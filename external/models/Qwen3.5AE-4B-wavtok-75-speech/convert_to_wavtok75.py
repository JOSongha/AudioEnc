"""Convert Qwen3.5AE-4B (DAC variant) → Qwen3.5AE-4B-wavtok-75-speech.

Strategy:
  1. Load existing Qwen3.5AE-4B safetensors (LLM weights).
  2. Build new AE config with WavTokenizer-75 AudioConfig.
  3. Instantiate AE model — SEANetEncoder is randomly initialized.
  4. Drop `model.audio_encoder.*` (DAC) keys from source state_dict, keep LLM keys.
  5. Load LLM weights via `load_state_dict(strict=False)`.
  6. Load WavTokenizer-75 checkpoint, merge weight_norm pairs, bake into encoder.
  7. save_pretrained → safetensors: LLM (from source) + SEANetEncoder (pretrained) + projector (random).

Usage:
    python convert_to_wavtok75.py \
        --source /mnt/fr20tb/audiollm/sanghyuk/Qwen3.5AE-4B \
        --wavtok-ckpt /mnt/tmp/hf_cache/wavtokenizer/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_v2.ckpt \
        --output-dir external/models/Qwen3.5AE-4B-wavtok-75-speech \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file

THIS_DIR = Path(__file__).parent.resolve()

WAVTOK75_RATIOS   = [8, 5, 4, 2]   # product=320, 75fps @ 24kHz
WAVTOK75_HOP      = 320
WAVTOK75_FPS      = 75
WAVTOK75_DIM      = 512
WAVTOK75_NFILTERS = 32
WAVTOK75_LSTM     = 2
WAVTOK75_SR       = 24000
WAVTOK75_CKPT     = "/mnt/tmp/hf_cache/wavtokenizer/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_v2.ckpt"


def _register_local_pkg() -> str:
    pkg = "qwen3_5_ae_wavtok75_local"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(THIS_DIR)]
        sys.modules[pkg] = m
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    return pkg


def merge_weight_norm(weight_g, weight_v, eps=1e-12):
    dims = tuple(range(1, len(weight_v.shape)))
    norm = torch.norm(weight_v, dim=dims, keepdim=True).clamp(min=eps)
    return weight_g * weight_v / norm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="/mnt/fr20tb/audiollm/sanghyuk/Qwen3.5AE-4B")
    parser.add_argument("--wavtok-ckpt", default=WAVTOK75_CKPT)
    parser.add_argument("--output-dir", default=str(THIS_DIR))
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    pkg = _register_local_pkg()

    cfg_mod = importlib.import_module(f"{pkg}.configuration_qwen3_5AE")
    modeling_mod = importlib.import_module(f"{pkg}.modeling_qwen3_5AE")

    src_path = Path(args.source)
    ckpt_path = Path(args.wavtok_ckpt)
    out_dir = Path(args.output_dir)
    print(f"[1/6] Source: {src_path}")
    print(f"      WavTok-75 checkpoint: {ckpt_path}")
    print(f"      Output: {out_dir}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"WavTokenizer checkpoint not found: {ckpt_path}")

    # 1) Build AE config
    src_cfg_path = src_path / "config.json"
    with open(src_cfg_path) as f:
        src_cfg = json.load(f)
    text_dict = src_cfg["text_config"]
    text_cfg = cfg_mod.Qwen3_5AETextConfig(**{
        k: v for k, v in text_dict.items()
        if k in cfg_mod.Qwen3_5AETextConfig.__init__.__code__.co_varnames
    })
    audio_cfg = cfg_mod.AudioConfig(
        audio_hidden_size=WAVTOK75_DIM,
        llm_embed_size=text_cfg.hidden_size,
        wavtok_ratios=WAVTOK75_RATIOS,
        wavtok_hop_length=WAVTOK75_HOP,
        wavtok_fps=WAVTOK75_FPS,
        wavtok_dimension=WAVTOK75_DIM,
        wavtok_n_filters=WAVTOK75_NFILTERS,
        wavtok_lstm=WAVTOK75_LSTM,
        wavtok_sample_rate=WAVTOK75_SR,
        wavtok_ckpt_path=str(ckpt_path),
    )
    ae_config = cfg_mod.Qwen3_5AEConfig(
        text_config=text_cfg,
        audio_config=audio_cfg,
        tie_word_embeddings=src_cfg.get("tie_word_embeddings", False),
        audio_pad_token_id=src_cfg.get("audio_pad_token_id", 248076),
    )
    ae_config.torch_dtype = str(dtype).removeprefix("torch.")
    print(f"[2/6] AE config: text.hidden={text_cfg.hidden_size}, audio.dim={WAVTOK75_DIM}, "
          f"ratios={WAVTOK75_RATIOS}, hop={WAVTOK75_HOP} ({WAVTOK75_FPS}fps)")

    # 2) Instantiate model
    print("[3/6] Instantiating model (SEANetEncoder random init)...")
    ae_model = modeling_mod.Qwen3_5AEForConditionalGeneration(ae_config).to(dtype)
    ae_model.eval()
    n_total  = sum(p.numel() for p in ae_model.parameters()) / 1e9
    n_enc    = sum(p.numel() for p in ae_model.model.audio_encoder.encoder.parameters()) / 1e6
    n_proj   = sum(p.numel() for p in ae_model.model.audio_encoder.projector.parameters()) / 1e6
    n_lm     = sum(p.numel() for p in ae_model.model.language_model.parameters()) / 1e9
    print(f"      total: {n_total:.2f}B, LM: {n_lm:.2f}B, encoder: {n_enc:.2f}M, projector: {n_proj:.2f}M")

    # 3) Load LLM weights; drop DAC audio_encoder keys
    shard_files = sorted(src_path.glob("model-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No model-*.safetensors in {src_path}")
    print(f"[4/6] Loading {len(shard_files)} source shard(s); dropping DAC audio_encoder keys...")

    llm_state = {}
    n_dropped = 0
    for sf in shard_files:
        part = load_file(str(sf), device="cpu")
        for k, v in part.items():
            if k.startswith("model.audio_encoder."):
                n_dropped += 1
                continue
            llm_state[k] = v.to(dtype)
    print(f"      kept {len(llm_state)} LLM keys, dropped {n_dropped} DAC audio_encoder keys")

    missing, unexpected = ae_model.load_state_dict(llm_state, strict=False)
    audio_missing = [k for k in missing if k.startswith("model.audio_encoder.")]
    other_missing = [k for k in missing if not k.startswith("model.audio_encoder.")]
    print(f"      audio_encoder keys to fill from WavTok: {len(audio_missing)}")
    if other_missing:
        print(f"      WARNING: {len(other_missing)} non-audio keys missing:")
        for k in other_missing[:5]:
            print(f"        - {k}")
    if unexpected:
        print(f"      WARNING: {len(unexpected)} unexpected keys")

    # 4) Load WavTokenizer-75 checkpoint and merge weight_norm
    print("[5/6] Loading WavTokenizer-75 checkpoint and merging weight_norm...")
    wt_ckpt = torch.load(str(ckpt_path), map_location="cpu")
    wt_state = wt_ckpt.get("state_dict", wt_ckpt)

    encoder_keys = [k for k in wt_state if k.startswith("feature_extractor.encodec.encoder.")]
    print(f"      found {len(encoder_keys)} encoder keys")

    wavtok_encoder_state = {}
    weight_norm_groups: dict[str, dict] = {}
    for k in encoder_keys:
        base = k.replace("feature_extractor.encodec.encoder.", "", 1)
        if base.endswith("_g"):
            prefix = base[:-2]
            weight_norm_groups.setdefault(prefix, {})["g"] = wt_state[k]
        elif base.endswith("_v"):
            prefix = base[:-2]
            weight_norm_groups.setdefault(prefix, {})["v"] = wt_state[k]
        else:
            wavtok_encoder_state[base] = wt_state[k]

    n_merged = 0
    for prefix, group in weight_norm_groups.items():
        if "g" in group and "v" in group:
            merged = merge_weight_norm(group["g"], group["v"])
            wavtok_encoder_state[prefix] = merged
            n_merged += 1
    print(f"      merged {n_merged} weight_norm pairs → {len(wavtok_encoder_state)} plain keys")

    encoder_model = ae_model.model.audio_encoder.encoder
    missing_enc, unexpected_enc = encoder_model.load_state_dict(wavtok_encoder_state, strict=True)
    if missing_enc:
        print(f"      WARNING: missing encoder keys: {missing_enc[:5]}")
    if unexpected_enc:
        print(f"      WARNING: unexpected encoder keys: {unexpected_enc[:5]}")
    print(f"      encoder loaded OK (strict=True)")

    if args.dry_run:
        print("[6/6] --dry-run: skipping save")
        return

    # 5) Save
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[6/6] Saving to {out_dir} ...")
    ae_model.save_pretrained(out_dir, safe_serialization=True)

    # Copy code files
    py_files = ["modeling_qwen3_5AE.py", "configuration_qwen3_5AE.py",
                "audio_encoder.py", "tokenization_qwen3_5AE.py"]
    for fname in py_files:
        src = THIS_DIR / fname
        dst = (out_dir / fname).resolve()
        if src.exists() and src.resolve() != dst:
            shutil.copy2(src, dst)

    # Copy wavtokenizer_modules
    wavtok_src = THIS_DIR / "wavtokenizer_modules"
    wavtok_dst = out_dir / "wavtokenizer_modules"
    if wavtok_src.exists() and wavtok_src.resolve() != wavtok_dst.resolve():
        if wavtok_dst.exists():
            shutil.rmtree(wavtok_dst)
        shutil.copytree(wavtok_src, wavtok_dst)

    # Copy tokenizer + aux files from source
    aux_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json",
        "chat_template.jinja", "generation_config.json",
    ]
    for fname in aux_files:
        src = src_path / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    # Patch config.json: auto_map + dtype + wavtok fields
    cfg_path = out_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["auto_map"] = {
        "AutoConfig": "configuration_qwen3_5AE.Qwen3_5AEConfig",
        "AutoModelForCausalLM": "modeling_qwen3_5AE.Qwen3_5AEForConditionalGeneration",
    }
    dtype_str = str(dtype).removeprefix("torch.")
    for section in (cfg, cfg.get("text_config", {}), cfg.get("audio_config", {})):
        section["torch_dtype"] = dtype_str
        section["dtype"] = dtype_str
    # Ensure wavtok fields are correct
    ac = cfg.setdefault("audio_config", {})
    ac.update({
        "wavtok_ratios": WAVTOK75_RATIOS,
        "wavtok_hop_length": WAVTOK75_HOP,
        "wavtok_fps": WAVTOK75_FPS,
        "wavtok_dimension": WAVTOK75_DIM,
        "wavtok_n_filters": WAVTOK75_NFILTERS,
        "wavtok_lstm": WAVTOK75_LSTM,
        "wavtok_sample_rate": WAVTOK75_SR,
        "wavtok_ckpt_path": str(ckpt_path),
    })
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"      patched config.json (auto_map, dtype={dtype_str}, wavtok 75fps fields)")
    print("Done.")


if __name__ == "__main__":
    main()
