"""Convert Qwen3.5AE-4B (DAC variant) → Qwen3.5AE-4B-whisper-small.

Strategy:
  1. Load existing Qwen3.5AE-4B safetensors (already has `model.language_model.*` LLM weights).
  2. Build new AE config with Whisper AudioConfig.
  3. Instantiate AE model — `AudioEncoder.__init__` downloads Whisper-small.en from HF and
     bakes pretrained encoder weights into the model.
  4. Drop `model.audio_encoder.*` (DAC) keys from source state_dict, keep LLM keys.
  5. Load LLM weights via `load_state_dict(strict=False)`. Whisper encoder + projector keep
     their fresh init (Whisper pretrained from HF, projector random).
  6. save_pretrained → new safetensors with LLM (from source) + Whisper (pretrained) + projector
     (random init, to be trained in Stage1).

Usage:
    python convert_to_whisper.py \
        --source external/models/Qwen3.5AE-4B \
        --output-dir external/models/Qwen3.5AE-4B-whisper-small \
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


def _register_local_pkg() -> str:
    pkg = "qwen3_5_ae_whisper_local"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(THIS_DIR)]
        sys.modules[pkg] = m
    return pkg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Local path of existing Qwen3.5AE-4B (DAC variant)")
    parser.add_argument("--output-dir", default=str(THIS_DIR), help="Output dir (default: this script's dir)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--dry-run", action="store_true", help="Skip save_pretrained, just verify state_dict")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    pkg = _register_local_pkg()

    cfg_mod = importlib.import_module(f"{pkg}.configuration_qwen3_5AE")
    modeling_mod = importlib.import_module(f"{pkg}.modeling_qwen3_5AE")

    src_path = Path(args.source)
    out_dir = Path(args.output_dir)
    print(f"[1/5] Source: {src_path}")
    print(f"      Output: {out_dir}")

    # 1) Build AE config: keep source text_config, replace audio_config with whisper variant
    src_cfg_path = src_path / "config.json"
    with open(src_cfg_path) as f:
        src_cfg = json.load(f)
    text_dict = src_cfg["text_config"]
    text_cfg = cfg_mod.Qwen3_5AETextConfig(**{
        k: v for k, v in text_dict.items()
        if k in cfg_mod.Qwen3_5AETextConfig.__init__.__code__.co_varnames
    })
    audio_cfg = cfg_mod.AudioConfig(llm_embed_size=text_cfg.hidden_size)
    ae_config = cfg_mod.Qwen3_5AEConfig(
        text_config=text_cfg,
        audio_config=audio_cfg,
        tie_word_embeddings=src_cfg.get("tie_word_embeddings", False),
        audio_pad_token_id=src_cfg.get("audio_pad_token_id", 248076),
    )
    ae_config.torch_dtype = str(dtype).removeprefix("torch.")
    print(f"[2/5] AE config built: text.hidden={text_cfg.hidden_size}, audio.hidden={audio_cfg.audio_hidden_size}")
    print(f"      whisper_model_id={audio_cfg.whisper_model_id}")

    # 2) Instantiate AE model — this triggers WhisperModel.from_pretrained inside AudioEncoder.__init__
    print("[3/5] Instantiating AE model (downloads Whisper-small.en if not cached)...")
    ae_model = modeling_mod.Qwen3_5AEForConditionalGeneration(ae_config).to(dtype)
    ae_model.eval()
    n_params = sum(p.numel() for p in ae_model.parameters()) / 1e9
    n_audio = sum(p.numel() for p in ae_model.model.audio_encoder.parameters()) / 1e6
    n_proj = sum(p.numel() for p in ae_model.model.audio_encoder.projector.parameters()) / 1e6
    n_lm = sum(p.numel() for p in ae_model.model.language_model.parameters()) / 1e9
    print(f"      total: {n_params:.2f}B, language_model: {n_lm:.2f}B, "
          f"audio_encoder: {n_audio:.2f}M (projector: {n_proj:.2f}M)")

    # 3) Load LLM weights from source AE; drop DAC audio_encoder.*
    shard_files = sorted(src_path.glob("model-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No model-*.safetensors in {src_path}")
    print(f"[4/5] Loading {len(shard_files)} shard(s) from source; dropping DAC audio_encoder keys...")

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
    print(f"      audio_encoder keys NOT loaded from source (filled by Whisper init / random): {len(audio_missing)}")
    if other_missing:
        print(f"      WARNING: {len(other_missing)} non-audio keys missing")
        for k in other_missing[:5]:
            print(f"        - {k}")
    if unexpected:
        print(f"      WARNING: {len(unexpected)} unexpected keys")
        for k in unexpected[:5]:
            print(f"        - {k}")

    if args.dry_run:
        print("[5/5] --dry-run: skip save_pretrained")
        return

    # 4) Save
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[5/5] Saving AE checkpoint to {out_dir}")
    ae_model.save_pretrained(out_dir, safe_serialization=True)

    # Copy code files (overwrite if changed)
    py_files = ["modeling_qwen3_5AE.py", "configuration_qwen3_5AE.py", "audio_encoder.py", "tokenization_qwen3_5AE.py"]
    for fname in py_files:
        src = THIS_DIR / fname
        if src.exists() and src != out_dir / fname:
            shutil.copy2(src, out_dir / fname)

    # Copy tokenizer + chat_template + generation_config from source
    aux_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json",
        "chat_template.jinja", "generation_config.json",
    ]
    copied = 0
    for fname in aux_files:
        src = src_path / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            copied += 1
    print(f"      copied {copied} auxiliary file(s) from source")

    # Patch config.json: auto_map + dtype
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
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"      patched auto_map and dtype={dtype_str}")
    print("Done.")


if __name__ == "__main__":
    main()
