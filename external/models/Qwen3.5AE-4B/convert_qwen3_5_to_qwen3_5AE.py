"""
Convert Qwen3.5-4B base model weights to the Qwen3.5AE-4B format.

Key remap: every `model.X` weight in the base model becomes `model.language_model.X`
in the AE model (since the AE wraps the text decoder under `model.language_model`).
The `model.audio_encoder.*` weights have no source in the base model, so they are
left at their random initialization and noted as "fresh".

Example:
    python convert_qwen3_5_to_qwen3_5AE.py \\
        --base-id Qwen/Qwen3.5-4B \\
        --output-dir ./pretrained \\
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import types
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

THIS_DIR = Path(__file__).parent.resolve()

# Register this directory as a synthetic package so relative imports inside the
# modeling file resolve (.audio_encoder / .configuration_qwen3_5AE).
_PKG = "qwen3_5_ae_local"
if _PKG not in sys.modules:
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(THIS_DIR)]
    sys.modules[_PKG] = pkg

import importlib

cfg_mod = importlib.import_module(f"{_PKG}.configuration_qwen3_5AE")
modeling_mod = importlib.import_module(f"{_PKG}.modeling_qwen3_5AE")


def resolve_base_path(base_id: str) -> Path:
    if os.path.isdir(base_id):
        return Path(base_id)
    return Path(snapshot_download(base_id))


def build_text_config(base_config: dict) -> "cfg_mod.Qwen3_5AETextConfig":
    """Build the AE text config from the base model's config.

    Handles both layouts:
      - VL base (Qwen3.5-4B): the text decoder config is nested under `text_config`
      - text-only base: fields are at the top level
    """
    text_dict = base_config.get("text_config", base_config)

    accepted = set(inspect.signature(cfg_mod.Qwen3_5AETextConfig.__init__).parameters) - {"self", "kwargs"}
    kwargs = {k: v for k, v in text_dict.items() if k in accepted}

    # tie_word_embeddings can live in either text_config or top-level
    kwargs["tie_word_embeddings"] = text_dict.get(
        "tie_word_embeddings", base_config.get("tie_word_embeddings", False)
    )

    # If rope_parameters is absent, synthesize it from rope_theta/rope_scaling/partial_rotary_factor
    if "rope_parameters" not in kwargs or kwargs["rope_parameters"] is None:
        rope_params = {
            "rope_type": "default",
            "rope_theta": text_dict.get("rope_theta", base_config.get("rope_theta", 10000.0)),
            "partial_rotary_factor": text_dict.get(
                "partial_rotary_factor", base_config.get("partial_rotary_factor", 0.25)
            ),
        }
        scaling = text_dict.get("rope_scaling") or base_config.get("rope_scaling")
        if scaling:
            rope_params.update(scaling)
            if "type" in rope_params and "rope_type" not in scaling:
                rope_params["rope_type"] = rope_params["type"]
        kwargs["rope_parameters"] = rope_params

    return cfg_mod.Qwen3_5AETextConfig(**kwargs)


def remap_key(k: str) -> str | None:
    """Remap a base-model key to the AE model key, or return None to drop.

    Handles both layouts:
      - Text-only base (e.g. Qwen3_5ForCausalLM): `model.X` → `model.language_model.X`
      - VL base (Qwen3_5ForConditionalGeneration): `model.language_model.X` stays;
        `model.visual.X` is dropped (no vision in AE).
      - `lm_head.X`: stays.
      - `mtp.X`: dropped (multi-token prediction head, not used in AE).
    """
    if k.startswith("mtp."):
        return None  # MTP head not in AE
    if k.startswith("model.visual."):
        return None  # Vision tower not in AE
    if k.startswith("model.language_model."):
        return k  # already correctly prefixed
    if k.startswith("model."):
        return "model.language_model." + k[len("model."):]
    return k  # lm_head.*, etc.


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-id", required=True, help="HF model id OR local path of base Qwen3.5-4B")
    parser.add_argument("--output-dir", default=str(THIS_DIR / "pretrained"), help="Output dir for AE checkpoint")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    # 1) Resolve base model path
    base_path = resolve_base_path(args.base_id)
    print(f"[1/5] Base model path: {base_path}")

    # 2) Build AE config
    with open(base_path / "config.json") as f:
        base_cfg = json.load(f)
    print(f"      base model_type={base_cfg.get('model_type')}, arch={base_cfg.get('architectures')}")

    text_cfg = build_text_config(base_cfg)
    audio_cfg = cfg_mod.AudioConfig(llm_embed_size=text_cfg.hidden_size)
    ae_config = cfg_mod.Qwen3_5AEConfig(
        text_config=text_cfg,
        audio_config=audio_cfg,
        tie_word_embeddings=text_cfg.tie_word_embeddings,
    )
    ae_config.torch_dtype = str(dtype).removeprefix("torch.")
    print(f"[2/5] AE config built (text.hidden={text_cfg.hidden_size}, n_layers={text_cfg.num_hidden_layers})")

    # 3) Instantiate AE model (random init; text weights overwritten next)
    print("[3/5] Instantiating AE model (this allocates ~4B params; may take a moment)...")
    ae_model = modeling_mod.Qwen3_5AEForConditionalGeneration(ae_config).to(dtype)
    ae_model.eval()
    n_params = sum(p.numel() for p in ae_model.parameters()) / 1e9
    print(f"      AE params: {n_params:.2f} B")

    # 4) Load base state_dict (sharded safetensors) and remap keys
    shard_files = sorted(base_path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors in {base_path}")
    print(f"[4/5] Loading {len(shard_files)} shard(s) from base and remapping keys...")

    remapped: dict[str, torch.Tensor] = {}
    dropped: dict[str, int] = {"mtp.": 0, "model.visual.": 0}
    from collections import Counter
    prefix_counter: Counter = Counter()
    for sf in shard_files:
        part = load_file(str(sf), device="cpu")
        for k, v in part.items():
            prefix_counter[".".join(k.split(".")[:2])] += 1
            new_k = remap_key(k)
            if new_k is None:
                for prefix in dropped:
                    if k.startswith(prefix):
                        dropped[prefix] += 1
                        break
                continue
            remapped[new_k] = v.to(dtype)

    print(f"      base key prefixes ({len(prefix_counter)} unique):")
    for prefix, cnt in prefix_counter.most_common():
        print(f"        {prefix:40s} {cnt:4d}")
    for prefix, cnt in dropped.items():
        if cnt:
            print(f"      dropped {prefix}* keys: {cnt}")

    missing, unexpected = ae_model.load_state_dict(remapped, strict=False)
    audio_missing = [k for k in missing if k.startswith("model.audio_encoder")]
    other_missing = [k for k in missing if not k.startswith("model.audio_encoder")]
    print(f"      audio_encoder keys left at random init: {len(audio_missing)}")

    # If lm_head.weight is missing because the base ties it to embeddings, re-tie now.
    if ae_model.config.tie_word_embeddings:
        ae_model.tie_weights()
        # `lm_head.weight` is now shared with the input embedding; remove from "missing"
        other_missing = [k for k in other_missing if k != "lm_head.weight"]
        print(f"      tied lm_head.weight to embed_tokens.weight (tie_word_embeddings=True)")

    if other_missing:
        print(f"      WARNING: {len(other_missing)} non-audio keys still missing, first 10:")
        for k in other_missing[:10]:
            print(f"        - {k}")
    if unexpected:
        print(f"      WARNING: {len(unexpected)} unexpected keys (not in AE model), first 10:")
        for k in unexpected[:10]:
            print(f"        - {k}")

    # 5) Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[5/5] Saving AE checkpoint to {out_dir}")
    ae_model.save_pretrained(out_dir, safe_serialization=True)

    # 5a) Copy modeling/config/audio_encoder Python files into the checkpoint dir so that
    # `AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True)` can find them.
    import shutil
    py_files = ["modeling_qwen3_5AE.py", "configuration_qwen3_5AE.py", "audio_encoder.py"]
    for fname in py_files:
        src = THIS_DIR / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            print(f"      copied {fname}")
        else:
            print(f"      WARNING: {fname} not found in {THIS_DIR}, skipping")

    # 5b) Add auto_map + fill both torch_dtype and dtype (transformers main serializes as
    # "dtype" only; transformers 4.57.x reads "torch_dtype"; fill both for cross-version compat).
    config_json_path = out_dir / "config.json"
    with open(config_json_path) as f:
        cfg = json.load(f)
    cfg["auto_map"] = {
        "AutoConfig": "configuration_qwen3_5AE.Qwen3_5AEConfig",
        "AutoModelForCausalLM": "modeling_qwen3_5AE.Qwen3_5AEForConditionalGeneration",
    }
    dtype_str = str(dtype).removeprefix("torch.")
    # Set dtype on top-level + all sub-configs. FA2's dispatch check reads
    # `self.config.dtype` on EACH sub-model (e.g., the language_model uses text_config).
    # If any sub-config's dtype is None, transformers warns at init.
    for section in (cfg, cfg.get("text_config", {}), cfg.get("audio_config", {})):
        section["torch_dtype"] = dtype_str
        section["dtype"] = dtype_str
    with open(config_json_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"      added auto_map and torch_dtype/dtype={dtype_str} to all configs")

    # 5c) Tokenizer: copy from base so users can `AutoTokenizer.from_pretrained(out_dir)`
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json",
    ]
    copied_tok = 0
    for fname in tokenizer_files:
        src = base_path / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            copied_tok += 1
    print(f"      copied {copied_tok} tokenizer file(s) from base")

    print("Done.")


if __name__ == "__main__":
    main()
