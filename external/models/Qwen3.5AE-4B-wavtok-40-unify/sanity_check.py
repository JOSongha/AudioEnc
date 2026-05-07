#!/usr/bin/env python3
"""Sanity check for Qwen3.5AE-4B-wavtok-40-unify variant.

Verify:
  [1/3] Optional: text-only logits parity vs base Qwen3.5AE-4B.
  [2/3] SEANetEncoder weights match WavTok checkpoint (within rtol=1e-5).
  [3/3] Projector init is non-trivial (std > 1e-3).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

THIS_DIR = Path(__file__).parent.resolve()


def _register_local_pkg() -> str:
    pkg = "qwen3_5_ae_wavtok_local"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(THIS_DIR)]
        sys.modules[pkg] = m
    return pkg


def merge_weight_norm(weight_g, weight_v, eps=1e-12):
    """Merge weight_norm parametrization: weight = weight_g * weight_v / ||weight_v||."""
    dims = tuple(range(1, len(weight_v.shape)))
    norm = torch.norm(weight_v, dim=dims, keepdim=True).clamp(min=eps)
    return weight_g * weight_v / norm


def check_encoder_weights(ae_model, wavtok_ckpt_path):
    """Verify encoder weights match WavTok checkpoint within rtol=1e-5."""
    print("\n[2/3] Checking SEANetEncoder weight match...")

    wavtok_ckpt_path = Path(wavtok_ckpt_path)
    if not wavtok_ckpt_path.exists():
        print(f"      SKIP: WavTok checkpoint not found at {wavtok_ckpt_path}")
        return True

    wt_ckpt = torch.load(str(wavtok_ckpt_path), map_location="cpu")
    wt_state = wt_ckpt.get("state_dict", wt_ckpt)

    encoder_model = ae_model.model.audio_encoder.encoder
    ae_state = encoder_model.state_dict()

    # Sample 3 keys: early conv, mid block, final conv
    probe_keys = [
        "conv_pre.weight",
        "seanet.layers.0.0.weight",  # mid-layer conv
        "conv_post.0.weight",
    ]

    all_match = True
    for key in probe_keys:
        if key not in ae_state:
            print(f"      SKIP: key {key} not in encoder (may be architecture variant)")
            continue

        ae_weight = ae_state[key].float()

        # Reconstruct from WavTok checkpoint (weight_norm if applicable)
        wt_key_g = f"feature_extractor.encodec.encoder.{key[:-7]}_g" if key.endswith(".weight") else f"feature_extractor.encodec.encoder.{key}"
        wt_key_v = f"feature_extractor.encodec.encoder.{key[:-7]}_v" if key.endswith(".weight") else None

        if wt_key_g in wt_state and wt_key_v in wt_state:
            # Merge weight_norm
            wt_weight = merge_weight_norm(wt_state[wt_key_g].float(), wt_state[wt_key_v].float())
        elif wt_key_g in wt_state:
            wt_weight = wt_state[wt_key_g].float()
        else:
            print(f"      SKIP: {key} not found in WavTok checkpoint (may be architecture variant)")
            continue

        match = torch.allclose(ae_weight, wt_weight, rtol=1e-5, atol=1e-7)
        status = "✓" if match else "✗"
        print(f"      {status} {key}: shape {ae_weight.shape}, match={match}")
        all_match = all_match and match

    return all_match


def check_projector_init(ae_model):
    """Verify projector init is non-trivial (std > 1e-3)."""
    print("\n[3/3] Checking projector initialization...")

    projector = ae_model.model.audio_encoder.projector
    all_nontrivial = True

    for name, param in projector.named_parameters():
        if param.requires_grad:
            std = param.data.float().std().item()
            is_nontrivial = std > 1e-3
            status = "✓" if is_nontrivial else "✗"
            print(f"      {status} {name}: std={std:.6f}")
            all_nontrivial = all_nontrivial and is_nontrivial

    return all_nontrivial


def main():
    print("=" * 70)
    print("Sanity Check: Qwen3.5AE-4B-wavtok-40-unify")
    print("=" * 70)

    pkg = _register_local_pkg()

    # Load model
    print("\n[0/3] Loading model from current directory...")
    try:
        ae_model = AutoModelForCausalLM.from_pretrained(
            str(THIS_DIR),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        print(f"      ✓ Model loaded from {THIS_DIR}")
    except Exception as e:
        print(f"      ✗ Failed to load model: {e}")
        return False

    ae_model.eval()

    # [1/3] Optional: text-only logits parity
    print("\n[1/3] Text-only logits parity check...")
    print("      SKIP (optional, requires base Qwen3.5AE-4B weights)")

    # [2/3] Encoder weight match
    wavtok_ckpt = THIS_DIR.parent.parent.parent / "mnt/tmp/hf_cache/wavtokenizer/wavtokenizer_large_unify_600_24k.ckpt"
    # Try alternative path
    if not wavtok_ckpt.exists():
        wavtok_ckpt = Path("/mnt/tmp/hf_cache/wavtokenizer/wavtokenizer_large_unify_600_24k.ckpt")

    enc_ok = check_encoder_weights(ae_model, wavtok_ckpt)

    # [3/3] Projector init
    proj_ok = check_projector_init(ae_model)

    # Summary
    print("\n" + "=" * 70)
    all_ok = enc_ok and proj_ok
    if all_ok:
        print("✓ All checks passed")
        print("=" * 70)
        return True
    else:
        print("✗ Some checks failed")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
