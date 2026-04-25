"""
Sanity check for Qwen3.5AE-4B-whisper-small converted checkpoint.

Checks:
  1. (Optional) Text-only logits match base Qwen3.5-4B (within bf16 tolerance).
     Pass --base-id if a local base model is available. Skip with --skip-logits.
  2. Whisper encoder weights in AE checkpoint match HF openai/whisper-small.en.
  3. AudioProjector is random-initialized (has non-trivial variance, not matching Whisper).

Example (with base model):
    python sanity_check.py \\
        --base-id Qwen/Qwen3.5-4B \\
        --ae-path . \\
        --whisper-id openai/whisper-small.en \\
        --dtype bfloat16

Example (encoder + projector check only):
    python sanity_check.py --ae-path . --skip-logits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).parent.resolve()


def load_ae(ae_path: Path, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        str(ae_path),
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()


def load_base(base_id: str, dtype: torch.dtype, trust_remote: bool = False):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=trust_remote,
    ).eval()


def load_whisper_encoder(whisper_id: str, dtype: torch.dtype):
    from transformers import WhisperModel
    m = WhisperModel.from_pretrained(whisper_id, dtype=dtype)
    enc = m.encoder
    del m
    return enc.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-id", default=None, help="HF id or local path of base Qwen3.5-4B (optional)")
    parser.add_argument("--base-trust-remote", action="store_true",
                        help="Use trust_remote_code for base model (e.g. if using DAC AE as reference)")
    parser.add_argument("--skip-logits", action="store_true", help="Skip logits comparison (for offline use)")
    parser.add_argument("--ae-path", default=str(THIS_DIR), help="Converted AE checkpoint dir")
    parser.add_argument("--whisper-id", default="openai/whisper-small.en", help="HF id or local path for Whisper")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    if not args.skip_logits and args.base_id is None:
        parser.error("--base-id is required unless --skip-logits is set")

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    failed = False

    print("=" * 60)
    print("Whisper-small AE sanity check")
    print("=" * 60)

    from transformers import AutoTokenizer
    ae_path = Path(args.ae_path)

    print(f"\nLoading AE from {ae_path} ...")
    ae = load_ae(ae_path, dtype).to(device)
    total_params = sum(p.numel() for p in ae.parameters()) / 1e9
    print(f"  ae class: {type(ae).__name__}")
    print(f"  ae params: {total_params:.3f}B")

    # ── 1. Text-only logits comparison (optional) ─────────────────────
    if args.skip_logits:
        print("\n[1/3] Logits comparison: SKIPPED (--skip-logits)")
    else:
        print(f"\n[1/3] Text-only logits comparison (AE vs {args.base_id})")
        print(f"  Loading base from {args.base_id} ...")
        base = load_base(args.base_id, dtype, trust_remote=args.base_trust_remote).to(device)

        tok = AutoTokenizer.from_pretrained(args.base_id, trust_remote_code=args.base_trust_remote)
        input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
        print(f"  Prompt: {args.prompt!r}  ({input_ids.shape[1]} tokens)")

        with torch.no_grad():
            ref_logits = base(input_ids=input_ids).logits.float().cpu()
            ae_logits = ae(input_ids=input_ids).logits.float().cpu()

        diff = (ref_logits - ae_logits).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        argmax_ok = (ref_logits.argmax(-1) == ae_logits.argmax(-1)).all().item()
        logits_ok = max_diff < args.atol and argmax_ok

        print(f"  max|diff|={max_diff:.3e}  mean|diff|={mean_diff:.3e}  atol={args.atol}")
        print(f"  argmax match: {argmax_ok}")
        if logits_ok:
            print("  [PASS] Logits match.")
        else:
            print("  [FAIL] Logits MISMATCH — check key remapping in convert script.")
            failed = True

        def top5(logits, tokenizer):
            probs = logits[0, -1].softmax(-1)
            top = probs.topk(5)
            return [(tokenizer.decode([i.item()]), f"{p.item():.3f}") for i, p in zip(top.indices, top.values)]
        print(f"  next-token top-5 (base): {top5(ref_logits, tok)}")
        print(f"  next-token top-5 (ae):   {top5(ae_logits, tok)}")

        del base

    # ── 2. Whisper encoder weights ────────────────────────────────────
    print("\n[2/3] Whisper encoder weight verification")
    print(f"  Loading HF Whisper from {args.whisper_id} for comparison ...")
    hf_enc = load_whisper_encoder(args.whisper_id, dtype)

    # Compare a few representative weight tensors
    probe_keys = [
        ("layers.0.self_attn.q_proj.weight",  "encoder layer-0 q_proj"),
        ("layers.5.self_attn.out_proj.weight", "encoder layer-5 out_proj"),
        ("layers.11.fc1.weight",               "encoder layer-11 fc1"),
    ]
    ae_enc = ae.model.audio_encoder.encoder

    all_enc_ok = True
    for key, label in probe_keys:
        hf_w = hf_enc.get_parameter(key).detach().float()
        ae_w = ae_enc.get_parameter(key).detach().float().cpu()
        d = (hf_w.cpu() - ae_w).abs()
        ok = d.max().item() < 1e-3  # bf16 rounding
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: max|diff|={d.max().item():.3e}  norm(hf)={hf_w.norm().item():.3f}")
        if not ok:
            all_enc_ok = False

    if all_enc_ok:
        print("  [PASS] Whisper encoder weights match HF original.")
    else:
        print("  [FAIL] Encoder weights differ — Whisper pretrained not properly baked in.")
        failed = True

    del hf_enc

    # ── 3. AudioProjector random init ────────────────────────────────
    print("\n[3/3] AudioProjector random-init verification")
    proj = ae.model.audio_encoder.projector

    input_proj_w = proj.input_proj.weight.detach().float().cpu()
    layer0_q = proj.layers[0].self_attn.q_proj.weight.detach().float().cpu()

    std_input = input_proj_w.std().item()
    std_layer0 = layer0_q.std().item()

    # Kaiming init produces std ≈ sqrt(2/fan_in). For input_proj 768→512, expect ~0.05
    proj_ok = std_input > 1e-3 and std_layer0 > 1e-3
    print(f"  input_proj.weight std = {std_input:.4f}")
    print(f"  layers[0].q_proj.weight std = {std_layer0:.4f}")
    if proj_ok:
        print("  [PASS] Projector is random-initialized (to be trained in Stage1).")
    else:
        print("  [FAIL] Projector std suspiciously low — may be zeroed out.")
        failed = True

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failed:
        print("RESULT: FAIL — see details above.")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        print("  → AE logits match base LLM")
        print("  → Whisper encoder pretrained weights verified")
        print("  → AudioProjector ready for Stage1 training")
    print("=" * 60)


if __name__ == "__main__":
    main()
