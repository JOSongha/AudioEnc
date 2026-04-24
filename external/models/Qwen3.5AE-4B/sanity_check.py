"""
Sanity check: compare the converted Qwen3.5AE-4B against the original Qwen3.5-4B.

Both models are loaded via the SAME path:
    AutoModelForCausalLM.from_pretrained(path, dtype=X, attn_implementation="eager",
                                          trust_remote_code=True)

so dtype handling, weight loading, attn-backend selection are all identical between
ref and ae. With the same dtype the logits should match bit-for-bit (or within
small fp accumulation noise).

For the AE checkpoint, this requires the converted directory to contain:
  - modeling_qwen3_5AE.py / configuration_qwen3_5AE.py / audio_encoder.py
  - config.json with `auto_map` pointing to those classes
The convert script (convert_qwen3_5_to_qwen3_5AE.py) writes all of these.

Run in an env with transformers that has `qwen3_5` built in (main branch).

Example:
    python sanity_check.py \\
        --base-id Qwen/Qwen3.5-4B \\
        --ae-path ./pretrained \\
        --prompt "The capital of France is" \\
        --dtype bfloat16 \\
        --atol 1e-2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

THIS_DIR = Path(__file__).parent.resolve()


def resolve_path(model_id: str) -> Path:
    if os.path.isdir(model_id):
        return Path(model_id)
    return Path(snapshot_download(model_id))


def load_via_from_pretrained(path: Path, dtype: torch.dtype, attn_impl: str, trust_remote: bool):
    """Single load path used for both ref and ae."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=trust_remote,
    ).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-id", required=True, help="HF id or local path of base Qwen3.5-4B")
    parser.add_argument("--ae-path", default=str(THIS_DIR / "pretrained"),
                        help="Converted AE checkpoint dir (must contain modeling .py + auto_map in config.json)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt to feed both models")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance; bf16 needs ~1e-2, fp32 ~1e-4")
    parser.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2"],
                        help="Force the same attention backend on both models for an apples-to-apples comparison")
    parser.add_argument("--layer-diag", action="store_true",
                        help="Hook every decoder layer and print max|diff| of hidden_states after each layer")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    import transformers
    from transformers import AutoTokenizer
    print(f"transformers: {transformers.__version__} from {transformers.__file__}")

    base_path = resolve_path(args.base_id)
    ae_path = Path(args.ae_path)

    print(f"\n[1/4] Loading base from {base_path} (attn={args.attn_impl}, dtype={args.dtype})")
    ref = load_via_from_pretrained(base_path, dtype, args.attn_impl, trust_remote=False).to(device)
    print(f"  ref class: {type(ref).__name__}")
    print(f"  ref params (B): {sum(p.numel() for p in ref.parameters()) / 1e9:.2f}")
    print(f"  ref params dtype: {next(ref.parameters()).dtype}")
    print(f"  ref inv_freq dtype: {ref.model.rotary_emb.inv_freq.dtype}")

    print(f"\n[2/4] Loading AE from {ae_path} (trust_remote_code=True)")
    ae = load_via_from_pretrained(ae_path, dtype, args.attn_impl, trust_remote=True).to(device)
    print(f"  ae class: {type(ae).__name__}")
    print(f"  ae params (B): {sum(p.numel() for p in ae.parameters()) / 1e9:.2f}")
    print(f"  ae params dtype: {next(ae.parameters()).dtype}")
    print(f"  ae inv_freq dtype: {ae.model.language_model.rotary_emb.inv_freq.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_path))

    print(f"\n[3/4] Tokenizing prompt: {args.prompt!r}")
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"  tokens ({input_ids.shape[1]}): {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")

    # Optional layer hooks
    ref_layer_out: list[torch.Tensor] = []
    ae_layer_out: list[torch.Tensor] = []
    hooks = []
    if args.layer_diag:
        for layer in ref.model.layers:
            hooks.append(layer.register_forward_hook(
                lambda _m, _i, out, buf=ref_layer_out:
                    buf.append((out[0] if isinstance(out, tuple) else out).detach().to(torch.float32).cpu())
            ))
        for layer in ae.model.language_model.layers:
            hooks.append(layer.register_forward_hook(
                lambda _m, _i, out, buf=ae_layer_out:
                    buf.append((out[0] if isinstance(out, tuple) else out).detach().to(torch.float32).cpu())
            ))
        print(f"  [diag] hooks installed on {len(ref.model.layers)} layers")

    print("\n[4/4] Running text-only forward on both models and comparing logits")
    with torch.no_grad():
        ref_logits = ref(input_ids=input_ids).logits.detach().to(torch.float32).cpu()
        ae_logits = ae(input_ids=input_ids).logits.detach().to(torch.float32).cpu()

    for h in hooks:
        h.remove()

    if args.layer_diag and ref_layer_out and ae_layer_out:
        print(f"\n  [diag] per-layer hidden_states max|diff| (ref vs ae):")
        for i, (r, a) in enumerate(zip(ref_layer_out, ae_layer_out)):
            d = (r - a).abs()
            print(f"    layer {i:2d}: max|diff|={d.max().item():.3e}  mean|diff|={d.mean().item():.3e}")

    diff = (ref_logits - ae_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    allclose = max_diff < args.atol
    argmax_match = (ref_logits.argmax(-1) == ae_logits.argmax(-1)).all().item()

    print(f"\n  logits: ref={tuple(ref_logits.shape)} ae={tuple(ae_logits.shape)}")
    print(f"  max|diff| = {max_diff:.3e}   mean|diff| = {mean_diff:.3e}")
    print(f"  allclose (atol={args.atol}): {allclose}")
    print(f"  argmax match: {argmax_match}")

    def top_k(logits, k=5):
        probs = logits[0, -1].softmax(-1)
        top = probs.topk(k)
        return [(tokenizer.decode([i.item()]), p.item()) for i, p in zip(top.indices, top.values)]

    print(f"\n  next-token top-5 (ref): {top_k(ref_logits)}")
    print(f"  next-token top-5 (ae):  {top_k(ae_logits)}")

    if allclose and argmax_match:
        print("\nOK: AE model matches real Qwen3.5-4B within tolerance.")
    else:
        print("\nMISMATCH — investigate forward differences.")
        sys.exit(1)


if __name__ == "__main__":
    main()
