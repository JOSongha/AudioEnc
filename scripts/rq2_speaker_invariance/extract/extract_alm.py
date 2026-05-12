#!/usr/bin/env python
"""Run a trained Qwen3.5AE checkpoint forward, capturing per-layer hidden states
at audio frame positions.

Methodology matches §4.3 inference prompt: we manually project the encoder
output and feed the LM `inputs_embeds = [<|audio_start|>] [projected audio]
[<|audio_end|>]` with no surrounding text prompt. Audio frames are the LM
positions `[1 .. 1+T_audio)`.

Forward hooks expose internal-layer activations that the §4.3 script did not
capture:
  - audio_encoder.encoder.layers[i]   → encoder layer i hidden  [B, T_enc, D_enc]
  - audio_encoder.projector.layers[i] → projector layer i        [B, T_enc, D_adapter]
  - language_model.layers[i]          → LM layer i               [B, 2+T_audio, D_llm]

Plus three "summary" tensors:
  - enc.out  : raw encoder output (post final-norm)
  - proj.out : projector output, before LM
  - llm.norm : final LM hidden (post-norm)

LoRA: pass `--lora <dir>` to merge a Stage2 LoRA adapter on top of the base
checkpoint (`peft.PeftModel.from_pretrained` + `merge_and_unload`).

Usage:
  python extract_alm.py \
      --ckpt   /mnt/tmp/Qwen3.5_whisper_tiny_v6_Stage1/.../checkpoint-64000 \
      --pairs  ../pairs/all_pairs.jsonl \
      --out    emb/alm_whisper_tiny_v6
  python extract_alm.py --ckpt <base> --lora <stage2_lora> --tag wtiny_s2 ...
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, WhisperFeatureExtractor

from _io import load_pairs, load_wav, mean_pool, save_layer_npz

# Stage2-trained boundary tokens. Same constants used by sec43_embeddings.
AUDIO_START_ID = 248070
AUDIO_END_ID = 248071

HOP = 320
MAX_ENC_FRAMES = 1500


def make_hook(store: dict, key: str):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        store[key] = output.detach()
    return hook


def patch_whisper_attn(impl_fallback: str = "sdpa"):
    """Some flash_attn 2 wheels link to libc symbols missing on this box. Force
    SDPA whenever AudioEncoder asks for flash_attention_2. Mirrors the patch in
    sec43_embeddings/extract_embeddings.py."""
    from transformers import WhisperModel
    orig = WhisperModel.from_pretrained.__func__ if hasattr(WhisperModel.from_pretrained, "__func__") \
        else WhisperModel.from_pretrained

    def patched(cls, *a, **kw):
        if kw.get("attn_implementation") == "flash_attention_2":
            kw["attn_implementation"] = impl_fallback
        return orig(cls, *a, **kw) if hasattr(orig, "__self__") else orig(*a, **kw)

    WhisperModel.from_pretrained = classmethod(patched)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF model dir (Qwen3.5AE-* checkpoint)")
    ap.add_argument("--lora", default=None, help="optional LoRA adapter dir; merge then unload")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="single-utt is simplest because projector receives variable T")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--attn", default="sdpa",
                    help="LM attention impl. 'sdpa' / 'eager' safe; flash kernels may interfere with hooks.")
    ap.add_argument("--llm_stride", type=int, default=2,
                    help="probe every Nth LLM layer (1 = all).")
    ap.add_argument("--max_sec", type=float, default=10.0,
                    help="audio truncation (existing §4.3 used 10s).")
    ap.add_argument("--patch_flash", action="store_true",
                    help="force whisper inside AudioEncoder to use sdpa not flash_attention_2")
    args = ap.parse_args()

    if args.patch_flash:
        patch_whisper_attn(impl_fallback="sdpa")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    ckpt_dir = Path(args.ckpt).resolve()
    tag = args.tag or ckpt_dir.parent.name
    out_dir = Path(args.out)

    print(f"[alm] loading {ckpt_dir}  (attn={args.attn})")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation=args.attn,
    )
    if args.lora:
        from peft import PeftModel
        print(f"[alm] merging LoRA from {args.lora}")
        model = PeftModel.from_pretrained(model, args.lora, is_trainable=False)
        model = model.merge_and_unload()
        tag = args.tag or f"{ckpt_dir.parent.name}+{Path(args.lora).parent.name}"
    model = model.to(args.device).eval()
    cfg = model.config
    whisper_id = cfg.audio_config.whisper_model_id
    print(f"[alm] tag={tag}  whisper={whisper_id}")
    feat_ex = WhisperFeatureExtractor.from_pretrained(whisper_id)

    base = getattr(model, "model", model)
    if not hasattr(base, "audio_encoder"):
        base = getattr(base, "model", base)
    audio_enc = base.audio_encoder
    encoder_inner = audio_enc.encoder
    projector = audio_enc.projector
    language_model = base.language_model
    embed_tokens = language_model.get_input_embeddings()

    n_enc = len(encoder_inner.layers)
    n_proj = len(projector.layers)
    n_llm = len(language_model.layers)
    print(f"[alm] enc_layers={n_enc} proj_layers={n_proj} llm_layers={n_llm}")

    as_emb = embed_tokens(torch.tensor([AUDIO_START_ID], device=args.device)).to(dtype)
    ae_emb = embed_tokens(torch.tensor([AUDIO_END_ID], device=args.device)).to(dtype)

    store: dict[str, torch.Tensor] = {}
    handles = []
    for i, mod in enumerate(encoder_inner.layers):
        handles.append(mod.register_forward_hook(make_hook(store, f"enc.L{i+1:02d}")))
    for i, mod in enumerate(projector.layers):
        handles.append(mod.register_forward_hook(make_hook(store, f"proj.L{i+1:02d}")))
    llm_probe_idx = sorted(set(list(range(0, n_llm, args.llm_stride)) + [n_llm - 1]))
    for i in llm_probe_idx:
        handles.append(language_model.layers[i].register_forward_hook(make_hook(store, f"llm.L{i+1:02d}")))
    handles.append(encoder_inner.register_forward_hook(make_hook(store, "enc.out")))
    handles.append(projector.register_forward_hook(make_hook(store, "proj.out")))
    handles.append(language_model.norm.register_forward_hook(make_hook(store, "llm.norm")))

    rows = load_pairs(args.pairs)
    print(f"[alm] rows={len(rows)}  max_sec={args.max_sec}")

    layer_keys: list[str] | None = None
    by_layer: dict[str, list[dict]] = {}

    for b0 in tqdm(range(0, len(rows), args.batch_size)):
        batch = rows[b0 : b0 + args.batch_size]
        assert len(batch) == 1, "manual flanking path runs one utterance at a time"
        r = batch[0]
        wav = load_wav(r["audio_path"], target_sr=16000, max_sec=args.max_sec)
        feats = feat_ex(wav, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(args.device, dtype=dtype)
        n_valid = min(MAX_ENC_FRAMES, math.ceil(len(wav) / HOP))

        store.clear()
        # Manually walk: encoder -> projector -> LM
        enc_hidden = encoder_inner(feats).last_hidden_state  # [1, 1500, D_enc]
        enc_hidden = enc_hidden[:, :n_valid, :]  # crop to valid frames
        proj_out, _ = projector(enc_hidden, use_cache=False)  # [1, T_audio, D_llm]
        proj_out = proj_out.to(dtype)

        seq_emb = torch.cat([as_emb[None], proj_out, ae_emb[None]], dim=1)  # [1, T+2, D]
        _ = language_model(inputs_embeds=seq_emb, use_cache=False)

        if layer_keys is None:
            layer_keys = sorted(store.keys())
            print(f"[alm] capturing {len(layer_keys)} layer tensors per utt")
            for k in layer_keys:
                by_layer[k] = []

        for key in layer_keys:
            h = store[key].float()
            if key.startswith("enc."):
                vlen = h.shape[1]  # encoder hook fires on full 1500 frames
                vlen = min(vlen, n_valid)
                emb = mean_pool(h[0], valid_len=vlen)
            elif key.startswith("proj."):
                # projector layers (and proj.out) operate on cropped enc_hidden,
                # so h.shape[1] == n_valid already.
                emb = mean_pool(h[0])
            elif key.startswith("llm."):
                # LM seq is [as, audio..., ae]. Pool only audio positions.
                emb = mean_pool(h[0, 1 : 1 + n_valid])
            else:
                emb = mean_pool(h[0])
            by_layer[key].append({
                "emb": emb.cpu().numpy(),
                "pair": r["transcript_id"],
                "spk": r["spk"],
                "src": r["src"],
            })

    for handle in handles:
        handle.remove()

    for key, layer_rows in by_layer.items():
        p = save_layer_npz(out_dir, tag, key, layer_rows)
        print(f"[alm] wrote {p}  N={len(layer_rows)}")


if __name__ == "__main__":
    sys.exit(main())
