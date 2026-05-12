#!/usr/bin/env python
"""Acoustic-encoder baseline (HuBERT / WavLM / wav2vec2-style).

For models that take raw 16 kHz waveform via `AutoFeatureExtractor` and expose
`output_hidden_states=True`. Default is `facebook/hubert-base-ls960`.

We treat this as the "acoustic encoder" axis in the cross-encoder comparison:
no ASR supervision → representations are more acoustic/phonetic than lexical.

Usage:
  python extract_ssl.py --model facebook/hubert-base-ls960 --pairs ... --out emb/hubert_base
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

from _io import load_pairs, load_wav, mean_pool, save_layer_npz


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/hubert-base-ls960")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max_sec", type=float, default=20.0,
                    help="cap waveform length; SSL models do not need 30s padding")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    out_dir = Path(args.out)

    feat_ex = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"[ssl] {args.model}  layers={n_layers}  d={model.config.hidden_size}")

    rows = load_pairs(args.pairs)
    print(f"[ssl] rows={len(rows)}")

    by_layer: dict[int, list[dict]] = {i: [] for i in range(n_layers + 1)}

    for b0 in tqdm(range(0, len(rows), args.batch_size)):
        batch = rows[b0 : b0 + args.batch_size]
        wavs = [load_wav(r["audio_path"], target_sr=16000, max_sec=args.max_sec) for r in batch]
        inputs = feat_ex(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(args.device, dtype=dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)

        out = model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # compute valid frame count per sample. HuBERT/WavLM/W2V2 share ~320 hop.
        if attention_mask is not None:
            valid_samples = attention_mask.sum(dim=1).tolist()
        else:
            valid_samples = [len(w) for w in wavs]
        T_out = out.hidden_states[0].size(1)
        # downsample ratio from input samples to output frames
        ratio = input_values.size(1) / T_out
        valid_frames = [min(T_out, max(1, int(vs / ratio))) for vs in valid_samples]

        for li, h in enumerate(out.hidden_states):
            h = h.float()
            for bi, r in enumerate(batch):
                emb = mean_pool(h[bi], valid_len=valid_frames[bi]).cpu().numpy()
                by_layer[li].append({
                    "emb": emb,
                    "pair": r["transcript_id"],
                    "spk": r["spk"],
                    "src": r["src"],
                })

    for li, layer_rows in by_layer.items():
        layer_tag = f"enc.L{li:02d}"
        p = save_layer_npz(out_dir, tag, layer_tag, layer_rows)
        print(f"[ssl] wrote {p}  N={len(layer_rows)}")


if __name__ == "__main__":
    main()
