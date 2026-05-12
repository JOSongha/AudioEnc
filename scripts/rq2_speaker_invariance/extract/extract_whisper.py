#!/usr/bin/env python
"""Extract per-layer Whisper encoder hidden states.

For each pair row we get [L+1, T=1500, D] (incl. embedding output). We mean-pool
over the first `valid_frames` frames (where `valid_frames = ceil(num_samples /
hop)`, capped at 1500) so that 30s-of-silence padding does not dilute the mean.

Writes one .npz per encoder layer (including layer 0 = post-conv input).

Usage:
  python extract_whisper.py --model openai/whisper-large-v2 --pairs ... --out emb/whisper_large_v2
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel

from _io import load_pairs, load_wav, mean_pool, save_layer_npz

HOP = 320  # samples per encoder frame after conv stride
MAX_ENC_FRAMES = 1500


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id, e.g. openai/whisper-large-v2")
    ap.add_argument("--pairs", required=True, help="path to all_pairs.jsonl")
    ap.add_argument("--out", required=True, help="output dir for .npz files")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--tag", default=None, help="model tag (defaults to model basename)")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    out_dir = Path(args.out)

    feat_ex = WhisperFeatureExtractor.from_pretrained(args.model)
    model = WhisperModel.from_pretrained(args.model, dtype=dtype).encoder.to(args.device).eval()
    n_layers = model.config.encoder_layers
    print(f"[whisper] {args.model}  layers={n_layers}  d={model.config.d_model}")

    rows = load_pairs(args.pairs)
    print(f"[whisper] rows={len(rows)}")

    # per layer collected rows
    by_layer: dict[int, list[dict]] = {i: [] for i in range(n_layers + 1)}

    for b0 in tqdm(range(0, len(rows), args.batch_size)):
        batch = rows[b0 : b0 + args.batch_size]
        wavs = [load_wav(r["audio_path"], target_sr=16000, max_sec=30.0) for r in batch]
        valid_frames = [min(MAX_ENC_FRAMES, math.ceil(len(w) / HOP)) for w in wavs]
        feats = feat_ex(wavs, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(args.device, dtype=dtype)

        out = model(feats, output_hidden_states=True, return_dict=True)
        # out.hidden_states: tuple of len (n_layers + 1), each [B, 1500, D]
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
        print(f"[whisper] wrote {p}  N={len(layer_rows)}")


if __name__ == "__main__":
    main()
