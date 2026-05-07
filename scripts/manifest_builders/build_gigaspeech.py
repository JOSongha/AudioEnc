#!/usr/bin/env python
"""Build GigaSpeech audio_asr shards (50 % random subsample) directly from nubes.

Source layout on nubes (`hyperscaleai-audiollm` bucket):
    /datasets/public/16kHz/gigaspeech/train/<id>.flac   (audio)
    /datasets/public/16kHz/gigaspeech/train/<id>.txt    (transcript, ~80 B each)

8,256,276 (<id>.flac, <id>.txt) pairs total. We:
  1. List the train directory recursively via the gateway list API
     (`?dir=...&max-contents=1000` + `X-Continuation-Token` pagination).
  2. seed=11 random 50 % subsample of .flac entries (~4,128,138 rows /
     5,000 h target).
  3. Fetch each kept .txt in parallel (default 64 workers).
  4. Emit `{"nubes_path": "...flac", "text": "...", "modality":
     "audio_asr", "source": "gigaspeech"}` rows in 88K-row shards under
     /mnt/tmp/datasets/manifests/v6/audio_asr/gigaspeech_NNNN.jsonl.

No dependency on /mnt/ddn/users/<person>/ caches. No dependency on
/mnt/ddn/omni_dataset/audio/gigaspeech/ jsonl. Only nubes and CPU/network.

Runtime: per-file .txt fetch over 4 M kept rows takes ~2-3 hours with
64 workers. Listing is fast (~10 minutes for 8 M entries with 1K page
size). Use --resume to skip shards that already exist.

Usage:
    python -m scripts.manifest_builders.build_gigaspeech \\
        --out /mnt/tmp/datasets/manifests/v6/audio_asr \\
        --sample-rate 0.5 --seed 11 --workers 64
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manifest_builders._nubes_helper import (  # noqa: E402
    fetch_text_files_parallel, flac_to_txt_path, list_files_recursive,
)


PREFIX_TRAIN = "datasets/public/16kHz/gigaspeech/train/"
OUT_DEFAULT = Path("/mnt/tmp/datasets/manifests/v6/audio_asr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--sample-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--shard-size", type=int, default=88000)
    ap.add_argument("--prefix", default="gigaspeech")
    ap.add_argument("--workers", type=int, default=64,
                    help="Parallel transcript fetch workers.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip shards that already exist on disk.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # 1) Enumerate flac entries on nubes.
    print(f"[gigaspeech] enumerating nubes:{PREFIX_TRAIN} ...", flush=True)
    flac_paths: list[str] = []
    for full in list_files_recursive(PREFIX_TRAIN, suffix=".flac"):
        flac_paths.append(full)
        if len(flac_paths) % 500_000 == 0:
            print(f"  listed {len(flac_paths):,} flac entries", flush=True)
    print(f"[gigaspeech] total flac entries: {len(flac_paths):,}", flush=True)

    # 2) Sample.
    sampled = [p for p in flac_paths if rng.random() < args.sample_rate]
    print(f"[gigaspeech] sampled {len(sampled):,} "
          f"(ratio={len(sampled)/max(len(flac_paths),1):.4f})", flush=True)

    # 3) Fetch transcripts and emit shards.
    txt_paths = [flac_to_txt_path(p) for p in sampled]
    flac_by_txt = dict(zip(txt_paths, sampled))

    shard_idx = 0
    shard_buf: list[str] = []

    def flush():
        nonlocal shard_idx, shard_buf
        if not shard_buf:
            return
        out_path = args.out / f"{args.prefix}_{shard_idx:04d}.jsonl"
        if args.resume and out_path.exists():
            print(f"  shard {shard_idx:04d}: skip (exists)", flush=True)
            shard_buf = []
            shard_idx += 1
            return
        if not args.dry_run:
            with open(out_path, "w") as f:
                f.writelines(shard_buf)
        print(f"  shard {shard_idx:04d}: {len(shard_buf):,} rows -> {out_path}",
              flush=True)
        shard_idx += 1
        shard_buf = []

    n_done = 0
    n_empty = 0
    for txt_path, text in fetch_text_files_parallel(
            txt_paths, workers=args.workers):
        flac = flac_by_txt[txt_path]
        if not text:
            n_empty += 1
        nubes_full = f"hyperscaleai-audiollm/{flac}"
        out_row = {
            "nubes_path": nubes_full,
            "text": text,
            "modality": "audio_asr",
            "source": "gigaspeech",
        }
        shard_buf.append(json.dumps(out_row, ensure_ascii=False) + "\n")
        n_done += 1
        if len(shard_buf) >= args.shard_size:
            flush()
        if n_done % 100_000 == 0:
            print(f"  fetched {n_done:,}/{len(sampled):,} "
                  f"(empty={n_empty:,})", flush=True)
    flush()

    print(f"[gigaspeech] DONE: enumerated={len(flac_paths):,} "
          f"sampled={len(sampled):,} fetched={n_done:,} "
          f"empty={n_empty:,} shards={shard_idx}", flush=True)


if __name__ == "__main__":
    main()
