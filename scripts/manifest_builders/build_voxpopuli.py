#!/usr/bin/env python
"""Build VoxPopuli (en transcribed) audio_asr shards directly from nubes.

Source layout on nubes:
    /datasets/public/16kHz/voxpopuli/train/<id>.flac (audio)
    /datasets/public/16kHz/voxpopuli/train/<id>.txt  (transcript)

182,466 (.flac, .txt) pairs total. Pattern matches build_gigaspeech.py
(recursive list + parallel txt fetch + shard emit) but no subsampling
(we use 100 % of train).

No /mnt/ddn/users/<person>/ caches. No /mnt/ddn/omni_dataset/audio/voxpopuli/
jsonl. Only nubes.

Runtime: ~360k transcript fetches with 64 workers ≈ 10-20 minutes.

Usage:
    python -m scripts.manifest_builders.build_voxpopuli \\
        --out /mnt/tmp/datasets/manifests/v6/audio_asr --workers 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manifest_builders._nubes_helper import (  # noqa: E402
    fetch_text_files_parallel, flac_to_txt_path, list_files_recursive,
)


PREFIX_TRAIN = "datasets/public/16kHz/voxpopuli/train/"
OUT_DEFAULT = Path("/mnt/tmp/datasets/manifests/v6/audio_asr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--shard-size", type=int, default=88000)
    ap.add_argument("--prefix", default="voxpopuli")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[voxpopuli] enumerating nubes:{PREFIX_TRAIN} ...", flush=True)
    flac_paths: list[str] = []
    for full in list_files_recursive(PREFIX_TRAIN, suffix=".flac"):
        flac_paths.append(full)
        if len(flac_paths) % 50_000 == 0:
            print(f"  listed {len(flac_paths):,} flac entries", flush=True)
    print(f"[voxpopuli] total flac entries: {len(flac_paths):,}", flush=True)

    txt_paths = [flac_to_txt_path(p) for p in flac_paths]
    flac_by_txt = dict(zip(txt_paths, flac_paths))

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
            "source": "voxpopuli",
        }
        shard_buf.append(json.dumps(out_row, ensure_ascii=False) + "\n")
        n_done += 1
        if len(shard_buf) >= args.shard_size:
            flush()
        if n_done % 20_000 == 0:
            print(f"  fetched {n_done:,}/{len(flac_paths):,} "
                  f"(empty={n_empty:,})", flush=True)
    flush()

    print(f"[voxpopuli] DONE: enumerated={len(flac_paths):,} "
          f"fetched={n_done:,} empty={n_empty:,} shards={shard_idx}",
          flush=True)


if __name__ == "__main__":
    main()
