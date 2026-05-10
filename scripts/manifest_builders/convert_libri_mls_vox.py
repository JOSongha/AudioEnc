#!/usr/bin/env python
"""DEPRECATED — sehyun-shard based. Use the per-source nubes-direct builders.

This script reads from `/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/
external/datasets/libri_mls_vox/shard_*.jsonl` (sehyun's personal cache),
which violates the "nubes-only" rule. Replacement builders that source from
nubes directly:

    build_mls.py        — MLS English (transcripts.txt single file)
    build_voxpopuli.py  — VoxPopuli en (per-file .txt fetch)
    build_libritts_r.py — LibriTTS-R (TODO: nubes /libriTTS/ recursive)

Existing v6 audio_asr/libri_mls_vox_*.jsonl shards (128 shards / 11.35M rows)
were built with this script and remain valid (nubes_path values are correct).
For future v6/v7 manifest rebuilds run the new per-source builders.

Original behaviour (kept for reproducibility / archival):
  Source : /mnt/ddn/users/sehyun/.../libri_mls_vox/shard_*.jsonl
  Output : /mnt/tmp/datasets/manifests/v3/libri_mls_vox_<NNNN>.jsonl
  Adds modality="audio_asr" + source inferred from nubes_path. 128 → 128
  shards, 1:1.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

SRC_DIR = Path(
    "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/libri_mls_vox"
)
OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
SHARD_GLOB = "shard_*.jsonl"


def infer_source(nubes_path: str) -> str:
    p_lower = nubes_path.lower()
    if "librispeech" in p_lower:
        return "librispeech"
    if "/mls/" in p_lower:
        return "mls"
    if "/voxpopuli/" in p_lower:
        return "voxpopuli"
    return "unknown_asr"


# Match shard_<digits>.jsonl, capture digits.
SHARD_RE = re.compile(r"^shard_(\d+)\.jsonl$")


def shard_index_from_name(name: str) -> int | None:
    m = SHARD_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def main() -> int:
    if not SRC_DIR.is_dir():
        print(f"[ERR] source dir not found: {SRC_DIR}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(SRC_DIR.glob(SHARD_GLOB))
    if not shard_paths:
        print(f"[ERR] no shards matched {SRC_DIR}/{SHARD_GLOB}", file=sys.stderr)
        return 1

    totals = {
        "librispeech": 0,
        "mls": 0,
        "voxpopuli": 0,
        "unknown_asr": 0,
    }
    n_rows = 0
    n_shards_out = 0

    for shard_path in shard_paths:
        idx = shard_index_from_name(shard_path.name)
        if idx is None:
            print(f"[WARN] skipping unparseable name: {shard_path.name}", file=sys.stderr)
            continue
        out_path = OUT_DIR / f"libri_mls_vox_{idx:04d}.jsonl"

        with shard_path.open("r", encoding="utf-8") as fin, \
             out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                obj = json.loads(line)
                nubes_path = obj.get("nubes_path", "")
                src = infer_source(nubes_path)
                totals[src] = totals.get(src, 0) + 1
                # Preserve original fields, prepend modality/source.
                new_obj = {
                    "modality": "audio_asr",
                    "source": src,
                }
                # Keep existing keys as-is (nubes_path, text, anything else).
                for k, v in obj.items():
                    new_obj[k] = v
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                n_rows += 1
        n_shards_out += 1

    print(f"[libri_mls_vox] shards in : {len(shard_paths)}")
    print(f"[libri_mls_vox] shards out: {n_shards_out}")
    print(f"[libri_mls_vox] rows total: {n_rows}")
    for k in ("librispeech", "mls", "voxpopuli", "unknown_asr"):
        print(f"[libri_mls_vox]   {k:13s}: {totals.get(k, 0)}")
    print(f"[libri_mls_vox] -> {OUT_DIR}/libri_mls_vox_*.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
