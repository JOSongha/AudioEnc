#!/usr/bin/env python
"""Build LAION Audiostock manifest (T4).

Source: /mnt/tmp/datasets/laion_audiostock/
  - audio/<id>.mp3 (already extracted)
  - meta.csv (id, status, size, caption, url)

Filters:
  - status == "ok"
  - non-empty caption
  - audio file exists and on-disk size > 1024 bytes

Output: /mnt/tmp/datasets/manifests/v3/laion_audiostock_0000.jsonl
Row format:
  {"modality": "audio_env_sound", "source": "laion_audiostock",
   "audio_path": ".../<id>.mp3", "captions": ["<caption>"]}
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

META_CSV = Path("/mnt/tmp/datasets/laion_audiostock/meta.csv")
AUDIO_DIR = Path("/mnt/tmp/datasets/laion_audiostock/audio")
OUT_PATH = Path("/mnt/tmp/datasets/manifests/v3/laion_audiostock_0000.jsonl")
MIN_BYTES = 1024


def main() -> int:
    if not META_CSV.exists():
        print(f"[ERR] meta.csv not found: {META_CSV}", file=sys.stderr)
        return 1
    if not AUDIO_DIR.is_dir():
        print(f"[ERR] audio dir not found: {AUDIO_DIR}", file=sys.stderr)
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_status_bad = 0
    n_caption_empty = 0
    n_missing_file = 0
    n_too_small = 0
    n_written = 0

    # CSV may have caption containing commas; csv.DictReader handles quoting.
    with META_CSV.open("r", newline="", encoding="utf-8") as f, \
         OUT_PATH.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            status = (row.get("status") or "").strip()
            if status != "ok":
                n_status_bad += 1
                continue
            caption = (row.get("caption") or "").strip()
            if not caption:
                n_caption_empty += 1
                continue
            audio_id = (row.get("id") or "").strip()
            if not audio_id:
                n_caption_empty += 1
                continue
            audio_path = AUDIO_DIR / f"{audio_id}.mp3"
            try:
                st = audio_path.stat()
            except FileNotFoundError:
                n_missing_file += 1
                continue
            except OSError:
                n_missing_file += 1
                continue
            if st.st_size <= MIN_BYTES:
                n_too_small += 1
                continue

            obj = {
                "modality": "audio_env_sound",
                "source": "laion_audiostock",
                "audio_path": str(audio_path),
                "captions": [caption],
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[audiostock] meta rows={n_total}")
    print(f"[audiostock]   status!=ok       : {n_status_bad}")
    print(f"[audiostock]   empty caption    : {n_caption_empty}")
    print(f"[audiostock]   missing audio    : {n_missing_file}")
    print(f"[audiostock]   size<={MIN_BYTES} bytes: {n_too_small}")
    print(f"[audiostock]   written          : {n_written}")
    print(f"[audiostock] -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
