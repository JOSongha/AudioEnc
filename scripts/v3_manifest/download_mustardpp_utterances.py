#!/usr/bin/env python
"""Download MUStARD++ utterance mp4s individually from GDrive (with retries).

Uses gdown's per-file API (more reliable than --folder under rate limits).
Reads the file ID list extracted from a previous --folder run's log
(utterance_ids.txt -- one "<id> <filename>" per line) and downloads each
mp4 to videos/augmented_utterance/<filename>, skipping files that already
exist with non-zero size.
"""
import os
import sys
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path("/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus")
ID_LIST = ROOT / "utterance_ids.txt"
OUT_DIR = ROOT / "videos" / "augmented_utterance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GDOWN = "/mnt/ddn/users/jos/miniforge3/envs/audio_lmf/bin/gdown"
MAX_RETRIES = 3
WORKERS = 4  # gdown rate-limits if too many parallel


def download_one(file_id: str, fname: str) -> tuple[str, bool, str]:
    out = OUT_DIR / fname
    if out.exists() and out.stat().st_size > 0:
        return fname, True, "already_exists"
    for attempt in range(MAX_RETRIES):
        try:
            cp = subprocess.run(
                [GDOWN, "--no-cookies", "-O", str(out), file_id],
                check=False, capture_output=True, timeout=180,
            )
            if cp.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return fname, True, "ok"
        except subprocess.TimeoutExpired:
            pass
        # Backoff
        time.sleep(2 ** attempt)
    # Cleanup zero-byte file if any
    if out.exists() and out.stat().st_size == 0:
        out.unlink()
    return fname, False, "failed_after_retries"


def main():
    items = []
    with open(ID_LIST) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            items.append(tuple(parts))
    print(f"[mustardpp_dl] total={len(items)} workers={WORKERS}")

    ok = 0
    fail = 0
    skipped = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_one, fid, fn): fn for fid, fn in items}
        for i, fut in enumerate(as_completed(futures)):
            fname, success, status = fut.result()
            if success:
                if status == "already_exists":
                    skipped += 1
                else:
                    ok += 1
            else:
                fail += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"[mustardpp_dl] {i+1}/{len(items)} ok={ok} skip={skipped} "
                      f"fail={fail} elapsed={elapsed:.0f}s", flush=True)

    print(f"[mustardpp_dl] DONE ok={ok} skipped={skipped} fail={fail}")


if __name__ == "__main__":
    main()
