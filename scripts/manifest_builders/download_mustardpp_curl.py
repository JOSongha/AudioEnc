#!/usr/bin/env python
"""Download MUStARD++ utterance mp4s individually via direct GDrive URL (curl).

Bypasses gdown's session-based rate-limiting by using the public
`uc?export=download&id=<id>` endpoint directly. Some files may need a
"confirm=t" follow-up if Drive serves a virus-scan interstitial; we
detect HTML responses and retry with confirm=t.
"""
import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path("/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus")
ID_LIST = ROOT / "utterance_ids.txt"
OUT_DIR = ROOT / "videos" / "augmented_utterance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 8
RETRIES = 4


def is_mp4(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        # Quick HTML check
        if b"<html" in head.lower() or b"<!doctype" in head.lower():
            return False
        # mp4 signature: contains 'ftyp' near start
        return b"ftyp" in head
    except Exception:
        return False


def download_one(file_id: str, fname: str) -> tuple[str, bool, str]:
    out = OUT_DIR / fname
    if is_mp4(out):
        return fname, True, "already_exists"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    confirm_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    for attempt in range(RETRIES):
        for u in (url, confirm_url):
            try:
                cp = subprocess.run(
                    ["curl", "-sSL", "-o", str(out),
                     "-A", "Mozilla/5.0",
                     "--max-time", "120",
                     u],
                    check=False, capture_output=True, timeout=150,
                )
                if cp.returncode == 0 and is_mp4(out):
                    return fname, True, "ok"
            except subprocess.TimeoutExpired:
                pass
        # cleanup garbage
        if out.exists() and not is_mp4(out):
            out.unlink()
        time.sleep(1.5 ** attempt)
    return fname, False, "failed_after_retries"


def main():
    items = []
    with open(ID_LIST) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            items.append(tuple(parts))
    print(f"[curl_dl] total={len(items)} workers={WORKERS}", flush=True)

    ok = 0
    fail = 0
    skipped = 0
    fail_list = []
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
                fail_list.append((fname, status))
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                rate = (ok + skipped) / max(1, el)
                print(f"[curl_dl] {i+1}/{len(items)} ok={ok} skip={skipped} "
                      f"fail={fail} rate={rate:.2f}/s elapsed={el:.0f}s",
                      flush=True)
    print(f"[curl_dl] DONE ok={ok} skipped={skipped} fail={fail}")
    if fail_list:
        with open(ROOT / "failed_ids.txt", "w") as f:
            for name, st in fail_list:
                f.write(f"{name}\t{st}\n")
        print(f"[curl_dl] first 10 failures:")
        for name, st in fail_list[:10]:
            print(f"  {name}: {st}")


if __name__ == "__main__":
    main()
