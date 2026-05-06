#!/usr/bin/env python
"""Extract 16kHz mono WAVs from MUStARD++ utterance mp4s in parallel.

Source: /mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/videos/augmented_utterance/<KEY>_u.mp4
Output: /mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/audio_wav/<KEY>_u.wav
Skips wavs that already exist with non-zero size.
"""
import os
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path("/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus")
SRC = ROOT / "videos" / "augmented_utterance"
DST = ROOT / "audio_wav"
DST.mkdir(parents=True, exist_ok=True)

FFMPEG = "/mnt/ddn/users/jos/miniforge3/bin/ffmpeg"
WORKERS = 8


def extract(mp4: Path) -> tuple[str, bool, str]:
    wav = DST / (mp4.stem + ".wav")
    if wav.exists() and wav.stat().st_size > 0:
        return mp4.name, True, "already_exists"
    try:
        cp = subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(mp4),
             "-ac", "1", "-ar", "16000", str(wav)],
            check=False, capture_output=True, timeout=120,
        )
        if cp.returncode == 0 and wav.exists() and wav.stat().st_size > 0:
            return mp4.name, True, "ok"
        if wav.exists() and wav.stat().st_size == 0:
            wav.unlink()
        return mp4.name, False, f"ffmpeg_rc={cp.returncode}"
    except subprocess.TimeoutExpired:
        if wav.exists() and wav.stat().st_size == 0:
            wav.unlink()
        return mp4.name, False, "timeout"
    except Exception as e:
        return mp4.name, False, f"exc:{e}"


def main():
    mp4s = sorted(SRC.glob("*_u.mp4"))
    print(f"[extract] mp4_count={len(mp4s)} workers={WORKERS}")
    ok = 0
    skip = 0
    fail = 0
    fail_list = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(extract, p): p.name for p in mp4s}
        for i, fut in enumerate(as_completed(futures)):
            name, success, status = fut.result()
            if success:
                if status == "already_exists":
                    skip += 1
                else:
                    ok += 1
            else:
                fail += 1
                fail_list.append((name, status))
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"[extract] {i+1}/{len(mp4s)} ok={ok} skip={skip} "
                      f"fail={fail} elapsed={el:.0f}s", flush=True)
    print(f"[extract] DONE ok={ok} skip={skip} fail={fail}")
    if fail_list:
        print(f"[extract] first 10 failures:")
        for name, st in fail_list[:10]:
            print(f"  {name}: {st}")


if __name__ == "__main__":
    main()
