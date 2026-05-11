#!/usr/bin/env python
"""Build MUStARD++ emotion MCQA manifest (v3).

Source CSV: mustard++_text.csv
We use only the *_u rows (single utterance). Emotion = Explicit_Emotion.

Audio is extracted from per-utterance mp4 videos (when available) into
/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/audio_wav/<KEY>.wav using
ffmpeg. Rows with no available source mp4 are skipped.
"""
import csv
import json
import os
import random
import subprocess
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus")
CSV_PATH = ROOT / "mustard++_text.csv"
CTX_DIR = ROOT / "videos" / "final_context_videos"
AUG_DIR = ROOT / "videos" / "augmented_utterance"
AUDIO_DIR = ROOT / "audio_wav"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "mustardpp"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
FFMPEG = "ffmpeg"


def extract_audio(mp4_path: Path, wav_path: Path) -> bool:
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return True
    try:
        cp = subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(mp4_path),
             "-ac", "1", "-ar", "16000", str(wav_path)],
            check=False, capture_output=True, timeout=120,
        )
        return cp.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0
    except Exception:
        return False


def main():
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            key = row["KEY"]
            if not key.endswith("_u"):
                continue
            label = row.get("Explicit_Emotion", "").strip()
            if not label:
                continue
            rows.append((key, label.lower()))

    classes = sorted({lab for _, lab in rows})
    print(f"[mustardpp] labeled rows={len(rows)} classes={len(classes)}: {classes}")

    items = []
    missing = 0
    for key, label in rows:
        wav_path = AUDIO_DIR / f"{key}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 0:
            items.append((str(wav_path), label))
            continue
        # Find candidate mp4
        mp4_aug = AUG_DIR / f"{key}.mp4"
        mp4_ctx = CTX_DIR / f"{key}.mp4"
        candidate = None
        if mp4_aug.exists():
            candidate = mp4_aug
        elif mp4_ctx.exists():
            candidate = mp4_ctx
        if candidate is None:
            missing += 1
            continue
        if extract_audio(candidate, wav_path):
            items.append((str(wav_path), label))
        else:
            missing += 1

    print(f"[mustardpp] kept={len(items)} missing/extract_failed={missing}")

    out_rows = []
    for idx, (ap, label) in enumerate(items):
        rng = random.Random(42 + idx)
        shuffled = list(classes)
        rng.shuffle(shuffled)
        choices = [f"{LETTERS[i]}. {c}" for i, c in enumerate(shuffled)]
        answer = LETTERS[shuffled.index(label)]
        out_rows.append({
            "modality": "audio_emotion",
            "source": SOURCE,
            "audio_path": ap,
            "question": QUESTION,
            "choices": choices,
            "answer": answer,
        })

    n = 0
    shard_idx = 0
    f = None
    for i, row in enumerate(out_rows):
        if i % SHARD_SIZE == 0:
            if f is not None:
                f.close()
            out_path = OUT_DIR / f"emotion_mustardpp_{shard_idx:04d}.jsonl"
            f = open(out_path, "w")
            print(f"[mustardpp] writing -> {out_path}")
            shard_idx += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    if f is not None:
        f.close()
    print(f"[mustardpp] DONE wrote={n} classes={len(classes)}")


if __name__ == "__main__":
    main()
