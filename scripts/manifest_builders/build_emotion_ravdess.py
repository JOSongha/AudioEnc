#!/usr/bin/env python
"""Build RAVDESS emotion MCQA manifest (v3).

Filename: 03-01-XX-YY-ZZ-AA-BB.wav
Position 3 (1-indexed) is emotion code:
  01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
"""
import json
import os
import random
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/emotion_raw/RAVDESS")
OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "ravdess"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

EMO_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def main():
    # v6: canonical split 없는 source 는 통째로 학습 (v5 leak-fix 폐기). 24 actors 모두 사용.
    wavs = sorted(ROOT.glob("Actor_*/*.wav"))
    print(f"[ravdess] found {len(wavs)} wavs (24 actors all)")

    items = []
    skipped = 0
    for wp in wavs:
        parts = wp.stem.split("-")
        if len(parts) < 3:
            skipped += 1
            continue
        emo_code = parts[2]
        label = EMO_MAP.get(emo_code)
        if label is None:
            skipped += 1
            continue
        if not wp.exists():
            skipped += 1
            continue
        items.append((str(wp), label))

    classes = sorted(set(EMO_MAP.values()))
    print(f"[ravdess] valid={len(items)} skipped={skipped} classes={len(classes)}: {classes}")

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
            out_path = OUT_DIR / f"emotion_ravdess_{shard_idx:04d}.jsonl"
            f = open(out_path, "w")
            print(f"[ravdess] writing -> {out_path}")
            shard_idx += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    if f is not None:
        f.close()
    print(f"[ravdess] DONE wrote={n} classes={len(classes)}")


if __name__ == "__main__":
    main()
