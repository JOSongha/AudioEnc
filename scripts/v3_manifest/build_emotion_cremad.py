#!/usr/bin/env python
"""Build CREMA-D emotion MCQA manifest (v3).

Filename: <ActorID>_<SentenceCode>_<Emotion>_<Intensity>.wav
Emotion codes: ANG=angry, DIS=disgust, FEA=fear, HAP=happy, NEU=neutral, SAD=sad
"""
import json
import os
import random
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/emotion_raw/CREMA-D/AudioWAV")
OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "cremad"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

EMO_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def main():
    wavs = sorted(ROOT.glob("*.wav"))
    print(f"[cremad] found {len(wavs)} wavs in {ROOT}")

    items = []
    skipped = 0
    for wp in wavs:
        parts = wp.stem.split("_")
        if len(parts) < 3:
            skipped += 1
            continue
        emo_code = parts[2]
        label = EMO_MAP.get(emo_code)
        if label is None:
            skipped += 1
            continue
        items.append((str(wp), label))

    classes = sorted(set(EMO_MAP.values()))
    print(f"[cremad] valid={len(items)} skipped={skipped} classes={len(classes)}: {classes}")

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
            out_path = OUT_DIR / f"emotion_cremad_{shard_idx:04d}.jsonl"
            f = open(out_path, "w")
            print(f"[cremad] writing -> {out_path}")
            shard_idx += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    if f is not None:
        f.close()
    print(f"[cremad] DONE wrote={n} classes={len(classes)}")


if __name__ == "__main__":
    main()
