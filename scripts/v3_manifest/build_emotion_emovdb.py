#!/usr/bin/env python
"""Build EmoV-DB emotion MCQA manifest (v3).

Layout: <speaker>/<Emotion>/<file>.wav
Speakers: bea, jenie, josh, sam
Emotions: Amused, Angry, Disgusted, Neutral, Sleepy
"""
import json
import os
import random
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/emotion_raw/EmoV-DB")
OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "emovdb"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
SPEAKERS = ["bea", "jenie", "josh", "sam"]
# eval_source_emotion.load_emov_jenie holds out the Jenie speaker; exclude here.
HELD_OUT_SPEAKERS = {"jenie"}
EMO_DIRS = ["Amused", "Angry", "Disgusted", "Neutral", "Sleepy"]
# Lowercase normalized labels
EMO_MAP = {
    "Amused": "amused",
    "Angry": "angry",
    "Disgusted": "disgusted",
    "Neutral": "neutral",
    "Sleepy": "sleepy",
}


def main():
    items = []
    n_held_out = 0
    for sp in SPEAKERS:
        if sp in HELD_OUT_SPEAKERS:
            held = sum(1 for emo in EMO_DIRS for _ in (ROOT / sp / emo).glob("*.wav") if (ROOT / sp / emo).is_dir())
            n_held_out += held
            continue
        for emo_dir in EMO_DIRS:
            d = ROOT / sp / emo_dir
            if not d.is_dir():
                continue
            label = EMO_MAP[emo_dir]
            for wp in sorted(d.glob("*.wav")):
                items.append((str(wp), label))
    print(f"[emovdb] held-out (eval) speakers={sorted(HELD_OUT_SPEAKERS)}, n_held={n_held_out}")
    classes = sorted(set(EMO_MAP.values()))
    print(f"[emovdb] valid={len(items)} classes={len(classes)}: {classes}")

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
            out_path = OUT_DIR / f"emotion_emovdb_{shard_idx:04d}.jsonl"
            f = open(out_path, "w")
            print(f"[emovdb] writing -> {out_path}")
            shard_idx += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    if f is not None:
        f.close()
    print(f"[emovdb] DONE wrote={n} classes={len(classes)}")


if __name__ == "__main__":
    main()
