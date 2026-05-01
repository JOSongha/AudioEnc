#!/usr/bin/env python
"""Build DailyTalk emotion MCQA manifest (v3).

metadata.json:
  { "<dialog_idx>": { "<utterance_idx>": { "emotion": ..., "speaker": ..., ... } } }
Audio path: data/<dialog_idx>/<utterance_idx>_<speaker>_d<dialog_idx>.wav
Emotions: anger, disgust, fear, happiness, no emotion, sadness, surprise
"""
import json
import os
import random
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/emotion_raw/DailyTalk/dailytalk")
META = ROOT / "metadata.json"
DATA = ROOT / "data"
OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "dailytalk"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def main():
    meta = json.load(open(META))
    items = []
    skipped_unknown = 0
    skipped_missing = 0
    classes_seen = set()
    for dialog_id, utts in meta.items():
        for utt_id, info in utts.items():
            label = info.get("emotion")
            if not label:
                skipped_unknown += 1
                continue
            classes_seen.add(label)
            speaker = info.get("speaker")
            dialog_idx = info.get("dialog_idx")
            utterance_idx = info.get("utterance_idx")
            if speaker is None or dialog_idx is None or utterance_idx is None:
                skipped_missing += 1
                continue
            wav_path = DATA / str(dialog_idx) / f"{utterance_idx}_{speaker}_d{dialog_idx}.wav"
            if not wav_path.exists():
                skipped_missing += 1
                continue
            items.append((str(wav_path), label))

    classes = sorted(classes_seen)
    print(f"[dailytalk] kept={len(items)} skipped_missing={skipped_missing} skipped_unknown={skipped_unknown}")
    print(f"[dailytalk] classes={len(classes)}: {classes}")

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
            out_path = OUT_DIR / f"emotion_dailytalk_{shard_idx:04d}.jsonl"
            f = open(out_path, "w")
            print(f"[dailytalk] writing -> {out_path}")
            shard_idx += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    if f is not None:
        f.close()
    print(f"[dailytalk] DONE wrote={n} classes={len(classes)}")


if __name__ == "__main__":
    main()
