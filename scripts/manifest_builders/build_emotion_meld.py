#!/usr/bin/env python
"""Build MELD emotion MCQA manifest (v6, jos own raw csv).

Source CSV: /mnt/tmp/datasets/emotion_raw/MELD/MELD.Raw/{train,dev}_sent_emo.csv
Audio:     /mnt/tmp/datasets/emotion_raw/MELD/audio/{train,dev}/dia<X>_utt<Y>.wav

train+dev only. test split is held out (eval_source_emotion.load_meld_test
reads test_sent_emo.csv + audio/test/ from the same root).

Replaces the prior shard-based builder that depended on
`/mnt/ddn/users/sehyun/.../meld_{train,dev}_shard_*.jsonl` for split metadata.

Output: $V6_OUT (default /mnt/tmp/datasets/manifests/v6_raw) / emotion_meld_<NNNN>.jsonl
"""
import csv
import json
import os
import random
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/emotion_raw/MELD")
SPLITS = ["train", "dev"]   # test reserved for eval (eval_source_emotion)
OUT_DIR = Path(os.environ.get("V6_OUT", "/mnt/tmp/datasets/manifests/v6_raw"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "meld"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def main():
    items = []
    skipped_missing = 0
    skipped_unknown = 0
    for split in SPLITS:
        csvp = ROOT / "MELD.Raw" / f"{split}_sent_emo.csv"
        adir = ROOT / "audio" / split
        with open(csvp) as fh:
            for r in csv.DictReader(fh):
                label = (r.get("Emotion") or "").strip().lower()
                if not label:
                    skipped_unknown += 1
                    continue
                stem = f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}"
                wav = adir / f"{stem}.wav"
                if not wav.exists():
                    skipped_missing += 1
                    continue
                items.append((str(wav), label))

    classes = sorted({lab for _, lab in items})
    print(f"[meld] kept={len(items)} skipped_missing={skipped_missing} "
          f"skipped_unknown={skipped_unknown} classes={len(classes)}: {classes}")

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
            out_path = OUT_DIR / f"emotion_meld_{shard_idx:04d}.jsonl"
            f = open(out_path, "w")
            print(f"[meld] writing -> {out_path}")
            shard_idx += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    if f is not None:
        f.close()
    print(f"[meld] DONE wrote={n} classes={len(classes)}")


if __name__ == "__main__":
    main()
