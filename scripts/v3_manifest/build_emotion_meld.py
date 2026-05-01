#!/usr/bin/env python
"""Build MELD emotion MCQA manifest (v3)."""
import json
import os
import random
from pathlib import Path

SHARDS = [
    "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/emotion/meld_train_shard_00000.jsonl",
    "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/emotion/meld_dev_shard_00000.jsonl",
]

OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "meld"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def main():
    rows = []
    for fp in SHARDS:
        with open(fp) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    classes = sorted({r["response"] for r in rows})
    print(f"[meld] loaded {len(rows)} rows, {len(classes)} classes: {classes}")

    out_rows = []
    skipped_missing = 0
    skipped_unknown = 0
    for idx, r in enumerate(rows):
        ap = r.get("audio_path")
        label = r.get("response")
        if not label or label not in classes:
            skipped_unknown += 1
            continue
        if not ap or not os.path.exists(ap):
            skipped_missing += 1
            continue
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

    print(f"[meld] kept {len(out_rows)} | skipped missing {skipped_missing} | unknown {skipped_unknown}")

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
