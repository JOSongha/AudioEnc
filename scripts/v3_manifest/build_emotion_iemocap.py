#!/usr/bin/env python
"""Build IEMOCAP emotion MCQA manifest (v3).

Source shards: existing emotion_classify rows.
Output: /mnt/tmp/datasets/manifests/v3/emotion_iemocap_<NNNN>.jsonl
"""
import json
import os
import random
from pathlib import Path

SHARDS = [
    "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/emotion/iemocap_shard_00000.jsonl",
    "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/emotion/iemocap_shard_00001.jsonl",
    "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/emotion/iemocap_shard_00002.jsonl",
]

OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "iemocap"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def load_rows():
    rows = []
    for fp in SHARDS:
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_rows()
    classes = sorted({r["response"] for r in rows})
    print(f"[iemocap] loaded {len(rows)} rows, {len(classes)} classes: {classes}")

    # The shards reference /mnt/ddn/users/sehyun/CACHE/iemocap/wav/<file>.wav,
    # but on this filesystem the actual IEMOCAP audio lives at
    # /mnt/ddn/kyudan/IEMOCAP/data/<file>.wav. Remap accordingly.
    REMAP_FROM = "/mnt/ddn/users/sehyun/CACHE/iemocap/wav/"
    REMAP_TO = "/mnt/ddn/kyudan/IEMOCAP/data/"

    out_rows = []
    skipped_missing = 0
    skipped_unknown = 0
    for idx, r in enumerate(rows):
        ap = r.get("audio_path")
        if ap and ap.startswith(REMAP_FROM):
            ap = REMAP_TO + ap[len(REMAP_FROM):]
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

    print(f"[iemocap] kept {len(out_rows)} | skipped missing {skipped_missing} | unknown {skipped_unknown}")

    # Write shards
    n = 0
    shard_idx = 0
    f = None
    for i, row in enumerate(out_rows):
        if i % SHARD_SIZE == 0:
            if f is not None:
                f.close()
            out_path = OUT_DIR / f"emotion_iemocap_{shard_idx:04d}.jsonl"
            f = open(out_path, "w")
            print(f"[iemocap] writing -> {out_path}")
            shard_idx += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    if f is not None:
        f.close()
    print(f"[iemocap] DONE wrote={n} classes={len(classes)}")


if __name__ == "__main__":
    main()
