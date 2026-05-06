"""Convert emotion_mcqa_manifest.jsonl → describe-format manifest.

Stage 2 used MCQA single-letter targets ("A", "B", ...) which biased the model
to fire EOS after very short outputs. For Stage 1 projector pretraining we want
natural-language emotion labels as targets so the model learns "long descriptive
output" alignment from the start.

Output rows:
    {"path": ..., "source": ..., "modality": "audio_emotion", "label": "happy"}
    (no `question` / `choices` / `answer` — omni_dataset.py picks the
     describe branch when MCQA fields are absent.)
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("/mnt/tmp/listen_analysis/train_manifest/emotion_mcqa_manifest.jsonl")
DST = Path("/mnt/tmp/listen_analysis/train_manifest/emotion_describe_manifest.jsonl")


def main():
    n_in = 0
    n_out = 0
    by_src = {}
    by_label = {}
    with SRC.open() as fin, DST.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            n_in += 1
            label = r.get("label")
            if not label:
                continue
            out = {
                "path": r["path"],
                "source": r["source"],
                "modality": "audio_emotion",
                "label": str(label).strip(),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
            by_src[out["source"]] = by_src.get(out["source"], 0) + 1
            by_label[out["label"]] = by_label.get(out["label"], 0) + 1
    print(f"in: {n_in}  out: {n_out}")
    print("by source:")
    for k, v in sorted(by_src.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v}")
    print("by label:")
    for k, v in sorted(by_label.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v}")
    print(f"-> {DST}")


if __name__ == "__main__":
    main()
