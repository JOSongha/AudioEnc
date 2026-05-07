#!/usr/bin/env python
"""Build IEMOCAP emotion MCQA manifest (v6, jos own download).

Source: /mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release/
Sessions 1-4 = training pool. Session 5 = Leave-session-out held-out (excluded).

EmoEvaluation/*.txt format (per dialog):
  [start - end]\t<utt_id>\t<emo_code>\t[V, A, D]

Codes mapped to native 10-class set:
  neu/ang/hap/sad/fea/sur/dis/exc/fru/oth → neutral/angry/happy/sad/fear/surprise/disgust/excited/frustrated/other.
  'xxx' (no annotator consensus) is dropped.

Output: $V6_OUT (default /mnt/tmp/datasets/manifests/v6_raw) / emotion_iemocap_<NNNN>.jsonl
"""
import json
import os
import random
import re
from pathlib import Path

ROOT = Path("/mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release")
TRAIN_SESSIONS = [1, 2, 3, 4]   # Session 5 held out for LSO eval
OUT_DIR = Path(os.environ.get("V6_OUT", "/mnt/tmp/datasets/manifests/v6_raw"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "iemocap"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

CODE_MAP = {
    "neu": "neutral",
    "ang": "angry",
    "hap": "happy",
    "sad": "sad",
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",
    "exc": "excited",
    "fru": "frustrated",
    "oth": "other",
}
LINE_RE = re.compile(r"^\[\s*[\d.]+\s*-\s*[\d.]+\s*\]\s+(\S+)\s+(\S+)\s+\[")


def main():
    items = []
    skipped_xxx = 0
    skipped_missing = 0
    held_out = 0
    for s in [1, 2, 3, 4, 5]:
        eval_dir = ROOT / f"Session{s}" / "dialog" / "EmoEvaluation"
        for label_fp in sorted(eval_dir.glob("*.txt")):
            dialog_id = label_fp.stem
            for line in label_fp.read_text().splitlines():
                m = LINE_RE.match(line)
                if not m:
                    continue
                utt_id, code = m.groups()
                if s not in TRAIN_SESSIONS:
                    held_out += 1
                    continue
                label = CODE_MAP.get(code)
                if label is None:
                    skipped_xxx += 1
                    continue
                wav = ROOT / f"Session{s}" / "sentences" / "wav" / dialog_id / f"{utt_id}.wav"
                if not wav.exists():
                    skipped_missing += 1
                    continue
                items.append((str(wav), label))

    classes = sorted(set(CODE_MAP.values()))
    print(f"[iemocap] sessions={TRAIN_SESSIONS} kept={len(items)} "
          f"xxx_skipped={skipped_xxx} missing={skipped_missing} session5_held_out={held_out} "
          f"classes={len(classes)}: {classes}")

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
