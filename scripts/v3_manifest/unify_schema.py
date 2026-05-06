"""Unify v3 manifest jsonl schemas — every row gets all 9 possible keys (None for missing).

Fix for HF datasets streaming IndexError: when shards have different schemas
(ASR uses nubes_path/text, sound uses audio_path/captions, emotion uses
audio_path/question/choices/answer), the batch builder produces uneven column
arrays → `array[i]` IndexError. Unifying makes every row have the same keys.
"""
import json, glob, os, sys
from pathlib import Path

UNIFIED_KEYS = [
    "modality",      # str: audio_asr | audio_env_sound | audio_emotion
    "source",        # str: dataset name
    "audio_path",    # str | null
    "nubes_path",    # str | null
    "text",          # str | null   (asr transcript)
    "captions",      # list[str] | null  (sound caption)
    "question",      # str | null  (emotion MCQA)
    "choices",       # list[str] | null
    "answer",        # str | null
]

ROOT = Path("/mnt/tmp/datasets/manifests/v3")


def unify_file(path: Path) -> tuple[int, int]:
    """Return (rows, bytes_written)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            unified = {k: r.get(k) for k in UNIFIED_KEYS}
            rows.append(unified)
    tmp = path.with_suffix(".jsonl.tmp")
    nb = 0
    with open(tmp, "w") as fh:
        for r in rows:
            s = json.dumps(r, ensure_ascii=False) + "\n"
            fh.write(s)
            nb += len(s.encode())
    os.replace(tmp, path)
    return len(rows), nb


def main():
    files = sorted(ROOT.glob("*.jsonl"))
    total_rows = 0
    total_bytes = 0
    for i, f in enumerate(files):
        n, b = unify_file(f)
        total_rows += n
        total_bytes += b
        if i % 20 == 0:
            print(f"[unify] {i+1}/{len(files)}  {f.name}  rows={n}", flush=True)
    print(f"[unify] DONE: {len(files)} files, {total_rows:,} rows, {total_bytes/1024/1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
