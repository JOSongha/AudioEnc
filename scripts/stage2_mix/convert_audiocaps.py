"""Convert AudioCaps (OpenSound/AudioCaps) → sound-captioning JSONL manifest.

Extracts embedded WAV bytes from train+validation parquets into individual .wav files
under CACHE/audiocaps/wav/<split>/<audiocap_id>.wav, then emits one manifest row per
clip: {task: "sound_caption", audio_path, response: caption}.

The test split is held out for evaluation and is NOT converted.
"""

import argparse
import glob
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

SRC_DIR = Path("/mnt/ddn/users/sehyun/CACHE/audiocaps/data")
WAV_ROOT = Path("/mnt/ddn/users/sehyun/CACHE/audiocaps/wav")
OUT_DIR = Path("/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/sound")

# Splits to emit (train + validation only; test held out)
INCLUDE_SPLITS = ("train", "validation")


def split_of(parquet_path: str) -> str:
    """Infer split name from filename e.g. 'train-00000-of-00412.parquet' → 'train'."""
    m = re.match(r"(train|validation|test)-\d+-of-\d+\.parquet", Path(parquet_path).name)
    if not m:
        raise ValueError(f"Cannot parse split from {parquet_path}")
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="Process only the first N shards (for smoke testing).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in INCLUDE_SPLITS:
        (WAV_ROOT / s).mkdir(parents=True, exist_ok=True)

    parquets = sorted(glob.glob(str(SRC_DIR / "*.parquet")))
    parquets = [p for p in parquets if split_of(p) in INCLUDE_SPLITS]
    if args.limit_shards:
        parquets = parquets[: args.limit_shards]

    total_rows = 0
    for i, pq_path in enumerate(parquets):
        split = split_of(pq_path)
        table = pq.read_table(pq_path).to_pandas()
        out_path = OUT_DIR / f"audiocaps_{split}_shard_{i:05d}.jsonl"
        n_written = 0
        with out_path.open("w") as f:
            for row in table.itertuples(index=False):
                audio = row.audio
                if not isinstance(audio, dict) or "bytes" not in audio:
                    continue
                audio_bytes = audio["bytes"]
                if not audio_bytes:
                    continue
                wav_path = WAV_ROOT / split / f"{row.audiocap_id}.wav"
                if not wav_path.exists():
                    wav_path.write_bytes(audio_bytes)
                rec = {
                    "task": "sound_caption",
                    "audio_path": str(wav_path),
                    "response": row.caption,
                    "source": f"audiocaps/{split}",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
        total_rows += n_written
        print(f"  [{i+1}/{len(parquets)}] {Path(pq_path).name} ({split}) → {out_path.name}: "
              f"{n_written} rows")

    print(f"\nTotal: {total_rows} rows written, wav under {WAV_ROOT}")


if __name__ == "__main__":
    main()
