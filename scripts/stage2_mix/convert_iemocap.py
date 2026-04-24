"""Convert IEMOCAP (AbstractTTS/IEMOCAP) → emotion-classification JSONL manifest.

Extracts embedded WAV bytes to individual .wav files under CACHE/iemocap/wav/.
Uses the `major_emotion` field as the response label (10-way, as-is).
"""

import argparse
import glob
import json
from pathlib import Path

import pyarrow.parquet as pq

SRC_DIR = Path("/mnt/ddn/users/sehyun/CACHE/iemocap/data")
WAV_DIR = Path("/mnt/ddn/users/sehyun/CACHE/iemocap/wav")
OUT_DIR = Path("/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/emotion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-shards", type=int, default=None)
    args = ap.parse_args()

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parquets = sorted(glob.glob(str(SRC_DIR / "*.parquet")))
    if args.limit_shards:
        parquets = parquets[: args.limit_shards]

    total = 0
    label_counts: dict[str, int] = {}
    for i, pq_path in enumerate(parquets):
        table = pq.read_table(pq_path).to_pandas()
        out_path = OUT_DIR / f"iemocap_shard_{i:05d}.jsonl"
        n = 0
        with out_path.open("w") as f:
            for row in table.itertuples(index=False):
                audio = row.audio
                if not isinstance(audio, dict) or not audio.get("bytes"):
                    continue
                label = row.major_emotion
                if not label:
                    continue
                wav_path = WAV_DIR / row.file
                if not wav_path.exists():
                    wav_path.write_bytes(audio["bytes"])
                rec = {
                    "task": "emotion_classify",
                    "audio_path": str(wav_path),
                    "response": str(label),
                    "source": "iemocap",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                label_counts[label] = label_counts.get(label, 0) + 1
        total += n
        print(f"  [{i+1}/{len(parquets)}] {Path(pq_path).name} → {out_path.name}: {n} rows")

    print(f"\nTotal: {total} rows, wav under {WAV_DIR}")
    print("Label distribution:")
    for lab, c in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {lab}: {c}")


if __name__ == "__main__":
    main()
