"""Convert MELD (declare-lab/MELD.Raw) → emotion-classification JSONL manifest.

Converts .mp4 utterance clips to 16 kHz mono .wav via ffmpeg, then emits manifest
rows using the 7-way `Emotion` label from the split CSV. Test split is held out
for evaluation.
"""

import argparse
import csv
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SRC_ROOT = Path("/mnt/ddn/users/sehyun/CACHE/meld/MELD.Raw")
WAV_ROOT = Path("/mnt/ddn/users/sehyun/CACHE/meld/wav")
OUT_DIR = Path("/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/emotion")

FFMPEG = shutil.which("ffmpeg") or "/mnt/tmp/miniconda3/envs/venv_torch210/bin/ffmpeg"

# (split_name, csv_file, mp4_dir_under_SRC_ROOT)
SPLITS = [
    ("train", "train_sent_emo.csv", "train_splits"),
    ("dev",   "dev_sent_emo.csv",   "dev_splits_complete"),
]


def mp4_to_wav(args: tuple[Path, Path]) -> tuple[Path, bool, str]:
    mp4_path, wav_path = args
    if wav_path.exists():
        return wav_path, True, ""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-nostdin", "-y", "-loglevel", "error",
        "-i", str(mp4_path),
        "-ac", "1", "-ar", "16000",
        "-vn", "-f", "wav",
        str(wav_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            return wav_path, False, r.stderr.strip()[:200]
        return wav_path, True, ""
    except Exception as e:
        return wav_path, False, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WAV_ROOT.mkdir(parents=True, exist_ok=True)

    # Pass 1: plan all mp4→wav jobs and manifest rows per split
    total_rows = 0
    for split, csv_name, mp4_dir in SPLITS:
        csv_path = SRC_ROOT / csv_name
        mp4_root = SRC_ROOT / mp4_dir
        out_path = OUT_DIR / f"meld_{split}_shard_00000.jsonl"

        if not csv_path.exists() or not mp4_root.exists():
            print(f"  [skip {split}] missing {csv_path} or {mp4_root}")
            continue

        # Gather records: (mp4_path, wav_path, emotion, utterance)
        planned: list[tuple[Path, Path, str, str]] = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                did = row["Dialogue_ID"]
                uid = row["Utterance_ID"]
                mp4 = mp4_root / f"dia{did}_utt{uid}.mp4"
                if not mp4.exists():
                    continue
                wav = WAV_ROOT / split / f"dia{did}_utt{uid}.wav"
                planned.append((mp4, wav, row["Emotion"], row["Utterance"]))
                if args.limit and len(planned) >= args.limit:
                    break

        # Convert mp4→wav in parallel
        jobs = [(mp4, wav) for mp4, wav, _, _ in planned if not wav.exists()]
        print(f"  [{split}] {len(planned)} utterances, {len(jobs)} to convert")
        failed: dict[str, str] = {}
        if jobs:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(mp4_to_wav, j): j for j in jobs}
                done = 0
                for fut in as_completed(futures):
                    wav_path, ok, err = fut.result()
                    done += 1
                    if not ok:
                        failed[wav_path.name] = err
                    if done % 1000 == 0:
                        print(f"    [{split}] converted {done}/{len(jobs)}")
            print(f"  [{split}] conversion done; failures: {len(failed)}")

        # Emit manifest rows
        n = 0
        label_counts: dict[str, int] = {}
        with out_path.open("w") as fout:
            for mp4, wav, emo, utt in planned:
                if not wav.exists():
                    continue
                rec = {
                    "task": "emotion_classify",
                    "audio_path": str(wav),
                    "response": emo,
                    "source": f"meld/{split}",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                label_counts[emo] = label_counts.get(emo, 0) + 1
        total_rows += n
        print(f"  [{split}] wrote {n} rows → {out_path.name}")
        for lab, c in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"    {lab}: {c}")

    print(f"\nTotal: {total_rows} rows, wav under {WAV_ROOT}")


if __name__ == "__main__":
    main()
