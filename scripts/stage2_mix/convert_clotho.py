"""Convert Clotho v2.1 (Zenodo) → sound-captioning JSONL manifest.

Extracts the development + validation .7z audio archives (eval held out).
Emits one manifest row per (clip, caption) pair — Clotho provides 5 captions per clip,
so each clip contributes 5 training rows.
"""

import argparse
import csv
import json
from pathlib import Path

import py7zr

SRC_DIR = Path("/mnt/ddn/users/sehyun/CACHE/clotho")
WAV_ROOT = Path("/mnt/ddn/users/sehyun/CACHE/clotho/wav")
OUT_DIR = Path("/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/sound")

# split_name → (7z archive stem, captions CSV stem, extracted subdir name inside archive)
SPLITS = {
    "development": ("clotho_audio_development", "clotho_captions_development", "development"),
    "validation":  ("clotho_audio_validation",  "clotho_captions_validation",  "validation"),
}


def extract_archive(archive_path: Path, extract_into: Path) -> None:
    """Extract a .7z archive (idempotent — skip if target dir already has contents)."""
    if extract_into.exists() and any(extract_into.iterdir()):
        print(f"  [skip extract] {extract_into} already populated")
        return
    extract_into.mkdir(parents=True, exist_ok=True)
    print(f"  [extract] {archive_path.name} → {extract_into}")
    with py7zr.SevenZipFile(archive_path, "r") as z:
        z.extractall(path=extract_into.parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N clips per split (for smoke testing).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WAV_ROOT.mkdir(parents=True, exist_ok=True)

    total = 0
    for split, (audio_stem, caption_stem, subdir) in SPLITS.items():
        # 1. Extract 7z → WAV_ROOT/<subdir>/<file>.wav
        archive = SRC_DIR / f"{audio_stem}.7z"
        extract_archive(archive, WAV_ROOT / subdir)

        # 2. Read captions CSV (file_name, caption_1..caption_5)
        captions_csv = SRC_DIR / f"{caption_stem}.csv"
        out_path = OUT_DIR / f"clotho_{split}_shard_00000.jsonl"
        n_clips = 0
        n_rows = 0
        with captions_csv.open() as fin, out_path.open("w") as fout:
            reader = csv.DictReader(fin)
            for clip in reader:
                fname = clip["file_name"]
                wav_path = WAV_ROOT / subdir / fname
                if not wav_path.exists():
                    # Should not happen after extract; log and skip
                    print(f"  [missing wav] {wav_path}")
                    continue
                n_clips += 1
                if args.limit and n_clips > args.limit:
                    break
                for k in ("caption_1", "caption_2", "caption_3", "caption_4", "caption_5"):
                    cap = clip.get(k, "").strip()
                    if not cap:
                        continue
                    rec = {
                        "task": "sound_caption",
                        "audio_path": str(wav_path),
                        "response": cap,
                        "source": f"clotho/{split}",
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_rows += 1
        total += n_rows
        print(f"  [{split}] {n_clips} clips → {n_rows} rows in {out_path.name}")

    print(f"\nTotal: {total} rows, wav under {WAV_ROOT}")


if __name__ == "__main__":
    main()
