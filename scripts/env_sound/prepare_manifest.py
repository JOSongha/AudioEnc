"""Build env-sound training manifest (FSD50K dev + Clotho dev + ESC-50 + AudioSet bal_train).

2026-04-25: pool expanded with AudioSet bal_train (~22 k clips, multi-label
AudioSet ontology). Per user direction the env-sound pool is now used "full"
(no TARGET subsample) — per-epoch sub-sampling is handled downstream by
build_epoch_random_manifest.py.

Output row format:
    {"path": "...", "source": "fsd50k|clotho|esc50|audioset",
     "modality": "audio_env_sound",
     "labels": [...],            # multi-label for fsd50k/audioset, single for esc50
     "captions": [...],          # 5 captions for clotho
     }

AudioSet audio is shipped inside the parquet as embedded FLAC bytes; this
builder extracts them once into <RAW>/AudioSet/audio/{video_id}.flac and
yields path-only rows after extraction (omni_dataset.py's torchaudio.load
needs files on disk).

Eval-side held out:
  Clotho evaluation + validation → Tier-3 captioning eval
  FSD50K eval                    → Tier-3 classification eval
  ESC-50                         → 5-fold CV (all folds contribute; just omit one fold
                                   per eval run). For the training manifest, all 5
                                   folds are included — fold selection happens at eval.
  AudioSet eval                  → reserved for separate Tier-3 zero-shot AudioSet eval
                                   if needed; not used in training.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pandas as pd

RAW = Path("/mnt/tmp/datasets/env_sound")
OUT = Path("/mnt/tmp/listen_analysis/train_manifest/env_sound_manifest.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 2026-04-25: keep all rows (no subsample). v2 manifest builder applies its
# own per-epoch fraction (env-frac=0.5 by user request).
TARGET = 10**9
random.seed(20260425)


def iter_fsd50k_dev():
    dev_dir = RAW / "FSD50K" / "FSD50K.dev_audio"
    gt_csv = RAW / "FSD50K" / "FSD50K.ground_truth" / "dev.csv"
    if not dev_dir.exists() or not gt_csv.exists():
        return
    df = pd.read_csv(gt_csv)
    # df columns: fname, labels, mids, split
    for _, r in df.iterrows():
        fname = str(r["fname"]) + ".wav"
        p = dev_dir / fname
        if not p.exists():
            continue
        yield {
            "path": str(p),
            "source": "fsd50k",
            "modality": "audio_env_sound",
            "labels": str(r["labels"]).split(","),
        }


def iter_clotho_dev():
    dev_dir = RAW / "Clotho" / "development"
    captions_csv = RAW / "Clotho" / "captions_development.csv"
    if not dev_dir.exists() or not captions_csv.exists():
        return
    df = pd.read_csv(captions_csv)
    for _, r in df.iterrows():
        fname = r["file_name"]
        p = dev_dir / fname
        if not p.exists():
            continue
        caps = [str(r[c]) for c in ("caption_1", "caption_2", "caption_3", "caption_4", "caption_5") if c in r]
        yield {
            "path": str(p),
            "source": "clotho",
            "modality": "audio_env_sound",
            "captions": caps,
        }


def iter_esc50():
    audio_dir = RAW / "ESC-50" / "audio"
    meta = RAW / "ESC-50" / "meta" / "esc50.csv"
    if not audio_dir.exists() or not meta.exists():
        return
    df = pd.read_csv(meta)
    # cols: filename, fold, target, category, esc10, src_file, take
    for _, r in df.iterrows():
        p = audio_dir / r["filename"]
        if not p.exists():
            continue
        yield {
            "path": str(p),
            "source": "esc50",
            "modality": "audio_env_sound",
            "labels": [r["category"]],
            "fold": int(r["fold"]),
        }


def iter_audioset_bal_train():
    """Yield AudioSet bal_train rows after extracting embedded FLAC bytes to disk.

    HF dataset `agkphysics/AudioSet` ships audio as FLAC bytes inside parquet.
    Our omni_dataset.py loads via torchaudio.load(path), so we materialize
    each FLAC as <RAW>/AudioSet/audio/{video_id}.flac on first run; subsequent
    runs skip extraction if the file already exists.

    `human_labels` is the human-readable label list (e.g. ["Speech", "Music"]).
    Use those as the `labels` field — matches FSD50K's plain-string label
    convention. Some videos have empty human_labels; skip them.
    """
    import pyarrow.parquet as pq
    parquet_dir = RAW / "AudioSet" / "data" / "bal_train"
    audio_dir = RAW / "AudioSet" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    if not parquet_dir.exists():
        return
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        return
    for pf in files:
        try:
            table = pq.read_table(pf)
        except Exception as e:
            print(f"  AudioSet: failed to read {pf.name}: {e}")
            continue
        for i in range(table.num_rows):
            vid = table.column("video_id")[i].as_py()
            audio_struct = table.column("audio")[i].as_py()
            human_labels = table.column("human_labels")[i].as_py()
            if not human_labels:
                continue
            out_path = audio_dir / f"{vid}.flac"
            if not out_path.exists():
                try:
                    out_path.write_bytes(audio_struct["bytes"])
                except Exception as e:
                    print(f"  AudioSet: write fail {vid}: {e}")
                    continue
            yield {
                "path": str(out_path),
                "source": "audioset",
                "modality": "audio_env_sound",
                "labels": [str(l) for l in human_labels],
            }


def main() -> None:
    rows_all: list[dict] = []
    src_counts = {}
    for gen in (iter_fsd50k_dev, iter_clotho_dev, iter_esc50, iter_audioset_bal_train):
        subset = list(gen())
        if subset:
            name = subset[0]["source"]
            src_counts[name] = len(subset)
            rows_all.extend(subset)
    print("available per source:")
    for k, v in src_counts.items():
        print(f"  {k:<8} {v:,}")
    print(f"total available: {len(rows_all):,}")

    random.shuffle(rows_all)
    kept = rows_all[:TARGET]

    with OUT.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_counts: dict[str, int] = {}
    for r in kept:
        final_counts[r["source"]] = final_counts.get(r["source"], 0) + 1
    print("\nafter subsample to target:")
    for k, v in final_counts.items():
        print(f"  {k:<8} {v:,}")
    print(f"total kept: {len(kept):,}")
    print(f"manifest:   {OUT}")


if __name__ == "__main__":
    main()
