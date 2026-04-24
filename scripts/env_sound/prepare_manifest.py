"""Build env-sound training manifest (FSD50K dev + Clotho dev + ESC-50).

Budget per user-set 2026-04-24 mix: env_sound target = emotion_pool (~41 k).

Output row format:
    {"path": "...", "source": "fsd50k|clotho|esc50",
     "modality": "audio_env_sound",
     "labels": [...],            # multi-label for fsd50k, single for esc50
     "captions": [...],          # 5 captions for clotho
     }

Eval-side held out:
  Clotho evaluation + validation → Tier-3 captioning eval
  FSD50K eval                    → Tier-3 classification eval
  ESC-50                         → 5-fold CV (all folds contribute; just omit one fold
                                   per eval run). For the training manifest, all 5
                                   folds are included — fold selection happens at eval.
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

TARGET = 41_000
random.seed(20260424)


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


def main() -> None:
    rows_all: list[dict] = []
    src_counts = {}
    for gen in (iter_fsd50k_dev, iter_clotho_dev, iter_esc50):
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
