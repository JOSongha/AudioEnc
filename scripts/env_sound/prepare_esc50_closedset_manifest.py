"""ESC-50 closed-set classification manifest for overfit-capacity experiments.

Each row carries a fully-rendered prompt that lists ALL 50 ESC-50 classes
(shuffled per-sample with a deterministic seed) and asks the model to answer
with the class name verbatim. Target = raw category string.

The shuffle prevents positional shortcut: with the same prompt-order every row,
a model could learn "this audio is the 7th item in the list" instead of
audio→label correspondence. Per-sample shuffling forces the latter.

Manifest schema
---------------
    {
      "path":     str,     # absolute wav path
      "source":   "esc50_closedset",
      "modality": "audio_env_sound",
      "prompt":   str,     # full instruction with 50 shuffled labels
      "target":   str,     # canonical class name, e.g. "chainsaw"
      "label":    str,     # same as target (kept for parsing compat)
      "fold":     int,     # 1..5, ESC-50 standard CV fold
      "filename": str,
    }

Output split
------------
We materialize 5 split files (one per held-out fold) plus an "all" file:
    esc50_closedset_train_fold{N}.jsonl   # rows where fold != N
    esc50_closedset_eval_fold{N}.jsonl    # rows where fold == N
    esc50_closedset_all.jsonl             # all 2000 rows (debug / overfit-all)

Usage
-----
    python scripts/env_sound/prepare_esc50_closedset_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ESC_ROOT = Path("/mnt/tmp/datasets/env_sound/ESC-50")
META_CSV = ESC_ROOT / "meta/esc50.csv"
AUDIO_DIR = ESC_ROOT / "audio"
OUT_DIR = Path("/mnt/tmp/listen_analysis/train_manifest/esc50_closedset")
DEFAULT_SEED = 20260430


def load_meta() -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    classes: set[str] = set()
    with META_CSV.open() as f:
        for r in csv.DictReader(f):
            ap = AUDIO_DIR / r["filename"]
            if not ap.exists():
                continue
            rows.append({
                "filename": r["filename"],
                "path": str(ap),
                "fold": int(r["fold"]),
                "label": r["category"],
            })
            classes.add(r["category"])
    return rows, sorted(classes)


def build_prompt(classes_shuffled: list[str]) -> str:
    """Render the closed-set classification prompt.

    Wraps the class list at ~5 names per line for readability (model still
    sees a flat token sequence; line wrapping is purely cosmetic for human
    inspection of the manifest).
    """
    lines = []
    chunk = 5
    for i in range(0, len(classes_shuffled), chunk):
        lines.append(", ".join(classes_shuffled[i:i + chunk]))
    body = ",\n".join(lines)
    return (
        "Listen to the audio. It belongs to one of these 50 classes:\n"
        f"{body}.\n"
        "Answer with the class name only."
    )


def write_shards(rows: list[dict], out_subdir: Path, n_shards: int) -> None:
    """Round-robin rows into N shard files inside `out_subdir/`."""
    out_subdir.mkdir(parents=True, exist_ok=True)
    for f in out_subdir.glob("shard_*.jsonl"):
        f.unlink()
    fhs = [open(out_subdir / f"shard_{i:05d}.jsonl", "w") for i in range(n_shards)]
    try:
        for i, r in enumerate(rows):
            fhs[i % n_shards].write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        for f in fhs:
            f.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-shards", type=int, default=16,
                   help="Shard each split into N files (must be ≥ world_size for "
                        "multi-GPU streaming).")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, classes = load_meta()
    print(f"loaded {len(rows)} rows, {len(classes)} classes")
    assert len(classes) == 50, f"expected 50 ESC-50 classes, got {len(classes)}"

    # deterministic per-sample shuffle: seed = master ^ row_index
    out_rows: list[dict] = []
    for i, r in enumerate(rows):
        rng = random.Random(args.seed ^ (i * 0x9E3779B1))
        order = list(classes)
        rng.shuffle(order)
        prompt = build_prompt(order)
        out_rows.append({
            "path": r["path"],
            "source": "esc50_closedset",
            "modality": "audio_env_sound",
            "prompt": prompt,
            "target": r["label"],
            "label": r["label"],
            "fold": r["fold"],
            "filename": r["filename"],
        })

    # 1) flat single-jsonl files (for offline parsing / debug)
    all_path = args.out_dir / "esc50_closedset_all.jsonl"
    with all_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} rows -> {all_path}")

    for fold in range(1, 6):
        train = [r for r in out_rows if r["fold"] != fold]
        held = [r for r in out_rows if r["fold"] == fold]
        tp = args.out_dir / f"esc50_closedset_train_fold{fold}.jsonl"
        ep = args.out_dir / f"esc50_closedset_eval_fold{fold}.jsonl"
        with tp.open("w") as f:
            for r in train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with ep.open("w") as f:
            for r in held:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  fold {fold}: train={len(train):4d}  eval={len(held):3d}")

    # 2) sharded directories (for multi-GPU streaming dataloader)
    print(f"\nsharding into {args.n_shards}-shard dirs...")
    write_shards(out_rows, args.out_dir / "shards_all", args.n_shards)
    print(f"  shards_all/  ({args.n_shards} shards, {len(out_rows)} rows)")
    for fold in range(1, 6):
        train = [r for r in out_rows if r["fold"] != fold]
        held = [r for r in out_rows if r["fold"] == fold]
        write_shards(train, args.out_dir / f"shards_train_fold{fold}", args.n_shards)
        write_shards(held,  args.out_dir / f"shards_eval_fold{fold}",  args.n_shards)
        print(f"  shards_train_fold{fold}/  shards_eval_fold{fold}/  ({len(train)}/{len(held)} rows)")

    # sanity dump of one sample prompt
    print("\n=== sample prompt (first row) ===")
    print(out_rows[0]["prompt"])
    print(f"target: {out_rows[0]['target']}")


if __name__ == "__main__":
    main()
