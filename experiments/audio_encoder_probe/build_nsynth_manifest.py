"""Build NSynth train + test manifests (family-balanced 30k train + full 4096 test)."""
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/mnt/tmp/datasets/music/nsynth")
OUT_DIR = Path(__file__).parent / "manifests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 families present in this HF mirror's test split (synth_lead absent in test)
TARGET_FAMILIES = ["bass", "brass", "flute", "guitar", "keyboard", "mallet",
                   "organ", "reed", "string", "vocal"]

# ── Load metadata jsonl ─────────────────────────────────────────────────────
def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

test_rows = load_jsonl(ROOT / "_test_rows.jsonl")
train_rows = load_jsonl(ROOT / "_train_rows.jsonl")

# ── Test manifest (full) ────────────────────────────────────────────────────
test_df = pd.DataFrame(test_rows)
# Filter to known families
test_df = test_df[test_df["instrument"].isin(TARGET_FAMILIES)].copy()
test_df["label"] = test_df["instrument"]
test_df["fold"] = 1  # single test fold
test_df = test_df[["utt_id", "audio_path", "label", "fold"]]
test_df.to_csv(OUT_DIR / "nsynth_test.csv", index=False)
print(f"Test: {len(test_df)} rows → {OUT_DIR / 'nsynth_test.csv'}")
print(f"  Family dist: {test_df.label.value_counts().to_dict()}")

# ── Train manifest (family-balanced 30k subsample) ──────────────────────────
train_df = pd.DataFrame(train_rows)
train_df = train_df[train_df["instrument"].isin(TARGET_FAMILIES)].copy()

# 30k target with min(target_per_family, available)
target_total = 30000
n_families = len(TARGET_FAMILIES)
target_per_family = target_total // n_families  # 2727

rng = np.random.default_rng(42)
selected = []
for fam in TARGET_FAMILIES:
    sub = train_df[train_df.instrument == fam]
    n_take = min(target_per_family, len(sub))
    idx = rng.choice(len(sub), size=n_take, replace=False)
    selected.append(sub.iloc[idx])

train_30k = pd.concat(selected, ignore_index=True)
train_30k["label"] = train_30k["instrument"]
# Internal "fold" for train probe: 5-fold stratified for CV (alternative: hold-out 90/10)
# Use 5-fold so probe.py works. Each fold ~6k.
train_30k["fold"] = 0
for fam in TARGET_FAMILIES:
    sub_idx = train_30k[train_30k.label == fam].index.to_numpy()
    rng.shuffle(sub_idx)
    fold_size = (len(sub_idx) + 4) // 5
    for i, ix in enumerate(sub_idx):
        train_30k.at[ix, "fold"] = (i // fold_size) + 1

train_30k = train_30k[["utt_id", "audio_path", "label", "fold"]]
train_30k.to_csv(OUT_DIR / "nsynth_train_30k.csv", index=False)
print(f"\nTrain 30k: {len(train_30k)} rows → {OUT_DIR / 'nsynth_train_30k.csv'}")
print(f"  Family dist: {train_30k.label.value_counts().to_dict()}")
print(f"  Fold dist: {train_30k.fold.value_counts().sort_index().to_dict()}")
