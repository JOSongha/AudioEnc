"""Build ESC-50 manifest from official meta/esc50.csv."""
from pathlib import Path
import pandas as pd

ESC_ROOT = Path("/mnt/tmp/datasets/env_sound/ESC-50")
OUT_CSV = Path(__file__).parent / "manifests/esc50.csv"

meta = pd.read_csv(ESC_ROOT / "meta/esc50.csv")
meta["utt_id"] = meta["filename"].str.replace(".wav", "", regex=False)
meta["audio_path"] = (ESC_ROOT / "audio" / meta["filename"]).astype(str)
meta["label"] = meta["target"]   # 0..49 integer
meta["category"] = meta["category"]

df = meta[["utt_id", "audio_path", "label", "category", "fold"]].copy()
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(f"Wrote {len(df)} rows to {OUT_CSV}")
print(f"Folds: {df.fold.value_counts().sort_index().to_dict()}")
print(f"Classes: {df.label.nunique()} (categories: {df.category.nunique()})")
