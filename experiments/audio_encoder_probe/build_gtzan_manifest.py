"""Build GTZAN manifest: 1000 clips, 10 genres, 5-fold stratified."""
import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/mnt/tmp/datasets/music/gtzan/genres")
OUT = Path(__file__).parent / "manifests/gtzan.csv"

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

rows = []
for g in GENRES:
    for wav in sorted((ROOT / g).glob("*.wav")):
        if wav.name.startswith("."):  # macOS dot-files
            continue
        m = re.match(r"^([a-z]+)\.(\d+)$", wav.stem)
        if not m:
            continue
        rows.append({
            "utt_id": wav.stem,
            "audio_path": str(wav),
            "label": g,
            "_idx": int(m.group(2)),
        })

df = pd.DataFrame(rows)
# 5-fold stratified by genre: each genre's 100 clips → 5 folds of 20 each
# deterministic by _idx (sorted)
rng = np.random.default_rng(42)
df["fold"] = 0
for g in GENRES:
    sub = df[df.label == g].sort_values("_idx").index.tolist()
    sub_arr = np.array(sub)
    rng.shuffle(sub_arr)
    for i, idx in enumerate(sub_arr):
        df.at[idx, "fold"] = (i // 20) + 1  # folds 1..5

df = df.drop(columns=["_idx"])
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"Wrote {len(df)} rows to {OUT}")
print(f"Genres: {df.label.value_counts().to_dict()}")
print(f"Folds:  {df.fold.value_counts().sort_index().to_dict()}")
print(f"Per-fold genre balance:")
print(df.groupby(["fold", "label"]).size().unstack(fill_value=0))
