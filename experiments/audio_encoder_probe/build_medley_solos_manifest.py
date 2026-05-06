"""Build Medley-solos-DB manifest (utt_id, audio_path, label, fold).

Dataset: https://zenodo.org/records/3464194
8 instrument classes, real recordings (non-synthetic).
Official splits: training / validation / test.

Fold assignment (for leave-one-fold-out probe):
  training   → fold 1 (train pool)
  validation → fold 2 (train pool)
  test       → fold 3 (held-out eval)

This gives two train-pool folds so probe.py trains on folds 1+2 and evaluates on fold 3.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

AUDIO_DIR = Path("/mnt/tmp/datasets/music/medley_solos")
META_CSV = AUDIO_DIR / "metadata.csv"
OUT_DIR = Path(__file__).parent / "manifests"

# Map official subset names → fold numbers
SUBSET_FOLD = {"training": 1, "validation": 2, "test": 3}

# Filename pattern: Medley-solos-DB_{subset}-{instrument_id}_{uuid4}.wav
_FNAME_RE = re.compile(
    r"Medley-solos-DB_(?P<subset>[a-z]+)-(?P<inst_id>\d+)_(?P<uuid4>[0-9a-f-]+)\.wav"
)


def build_manifest(audio_dir: Path, meta_csv: Path, out_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)

    rows = []
    for _, r in meta.iterrows():
        fname = f"Medley-solos-DB_{r['subset']}-{r['instrument_id']}_{r['uuid4']}.wav"
        audio_path = audio_dir / fname
        if not audio_path.exists():
            continue
        rows.append(
            {
                "utt_id": f"medley_{r['subset']}_{r['instrument_id']}_{r['uuid4']}",
                "audio_path": str(audio_path),
                "label": r["instrument"],
                "fold": SUBSET_FOLD[r["subset"]],
            }
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows → {out_csv}")
    print("Label dist:", df["label"].value_counts().to_dict())
    print("Fold dist:", df["fold"].value_counts().sort_index().to_dict())
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio_dir", type=Path, default=AUDIO_DIR)
    p.add_argument("--meta_csv", type=Path, default=META_CSV)
    p.add_argument("--out_csv", type=Path, default=OUT_DIR / "medley_solos.csv")
    args = p.parse_args()
    build_manifest(args.audio_dir, args.meta_csv, args.out_csv)


if __name__ == "__main__":
    main()
