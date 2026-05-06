"""Descriptive statistics on the 4 W x A sample sets (no hypothesis tests)."""

import argparse
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


SET_LABELS = ("W+A+", "W+A-", "W-A+", "W-A-")


def set_sizes(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return DataFrame with columns (set, count, fraction)."""
    total = sum(len(df) for df in sets.values())
    rows = [
        {"set": label, "count": len(sets[label]), "fraction": len(sets[label]) / total}
        for label in SET_LABELS
    ]
    return pd.DataFrame(rows)


def dac_confusion_on_set(set_df: pd.DataFrame, all_classes: list[str]) -> pd.DataFrame:
    """11x11 confusion matrix (rows = true_label, cols = dacvae_pred) on a single set.

    Returns DataFrame indexed by class with class columns; values = counts.
    Missing classes are filled with 0.
    """
    cm = pd.crosstab(
        set_df["true_label"],
        set_df["dacvae_pred"],
        rownames=["true_label"],
        colnames=["dacvae_pred"],
        dropna=False,
    )
    return cm.reindex(index=all_classes, columns=all_classes, fill_value=0)


def proba_distribution_summary(
    sets: dict[str, pd.DataFrame], encoder_col: str
) -> pd.DataFrame:
    """Per-set summary stats (mean, median, q25, q75, std) of an encoder's pred_proba."""
    rows = []
    for label in SET_LABELS:
        s = sets[label][encoder_col] if label in sets else pd.Series([], dtype=float)
        rows.append({
            "set": label,
            "n": len(s),
            "mean": float(s.mean()) if len(s) else float("nan"),
            "median": float(s.median()) if len(s) else float("nan"),
            "q25": float(s.quantile(0.25)) if len(s) else float("nan"),
            "q75": float(s.quantile(0.75)) if len(s) else float("nan"),
            "std": float(s.std()) if len(s) else float("nan"),
        })
    return pd.DataFrame(rows)


def top_confused_pairs(
    confusion: pd.DataFrame, k: int = 3
) -> list[tuple[str, str, int]]:
    """Top-k off-diagonal cells of a confusion matrix as (true, pred, count) sorted desc.

    Diagonal cells (true == pred) are excluded. By construction, on the W+A- set
    the diagonal is already 0 (DAC was wrong by definition on those samples).
    """
    pairs: list[tuple[str, str, int]] = []
    for t in confusion.index:
        for p in confusion.columns:
            if t == p:
                continue
            v = int(confusion.loc[t, p])
            if v > 0:
                pairs.append((t, p, v))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:k]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    from experiments.audio_encoder_probe.phase_a.sample_sets import build_sample_sets

    df = pd.read_csv(args.combined_csv)
    sets = build_sample_sets(df)
    classes = sorted(df["true_label"].unique())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_sizes(sets).to_csv(args.out_dir / "set_sizes.csv", index=False)
    dac_confusion_on_set(sets["W+A-"], classes).to_csv(
        args.out_dir / "dac_confusion_W+A-.csv"
    )
    proba_distribution_summary(sets, "whisper_small_proba").to_csv(
        args.out_dir / "whisper_proba_distribution.csv", index=False
    )
    top = top_confused_pairs(dac_confusion_on_set(sets["W+A-"], classes), k=10)
    pd.DataFrame(top, columns=["true_label", "dacvae_pred", "count"]).to_csv(
        args.out_dir / "top_confused_pairs.csv", index=False
    )
    logger.info("Descriptive outputs -> %s", args.out_dir)


if __name__ == "__main__":
    main()
