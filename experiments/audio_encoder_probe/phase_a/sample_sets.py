"""Partition combined predictions CSV into the 4 W x A correctness sets."""

import argparse
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


SET_LABELS = ("W+A+", "W+A-", "W-A+", "W-A-")
REQUIRED_COLS = (
    "utt_id", "audio_path", "true_label", "fold",
    "whisper_small_pred", "whisper_small_proba", "whisper_small_correct",
    "dacvae_pred", "dacvae_proba", "dacvae_correct",
)


def build_sample_sets(combined_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Partition rows by (whisper_small_correct, dacvae_correct).

    Args:
        combined_df: Combined-predictions DataFrame (see REQUIRED_COLS).

    Returns:
        Dict {"W+A+", "W+A-", "W-A+", "W-A-"} -> DataFrame slice (preserves all columns).
        The 4 slices form a disjoint partition of input rows.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = [c for c in REQUIRED_COLS if c not in combined_df.columns]
    if missing:
        raise ValueError(f"combined_df missing required columns: {missing}")

    w = combined_df["whisper_small_correct"].astype(bool)
    a = combined_df["dacvae_correct"].astype(bool)

    sets = {
        "W+A+": combined_df[w & a].copy(),
        "W+A-": combined_df[w & ~a].copy(),
        "W-A+": combined_df[~w & a].copy(),
        "W-A-": combined_df[~w & ~a].copy(),
    }
    n_total = sum(len(v) for v in sets.values())
    assert n_total == len(combined_df), "partition size mismatch (bug)"
    logger.info(
        "Partition: total=%d  W+A+=%d  W+A-=%d  W-A+=%d  W-A-=%d",
        n_total, *(len(sets[k]) for k in SET_LABELS),
    )
    return sets


def to_long_csv(sets: dict[str, pd.DataFrame], out_csv: Path) -> None:
    """Write a long-format CSV with columns (utt_id, set_label, true_label)."""
    parts = []
    for label, df in sets.items():
        parts.append(
            pd.DataFrame({
                "utt_id": df["utt_id"].values,
                "set_label": label,
                "true_label": df["true_label"].values,
            })
        )
    long_df = pd.concat(parts, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows -> %s", len(long_df), out_csv)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    args = p.parse_args()
    sets = build_sample_sets(pd.read_csv(args.combined_csv))
    to_long_csv(sets, args.out_csv)


if __name__ == "__main__":
    main()
