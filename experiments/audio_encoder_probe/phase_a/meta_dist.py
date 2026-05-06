"""Meta-attribute distribution comparison + chi-square overrepresentation test."""

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Default pitch bins: 8 bins covering MIDI 0-127 (right edge 128 is inclusive).
DEFAULT_PITCH_BINS = (0, 16, 32, 48, 64, 80, 96, 112, 128)


def _bin_pitch(pitch: pd.Series, bins: Iterable[int] = DEFAULT_PITCH_BINS) -> pd.Series:
    """Categorical pitch bin label like '[0,16)'."""
    edges = list(bins)
    labels = [f"[{edges[i]},{edges[i + 1]})" for i in range(len(edges) - 1)]
    return pd.cut(pitch, bins=edges, labels=labels, right=False, include_lowest=True)


def join_sets_with_meta(
    sample_sets_csv: Path,
    meta_csv: Path,
    pitch_bins: Iterable[int] = DEFAULT_PITCH_BINS,
) -> pd.DataFrame:
    """Join Stage-2 sample_sets.csv with Stage-1 nsynth_train_30k_meta.csv on utt_id.

    Returns DataFrame with columns: utt_id, set_label, true_label,
    family, source, pitch, velocity, pitch_bin.

    Raises:
        ValueError: If any utt_id in sample_sets_csv has no matching row in meta_csv.
    """
    sets = pd.read_csv(sample_sets_csv)
    meta = pd.read_csv(meta_csv)
    out = sets.merge(
        meta[["utt_id", "family", "source", "pitch", "velocity"]],
        on="utt_id",
        how="left",
        validate="one_to_one",
    )
    n_missing = int(out["family"].isna().sum())
    if n_missing > 0:
        raise ValueError(f"{n_missing} rows in sample_sets had no meta join match")
    out["pitch_bin"] = _bin_pitch(out["pitch"], bins=pitch_bins).astype(str)
    return out


def meta_distribution(joined_df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Long-format cross-tab: (set_label, attribute_value, count, fraction_within_set).

    Args:
        joined_df: Output of join_sets_with_meta().
        attribute: Column name, e.g. "source", "velocity", "pitch_bin".

    Returns:
        DataFrame with columns: set_label, <attribute>, count, fraction.
    """
    if attribute not in joined_df.columns:
        raise ValueError(f"attribute {attribute!r} not in joined_df columns")
    grp = (
        joined_df.groupby(["set_label", attribute], observed=False)
        .size()
        .reset_index(name="count")
    )
    set_totals = grp.groupby("set_label")["count"].transform("sum")
    grp["fraction"] = grp["count"] / set_totals
    return grp


def chi_square_test(
    joined_df: pd.DataFrame,
    attribute: str,
    target_set: str = "W+A-",
) -> dict[str, float]:
    """Chi-square test for ``target_set vs rest`` independence with ``attribute``.

    Builds a 2 x |attribute| contingency table:
        row 0 = counts in target_set
        row 1 = counts in all other sets combined
    Drops attribute values with zero in BOTH rows (undefined dof otherwise).

    Returns:
        Dict with keys chi2, p, dof, n, n_categories_used.

    Raises:
        ValueError: If target_set has no rows or fewer than 2 non-empty categories remain.
    """
    if attribute not in joined_df.columns:
        raise ValueError(f"attribute {attribute!r} not in joined_df columns")
    is_target = joined_df["set_label"] == target_set
    if not is_target.any():
        raise ValueError(f"target_set {target_set!r} has no rows in joined_df")
    table = pd.crosstab(is_target, joined_df[attribute], dropna=False)
    nonzero = table.sum(axis=0) > 0
    table = table.loc[:, nonzero]
    if table.shape[1] < 2:
        raise ValueError(
            f"attribute {attribute!r} has <2 non-empty categories — chi2 undefined"
        )
    chi2, p, dof, _ = chi2_contingency(table.values)
    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "n": int(table.values.sum()),
        "n_categories_used": int(table.shape[1]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample_sets_csv", type=Path, required=True)
    p.add_argument("--meta_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    joined = join_sets_with_meta(args.sample_sets_csv, args.meta_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        meta_distribution(joined, attr).assign(attribute=attr)
        for attr in ("source", "velocity", "pitch_bin")
    ]
    pd.concat(parts, ignore_index=True).to_csv(
        args.out_dir / "meta_distribution.csv", index=False
    )

    chi2_rows = [
        {"attribute": attr, "target_set": "W+A-", **chi_square_test(joined, attr, "W+A-")}
        for attr in ("source", "velocity", "pitch_bin")
    ]
    pd.DataFrame(chi2_rows).to_csv(args.out_dir / "chi_square_results.csv", index=False)
    logger.info("Meta-dist outputs -> %s", args.out_dir)


if __name__ == "__main__":
    main()
