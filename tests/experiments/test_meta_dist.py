"""Tests for meta distribution + chi-square overrepresentation test."""

from pathlib import Path

import pandas as pd
import pytest

from experiments.audio_encoder_probe.phase_a.meta_dist import (
    chi_square_test,
    join_sets_with_meta,
    meta_distribution,
)


def _make_joined(imbalanced: bool) -> pd.DataFrame:
    """Build a joined DataFrame for chi-square tests.

    If imbalanced: W+A- is all electronic (30 rows), rest uniform across 3 sources.
    If balanced: all 4 sets have identical source distribution.
    N=120 total, large enough for reliable chi-square.
    """
    rows = []
    if imbalanced:
        # W+A-: 30 rows, all electronic
        for i in range(30):
            rows.append({"utt_id": f"t{i}", "set_label": "W+A-", "source": "electronic"})
        # rest: 90 rows (30 per set), 10 per source per set
        for j, label in enumerate(["W+A+", "W-A+", "W-A-"]):
            for k in range(30):
                src = ["acoustic", "electronic", "synthetic"][k % 3]
                rows.append({"utt_id": f"u{j}_{k}", "set_label": label, "source": src})
    else:
        # All 4 sets: 30 rows each, uniform across 3 sources
        for j, label in enumerate(["W+A+", "W+A-", "W-A+", "W-A-"]):
            for k in range(30):
                src = ["acoustic", "electronic", "synthetic"][k % 3]
                rows.append({"utt_id": f"u{j}_{k}", "set_label": label, "source": src})
    return pd.DataFrame(rows)


def test_meta_distribution_fractions_sum_to_one_per_set() -> None:
    joined = _make_joined(imbalanced=False)
    out = meta_distribution(joined, "source")
    for label in out["set_label"].unique():
        s = out[out["set_label"] == label]["fraction"].sum()
        assert abs(s - 1.0) < 1e-9, f"fractions in {label} sum to {s}"


def test_chi_square_imbalanced_significant() -> None:
    joined = _make_joined(imbalanced=True)
    res = chi_square_test(joined, "source", "W+A-")
    assert res["p"] < 0.01, f"expected p<0.01 for clear imbalance, got {res['p']}"
    assert res["dof"] == 2  # 3 sources, target vs rest


def test_chi_square_balanced_not_significant() -> None:
    joined = _make_joined(imbalanced=False)
    res = chi_square_test(joined, "source", "W+A-")
    assert res["p"] > 0.05, f"expected p>0.05 for balanced data, got {res['p']}"


def test_chi_square_missing_target_set_raises() -> None:
    joined = _make_joined(imbalanced=False)
    joined = joined[joined["set_label"] != "W+A-"]
    with pytest.raises(ValueError, match="has no rows"):
        chi_square_test(joined, "source", "W+A-")


def test_join_sets_with_meta(tmp_path: Path) -> None:
    sets = tmp_path / "sets.csv"
    pd.DataFrame({
        "utt_id": ["a", "b"],
        "set_label": ["W+A+", "W+A-"],
        "true_label": ["bass", "guitar"],
    }).to_csv(sets, index=False)
    meta = tmp_path / "meta.csv"
    pd.DataFrame({
        "utt_id": ["a", "b"],
        "family": ["bass", "guitar"],
        "source": ["acoustic", "electronic"],
        "pitch": [60, 70],
        "velocity": [100, 75],
    }).to_csv(meta, index=False)
    joined = join_sets_with_meta(sets, meta)
    assert set(joined.columns) >= {"set_label", "source", "pitch_bin"}
    assert len(joined) == 2


def test_join_sets_with_missing_meta_raises(tmp_path: Path) -> None:
    sets = tmp_path / "sets.csv"
    pd.DataFrame({
        "utt_id": ["a", "missing"],
        "set_label": ["W+A+", "W+A-"],
        "true_label": ["bass", "guitar"],
    }).to_csv(sets, index=False)
    meta = tmp_path / "meta.csv"
    pd.DataFrame({
        "utt_id": ["a"],
        "family": ["bass"],
        "source": ["acoustic"],
        "pitch": [60],
        "velocity": [100],
    }).to_csv(meta, index=False)
    with pytest.raises(ValueError, match="no meta join match"):
        join_sets_with_meta(sets, meta)
