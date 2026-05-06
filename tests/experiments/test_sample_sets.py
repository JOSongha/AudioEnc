"""Tests for the 4-set partition of combined predictions."""

from pathlib import Path

import pandas as pd
import pytest

from experiments.audio_encoder_probe.phase_a.sample_sets import (
    REQUIRED_COLS,
    build_sample_sets,
    to_long_csv,
)


def _make_synthetic(n_each: int = 2) -> pd.DataFrame:
    """4*n_each rows covering all (W,A) combinations."""
    combos = [(True, True), (True, False), (False, True), (False, False)]
    rows = []
    for i, (w, a) in enumerate(combos):
        for j in range(n_each):
            rows.append(
                dict(
                    utt_id=f"u_{i}_{j}",
                    audio_path=f"/x/u_{i}_{j}.wav",
                    true_label="bass",
                    fold=1,
                    whisper_small_pred="bass" if w else "guitar",
                    whisper_small_proba=0.9,
                    whisper_small_correct=w,
                    dacvae_pred="bass" if a else "guitar",
                    dacvae_proba=0.5,
                    dacvae_correct=a,
                )
            )
    return pd.DataFrame(rows)


def test_partition_sizes() -> None:
    df = _make_synthetic(n_each=3)
    sets = build_sample_sets(df)
    for label in ("W+A+", "W+A-", "W-A+", "W-A-"):
        assert len(sets[label]) == 3


def test_partition_disjoint_and_complete() -> None:
    df = _make_synthetic(n_each=2)
    sets = build_sample_sets(df)
    all_ids: set[str] = set()
    for sub in sets.values():
        ids = set(sub["utt_id"])
        assert ids.isdisjoint(all_ids), "partitions must be disjoint"
        all_ids |= ids
    assert all_ids == set(df["utt_id"]), "partition must cover all rows"


def test_missing_column_raises() -> None:
    df = _make_synthetic().drop(columns=["whisper_small_correct"])
    with pytest.raises(ValueError, match="missing required columns"):
        build_sample_sets(df)


def test_to_long_csv_round_trip(tmp_path: Path) -> None:
    df = _make_synthetic(n_each=1)
    sets = build_sample_sets(df)
    out = tmp_path / "sample_sets.csv"
    to_long_csv(sets, out)
    long = pd.read_csv(out)
    assert set(long.columns) == {"utt_id", "set_label", "true_label"}
    assert len(long) == 4
    assert set(long["set_label"]) == {"W+A+", "W+A-", "W-A+", "W-A-"}


def test_required_cols_constant_matches_schema() -> None:
    df = _make_synthetic()
    for col in REQUIRED_COLS:
        assert col in df.columns
