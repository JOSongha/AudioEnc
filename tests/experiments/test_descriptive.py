"""Tests for set-level descriptive statistics."""

import pandas as pd

from experiments.audio_encoder_probe.phase_a.descriptive import (
    SET_LABELS,
    dac_confusion_on_set,
    proba_distribution_summary,
    set_sizes,
    top_confused_pairs,
)


def _toy_sets() -> dict[str, pd.DataFrame]:
    def df(true: list[str], pred: list[str], proba: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {"true_label": true, "dacvae_pred": pred, "whisper_small_proba": proba}
        )

    return {
        "W+A+": df(["bass", "guitar"], ["bass", "guitar"], [0.9, 0.85]),
        "W+A-": df(
            ["bass", "bass", "organ"],
            ["guitar", "guitar", "keyboard"],
            [0.7, 0.6, 0.55],
        ),
        "W-A+": df(["bass"], ["bass"], [0.3]),
        "W-A-": df(["bass"], ["guitar"], [0.4]),
    }


def test_set_sizes_sums_and_fractions() -> None:
    out = set_sizes(_toy_sets())
    assert list(out["set"]) == list(SET_LABELS)
    assert out["count"].sum() == 7
    assert abs(out["fraction"].sum() - 1.0) < 1e-9


def test_dac_confusion_on_W_plus_A_minus_diagonal_zero() -> None:
    sets = _toy_sets()
    classes = ["bass", "guitar", "keyboard", "organ"]
    cm = dac_confusion_on_set(sets["W+A-"], classes)
    for c in classes:
        assert cm.loc[c, c] == 0, f"diagonal must be 0 in W+A-, but {c} has {cm.loc[c, c]}"
    assert cm.loc["bass", "guitar"] == 2
    assert cm.loc["organ", "keyboard"] == 1


def test_dac_confusion_unknown_classes_zero() -> None:
    """Classes in all_classes but absent from set_df should have zero row/col sum."""
    sets = _toy_sets()
    classes = ["bass", "guitar", "keyboard", "organ", "flute"]
    cm = dac_confusion_on_set(sets["W+A-"], classes)
    assert cm.loc["flute"].sum() == 0
    assert cm["flute"].sum() == 0


def test_proba_summary_quartiles() -> None:
    sets = _toy_sets()
    out = proba_distribution_summary(sets, "whisper_small_proba")
    row = out[out["set"] == "W+A-"].iloc[0]
    # values: 0.7, 0.6, 0.55 -> median 0.6
    assert abs(row["median"] - 0.6) < 1e-9
    assert row["n"] == 3


def test_top_confused_pairs_excludes_diagonal_and_sorts() -> None:
    sets = _toy_sets()
    classes = ["bass", "guitar", "keyboard", "organ"]
    cm = dac_confusion_on_set(sets["W+A-"], classes)
    top = top_confused_pairs(cm, k=2)
    assert top[0] == ("bass", "guitar", 2)
    assert top[1] == ("organ", "keyboard", 1)
