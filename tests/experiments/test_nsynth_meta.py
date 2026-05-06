"""Tests for NSynth utt_id parsing and manifest augmentation."""

from pathlib import Path

import pandas as pd
import pytest

from experiments.audio_encoder_probe.phase_a.nsynth_meta import (
    augment_manifest,
    parse_nsynth_utt_id,
)


def test_parse_simple_family() -> None:
    out = parse_nsynth_utt_id("bass_electronic_034-061-100")
    assert out == {
        "family": "bass",
        "source": "electronic",
        "inst_id": 34,
        "pitch": 61,
        "velocity": 100,
    }


def test_parse_family_with_underscore() -> None:
    """synth_lead is the only NSynth family containing an underscore."""
    out = parse_nsynth_utt_id("synth_lead_synthetic_001-060-127")
    assert out["family"] == "synth_lead"
    assert out["source"] == "synthetic"
    assert out["pitch"] == 60
    assert out["velocity"] == 127


def test_parse_all_three_sources() -> None:
    for source in ("acoustic", "electronic", "synthetic"):
        out = parse_nsynth_utt_id(f"vocal_{source}_007-039-100")
        assert out["source"] == source


def test_parse_invalid_format_raises() -> None:
    with pytest.raises(ValueError, match="does not match NSynth format"):
        parse_nsynth_utt_id("not-a-valid-utt-id")


def test_parse_unknown_source_raises() -> None:
    with pytest.raises(ValueError):
        parse_nsynth_utt_id("bass_unknown_034-061-100")


def test_augment_manifest_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    pd.DataFrame(
        {
            "utt_id": [
                "bass_electronic_034-061-100",
                "synth_lead_synthetic_001-060-127",
                "vocal_acoustic_007-039-050",
            ],
            "audio_path": ["/x/a.wav", "/x/b.wav", "/x/c.wav"],
            "label": ["bass", "synth_lead", "vocal"],
            "fold": [1, 2, 3],
        }
    ).to_csv(src, index=False)

    out_csv = tmp_path / "out.csv"
    df = augment_manifest(src, out_csv)

    assert out_csv.exists()
    assert set(df.columns) == {
        "utt_id", "audio_path", "label", "fold",
        "family", "source", "inst_id", "pitch", "velocity",
    }
    assert len(df) == 3
    assert df.loc[1, "family"] == "synth_lead"
    assert df.loc[1, "velocity"] == 127
