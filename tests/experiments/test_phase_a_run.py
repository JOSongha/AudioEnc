"""Smoke test for Phase A orchestrator."""

from pathlib import Path

import pandas as pd

from experiments.audio_encoder_probe.phase_a.run import main


def _make_synthetic_csvs(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal combined + meta CSVs (50 rows) for smoke testing."""
    families = ["bass", "guitar", "organ", "keyboard", "flute"]
    sources = ["acoustic", "electronic", "synthetic"]

    rows_combined = []
    rows_meta = []
    for i in range(50):
        true_idx = i % len(families)
        family = families[true_idx]
        source = sources[i % 3]
        pitch = (i * 5) % 128
        velocity = [25, 50, 75, 100, 127][i % 5]
        utt_id = f"{family}_{source}_{i:03d}-{pitch:03d}-{velocity:03d}"

        # Pattern: 30% W+A+, 40% W+A-, 10% W-A+, 20% W-A-
        if i < 15:
            wc, ac = True, True
            wp, dp = family, family
        elif i < 35:
            wc, ac = True, False
            wp = family
            dp = families[(true_idx + 1) % len(families)]
        elif i < 40:
            wc, ac = False, True
            wp = families[(true_idx + 1) % len(families)]
            dp = family
        else:
            wc, ac = False, False
            wp = families[(true_idx + 1) % len(families)]
            dp = families[(true_idx + 2) % len(families)]

        rows_combined.append(dict(
            utt_id=utt_id,
            audio_path=f"/mnt/tmp/datasets/music/nsynth/train_30k/{utt_id}.wav",
            true_label=family,
            fold=(i % 5) + 1,
            whisper_small_pred=wp,
            whisper_small_proba=0.7 + 0.01 * (i % 30),
            whisper_small_correct=wc,
            dacvae_pred=dp,
            dacvae_proba=0.4 + 0.01 * (i % 30),
            dacvae_correct=ac,
        ))
        rows_meta.append(dict(
            utt_id=utt_id,
            audio_path=rows_combined[-1]["audio_path"],
            label=family,
            fold=(i % 5) + 1,
            family=family,
            source=source,
            inst_id=i,
            pitch=pitch,
            velocity=velocity,
        ))

    combined_csv = tmp_path / "combined.csv"
    meta_csv = tmp_path / "meta.csv"
    pd.DataFrame(rows_combined).to_csv(combined_csv, index=False)
    pd.DataFrame(rows_meta).to_csv(meta_csv, index=False)
    return combined_csv, meta_csv


def test_smoke_end_to_end(tmp_path: Path) -> None:
    """50-row synthetic CSV -> run.main produces all expected outputs."""
    combined_csv, meta_csv = _make_synthetic_csvs(tmp_path)
    out_dir = tmp_path / "out"

    main([
        "--combined_csv", str(combined_csv),
        "--meta_csv", str(meta_csv),
        "--out_dir", str(out_dir),
    ])

    expected_csvs = [
        "sample_sets.csv",
        "set_sizes.csv",
        "dac_confusion_W+A-.csv",
        "whisper_proba_distribution.csv",
        "top_confused_pairs.csv",
        "meta_distribution.csv",
        "chi_square_results.csv",
    ]
    for name in expected_csvs:
        f = out_dir / name
        assert f.exists() and f.stat().st_size > 0, f"{name} missing or empty"

    html = out_dir / "confused_pair_examples.html"
    assert html.exists() and html.stat().st_size > 0, "HTML viewer missing or empty"

    figs_dir = out_dir / "figs"
    expected_figs = [
        "F-A1_set_sizes.png",
        "F-A2_dac_confusion.png",
        "F-A3_meta_distribution.png",
        "F-A4_whisper_proba.png",
    ]
    for name in expected_figs:
        f = figs_dir / name
        assert f.exists() and f.stat().st_size > 1024, (
            f"{name} missing or suspiciously small (<1KB)"
        )
