"""Smoke test for Phase B orchestrator."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.audio_encoder_probe.phase_b.metrics import compute_metrics_for_utt
from experiments.audio_encoder_probe.phase_b.run import main


FAMILIES = ["bass", "guitar", "organ", "keyboard", "flute",
            "brass", "mallet", "reed", "string", "vocal", "synth_lead"]
SOURCES = ["acoustic", "electronic", "synthetic"]


def _make_synthetic_data(tmp_path: Path, n_utts: int = 30):
    """Build minimal synthetic frame npzs, manifest, combined CSV, and sample_sets."""
    rng = np.random.default_rng(0)

    manifest_rows, combined_rows, sets_rows = [], [], []
    frames_W = tmp_path / "frames_W"
    frames_D = tmp_path / "frames_D"
    frames_W.mkdir()
    frames_D.mkdir()

    # n_utts: 60% W+A+, 40% W+A-
    for i in range(n_utts):
        fam = FAMILIES[i % len(FAMILIES)]
        src = SOURCES[i % 3]
        pitch, vel = (i * 3) % 128, [25, 50, 75, 100, 127][i % 5]
        utt_id = f"{fam}_{src}_{i:03d}-{pitch:03d}-{vel:03d}"
        fold = (i % 5) + 1
        wc = i < int(n_utts * 0.6)
        ac = i < int(n_utts * 0.6)
        wp = fam if wc else FAMILIES[(i + 1) % len(FAMILIES)]
        dp = fam if ac else FAMILIES[(i + 2) % len(FAMILIES)]
        set_label = "W+A+" if (wc and ac) else "W+A-"

        # Write frame npzs (T=10 for whisper, T=5 for dacvae)
        for d, T, fdir in [(768, 10, frames_W), (128, 5, frames_D)]:
            frames = rng.random((T, d), dtype=np.float32).astype(np.float16)
            np.savez_compressed(
                fdir / f"{utt_id}.npz",
                frames=frames, n_valid=np.int32(T),
            )

        manifest_rows.append({"utt_id": utt_id, "audio_path": f"/x/{utt_id}.wav",
                               "label": fam, "fold": fold})
        combined_rows.append({
            "utt_id": utt_id, "audio_path": f"/x/{utt_id}.wav",
            "true_label": fam, "fold": fold,
            "whisper_small_pred": wp, "whisper_small_proba": 0.8 + 0.01 * i,
            "whisper_small_correct": wc,
            "dacvae_pred": dp, "dacvae_proba": 0.5,
            "dacvae_correct": ac,
        })
        sets_rows.append({"utt_id": utt_id, "set_label": set_label, "true_label": fam})

    manifest_csv = tmp_path / "manifest.csv"
    combined_csv = tmp_path / "combined.csv"
    sets_csv = tmp_path / "sample_sets.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    pd.DataFrame(combined_rows).to_csv(combined_csv, index=False)
    pd.DataFrame(sets_rows).to_csv(sets_csv, index=False)
    return manifest_csv, combined_csv, sets_csv, frames_W, frames_D


def _plant_fake_classifiers(clf_dir: Path, manifest_csv: Path, frames_W: Path, frames_D: Path):
    """Write minimal pre-trained pkl classifiers so run.py can skip training."""
    manifest_df = pd.read_csv(manifest_csv)
    clf_dir.mkdir(parents=True, exist_ok=True)

    for encoder, d, fdir, fps in [
        ("whisper_small", 768, frames_W, 50.0),
        ("dacvae", 128, frames_D, 25.0),
    ]:
        # Build a tiny training set from all utts
        X, y = [], []
        for row in manifest_df.to_dict("records"):
            npz_path = fdir / f"{row['utt_id']}.npz"
            if npz_path.exists():
                frames = np.load(str(npz_path))["frames"].astype(np.float32)
                X.append(frames)
                y.extend([row["label"]] * len(frames))
        X = np.concatenate(X)
        y = np.array(y)
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=10, random_state=0)
        clf.fit(scaler.transform(X), y)
        for fold in sorted(manifest_df["fold"].unique()):
            pkl = clf_dir / f"{encoder}_fold{fold}.pkl"
            with open(pkl, "wb") as fh:
                pickle.dump({"scaler": scaler, "clf": clf, "fold": fold, "encoder": encoder}, fh)


def test_smoke_end_to_end(tmp_path: Path) -> None:
    """30-utt synthetic data → run.main produces metrics CSV and figures."""
    manifest_csv, combined_csv, sets_csv, frames_W, frames_D = _make_synthetic_data(tmp_path)

    out_dir = tmp_path / "out"
    clf_dir = out_dir / "classifiers"

    # Patch FRAMES_ROOT so run.py finds our synthetic frames
    import experiments.audio_encoder_probe.phase_b.frame_probe as fp_mod
    import experiments.audio_encoder_probe.phase_b.trajectories as tr_mod
    orig_FR_fp = fp_mod.FRAMES_ROOT
    orig_FR_tr = tr_mod.FRAMES_ROOT

    try:
        fp_mod.FRAMES_ROOT = tmp_path / "FAKE"  # won't be used — we pre-plant classifiers
        tr_mod.FRAMES_ROOT = tmp_path

        # Plant classifiers with correct paths pointing to our synthetic frames
        # Need to temporarily override FRAMES_ROOT for the clf dir lookup in trajectories
        # Instead: just plant classifiers and override FRAMES_ROOT in trajectories
        tr_mod.FRAMES_ROOT = tmp_path

        # Pre-plant classifiers (bypass train_all_folds for speed)
        _plant_fake_classifiers(clf_dir, manifest_csv, frames_W, frames_D)

        # Create fake sub-dirs matching encoder names
        (tmp_path / "whisper_small" / "nsynth_train_30k").mkdir(parents=True)
        (tmp_path / "dacvae" / "nsynth_train_30k").mkdir(parents=True)
        for f in frames_W.glob("*.npz"):
            (tmp_path / "whisper_small" / "nsynth_train_30k" / f.name).symlink_to(f)
        for f in frames_D.glob("*.npz"):
            (tmp_path / "dacvae" / "nsynth_train_30k" / f.name).symlink_to(f)

        # Monkey-patch top_confused_pairs path in run.py
        import experiments.audio_encoder_probe.phase_b.run as run_mod
        orig_probe_root = run_mod.PROBE_ROOT
        run_mod.PROBE_ROOT = tmp_path

        # Write fake top_confused_pairs.csv
        phase_a_dir = tmp_path / "_results" / "phase_a"
        phase_a_dir.mkdir(parents=True)
        pd.DataFrame([{"true_label": "bass", "dacvae_pred": "guitar", "count": 5}]).to_csv(
            phase_a_dir / "top_confused_pairs.csv", index=False
        )

        main([
            "--combined_csv", str(combined_csv),
            "--sample_sets_csv", str(sets_csv),
            "--manifest_csv", str(manifest_csv),
            "--out_dir", str(out_dir),
            "--frames_per_utt", "3",
        ])
    finally:
        fp_mod.FRAMES_ROOT = orig_FR_fp
        tr_mod.FRAMES_ROOT = orig_FR_tr
        run_mod.PROBE_ROOT = orig_probe_root

    assert (out_dir / "frame_metrics.csv").exists()
    metrics = pd.read_csv(out_dir / "frame_metrics.csv")
    assert "truth_prob_gap" in metrics.columns
    assert len(metrics) > 0
    assert (out_dir / "figs" / "F-B1_metric_violins.png").exists()


def test_compute_metrics_for_utt(tmp_path: Path) -> None:
    """Unit test for compute_metrics_for_utt with synthetic trajectories."""
    classes = np.array(["bass", "guitar", "organ"])
    rng = np.random.default_rng(1)

    # Whisper: high confidence on bass (index 0)
    probs_W = rng.dirichlet([10, 1, 1], size=20).astype(np.float32)
    # DAC: low/confused on bass
    probs_D = rng.dirichlet([1, 1, 1], size=10).astype(np.float32)

    traj_W = tmp_path / "w.npz"
    traj_D = tmp_path / "d.npz"
    np.savez(traj_W, probs=probs_W, classes=classes, n_valid=np.int32(20), fps=np.float32(50.0))
    np.savez(traj_D, probs=probs_D, classes=classes, n_valid=np.int32(10), fps=np.float32(25.0))

    m = compute_metrics_for_utt("test_utt", "bass", traj_W, traj_D)
    assert m["truth_prob_gap"] > 0, "Whisper should have higher truth prob than DAC"
    assert 0.0 <= m["frame_agreement"] <= 1.0
    assert m["sharpness_W"] >= 0.0
    assert m["argmax_time_W"] >= 0.0
