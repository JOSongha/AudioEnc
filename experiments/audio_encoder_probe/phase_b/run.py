"""Phase B orchestrator: frame-level probe → trajectories → metrics → figures.

Steps:
  1. Train per-fold frame classifiers (whisper_small + dacvae)
  2. Extract probability trajectories for W+A- and W+A+ sets
  3. Compute per-utt metrics
  4. Produce figures F-B1 (metric violins) and F-B2 (example timelines)
"""

import argparse  # noqa: I001
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001
import numpy as np
import pandas as pd

from experiments.audio_encoder_probe.phase_b.frame_probe import train_all_folds
from experiments.audio_encoder_probe.phase_b.metrics import compute_metrics_for_sets
from experiments.audio_encoder_probe.phase_b.trajectories import extract_trajectories

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[3]
PROBE_ROOT = ROOT / "experiments" / "audio_encoder_probe"

SET_COLORS = {
    "W+A+": "#22c55e",
    "W+A-": "#ef4444",
    "W-A+": "#3b82f6",
    "W-A-": "#9ca3af",
}


def _save(fig: plt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_metric_violin_fig(metrics_df: pd.DataFrame, out_png: Path) -> None:
    """F-B1: violin plots of 4 key metrics for W+A- vs W+A+."""
    plot_metrics = ["truth_prob_gap", "sharpness_W", "sharpness_D", "frame_agreement"]
    labels_pretty = {
        "truth_prob_gap": "Truth-prob gap\n(W - D, mean over frames)",
        "sharpness_W": "Confidence sharpness\n(Whisper)",
        "sharpness_D": "Confidence sharpness\n(DAC)",
        "frame_agreement": "Frame agreement\n(argmax W == argmax D)",
    }
    sets = ["W+A-", "W+A+"]
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(16, 4))
    for ax, metric in zip(axes, plot_metrics):
        data = [metrics_df.loc[metrics_df["set_label"] == s, metric].dropna().values for s in sets]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body, s in zip(parts["bodies"], sets):
            body.set_facecolor(SET_COLORS[s])
            body.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(sets)
        ax.set_title(labels_pretty[metric], fontsize=9)
        # Annotate median
        for i, (d, s) in enumerate(zip(data, sets), 1):
            if len(d):
                ax.text(i, np.median(d), f"{np.median(d):.3f}", ha="center", va="bottom", fontsize=7, color="black")
    fig.suptitle("F-B1: Frame-level metrics — W+A- vs W+A+")
    _save(fig, out_png)


def make_timeline_fig(
    utt_id: str,
    true_label: str,
    traj_W_npz: Path,
    traj_D_npz: Path,
    out_png: Path,
) -> None:
    """F-B2: 2-panel class-prob heatmap timeline for one utterance."""
    if not traj_W_npz.exists() or not traj_D_npz.exists():
        return

    nW = np.load(str(traj_W_npz))
    nD = np.load(str(traj_D_npz))
    probs_W = nW["probs"]  # [T_W, C]
    probs_D = nD["probs"]  # [T_D, C]
    classes_W = list(nW["classes"])
    classes_D = list(nD["classes"])
    fps_W = float(nW["fps"])
    fps_D = float(nD["fps"])

    # Shared class order
    all_classes = sorted(set(classes_W) | set(classes_D))

    def align_probs(probs, classes, target_classes):
        aligned = np.zeros((len(probs), len(target_classes)), dtype=np.float32)
        for i, c in enumerate(target_classes):
            if c in classes:
                aligned[:, i] = probs[:, classes.index(c)]
        return aligned

    pW = align_probs(probs_W, classes_W, all_classes)
    pD = align_probs(probs_D, classes_D, all_classes)

    t_W = np.arange(len(pW)) / fps_W
    t_D = np.arange(len(pD)) / fps_D
    true_idx = all_classes.index(true_label) if true_label in all_classes else -1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    for ax, probs, fps, title, t_axis in [
        (ax1, pW, fps_W, "Whisper", t_W),
        (ax2, pD, fps_D, "DAC", t_D),
    ]:
        im = ax.imshow(
            probs.T,
            aspect="auto",
            origin="lower",
            extent=[0, len(probs) / fps, -0.5, len(all_classes) - 0.5],
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
        )
        ax.set_yticks(range(len(all_classes)))
        ax.set_yticklabels(all_classes, fontsize=7)
        ax.set_xlabel("time (s)")
        ax.set_title(f"{title}  |  utt: {utt_id}  |  true: {true_label}", fontsize=8)
        if true_idx >= 0:
            ax.axhline(true_idx, color="cyan", linewidth=1.5, linestyle="--", alpha=0.8)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

    fig.suptitle(f"F-B2: Frame probability timeline — {utt_id}", fontsize=9)
    _save(fig, out_png)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--combined_csv",
        type=Path,
        default=PROBE_ROOT / "_results" / "predictions" / "combined_whisper_small_dacvae_nsynth_train_30k.csv",
    )
    p.add_argument("--sample_sets_csv", type=Path, default=PROBE_ROOT / "_results" / "phase_a" / "sample_sets.csv")
    p.add_argument("--manifest_csv", type=Path, default=PROBE_ROOT / "manifests" / "nsynth_train_30k.csv")
    p.add_argument("--out_dir", type=Path, default=PROBE_ROOT / "_results" / "phase_b")
    p.add_argument(
        "--top_pairs_k", type=int, default=3, help="number of timeline examples (one per top confused pair)"
    )
    p.add_argument("--frames_per_utt", type=int, default=20)
    p.add_argument("--C", type=float, default=0.1)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = args.out_dir / "figs"

    combined_df = pd.read_csv(args.combined_csv)
    sets_df = pd.read_csv(args.sample_sets_csv)
    manifest_df = pd.read_csv(args.manifest_csv)
    fold_map = dict(zip(manifest_df["utt_id"], manifest_df["fold"]))

    clf_dir = args.out_dir / "classifiers"

    # ── Step 1: Train classifiers ──────────────────────────────────────────
    for encoder in ("whisper_small", "dacvae"):
        logger.info("=== Training %s classifiers ===", encoder)
        train_all_folds(encoder, args.manifest_csv, clf_dir, args.frames_per_utt, args.C)

    # ── Step 2: Extract trajectories for W+A- and W+A+ ────────────────────
    target_utt_ids = sets_df[sets_df["set_label"].isin(["W+A-", "W+A+"])]["utt_id"].tolist()
    for encoder in ("whisper_small", "dacvae"):
        logger.info("=== Extracting trajectories: %s ===", encoder)
        extract_trajectories(
            encoder,
            target_utt_ids,
            fold_map,
            clf_dir,
            args.out_dir / "trajectories" / encoder,
        )

    # ── Step 3: Compute metrics ────────────────────────────────────────────
    logger.info("=== Computing metrics ===")
    traj_W_dir = args.out_dir / "trajectories" / "whisper_small"
    traj_D_dir = args.out_dir / "trajectories" / "dacvae"
    metrics_df = compute_metrics_for_sets(
        sets_df,
        combined_df,
        traj_W_dir,
        traj_D_dir,
        target_sets=["W+A-", "W+A+"],
    )
    metrics_df.to_csv(args.out_dir / "frame_metrics.csv", index=False)
    logger.info("Saved frame_metrics.csv  shape=%s", metrics_df.shape)

    # Quick summary
    for s in ["W+A-", "W+A+"]:
        sub = metrics_df[metrics_df["set_label"] == s]
        logger.info(
            "%s (n=%d): gap=%.3f±%.3f  sharpW=%.3f  sharpD=%.3f  agree=%.3f",
            s,
            len(sub),
            sub["truth_prob_gap"].mean(),
            sub["truth_prob_gap"].std(),
            sub["sharpness_W"].mean(),
            sub["sharpness_D"].mean(),
            sub["frame_agreement"].mean(),
        )

    # ── Step 4: Figures ────────────────────────────────────────────────────
    make_metric_violin_fig(metrics_df, figs_dir / "F-B1_metric_violins.png")

    # Timeline examples: highest-Whisper-confidence utt from each top confused pair
    top_pairs_csv = PROBE_ROOT / "_results" / "phase_a" / "top_confused_pairs.csv"
    if top_pairs_csv.exists():
        top_pairs = pd.read_csv(top_pairs_csv).head(args.top_pairs_k)
        wam = sets_df[sets_df["set_label"] == "W+A-"][["utt_id", "set_label"]].copy()
        wam = wam.merge(
            combined_df[["utt_id", "whisper_small_proba", "dacvae_pred", "true_label"]], on="utt_id", how="left"
        )
        for _, pair_row in top_pairs.iterrows():
            true_c, pred_c = pair_row["true_label"], pair_row["dacvae_pred"]
            sub = wam[(wam["true_label"] == true_c) & (wam["dacvae_pred"] == pred_c)]
            if sub.empty:
                continue
            example = sub.nlargest(1, "whisper_small_proba").iloc[0]
            uid = example["utt_id"]
            make_timeline_fig(
                uid,
                true_c,
                traj_W_dir / f"{uid}.npz",
                traj_D_dir / f"{uid}.npz",
                figs_dir / f"F-B2_timeline_{true_c}_vs_{pred_c}.png",
            )

    logger.info("Phase B complete → %s", args.out_dir)


if __name__ == "__main__":
    main()
