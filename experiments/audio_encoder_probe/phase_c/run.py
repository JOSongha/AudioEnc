"""Phase C orchestrator: decisive-frame analysis + figures (F-C1..F-C4).

Builds on Phase B trajectories to answer WHY Whisper wins on W+A- samples.
Key questions:
  Q1. Where in the 4s clip does Whisper make its decision? (onset / sustain / release)
  Q2. What is DAC doing at that exact frame? (entropy high → DAC sees noise there)
  Q3. Does timing correlate with NSynth meta (pitch / velocity / source)?
  Q4. Which instrument families show early vs late decisive frames?
"""

import argparse  # noqa: I001
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001
import numpy as np
import pandas as pd
from scipy import stats

from experiments.audio_encoder_probe.phase_c.decisive_frame import compute_decisive_frame_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[3]
PROBE_ROOT = ROOT / "experiments" / "audio_encoder_probe"

SET_COLORS = {"W+A-": "#ef4444", "W+A+": "#22c55e"}
SOURCE_COLORS = {"acoustic": "#3b82f6", "electronic": "#f97316", "synthetic": "#8b5cf6"}


def _save(fig: plt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_timing_fig(decisive_W: pd.DataFrame, decisive_Wap: pd.DataFrame, out_png: Path) -> None:
    """F-C1: argmax_time_W distribution (W+A- vs W+A+) + onset/sustain/release breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: overlapping KDE / histogram
    ax = axes[0]
    bins = np.linspace(0, 4, 41)
    ax.hist(
        decisive_W["argmax_time_W"].dropna(),
        bins=bins,
        density=True,
        alpha=0.5,
        color=SET_COLORS["W+A-"],
        label="W+A-",
    )
    ax.hist(
        decisive_Wap["argmax_time_W"].dropna(),
        bins=bins,
        density=True,
        alpha=0.5,
        color=SET_COLORS["W+A+"],
        label="W+A+",
    )
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="onset boundary")
    ax.axvline(1.5, color="gray", linestyle=":", linewidth=0.8, label="sustain boundary")
    ax.set_xlabel("Whisper decisive frame (s)")
    ax.set_ylabel("density")
    ax.set_title("F-C1a: Whisper decisive-frame timing distribution")
    ax.legend(fontsize=8)

    # Right: W+A- breakdown by source
    ax = axes[1]
    if "source" in decisive_W.columns:
        for src, grp in decisive_W.groupby("source"):
            ax.hist(
                grp["argmax_time_W"].dropna(),
                bins=bins,
                density=True,
                alpha=0.5,
                color=SOURCE_COLORS.get(src, "gray"),
                label=src,
            )
        ax.set_xlabel("Whisper decisive frame (s)")
        ax.set_ylabel("density")
        ax.set_title("F-C1b: W+A- by source")
        ax.legend(fontsize=8)
    _save(fig, out_png)


def make_dac_entropy_fig(decisive_W: pd.DataFrame, decisive_Wap: pd.DataFrame, out_png: Path) -> None:
    """F-C2: DAC entropy at Whisper's decisive frame — W+A- vs W+A+."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Violin: dac_entropy_at_decisive
    ax = axes[0]
    data = [
        decisive_W["dac_entropy_at_decisive"].dropna().values,
        decisive_Wap["dac_entropy_at_decisive"].dropna().values,
    ]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, s in zip(parts["bodies"], ["W+A-", "W+A+"]):
        body.set_facecolor(SET_COLORS[s])
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["W+A-", "W+A+"])
    for i, d in enumerate(data, 1):
        ax.text(i, np.median(d), f"{np.median(d):.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Shannon entropy (nats)")
    ax.set_title("F-C2a: DAC entropy at Whisper's decisive frame")

    # Scatter: DAC truth prob vs Whisper truth prob at decisive frame (W+A- only)
    ax = axes[1]
    sub = decisive_W.dropna(subset=["whisper_truth_prob_at_decisive", "dac_truth_prob_at_decisive"])
    ax.scatter(
        sub["whisper_truth_prob_at_decisive"],
        sub["dac_truth_prob_at_decisive"],
        alpha=0.05,
        s=3,
        color=SET_COLORS["W+A-"],
    )
    lim = [0, 1]
    ax.plot(lim, lim, "k--", linewidth=0.8, label="equal")
    ax.set_xlabel("Whisper P(true | decisive frame)")
    ax.set_ylabel("DAC P(true | Whisper's decisive frame)")
    ax.set_title("F-C2b: Whisper vs DAC truth prob at decisive frame (W+A-)")
    ax.legend(fontsize=8)
    _save(fig, out_png)


def make_meta_correlation_fig(decisive_W: pd.DataFrame, out_png: Path) -> None:
    """F-C3: Meta correlations — timing / sharpness by source, velocity, pitch."""
    if not all(c in decisive_W.columns for c in ("source", "velocity", "pitch")):
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Source: sharpness_W and dac_entropy
    ax = axes[0]
    sources = sorted(decisive_W["source"].dropna().unique())
    x = np.arange(len(sources))
    sharp_means = [decisive_W[decisive_W.source == s]["dac_entropy_at_decisive"].mean() for s in sources]
    ax.bar(x, sharp_means, color=[SOURCE_COLORS.get(s, "gray") for s in sources])
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylabel("mean DAC entropy at decisive frame")
    ax.set_title("F-C3a: DAC entropy at decisive frame by source (W+A-)")

    # Velocity: argmax_time_W
    ax = axes[1]
    vels = sorted(decisive_W["velocity"].dropna().unique())
    v_means = [decisive_W[decisive_W.velocity == v]["argmax_time_W"].mean() for v in vels]
    v_sems = [decisive_W[decisive_W.velocity == v]["argmax_time_W"].sem() for v in vels]
    ax.errorbar(range(len(vels)), v_means, yerr=v_sems, fmt="o-", capsize=4, color="#ef4444")
    ax.set_xticks(range(len(vels)))
    ax.set_xticklabels([str(v) for v in vels])
    ax.set_xlabel("velocity")
    ax.set_ylabel("argmax_time_W (s)")
    ax.set_title("F-C3b: Decisive frame timing vs velocity (W+A-)")

    # Pitch: scatter with regression line
    ax = axes[2]
    sub = decisive_W.dropna(subset=["pitch", "argmax_time_W"])
    ax.scatter(sub["pitch"], sub["argmax_time_W"], alpha=0.03, s=2, color="#ef4444")
    slope, intercept, r, p, _ = stats.linregress(sub["pitch"], sub["argmax_time_W"])
    px = np.array([sub.pitch.min(), sub.pitch.max()])
    ax.plot(px, slope * px + intercept, "k-", linewidth=1.5, label=f"r={r:.3f}  p={p:.2e}")
    ax.set_xlabel("pitch (MIDI)")
    ax.set_ylabel("argmax_time_W (s)")
    ax.set_title("F-C3c: Decisive frame timing vs pitch (W+A-)")
    ax.legend(fontsize=8)
    _save(fig, out_png)


def make_family_timing_fig(decisive_W: pd.DataFrame, out_png: Path) -> None:
    """F-C4: Per-family median decisive-frame timing and DAC entropy (W+A- only)."""
    if "true_label" not in decisive_W.columns:
        return

    grp = (
        decisive_W.groupby("true_label")
        .agg(
            timing_median=("argmax_time_W", "median"),
            dac_entropy_median=("dac_entropy_at_decisive", "median"),
            n=("utt_id", "count"),
        )
        .reset_index()
        .sort_values("timing_median")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.barh(grp["true_label"], grp["timing_median"], color="#ef4444", alpha=0.8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(1.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("median argmax_time_W (s)")
    ax.set_title("F-C4a: Decisive-frame timing by family (W+A-, sorted)")

    ax = axes[1]
    grp2 = (
        decisive_W.groupby("true_label")
        .agg(
            dac_entropy_median=("dac_entropy_at_decisive", "median"),
        )
        .reset_index()
        .sort_values("dac_entropy_median", ascending=False)
    )
    ax.barh(grp2["true_label"], grp2["dac_entropy_median"], color="#3b82f6", alpha=0.8)
    ax.set_xlabel("median DAC entropy at decisive frame (nats)")
    ax.set_title("F-C4b: DAC entropy at decisive frame by family (W+A-)")
    _save(fig, out_png)


def print_summary(decisive_W: pd.DataFrame, decisive_Wap: pd.DataFrame) -> None:
    """Print Phase C key findings to stdout."""
    logger.info("=== Phase C Summary ===")
    for label, sub in [("W+A-", decisive_W), ("W+A+", decisive_Wap)]:
        sub_clean = sub.dropna(subset=["argmax_time_W", "dac_entropy_at_decisive"])
        logger.info(
            "%s: argmax_time_W median=%.3fs  dac_entropy_at_decisive median=%.4f  n=%d",
            label,
            sub_clean["argmax_time_W"].median(),
            sub_clean["dac_entropy_at_decisive"].median(),
            len(sub_clean),
        )
    # Mann-Whitney U test on DAC entropy
    from scipy.stats import mannwhitneyu

    e_wam = decisive_W["dac_entropy_at_decisive"].dropna().values
    e_wap = decisive_Wap["dac_entropy_at_decisive"].dropna().values
    u, p = mannwhitneyu(e_wam, e_wap, alternative="greater")
    logger.info("Mann-Whitney U (W+A- entropy > W+A+): U=%.0f  p=%.2e", u, p)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--combined_csv",
        type=Path,
        default=PROBE_ROOT / "_results" / "predictions" / "combined_whisper_small_dacvae_nsynth_train_30k.csv",
    )
    p.add_argument("--metrics_csv", type=Path, default=PROBE_ROOT / "_results" / "phase_b" / "frame_metrics.csv")
    p.add_argument("--meta_csv", type=Path, default=PROBE_ROOT / "manifests" / "nsynth_train_30k_meta.csv")
    p.add_argument(
        "--traj_W_dir", type=Path, default=PROBE_ROOT / "_results" / "phase_b" / "trajectories" / "whisper_small"
    )
    p.add_argument("--traj_D_dir", type=Path, default=PROBE_ROOT / "_results" / "phase_b" / "trajectories" / "dacvae")
    p.add_argument("--out_dir", type=Path, default=PROBE_ROOT / "_results" / "phase_c")
    p.add_argument("--no_audio", action="store_true", help="skip mel_energy / spectral_centroid (faster)")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = args.out_dir / "figs"

    combined_df = pd.read_csv(args.combined_csv)
    metrics_df = pd.read_csv(args.metrics_csv)
    meta_df = pd.read_csv(args.meta_csv)

    logger.info("Computing decisive-frame features for W+A- ...")
    decisive_W = compute_decisive_frame_features(
        metrics_df,
        combined_df,
        args.traj_W_dir,
        args.traj_D_dir,
        target_set="W+A-",
        compute_audio_features=not args.no_audio,
    )
    decisive_W = decisive_W.merge(meta_df[["utt_id", "pitch", "velocity", "source"]], on="utt_id", how="left")

    logger.info("Computing decisive-frame features for W+A+ ...")
    decisive_Wap = compute_decisive_frame_features(
        metrics_df,
        combined_df,
        args.traj_W_dir,
        args.traj_D_dir,
        target_set="W+A+",
        compute_audio_features=False,  # W+A+ audio analysis optional
    )
    decisive_Wap = decisive_Wap.merge(meta_df[["utt_id", "pitch", "velocity", "source"]], on="utt_id", how="left")

    decisive_W.to_csv(args.out_dir / "decisive_frame_wam.csv", index=False)
    decisive_Wap.to_csv(args.out_dir / "decisive_frame_wap.csv", index=False)
    logger.info("Saved decisive_frame CSVs")

    print_summary(decisive_W, decisive_Wap)

    make_timing_fig(decisive_W, decisive_Wap, figs_dir / "F-C1_timing_distribution.png")
    make_dac_entropy_fig(decisive_W, decisive_Wap, figs_dir / "F-C2_dac_entropy.png")
    make_meta_correlation_fig(decisive_W, figs_dir / "F-C3_meta_correlation.png")
    make_family_timing_fig(decisive_W, figs_dir / "F-C4_family_timing.png")

    logger.info("Phase C complete → %s", args.out_dir)


if __name__ == "__main__":
    main()
