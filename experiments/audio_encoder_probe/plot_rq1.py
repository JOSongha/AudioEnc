"""Visualization for RQ1 layer-wise distance analysis.

Spec: docs/analysis/RQ1_layer_distance.md §6 (visualization).

Inputs (from compute_cka.py):
  cka/{family}_within_{pooling}_{subset}.npy   shape (15, 15)
  cka/cross_position_{pooling}_{subset}.npy    shape (6, 15)
  cka/layer_positions.npy, cka/family_pairs.npy

Outputs (to {out_dir}/figs/):
  Main (overall):
    within_{family}_{pooling}.png       (4 family × 2 pooling = 8 heatmaps)
    cross_position_{pooling}.png        (2 cross heatmaps)
    cross_position_line_{pooling}.png   (2 cross line plots)
    smoothness_{pooling}.png            (2 adjacent-layer CKA line plots)
  Appendix (emotion-conditioned):
    appendix/within_{family}_{pooling}_{emotion}.png   (32 heatmaps)
    appendix/cross_position_{pooling}_{emotion}.png    (8 cross heatmaps)

Usage:
    python experiments/audio_encoder_probe/plot_rq1.py
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FAMILIES = ["whisper_tiny", "whisper_small", "dacvae", "wavtok"]
POOLINGS = ("mean", "last")
EMOTIONS = ("angry", "happy", "sad", "neutral")
# Group boundaries: enc (0..4), proj (5..9), llm (10..14)
GROUP_SPLITS = (5, 10)
CMAP = "viridis"


def plot_within(M: np.ndarray, layers: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(layers, fontsize=8)
    for s in GROUP_SPLITS:
        ax.axhline(s - 0.5, color="white", linewidth=1.5)
        ax.axvline(s - 0.5, color="white", linewidth=1.5)
    for i in range(len(layers)):
        for j in range(len(layers)):
            color = "white" if M[i, j] < 0.5 else "black"
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color=color, fontsize=6)
    plt.colorbar(im, ax=ax, label="Linear CKA", shrink=0.85)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross(C: np.ndarray, layers: list[str], pairs: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(C, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(pairs)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([p.replace("__", " ↔ ") for p in pairs], fontsize=8)
    for s in GROUP_SPLITS:
        ax.axvline(s - 0.5, color="white", linewidth=1.5)
    for i in range(len(pairs)):
        for j in range(len(layers)):
            color = "white" if C[i, j] < 0.5 else "black"
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", color=color, fontsize=6)
    plt.colorbar(im, ax=ax, label="Linear CKA", shrink=0.85)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_line(C: np.ndarray, layers: list[str], pairs: list[str],
                    title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(layers))
    for k, pair in enumerate(pairs):
        ax.plot(x, C[k], marker="o", label=pair.replace("__", " ↔ "))
    for s in GROUP_SPLITS:
        ax.axvline(s - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer position")
    ax.set_ylabel("Linear CKA")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_smoothness(mats: dict, layers: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(layers) - 1)
    for fam, M in mats.items():
        adj = [M[i, i + 1] for i in range(len(layers) - 1)]
        ax.plot(x, adj, marker="o", label=fam)
    for s in GROUP_SPLITS:
        ax.axvline(s - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{layers[i]}->{layers[i + 1]}" for i in range(len(layers) - 1)],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer transition")
    ax.set_ylabel("Linear CKA (adjacent)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cka-dir", default=str(REPO / "experiments/audio_encoder_probe/cka"))
    parser.add_argument("--out-dir", default=str(REPO / "experiments/audio_encoder_probe/figs"))
    args = parser.parse_args()

    cka_dir = Path(args.cka_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir = out_dir / "appendix"
    appendix_dir.mkdir(exist_ok=True)

    layers = list(np.load(cka_dir / "layer_positions.npy"))
    pairs = list(np.load(cka_dir / "family_pairs.npy"))

    for pooling in POOLINGS:
        within_mats = {}
        for fam in FAMILIES:
            M = np.load(cka_dir / f"{fam}_within_{pooling}_overall.npy")
            plot_within(M, layers, f"{fam} - Axis 1 ({pooling}, overall, N=2000)",
                        out_dir / f"within_{fam}_{pooling}.png")
            within_mats[fam] = M
        C = np.load(cka_dir / f"cross_position_{pooling}_overall.npy")
        plot_cross(C, layers, pairs, f"Axis 2 cross-family ({pooling}, overall, N=2000)",
                   out_dir / f"cross_position_{pooling}.png")
        plot_cross_line(C, layers, pairs, f"Axis 2 cross-family ({pooling}, overall, N=2000)",
                        out_dir / f"cross_position_line_{pooling}.png")
        plot_smoothness(within_mats, layers,
                        f"Adjacent-layer CKA smoothness ({pooling}, overall, N=2000)",
                        out_dir / f"smoothness_{pooling}.png")
    logger.info(f"main: {sum(1 for _ in out_dir.glob('*.png'))} png files")

    for emo in EMOTIONS:
        for pooling in POOLINGS:
            for fam in FAMILIES:
                M = np.load(cka_dir / f"{fam}_within_{pooling}_{emo}.npy")
                plot_within(M, layers, f"{fam} - Axis 1 ({pooling}, {emo}, N=500)",
                            appendix_dir / f"within_{fam}_{pooling}_{emo}.png")
            C = np.load(cka_dir / f"cross_position_{pooling}_{emo}.npy")
            plot_cross(C, layers, pairs, f"Axis 2 ({pooling}, {emo}, N=500)",
                       appendix_dir / f"cross_position_{pooling}_{emo}.png")
    logger.info(f"appendix: {sum(1 for _ in appendix_dir.glob('*.png'))} png files")
    logger.info(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
