"""Plot Stage-2 v1 ckpt trajectories as a multi-panel PDF figure.

Reads the long-format CSV emitted by `aggregate_results.py` and renders one
panel per task with the metric on the y-axis and ckpt step on the x-axis.
Best ckpt per panel is annotated.

Usage:
    python -m evaluation.stage2.plot_trajectories \
        --csv /mnt/tmp/.../analysis/results.csv \
        --out /mnt/tmp/.../analysis/trajectories.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Panel layout: list of (panel_title, [(metric, label, lower_is_better, color)])
# Each panel can show multiple lines (e.g. per-corpus emotion).
PANELS = [
    ("Source-corpus emotion (macro-F1)", [
        ("MELD_F1",      "MELD",      False, "#1f77b4"),
        ("DailyTalk_F1", "DailyTalk", False, "#ff7f0e"),
        ("EmoV_F1",      "EmoV-DB",   False, "#2ca02c"),
        ("RAVDESS_F1",   "RAVDESS",   False, "#d62728"),
    ]),
    ("Source-corpus emotion (accuracy)", [
        ("MELD_acc",      "MELD",      False, "#1f77b4"),
        ("DailyTalk_acc", "DailyTalk", False, "#ff7f0e"),
        ("EmoV_acc",      "EmoV-DB",   False, "#2ca02c"),
        ("RAVDESS_acc",   "RAVDESS",   False, "#d62728"),
    ]),
    ("ESC-50 accuracy", [
        ("ESC50_acc", "ESC-50", False, "#1f77b4"),
    ]),
    ("FSD50K (greedy F1)", [
        ("FSD50K_F1mi", "F1-micro", False, "#1f77b4"),
        ("FSD50K_F1ma", "F1-macro", False, "#ff7f0e"),
        ("FSD50K_Jacc", "Jaccard",  False, "#2ca02c"),
    ]),
    ("FSD50K (sequence-scoring mAP)", [
        ("FSD50K_mAPma", "mAP-macro", False, "#1f77b4"),
        ("FSD50K_mAPmi", "mAP-micro", False, "#ff7f0e"),
    ]),
    ("Clotho captioning", [
        ("Clotho_BLEU1", "BLEU-1", False, "#1f77b4"),
        ("Clotho_BLEU4", "BLEU-4 ×5", False, "#ff7f0e"),  # scale BLEU-4 ×5 for legibility
    ]),
    ("LibriSpeech WER (lower better)", [
        ("WER_clean", "test-clean", True, "#1f77b4"),
        ("WER_other", "test-other", True, "#ff7f0e"),
    ]),
    ("Text retention (mean of 6)", [
        ("text_mean", "mean acc", False, "#1f77b4"),
    ]),
    ("LISTEN-test (training-parallel)", [
        ("LISTEN_acc", "accuracy", False, "#1f77b4"),
        ("LISTEN_F1",  "macro-F1", False, "#ff7f0e"),
    ]),
    ("LISTEN-official (audio-only mean across 5 exp)", [
        ("LISTENo_WAmean", "WA mean",  False, "#1f77b4"),
        ("LISTENo_F1mean", "F1 mean",  False, "#ff7f0e"),
    ]),
]


def load_csv(path: Path) -> dict[str, dict[int, float]]:
    """Returns {metric: {ckpt: value}}."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                out[row["metric"]][int(row["ckpt"])] = float(row["value"])
            except (ValueError, KeyError):
                continue
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=2,
                   help="number of subplot columns")
    args = p.parse_args()

    data = load_csv(Path(args.csv))
    panels = [(title, lines) for title, lines in PANELS
              if any(metric in data for metric, *_ in lines)]
    n = len(panels)
    cols = args.cols
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (title, lines) in enumerate(panels):
        ax = axes[idx]
        any_data = False
        for metric, label, lower, color in lines:
            series = data.get(metric, {})
            if not series:
                continue
            ckpts = sorted(series.keys())
            values = [series[c] for c in ckpts]
            # Special-case: scale Clotho BLEU-4 ×5 to share the BLEU-1 panel
            if metric == "Clotho_BLEU4":
                values = [v * 5 for v in values]
            ax.plot([c / 1000 for c in ckpts], values,
                    "o-", color=color, label=label, markersize=4, linewidth=1.4)
            # Mark best
            if lower:
                bi = min(range(len(values)), key=lambda i: values[i])
            else:
                bi = max(range(len(values)), key=lambda i: values[i])
            ax.scatter([ckpts[bi] / 1000], [values[bi]],
                       s=80, marker="*", color=color, edgecolor="black",
                       linewidth=0.5, zorder=5)
            any_data = True

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ckpt (×1k step)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if any_data:
            ax.legend(fontsize=7, loc="best", framealpha=0.85)

    # Hide unused axes
    for k in range(len(panels), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        "Stage-2 v1 — eval trajectory (1k → 25k step, ckpt-1000 step)\n"
        "★ marks per-line best ckpt. Lower-is-better for WER. "
        "Clotho BLEU-4 scaled ×5 to share BLEU-1 panel.",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"[plot] wrote {out} ({rows}×{cols} panels, {n} populated)")


if __name__ == "__main__":
    main()
