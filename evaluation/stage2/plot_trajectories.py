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
import matplotlib.patheffects as pe

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
    p.add_argument("--title", default=None,
                   help="custom suptitle (line 1). Default: derived from data span.")
    p.add_argument("--exclude", default="",
                   help="comma-separated panel-title substrings to skip "
                        "(case-insensitive, e.g. 'ESC-50,Text retention').")
    args = p.parse_args()

    data = load_csv(Path(args.csv))
    excludes = [e.strip().lower() for e in args.exclude.split(",") if e.strip()]
    def _excluded(title: str) -> bool:
        t = title.lower()
        return any(e in t for e in excludes)
    panels = [(title, lines) for title, lines in PANELS
              if any(metric in data for metric, *_ in lines)
              and not _excluded(title)]
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
            raw_values = [series[c] for c in ckpts]
            values = list(raw_values)
            # Special-case: scale Clotho BLEU-4 ×5 to share the BLEU-1 panel
            if metric == "Clotho_BLEU4":
                values = [v * 5 for v in values]
            # WER reported as percent (×100). y-axis clipped to [1, 100]
            # below so early-training spikes (>100%) drop off-axis.
            is_pct = metric.startswith("WER_") or metric.startswith("CER_")
            if is_pct:
                values = [v * 100 for v in values]
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
            # Annotate best value next to the star.
            best_step = ckpts[bi] // 1000
            if is_pct:
                txt = f"{values[bi]:.2f}% @{best_step}k"
            else:
                # show unscaled value (raw_values for BLEU-4 ×5)
                txt = f"{raw_values[bi]:.3f} @{best_step}k"
            ax.annotate(txt, xy=(ckpts[bi] / 1000, values[bi]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=color,
                        path_effects=[
                            pe.withStroke(linewidth=1.6, foreground="white")
                        ])
            any_data = True

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ckpt (×1k step)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        # WER reported in percent. Clip y-axis to [1, 100] so early-training
        # spikes (>100% at ckpt-1k from insertions) drop off-axis instead of
        # squashing the converged region.
        if "WER" in title:
            ax.set_ylim(1, 100)
            ax.set_ylabel("%", fontsize=8)
        if any_data:
            ax.legend(fontsize=7, loc="best", framealpha=0.85)

    # Hide unused axes
    for k in range(len(panels), len(axes)):
        axes[k].set_visible(False)

    all_ckpts = sorted({c for series in data.values() for c in series})
    if args.title:
        line1 = args.title
    elif all_ckpts:
        cmin, cmax = all_ckpts[0], all_ckpts[-1]
        # detect stride from most common gap
        gaps = [b-a for a,b in zip(all_ckpts, all_ckpts[1:])]
        stride = min(gaps) if gaps else 1000
        line1 = f"Eval trajectory ({cmin//1000}k → {cmax//1000}k step, ckpt-{stride} step)"
    else:
        line1 = "Eval trajectory"
    fig.suptitle(
        line1 + "\n"
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
