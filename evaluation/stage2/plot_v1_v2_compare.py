"""Overlay v1 vs v2 ckpt trajectories as a multi-panel PDF.

Each panel renders one metric (or related metrics) with v1 and v2 as
two distinguishable series, sharing the same axes for direct comparison.
Best-per-run ckpt marked with star.

Usage:
    python -m evaluation.stage2.plot_v1_v2_compare \
        --v1-csv /mnt/tmp/.../analysis/results.csv \
        --v2-csv /mnt/tmp/.../analysis/results.csv \
        --out /mnt/tmp/.../v1_vs_v2.pdf
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (panel_title, [(metric, label, lower_is_better)])
PANELS = [
    ("Source-corpus emotion (macro-F1)", [
        ("MELD_F1",      "MELD",      False),
        ("DailyTalk_F1", "DailyTalk", False),
        ("EmoV_F1",      "EmoV-DB",   False),
        ("RAVDESS_F1",   "RAVDESS",   False),
    ]),
    ("Source-corpus emotion (accuracy)", [
        ("MELD_acc",      "MELD",      False),
        ("DailyTalk_acc", "DailyTalk", False),
        ("EmoV_acc",      "EmoV-DB",   False),
        ("RAVDESS_acc",   "RAVDESS",   False),
    ]),
    ("ESC-50 accuracy", [
        ("ESC50_acc", "ESC-50", False),
    ]),
    ("FSD50K (sequence-scoring mAP)", [
        ("FSD50K_mAPmi", "mAP-micro", False),
        ("FSD50K_mAPma", "mAP-macro", False),
    ]),
    ("Clotho captioning (BLEU-4)", [
        ("Clotho_BLEU4", "BLEU-4", False),
    ]),
    ("LibriSpeech WER (↓ better)", [
        ("WER_clean", "test-clean", True),
        ("WER_other", "test-other", True),
    ]),
    ("Text retention (mean of 6)", [
        ("text_mean", "mean acc", False),
    ]),
    ("LISTEN-test MCQA", [
        ("LISTEN_acc", "accuracy", False),
        ("LISTEN_F1",  "macro-F1", False),
    ]),
    ("LISTEN-official audio-only", [
        ("LISTENo_WAmean", "WA mean",  False),
        ("LISTENo_F1mean", "F1 mean",  False),
    ]),
]

V1_COLOR = "#1f77b4"  # blue
V2_COLOR = "#d62728"  # red


def load_csv(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                out[row["metric"]][int(row["ckpt"])] = float(row["value"])
            except (ValueError, KeyError):
                continue
    return out


def plot_series(ax, series, color, label_prefix, lower, marker_style):
    if not series:
        return None
    ckpts = sorted(series.keys())
    values = [series[c] for c in ckpts]
    line, = ax.plot([c / 1000 for c in ckpts], values, marker_style,
                    color=color, label=label_prefix, markersize=3.5,
                    linewidth=1.2, alpha=0.85)
    if lower:
        bi = min(range(len(values)), key=lambda i: values[i])
    else:
        bi = max(range(len(values)), key=lambda i: values[i])
    ax.scatter([ckpts[bi] / 1000], [values[bi]],
               s=90, marker="*", color=color, edgecolor="black",
               linewidth=0.5, zorder=5)
    return (ckpts[bi], values[bi])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1-csv", required=True)
    p.add_argument("--v2-csv", required=True)
    p.add_argument("--v1-label", default="v1 (asr14/emo34/env35/txt17)")
    p.add_argument("--v2-label", default="v2 (emoFull/asr033/env05/txt03)")
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=3)
    args = p.parse_args()

    v1 = load_csv(Path(args.v1_csv))
    v2 = load_csv(Path(args.v2_csv))
    panels = [(t, ms) for t, ms in PANELS
              if any(m in v1 or m in v2 for m, *_ in ms)]
    n = len(panels)
    cols = args.cols
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.4))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (title, metrics) in enumerate(panels):
        ax = axes[idx]
        # When a panel has multiple metrics, plot each with v1 solid + v2 dashed
        # and append metric label to the legend entry.
        for metric, sublabel, lower in metrics:
            v1_best = plot_series(
                ax, v1.get(metric, {}), V1_COLOR,
                f"v1 {sublabel}" if len(metrics) > 1 else "v1",
                lower, "o-",
            )
            v2_best = plot_series(
                ax, v2.get(metric, {}), V2_COLOR,
                f"v2 {sublabel}" if len(metrics) > 1 else "v2",
                lower, "s--",
            )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ckpt (×1k step)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.5, loc="best", framealpha=0.85, ncol=2 if len(metrics) > 1 else 1)

    for k in range(len(panels), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f"Stage-2 v1 vs v2 — overlaid eval trajectories\n"
        f"v1 ({args.v1_label}) solid blue ●  vs  v2 ({args.v2_label}) dashed red ■   ★ = per-run best ckpt",
        fontsize=10, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"[plot] wrote {out} ({rows}×{cols} panels, {n} populated)")


if __name__ == "__main__":
    main()
