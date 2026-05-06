"""Phase A orchestrator: figures (F-A1..F-A4) + HTML viewer (F-A5).

Reads combined predictions CSV and meta-augmented manifest; produces deliverables
under --out_dir. Calls Stage 1-4 modules' pure functions directly.
"""

import argparse  # noqa: I001
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001
import numpy as np
import pandas as pd

from experiments.audio_encoder_probe._html_utils import audio_relpath
from experiments.audio_encoder_probe.phase_a.descriptive import (
    SET_LABELS,
    dac_confusion_on_set,
    proba_distribution_summary,
    set_sizes,
    top_confused_pairs,
)
from experiments.audio_encoder_probe.phase_a.meta_dist import (
    chi_square_test,
    join_sets_with_meta,
    meta_distribution,
)
from experiments.audio_encoder_probe.phase_a.sample_sets import build_sample_sets, to_long_csv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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


def make_set_sizes_fig(sizes_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [SET_COLORS[s] for s in sizes_df["set"]]
    ax.bar(sizes_df["set"], sizes_df["count"], color=colors)
    for i, (cnt, frac) in enumerate(zip(sizes_df["count"], sizes_df["fraction"])):
        ax.text(i, cnt, f"{cnt}\n({frac:.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title("F-A1: Sample set sizes (Whisper × DAC correctness on NSynth)")
    _save(fig, out_png)


def make_dac_confusion_fig(cm: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("DAC predicted family")
    ax.set_ylabel("true family")
    ax.set_title("F-A2: DAC confusion on W+A- only (diagonal=0 by definition)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    max_val = cm.values.max()
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            v = cm.iat[i, j]
            if v > 0:
                color = "white" if (max_val > 0 and v > max_val / 2) else "black"
                ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=8)
    _save(fig, out_png)


def make_meta_distribution_fig(
    joined_df: pd.DataFrame,
    chi2_results: dict[str, dict],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, attr in zip(axes, ("source", "velocity", "pitch_bin")):
        dist = meta_distribution(joined_df, attr)
        piv = dist.pivot(index=attr, columns="set_label", values="fraction").fillna(0.0)
        x = np.arange(len(piv.index))
        width = 0.2
        for i, label in enumerate(SET_LABELS):
            if label not in piv.columns:
                continue
            ax.bar(
                x + (i - 1.5) * width,
                piv[label].values,
                width,
                label=label,
                color=SET_COLORS[label],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in piv.index], rotation=30, ha="right")
        ax.set_ylabel("fraction within set")
        p = chi2_results.get(attr, {}).get("p", float("nan"))
        ax.set_title(f"{attr}\nχ² p={p:.2e} (W+A- vs rest)")
        ax.legend(fontsize=8)
    fig.suptitle("F-A3: Meta-attribute distribution by set")
    _save(fig, out_png)


def make_proba_violin_fig(sets: dict[str, pd.DataFrame], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [sets[label]["whisper_small_proba"].values for label in SET_LABELS]
    parts = ax.violinplot(data, showmedians=True)
    for body, label in zip(parts["bodies"], SET_LABELS):
        body.set_facecolor(SET_COLORS[label])
        body.set_alpha(0.7)
    ax.set_xticks(range(1, len(SET_LABELS) + 1))
    ax.set_xticklabels(SET_LABELS)
    ax.set_ylabel("whisper_small pred_proba")
    ax.set_title("F-A4: Whisper proba distribution by set")
    _save(fig, out_png)


def make_html_viewer(
    sets: dict[str, pd.DataFrame],
    top_pairs: list[tuple[str, str, int]],
    examples_per_pair: int,
    out_html: Path,
) -> None:
    """Build HTML grouping audio players by top confused (true → DAC pred) family pairs."""
    target = sets["W+A-"]
    blocks = []
    for true_c, pred_c, count in top_pairs:
        sub = target[(target["true_label"] == true_c) & (target["dacvae_pred"] == pred_c)]
        sub = sub.nlargest(examples_per_pair, "whisper_small_proba")
        rows = []
        for _, r in sub.iterrows():
            url = audio_relpath(r["audio_path"])
            rows.append(
                f'<div style="margin:4px 0;">'
                f"<small>{r['utt_id']} | Whisper: {r['whisper_small_pred']} "
                f"(p={r['whisper_small_proba']:.3f}) | DAC: {r['dacvae_pred']} "
                f"(p={r['dacvae_proba']:.3f})</small><br>"
                f'<audio controls preload="none" style="width:280px;height:30px;">'
                f'<source src="{url}"></audio></div>'
            )
        blocks.append(
            f'<div style="margin:12px 0;padding:8px;border-left:4px solid #ef4444;'
            f'background:#fef2f2;">'
            f"<b>true: {true_c} → DAC predicted: {pred_c}</b> ({count} samples in W+A-, "
            f"top {min(count, examples_per_pair)} by Whisper confidence)<br>" + "".join(rows) + "</div>"
        )
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>Phase A — Confused Pair Examples (W+A-)</title>"
        "<style>body{font-family:-apple-system,sans-serif;max-width:1200px;"
        "margin:20px auto;padding:0 20px;}audio{vertical-align:middle;}</style>"
        "</head><body><h1>Phase A — Confused pair examples (W+A- only)</h1>"
        "<p>Audio players use relative paths via known dataset symlinks. "
        "Run <code>python -m http.server</code> from the project root to serve.</p>"
        + "".join(blocks)
        + "</body></html>"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined_csv", type=Path, required=True)
    p.add_argument("--meta_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--top_pairs_k", type=int, default=3)
    p.add_argument("--examples_per_pair", type=int, default=8)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = args.out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.combined_csv)
    sets = build_sample_sets(df)
    classes = sorted(df["true_label"].unique())

    to_long_csv(sets, args.out_dir / "sample_sets.csv")
    set_sizes_df = set_sizes(sets)
    set_sizes_df.to_csv(args.out_dir / "set_sizes.csv", index=False)

    cm_wam = dac_confusion_on_set(sets["W+A-"], classes)
    cm_wam.to_csv(args.out_dir / "dac_confusion_W+A-.csv")

    proba_summary = proba_distribution_summary(sets, "whisper_small_proba")
    proba_summary.to_csv(args.out_dir / "whisper_proba_distribution.csv", index=False)

    top_pairs = top_confused_pairs(cm_wam, k=args.top_pairs_k)
    pd.DataFrame(top_pairs, columns=["true_label", "dacvae_pred", "count"]).to_csv(
        args.out_dir / "top_confused_pairs.csv", index=False
    )

    joined = join_sets_with_meta(args.out_dir / "sample_sets.csv", args.meta_csv)
    parts = [meta_distribution(joined, a).assign(attribute=a) for a in ("source", "velocity", "pitch_bin")]
    pd.concat(parts, ignore_index=True).to_csv(args.out_dir / "meta_distribution.csv", index=False)
    chi2_results = {a: chi_square_test(joined, a, "W+A-") for a in ("source", "velocity", "pitch_bin")}
    pd.DataFrame([{"attribute": a, "target_set": "W+A-", **chi2_results[a]} for a in chi2_results]).to_csv(
        args.out_dir / "chi_square_results.csv", index=False
    )

    make_set_sizes_fig(set_sizes_df, figs_dir / "F-A1_set_sizes.png")
    make_dac_confusion_fig(cm_wam, figs_dir / "F-A2_dac_confusion.png")
    make_meta_distribution_fig(joined, chi2_results, figs_dir / "F-A3_meta_distribution.png")
    make_proba_violin_fig(sets, figs_dir / "F-A4_whisper_proba.png")

    make_html_viewer(sets, top_pairs, args.examples_per_pair, args.out_dir / "confused_pair_examples.html")
    logger.info("phase A run complete → %s", args.out_dir)


if __name__ == "__main__":
    main()
