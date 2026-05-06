"""
Combine multiple encoder predictions into a single HTML viewer.
Shows for each audio: true_label, [enc1_pred, enc2_pred, ...], audio player.

Inputs: existing per-encoder prediction CSVs from error_analysis.py
Output: combined CSV + HTML

Usage:
  python combine_predictions_html.py --encoders whisper_small dacvae \
      --dataset nsynth_train_30k
"""

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
PRED_DIR = ROOT / "_results/predictions"


def _relpath(audio_path: str) -> str:
    p = audio_path
    if p.startswith("/mnt/tmp/datasets/music/"):
        return "nsynth_audio_root/" + p[len("/mnt/tmp/datasets/music/"):]
    if p.startswith("/mnt/tmp/datasets/"):
        return "audio_root/" + p[len("/mnt/tmp/datasets/"):]
    return "file://" + p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--max_examples_per_section", type=int, default=8)
    args = p.parse_args()

    # Load each encoder's predictions
    dfs = {}
    for enc in args.encoders:
        csvp = PRED_DIR / f"{enc}_{args.dataset}.csv"
        if not csvp.exists():
            raise FileNotFoundError(f"Missing: {csvp}. Run error_analysis.py first.")
        df = pd.read_csv(csvp)
        dfs[enc] = df.set_index("utt_id")
        print(f"  {enc}: {len(df)} predictions, acc={df.correct.mean():.4f}")

    # Combine on utt_id
    base = dfs[args.encoders[0]][["audio_path", "true_label", "fold"]].copy()
    for enc in args.encoders:
        d = dfs[enc]
        base[f"{enc}_pred"] = d["pred_label"]
        base[f"{enc}_proba"] = d["pred_proba"]
        base[f"{enc}_correct"] = d["correct"]

    combined = base.reset_index()
    out_csv = PRED_DIR / f"combined_{'_'.join(args.encoders)}_{args.dataset}.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nCombined CSV: {out_csv}")

    # ── Sections ───────────────────────────────────────────────────────────────
    # 1. ALL correct (any sample where all encoders correct)
    correct_cols = [f"{e}_correct" for e in args.encoders]
    all_correct = combined[combined[correct_cols].all(axis=1)]
    all_wrong = combined[(~combined[correct_cols]).all(axis=1)] if len(args.encoders) > 1 else pd.DataFrame()
    # 2. Only encoder X correct (others wrong)
    only_correct = {}
    for enc in args.encoders:
        cond = combined[f"{enc}_correct"]
        for other in args.encoders:
            if other != enc:
                cond = cond & (~combined[f"{other}_correct"])
        only_correct[enc] = combined[cond]

    print(f"\nSection sizes:")
    print(f"  All {len(args.encoders)} correct: {len(all_correct)}")
    if len(args.encoders) > 1:
        print(f"  All wrong: {len(all_wrong)}")
        for e in args.encoders:
            print(f"  Only {e} correct: {len(only_correct[e])}")

    # ── HTML ──────────────────────────────────────────────────────────────────
    html = _build_html(combined, args.encoders, args.dataset, args.max_examples_per_section,
                      all_correct, all_wrong, only_correct)
    out_html = PRED_DIR / f"combined_{'_'.join(args.encoders)}_{args.dataset}.html"
    out_html.write_text(html)
    print(f"HTML: {out_html}")


def _audio_row(r, encoders):
    parts = [f'<small>{r["utt_id"]}</small>']
    parts.append(f'<b>True: {r["true_label"]}</b>')
    for e in encoders:
        ok = r[f"{e}_correct"]
        color = "#22c55e" if ok else "#ef4444"
        sym = "✓" if ok else "✗"
        parts.append(f'<span style="color:{color};">{sym} {e}: <b>{r[f"{e}_pred"]}</b> ({r[f"{e}_proba"]:.2f})</span>')
    parts.append(
        f'<audio controls style="width:280px; height:30px;" preload="none">'
        f'<source src="{_relpath(r["audio_path"])}"></audio>'
    )
    return '<div style="margin:6px 0; padding:6px; border-bottom:1px solid #e5e7eb;">' + \
           ' &middot; '.join(parts) + '</div>'


def _build_html(combined, encoders, dataset, max_per_section,
                all_correct, all_wrong, only_correct):
    classes = sorted(combined["true_label"].unique())

    # Per-encoder accuracy summary
    acc_lines = []
    for e in encoders:
        acc = combined[f"{e}_correct"].mean()
        acc_lines.append(f"<li><b>{e}</b>: {acc:.4f}</li>")

    # Overall agreement
    if len(encoders) >= 2:
        agree = (combined[f"{encoders[0]}_pred"] == combined[f"{encoders[1]}_pred"]).mean()
        agreement_html = f"<p>Agreement between {encoders[0]} and {encoders[1]}: {agree:.4f}</p>"
    else:
        agreement_html = ""

    # Section: only-encoder-X-correct (most informative for comparing encoders)
    only_blocks = []
    for e in encoders:
        sub = only_correct[e].copy()
        if len(sub) == 0:
            continue
        # Sort by encoder confidence (highest first)
        sub = sub.sort_values(f"{e}_proba", ascending=False)
        # Group by true_label → pred of OTHER encoder for context
        # For brevity just show top samples overall, but break down per true class
        per_class_blocks = []
        for tl in classes:
            cls_sub = sub[sub.true_label == tl].head(max_per_section)
            if len(cls_sub) == 0:
                continue
            n_total = len(sub[sub.true_label == tl])
            rows_html = ''.join(_audio_row(r, encoders) for _, r in cls_sub.iterrows())
            per_class_blocks.append(
                f'<div style="margin:8px 0; padding:6px; background:#f9fafb;">'
                f'<b>True={tl}</b> ({n_total} where only {e} got right; top {min(n_total, max_per_section)} by {e} confidence)'
                + rows_html + '</div>')
        if per_class_blocks:
            only_blocks.append(
                f'<h3 style="background:#dbeafe; padding:6px;">'
                f'Only {e} correct ({len(sub)} samples)'
                f'</h3>' + ''.join(per_class_blocks))

    # Section: all wrong (per true class)
    all_wrong_blocks = []
    if len(all_wrong) > 0:
        for tl in classes:
            cls_sub = all_wrong[all_wrong.true_label == tl].head(max_per_section)
            if len(cls_sub) == 0:
                continue
            n_total = len(all_wrong[all_wrong.true_label == tl])
            rows_html = ''.join(_audio_row(r, encoders) for _, r in cls_sub.iterrows())
            all_wrong_blocks.append(
                f'<div style="margin:8px 0; padding:6px; background:#fef2f2;">'
                f'<b>True={tl}</b> (all encoders wrong, {n_total} samples; top {min(n_total, max_per_section)})'
                + rows_html + '</div>')

    # Section: all correct (just a few examples per class for sanity)
    all_correct_blocks = []
    if len(all_correct) > 0:
        for tl in classes:
            cls_sub = all_correct[all_correct.true_label == tl].head(3)
            if len(cls_sub) == 0:
                continue
            rows_html = ''.join(_audio_row(r, encoders) for _, r in cls_sub.iterrows())
            all_correct_blocks.append(
                f'<div style="margin:6px 0; padding:6px; background:#f0fdf4;">'
                f'<b>True={tl}</b> (all encoders correct; 3 examples)' + rows_html + '</div>')

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{dataset} — Combined: {' vs '.join(encoders)}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width:1200px; margin:20px auto; padding:0 20px; }}
h1, h2, h3 {{ margin-top: 1.5em; }}
audio {{ vertical-align: middle; }}
.summary {{ background:#eff6ff; padding:12px; border-radius:8px; margin: 12px 0; }}
.toc a {{ display:inline-block; margin-right:12px; padding:4px 8px; background:#e5e7eb; border-radius:4px; text-decoration:none; color:#111; }}
</style>
</head>
<body>
<h1>{dataset} — comparison</h1>
<div class="summary">
  <h3>5-fold CV accuracy</h3>
  <ul>{''.join(acc_lines)}</ul>
  {agreement_html}
  <p>Total samples: {len(combined)}</p>
</div>

<div class="toc">
<a href="#only">Only-X-correct</a>
<a href="#allwrong">All-wrong</a>
<a href="#allcorrect">All-correct</a>
</div>

<h2 id="only">Only one encoder correct (most informative)</h2>
{''.join(only_blocks) if only_blocks else '<p>None</p>'}

<h2 id="allwrong">All encoders wrong</h2>
{''.join(all_wrong_blocks) if all_wrong_blocks else '<p>None</p>'}

<h2 id="allcorrect">All encoders correct (a few per class)</h2>
{''.join(all_correct_blocks) if all_correct_blocks else '<p>None</p>'}

</body>
</html>"""
    return html


if __name__ == "__main__":
    main()
