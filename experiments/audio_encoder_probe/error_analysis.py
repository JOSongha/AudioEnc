"""
Per-sample prediction CSV + HTML audio viewer for one encoder × dataset.

Trains linear probe with given C, runs 5-fold CV to get predictions for every
sample (each held-out exactly once), then:
  - saves CSV: utt_id, audio_path, true_label, pred_label, correct
  - generates HTML viewer with audio players for misclassified samples

Usage:
  python error_analysis.py --encoder whisper_small --dataset nsynth_train_30k --C 0.1
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).parent
EMB_ROOT = ROOT / "embeds"
MANIFEST_DIR = ROOT / "manifests"
OUT_DIR = ROOT / "_results/predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = {"iemocap_4class": "emotion", "ravdess": "emotion",
             "cremad": "emotion", "esc50": "label",
             "gtzan": "label", "nsynth_train_30k": "label", "nsynth_test": "label"}
FOLD_COL = {"iemocap_4class": "session", "ravdess": "fold",
            "cremad": "fold", "esc50": "fold",
            "gtzan": "fold", "nsynth_train_30k": "fold", "nsynth_test": "fold"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--C", type=float, default=0.1)
    p.add_argument("--max_examples_per_class", type=int, default=10,
                   help="Per (true,pred) cell: max audio examples in HTML")
    args = p.parse_args()

    enc, ds = args.encoder, args.dataset
    df = pd.read_csv(MANIFEST_DIR / f"{ds}.csv")
    label_col, fold_col = LABEL_COL[ds], FOLD_COL[ds]
    classes = sorted(df[label_col].unique())
    cls2idx = {c: i for i, c in enumerate(classes)}
    df["idx"] = df[label_col].map(cls2idx)

    emb_dir = EMB_ROOT / enc / ds
    X = []
    keep = []
    for row in df.to_dict("records"):
        npy = emb_dir / f"{row['utt_id']}.npy"
        if not npy.exists():
            continue
        X.append(np.load(str(npy)))
        keep.append(row)
    X = np.stack(X).astype(np.float32)
    df_keep = pd.DataFrame(keep).reset_index(drop=True)
    y = df_keep["idx"].values
    fold = df_keep[fold_col].values
    logger.info(f"Loaded {len(X)} samples, dim={X.shape[1]}, classes={len(classes)}")

    # 5-fold CV: predict each held-out fold
    pred_idx = np.full(len(X), -1, dtype=np.int64)
    proba_max = np.zeros(len(X), dtype=np.float32)
    for f in sorted(set(fold.tolist())):
        tr = fold != f; te = fold == f
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(C=args.C, max_iter=5000, random_state=42).fit(Xtr, y[tr])
        pred_idx[te] = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        proba_max[te] = proba.max(axis=1)
        logger.info(f"  fold {f}: train={tr.sum()} test={te.sum()} acc={accuracy_score(y[te], pred_idx[te]):.4f}")

    overall = accuracy_score(y, pred_idx)
    logger.info(f"\nOverall 5-fold CV acc: {overall:.4f}")

    # Save per-sample CSV
    df_keep["true_label"] = [classes[i] for i in y]
    df_keep["pred_label"] = [classes[i] for i in pred_idx]
    df_keep["pred_proba"] = proba_max
    df_keep["correct"] = (y == pred_idx)
    df_keep["fold"] = fold
    out_csv = OUT_DIR / f"{enc}_{ds}.csv"
    df_keep[["utt_id","audio_path","true_label","pred_label","pred_proba","correct","fold"]].to_csv(out_csv, index=False)
    logger.info(f"Wrote per-sample predictions to {out_csv}")

    # Confusion matrix
    cm = confusion_matrix(y, pred_idx, labels=range(len(classes)))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    out_cm = OUT_DIR / f"{enc}_{ds}_confusion.csv"
    cm_df.to_csv(out_cm)
    logger.info(f"Wrote confusion matrix to {out_cm}")

    # ── HTML viewer ────────────────────────────────────────────────────────────
    html = _build_html(df_keep, classes, cm, overall, args.max_examples_per_class, enc, ds)
    out_html = OUT_DIR / f"{enc}_{ds}.html"
    out_html.write_text(html)
    logger.info(f"Wrote HTML viewer to {out_html}")
    logger.info(f"To view: cd {OUT_DIR.parent.parent}/.. && python -m http.server 8765 → open http://localhost:8765/{out_html.relative_to(ROOT.parent.parent)}")


def _relpath(audio_path: str) -> str:
    """Convert absolute audio_path to a path relative to OUT_DIR (for HTTP serving).
    Maps /mnt/tmp/datasets/music/... to nsynth_audio_root/... via symlink."""
    p = audio_path
    # nsynth + future music dataset roots live under /mnt/tmp/datasets/music/
    if p.startswith("/mnt/tmp/datasets/music/"):
        return "nsynth_audio_root/" + p[len("/mnt/tmp/datasets/music/"):]
    if p.startswith("/mnt/tmp/datasets/"):
        return "audio_root/" + p[len("/mnt/tmp/datasets/"):]
    if p.startswith("/mnt/ddn/kyudan/IEMOCAP/"):
        return "iemocap_audio_root/" + p[len("/mnt/ddn/kyudan/IEMOCAP/"):]
    # fallback: file:// URL (browser may block)
    return "file://" + p


def _build_html(df, classes, cm, overall_acc, max_per_cell, enc, ds):
    """Generate self-contained HTML with audio players + confusion matrix.

    NOTE: requires symlinks at OUT_DIR pointing to dataset roots:
      OUT_DIR/nsynth_audio_root → /mnt/tmp/datasets/music
      OUT_DIR/audio_root → /mnt/tmp/datasets
      OUT_DIR/iemocap_audio_root → /mnt/ddn/kyudan/IEMOCAP
    Run HTTP server from OUT_DIR (or any parent) to view.
    """
    # Confusion matrix HTML
    cm_rows = []
    for i, true_c in enumerate(classes):
        row_total = cm[i].sum()
        cells = []
        for j, pred_c in enumerate(classes):
            v = cm[i, j]
            pct = (v / row_total * 100) if row_total else 0
            color = "#fef2f2" if i != j and v > 0 else ("#dcfce7" if i == j else "#ffffff")
            if i == j:
                cell_html = f'<td style="background:{color}; text-align:center;"><b>{v}</b><br><small>{pct:.1f}%</small></td>'
            else:
                cell_html = f'<td style="background:{color}; text-align:center;">{v}<br><small>{pct:.1f}%</small></td>'
            cells.append(cell_html)
        cm_rows.append(f"<tr><th>{true_c}</th>{''.join(cells)}</tr>")

    cm_table = (
        '<table border="1" style="border-collapse:collapse; font-size:12px;">'
        f'<tr><th></th>' + ''.join(f'<th>{c}</th>' for c in classes) + '</tr>'
        + ''.join(cm_rows) + '</table>'
    )

    # Per-class error groups (true → pred)
    err = df[df.correct == False].copy()
    correct = df[df.correct == True].copy()

    error_blocks = []
    for true_c in classes:
        sub = err[err.true_label == true_c]
        if sub.empty:
            continue
        pred_groups = sub.groupby("pred_label")
        for pred_c, gp in sorted(pred_groups, key=lambda x: -len(x[1])):
            n = len(gp)
            if n == 0:
                continue
            samples = gp.nlargest(max_per_cell, "pred_proba")  # most confident wrongs
            audio_rows = []
            for _, r in samples.iterrows():
                path = r.audio_path
                proba = r.pred_proba
                audio_rows.append(
                    f'<div style="margin:4px 0;">'
                    f'<small>{r.utt_id} (proba={proba:.3f})</small><br>'
                    f'<audio controls style="width:280px; height:30px;" preload="none"><source src="{_relpath(r.audio_path)}"></audio>'
                    f'</div>'
                )
            error_blocks.append(
                f'<div style="margin:12px 0; padding:8px; border-left:4px solid #ef4444; background:#fef2f2;">'
                f'<b>True: {true_c} → Predicted: {pred_c}</b> ({n} samples, top {min(n, max_per_cell)} shown by confidence)<br>'
                + ''.join(audio_rows) +
                '</div>'
            )

    # Per-class correct examples (a few each)
    correct_blocks = []
    for true_c in classes:
        sub = correct[correct.true_label == true_c]
        if sub.empty:
            continue
        samples = sub.nlargest(5, "pred_proba")
        audio_rows = []
        for _, r in samples.iterrows():
            audio_rows.append(
                f'<div style="margin:4px 0;">'
                f'<small>{r.utt_id} (proba={r.pred_proba:.3f})</small><br>'
                f'<audio controls style="width:280px; height:30px;" preload="none"><source src="{_relpath(r.audio_path)}"></audio>'
                f'</div>'
            )
        correct_blocks.append(
            f'<div style="margin:12px 0; padding:8px; border-left:4px solid #22c55e; background:#f0fdf4;">'
            f'<b>{true_c} ✓ (n={len(sub)} correct)</b><br>'
            + ''.join(audio_rows) +
            '</div>'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{enc} × {ds} — Error Analysis</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width:1200px; margin:20px auto; padding:0 20px; }}
h1, h2, h3 {{ margin-top: 1.5em; }}
table {{ border-collapse: collapse; }}
th, td {{ padding: 4px 8px; border: 1px solid #ccc; }}
audio {{ vertical-align: middle; }}
.summary {{ background:#eff6ff; padding:12px; border-radius:8px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>{enc} × {ds}</h1>
<div class="summary">
  <b>Overall 5-fold CV accuracy: {overall_acc:.4f}</b> ({len(df)} samples, {len(classes)} classes)
</div>

<h2>Confusion matrix (rows=true, cols=pred)</h2>
{cm_table}

<h2>Misclassified samples (per true → pred cell, top {max_per_cell} by confidence)</h2>
<p>⚠️ Audio paths use <code>file://</code> URLs. Browser may block — see CSV for paths.</p>
{''.join(error_blocks) if error_blocks else '<p>No misclassifications.</p>'}

<h2>Correct samples (per class, top 5 by confidence)</h2>
{''.join(correct_blocks)}

</body>
</html>"""
    return html


if __name__ == "__main__":
    main()
