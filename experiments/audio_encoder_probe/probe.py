"""
Linear probe over pre-extracted audio-encoder embeddings.

For each (encoder × dataset × C × fold):
  - Load mean-pooled embeddings
  - Logistic regression with L2 (specified C)
  - Train on K-1 folds, eval on held-out fold
  - Standard scaler optional (default on)

Datasets:
  iemocap_4class : 5-fold leave-session-out (sessions Ses01..Ses05)
  esc50          : 5-fold official CV (fold column 1..5)

Output:
  _results/probe_results.csv
  columns: encoder, dataset, fold, C, scaler, n_train, n_test, accuracy, macro_f1, weighted_f1

Usage:
  python probe.py [--encoders whisper_small dacvae] [--datasets iemocap_4class esc50] [--C_grid 0.01 0.1 1 10]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).parent
EMB_ROOT = ROOT / "embeds"
MANIFEST_DIR = ROOT / "manifests"
RESULTS_CSV = ROOT / "_results/probe_results.csv"


def load_split(encoder: str, dataset: str):
    """Load X (N,d), y (N,), fold (N,) for one encoder×dataset."""
    df = pd.read_csv(MANIFEST_DIR / f"{dataset}.csv")
    emb_dir = EMB_ROOT / encoder / dataset

    X, y, fold, kept_ids = [], [], [], []
    LABEL_COL = {
        "iemocap_4class": "emotion",
        "ravdess": "emotion",
        "cremad": "emotion",
        "esc50": "label",
        "gtzan": "label",
        "nsynth_train_30k": "label",
        "nsynth_test": "label",
        "medley_solos": "label",
    }
    FOLD_COL = {
        "iemocap_4class": "session",
        "ravdess": "fold",
        "cremad": "fold",
        "esc50": "fold",
        "gtzan": "fold",
        "nsynth_train_30k": "fold",
        "nsynth_test": "fold",
        "medley_solos": "fold",
    }
    label_col = LABEL_COL[dataset]
    fold_col = FOLD_COL[dataset]

    for row in df.to_dict("records"):
        npy = emb_dir / f"{row['utt_id']}.npy"
        if not npy.exists():
            continue
        X.append(np.load(str(npy)))
        y.append(row[label_col])
        fold.append(row[fold_col])
        kept_ids.append(row["utt_id"])

    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    fold = np.array(fold)
    return X, y, fold, kept_ids


def probe_fold(X_train, y_train, X_test, y_test, C: float, scaler: bool = True,
               max_iter: int = 5000, seed: int = 42):
    if scaler:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    clf = LogisticRegression(
        C=C, penalty="l2", solver="lbfgs",
        max_iter=max_iter, n_jobs=-1, random_state=seed,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
    }


def run(encoders, datasets, C_grid, use_scaler=True, append=False):
    rows = []
    for enc in encoders:
        for ds in datasets:
            try:
                X, y, fold, _ = load_split(enc, ds)
            except Exception as e:
                logger.warning(f"{enc} × {ds}: load failed: {e}")
                continue

            unique_folds = sorted(set(fold.tolist()))
            logger.info(f"\n=== {enc} × {ds} ===  N={len(X)}, d={X.shape[1]}, "
                        f"classes={len(set(y))}, folds={unique_folds}")

            for f in unique_folds:
                test_mask = fold == f
                Xtr, ytr = X[~test_mask], y[~test_mask]
                Xte, yte = X[test_mask], y[test_mask]

                for C in C_grid:
                    metrics = probe_fold(Xtr, ytr, Xte, yte, C=C, scaler=use_scaler)
                    rows.append({
                        "encoder": enc, "dataset": ds, "fold": f, "C": C,
                        "scaler": use_scaler,
                        "n_train": len(ytr), "n_test": len(yte),
                        **metrics,
                    })
                    logger.info(f"  fold={f} C={C}: acc={metrics['accuracy']:.4f} "
                                f"macro_f1={metrics['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if append and RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        # Drop rows for encoders/datasets being re-run to avoid duplicates
        keys = set(zip(df.encoder, df.dataset))
        existing = existing[~existing.apply(
            lambda r: (r.encoder, r.dataset) in keys, axis=1
        )]
        df = pd.concat([existing, df], ignore_index=True)
        logger.info(f"Appended; total rows now {len(df)}")
    df.to_csv(RESULTS_CSV, index=False)
    logger.info(f"\nWrote {len(df)} rows to {RESULTS_CSV}")

    # Summary: encoder × dataset, mean over folds at best C
    print("\n=== Summary (mean over folds, best C per encoder×dataset) ===")
    for (enc, ds), g in df.groupby(["encoder", "dataset"]):
        best_C = g.groupby("C").accuracy.mean().idxmax()
        sub = g[g.C == best_C]
        print(f"{enc} × {ds}  [best C={best_C}]: "
              f"acc = {sub.accuracy.mean():.4f} ± {sub.accuracy.std():.4f}  |  "
              f"macro_f1 = {sub.macro_f1.mean():.4f} ± {sub.macro_f1.std():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=["whisper_small", "dacvae"])
    p.add_argument("--datasets", nargs="+", default=["iemocap_4class", "esc50"])
    p.add_argument("--C_grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    p.add_argument("--no_scaler", action="store_true")
    p.add_argument("--append", action="store_true",
                   help="Append to existing probe_results.csv (replaces matching encoder×dataset)")
    args = p.parse_args()
    run(args.encoders, args.datasets, args.C_grid,
        use_scaler=not args.no_scaler, append=args.append)


if __name__ == "__main__":
    main()
