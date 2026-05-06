"""
Non-linear probes (MLP + kNN) over pre-extracted audio-encoder embeddings.

Tests "info-present-but-entangled" hypothesis: if reconstruction encoder gap
narrows under MLP/kNN vs linear probe, then info IS preserved but in a
non-linearly-accessible form.

Output:
  _results/probe_nonlinear_results.csv
  columns: encoder, dataset, fold, method, hyper, accuracy, macro_f1, weighted_f1
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).parent
EMB_ROOT = ROOT / "embeds"
MANIFEST_DIR = ROOT / "manifests"
RESULTS_CSV = ROOT / "_results/probe_nonlinear_results.csv"

LABEL_COL = {"iemocap_4class": "emotion", "ravdess": "emotion",
             "cremad": "emotion", "esc50": "label"}
FOLD_COL = {"iemocap_4class": "session", "ravdess": "fold",
            "cremad": "fold", "esc50": "fold"}


def load_split(encoder, dataset):
    df = pd.read_csv(MANIFEST_DIR / f"{dataset}.csv")
    emb_dir = EMB_ROOT / encoder / dataset
    X, y, fold = [], [], []
    for row in df.to_dict("records"):
        npy = emb_dir / f"{row['utt_id']}.npy"
        if not npy.exists():
            continue
        X.append(np.load(str(npy)))
        y.append(row[LABEL_COL[dataset]])
        fold.append(row[FOLD_COL[dataset]])
    return np.stack(X).astype(np.float32), np.array(y), np.array(fold)


def _encode(ytr, yte):
    """LabelEncoder fit on train, transform both — avoids sklearn MLP isnan bug."""
    le = LabelEncoder()
    ytr_i = le.fit_transform(ytr)
    yte_i = le.transform(yte)
    return ytr_i, yte_i


def fit_mlp(Xtr, ytr, Xte, yte, hidden=(512,), alpha=1e-4, seed=42):
    """Adam, no early-stopping. hidden can be tuple for deep MLPs."""
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    ytr_i, yte_i = _encode(ytr, yte)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=1e-3,
        batch_size=128,
        max_iter=500,
        n_iter_no_change=30,
        tol=1e-5,
        random_state=seed,
        early_stopping=False,
    )
    clf.fit(Xtr, ytr_i)
    pred = clf.predict(Xte)
    return {
        "accuracy": accuracy_score(yte_i, pred),
        "macro_f1": f1_score(yte_i, pred, average="macro"),
        "weighted_f1": f1_score(yte_i, pred, average="weighted"),
        "n_iter": int(clf.n_iter_),
    }


def fit_knn(Xtr, ytr, Xte, yte, k=11, metric="cosine"):
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    ytr_i, yte_i = _encode(ytr, yte)
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights="distance", n_jobs=-1)
    clf.fit(Xtr, ytr_i)
    pred = clf.predict(Xte)
    return {
        "accuracy": accuracy_score(yte_i, pred),
        "macro_f1": f1_score(yte_i, pred, average="macro"),
        "weighted_f1": f1_score(yte_i, pred, average="weighted"),
    }


ARCH_MAP = {"h512": (512,), "h64x3": (64, 64, 64)}


def run(encoders, datasets, methods=("mlp", "knn"), archs=("h512", "h64x3"), append=False):
    rows = []
    for enc in encoders:
        for ds in datasets:
            try:
                X, y, fold = load_split(enc, ds)
            except Exception as e:
                logger.warning(f"{enc} × {ds}: load failed: {e}")
                continue
            unique_folds = sorted(set(fold.tolist()), key=str)
            logger.info(f"\n=== {enc} × {ds} === N={len(X)}, d={X.shape[1]}, folds={unique_folds}")

            for f in unique_folds:
                test_mask = fold == f
                Xtr, ytr = X[~test_mask], y[~test_mask]
                Xte, yte = X[test_mask], y[test_mask]

                if "mlp" in methods:
                    for tag in archs:
                        hidden = ARCH_MAP[tag]
                        for alpha in (1e-5, 1e-4, 1e-3, 1e-2):
                            m = fit_mlp(Xtr, ytr, Xte, yte, hidden=hidden, alpha=alpha)
                            rows.append({
                                "encoder": enc, "dataset": ds, "fold": f,
                                "method": "mlp", "hyper": f"{tag}_a{alpha:.0e}",
                                **m,
                            })
                            logger.info(f"  fold={f} mlp({tag},α={alpha:.0e}): acc={m['accuracy']:.4f} iter={m['n_iter']}")

                if "knn" in methods:
                    for k in (11,):
                        m = fit_knn(Xtr, ytr, Xte, yte, k=k)
                        rows.append({
                            "encoder": enc, "dataset": ds, "fold": f,
                            "method": "knn", "hyper": f"k{k}",
                            **m,
                        })
                        logger.info(f"  fold={f} knn(k={k}): acc={m['accuracy']:.4f}")

    df = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if append and RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        # drop matching (encoder, dataset, method, hyper) rows
        keys = set(zip(df.encoder, df.dataset, df.method, df.hyper))
        existing = existing[~existing.apply(
            lambda r: (r.encoder, r.dataset, r.method, r.hyper) in keys, axis=1
        )]
        df = pd.concat([existing, df], ignore_index=True)
        logger.info(f"Appended; total rows now {len(df)}")
    df.to_csv(RESULTS_CSV, index=False)
    logger.info(f"\nWrote {len(df)} rows to {RESULTS_CSV}")

    # Summary
    print("\n=== Summary (5-fold mean accuracy per encoder × dataset × method) ===")
    for (enc, ds, m), g in df.groupby(["encoder", "dataset", "method"]):
        print(f"  {enc:18s} × {ds:18s} × {m:5s}: acc={g.accuracy.mean():.4f} ± {g.accuracy.std():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+",
                   default=["mel_only", "whisper_tiny", "whisper_small",
                            "wavtok_40_unify", "encodec_24k", "dacvae"])
    p.add_argument("--datasets", nargs="+",
                   default=["iemocap_4class", "ravdess", "cremad", "esc50"])
    p.add_argument("--methods", nargs="+", default=["mlp", "knn"],
                   choices=["mlp", "knn"])
    p.add_argument("--archs", nargs="+", default=["h512", "h64x3"],
                   choices=list(ARCH_MAP.keys()))
    p.add_argument("--append", action="store_true")
    args = p.parse_args()
    run(args.encoders, args.datasets, methods=tuple(args.methods),
        archs=tuple(args.archs), append=args.append)


if __name__ == "__main__":
    main()
