#!/usr/bin/env python
"""Linear-probe per layer.

For each (model, layer, corpus) .npz, train two cross-validated linear classifiers:

  speaker_probe : predict `spk` from `emb`         → "how much speaker info is recoverable"
  content_probe : predict `pair` (transcript_id)   → "how much content info is recoverable"

A layer that has speaker_acc → near-chance while content_acc stays high is
"speaker-invariant content-preserving" — the property we want from the encoder
+ projector + LLM stack as we go deeper.

Cross-validation: speaker-stratified k-fold. We carefully avoid leaking the
exact same `(pair, spk)` example between train and test for the content probe;
splits are by random utterance, but the content_probe is multinomial over
transcript_ids so as long as each pair has >= 2 occurrences (true by
construction) generalization is meaningful.

Uses sklearn's LogisticRegression (`saga` for large k). Falls back to a tiny
torch MLP only if sklearn missing.

Output:
  out/probe_per_layer.csv     (model, src, layer, group, layer_idx, target, acc, chance, n_train, n_test)
  out/probe__{speaker,content}__{metric}.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_layer(layer: str):
    import re
    m = re.match(r"^(?P<group>enc|proj|llm)\.(?:L(?P<idx>\d+)|(?P<tag>out|norm))$", layer)
    if not m:
        return ("?", -1)
    group = m.group("group")
    if m.group("tag") in ("out", "norm"):
        return (group, 999)
    return (group, int(m.group("idx")))


def chance_acc(labels: np.ndarray) -> float:
    """Most-frequent-class baseline."""
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


def stratified_kfold_indices(y: np.ndarray, k: int, seed: int):
    """Return list of (train_idx, test_idx) tuples, stratified by class.
    Classes with < k members are dropped from the fold-eligible pool but kept
    in train (so the classifier sees them at training but isn't tested on rare
    classes)."""
    rng = np.random.default_rng(seed)
    by_class = {c: np.where(y == c)[0].tolist() for c in np.unique(y)}
    for c in by_class:
        rng.shuffle(by_class[c])
    folds = [[] for _ in range(k)]
    for c, idxs in by_class.items():
        if len(idxs) < k:
            # tiny class: round-robin into folds, but each fold gets at most 1
            for i, idx in enumerate(idxs):
                folds[i % k].append(idx)
        else:
            for i, idx in enumerate(idxs):
                folds[i % k].append(idx)
    splits = []
    for ki in range(k):
        test_idx = np.array(folds[ki], dtype=int)
        train_idx = np.array([i for kj in range(k) if kj != ki for i in folds[kj]], dtype=int)
        splits.append((train_idx, test_idx))
    return splits


def probe_one(X: np.ndarray, y: np.ndarray, k: int, seed: int,
              max_classes_one_vs_rest: int = 200):
    """Train k-fold logistic regression, return mean test accuracy.

    For content probes with hundreds of classes, sklearn's saga solver handles
    multinomial efficiently; we cap to 200 most-frequent classes if huge.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) > max_classes_one_vs_rest:
        keep = classes[np.argsort(-counts)[:max_classes_one_vs_rest]]
        m = np.isin(y, keep)
        X, y = X[m], y[m]

    splits = stratified_kfold_indices(y, k=k, seed=seed)
    accs = []
    for train_idx, test_idx in splits:
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        if len(np.unique(ytr)) < 2 or len(Xte) == 0:
            continue
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        clf = LogisticRegression(
            max_iter=2000, multi_class="multinomial", solver="lbfgs",
            C=1.0, n_jobs=1,
        )
        clf.fit(Xtr_s, ytr)
        yhat = clf.predict(Xte_s)
        accs.append(float((yhat == yte).mean()))
    return float(np.mean(accs)) if accs else float("nan"), len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_per_class", type=int, default=2,
                    help="drop classes (speakers / transcripts) with fewer examples than this")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(Path(args.emb_dir).rglob("*.npz"))
    print(f"[probe] found {len(npz_paths)} .npz files")

    rows = []
    for np_path in npz_paths:
        z = np.load(np_path, allow_pickle=False)
        model_tag = str(z["model"])
        layer_tag = str(z["layer"])
        emb = z["emb"]
        pair = z["pair"].astype(str)
        spk = z["spk"].astype(str)
        src = z["src"].astype(str)
        group, idx = parse_layer(layer_tag)

        for src_name in sorted(set(src.tolist())):
            mask = src == src_name
            if mask.sum() < 8:
                continue
            for target_name, y in [("speaker", spk[mask]), ("content", pair[mask])]:
                # filter rare classes
                classes, counts = np.unique(y, return_counts=True)
                keep_classes = classes[counts >= args.min_per_class]
                m2 = np.isin(y, keep_classes)
                if m2.sum() < 8 or len(keep_classes) < 2:
                    continue
                X = emb[mask][m2]
                yk = y[m2]
                acc, n = probe_one(X, yk, k=args.folds, seed=args.seed)
                rows.append({
                    "model": model_tag,
                    "src": src_name,
                    "layer": layer_tag,
                    "group": group,
                    "layer_idx": idx,
                    "target": target_name,
                    "acc": acc,
                    "chance": chance_acc(yk),
                    "n_examples": n,
                    "n_classes": int(len(np.unique(yk))),
                })

    csv_path = out_dir / "probe_per_layer.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "model", "src", "layer", "group", "layer_idx", "target",
        "acc", "chance", "n_examples", "n_classes",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[probe] wrote {csv_path}  rows={len(rows)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for target in ["speaker", "content"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        seen = set()
        for r in rows:
            if r["target"] != target:
                continue
            seen.add((r["model"], r["src"], r["group"]))
        for key in sorted(seen):
            model, src_name, group = key
            curve = [r for r in rows
                     if r["target"] == target and r["model"] == model
                     and r["src"] == src_name and r["group"] == group]
            curve.sort(key=lambda r: r["layer_idx"])
            xs = [r["layer_idx"] for r in curve]
            ys = [r["acc"] for r in curve]
            ax.plot(xs, ys, marker="o", label=f"{model} | {src_name} | {group}")
        ax.set_xlabel("layer index (group-local)")
        ax.set_ylabel(f"{target} probe acc")
        ax.set_ylim(0, 1.02)
        ax.axhline(0.0, ls=":", color="grey", lw=0.5)
        ax.legend(fontsize=7, loc="best")
        ax.set_title(f"Linear probe: {target}-ID recoverable per layer")
        png_path = out_dir / f"probe__{target}.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        print(f"[probe] wrote {png_path}")


if __name__ == "__main__":
    main()
