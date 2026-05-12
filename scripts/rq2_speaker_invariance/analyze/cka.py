#!/usr/bin/env python
"""Linear CKA between (model_a, layer_a) and (model_b, layer_b) for the
same set of utterances.

Use cases:
  1. raw HF Whisper-tiny encoder layer i  vs  ALM-trained Whisper-tiny encoder layer i
       → how much did Stage1/Stage2 training move the encoder representation
  2. Whisper-large-v2 encoder layer i      vs  Whisper-tiny encoder layer j
       → does small encoder land at a different point in representation space
  3. ALM proj.out                          vs  ALM llm.LXX
       → at which LM layer does audio merge with text-space

For each .npz the analyzer needs same `pair, spk, src` ordering. We canonicalize
by sorting on `(src, pair, spk)` before computing CKA so utterance order
matches across npz files even if extractors processed them in different
batch orders.

Outputs:
  out/cka.csv         (model_a, layer_a, model_b, layer_b, cka, n)
  out/cka_heatmap__{model_a}_vs_{model_b}.png
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_layer(layer: str):
    m = re.match(r"^(?P<group>enc|proj|llm)\.(?:L(?P<idx>\d+)|(?P<tag>out|norm))$", layer)
    if not m:
        return ("?", -1)
    group = m.group("group")
    if m.group("tag") in ("out", "norm"):
        return (group, 999)
    return (group, int(m.group("idx")))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019). X:[N,Dx], Y:[N,Dy], same row order."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    num = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(num / (den + 1e-12))


def canonical_sort(z) -> np.ndarray:
    """Return permutation that sorts rows by (src, pair, spk)."""
    keys = np.array([f"{z['src'][i]}|{z['pair'][i]}|{z['spk'][i]}" for i in range(len(z["pair"]))])
    return np.argsort(keys, kind="stable")


def load_emb(np_path: Path):
    z = np.load(np_path, allow_pickle=False)
    perm = canonical_sort(z)
    return {
        "model": str(z["model"]),
        "layer": str(z["layer"]),
        "emb": z["emb"][perm],
        "pair": z["pair"][perm],
        "spk": z["spk"][perm],
        "src": z["src"][perm],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair_models", nargs="*", default=None,
                    help="explicit list of (model_a, model_b) pairs to compute, "
                         "given as colon-separated, e.g. "
                         "'whisper_tiny_en:alm_whisper_tiny_v6 hubert_base_ls960:alm_whisper_tiny_v6'. "
                         "If omitted, compute every distinct ordered pair (model_a < model_b lexicographically).")
    ap.add_argument("--heatmap", action="store_true",
                    help="render heatmap (model_a layers x model_b layers) per pair")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_by_model: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(Path(args.emb_dir).rglob("*.npz")):
        z_peek = np.load(p, allow_pickle=False)
        files_by_model[str(z_peek["model"])].append(p)
    print(f"[cka] models found: {sorted(files_by_model.keys())}")

    if args.pair_models:
        pairs = [tuple(p.split(":")) for p in args.pair_models]
    else:
        models = sorted(files_by_model.keys())
        pairs = list(itertools.combinations(models, 2))
    print(f"[cka] pairs to compute: {pairs}")

    rows = []
    for model_a, model_b in pairs:
        if model_a not in files_by_model or model_b not in files_by_model:
            print(f"[cka] WARN missing model in pair: {model_a} / {model_b}")
            continue
        # Load all layers of both models
        layers_a = {load_emb(p)["layer"]: load_emb(p) for p in files_by_model[model_a]}
        layers_b = {load_emb(p)["layer"]: load_emb(p) for p in files_by_model[model_b]}
        # Verify alignment by checking pair/spk arrays equal across one common layer
        any_a = next(iter(layers_a.values()))
        any_b = next(iter(layers_b.values()))
        # Restrict to the intersection of (src, pair, spk) keys
        key_a = np.array([f"{any_a['src'][i]}|{any_a['pair'][i]}|{any_a['spk'][i]}"
                          for i in range(len(any_a['pair']))])
        key_b = np.array([f"{any_b['src'][i]}|{any_b['pair'][i]}|{any_b['spk'][i]}"
                          for i in range(len(any_b['pair']))])
        common, ia, ib = np.intersect1d(key_a, key_b, return_indices=True)
        if len(common) < 8:
            print(f"[cka] WARN {model_a} vs {model_b}: only {len(common)} common utts, skipping")
            continue
        print(f"[cka] {model_a} vs {model_b}: {len(common)} common utts, "
              f"{len(layers_a)} x {len(layers_b)} layer combinations")

        for la, da in sorted(layers_a.items(), key=lambda kv: parse_layer(kv[0])):
            for lb, db in sorted(layers_b.items(), key=lambda kv: parse_layer(kv[0])):
                X = da["emb"][ia]
                Y = db["emb"][ib]
                cka = linear_cka(X, Y)
                rows.append({
                    "model_a": model_a, "layer_a": la,
                    "model_b": model_b, "layer_b": lb,
                    "cka": cka,
                    "n": int(len(common)),
                })

        if args.heatmap:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                continue
            la_keys = sorted(layers_a.keys(), key=parse_layer)
            lb_keys = sorted(layers_b.keys(), key=parse_layer)
            mat = np.zeros((len(la_keys), len(lb_keys)), dtype=float)
            for r in rows:
                if r["model_a"] != model_a or r["model_b"] != model_b:
                    continue
                i = la_keys.index(r["layer_a"])
                j = lb_keys.index(r["layer_b"])
                mat[i, j] = r["cka"]
            fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(lb_keys)),
                                            max(5, 0.35 * len(la_keys))))
            im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(lb_keys)))
            ax.set_xticklabels(lb_keys, rotation=90, fontsize=6)
            ax.set_yticks(range(len(la_keys)))
            ax.set_yticklabels(la_keys, fontsize=6)
            ax.set_xlabel(model_b)
            ax.set_ylabel(model_a)
            ax.set_title(f"linear CKA  {model_a}  vs  {model_b}  (N={len(common)})")
            fig.colorbar(im, ax=ax, shrink=0.85)
            png_path = out_dir / f"cka__{model_a}__vs__{model_b}.png"
            fig.tight_layout()
            fig.savefig(png_path, dpi=160)
            plt.close(fig)
            print(f"[cka] wrote {png_path}")

    csv_path = out_dir / "cka.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_a", "layer_a", "model_b", "layer_b", "cka", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[cka] wrote {csv_path}  rows={len(rows)}")


if __name__ == "__main__":
    main()
