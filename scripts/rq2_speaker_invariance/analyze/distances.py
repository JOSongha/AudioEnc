#!/usr/bin/env python
"""Per-layer cosine-distance analysis.

Three pair populations per (model, layer, corpus):

  same_t_diff_s : same transcript, different speaker   (§4.3 "across-speaker")
  same_s_diff_t : same speaker, different transcript   (§4.3 "within-speaker")
  diff_s_diff_t : different speaker, different transcript

Reported ratios:

  ratio_paper   = mean(same_t_diff_s) / mean(same_s_diff_t)
                  > 1 ⇒ speaker dominates over content
                  matches existing sec43_embeddings analyzer convention
  ratio_inv     = mean(same_t_diff_s) / mean(diff_s_diff_t)
                  < 1 ⇒ same-content clusters tighter than random control
                  (no matched-speaker baseline; just "do same-transcript
                   utterances cluster, period")

Outputs:
  out/dist_per_layer.csv
  out/dist_per_layer__{ratio_kind}__{metric}.png

Inputs: one or more .npz files written by the extractors (each = one
(model, layer); arrays `emb, pair, spk, src`).
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


LAYER_RE = re.compile(r"^(?P<group>enc|proj|llm)\.(?:L(?P<idx>\d+)|(?P<tag>out|norm))$")


def parse_layer(layer: str):
    m = LAYER_RE.match(layer)
    if not m:
        return ("?", -1)
    group = m.group("group")
    if m.group("tag") in ("out", "norm"):
        return (group, 999)
    return (group, int(m.group("idx")))


def cosine_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """X:[N,D] Y:[M,D] -> [N,M] cosine distance in [0, 2]."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Xn @ Yn.T


def l2n_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return np.linalg.norm(Xn[:, None, :] - Yn[None, :, :], axis=-1) ** 2


METRICS = {"cos": cosine_dist, "l2n": l2n_dist}


def population_means(emb, pair, spk, *, metric_fn):
    """Compute mean distance for the three contrast populations.

    Uses upper-triangle pairs (i < j). For embeddings of size >2k this is
    O(N^2 * D) memory; we keep N small (~1k) per (model, layer, corpus).
    """
    n = emb.shape[0]
    if n < 4:
        return None
    D = metric_fn(emb, emb)
    iu = np.triu_indices(n, k=1)
    d = D[iu]
    same_t = (pair[:, None] == pair[None, :])[iu]
    same_s = (spk[:, None] == spk[None, :])[iu]

    m_stds = same_t & ~same_s
    m_ssdt = ~same_t & same_s
    m_dsdt = ~same_t & ~same_s

    out = {}
    for name, mask in [("same_t_diff_s", m_stds), ("same_s_diff_t", m_ssdt),
                       ("diff_s_diff_t", m_dsdt)]:
        if mask.any():
            out[name] = (float(d[mask].mean()), int(mask.sum()))
        else:
            out[name] = (float("nan"), 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(Path(args.emb_dir).rglob("*.npz"))
    print(f"[dist] found {len(npz_paths)} .npz files")

    rows = []
    for np_path in npz_paths:
        z = np.load(np_path, allow_pickle=False)
        model_tag = str(z["model"])
        layer_tag = str(z["layer"])
        emb = z["emb"]
        pair = z["pair"].astype(str)
        spk = z["spk"].astype(str)
        src = z["src"].astype(str)

        for src_name in sorted(set(src.tolist())):
            mask = src == src_name
            if mask.sum() < 4:
                continue
            for metric_name, fn in METRICS.items():
                pops = population_means(emb[mask], pair[mask], spk[mask], metric_fn=fn)
                if pops is None:
                    continue
                stds_m, stds_n = pops["same_t_diff_s"]
                ssdt_m, ssdt_n = pops["same_s_diff_t"]
                dsdt_m, dsdt_n = pops["diff_s_diff_t"]
                ratio_paper = stds_m / ssdt_m if ssdt_m > 0 else float("nan")
                ratio_inv = stds_m / dsdt_m if dsdt_m > 0 else float("nan")
                group, idx = parse_layer(layer_tag)
                rows.append({
                    "model": model_tag,
                    "src": src_name,
                    "layer": layer_tag,
                    "group": group,
                    "layer_idx": idx,
                    "metric": metric_name,
                    "same_t_diff_s": stds_m, "n_stds": stds_n,
                    "same_s_diff_t": ssdt_m, "n_ssdt": ssdt_n,
                    "diff_s_diff_t": dsdt_m, "n_dsdt": dsdt_n,
                    "ratio_paper": ratio_paper,
                    "ratio_inv": ratio_inv,
                })

    csv_path = out_dir / "dist_per_layer.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "model", "src", "layer", "group", "layer_idx", "metric",
        "same_t_diff_s", "n_stds", "same_s_diff_t", "n_ssdt",
        "diff_s_diff_t", "n_dsdt", "ratio_paper", "ratio_inv",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[dist] wrote {csv_path}  rows={len(rows)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[dist] matplotlib missing, skipping plot")
        return

    for metric_name in METRICS:
        for ratio_key, ylabel in [
            ("ratio_paper",
             f"{metric_name} ratio  same-t-diff-s / same-s-diff-t  (>1 = speaker dominates)"),
            ("ratio_inv",
             f"{metric_name} ratio  same-t-diff-s / diff-s-diff-t  (<1 = transcript clusters)"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            seen = set()
            for r in rows:
                if r["metric"] != metric_name:
                    continue
                seen.add((r["model"], r["src"], r["group"]))
            for key in sorted(seen):
                model, src_name, group = key
                curve = [r for r in rows
                         if r["metric"] == metric_name and r["model"] == model
                         and r["src"] == src_name and r["group"] == group]
                curve.sort(key=lambda r: r["layer_idx"])
                xs = [r["layer_idx"] for r in curve]
                ys = [r[ratio_key] for r in curve]
                ax.plot(xs, ys, marker="o", label=f"{model} | {src_name} | {group}")
            ax.set_xlabel("layer index (group-local)")
            ax.set_ylabel(ylabel)
            ax.axhline(1.0, ls="--", color="grey", lw=0.8)
            ax.legend(fontsize=7, loc="best")
            ax.set_title(f"{ratio_key}  ({metric_name})")
            png_path = out_dir / f"dist__{ratio_key}__{metric_name}.png"
            fig.tight_layout()
            fig.savefig(png_path, dpi=160)
            plt.close(fig)
            print(f"[dist] wrote {png_path}")


if __name__ == "__main__":
    main()
