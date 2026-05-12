#!/usr/bin/env python
"""PCA scatter per (model, layer): each point is one utterance, colored by
transcript_id, marker shape by speaker. With same-transcript clustering you
want points of the same color to be close (== speaker invariance).

Inputs: dir with .npz files (same format as distances.py).
Outputs: one png per (model, layer) selected by --layer regex.
"""

import argparse
import re
from pathlib import Path

import numpy as np


def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:2].T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer_re", default=r"^(enc\.L(00|03|06|09|12|24|32)|proj\.L(01|04)|proj\.out|llm\.L(01|08|16|24|32))$",
                    help="regex on layer tag — selects which layers to render")
    ap.add_argument("--max_pairs", type=int, default=20,
                    help="show only the first K transcript_ids per (model, src) for legibility")
    ap.add_argument("--per_src", action="store_true",
                    help="render one figure per corpus")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_re = re.compile(args.layer_re)
    paths = sorted(Path(args.emb_dir).rglob("*.npz"))
    print(f"[pca] {len(paths)} files. layer filter: {args.layer_re}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("[pca] matplotlib required")

    for p in paths:
        z = np.load(p, allow_pickle=False)
        layer_tag = str(z["layer"])
        if not layer_re.match(layer_tag):
            continue
        model_tag = str(z["model"])
        emb = z["emb"]
        pair = z["pair"].astype(str)
        spk = z["spk"].astype(str)
        src = z["src"].astype(str)

        src_groups = [s for s in sorted(set(src.tolist()))] if args.per_src else [None]
        for s in src_groups:
            mask = (src == s) if s is not None else np.ones(len(src), dtype=bool)
            if mask.sum() < 4:
                continue
            E = emb[mask]
            P = pair[mask]
            S = spk[mask]
            keep_pairs = []
            seen_pairs = []
            for p_ in P:
                if p_ in seen_pairs:
                    continue
                seen_pairs.append(p_)
                if len(seen_pairs) <= args.max_pairs:
                    keep_pairs.append(p_)
            keep_mask = np.array([pp in set(keep_pairs) for pp in P])
            if keep_mask.sum() < 4:
                continue
            X2 = pca_2d(E[keep_mask])
            P2 = P[keep_mask]
            S2 = S[keep_mask]

            fig, ax = plt.subplots(figsize=(7, 6))
            unique_pairs = sorted(set(P2.tolist()))
            cmap = plt.get_cmap("tab20" if len(unique_pairs) <= 20 else "hsv")
            color_of = {p: cmap(i / max(1, len(unique_pairs) - 1)) for i, p in enumerate(unique_pairs)}
            unique_spk = sorted(set(S2.tolist()))
            markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]
            marker_of = {sp: markers[i % len(markers)] for i, sp in enumerate(unique_spk)}

            for p_, sp_ in [(p, s) for p in unique_pairs for s in unique_spk]:
                m = (P2 == p_) & (S2 == sp_)
                if not m.any():
                    continue
                ax.scatter(X2[m, 0], X2[m, 1], color=color_of[p_], marker=marker_of[sp_],
                           s=42, alpha=0.85, edgecolors="black", linewidths=0.3)
            title = f"{model_tag} | {layer_tag}"
            if s is not None:
                title += f" | {s}"
            ax.set_title(title)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            # tiny legend showing marker = speaker
            from matplotlib.lines import Line2D
            legend = [Line2D([0], [0], marker=marker_of[sp], color="w", label=sp,
                              markerfacecolor="grey", markeredgecolor="black", markersize=7)
                       for sp in unique_spk]
            ax.legend(handles=legend, title="speaker", fontsize=7, loc="best")

            stem = f"pca__{model_tag}__{layer_tag.replace('.', '_')}"
            if s is not None:
                stem += f"__{s}"
            fig_path = out_dir / f"{stem}.png"
            fig.tight_layout()
            fig.savefig(fig_path, dpi=160)
            plt.close(fig)
            print(f"[pca] wrote {fig_path}")


if __name__ == "__main__":
    main()
