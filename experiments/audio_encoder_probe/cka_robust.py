"""Statistical robustness for RQ1 — bootstrap CI + permutation test on key Axis 2 results.

Spec: docs/analysis/RQ1_layer_distance.md §9.

Sanity checks already covered elsewhere:
  - self-similarity, random baseline, range, symmetry      : tests/experiments/test_cka.py
  - DACVAE post-VAE z reproducibility (seed = stable_hash) : tests/experiments/test_extract_layers.py

This script computes for 6 family pairs × 4 key positions × 2 poolings = 48 results:
  - observed CKA (point estimate)
  - bootstrap 5/95 percentile CI (n=1000, with-replacement resampling)
  - permutation p-value (n=1000, null = utterance-shuffled)

Uses GPU torch for efficiency — N×N gram (kernel) form. Bootstrap/permutation
resample indices of pre-computed gram matrices instead of recomputing inner products.

Outputs:
  cka/axis2_ci.json   list of 48 dicts with CI + p-value

Usage:
  python experiments/audio_encoder_probe/cka_robust.py [--n-boot 1000] [--n-perm 1000]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "experiments/audio_encoder_probe"))
from compute_cka import (  # noqa: E402
    FAMILIES, FAMILY_PAIRS, load_all_npz, stack_layer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KEY_POSITIONS = ("enc_4", "proj_3", "proj_out", "llm_31")
POOLINGS = ("mean", "last")


def _center_kernel_t(K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """O(N^2) centering via H K H = K - r - c + g. Returns (K_c, hsic_kk) tensors."""
    row_m = K.mean(dim=0, keepdim=True)
    col_m = K.mean(dim=1, keepdim=True)
    grand = K.mean()
    K_c = K - row_m - col_m + grand
    return K_c, (K_c * K_c).sum()


def gram_and_center_t(X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Xt = torch.from_numpy(X).to(DEVICE, dtype=torch.float32)
    K = Xt @ Xt.T
    K_c, hsic = _center_kernel_t(K)
    return K, K_c, hsic


def cka_from_centered_t(K_c, L_c, hsic_kk, hsic_ll) -> torch.Tensor:
    hsic_kl = (K_c * L_c).sum()
    denom = torch.sqrt(hsic_kk * hsic_ll)
    return torch.where(denom > 0, hsic_kl / denom, torch.zeros_like(hsic_kl))


def bootstrap_ci_t(K: torch.Tensor, L: torch.Tensor, n_boot: int, seed: int) -> dict:
    """Bootstrap 5/95 CI on GPU. K, L: (N, N) torch on DEVICE."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    N = K.shape[0]
    vals = torch.empty(n_boot, device=DEVICE)
    for i in range(n_boot):
        idx = torch.randint(0, N, (N,), device=DEVICE, generator=gen)
        Kb = K.index_select(0, idx).index_select(1, idx)
        Lb = L.index_select(0, idx).index_select(1, idx)
        Kb_c, hkk = _center_kernel_t(Kb)
        Lb_c, hll = _center_kernel_t(Lb)
        vals[i] = cka_from_centered_t(Kb_c, Lb_c, hkk, hll)
    vals_np = vals.cpu().numpy()
    return {"p5": float(np.percentile(vals_np, 5)),
            "p95": float(np.percentile(vals_np, 95)),
            "mean": float(np.mean(vals_np))}


def permutation_p_t(K_c: torch.Tensor, L_c: torch.Tensor,
                    hsic_kk: torch.Tensor, hsic_ll: torch.Tensor,
                    observed: torch.Tensor, n_perm: int, seed: int) -> float:
    """p = P(CKA(K, shuffled L) >= observed) under null. GPU."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    N = K_c.shape[0]
    vals = torch.empty(n_perm, device=DEVICE)
    for i in range(n_perm):
        idx = torch.randperm(N, device=DEVICE, generator=gen)
        L_perm = L_c.index_select(0, idx).index_select(1, idx)
        vals[i] = cka_from_centered_t(K_c, L_perm, hsic_kk, hsic_ll)
    return float((vals >= observed).float().mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(
        REPO / "experiments/audio_encoder_probe/manifests/iemocap_4class_balanced500.csv"))
    parser.add_argument("--embeds-dir", default=str(
        REPO / "experiments/audio_encoder_probe/embeds_layers"))
    parser.add_argument("--cka-dir", default=str(REPO / "experiments/audio_encoder_probe/cka"))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cka_dir = Path(args.cka_dir)
    cka_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest).sort_values("utt_id").reset_index(drop=True)
    utts = df["utt_id"].tolist()
    logger.info(f"Manifest: {len(utts)} utterances")

    logger.info("Loading npz...")
    npz_data = load_all_npz(FAMILIES, utts, Path(args.embeds_dir))

    logger.info(f"Computing gram matrices on {DEVICE}...")
    cache: dict[tuple, tuple] = {}
    for fam in FAMILIES:
        for pos in KEY_POSITIONS:
            for pool in POOLINGS:
                X = stack_layer(npz_data, fam, pos, pool, utts).astype(np.float32)
                cache[(fam, pos, pool)] = gram_and_center_t(X)
    logger.info(f"  cached {len(cache)} gram matrices")

    results: list[dict] = []
    t0 = time.time()
    total = len(POOLINGS) * len(KEY_POSITIONS) * len(FAMILY_PAIRS)
    n = 0
    for pool in POOLINGS:
        for pos in KEY_POSITIONS:
            for a, b in FAMILY_PAIRS:
                n += 1
                K_full, K_c, hkk = cache[(a, pos, pool)]
                L_full, L_c, hll = cache[(b, pos, pool)]
                observed = cka_from_centered_t(K_c, L_c, hkk, hll)
                boot = bootstrap_ci_t(K_full, L_full, args.n_boot, seed=args.seed)
                p_val = permutation_p_t(K_c, L_c, hkk, hll, observed, args.n_perm, seed=args.seed)
                observed_f = float(observed.item())
                results.append({
                    "pair": f"{a}__{b}",
                    "position": pos,
                    "pooling": pool,
                    "observed_cka": observed_f,
                    "bootstrap_p5": boot["p5"],
                    "bootstrap_p95": boot["p95"],
                    "bootstrap_mean": boot["mean"],
                    "permutation_p_value": float(p_val),
                    "n_boot": args.n_boot,
                    "n_perm": args.n_perm,
                })
                logger.info(
                    f"[{n:2d}/{total}] {a}↔{b} {pos:8s} {pool}: "
                    f"CKA={observed_f:.4f} CI=[{boot['p5']:.4f}, {boot['p95']:.4f}] p={p_val:.4f}")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    out_path = cka_dir / "axis2_ci.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
