"""Linear CKA computation for RQ1.

Spec: docs/analysis/RQ1_layer_distance.md §4 / §5 / §6.

Two analysis axes:
  Axis 1 (within-model)  : per-family 15×15 matrix, layer × layer
  Axis 2 (between-model) : per-position cross-family CKA, 6 pairs × 15 positions

Variants:
  pooling : mean | last (encoder reuses mean for "last" since non-causal)
  subset  : overall (N=2000) | angry | happy | sad | neutral (each N=500)

Outputs (to {out_dir}/cka/):
  {family}_within_{pooling}_{subset}.npy   shape (15, 15)
  cross_position_{pooling}_{subset}.npy    shape (6, 15)
  layer_positions.npy                      shape (15,)
  family_pairs.npy                         shape (6,)

Total: 4 fam × 2 pooling × 5 subset = 40 within + 10 cross + 2 metadata = 52 files.

Usage:
    python experiments/audio_encoder_probe/compute_cka.py
"""

import argparse
import itertools
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FAMILIES = ["whisper_tiny", "whisper_small", "dacvae", "wavtok"]
FAMILY_PAIRS = list(itertools.combinations(FAMILIES, 2))  # 6 pairs

LAYER_POSITIONS = [
    "enc_0", "enc_1", "enc_2", "enc_3", "enc_4",
    "proj_0", "proj_1", "proj_2", "proj_3", "proj_out",
    "llm_0", "llm_8", "llm_15", "llm_23", "llm_31",
]
POOLINGS = ("mean", "last")
EMOTIONS = ("angry", "happy", "sad", "neutral")
SUBSETS = ("overall",) + EMOTIONS


def center(X: np.ndarray) -> np.ndarray:
    """Column-wise mean removal. Input/output (N, D)."""
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(
    X_c: np.ndarray,
    Y_c: np.ndarray,
    self_xx: float | None = None,
    self_yy: float | None = None,
) -> float:
    """Linear CKA, assumes X_c and Y_c are pre-centered.

    CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    Pass precomputed self-Frobenius terms to skip redundant computation.
    """
    XtY = X_c.T @ Y_c
    num = float(np.sum(XtY * XtY))
    if self_xx is None:
        XtX = X_c.T @ X_c
        self_xx = float(np.sum(XtX * XtX))
    if self_yy is None:
        YtY = Y_c.T @ Y_c
        self_yy = float(np.sum(YtY * YtY))
    denom = np.sqrt(self_xx * self_yy)
    if denom == 0.0:
        return 0.0
    return num / denom


def layer_key(pos: str, pooling: str) -> str:
    """Map layer position to npz key. Encoder layers always use mean (non-causal)."""
    if pos.startswith("enc_") or pooling == "mean":
        return f"{pos}_mean"
    return f"{pos}_last"


def load_all_npz(families: list[str], utt_ids: list[str], embeds_dir: Path) -> dict:
    """Load 4*N npz files into nested dict[family][utt_id][key] = array."""
    data: dict = {fam: {} for fam in families}
    total = len(families) * len(utt_ids)
    n = 0
    t0 = time.time()
    for fam in families:
        for uid in utt_ids:
            with np.load(embeds_dir / fam / f"{uid}.npz") as z:
                data[fam][uid] = {k: z[k] for k in z.files}
            n += 1
            if n % 2000 == 0:
                logger.info(f"  loaded {n}/{total} ({(time.time() - t0):.1f}s)")
    logger.info(f"  loaded all {n} files in {(time.time() - t0):.1f}s")
    return data


def stack_layer(npz_data: dict, family: str, pos: str, pooling: str, utt_ids: list[str]) -> np.ndarray:
    key = layer_key(pos, pooling)
    return np.stack([npz_data[family][uid][key] for uid in utt_ids]).astype(np.float32)


def build_features(npz_data: dict, families: list[str], utt_ids: list[str], pooling: str) -> tuple[dict, dict]:
    """Returns (features, self_hsic) where features[fam][pos] = (N, D) centered,
    self_hsic[fam][pos] = ||X^T X||_F^2."""
    features: dict[str, dict] = {}
    self_hsic: dict[str, dict] = {}
    for fam in families:
        features[fam] = {}
        self_hsic[fam] = {}
        for pos in LAYER_POSITIONS:
            X = stack_layer(npz_data, fam, pos, pooling, utt_ids)
            Xc = center(X)
            XtX = Xc.T @ Xc
            features[fam][pos] = Xc
            self_hsic[fam][pos] = float(np.sum(XtX * XtX))
    return features, self_hsic


def compute_within(features: dict, self_hsic: dict, family: str) -> np.ndarray:
    """15×15 symmetric CKA matrix for one family."""
    n = len(LAYER_POSITIONS)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            pi, pj = LAYER_POSITIONS[i], LAYER_POSITIONS[j]
            v = linear_cka(features[family][pi], features[family][pj],
                           self_hsic[family][pi], self_hsic[family][pj])
            M[i, j] = M[j, i] = v
    return M


def compute_cross(features: dict, self_hsic: dict) -> np.ndarray:
    """(6, 15) cross-family CKA per layer position."""
    cross = np.zeros((len(FAMILY_PAIRS), len(LAYER_POSITIONS)), dtype=np.float32)
    for k, (a, b) in enumerate(FAMILY_PAIRS):
        for i, pos in enumerate(LAYER_POSITIONS):
            cross[k, i] = linear_cka(features[a][pos], features[b][pos],
                                     self_hsic[a][pos], self_hsic[b][pos])
    return cross


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(
        REPO / "experiments/audio_encoder_probe/manifests/iemocap_4class_balanced500.csv"))
    parser.add_argument("--embeds-dir", default=str(
        REPO / "experiments/audio_encoder_probe/embeds_layers"))
    parser.add_argument("--out-dir", default=str(
        REPO / "experiments/audio_encoder_probe/cka"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest).sort_values("utt_id").reset_index(drop=True)
    logger.info(f"Manifest: {len(df)} utterances")

    all_utts = df["utt_id"].tolist()
    logger.info(f"Loading all npz files (4 families × {len(all_utts)} utts)...")
    npz_data = load_all_npz(FAMILIES, all_utts, Path(args.embeds_dir))

    utts_per_subset = {"overall": all_utts}
    for emo in EMOTIONS:
        utts_per_subset[emo] = df[df["emotion"] == emo]["utt_id"].tolist()
        logger.info(f"  '{emo}': {len(utts_per_subset[emo])} utts")

    np.save(out_dir / "layer_positions.npy", np.array(LAYER_POSITIONS))
    np.save(out_dir / "family_pairs.npy", np.array([f"{a}__{b}" for a, b in FAMILY_PAIRS]))

    t0 = time.time()
    for subset in SUBSETS:
        utt_ids = utts_per_subset[subset]
        for pooling in POOLINGS:
            logger.info(f"=== {subset} × {pooling} (N={len(utt_ids)}) ===")
            features, self_hsic = build_features(npz_data, FAMILIES, utt_ids, pooling)
            for fam in FAMILIES:
                M = compute_within(features, self_hsic, fam)
                if (M < -1e-4).any() or (M > 1.0 + 1e-4).any():
                    logger.warning(f"{fam}_within_{pooling}_{subset}: values out of [0, 1]")
                np.save(out_dir / f"{fam}_within_{pooling}_{subset}.npy", M)
            cross = compute_cross(features, self_hsic)
            np.save(out_dir / f"cross_position_{pooling}_{subset}.npy", cross)
    logger.info(f"Done in {(time.time() - t0)/60:.2f} min. Output: {out_dir}")


if __name__ == "__main__":
    main()
