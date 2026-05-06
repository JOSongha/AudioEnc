"""Train per-fold frame-level logistic regression classifiers.

For each (encoder, fold), trains LogisticRegression where each frame is a sample
and the label is the utterance-level instrument family. Classifiers are saved to
disk so trajectories.py can apply them without re-training.

Memory note: whisper_small frames are [T=200, 768] float16 per utt. To keep
training memory under ~1GB we subsample FRAMES_PER_UTT frames per utterance
during training only; inference uses all valid frames.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[3]
PROBE_ROOT = ROOT / "experiments" / "audio_encoder_probe"
FRAMES_ROOT = PROBE_ROOT / "embeds_frames"
MANIFEST_DIR = PROBE_ROOT / "manifests"
PHASE_B_DIR = PROBE_ROOT / "_results" / "phase_b"

FRAMES_PER_UTT = 20  # frames subsampled per utt during training


def load_frames_for_fold(
    encoder: str,
    manifest_df: pd.DataFrame,
    test_fold: int,
    frames_per_utt: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and subsample frames for training folds (all folds except test_fold).

    Returns:
        X: (N_frames, d) float32
        y: (N_frames,) str labels
    """
    frames_dir = FRAMES_ROOT / encoder / "nsynth_train_30k"
    train_df = manifest_df[manifest_df["fold"] != test_fold]

    X_parts, y_parts = [], []
    missing = 0
    for row in train_df.to_dict("records"):
        npz_path = frames_dir / f"{row['utt_id']}.npz"
        if not npz_path.exists():
            missing += 1
            continue
        npz = np.load(str(npz_path))
        frames = npz["frames"].astype(np.float32)  # [T, d]
        n_valid = int(npz["n_valid"])
        frames = frames[:n_valid]
        if len(frames) == 0:
            continue
        # Subsample
        idx = rng.choice(len(frames), size=min(frames_per_utt, len(frames)), replace=False)
        X_parts.append(frames[idx])
        y_parts.extend([row["label"]] * len(idx))

    if missing:
        logger.warning("fold=%d: %d/%d utterances missing npz", test_fold, missing, len(train_df))

    X = np.concatenate(X_parts, axis=0)
    y = np.array(y_parts)
    return X, y


def train_fold_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 0.1,
    max_iter: int = 1000,
    seed: int = 42,
) -> tuple[StandardScaler, LogisticRegression]:
    """Fit scaler + logistic regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_scaled, y_train)
    return scaler, clf


def train_all_folds(
    encoder: str,
    manifest_csv: Path,
    out_dir: Path,
    frames_per_utt: int = FRAMES_PER_UTT,
    C: float = 0.1,
    seed: int = 42,
) -> None:
    """Train one classifier per fold and save to out_dir/{encoder}_fold{k}.pkl."""
    manifest_df = pd.read_csv(manifest_csv)
    rng = np.random.default_rng(seed)
    folds = sorted(manifest_df["fold"].unique())
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        out_pkl = out_dir / f"{encoder}_fold{fold}.pkl"
        if out_pkl.exists():
            logger.info("skip fold=%d (already exists: %s)", fold, out_pkl)
            continue
        logger.info("encoder=%s fold=%d: loading frames ...", encoder, fold)
        X, y = load_frames_for_fold(encoder, manifest_df, fold, frames_per_utt, rng)
        logger.info("  X=%s  classes=%d", X.shape, len(set(y)))
        scaler, clf = train_fold_classifier(X, y, C=C)
        train_acc = (clf.predict(scaler.transform(X)) == y).mean()
        logger.info("  train_acc=%.4f  n_iter=%s", train_acc, clf.n_iter_)
        with open(out_pkl, "wb") as fh:
            pickle.dump({"scaler": scaler, "clf": clf, "fold": fold, "encoder": encoder}, fh)
        logger.info("  saved → %s", out_pkl)


def load_classifier(pkl_path: Path) -> tuple[StandardScaler, LogisticRegression, list[str]]:
    """Return (scaler, clf, classes) from a saved pkl."""
    with open(pkl_path, "rb") as fh:
        obj = pickle.load(fh)
    clf = obj["clf"]
    return obj["scaler"], clf, list(clf.classes_)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True, choices=["whisper_small", "dacvae"])
    p.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "nsynth_train_30k.csv")
    p.add_argument("--out_dir", type=Path, default=PHASE_B_DIR / "classifiers")
    p.add_argument("--frames_per_utt", type=int, default=FRAMES_PER_UTT)
    p.add_argument("--C", type=float, default=0.1)
    args = p.parse_args()
    train_all_folds(args.encoder, args.manifest, args.out_dir, args.frames_per_utt, args.C)


if __name__ == "__main__":
    main()
