"""Extract per-utterance frame-level probability trajectories.

For each target utt, applies the fold-appropriate classifier (trained on other folds)
to every valid frame, producing P(t, c) in [T, n_classes].

Output per utt: {utt_id}.npz with keys:
  probs    : [T, n_classes] float32  — softmax probabilities
  classes  : [n_classes]   str      — class names (alphabetical, matches LogReg order)
  n_valid  : scalar int              — number of valid frames used
  fps      : scalar float            — frames per second for this encoder
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.audio_encoder_probe.phase_b.frame_probe import load_classifier


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[3]
PROBE_ROOT = ROOT / "experiments" / "audio_encoder_probe"
FRAMES_ROOT = PROBE_ROOT / "embeds_frames"
PHASE_B_DIR = PROBE_ROOT / "_results" / "phase_b"

ENCODER_FPS = {"whisper_small": 50.0, "dacvae": 25.0}


def extract_trajectories(
    encoder: str,
    utt_ids: list[str],
    fold_map: dict[str, int],
    clf_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> None:
    """Extract and save probability trajectories for a list of utterances.

    Args:
        encoder: "whisper_small" or "dacvae"
        utt_ids: utterance IDs to process
        fold_map: {utt_id → fold} mapping (to pick correct held-out classifier)
        clf_dir: directory with {encoder}_fold{k}.pkl classifiers
        out_dir: output directory for per-utt npz files
        overwrite: if False, skip already-existing files
    """
    frames_dir = FRAMES_ROOT / encoder / "nsynth_train_30k"
    fps = ENCODER_FPS[encoder]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache classifiers by fold
    clf_cache: dict[int, tuple] = {}

    done, skipped, missing = 0, 0, 0
    for utt_id in utt_ids:
        out_npz = out_dir / f"{utt_id}.npz"
        if not overwrite and out_npz.exists():
            skipped += 1
            continue

        frame_npz = frames_dir / f"{utt_id}.npz"
        if not frame_npz.exists():
            missing += 1
            continue

        fold = fold_map[utt_id]
        if fold not in clf_cache:
            pkl = clf_dir / f"{encoder}_fold{fold}.pkl"
            clf_cache[fold] = load_classifier(pkl)
        scaler, clf, classes = clf_cache[fold]

        npz = np.load(str(frame_npz))
        frames = npz["frames"].astype(np.float32)
        n_valid = int(npz["n_valid"])
        frames = frames[:n_valid]

        if len(frames) == 0:
            missing += 1
            continue

        X_scaled = scaler.transform(frames)
        log_probs = clf.predict_log_proba(X_scaled)
        # Numerically stable softmax (log_probs already log-softmax from sklearn)
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        np.savez_compressed(
            out_npz,
            probs=probs.astype(np.float32),
            classes=np.array(classes),
            n_valid=np.int32(n_valid),
            fps=np.float32(fps),
        )
        done += 1
        if done % 500 == 0:
            logger.info("encoder=%s: %d done, %d skipped, %d missing", encoder, done, skipped, missing)

    logger.info("encoder=%s: done=%d skipped=%d missing=%d → %s", encoder, done, skipped, missing, out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True, choices=["whisper_small", "dacvae"])
    p.add_argument("--sample_sets_csv", type=Path, default=PROBE_ROOT / "_results" / "phase_a" / "sample_sets.csv")
    p.add_argument("--manifest_csv", type=Path, default=PROBE_ROOT / "manifests" / "nsynth_train_30k.csv")
    p.add_argument("--clf_dir", type=Path, default=PHASE_B_DIR / "classifiers")
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--sets", nargs="+", default=["W+A-", "W+A+"], help="which sets to process")
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = PHASE_B_DIR / "trajectories" / args.encoder

    sets_df = pd.read_csv(args.sample_sets_csv)
    manifest_df = pd.read_csv(args.manifest_csv)
    fold_map = dict(zip(manifest_df["utt_id"], manifest_df["fold"]))

    target = sets_df[sets_df["set_label"].isin(args.sets)]
    utt_ids = target["utt_id"].tolist()
    logger.info("Processing %d utterances for sets %s", len(utt_ids), args.sets)

    extract_trajectories(args.encoder, utt_ids, fold_map, args.clf_dir, args.out_dir)


if __name__ == "__main__":
    main()
