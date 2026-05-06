"""Compute per-utterance frame-level comparison metrics (Phase B §6.4).

Four metrics comparing Whisper and DAC trajectories on the same utterance:
  - truth_prob_gap  : mean_t[P_W(t,c*) - P_D(t,c*)]
  - argmax_time_W   : t* = argmax_t P_W(t,c*) in seconds
  - argmax_time_D   : t* = argmax_t P_D(t,c*) in seconds
  - sharpness_W     : P_W(t*,c*) - mean_t P_W(t,c*)
  - sharpness_D     : P_D(t*,c*) - mean_t P_D(t,c*)
  - frame_agreement : fraction of frames where argmax_W == argmax_D
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _class_index(classes: np.ndarray, label: str) -> int:
    idx = np.where(classes == label)[0]
    if len(idx) == 0:
        raise ValueError(f"label {label!r} not in classes {list(classes)}")
    return int(idx[0])


def compute_metrics_for_utt(
    utt_id: str,
    true_label: str,
    traj_W_npz: Path,
    traj_D_npz: Path,
) -> dict:
    """Compute all 6 metrics for one utterance.

    Args:
        utt_id: utterance identifier
        true_label: ground-truth instrument family
        traj_W_npz: Whisper trajectory npz (keys: probs, classes, n_valid, fps)
        traj_D_npz: DAC trajectory npz

    Returns:
        Dict with utt_id + 6 metric fields. Returns NaN dict if files missing.
    """
    base = {"utt_id": utt_id}
    nan_row = {
        **base,
        **{
            k: float("nan")
            for k in (
                "truth_prob_gap",
                "argmax_time_W",
                "argmax_time_D",
                "sharpness_W",
                "sharpness_D",
                "frame_agreement",
            )
        },
    }

    if not traj_W_npz.exists() or not traj_D_npz.exists():
        return nan_row

    nW = np.load(str(traj_W_npz))
    nD = np.load(str(traj_D_npz))

    probs_W = nW["probs"]  # [T_W, C]
    probs_D = nD["probs"]  # [T_D, C]
    classes_W = nW["classes"]
    classes_D = nD["classes"]
    fps_W = float(nW["fps"])
    fps_D = float(nD["fps"])

    try:
        ci_W = _class_index(classes_W, true_label)
        ci_D = _class_index(classes_D, true_label)
    except ValueError:
        return nan_row

    p_W = probs_W[:, ci_W]  # [T_W]
    p_D = probs_D[:, ci_D]  # [T_D]

    # Resample D to W length for gap (linear interpolation)
    t_W = np.arange(len(p_W)) / fps_W
    t_D = np.arange(len(p_D)) / fps_D
    p_D_resampled = np.interp(t_W, t_D, p_D) if len(p_D) > 1 else np.full_like(p_W, p_D.mean())

    truth_prob_gap = float(np.mean(p_W - p_D_resampled))

    tstar_W = float(np.argmax(p_W) / fps_W)
    tstar_D = float(np.argmax(p_D) / fps_D)

    sharpness_W = float(p_W.max() - p_W.mean())
    sharpness_D = float(p_D.max() - p_D.mean())

    # Frame agreement: compare argmax at Whisper timestamps (align D to W time grid)
    argmax_W = np.argmax(probs_W, axis=1)  # [T_W] → class indices
    # Interpolate D probs to W time grid, then take argmax
    probs_D_aligned = np.stack(
        [
            np.interp(t_W, t_D, probs_D[:, c]) if len(p_D) > 1 else np.full(len(t_W), probs_D[:, c].mean())
            for c in range(probs_D.shape[1])
        ],
        axis=1,
    )  # [T_W, C_D]
    # Map class indices via shared class names
    classes_W_list = list(classes_W)
    classes_D_list = list(classes_D)
    # For each W frame, get D's argmax class name, compare with W's argmax class name
    argmax_D_aligned = np.argmax(probs_D_aligned, axis=1)
    # Both encoders trained on same classes but class order may differ
    agree = np.array(
        [classes_W_list[w_idx] == classes_D_list[d_idx] for w_idx, d_idx in zip(argmax_W, argmax_D_aligned)]
    )
    frame_agreement = float(agree.mean())

    return {
        **base,
        "truth_prob_gap": truth_prob_gap,
        "argmax_time_W": tstar_W,
        "argmax_time_D": tstar_D,
        "sharpness_W": sharpness_W,
        "sharpness_D": sharpness_D,
        "frame_agreement": frame_agreement,
    }


def compute_metrics_for_sets(
    sets_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    traj_W_dir: Path,
    traj_D_dir: Path,
    target_sets: list[str] = ("W+A-", "W+A+"),
) -> pd.DataFrame:
    """Batch-compute metrics for all utterances in target_sets.

    Args:
        sets_df: sample_sets.csv (utt_id, set_label, true_label)
        combined_df: combined predictions CSV (for true_label)
        traj_W_dir: directory of Whisper trajectory npzs
        traj_D_dir: directory of DAC trajectory npzs
        target_sets: which set labels to process

    Returns:
        DataFrame with columns: utt_id, set_label, true_label, + 6 metrics
    """
    label_map = dict(zip(combined_df["utt_id"], combined_df["true_label"]))
    rows = []
    subset = sets_df[sets_df["set_label"].isin(target_sets)]
    for i, (_, row) in enumerate(subset.iterrows()):
        utt_id = row["utt_id"]
        true_label = label_map.get(utt_id, row.get("true_label", ""))
        m = compute_metrics_for_utt(
            utt_id,
            true_label,
            traj_W_dir / f"{utt_id}.npz",
            traj_D_dir / f"{utt_id}.npz",
        )
        m["set_label"] = row["set_label"]
        m["true_label"] = true_label
        rows.append(m)
        if (i + 1) % 1000 == 0:
            logger.info("%d / %d done", i + 1, len(subset))
    return pd.DataFrame(rows)
