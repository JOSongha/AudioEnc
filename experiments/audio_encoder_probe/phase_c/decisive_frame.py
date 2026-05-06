"""Characterise the Whisper-decisive frame for each W+A- utterance.

For each utt in W+A-, loads the two encoder trajectories and computes:
  - dac_entropy_at_decisive : Shannon entropy of DAC's P(t_W*, c) at Whisper's argmax time
  - dac_truth_prob_at_decisive : P_D(t_W*, c*)  — DAC probability on correct class
  - whisper_truth_prob_at_decisive : P_W(t_W*, c*)
  - frame_position_norm : argmax_time_W / 4.0  (0=onset, 1=end, NSynth clips = 4s)
  - mel_energy_at_decisive : mean log-mel energy at the decisive frame (16kHz audio)
  - spectral_centroid_at_decisive : spectral centroid (Hz) at decisive frame
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NSYNTH_AUDIO_DIR = Path("/mnt/tmp/datasets/music/nsynth/train_30k")
AUDIO_SR = 16000
CLIP_DURATION = 4.0  # seconds, all NSynth clips


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy (nats) of a probability vector."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _load_trajectory(npz_path: Path) -> tuple[np.ndarray, list[str], float] | None:
    """Returns (probs [T,C], classes, fps) or None if file missing."""
    if not npz_path.exists():
        return None
    npz = np.load(str(npz_path))
    return npz["probs"].astype(np.float32), list(npz["classes"]), float(npz["fps"])


def _mel_features_at_time(audio_path: str, t_sec: float, sr: int = AUDIO_SR) -> dict:
    """Compute log-mel energy and spectral centroid at a given time offset.

    Uses a 25ms window centred on t_sec. Falls back gracefully if audio unreadable.
    """
    try:
        import torchaudio

        wav, file_sr = torchaudio.load(audio_path)
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)
        wav = wav[0]  # mono

        win = int(0.025 * sr)  # 25ms
        centre = int(t_sec * sr)
        lo = max(0, centre - win // 2)
        hi = min(len(wav), lo + win)
        chunk = wav[lo:hi].numpy().astype(np.float32)
        if len(chunk) == 0:
            return {"mel_energy": float("nan"), "spectral_centroid": float("nan")}

        # log-mel energy
        fft = np.abs(np.fft.rfft(chunk, n=512))
        mel_energy = float(np.log1p(np.mean(fft**2)))

        # spectral centroid
        freqs = np.fft.rfftfreq(512, d=1.0 / sr)
        sc = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-9))
        return {"mel_energy": mel_energy, "spectral_centroid": sc}
    except Exception:
        return {"mel_energy": float("nan"), "spectral_centroid": float("nan")}


def compute_decisive_frame_features(
    metrics_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    traj_W_dir: Path,
    traj_D_dir: Path,
    audio_dir: Path = NSYNTH_AUDIO_DIR,
    target_set: str = "W+A-",
    compute_audio_features: bool = True,
) -> pd.DataFrame:
    """For each utt in target_set, compute decisive-frame characterisation.

    Args:
        metrics_df: frame_metrics.csv (has argmax_time_W, set_label, utt_id)
        combined_df: combined predictions CSV (has true_label, audio_path)
        traj_W_dir: Whisper trajectory npz directory
        traj_D_dir: DAC trajectory npz directory
        audio_dir: NSynth audio root
        target_set: "W+A-" or "W+A+"
        compute_audio_features: if True, load WAV to compute mel_energy/centroid

    Returns:
        DataFrame with per-utt decisive-frame features.
    """
    subset = metrics_df[metrics_df["set_label"] == target_set].dropna(subset=["argmax_time_W"])
    label_map = dict(zip(combined_df["utt_id"], combined_df["true_label"]))
    audio_map = dict(zip(combined_df["utt_id"], combined_df["audio_path"]))

    rows = []
    for i, (_, row) in enumerate(subset.iterrows()):
        utt_id = row["utt_id"]
        t_star = float(row["argmax_time_W"])
        true_label = label_map.get(utt_id, "")

        res = {
            "utt_id": utt_id,
            "set_label": target_set,
            "true_label": true_label,
            "argmax_time_W": t_star,
            "frame_position_norm": t_star / CLIP_DURATION,
        }

        # DAC features at Whisper's decisive frame
        traj_D = _load_trajectory(traj_D_dir / f"{utt_id}.npz")
        traj_W = _load_trajectory(traj_W_dir / f"{utt_id}.npz")

        if traj_D is not None and traj_W is not None:
            probs_D, classes_D, fps_D = traj_D
            probs_W, classes_W, fps_W = traj_W

            # Map t_star to DAC frame index
            t_star_D_idx = min(int(t_star * fps_D), len(probs_D) - 1)
            t_star_W_idx = min(int(t_star * fps_W), len(probs_W) - 1)

            dac_probs_at_decisive = probs_D[t_star_D_idx]
            whi_probs_at_decisive = probs_W[t_star_W_idx]

            res["dac_entropy_at_decisive"] = _entropy(dac_probs_at_decisive)

            # DAC truth prob at decisive frame
            if true_label in classes_D:
                ci_D = classes_D.index(true_label)
                res["dac_truth_prob_at_decisive"] = float(dac_probs_at_decisive[ci_D])
            else:
                res["dac_truth_prob_at_decisive"] = float("nan")

            if true_label in classes_W:
                ci_W = classes_W.index(true_label)
                res["whisper_truth_prob_at_decisive"] = float(whi_probs_at_decisive[ci_W])
            else:
                res["whisper_truth_prob_at_decisive"] = float("nan")
        else:
            for k in ("dac_entropy_at_decisive", "dac_truth_prob_at_decisive", "whisper_truth_prob_at_decisive"):
                res[k] = float("nan")

        # Audio spectral features
        if compute_audio_features:
            audio_path = audio_map.get(utt_id, str(audio_dir / f"{utt_id}.wav"))
            feats = _mel_features_at_time(audio_path, t_star)
            res.update(feats)

        rows.append(res)
        if (i + 1) % 1000 == 0:
            logger.info("%d / %d processed", i + 1, len(subset))

    return pd.DataFrame(rows)
