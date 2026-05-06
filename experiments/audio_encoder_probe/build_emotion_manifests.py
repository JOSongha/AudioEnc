"""
Build RAVDESS and CREMA-D manifests with leave-N-speakers-out folds.

RAVDESS filename: 03-01-06-02-01-02-15.wav
  fields: modality-channel-emotion-intensity-statement-repetition-actor
  emotion codes: 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful
                 07=disgust 08=surprised
  actor: 01-24 (odd=male, even=female)
  Use 8-class.

CREMA-D filename: 1001_IEO_ANG_LO.wav
  fields: actorID_sentence_emotion_intensity
  emotion codes: ANG, DIS, FEA, HAP, NEU, SAD
  actorID: 1001-1091 (91 actors)
  Use 6-class.

Folds (leave-N-speakers-out):
  RAVDESS — 5 folds, ~5 speakers per fold (24/5 ≈ 5)
  CREMA-D — 5 folds, ~18 speakers per fold (91/5 ≈ 18)
"""

import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "manifests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── RAVDESS ──────────────────────────────────────────────────────────────────

RAVDESS_DIR = Path("/mnt/tmp/datasets/emotion_raw/RAVDESS/AudioSpeech")
RAVDESS_EMO = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

def build_ravdess():
    rows = []
    for wav in sorted(RAVDESS_DIR.glob("*.wav")):
        m = re.match(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$", wav.stem)
        if not m:
            continue
        modality, channel, emo, intensity, statement, repetition, actor = m.groups()
        if modality != "03":  # 03=audio-only
            continue
        if emo not in RAVDESS_EMO:
            continue
        actor_int = int(actor)
        rows.append({
            "utt_id": wav.stem,
            "audio_path": str(wav),
            "emotion": RAVDESS_EMO[emo],
            "intensity": "normal" if intensity == "01" else "strong",
            "actor": actor_int,
            "gender": "M" if actor_int % 2 == 1 else "F",
        })

    df = pd.DataFrame(rows)
    # 5-fold by actor: actors 1-5, 6-10, 11-15, 16-20, 21-24
    bins = pd.cut(df.actor, bins=[0, 5, 10, 15, 20, 24], labels=[1, 2, 3, 4, 5])
    df["fold"] = bins.astype(int)
    out = OUT_DIR / "ravdess.csv"
    df.to_csv(out, index=False)
    print(f"RAVDESS: wrote {len(df)} rows to {out}")
    print(f"  Class dist: {df.emotion.value_counts().to_dict()}")
    print(f"  Fold dist: {df.fold.value_counts().sort_index().to_dict()}")
    print(f"  Actors per fold: {df.groupby('fold').actor.nunique().to_dict()}")


# ── CREMA-D ──────────────────────────────────────────────────────────────────

CREMAD_DIR = Path("/mnt/tmp/datasets/emotion_raw/CREMA-D/AudioWAV")
CREMAD_EMO = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

def build_cremad():
    rows = []
    for wav in sorted(CREMAD_DIR.glob("*.wav")):
        m = re.match(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]+)$", wav.stem)
        if not m:
            continue
        actor_id, sentence, emo, intensity = m.groups()
        if emo not in CREMAD_EMO:
            continue
        rows.append({
            "utt_id": wav.stem,
            "audio_path": str(wav),
            "emotion": CREMAD_EMO[emo],
            "intensity": intensity,
            "actor": int(actor_id),
        })

    df = pd.DataFrame(rows)
    # 5-fold by actor: 91 actors → ~18 per fold
    actors = sorted(df.actor.unique())
    fold_size = (len(actors) + 4) // 5
    fold_map = {a: (i // fold_size) + 1 for i, a in enumerate(actors)}
    df["fold"] = df.actor.map(fold_map).astype(int)
    out = OUT_DIR / "cremad.csv"
    df.to_csv(out, index=False)
    print(f"\nCREMA-D: wrote {len(df)} rows to {out}")
    print(f"  Class dist: {df.emotion.value_counts().to_dict()}")
    print(f"  Fold dist: {df.fold.value_counts().sort_index().to_dict()}")
    print(f"  Actors per fold: {df.groupby('fold').actor.nunique().to_dict()}")


if __name__ == "__main__":
    build_ravdess()
    build_cremad()
