"""
Build IEMOCAP 4-class manifest from raw jsonl shards.

Source:
  /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/emotion/iemocap_shard_*.jsonl
  Each row: {task, audio_path (stale), response, source}

Audio files (real location):
  /mnt/ddn/kyudan/IEMOCAP/data/<utt_id>.wav

Output:
  manifests/iemocap_4class.csv
  columns: utt_id, audio_path, emotion, session, gender, duration_s
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
import torchaudio

REPO = Path(__file__).resolve().parent.parent.parent
SRC_JSONL = REPO / "external/datasets/emotion"
WAV_ROOT = Path("/mnt/ddn/kyudan/IEMOCAP/data")
OUT_CSV = Path(__file__).parent / "manifests/iemocap_4class.csv"

# Standard 4-class mapping (Yoon 2018 / Tseng 2017)
EMO_MAP = {
    "angry": "angry",
    "happy": "happy",
    "excited": "happy",   # merge per convention
    "sad": "sad",
    "neutral": "neutral",
}

UTT_RE = re.compile(r"^Ses(\d{2})([FM])_.*?_([FM])\d+$")


def parse_utt_id(utt_id: str):
    m = UTT_RE.match(utt_id)
    if not m:
        return None, None
    session = f"Ses{m.group(1)}"
    speaker_gender = m.group(3)  # gender of the speaker for this utterance
    return session, speaker_gender


def main():
    rows = []
    for shard in sorted(SRC_JSONL.glob("iemocap_shard_*.jsonl")):
        with open(shard) as f:
            for line in f:
                d = json.loads(line)
                utt_id = Path(d["audio_path"]).stem  # e.g. Ses01F_impro01_F000
                emo_raw = d["response"]
                if emo_raw not in EMO_MAP:
                    continue
                emo = EMO_MAP[emo_raw]

                session, gender = parse_utt_id(utt_id)
                if session is None:
                    print(f"Skip malformed utt_id: {utt_id}")
                    continue

                wav_path = WAV_ROOT / f"{utt_id}.wav"
                if not wav_path.exists():
                    print(f"Missing wav: {wav_path}")
                    continue

                try:
                    info = torchaudio.info(str(wav_path))
                    duration = info.num_frames / info.sample_rate
                except Exception as e:
                    print(f"Audio info error {utt_id}: {e}")
                    continue

                rows.append({
                    "utt_id": utt_id,
                    "audio_path": str(wav_path),
                    "emotion": emo,
                    "session": session,
                    "gender": gender,
                    "duration_s": round(duration, 3),
                })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {len(df)} rows to {OUT_CSV}")
    print("\nClass distribution:")
    print(df.emotion.value_counts())
    print("\nSession distribution:")
    print(df.session.value_counts().sort_index())
    print(f"\nDuration: min={df.duration_s.min():.2f}s "
          f"max={df.duration_s.max():.2f}s mean={df.duration_s.mean():.2f}s")


if __name__ == "__main__":
    main()
