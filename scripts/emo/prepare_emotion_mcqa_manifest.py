"""Emotion MCQA manifest (Option C — data-driven templating).

Takes the base training manifest (path, source, relpath) and emits a row-level
MCQA-ready manifest: each row carries fully-rendered question / choices list /
letter-answer so the generic ChatML template can render without per-source
branching in `omni_dataset.py`.

Output row schema:
    {
      "path":       str,                 # absolute wav path
      "source":     str,                 # "MELD" | "DailyTalk" | "EmoV-DB" | "RAVDESS" | "MUStARD"
      "modality":   "audio_emotion",
      "question":   str,                 # "What emotion does the speaker convey?"
      "choices":    ["A) neutral", ...], # per-source emotion set, lettered
      "answer":     "A" | "B" | ...,
      "label":      str,                 # raw label (for logging)
      "transcript": str | None,          # optional; when available
      "rationale":  str | None,          # filled by a later offline pass
    }

Held-out splits are already enforced upstream by build_training_manifest.py, so
we inherit the file list from train_manifest.jsonl.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

BASE_MANIFEST = Path("/mnt/tmp/listen_analysis/train_manifest/train_manifest.jsonl")
OUT = Path("/mnt/tmp/listen_analysis/train_manifest/emotion_mcqa_manifest.jsonl")
RAW = Path("/mnt/tmp/datasets/emotion_raw")

QUESTION = "What emotion does the speaker convey?"


# ----- per-corpus label maps -------------------------------------------------
MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

DAILYTALK_EMOTIONS = ["no emotion", "happiness", "sadness", "anger", "surprise",
                      "fear", "disgust"]

EMOV_EMOTIONS = ["amused", "angry", "disgusted", "neutral", "sleepy"]

RAVDESS_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful",
                    "disgust", "surprised"]   # indices 1..8

MUSTARD_CHOICES = ["sarcastic", "not sarcastic"]

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def mcqa_row(path: str, source: str, label: str, choices: list[str],
             transcript: str | None = None) -> dict | None:
    """Compose the output row. Returns None if label isn't in choices."""
    if label not in choices:
        return None
    idx = choices.index(label)
    lettered = [f"{LETTERS[i]}) {c}" for i, c in enumerate(choices)]
    return {
        "path": path,
        "source": source,
        "modality": "audio_emotion",
        "question": QUESTION,
        "choices": lettered,
        "answer": LETTERS[idx],
        "label": label,
        "transcript": transcript,
        "rationale": None,   # filled by a later offline pass
    }


# ----- MELD ------------------------------------------------------------------
def build_meld_index() -> dict[str, dict]:
    """Return {wav_stem: {emotion, utterance}} for train+dev (test held out)."""
    idx = {}
    for split in ("train", "dev"):
        csvp = RAW / "MELD" / "MELD.Raw" / f"{split}_sent_emo.csv"
        if not csvp.exists():
            print(f"  [MELD] {csvp} not found — skipping MELD")
            return {}
        df = pd.read_csv(csvp)
        for _, r in df.iterrows():
            stem = f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}"
            idx[stem] = {"emotion": str(r["Emotion"]), "utterance": str(r["Utterance"])}
    return idx


# ----- DailyTalk -------------------------------------------------------------
_DAILYTALK_META_PATH = RAW / "DailyTalk" / "dailytalk" / "metadata.json"

def build_dailytalk_index() -> dict[str, dict]:
    """Return {wav_stem: {emotion, text}} keyed on the filename stem used by
    DailyTalk's on-disk layout (e.g. '0_1_d0'). metadata.json is keyed by
    (dialog_id, utterance_id)."""
    idx = {}
    if not _DAILYTALK_META_PATH.exists():
        print(f"  [DailyTalk] {_DAILYTALK_META_PATH} not found — skipping DailyTalk")
        return {}
    meta = json.loads(_DAILYTALK_META_PATH.read_text())
    for dialog_id, utts in meta.items():
        for utt_id, row in utts.items():
            speaker = row.get("speaker", 0)
            # filename convention observed on disk: "{utt_id}_{speaker}_d{dialog_id}.wav"
            stem = f"{int(utt_id)}_{int(speaker)}_d{int(dialog_id)}"
            idx[stem] = {"emotion": row.get("emotion", "no emotion"),
                         "text": row.get("text", "")}
    return idx


# ----- EmoV-DB ---------------------------------------------------------------
def emov_emotion_from_path(path: str) -> str | None:
    # Two layouts observed in the wild:
    #   (a) /EmoV-DB/<speaker>/<Emotion>/<file>.wav  (subdirectory layout)
    #   (b) /EmoV-DB/<emotion>_<range>_<num>.wav     (flat OpenSLR 115 layout)
    # Filename prefixes observed: amused, anger, disgust, Disgust, neutral, Neutral, sleepiness
    EMOV_EMOTIONS = {"amused", "angry", "disgusted", "neutral", "sleepy"}
    PREFIX_MAP = {
        "amused": "amused", "anger": "angry", "angry": "angry",
        "disgust": "disgusted", "disgusted": "disgusted",
        "neutral": "neutral", "sleepy": "sleepy", "sleepiness": "sleepy",
    }
    parts = path.split("/")
    # Try subdirectory layout first (parent dir = emotion)
    try:
        emotion_dir = parts[-2].lower()
        if emotion_dir in EMOV_EMOTIONS:
            return emotion_dir
        if emotion_dir in PREFIX_MAP:
            return PREFIX_MAP[emotion_dir]
    except IndexError:
        pass
    # Fall back to filename prefix (e.g. "amused_1-15_0001.wav" → "amused")
    stem = Path(path).stem.lower()
    for prefix, canonical in PREFIX_MAP.items():
        if stem.startswith(prefix):
            return canonical
    return None


# ----- RAVDESS ---------------------------------------------------------------
_RAV_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")

def ravdess_emotion_from_path(path: str) -> str | None:
    m = _RAV_RE.search(Path(path).stem)
    if not m:
        return None
    emo_idx = int(m.group(3))  # 3rd field = emotion 01-08
    if 1 <= emo_idx <= 8:
        return RAVDESS_EMOTIONS[emo_idx - 1]
    return None


# ----- MUStARD ---------------------------------------------------------------
def build_mustard_index() -> dict[str, dict]:
    csvp = RAW / "MUStARD_Plus_Plus" / "mustard++_text.csv"
    idx = {}
    if not csvp.exists():
        print(f"  [MUStARD] {csvp} not found — skipping MUStARD")
        return {}
    with csvp.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            key = r["KEY"]
            sar_raw = r.get("Sarcasm", "").strip()
            if not sar_raw:
                continue
            try:
                sar = int(sar_raw)
            except ValueError:
                continue
            idx[key] = {"sarcasm": sar, "sentence": r.get("SENTENCE", "")}
    return idx


def mustard_label_from_path(path: str, index: dict[str, dict]) -> tuple[str | None, str | None]:
    """MUStARD filename {scene}_{u|c}.mp4 maps to CSV key {scene}_u (utterance, labeled)
    or {scene}_c_## (context rows, Sarcasm column empty → we drop them here).
    Only `*_u.mp4` files carry a sarcasm label in the CSV.
    """
    stem = Path(path).stem
    if not stem.endswith("_u"):
        return (None, None)
    row = index.get(stem)
    if row is None:
        return (None, None)
    return ("sarcastic" if row["sarcasm"] else "not sarcastic", row["sentence"])


# ----- main ------------------------------------------------------------------
def main() -> None:
    meld_idx = build_meld_index()
    daily_idx = build_dailytalk_index()
    mustard_idx = build_mustard_index()

    out_rows = []
    stats = {"total_in": 0, "total_out": 0, "by_src": {}}
    missed_label = {"by_src": {}}

    with BASE_MANIFEST.open() as f:
        for line in f:
            base = json.loads(line)
            stats["total_in"] += 1
            src = base["source"]
            path = base["path"]
            row = None

            if src == "MELD":
                stem = Path(path).stem
                m = meld_idx.get(stem)
                if m:
                    row = mcqa_row(path, src, m["emotion"], MELD_EMOTIONS,
                                   transcript=m["utterance"])
            elif src == "DailyTalk":
                stem = Path(path).stem
                m = daily_idx.get(stem)
                if m:
                    row = mcqa_row(path, src, m["emotion"], DAILYTALK_EMOTIONS,
                                   transcript=m.get("text"))
            elif src == "EmoV-DB":
                emo = emov_emotion_from_path(path)
                if emo:
                    row = mcqa_row(path, src, emo, EMOV_EMOTIONS)
            elif src == "RAVDESS":
                emo = ravdess_emotion_from_path(path)
                if emo:
                    row = mcqa_row(path, src, emo, RAVDESS_EMOTIONS)
            elif src == "MUStARD":
                label, sentence = mustard_label_from_path(path, mustard_idx)
                if label:
                    row = mcqa_row(path, src, label, MUSTARD_CHOICES, transcript=sentence)
            else:
                continue

            if row is None:
                missed_label["by_src"][src] = missed_label["by_src"].get(src, 0) + 1
                continue
            out_rows.append(row)
            stats["by_src"][src] = stats["by_src"].get(src, 0) + 1
            stats["total_out"] += 1

    with OUT.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"in:  {stats['total_in']:,}")
    print(f"out: {stats['total_out']:,}")
    print("by source:")
    for k, v in sorted(stats["by_src"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12} {v:>6,}")
    if missed_label["by_src"]:
        print("\n⚠ rows with label-not-found (skipped):")
        for k, v in missed_label["by_src"].items():
            print(f"  {k:<12} {v:>6,}")
    print(f"\nmanifest: {OUT}")


if __name__ == "__main__":
    main()
