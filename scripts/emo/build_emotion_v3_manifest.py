#!/usr/bin/env python
"""T11 — Stage-1 v3 Emotion MCQA manifest (7 datasets, unified canonical 7-class).

Datasets:
  - iemocap       (sehyun shards + kyudan IEMOCAP wavs)
  - meld          (extracted wavs under emotion_raw/MELD/audio/{train,dev,test})
  - cremad        (shkim CREMA-D AudioWAV mirror)
  - dailytalk     (emotion_raw/DailyTalk/dailytalk/data/<dialog>/<utt>_<spk>_d<dialog>.wav)
  - emov_db       (emotion_raw/EmoV-DB/<speaker>/<Emotion>/<file>.wav)
  - mustard       (emotion_raw/MUStARD_Plus_Plus/videos/.../*.mp4 with mustard++_text.csv)
  - ravdess       (emotion_raw/RAVDESS/Actor_*/<filename>.wav)

Canonical 7-class set (FIXED letter order across all sources):
  A. happy   B. sad   C. angry   D. neutral   E. fear   F. disgust   G. surprise

Per § v3 포맷 스펙:
  {"modality":"audio_emotion","source":"<src>","audio_path":"...",
   "question":"What is the emotion expressed?",
   "choices":["A. happy", ...],"answer":"<letter>"}

Output:
  /mnt/tmp/datasets/manifests/v3/emotion_<src>_<NNNN>.jsonl   (15k rows / shard)

Notes on canonical merges:
  - RAVDESS `calm` -> neutral (per task spec)
  - IEMOCAP `excited` -> happy, `frustrated` -> angry; `other` dropped (not mappable)
  - EmoV-DB `Amused` -> happy, `Sleepy` dropped (no canonical bucket)
  - MUStARD uses Explicit_Emotion column; `Excitement` -> happy, `Frustration` -> angry
  - Labels not mappable to canonical 7 are dropped and reported.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

# -----------------------------------------------------------------------------
# Canonical schema
# -----------------------------------------------------------------------------
CANON = ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"]
LETTERS = ["A", "B", "C", "D", "E", "F", "G"]
CHOICES = [f"{L}. {c}" for L, c in zip(LETTERS, CANON)]
ANSWER_OF = {c: L for L, c in zip(LETTERS, CANON)}
QUESTION = "What is the emotion expressed?"

OUT_DIR = Path("/mnt/tmp/datasets/manifests/v3")
SHARD_SIZE = 15000


# -----------------------------------------------------------------------------
# Per-dataset label maps -> canonical, or None to drop
# -----------------------------------------------------------------------------
MELD_MAP = {
    "neutral": "neutral", "joy": "happy", "sadness": "sad", "anger": "angry",
    "fear": "fear", "disgust": "disgust", "surprise": "surprise",
}
DAILYTALK_MAP = {
    "no emotion": "neutral", "happiness": "happy", "sadness": "sad",
    "anger": "angry", "surprise": "surprise", "fear": "fear", "disgust": "disgust",
}
EMOV_MAP = {
    "amused": "happy", "angry": "angry", "disgusted": "disgust",
    "neutral": "neutral",
    # "sleepy" -> drop (no canonical bucket)
}
RAVDESS_LIST = ["neutral", "calm", "happy", "sad", "angry", "fearful",
                "disgust", "surprised"]  # 1..8 indices
RAVDESS_MAP = {
    "neutral": "neutral", "calm": "neutral",  # merged per task spec
    "happy": "happy", "sad": "sad", "angry": "angry", "fearful": "fear",
    "disgust": "disgust", "surprised": "surprise",
}
CREMAD_MAP = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}
IEMOCAP_MAP = {
    "neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
    "fear": "fear", "disgust": "disgust", "surprise": "surprise",
    "excited": "happy", "frustrated": "angry",
    # "other" -> drop
}
MUSTARD_MAP = {
    "Neutral": "neutral", "Happiness": "happy", "Sadness": "sad",
    "Anger": "angry", "Fear": "fear", "Disgust": "disgust",
    "Surprise": "surprise",
    "Excitement": "happy", "Frustration": "angry",
    # "Ridicule" -> drop
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_row(audio_path: str, source: str, canon_label: str) -> dict:
    return {
        "modality": "audio_emotion",
        "source": source,
        "audio_path": audio_path,
        "question": QUESTION,
        "choices": list(CHOICES),
        "answer": ANSWER_OF[canon_label],
    }


def write_shards(rows: list[dict], src: str, split: str = "all") -> int:
    if not rows:
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    f = None
    shard_idx = 0
    for i, row in enumerate(rows):
        if i % SHARD_SIZE == 0:
            if f is not None:
                f.close()
            out_path = OUT_DIR / f"emotion_{src}_{split}_{shard_idx:04d}.jsonl"
            f = open(out_path, "w")
            print(f"[{src}] writing -> {out_path}")
            shard_idx += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
    if f is not None:
        f.close()
    return written


def report(src: str, kept: int, drop_label: int, miss_path: int,
           label_hist: Counter, drop_hist: Counter) -> None:
    print(f"[{src}] kept={kept:,} drop_label={drop_label:,} miss_path={miss_path:,}")
    if label_hist:
        print(f"[{src}]   canonical: {dict(label_hist)}")
    if drop_hist:
        print(f"[{src}]   dropped labels: {dict(drop_hist)}")


# -----------------------------------------------------------------------------
# IEMOCAP — sehyun shards + path remap to /mnt/ddn/kyudan/IEMOCAP/data
# -----------------------------------------------------------------------------
def build_iemocap() -> int:
    src = "iemocap"
    shards = [
        "/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/emotion/"
        f"iemocap_shard_{i:05d}.jsonl" for i in range(3)
    ]
    REMAP_FROM = "/mnt/ddn/users/sehyun/CACHE/iemocap/wav/"
    REMAP_TO = "/mnt/ddn/kyudan/IEMOCAP/data/"
    rows_out, label_hist, drop_hist = [], Counter(), Counter()
    drop_label = miss_path = 0
    for sh in shards:
        if not os.path.exists(sh):
            print(f"[{src}] WARN missing shard: {sh}")
            continue
        with open(sh) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                raw = r.get("response", "")
                canon = IEMOCAP_MAP.get(raw)
                if canon is None:
                    drop_label += 1
                    drop_hist[raw] += 1
                    continue
                ap = r.get("audio_path", "")
                if ap.startswith(REMAP_FROM):
                    ap = REMAP_TO + ap[len(REMAP_FROM):]
                if not os.path.exists(ap):
                    miss_path += 1
                    continue
                rows_out.append(make_row(ap, src, canon))
                label_hist[canon] += 1
    write_shards(rows_out, src, "all")
    report(src, len(rows_out), drop_label, miss_path, label_hist, drop_hist)
    return len(rows_out)


# -----------------------------------------------------------------------------
# MELD — train/dev wavs + sent_emo CSVs (test held out per existing convention,
#         but task asks for full datasets; we include train+dev+test so eval set
#         is decided downstream)
# -----------------------------------------------------------------------------
def build_meld() -> int:
    src = "meld"
    import pandas as pd
    BASE = Path("/mnt/tmp/datasets/emotion_raw/MELD")
    splits = {"train": "train_splits", "dev": "dev_splits_complete",
              "test": "output_repeated_splits_test"}
    audio_root = BASE / "audio"   # extracted wavs by extract_meld_audio.sh
    rows_out, label_hist, drop_hist = [], Counter(), Counter()
    drop_label = miss_path = 0
    total_kept = 0
    for split, _vid_dir in splits.items():
        csvp = BASE / "MELD.Raw" / f"{split if split != 'test' else 'test'}_sent_emo.csv"
        if not csvp.exists():
            print(f"[{src}] WARN missing csv: {csvp}")
            continue
        df = pd.read_csv(csvp)
        split_rows = []
        for _, r in df.iterrows():
            stem = f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}"
            wav = audio_root / split / f"{stem}.wav"
            raw = str(r["Emotion"])
            canon = MELD_MAP.get(raw)
            if canon is None:
                drop_label += 1
                drop_hist[raw] += 1
                continue
            if not wav.exists():
                miss_path += 1
                continue
            split_rows.append(make_row(str(wav), src, canon))
            label_hist[canon] += 1
        write_shards(split_rows, src, split)
        total_kept += len(split_rows)
        rows_out.extend(split_rows)
    report(src, total_kept, drop_label, miss_path, label_hist, drop_hist)
    return total_kept


# -----------------------------------------------------------------------------
# CREMA-D — shkim mirror, filename-based label
# -----------------------------------------------------------------------------
def build_cremad() -> int:
    src = "crema_d"
    ROOT = Path(
        "/mnt/ddn/shkim/omni_acoustic_bench_v4_train_evaluator/"
        "train_eval_model/dataset_emotion/Source/CREMA-D/AudioWAV"
    )
    if not ROOT.exists():
        print(f"[{src}] ERROR not found: {ROOT}")
        return 0
    rows_out, label_hist, drop_hist = [], Counter(), Counter()
    drop_label = miss_path = 0
    wavs = sorted(ROOT.glob("*.wav"))
    print(f"[{src}] scanning {len(wavs):,} wavs in {ROOT}")
    for wp in wavs:
        parts = wp.stem.split("_")
        if len(parts) < 3:
            drop_label += 1
            drop_hist["BAD_NAME"] += 1
            continue
        canon = CREMAD_MAP.get(parts[2])
        if canon is None:
            drop_label += 1
            drop_hist[parts[2]] += 1
            continue
        if not wp.exists():
            miss_path += 1
            continue
        rows_out.append(make_row(str(wp), src, canon))
        label_hist[canon] += 1
    write_shards(rows_out, src, "all")
    report(src, len(rows_out), drop_label, miss_path, label_hist, drop_hist)
    return len(rows_out)


# -----------------------------------------------------------------------------
# DailyTalk — metadata.json keyed by (dialog_id, utt_id)
# -----------------------------------------------------------------------------
def build_dailytalk() -> int:
    src = "dailytalk"
    BASE = Path("/mnt/tmp/datasets/emotion_raw/DailyTalk/dailytalk")
    meta_p = BASE / "metadata.json"
    data_root = BASE / "data"
    if not meta_p.exists():
        print(f"[{src}] ERROR no metadata: {meta_p}")
        return 0
    meta = json.loads(meta_p.read_text())
    rows_out, label_hist, drop_hist = [], Counter(), Counter()
    drop_label = miss_path = 0
    for dialog_id, utts in meta.items():
        for utt_id, row in utts.items():
            speaker = int(row.get("speaker", 0))
            stem = f"{int(utt_id)}_{int(speaker)}_d{int(dialog_id)}"
            wav = data_root / str(int(dialog_id)) / f"{stem}.wav"
            raw = str(row.get("emotion", "no emotion"))
            canon = DAILYTALK_MAP.get(raw)
            if canon is None:
                drop_label += 1
                drop_hist[raw] += 1
                continue
            if not wav.exists():
                miss_path += 1
                continue
            rows_out.append(make_row(str(wav), src, canon))
            label_hist[canon] += 1
    write_shards(rows_out, src, "all")
    report(src, len(rows_out), drop_label, miss_path, label_hist, drop_hist)
    return len(rows_out)


# -----------------------------------------------------------------------------
# EmoV-DB — directory-coded labels
# -----------------------------------------------------------------------------
def build_emov_db() -> int:
    src = "emov_db"
    ROOT = Path("/mnt/tmp/datasets/emotion_raw/EmoV-DB")
    if not ROOT.exists():
        print(f"[{src}] ERROR not found: {ROOT}")
        return 0
    rows_out, label_hist, drop_hist = [], Counter(), Counter()
    drop_label = miss_path = 0
    # speaker dirs at top-level: bea, jenie, josh, sam (skip 'repo')
    speakers = [d for d in ROOT.iterdir()
                if d.is_dir() and d.name.lower() in {"bea", "jenie", "josh", "sam"}]
    for spk in speakers:
        for emo_dir in spk.iterdir():
            if not emo_dir.is_dir():
                continue
            raw = emo_dir.name.lower()
            canon = EMOV_MAP.get(raw)
            for wp in sorted(emo_dir.glob("*.wav")):
                if canon is None:
                    drop_label += 1
                    drop_hist[raw] += 1
                    continue
                if not wp.exists():
                    miss_path += 1
                    continue
                rows_out.append(make_row(str(wp), src, canon))
                label_hist[canon] += 1
    write_shards(rows_out, src, "all")
    report(src, len(rows_out), drop_label, miss_path, label_hist, drop_hist)
    return len(rows_out)


# -----------------------------------------------------------------------------
# MUStARD — videos/*.mp4 + mustard++_text.csv (Explicit_Emotion)
# -----------------------------------------------------------------------------
def build_mustard() -> int:
    src = "mustard"
    BASE = Path("/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus")
    csvp = BASE / "mustard++_text.csv"
    if not csvp.exists():
        print(f"[{src}] ERROR no csv: {csvp}")
        return 0
    rows_out, label_hist, drop_hist = [], Counter(), Counter()
    drop_label = miss_path = 0
    vids_aug = BASE / "videos" / "augmented_utterance"
    vids_ctx = BASE / "videos" / "final_context_videos"
    with open(csvp) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            raw = (r.get("Explicit_Emotion") or "").strip()
            if not raw:
                continue  # unlabeled context row
            canon = MUSTARD_MAP.get(raw)
            if canon is None:
                drop_label += 1
                drop_hist[raw] += 1
                continue
            key = r["KEY"]
            cand = vids_aug / f"{key}.mp4"
            if not cand.exists():
                cand = vids_ctx / f"{key}.mp4"
            if not cand.exists():
                miss_path += 1
                continue
            rows_out.append(make_row(str(cand), src, canon))
            label_hist[canon] += 1
    write_shards(rows_out, src, "all")
    report(src, len(rows_out), drop_label, miss_path, label_hist, drop_hist)
    return len(rows_out)


# -----------------------------------------------------------------------------
# RAVDESS — filename-coded label (3rd field)
# -----------------------------------------------------------------------------
_RAV_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")

def build_ravdess() -> int:
    src = "ravdess"
    ROOT = Path("/mnt/tmp/datasets/emotion_raw/RAVDESS")
    if not ROOT.exists():
        print(f"[{src}] ERROR not found: {ROOT}")
        return 0
    rows_out, label_hist, drop_hist = [], Counter(), Counter()
    drop_label = miss_path = 0
    for wp in sorted(ROOT.rglob("*.wav")):
        m = _RAV_RE.search(wp.stem)
        if not m:
            drop_label += 1
            drop_hist["BAD_NAME"] += 1
            continue
        idx = int(m.group(3))
        if not 1 <= idx <= 8:
            drop_label += 1
            drop_hist[f"IDX_{idx}"] += 1
            continue
        raw = RAVDESS_LIST[idx - 1]
        canon = RAVDESS_MAP.get(raw)
        if canon is None:
            drop_label += 1
            drop_hist[raw] += 1
            continue
        if not wp.exists():
            miss_path += 1
            continue
        rows_out.append(make_row(str(wp), src, canon))
        label_hist[canon] += 1
    write_shards(rows_out, src, "all")
    report(src, len(rows_out), drop_label, miss_path, label_hist, drop_hist)
    return len(rows_out)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
DATASETS = {
    "iemocap":  build_iemocap,
    "meld":     build_meld,
    "crema_d":  build_cremad,
    "dailytalk": build_dailytalk,
    "emov_db":  build_emov_db,
    "mustard":  build_mustard,
    "ravdess":  build_ravdess,
}


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(DATASETS.keys()) + ["all"],
                    default=["all"])
    args = ap.parse_args(argv)
    targets = list(DATASETS.keys()) if "all" in args.only else args.only

    print(f"# canonical 7-class: {CANON}")
    print(f"# choices (fixed):  {CHOICES}")
    print(f"# question:         {QUESTION!r}")
    print(f"# OUT_DIR:          {OUT_DIR}")
    print(f"# targets:          {targets}\n")

    grand = {}
    for name in targets:
        print(f"\n=== {name} ===")
        try:
            grand[name] = DATASETS[name]()
        except Exception as e:
            import traceback
            traceback.print_exc()
            grand[name] = -1
            print(f"[{name}] FAILED: {e}")

    print("\n========== SUMMARY ==========")
    total = 0
    for k, v in grand.items():
        marker = "OK " if v >= 0 else "ERR"
        print(f"  {marker}  {k:<10} kept={v}")
        if v > 0:
            total += v
    print(f"  TOTAL kept: {total:,}")


if __name__ == "__main__":
    main()
