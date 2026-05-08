#!/usr/bin/env python
"""Build MELD emotion MCQA manifest from nubes (nubes-direct).

Source:
    csv:   nubes /users/jos/AudioEnc/MELD/CSV/{train,dev}_sent_emo.csv
    audio: nubes /users/jos/AudioEnc/MELD/audio/{train,dev}/dia<N>_utt<M>.wav

train+dev 만 학습. test 는 Stage-2 eval (eval_source_emotion.load_meld_test)
가 별도 사용.

v6 부터 csv + audio 모두 nubes 직접 인용. 2026-05-08 까지는 audio 가 nubes
public dir 의 mp3 를 가리켰으나 (`/MELD.Raw/<split>/<stem>.mp3`) torchaudio
default backend (libsndfile) 가 mp3 디코드 inconsistent — wav 본을 사용자
영역에 직접 업로드 (`/users/jos/AudioEnc/MELD/audio/<split>/<stem>.wav`,
§ 12.13).

Output schema:
    {"modality": "audio_emotion",
     "source": "meld",
     "nubes_path": "hyperscaleai-audiollm/users/jos/AudioEnc/MELD/audio/<split>/dia<N>_utt<M>.wav",
     "question": "...",
     "choices": ["A. ...", ...],
     "answer": "<letter>"}
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import urllib.parse
import urllib.request
from pathlib import Path

NUBES_GATEWAY = "http://c.nubes.sto.navercorp.com:8000/v1"
BUCKET = "hyperscaleai-audiollm"

CSV_PATHS = {
    "train": "users/jos/AudioEnc/MELD/CSV/train_sent_emo.csv",
    "dev":   "users/jos/AudioEnc/MELD/CSV/dev_sent_emo.csv",
}
AUDIO_DIR_PATHS = {
    "train": "users/jos/AudioEnc/MELD/audio/train",
    "dev":   "users/jos/AudioEnc/MELD/audio/dev",
}
NUBES_PREFIX = f"{BUCKET}/users/jos/AudioEnc/MELD/audio"

OUT_DIR = Path(os.environ.get("V6_OUT", "/mnt/tmp/datasets/manifests/v6_raw"))
SHARD_SIZE = 15000
QUESTION = "What emotion is expressed in this audio clip?"
SOURCE = "meld"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def fetch_csv(nubes_path: str) -> str:
    url = f"{NUBES_GATEWAY}/{BUCKET}/{nubes_path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def list_audio_set(prefix: str) -> set[str]:
    """List all .wav basenames (stem) under a nubes audio dir, with pagination."""
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    out: set[str] = set()
    token = None
    while True:
        params = {"dir": prefix, "max-contents": "1000"}
        if token:
            params["continuation-token"] = token
        url = f"{NUBES_GATEWAY}/{BUCKET}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read()
            headers = r.headers
        try:
            entries = json.loads(body) if body else []
        except json.JSONDecodeError:
            entries = []
        if not entries:
            break
        for e in entries:
            if not e.get("IsDir") and e["Name"].endswith(".wav"):
                out.add(e["Name"][:-4])  # strip ".wav"
        token = headers.get("X-Continuation-Token")
        if not token:
            break
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, str]] = []  # (nubes_path, label)
    skipped_missing = 0
    skipped_unknown = 0

    for split in ["train", "dev"]:
        print(f"[meld] fetching csv: {CSV_PATHS[split]}", flush=True)
        csv_text = fetch_csv(CSV_PATHS[split])

        print(f"[meld] listing audio: {AUDIO_DIR_PATHS[split]}", flush=True)
        audio_stems = list_audio_set(AUDIO_DIR_PATHS[split])
        audio_subdir = AUDIO_DIR_PATHS[split].rsplit("/", 1)[-1]
        print(f"[meld] {split}: csv parsed, audio stems={len(audio_stems)}", flush=True)

        for r in csv.DictReader(io.StringIO(csv_text)):
            label = (r.get("Emotion") or "").strip().lower()
            if not label:
                skipped_unknown += 1
                continue
            try:
                stem = f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}"
            except (KeyError, ValueError):
                skipped_unknown += 1
                continue
            if stem not in audio_stems:
                skipped_missing += 1
                continue
            nubes_path = f"{NUBES_PREFIX}/{audio_subdir}/{stem}.wav"
            items.append((nubes_path, label))

    classes = sorted({lab for _, lab in items})
    print(f"[meld] kept={len(items)} skipped_missing={skipped_missing} "
          f"skipped_unknown={skipped_unknown} classes={len(classes)}: {classes}",
          flush=True)

    out_rows = []
    for idx, (np_, label) in enumerate(items):
        rng = random.Random(42 + idx)
        shuffled = list(classes)
        rng.shuffle(shuffled)
        choices = [f"{LETTERS[i]}. {c}" for i, c in enumerate(shuffled)]
        answer = LETTERS[shuffled.index(label)]
        out_rows.append({
            "modality": "audio_emotion",
            "source": SOURCE,
            "nubes_path": np_,
            "question": QUESTION,
            "choices": choices,
            "answer": answer,
        })

    n = 0
    shard_idx = 0
    f = None
    for i, row in enumerate(out_rows):
        if i % SHARD_SIZE == 0:
            if f is not None:
                f.close()
            out_path = OUT_DIR / f"emotion_meld_{shard_idx:04d}.jsonl"
            f = open(out_path, "w")
            print(f"[meld] writing -> {out_path}", flush=True)
            shard_idx += 1
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    if f is not None:
        f.close()
    print(f"[meld] DONE wrote={n} classes={len(classes)}", flush=True)


if __name__ == "__main__":
    main()
