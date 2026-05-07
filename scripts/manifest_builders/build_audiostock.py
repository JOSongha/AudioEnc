#!/usr/bin/env python
"""Build LAION Audiostock manifest from nubes train.jsonl + test.jsonl.

v5 까지는 ddn 의 `/mnt/tmp/datasets/laion_audiostock/{audio/, meta.csv}` 를
사용했으나 LAION 공식 ~10K 중 ddn 다운로드 단계에서 ~860 fail (로컬 audio
9,139 만 보존). v6 부터 nubes 의 LAION 공식 본 (10K) 직접 인용:

    nubes train.jsonl  (9,000 row)
    nubes test.jsonl     (999 row)
    합 9,999 row (LAION 표준 ~10K 수준)

Output schema:
    {"modality": "audio_env_sound",
     "source": "laion_audiostock",
     "nubes_path": "hyperscaleai-audiollm/datasets/public/LAION-Audio-630k/audiostock/audio/train/<id>.flac",
     "captions": ["<caption>"]}

omni_dataset.py 가 `load_from_nubes=True` 시 nubes_path 로 fetch. 다른
audio_env_sound source 는 여전히 audio_path (로컬) 사용 — row 별 mix OK.

No /mnt/ddn/users/<person>/ caches. No /mnt/ddn/omni_dataset/audio/...
caches. Only nubes.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

NUBES_GATEWAY = "http://c.nubes.sto.navercorp.com:8000/v1"
BUCKET = "hyperscaleai-audiollm"
TRAIN_JSONL = "datasets/public/LAION-Audio-630k/audiostock/train.jsonl"
TEST_JSONL = "datasets/public/LAION-Audio-630k/audiostock/test.jsonl"

# s3 fileuri prefix → nubes prefix 변환 (LAION-Audio s3 mirror 의 path 구조).
_S3_PREFIX = "s3://hcx-audio/public/LAION-Audio/audiostock/"
_NUBES_PREFIX = f"{BUCKET}/datasets/public/LAION-Audio-630k/audiostock/"

OUT_PATH = Path("/mnt/tmp/datasets/manifests/v3/laion_audiostock_0000.jsonl")


def s3_to_nubes(fileuri: str) -> str | None:
    """nubes train.jsonl 의 s3://hcx-audio/.../audiostock/audio/train/<id>.flac
    → hyperscaleai-audiollm/datasets/public/LAION-Audio-630k/audiostock/audio/train/<id>.flac."""
    if not fileuri.startswith(_S3_PREFIX):
        return None
    return _NUBES_PREFIX + fileuri[len(_S3_PREFIX):]


def stream_nubes_jsonl(path: str):
    """Yield each row from a nubes-hosted jsonl via HTTP gateway."""
    url = f"{NUBES_GATEWAY}/{BUCKET}/{path}"
    with urllib.request.urlopen(url, timeout=120.0) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_row(row: dict) -> dict | None:
    """Extract (nubes_path, caption) from one HCX-style sound_caption row."""
    msgs = row.get("messages", [])
    if len(msgs) < 3:
        return None
    user = msgs[1]
    assistant = msgs[2]
    audio_uri = None
    for c in user.get("content", []):
        if c.get("type") == "audio":
            audio_uri = c.get("fileuri")
            break
    if audio_uri is None:
        return None
    caption = None
    for c in assistant.get("content", []):
        if c.get("type") == "text":
            caption = (c.get("text") or "").strip()
            break
    if not caption:
        return None
    nubes_path = s3_to_nubes(audio_uri)
    if nubes_path is None:
        return None
    return {
        "modality": "audio_env_sound",
        "source": "laion_audiostock",
        "nubes_path": nubes_path,
        "captions": [caption],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_train = n_test = n_skip = 0
    out_rows: list[dict] = []

    print(f"[audiostock] streaming nubes:{TRAIN_JSONL} ...", flush=True)
    for row in stream_nubes_jsonl(TRAIN_JSONL):
        out = parse_row(row)
        if out is None:
            n_skip += 1
            continue
        out_rows.append(out)
        n_train += 1

    print(f"[audiostock] streaming nubes:{TEST_JSONL} ...", flush=True)
    for row in stream_nubes_jsonl(TEST_JSONL):
        out = parse_row(row)
        if out is None:
            n_skip += 1
            continue
        out_rows.append(out)
        n_test += 1

    total = n_train + n_test
    print(f"[audiostock] train={n_train}, test={n_test}, total={total}, skipped={n_skip}", flush=True)

    if not args.dry_run:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[audiostock] DONE wrote={total} -> {args.out}", flush=True)
    else:
        print(f"[audiostock] DRY RUN (no file written)", flush=True)


if __name__ == "__main__":
    main()
