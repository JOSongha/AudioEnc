#!/usr/bin/env python
"""Build AudioCaps train+val manifests (nubes-direct).

v6 (nubes-only):
    - parquet 은 nubes `users/jos/AudioEnc/AudioCaps/data/{train,validation}-*.parquet`
      에서 stream fetch (audio bytes 추출 안 함, schema 의 caption + youtube_id +
      start_time 컬럼만 읽음)
    - audio 는 nubes `users/jos/AudioEnc/AudioCaps/audio/<youtube_id>_<start_time>.flac`
      에 이미 보존되어 있어 별도 추출 불필요
    - 학습 row 의 audio path 는 `nubes_path` 로 직접 인용

**Leak prevention**: train + validation parquet 만 명시 enumerate. test parquet
(nubes 의 `data/test-*.parquet` 41 file) 절대 안 읽음. test FLAC 도 nubes 에
보존되어 있지만 본 builder 가 enumerate 하지 않으므로 학습 풀에 안 들어감.
nubes-direct 빌더 작성 시 흔한 실수 — `data/*.parquet` 또는 `audio/*.flac` 으로
glob 하면 test 도 들어가 leak 발생. 반드시 split 별 prefix 명시.

Output schema:
    {"modality": "audio_env_sound",
     "source": "audiocaps",
     "nubes_path": "hyperscaleai-audiollm/users/jos/AudioEnc/AudioCaps/audio/<ytid>_<start>.flac",
     "captions": [<caption1>, ...]}

Local fallback: env `AUDIOCAPS_LOCAL_PARQUET_DIR` 설정 시 그 경로의 parquet 사용
(예: `/mnt/tmp/datasets/audiocaps/data`). 이전 v5 까지의 ddn 동작 호환.

Note (이전 v5 까지):
    parquet 의 audio bytes 를 추출해서 ddn 의 `/mnt/tmp/datasets/laion_extracted/
    audiocaps/<ytid>_<start>.flac` 으로 보존, audio_path 로 인용. nubes 에 audio
    가 이미 있어 추출 불필요. 본 builder 는 metadata-only.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

NUBES_GATEWAY = "http://c.nubes.sto.navercorp.com:8000/v1"
BUCKET = "hyperscaleai-audiollm"
NUBES_DATA_PREFIX = "users/jos/AudioEnc/AudioCaps/data"
NUBES_AUDIO_PREFIX = "hyperscaleai-audiollm/users/jos/AudioEnc/AudioCaps/audio"

# train + validation parquet 갯수 (HF OpenSound/AudioCaps).
SPLIT_PARQUET_COUNT = {"train": 412, "validation": 20}
# test 41 parquet 절대 enumerate 안 함.

OUT_DIR_DEFAULT = Path("/mnt/tmp/datasets/manifests/v3")
SHARD_ROWS = 15000


def list_split_parquets(split: str) -> list[str]:
    """Return list of nubes parquet paths for a single split.

    train + validation 만 허용. test 는 KeyError 발생 (leak prevention).
    """
    if split not in SPLIT_PARQUET_COUNT:
        raise ValueError(f"split must be train or validation (test is held out for "
                         f"Stage-2 eval, never enumerated by this builder). got: {split}")
    n = SPLIT_PARQUET_COUNT[split]
    return [f"{NUBES_DATA_PREFIX}/{split}-{i:05d}-of-{n:05d}.parquet" for i in range(n)]


def fetch_parquet_table(nubes_path: str):
    """Fetch single parquet from nubes gateway, return pyarrow Table.

    Local fallback via env `AUDIOCAPS_LOCAL_PARQUET_DIR` (e.g. ddn cache).
    """
    local_dir = os.environ.get("AUDIOCAPS_LOCAL_PARQUET_DIR")
    if local_dir:
        fname = Path(nubes_path).name
        local_path = Path(local_dir) / fname
        if local_path.is_file():
            return pq.read_table(local_path,
                                 columns=["youtube_id", "start_time", "caption"])
    url = f"{NUBES_GATEWAY}/{BUCKET}/{nubes_path}"
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read()
    return pq.read_table(io.BytesIO(body),
                         columns=["youtube_id", "start_time", "caption"])


def process_split(split: str, manifest_prefix: str,
                  out_dir: Path) -> tuple[int, int]:
    """Returns (n_unique_audios, n_shards). nubes-direct."""
    parquets = list_split_parquets(split)
    print(f"[audiocaps:{split}] parquets={len(parquets)} (nubes-direct)", flush=True)

    # key: (youtube_id, start_time) → list of captions
    captions_per_key: dict[tuple[str, int], list[str]] = defaultdict(list)

    n_rows = 0
    n_no_caption = 0

    for i, nubes_path in enumerate(parquets):
        try:
            t = fetch_parquet_table(nubes_path)
        except Exception as e:
            print(f"  [{split}] FAIL fetch {nubes_path}: {e}", flush=True)
            continue
        ytids = t.column("youtube_id").to_pylist()
        starts = t.column("start_time").to_pylist()
        caps = t.column("caption").to_pylist()

        for ytid, start, cap in zip(ytids, starts, caps):
            n_rows += 1
            if not cap or not str(cap).strip():
                n_no_caption += 1
                continue
            key = (str(ytid), int(start))
            captions_per_key[key].append(str(cap).strip())

        if (i + 1) % 50 == 0 or (i + 1) == len(parquets):
            print(f"  [{split}] {i+1}/{len(parquets)} parquets, "
                  f"unique_audios={len(captions_per_key)}", flush=True)

    # build manifest rows: one per unique key
    rows: list[dict] = []
    for (ytid, start), caps in captions_per_key.items():
        nubes_path = f"{NUBES_AUDIO_PREFIX}/{ytid}_{start}.flac"
        rows.append({
            "modality": "audio_env_sound",
            "source": "audiocaps",
            "nubes_path": nubes_path,
            "captions": caps,
        })

    print(
        f"[audiocaps:{split}] rows={n_rows} no_caption={n_no_caption} "
        f"unique_audios={len(rows)}",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = 0
    for i in range(0, len(rows), SHARD_ROWS):
        chunk = rows[i:i + SHARD_ROWS]
        out = out_dir / f"{manifest_prefix}_{n_shards:04d}.jsonl"
        with open(out, "w") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_shards += 1
    print(f"[audiocaps:{split}] wrote {n_shards} shards -> {out_dir}", flush=True)
    if rows:
        print(f"[audiocaps:{split}] sample: {json.dumps(rows[0], ensure_ascii=False)}",
              flush=True)
    return len(rows), n_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT,
                    help="Output manifest directory.")
    args = ap.parse_args()

    train_n, train_s = process_split("train", "audiocaps_train", args.out)
    val_n, val_s = process_split("validation", "audiocaps_val", args.out)
    print(f"[audiocaps] DONE train={train_n} ({train_s} shards), "
          f"val={val_n} ({val_s} shards)", flush=True)


if __name__ == "__main__":
    main()
