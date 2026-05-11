#!/usr/bin/env python
"""Build MACS manifest from MACS.yaml + nubes audio (nubes-direct).

MACS = TAU Urban Acoustic Scenes 2019 development 의 일부 scene
(airport / park / public_square) 의 3,930 audio 에 사람이 caption 을 붙인
캡션 데이터셋. nubes `/datasets/public/MACS/audio/` 에 14,400 audio (TAU2019
의 모든 scene) 가 보존되어 있고, 그중 MACS yaml 이 가리키는 3,930 만 v6
학습 풀에 사용.

Inputs (nubes-only):
    nubes /users/jos/AudioEnc/MACS/MACS.yaml  (2.7 MB, 3,930 entry)
        — 각 entry 에 filename + 2-5 annotator caption
        — ddn 로컬 (`/mnt/tmp/datasets/env_sound/MACS/MACS.yaml`) 도 동일 내용
          이라 fallback 으로 인용 가능 (env var `MACS_YAML_LOCAL`).

Output schema:
    {"modality": "audio_env_sound",
     "source": "macs",
     "nubes_path": "hyperscaleai-audiollm/datasets/public/MACS/audio/<fname>.wav",
     "captions": [<caption1>, <caption2>, ...]}

Note (이전 v5 까지):
    yaml 의 fname 을 ddn 의 zip download (TAU2019 21 zip, 35+ GB) 으로
    추출해서 `/mnt/tmp/datasets/laion_extracted/macs/<fname>.wav` 보존,
    그것을 audio_path 로 인용. nubes audio 가 이미 있어서 불필요.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import urllib.request
from pathlib import Path

import yaml

MACS_YAML_NUBES_URL = (
    "http://c.nubes.sto.navercorp.com:8000/v1/"
    "hyperscaleai-audiollm/users/jos/AudioEnc/MACS/MACS.yaml"
)
MACS_YAML_LOCAL_FALLBACK = Path("/mnt/tmp/datasets/env_sound/MACS/MACS.yaml")
OUT_PATH = Path("/mnt/tmp/datasets/manifests/v3/macs_0000.jsonl")
NUBES_PREFIX = "hyperscaleai-audiollm/datasets/public/MACS/audio/"


def _fetch_yaml() -> dict:
    """Fetch MACS.yaml from nubes (default), with ddn local fallback if env var set."""
    fallback = os.environ.get("MACS_YAML_LOCAL")
    if fallback and Path(fallback).is_file():
        with open(fallback) as f:
            return yaml.safe_load(f)
    try:
        with urllib.request.urlopen(MACS_YAML_NUBES_URL, timeout=60) as r:
            return yaml.safe_load(io.BytesIO(r.read()))
    except Exception as e:
        if MACS_YAML_LOCAL_FALLBACK.is_file():
            print(f"[macs] nubes yaml fetch failed ({e}), falling back to "
                  f"{MACS_YAML_LOCAL_FALLBACK}", flush=True)
            with open(MACS_YAML_LOCAL_FALLBACK) as f:
                return yaml.safe_load(f)
        raise


def load_targets() -> dict[str, list[str]]:
    """Returns {basename.wav: [caption1, caption2, ...]} from MACS.yaml."""
    d = _fetch_yaml()
    out: dict[str, list[str]] = {}
    for entry in d["files"]:
        fname = entry["filename"]
        caps = []
        for a in entry.get("annotations", []):
            s = a.get("sentence", "").strip()
            if s:
                caps.append(s)
        if caps:
            out[fname] = caps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    targets = load_targets()
    print(f"[macs] yaml entries: {len(targets)}", flush=True)

    rows = []
    for fname, caps in targets.items():
        rows.append({
            "modality": "audio_env_sound",
            "source": "macs",
            "nubes_path": NUBES_PREFIX + fname,
            "captions": caps,
        })

    print(f"[macs] manifest rows: {len(rows)}", flush=True)

    if not args.dry_run:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[macs] DONE wrote={len(rows)} -> {args.out}", flush=True)
    else:
        print(f"[macs] DRY RUN (no file written)", flush=True)


if __name__ == "__main__":
    main()
