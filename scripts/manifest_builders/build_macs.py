#!/usr/bin/env python
"""Build MACS manifest from MACS.yaml + nubes audio (nubes-direct).

MACS = TAU Urban Acoustic Scenes 2019 development 의 일부 scene
(airport / park / public_square) 의 3,930 audio 에 사람이 caption 을 붙인
캡션 데이터셋. nubes `/datasets/public/MACS/audio/` 에 14,400 audio (TAU2019
의 모든 scene) 가 보존되어 있고, 그중 MACS yaml 이 가리키는 3,930 만 v6
학습 풀에 사용.

Inputs:
    /mnt/tmp/datasets/env_sound/MACS/MACS.yaml
        — 3,930 entry, 각 entry 에 filename + 2-5 annotator caption
        — yaml 자체는 작은 metadata 라 ddn 의존 유지 (또는 향후 nubes 에 직접
          올려서 nubes-only 화 가능)

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
import json
from pathlib import Path

import yaml

MACS_YAML = Path("/mnt/tmp/datasets/env_sound/MACS/MACS.yaml")
OUT_PATH = Path("/mnt/tmp/datasets/manifests/v3/macs_0000.jsonl")
NUBES_PREFIX = "hyperscaleai-audiollm/datasets/public/MACS/audio/"


def load_targets() -> dict[str, list[str]]:
    """Returns {basename.wav: [caption1, caption2, ...]} from MACS.yaml."""
    with open(MACS_YAML) as f:
        d = yaml.safe_load(f)
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
