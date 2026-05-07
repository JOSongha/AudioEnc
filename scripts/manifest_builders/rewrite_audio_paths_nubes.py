#!/usr/bin/env python
"""Rewrite v6 manifest rows: add `nubes_path` field from `audio_path` prefix swap.

`omni_dataset.py` 가 `load_from_nubes=True` + `nubes_path` 있으면 nubes 게이트웨이
fetch, 없거나 fetch 실패 시 `audio_path` fallback. 이 스크립트는 v6 의 모든 shard
를 스캔해서 source-별 prefix mapping 으로 nubes path 를 row 에 추가.

Source-별 nubes 위치 (2026-05-07 시점, /users/jos/AudioEnc/ 업로드 + nubes 기존
public 영역):

| Source         | audio_path prefix                                              | → nubes_path prefix                                                          |
|----------------|---------------------------------------------------------------|------------------------------------------------------------------------------|
| audioset       | /mnt/tmp/datasets/env_sound/AudioSet/audio/                    | hyperscaleai-audiollm/users/jos/AudioEnc/AudioSet/audio/                     |
| laion_bbc      | /mnt/tmp/datasets/laion_extracted/bbc/                         | hyperscaleai-audiollm/users/jos/AudioEnc/LAION-BBC/audio/                    |
| laion_epidemic | /mnt/tmp/datasets/laion_extracted/epidemic/                    | hyperscaleai-audiollm/datasets/public/LAION-Audio-630k/epidemic_sound_effects/audio/ |
| fsd50k         | /mnt/tmp/datasets/env_sound/FSD50K/FSD50K.dev_audio/           | hyperscaleai-audiollm/datasets/public/FSD50K/audio/                          |
| clotho         | /mnt/tmp/datasets/env_sound/Clotho/development/                | hyperscaleai-audiollm/datasets/public/Clotho-v2/audio/                       |
| iemocap        | /mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release/    | hyperscaleai-audiollm/users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/       |
| ravdess        | /mnt/tmp/datasets/emotion_raw/RAVDESS/                         | hyperscaleai-audiollm/users/jos/AudioEnc/RAVDESS/                            |
| emovdb         | /mnt/tmp/datasets/emotion_raw/EmoV-DB/                         | hyperscaleai-audiollm/users/jos/AudioEnc/EmoV-DB/                            |
| mustardpp      | /mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/               | hyperscaleai-audiollm/users/jos/AudioEnc/MUStARD_Plus_Plus/                  |

미매핑 source (nubes 부재 / 매핑 미확인): dailytalk (zero-byte placeholder),
audiocaps, laion_freesound, laion_audiostock (audio_path 빈 문자열), macs
(audio_path 빈 문자열), meld (path 구조 다름 — 별도 builder 필요).

미매핑 row 는 `audio_path` 만 유지, `nubes_path` 추가 안 함 → 학습 시 local fallback.

ASR (audio_asr) shard 의 row 는 이미 `nubes_path` 필드 갖고 있어 변경 불필요.
이 스크립트는 audio_env_sound + audio_emotion 만 대상.

Usage:
    # default: in-place 안 하고 v6_nubes/ 에 별도 출력 (safety)
    python -m scripts.manifest_builders.rewrite_audio_paths_nubes

    # in-place rewrite (백업 권장)
    python -m scripts.manifest_builders.rewrite_audio_paths_nubes --in-place

    # 사용자 정의 출력 dir
    python -m scripts.manifest_builders.rewrite_audio_paths_nubes \
        --src /mnt/tmp/datasets/manifests/v6 \
        --out /mnt/tmp/datasets/manifests/v6_nubes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PREFIX_MAPPINGS: dict[str, tuple[str, str] | None] = {
    # === 사용자 영역 (현재 세션 업로드) ===
    "audioset": (
        "/mnt/tmp/datasets/env_sound/AudioSet/audio/",
        "hyperscaleai-audiollm/users/jos/AudioEnc/AudioSet/audio/",
    ),
    "laion_bbc": (
        "/mnt/tmp/datasets/laion_extracted/bbc/",
        "hyperscaleai-audiollm/users/jos/AudioEnc/LAION-BBC/audio/",
    ),
    "iemocap": (
        "/mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release/",
        "hyperscaleai-audiollm/users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/",
    ),
    "ravdess": (
        "/mnt/tmp/datasets/emotion_raw/RAVDESS/",
        "hyperscaleai-audiollm/users/jos/AudioEnc/RAVDESS/",
    ),
    # === 사용자 영역 (다른 세션 업로드) ===
    "emovdb": (
        "/mnt/tmp/datasets/emotion_raw/EmoV-DB/",
        "hyperscaleai-audiollm/users/jos/AudioEnc/EmoV-DB/",
    ),
    "mustardpp": (
        "/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/",
        "hyperscaleai-audiollm/users/jos/AudioEnc/MUStARD_Plus_Plus/",
    ),
    # === nubes 기존 (public 영역) ===
    "fsd50k": (
        "/mnt/tmp/datasets/env_sound/FSD50K/FSD50K.dev_audio/",
        "hyperscaleai-audiollm/datasets/public/FSD50K/audio/",
    ),
    "clotho": (
        "/mnt/tmp/datasets/env_sound/Clotho/development/",
        "hyperscaleai-audiollm/datasets/public/Clotho-v2/audio/",
    ),
    "laion_epidemic": (
        "/mnt/tmp/datasets/laion_extracted/epidemic/",
        "hyperscaleai-audiollm/datasets/public/LAION-Audio-630k/epidemic_sound_effects/audio/",
    ),
    # === 미매핑 (skip) ===
    "dailytalk": None,
    "audiocaps": None,
    "laion_freesound": None,
    "laion_audiostock": None,
    "macs": None,
    "meld": None,
}


def rewrite_row(row: dict) -> tuple[dict, str]:
    """Return (new_row, status). status ∈ {'mapped', 'skip-no-source',
    'skip-unmapped', 'skip-no-prefix', 'skip-already-has-nubes'}."""
    if row.get("nubes_path"):
        return row, "skip-already-has-nubes"
    src = row.get("source")
    if not src:
        return row, "skip-no-source"
    mapping = PREFIX_MAPPINGS.get(src)
    if mapping is None:
        return row, "skip-unmapped"
    local_prefix, nubes_prefix = mapping
    audio_path = row.get("audio_path", "")
    if not audio_path.startswith(local_prefix):
        return row, "skip-no-prefix"
    suffix = audio_path[len(local_prefix):]
    new_row = dict(row)
    new_row["nubes_path"] = nubes_prefix + suffix
    return new_row, "mapped"


def process_shard(in_path: Path, out_path: Path | None, dry_run: bool) -> Counter:
    counter: Counter = Counter()
    out_lines: list[str] = []
    with open(in_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counter["json-error"] += 1
                continue
            new_row, status = rewrite_row(row)
            counter[status] += 1
            out_lines.append(json.dumps(new_row, ensure_ascii=False) + "\n")
    if not dry_run and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.writelines(out_lines)
    return counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("/mnt/tmp/datasets/manifests/v6"))
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/tmp/datasets/manifests/v6_nubes"))
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite shards in --src dir (skip --out)")
    ap.add_argument("--modalities", nargs="+",
                    default=["audio_env_sound", "audio_emotion"],
                    help="Manifest sub-dirs to process (audio_asr 는 이미 nubes_path 보유)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.in_place:
        out_root = args.src
    else:
        out_root = args.out

    print(f"[rewrite] src   = {args.src}", flush=True)
    print(f"[rewrite] out   = {out_root}{' (in-place)' if args.in_place else ''}", flush=True)
    print(f"[rewrite] dry?  = {args.dry_run}", flush=True)
    print(f"[rewrite] mods  = {args.modalities}", flush=True)
    print(f"[rewrite] mapped sources ({sum(1 for v in PREFIX_MAPPINGS.values() if v)}): "
          f"{[k for k, v in PREFIX_MAPPINGS.items() if v]}", flush=True)
    print(f"[rewrite] skipped sources ({sum(1 for v in PREFIX_MAPPINGS.values() if v is None)}): "
          f"{[k for k, v in PREFIX_MAPPINGS.items() if v is None]}", flush=True)

    grand: Counter = Counter()
    if not args.in_place and not args.dry_run:
        # Copy the audio_asr/ unchanged (already nubes-direct), so v6_nubes is a
        # complete drop-in replacement.
        asr_src = args.src / "audio_asr"
        asr_dst = out_root / "audio_asr"
        if asr_src.exists() and not asr_dst.exists():
            print(f"[rewrite] copying audio_asr/ unchanged: {asr_src} -> {asr_dst}",
                  flush=True)
            shutil.copytree(asr_src, asr_dst)

    for mod in args.modalities:
        mod_src = args.src / mod
        if not mod_src.exists():
            print(f"  skip {mod}: not found", flush=True)
            continue
        shards = sorted(mod_src.glob("*.jsonl"))
        print(f"[{mod}] {len(shards)} shards", flush=True)
        for shard in shards:
            if args.in_place:
                out_shard = shard
            else:
                out_shard = out_root / mod / shard.name
            cnt = process_shard(shard, out_shard, args.dry_run)
            grand.update(cnt)
            mapped = cnt.get("mapped", 0)
            total = sum(cnt.values())
            print(f"  {shard.name:<40s} mapped={mapped:>6d}/{total:>6d}  "
                  f"unmapped={cnt.get('skip-unmapped',0)}  "
                  f"no-prefix={cnt.get('skip-no-prefix',0)}  "
                  f"already={cnt.get('skip-already-has-nubes',0)}",
                  flush=True)

    print(f"\n[rewrite] DONE. grand totals:", flush=True)
    for k, v in grand.most_common():
        print(f"  {k:<28s} {v:>10,}", flush=True)


if __name__ == "__main__":
    main()
