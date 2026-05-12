"""Nubes-aware audio / metadata loader for Stage-2 eval scripts.

각 eval_*.py 가 hardcoded local path 대신 nubes 게이트웨이에서 직접 fetch 할 수
있도록 공통 helper 제공. 환경변수 `EVAL_USE_NUBES=1` 설정 시 nubes 모드 활성.
nubes 모드에서도 fetch 실패 시 local fallback (있는 경우).

같은 nubes API 를 공유하는 sibling helper:
- `scripts/manifest_builders/_nubes_helper.py` — Stage-1 manifest builder 용
  (recursive list + parallel txt fetch). 본 모듈은 Stage-2 eval 용 (audio
  bytes / parquet / csv 단건 fetch + 캐시).

Nubes URL 매핑 (2026-05-07 시점):
- gateway: http://c.nubes.sto.navercorp.com:8000/v1/
- bucket : hyperscaleai-audiollm

각 source 의 nubes prefix 는 `NUBES_BASES` dict 참고. 미업로드 source 는 미정의
(eval 시 local fallback 강제).
"""
from __future__ import annotations

import io
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

NUBES_GATEWAY = "http://c.nubes.sto.navercorp.com:8000/v1"
NUBES_BUCKET = "hyperscaleai-audiollm"

# 환경변수로 nubes 모드 on/off
USE_NUBES = os.environ.get("EVAL_USE_NUBES", "0") == "1"

# 로컬 캐시 디렉터리 (download 후 재사용)
NUBES_CACHE_ROOT = Path(os.environ.get(
    "EVAL_NUBES_CACHE", "/mnt/tmp/nubes_eval_cache"))

# Source-별 nubes prefix (audio + metadata).
# 형식: source -> {"audio": <prefix>, "metadata": <prefix or None>, ...}
NUBES_BASES: dict[str, dict[str, str]] = {
    # === Stage-2 eval source 들 ===
    "fsd50k_eval": {
        "audio": "users/jos/AudioEnc/FSD50K/FSD50K.eval_audio/",
        "ground_truth": "users/jos/AudioEnc/FSD50K/FSD50K.ground_truth/",
        "metadata": "users/jos/AudioEnc/FSD50K/FSD50K.metadata/",
    },
    "audioset_eval": {
        "parquet": "users/jos/AudioEnc/AudioSet/data/eval/",
        "ontology": "users/jos/AudioEnc/AudioSet/ontology.json",
    },
    "iemocap": {
        # Session5 eval 용. Sessions 1-5 모두 사용자 영역에 있음.
        "root": "users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/",
    },
    "ravdess": {
        "root": "users/jos/AudioEnc/RAVDESS/",
    },
    "emovdb": {
        "root": "users/jos/AudioEnc/EmoV-DB/",
    },
    "meld": {
        # nubes 기존 public 영역 (test split)
        "test_audio": "datasets/public/MELD.Raw/output_repeated_splits_test/",
        "train_audio": "datasets/public/MELD.Raw/train_splits/",
        "dev_audio": "datasets/public/MELD.Raw/dev_splits_complete/",
    },
    "librispeech": {
        # nubes 기존 public — eval test split 만 사용
        "test_clean": "datasets/public/librispeech_asr/clean/test/",
        "test_other": "datasets/public/librispeech_asr/other/test/",
    },
    "clotho": {
        # nubes public — dev (학습), val (학습), eval (Stage-2). 2026-05-08 § 12.12
        # 업로드로 eval+val csv + audio subdir 추가. dev/eval/val 파일명 충돌 4건
        # 회피 위해 split 별 audio_{evaluation,validation}/ 분리.
        "audio":          "datasets/public/Clotho-v2/audio/",            # dev
        "audio_eval":     "datasets/public/Clotho-v2/audio_evaluation/",
        "audio_val":      "datasets/public/Clotho-v2/audio_validation/",
        "captions_dev":   "datasets/public/Clotho-v2/clotho_captions_development.csv",
        "captions_eval":  "datasets/public/Clotho-v2/clotho_captions_evaluation.csv",
        "captions_val":   "datasets/public/Clotho-v2/clotho_captions_validation.csv",
    },
}


def fetch_nubes_bytes(nubes_path: str, *, timeout: float = 30.0,
                      cache: bool = True) -> bytes:
    """Fetch raw bytes from nubes gateway. Cache to local disk if cache=True."""
    if nubes_path.startswith(NUBES_BUCKET + "/"):
        nubes_path = nubes_path[len(NUBES_BUCKET) + 1:]
    nubes_path = nubes_path.lstrip("/")

    if cache:
        cache_path = NUBES_CACHE_ROOT / nubes_path
        if cache_path.exists():
            return cache_path.read_bytes()

    url = f"{NUBES_GATEWAY}/{NUBES_BUCKET}/{nubes_path}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read()

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(body)
    return body


def fetch_nubes_audio_tensor(nubes_path: str, *, target_sr: Optional[int] = None,
                             timeout: float = 30.0):
    """Fetch audio file from nubes and return (waveform, sr). Optional resample."""
    import torchaudio
    body = fetch_nubes_bytes(nubes_path, timeout=timeout)
    wav, sr = torchaudio.load(io.BytesIO(body))
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def fetch_nubes_text(nubes_path: str, *, encoding: str = "utf-8",
                     timeout: float = 30.0) -> str:
    body = fetch_nubes_bytes(nubes_path, timeout=timeout)
    return body.decode(encoding, errors="replace")


def list_nubes_dir(prefix: str, *, suffix: Optional[str] = None,
                   max_contents: int = 1000, recursive: bool = False):
    """Yield nubes object names under prefix (with continuation-token pagination).

    prefix 는 bucket 이후 path. suffix 로 필터 (예: '.flac').
    recursive=True 면 sub-dir 모두 들어감.
    """
    import json
    if not prefix.endswith("/"):
        prefix += "/"
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    stack = [prefix]
    while stack:
        cur = stack.pop()
        token = None
        while True:
            params = {"dir": cur, "max-contents": str(max_contents)}
            if token:
                params["continuation-token"] = token
            url = (f"{NUBES_GATEWAY}/{NUBES_BUCKET}?"
                   f"{urllib.parse.urlencode(params)}")
            with urllib.request.urlopen(url, timeout=60.0) as r:
                body = r.read()
                headers = r.headers
            try:
                entries = json.loads(body) if body else []
            except json.JSONDecodeError:
                entries = []
            if not entries:
                break
            for e in entries:
                full = cur + e["Name"]
                if e.get("IsDir"):
                    if recursive:
                        stack.append(full + "/")
                    continue
                if suffix is not None and not full.endswith(suffix):
                    continue
                yield full.lstrip("/")
            token = headers.get("X-Continuation-Token")
            if not token:
                break


def resolve_audio_path(local_path: Path, nubes_key: str,
                       *, nubes_subdir: str = "audio") -> str:
    """Map a local-filesystem audio path to nubes path via NUBES_BASES.

    Useful when an eval script uses fname-based addressing (e.g. fsd50k_map.py
    `EVAL_AUDIO_DIR / f"{fname}.wav"`). Replace `EVAL_AUDIO_DIR` with the nubes
    prefix.

    Example:
        resolve_audio_path(Path("100.wav"), "fsd50k_eval", nubes_subdir="audio")
        # -> "users/jos/AudioEnc/FSD50K/FSD50K.eval_audio/100.wav"
    """
    base = NUBES_BASES.get(nubes_key, {}).get(nubes_subdir)
    if base is None:
        raise KeyError(f"NUBES_BASES[{nubes_key!r}][{nubes_subdir!r}] not defined")
    return f"{base.rstrip('/')}/{local_path.name}"
