# Clotho-v2 (eval + val splits)

업로드 완료: 2026-05-08. nubes 업로드 대상 = `hyperscaleai-audiollm/datasets/public/Clotho-v2/`.

## 배경

audiollm-trainer 의 Clotho 사용:
- **Stage-1 학습**: dev (3,839) + val (1,045) = 4,884 row (datasets.md § 3 sound captioning)
- **Stage-2 평가**: evaluation (1,045) — eval_clotho_caption.py

nubes 의 `/datasets/public/Clotho-v2/` 에 dev 만 있었음. **eval 용 evaluation csv + audio 부재** → Stage-2 eval nubes-direct 불가. Stage-1 학습은 dev 만 nubes-mapped, val 1,044 row 는 local fallback.

2026-05-08 eval + val splits 추가 업로드.

## 옵션 비교 + subdir 분리 결정 사유

| 옵션 | 내용 | 결정 |
|---|---|---|
| A | 같은 `audio/` 에 dev/eval/val wav 모두 합쳐 업로드 | **X** — 파일명 충돌 4건 (dev∩eval=1, dev∩val=1, eval∩val=2) → 덮어쓰기 위험 |
| **B (선택)** | split 별 subdir 분리 (`audio/`, `audio_evaluation/`, `audio_validation/`) | ✓ 충돌 0 |

## 내용

| 파일/디렉터리 | 파일 수 | 크기 | 설명 |
|---|---:|---:|---|
| `audio/` (기존 dev) | 3,839 wav | (기존) | Stage-1 dev split |
| `audio_evaluation/` (신규) | 1,045 wav | ~2.0 GB | Stage-2 eval split (held-out) |
| `audio_validation/` (신규) | 1,045 wav | ~2.0 GB | Stage-1 val split |
| `clotho_captions_development.csv` (기존) | 1 | 1.3 MB | dev 5-caption per clip |
| `clotho_captions_evaluation.csv` (신규) | 1 | 362 KB | eval 5-caption per clip |
| `clotho_captions_validation.csv` (신규) | 1 | 368 KB | val 5-caption per clip |
| `clotho_metadata_development.csv` (기존) | 1 | 831 KB | dev metadata |

## audiollm-trainer 사용

- **Stage-1 학습**: dev + val (4,884 row). builder 가 dev → `audio/`, val → `audio_validation/` 매핑.
- **Stage-2 평가**: [`eval_clotho_caption.py:128-170`](../../evaluation/stage2/eval_clotho_caption.py#L128-L170) `load_clotho_split()` 가 `_NUBES_SPLIT_KEYS` table 로 dev / eval / val 모두 nubes 분기.
- [`build_clotho.py`](../../scripts/manifest_builders/build_clotho.py): dev/val 둘 다 `nubes_path` 박음.
- [`rewrite_audio_paths_nubes.py`](../../scripts/manifest_builders/rewrite_audio_paths_nubes.py): clotho mapping list-form (dev → audio/, val → audio_validation/).

## 검증

- 4 파일/디렉토리 업로드 nubes listing 확인 ✓
- audio 첫 5 wav 정상 size (1.5-2.5 MB)
- 파일명 충돌 4건 회피 (split 별 subdir)
- eval harness smoke: `decode_audio()` 가 nubes_path prefix 로 자동 routing → gateway HTTP fetch + torchaudio decode (8 sample 모두 성공)

## Source

- 로컬: `/mnt/tmp/datasets/env_sound/Clotho/{development,evaluation,validation}/` + `captions_*.csv`
- upstream: [Clotho v2.1 (Zenodo)](https://zenodo.org/record/4783391) (Drossos et al. 2020)

## 라이선스

Tampere Univ. dataset license (research only).

## 관련 문서

- [`../nubes_upload.md` § 12.12](../nubes_upload.md)
- [`../datasets.md` § 3 Clotho row](../datasets.md)
