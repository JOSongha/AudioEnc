# MACS (caption metadata yaml backup, 옵션 C)

업로드 완료: 2026-05-08. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/MACS/MACS.yaml`.

## 배경

MACS 의 audio (3,930 wav, TAU2019 의 3 scene = airport / park / public_square) 는
nubes `/datasets/public/MACS/audio/` 에 이미 14,400 (TAU2019 source `a` 전체 dump)
보존. v6 학습은 그중 `MACS.yaml` 정의된 3,930 만 인용.

따라서 audio 중복 업로드 불필요. **caption metadata yaml 만** 사용자 영역에
backup 으로 업로드 (옵션 C, ddn audio extract dir 의존성 폐기).

## 옵션 비교 + C 결정 사유

| 옵션 | 내용 | 비고 |
|---|---|---|
| A | yaml + audio 모두 사용자 영역 (4 GB) | 중복 업로드. nubes audio 이미 충분 |
| B | yaml + audio 의 3 scene subset (1 GB) | nubes public 의 14,400 superset 으로 충분, subset 별도 업로드 무의미 |
| **C (선택)** | yaml 만 backup (2.7 MB) | nubes audio 그대로 사용 + builder 가 yaml fetch (nubes URL 우선 + ddn fallback) |

**C 선택 사유**: audio 는 nubes 에 이미 있고 builder 가 path 만 매핑하면 됨.
yaml backup 은 ddn 사라져도 builder 가 동작하도록 fallback 보강.

## 내용

| 파일 | 크기 | 설명 |
|---|---:|---|
| `MACS.yaml` | 2.7 MB | 3,930 entry caption metadata (audio_filename, captions[], annotators) |

## audiollm-trainer 사용

- **Stage-1 학습**: 3,930 row (datasets.md § 3, env_sound)
- **Builder**: [`build_macs.py`](../../scripts/manifest_builders/build_macs.py) — `_fetch_yaml()` 가 nubes URL 우선 fetch + `MACS_YAML_LOCAL` env var fallback (ddn 도 사용 가능). nubes 만으로 동작 가능.

## 검증

- `MACS.yaml` HEAD 200 OK (2.7 MB)
- builder smoke test 통과: 3,930 entry / captions parsed

## Source

- 로컬 (ddn local 또는 nubes mirror): `/mnt/tmp/datasets/env_sound/MACS/MACS.yaml`
- 원본: TAU Urban Acoustic Scenes 2019 development (Mesaros et al.) + MACS caption (Martín-Morató & Mesaros 2021)

## 라이선스

CC BY 4.0 (TAU2019 base + MACS caption).

## 관련 문서

- [`../nubes_upload.md` § 12.8](../nubes_upload.md)
- [`../datasets.md` § 3 MACS row](../datasets.md)
