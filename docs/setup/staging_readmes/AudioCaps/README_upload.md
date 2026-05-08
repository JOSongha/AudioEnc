# AudioCaps (audio + parquets, 표준 dist 보존)

업로드 완료: 2026-05-08. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/AudioCaps/`.

## 배경

audiollm-trainer 의 AudioCaps 사용:
- **Stage-1 학습**: train + val (45,623 row, datasets.md § 3 sound captioning)
- test 는 학습 풀 미포함

nubes 부재 (`/audiocaps/`, `/AudioCaps/`, `/AudioSet_SL/audiocaps/`, `/16kHz/audiocaps/` 모두 404). 옵션 C2 (audio 46,506 flac + HF parquet 473 file) 로 업로드.

## 옵션 비교 + C2 결정 사유

| 옵션 | 내용 | 크기 | 결정 |
|---|---|---:|---|
| A | extracted flac 만 (45,623 train+val) | 40 GB | 학습 데이터만 — test 는 별도 |
| B | extracted flac 전체 (46,506 = 45,623 + 883 test) | 40 GB | test 도 보존 |
| C1 | audio + 표준 dist parquet (HF `OpenSound/AudioCaps`) | ~80 GB | 학습 데이터 + 표준 dist 둘 다 |
| **C2 (선택)** | audio (46,506 flac, all splits) + parquet (473 file = 412+20+41 train/val/test, audio bytes 포함 표준 dist) | ~81 GB | 학습용 flac + 재현용 parquet 둘 다 보존. HF `OpenSound/AudioCaps` test 41 parquet 신규 다운 + 추출 |

**C2 선택 사유**:
1. 학습용은 extracted flac (random access 빠름)
2. 표준 dist 인용은 parquet (audio bytes + caption 모두 포함, HF datasets API 호환)
3. 둘 다 보존하면 builder 재작성 시 어느 형식이든 사용 가능

## 내용

| 디렉터리 | 파일 수 | 크기 | 설명 |
|---|---:|---:|---|
| `audio/` | 46,506 flac | ~40 GB | flat dir, `<id>.flac` (45,623 train+val + 883 test) |
| `data/` | 473 parquet | ~41 GB | train (412) + val (20) + test (41) HF parquet, audio bytes embedded |
| `README_upload.md` | 1 | ~3 KB | 본 문서 |

## audiollm-trainer 사용

- **Stage-1 학습**: train + val (45,623 row). builder 가 audio/ 의 train/val flac id list + caption metadata 파싱
- **Stage-2 평가**: AudioCaps test 미사용 (다른 caption corpora 사용)

## 검증

- `audio/` 46,506 flac (확인 완료)
- `data/` 473 parquet (412+20+41 = 473)
- 첫 sample HEAD: `--1_cCGK4M_0.flac` 960,044 byte (정상 size)

## Source

- 로컬 extracted: `/mnt/tmp/datasets/laion_extracted/audiocaps/` (40 GB)
- 로컬 raw: `/mnt/tmp/datasets/audiocaps/` (38 GB)
- HF parquet: `OpenSound/AudioCaps` (test 41 parquet 신규 download)

## 라이선스

MIT (annotation), 원본 YouTube audio 는 별도 라이선스.

## 관련 문서

- [`../nubes_upload.md` § 12.9](../nubes_upload.md)
- [`../datasets.md` § 3 AudioCaps row](../datasets.md)
