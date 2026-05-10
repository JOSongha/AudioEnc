# DailyTalk (Stage-1 emotion utterance wav + metadata)

업로드 완료: 2026-05-08. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/DailyTalk/`.

## 배경

audiollm-trainer 의 DailyTalk 사용:
- **Stage-1 학습**: 23,773 utterance wav 통째로 (datasets.md § 4 — v6 룰: canonical split 없음 + leak-fix 폐기)

nubes 의 `/datasets/public/DailyTalk/audio/` 에 2,541 wav 가 **모두 0 byte placeholder** (사실상 부재). v6 학습 위해 utterance audio + metadata 신규 업로드 필수.

## 옵션 비교 + 결정 사유

| 옵션 | 내용 | 크기 | 결정 |
|---|---|---:|---|
| A | dialogue 단위 (2,541 wav, nubes 기존 형식) + leak-fix 5% cutoff | ~6 GB | **X** — nubes wav 가 zero-byte placeholder, 사용 불가. v5 leak-fix 폐기로 cutoff 도 의미 없음 |
| **B (선택)** | **utterance 단위** (23,773 wav + 23,773 txt + metadata.json + README) | ~6.6 GB | ✓ |

**B 선택 사유**:
1. v6 룰: canonical split 없음 + leak-fix 폐기 → 23,773 utt 모두 학습 풀
2. utterance-level audio 가 emotion label 과 1:1 매핑 (dialogue-level 은 cutoff 의 ambiguity 유발)
3. nubes 기존 dialogue 영역은 zero-byte 라 어차피 사용 불가

## 내용

| 디렉터리 | 파일 수 | 크기 | 설명 |
|---|---:|---:|---|
| `data/<dialog>/<utt>.wav` | 23,773 wav, 2,541 dialog dirs | ~6.5 GB | utterance audio |
| `data/<dialog>/<utt>.txt` | 23,773 txt | ~10 MB | per-utt transcript |
| `metadata.json` | 1 | 7.85 MB | dialogue-level metadata (speaker, emotion, transcript) |
| `README_upload.md` | 1 | ~3 KB | 본 문서 |

총 47,547 file (recursive count 검증 완료).

## audiollm-trainer 사용

- **Stage-1 학습**: 23,773 row, datasets.md § 4
- **Builder**: [`build_emotion_dailytalk.py`](../../scripts/manifest_builders/build_emotion_dailytalk.py) — v5 cutoff 로직 제거, 통째 학습. metadata.json 에서 utterance + emotion 라벨 추출.
- **Stage-2 평가**: 미사용 (eval_source_emotion 의 DT held-out 평가 v5 leak-fix 폐기로 함께 폐기)

## 검증

- nubes recursive list: 47,547 file (= 23,773 wav + 23,773 txt + metadata + README) ✓
- 2,541 dialog subdirs ✓
- sample HEAD: `data/813/0_0_d813.wav` 385,366 byte (정상 size, NOT zero-byte)

## Source

- 로컬: `/mnt/tmp/datasets/emotion_raw/DailyTalk/dailytalk/data/<dialog>/<utt>.wav` (jos own download)
- upstream: [DailyTalk GitHub](https://github.com/keonlee9420/DailyTalk) (Lee et al. 2023)

## 라이선스

Academic-only EULA.

## 관련 문서

- [`../nubes_upload.md` § 12.11](../nubes_upload.md)
- [`../datasets.md` § 4 DailyTalk row](../datasets.md)
