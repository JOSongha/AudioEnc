# MELD CSV (Stage-1 학습 + Stage-2 평가 라벨)

업로드 완료: 2026-05-08. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/MELD/CSV/`.

## 배경

audiollm-trainer 의 MELD 사용:
- **Stage-1 학습**: train + dev (11,096 row, datasets.md § 4 emotion)
- **Stage-2 평가**: test (2,610 row, eval_source_emotion.load_meld_test)

nubes 의 `/datasets/public/MELD.Raw/` 에 audio (mp3) 는 있으나 emotion label csv 부재. v6 builder + eval 가 csv 직접 인용하도록 사용자 영역에 업로드.

## 내용

| 파일 | row | 설명 |
|---|---:|---|
| `train_sent_emo.csv` | 9,988 | Stage-1 학습 train split, Dialogue_ID + Utterance_ID + Emotion + Utterance |
| `dev_sent_emo.csv` | 1,108 | Stage-1 학습 dev split |
| `test_sent_emo.csv` | 2,610 | Stage-2 eval test split (held-out) |

총 row 13,706. v5 sehyun shard 의 row count 9,988 + 1,108 = 11,096 (train+dev) 일치.

## audiollm-trainer 사용

- **Builder**: [`build_emotion_meld.py`](../../scripts/manifest_builders/build_emotion_meld.py) — csv 를 nubes URL 로 직접 fetch (`urllib.request.urlopen`), audio_filename → nubes_path 매핑.
- **Stage-2 eval**: [`eval_source_emotion.load_meld_test`](../../evaluation/stage2/eval_source_emotion.py) — test csv 동일 방식 fetch.
- v5 sehyun cache 의존성 (`/mnt/ddn/users/sehyun/.../meld_*_shard_*.jsonl`) 폐기 — v6 부터 csv 직접 파싱.

## 검증

- 3 csv HEAD 200 OK
- builder smoke test: train+dev 통합 = 11,096 row (datasets.md § 4 일치)
- eval smoke: load_meld_test() → 2,610 row + 7 emotion labels

## Source

- 원본: `/mnt/tmp/datasets/emotion_raw/MELD/MELD.Raw/{train,dev,test}_sent_emo.csv` (jos own download)
- upstream: [MELD GitHub](https://github.com/declare-lab/MELD) (Poria et al. 2019)

## 라이선스

Academic-only EULA (MELD).

## 관련 문서

- [`../nubes_upload.md` § 12.10](../nubes_upload.md)
- [`./MELD-audio-wav/`](../MELD-audio-wav/README_upload.md) — audio wav 업로드 (mp3 → wav 전환, § 12.14)
- [`../datasets.md` § 4 MELD row](../datasets.md)
