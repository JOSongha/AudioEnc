# MELD audio (wav, 사용자 영역)

업로드 완료: 2026-05-08. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/MELD/audio/`.

## 배경

§ 12.10 에서 MELD csv 만 사용자 영역에 업로드하고 audio 는 **nubes public dir 의 mp3** (`/datasets/public/MELD.Raw/{train_splits, dev_splits_complete, output_repeated_splits_test}/*.mp3`) 를 가리켰음.

v6 stage1 학습 launch 시 torchaudio default backend (libsndfile) 가 mp3 디코드 inconsistent — multi-worker dataloader 환경에서 `Format not recognised` 에러 발생, 11k MELD row 가 학습에서 모두 skip. ffmpeg backend 도 audio_lmf env 에 미등록.

대신 **wav 본을 사용자 영역에 직접 업로드** + builder/eval 의 nubes_path 를 wav 가리키게 변경.

## 옵션 비교 + 결정 사유

| 옵션 | 내용 | 결정 |
|---|---|---|
| A | mp3 디코드 fallback 코드 추가 (omni_dataset, audio_io 의 ffmpeg backend) | **X** — env 에 ffmpeg backend 미설치, 추가 dependency 부담 |
| B | pyav backend 도입 | **X** — 추가 코드 복잡 |
| **C (선택)** | wav 본 사용자 영역 직접 업로드 | ✓ 가장 단순, libsndfile 만으로 동작 |

**C 선택 사유**: 코드 단순화 (mp3 fallback 인프라 전부 제거), env 의존 감소, multi-worker 환경 안정성.

## 내용

| 디렉터리 | 파일 수 | 크기 | 설명 |
|---|---:|---:|---|
| `audio/train/<stem>.wav` | 9,988 | 982 MB | Stage-1 학습 train split |
| `audio/dev/<stem>.wav` | 1,112 | 109 MB | Stage-1 학습 dev split |
| `audio/test/<stem>.wav` | 2,747 | 284 MB | Stage-2 eval split |

총 13,847 wav, ~1.4 GB. 업로드 시간 17분.

## audiollm-trainer 사용

- **Stage-1 학습**: train + dev (11,096 row, datasets.md § 4)
- **Stage-2 평가**: test (2,610 row, eval_source_emotion.load_meld_test)
- [`build_emotion_meld.py`](../../scripts/manifest_builders/build_emotion_meld.py) — `AUDIO_DIR_PATHS` / `NUBES_PREFIX` 가 `/users/jos/AudioEnc/MELD/audio/{train,dev}/` 가리킴, `.wav` 확장자.
- [`eval_source_emotion.load_meld_test`](../../evaluation/stage2/eval_source_emotion.py) — `AUDIO_PREFIX` `/users/jos/AudioEnc/MELD/audio/test`, `.wav`.
- [`rewrite_audio_paths_nubes.py`](../../scripts/manifest_builders/rewrite_audio_paths_nubes.py) — `_meld_transform` callable 제거 (builder 가 처음부터 nubes_path 박으니 rewrite 불필요).

## mp3 fallback 코드 제거

- [`src/llamafactory/data/audio_io.py`](../../src/llamafactory/data/audio_io.py): `_ta_load` ffmpeg fallback revert
- [`src/llamafactory/data/omni_dataset.py`](../../src/llamafactory/data/omni_dataset.py): `_download_from_nubes` 의 ext 분기 + ffmpeg fallback revert

§ 12.13 #6 (mp3 디코드 fallback chain) 도 obsolete.

## 검증

- nubes count: train 9,988 / dev 1,112 / test 2,747 ✓
- 첫 wav HEAD: `dia0_utt0.wav` 181,668 byte (정상 size)
- 학습 launch (step 200+) Format not recognised 에러 없음 ✓
- emotion token 통계 정상 (audio_emotion 348/22k, MELD wav 정상 fetch + decode)

## Source

- 로컬: `/mnt/tmp/datasets/emotion_raw/MELD/audio/{train,dev,test}/*.wav` (jos own download, MELD.Raw 변환)
- upstream: [MELD GitHub](https://github.com/declare-lab/MELD) (Poria et al. 2019)

## 라이선스

Academic-only EULA.

## 관련 문서

- [`../nubes_upload.md` § 12.14](../nubes_upload.md)
- [`./MELD-CSV/`](../MELD-CSV/README_upload.md) — emotion label csv (§ 12.10)
- [`../datasets.md` § 4 MELD row](../datasets.md)
