# RAVDESS (24 actors, full speech corpus)

업로드 준비: 2026-05-07. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/RAVDESS/`.

## 배경

nubes 부재. v6 학습 풀에 1,440 wav 사용 (datasets.md § 4 — v6 룰: canonical split 없음 + leak-fix 폐기 → 24 actors 모두 학습 풀, eval held-out 없음).

## 내용

| 디렉터리 | 파일 수 | 크기 | 설명 |
|---|---:|---:|---|
| `Actor_01/` ~ `Actor_24/` | 60 wav × 24 = **1,440 wav** | 565 MB | RAVDESS-Speech full set (Audio-only, modality 03) |

**제외 항목**: `speech.zip` (365 MB, redundant archive — 개별 wav 가 이미 모든 데이터 포함). 표준 distribution 의 zip archive form 이 필요할 때만 별도 업로드.

## Filename schema

RAVDESS 표준 7-segment filename: `<modality>-<channel>-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav`

| Segment | 의미 | 값 |
|---|---|---|
| Modality | 03 = audio-only (uploaded) / 01 = audio+video / 02 = video-only (not uploaded) | 03 |
| Vocal channel | 01 = speech / 02 = song | 01 (speech) |
| Emotion | 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised | 01-08 |
| Intensity | 01 = normal, 02 = strong (neutral 은 strong 없음) | 01-02 |
| Statement | 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door" | 01-02 |
| Repetition | 01 = first take, 02 = second take | 01-02 |
| Actor | 01-24 (홀수 = male, 짝수 = female) | 01-24 |

예: `03-01-04-01-02-01-12.wav` = audio-only / speech / sad / normal / "dogs..." / 1st take / actor 12 (female).

## Source

- 로컬: `/mnt/tmp/datasets/emotion_raw/RAVDESS/Actor_*` (jos own download)
- 원본: [Zenodo RAVDESS Audio_Speech_Actors_01-24.zip](https://zenodo.org/record/1188976) (CC BY-NC-SA 4.0, Livingstone & Russo 2018)

## audiollm-trainer 사용

- **Stage-1 학습**: 1,440 wav 모두 학습 풀 (datasets.md § 4 v6 룰)
- **Stage-2 평가**: [`eval_source_emotion.py`](../../audiollm-trainer/evaluation/stage2/eval_source_emotion.py) RAVDESS corpus 평가 (8-class). v5 leak-fix (Actors 21-24 held-out) 는 **v6 에서 폐기됨** — 24 actors 모두 학습. eval 시 자기 자신 (self-held-out) 또는 cross-corpus 통한 평가
- **plot / aggregate**: [aggregate_results.py](../../audiollm-trainer/evaluation/stage2/aggregate_results.py), [plot_trajectories.py](../../audiollm-trainer/evaluation/stage2/plot_trajectories.py), [plot_v1_v2_compare.py](../../audiollm-trainer/evaluation/stage2/plot_v1_v2_compare.py) 가 RAVDESS_acc / RAVDESS_F1 컬럼 사용

## Row count 검증

- 24 actors × 60 wav = **1,440** (정확)
- size: 565 MB (zip 제외)

## 라이선스

CC BY-NC-SA 4.0 — academic / non-commercial only. 상업 배포 시 별도 라이선스 검토.
