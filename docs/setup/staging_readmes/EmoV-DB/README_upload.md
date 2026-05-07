# EmoV-DB (Stage-1 emotion 학습)

업로드 준비: 2026-05-07. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/EmoV-DB/`.

## 배경

datasets.md § 4 emotion 6 source 중 하나. v6 룰 (canonical split 없는 source 통째로 학습, leak-fix 폐기) 적용으로 **4 화자 모두 학습 풀에 포함** (이전 v5 룰: Jenie 화자 held-out 폐기). 6,893 utt 학습. nubes 부재라 신규 업로드.

## 내용

| 항목 | 파일 수 / 크기 | 설명 |
|---|---|---|
| `bea/{Amused,Angry,Disgusted,Neutral,Sleepy}/*.wav` | 1,787 wav / 2.9 GB | 화자 1 (영어 여성). 5 emotion 모두 |
| `jenie/{Amused,Angry,Disgusted,Neutral,Sleepy}/*.wav` | 1,790 wav / 1.1 GB | 화자 2 (프랑스어 모국 여성, 영어 발화). 5 emotion |
| `josh/{Amused,Neutral,Sleepy}/*.wav` | 863 wav / 431 MB | 화자 3 (영어 남성). **3 emotion 만** (Angry / Disgusted 부재) |
| `sam/{Amused,Angry,Disgusted,Neutral,Sleepy}/*.wav` | 2,453 wav / 1.6 GB | 화자 4 (영어 남성). 5 emotion |
| `repo/{LICENSE.md, README.md, align_db.py, emov_mfa_alignment.py}` | 4 file / 156 KB | 표준 distribution doc + Montreal Forced Aligner 정렬 스크립트 |
| `README_upload.md` | (이 파일) | 업로드 정보 |

총 **6,893 wav / ~5.9 GB** (`*.tar.gz` archive 는 이미 추출 완료라 제외, `repo/.git` 제외).

## Source

- 로컬: `/mnt/tmp/datasets/emotion_raw/EmoV-DB/`
- Upstream: [github.com/numediart/EmoV-DB](https://github.com/numediart/EmoV-DB) (Adigwe et al. 2018, "The Emotional Voices Database (EmoV-DB)")
- 라이선스: 학술 사용 (datasets.md § 6 "academic-only EULA")

## 제외 항목

- `*.tar.gz` (4 화자 × 5 emotion 합 ~1.5 GB) — 이미 wav 추출되어 있고, archive 보존 의의 낮음
- `repo/.git` (history) — repo metadata 불필요

## 사용 방법 (Stage-1 학습)

[`build_emotion_emovdb.py`](../../../audiollm-trainer/scripts/manifest_builders/build_emotion_emovdb.py) (v6 갱신본):

```python
ROOT = Path("/mnt/tmp/datasets/emotion_raw/EmoV-DB")
SPEAKERS = ["bea", "jenie", "josh", "sam"]   # v6 룰: 4 화자 모두 학습
EMO_DIRS = ["Amused", "Angry", "Disgusted", "Neutral", "Sleepy"]
# audio: <ROOT>/<speaker>/<EmoDir>/<file>.wav
# josh 의 Angry/Disgusted 디렉터리는 없음 (skip)
```

nubes-direct 사용 시 builder 가 nubes 의 `users/jos/AudioEnc/EmoV-DB/<speaker>/<EmoDir>/<file>.wav` 를 인용하도록 path 수정 필요 (별개 작업).

## Row count 검증

| 화자 | wav 수 | emotion |
|---|---:|---|
| bea | 1,787 | 5 (Amused / Angry / Disgusted / Neutral / Sleepy) |
| jenie | 1,790 | 5 |
| josh | 863 | 3 (Amused / Neutral / Sleepy) |
| sam | 2,453 | 5 |
| **합계** | **6,893** | (datasets.md row count 일치) |
