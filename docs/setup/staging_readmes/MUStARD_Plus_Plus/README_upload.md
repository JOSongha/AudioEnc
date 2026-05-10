# MUStARD++ (Stage-1 emotion 학습)

업로드 준비: 2026-05-07. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/MUStARD_Plus_Plus/`.

## 배경

datasets.md § 4 의 emotion 6 source 중 하나. v6 학습 풀에 1,200 utt 사용 (canonical split 없음, 전체). nubes 에 부재라 신규 업로드.

## 내용

| 항목 | 파일 수 / 크기 | 설명 |
|---|---|---|
| `audio_wav/<scene>_<key>_u.wav` | 1,201 wav / 172 MB | 발화 단위 audio (videos/ 의 utterance mp4 에서 ffmpeg 추출 결과). 우리 학습 풀 핵심. [`build_emotion_mustardpp.py`](../../../audiollm-trainer/scripts/manifest_builders/build_emotion_mustardpp.py) 가 `audio_wav/<KEY>.wav` 사용 |
| `mustard++_text.csv` | 608 KB | 라벨 csv. 컬럼: SCENE, KEY, SENTENCE, END_TIME, SPEAKER, SHOW, Sarcasm, Sarcasm_Type, Implicit_Emotion, Explicit_Emotion, Valence, Arousal. utterance + context row 합 6,202 (학습 시 utterance row 만 추출) |
| `utterance_ids.txt` | 59 KB | Google Drive ID ↔ filename 매핑 (1,202 line, 표준 distribution 의 video 다운로드 manifest) |
| `README.md` | 1.5 KB | 표준 distribution README (upstream) |
| `README_upload.md` | (이 파일) | 업로드 정보 (보완) |

총 ~172 MB.

## Source

- 로컬: `/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/`
- Upstream: [github.com/cfiltnlp/MUStARD_Plus_Plus](https://github.com/cfiltnlp/MUStARD_Plus_Plus) (Ray et al. 2022, "A Multimodal Corpus for Emotion Recognition in Sarcasm")
- 라이선스: 학술 사용 only (datasets.md § 6 "academic-only EULA")

## 제외 항목

다음 디렉터리 / 파일은 업로드 안 함:

- `.git/` (392 KB) — repo metadata 불필요
- `MPP_Code/` (204 KB) — upstream repo 의 baseline training code (우리 학습 풀과 무관)
- `videos/` (103 MB) — utterance / context mp4. 로컬에 1+64 = 65 mp4 만 (utterance_ids.txt 의 1,202 entry 중 일부) 라 표준 distribution 완전성 부족. audio_wav 가 이미 1,201 utterance 모두 추출 완료라 보존 가치 낮음

## 사용 방법 (Stage-1 학습)

[`build_emotion_mustardpp.py`](../../../audiollm-trainer/scripts/manifest_builders/build_emotion_mustardpp.py):

```python
ROOT = Path("/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus")
CSV_PATH = ROOT / "mustard++_text.csv"
# audio: audio_wav/<KEY>.wav
```

nubes-direct 사용 시 builder 가 nubes 의 `users/jos/AudioEnc/MUStARD_Plus_Plus/audio_wav/<KEY>.wav` 를 인용하도록 path 수정 필요 (별개 작업).

## Row count 검증

| 항목 | Count |
|---|---:|
| audio_wav 의 .wav | 1,201 |
| utterance_ids.txt entries | 1,202 (header 포함) |
| mustard++_text.csv row (header 제외) | 6,202 (utterance + context 합) |

datasets.md row count = 1,200 (학습 사용 row). audio_wav 의 1,201 중 1 은 다운로드 시 추가된 잔여물 가능성, builder 가 csv 의 valid utterance row 1,200 만 사용.
