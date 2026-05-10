# AudioSet (audio + bal_train parquet + eval parquet + ontology)

업로드 준비: 2026-05-07. nubes 업로드 대상 = `hyperscaleai-audiollm/users/jos/AudioEnc/AudioSet/`.

## 배경

nubes 의 `/datasets/public/AudioSet_SL/` 에 audio 108,317 flac (bal_train + unbal_train + eval **통합 dir**) + AF-Think jsonl + naiveInst_AudioSet_SL.jsonl. v6 학습은 **bal_train 18,683 만** 사용 (datasets.md § 3). nubes 통합 dir 에서 학습 split 매핑하려면 metadata jsonl 의 라벨로 필터링 필수 → builder bug 시 eval row leak 위험 (§ 9.8).

본 업로드는 사용자 영역에 **split-aware** 형태로 별도 보존:
- `audio/` = bal_train 18,683 flac (학습 split 격리)
- `data/bal_train/` = bal_train 18,683 row HF parquet (audio bytes embedded)
- `data/eval/` = eval 17,141 row HF parquet (Stage-2 [`eval_audioset_map.py`](../../audiollm-trainer/evaluation/stage2/eval_audioset_map.py) 가 사용)
- `ontology.json` = 632 label vocab

builder 가 `audio/` 또는 `data/bal_train/` 만 보면 학습. `data/eval/` 만 보면 eval. **path 자체로 split 분리** → leak 안전.

## 내용

| 디렉터리 | 파일 수 | 크기 | 설명 |
|---|---:|---:|---|
| `audio/<youtube_id>.flac` | **18,683** | 25 GB | bal_train 의 audio bytes 만 추출한 flac. v6 manifest 의 `audio_path` 가 인용 |
| `data/bal_train/*.parquet` | 38 | 25 GB | bal_train HF parquet (`video_id, audio, labels, human_labels`), 18,683 row |
| `data/eval/*.parquet` | 35 | 23 GB | eval HF parquet, **17,141 row** (Stage-2 eval) |
| `ontology.json` | 1 | 343 KB | AudioSet 632-label hierarchical vocab + `freebase_mid` |
| `README.md` | 1 | 5 KB | upstream README |
| `README_upload.md` | 1 | ~3 KB | 본 문서 |

총 ~71 GB.

## audio/ 와 data/bal_train/ 의 관계

같은 데이터 두 형식:
- `audio/<id>.flac` = decoded audio (flac 추출)
- `data/bal_train/*.parquet` = HF `datasets` library 가 사용하는 형식 (audio bytes embedded + label metadata)

학습 시 v6 manifest 가 `audio_path: ...AudioSet/audio/<id>.flac` 으로 flac 사용. parquet 는 HF dataset API 호환 / 다른 task 용 redundant 보존.

## Source

- 로컬: `/mnt/tmp/datasets/env_sound/AudioSet/` (jos own download)
- 원본: [AudioSet by Google Research](https://research.google.com/audioset/) (CC BY 4.0 annotations / YouTube 원본 라이선스 별도)

## 사용 방법

### 학습 (Stage-1)

`audio/<id>.flac` 인용. v6 manifest:
```
{"modality": "audio_env_sound", "source": "audioset",
 "audio_path": "/mnt/tmp/datasets/env_sound/AudioSet/audio/<id>.flac",
 "captions": [...]}
```
nubes path 로 변환 시: `audio_path` → `nubes_path: hyperscaleai-audiollm/users/jos/AudioEnc/AudioSet/audio/<id>.flac` (별도 builder rewrite 필요).

### 평가 (Stage-2)

[`eval_audioset_map.py`](../../audiollm-trainer/evaluation/stage2/eval_audioset_map.py) 의 path:
```python
AUDIOSET_ROOT = Path("/mnt/tmp/datasets/env_sound/AudioSet")
EVAL_PARQUET_DIR = Path(os.environ.get(
    "AUDIOSET_EVAL_PARQUET_DIR",
    str(AUDIOSET_ROOT / "data/eval"),
))
ONTOLOGY_JSON = AUDIOSET_ROOT / "ontology.json"
```

nubes 사용 시 `AUDIOSET_EVAL_PARQUET_DIR` env override 또는 다운로드 후 로컬 캐시. parquet `audio` 컬럼 안의 bytes 가 audio.

## Row count 검증

- audio/ flac: 18,683 (= v6 학습 row count 정확 일치)
- data/bal_train/ rows: 18,683 (audio/ 와 1:1)
- data/eval/ rows: 17,141 (Stage-2 eval)
- ontology.json: 632 label (Audio events hierarchical)

## Leak 위험 / 안전성

본 업로드는 **path-level split 분리** 로 leak 안전:
- 학습 builder 가 `audio/` 또는 `data/bal_train/` 만 인용 → 절대 eval 진입 X
- eval 코드 가 `data/eval/` 만 인용 → 학습 데이터 진입 X
- nubes 통합 dir (`/datasets/public/AudioSet_SL/audio/` 108,317 통합) 의 metadata jsonl 라벨 필터링 의존성 회피

## 라이선스

annotations: CC BY 4.0. audio: YouTube 원본 라이선스 별도. 상업 배포 시 audio 라이선스 검토 필요.
