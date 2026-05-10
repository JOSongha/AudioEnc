# Stage-1 DAC-VAE v3 — 학습 데이터 카탈로그

> Config: [`configs/ASR/stage1_dac_vae_v3.yaml`](../../configs/ASR/stage1_dac_vae_v3.yaml)
> Run script: [`scripts/ASR/run_stage1_dac_vae_v3.sh`](../../scripts/ASR/run_stage1_dac_vae_v3.sh)
> Manifest dir: `/mnt/tmp/datasets/manifests/v3/`
> Build log: [`docs/v3_progress_log.md`](../v3_progress_log.md)

## 개요

v2 (libri+mls+vox 단독)에서 sound captioning 9 dataset + emotion MCQA 7 dataset 추가. 모달리티 3종을 단일 manifest dir에서 IterableDataset로 mix.

| Modality | Field schema | Loader path |
|---|---|---|
| `audio_asr` | `nubes_path`/`audio_path` + `text` | [`omni_dataset.py:_build_prompt_targets:audio_asr`](../../src/llamafactory/data/omni_dataset.py#L262-L266) |
| `audio_env_sound` | `audio_path` + `captions:[...]` | [`omni_dataset.py:282-307`](../../src/llamafactory/data/omni_dataset.py#L282-L307) (caption pool) |
| `audio_emotion` | `audio_path` + `question`/`choices`/`answer` | [`omni_dataset.py:268-280`](../../src/llamafactory/data/omni_dataset.py#L268-L280) (MCQA) |

## 데이터셋별 통계 (2026-05-01 기준)

### ASR (audio_asr) — 11.35M rows

| Source | rows | shards | 비고 |
|---|---:|---:|---|
| MLS English | 10,808,037 | – | nubes_path → libri_mls_vox_*.jsonl |
| LibriTTS-R | 354,721 | – | 동일 nubes shard, source="unknown_asr" (TTS 데이터) |
| VoxPopuli | 182,466 | – | 동일 nubes shard |
| **합계** | **11,345,224** | **128** | |

**참고**: vanilla LibriSpeech 학습 데이터는 nubes 미포함. eval (`eval_librispeech_wer.py`)은 test.clean / test.other 평가하는데 직접 학습 셋이 없어 MLS+LibriTTS-R로 일반화 의존. 분배 외에서 빌드된 vanilla `librispeech_train_*` 19 shards (281k rows) 는 사용자 결정으로 격리됨.

### Sound captioning (audio_env_sound) — 685k rows

| Source | split | rows | shards | Caption 출처 |
|---|---|---:|---:|---|
| laion_freesound | train_1 | ~235,000 | 16 | tar 안 `<id>.json:text` (1-2 caps) |
| laion_freesound | train_2 | ~178,000 | 12 | 〃 |
| laion_freesound | test | ~46,000 | 4 | 〃 |
| laion_bbc | train | 28,742 | 2 | tar JSON `text` (1 cap) |
| laion_bbc | test | 3,194 | 1 | 〃 |
| laion_epidemic | train | 68,081 | 5 | parquet `text` (T5-debiased) |
| laion_epidemic | test | 7,564 | 1 | 〃 |
| laion_audiostock | – | 9,139 | 1 | LAION csv `caption` 컬럼 |
| audiocaps | train | 45,178 | 4 | 1 cap/clip |
| audiocaps | val | 445 | 1 | 5 caps/clip (list로 결합) |
| clotho | dev | 3,837 | 1 | 5 caps/clip (list) |
| clotho | val | 1,044 | 1 | 〃 |
| audioset | bal_train | 18,683 | 2 | ontology→`"sound of X, Y, and Z"` |
| fsd50k | dev | 40,966 | 3 | 〃 (AudioSet ontology = FSD50K superset) |
| macs | – | _pending_ | _pending_ | TAU2019 zip DL → MACS 클립 추출 (Agent C 진행 중) |
| **subtotal** | | **~685,000** | **~54** | |

### Emotion MCQA (audio_emotion) — 43,203 rows (Set B: per-dataset native 라벨)

각 데이터셋 native 클래스 셋 유지 (canonical 7-class 매핑 안 함). 라벨 강제 매핑으로 인한 정보 손실 방지 + Stage-2 `eval_source_emotion.py`의 native 라벨 평가와 분포 일치.

| Source | rows | shards | 클래스 (native) | 비고 |
|---|---:|---:|---|---|
| meld | 11,096 | 1 | 7: anger, disgust, fear, joy, neutral, sadness, surprise | train+dev only (test split = Stage-2 eval) |
| dailytalk | 23,773 | 2 | 7: anger, disgust, fear, happiness, no emotion, sadness, surprise | – |
| emovdb | 6,893 | 1 | 5: amused, angry, disgusted, neutral, sleepy | – |
| mustardpp | **1** ⚠ | 1 | 9: anger, disgust, excitement, fear, frustration, happiness, neutral, sadness, surprise | audio 1/1202 (Agent F가 GDrive에서 보강 중) |
| ravdess | 1,440 | 1 | 8: angry, calm, disgust, fearful, happy, neutral, sad, surprised | – |
| **subtotal** | **43,203** | **6** | | |

> **CREMA-D 제외** (2026-05-02 14:48, user 지시): 원본 path가 다른 user (`/mnt/ddn/shkim/...`) 소유라 영구 제외.
> **IEMOCAP 제외** (2026-05-02 14:55, 동일 정책): 원본 audio path가 `/mnt/ddn/kyudan/IEMOCAP/...` (kyudan user 소유)라 영구 제외. `emotion_iemocap_*` shards는 v3_quarantine 으로 이동.

## 격리 (`/mnt/tmp/datasets/manifests/v3_quarantine/`)

학습에 들어가지 않을 shards. 287 files 격리됨.

| 카테고리 | files | rows | 격리 이유 |
|---|---:|---:|---|
| LibriSpeech eval splits | 4 | ~5k | `test.clean`/`test.other`/`validation.*` — Stage-2 `eval_librispeech_wer.py`와 동일 셋 (leak) |
| LibriSpeech train (vanilla) | 19 | 281,241 | 사용자 결정 ("1: 안 넣어도 될듯") — libri_mls_vox에 LibriSpeech 미포함이지만 외부 build로 추가됨 |
| MLS local train | 248 | 3,706,608 | libri_mls_vox MLS 10.8M와 중복 가능성, 사용자 "ㅇㅇ 그래라" |
| Emotion Set A (canonical 7-class) | 10 | ~62k | Set B (native) 채택 결정. 추가로 `meld_test_*` (2,610 rows)는 Stage-2 MELD eval과 정확 일치 (leak) |
| Emotion Set B (Agent E 1차) | 6 | – | 2차 빌드 (07:41-42)로 superseded |

## 포맷 스펙 (manifest row)

```jsonc
// 1) Sound captioning — captions: list of strings
{"modality":"audio_env_sound","source":"<src>","audio_path":"...","captions":["..."]}

// 2) Speech ASR — text 단일 string
{"modality":"audio_asr","source":"librispeech|mls|voxpopuli|unknown_asr",
 "audio_path|nubes_path":"...","text":"..."}

// 3) Emotion MCQA — choices 셔플됨 (seed=42 + row_index), answer = letter
{"modality":"audio_emotion","source":"<src>","audio_path":"...",
 "question":"What emotion is expressed in this audio clip?",
 "choices":["A. anger","B. happy",...],"answer":"D"}
```

Shard 크기: 15,000 rows. 파일명: `<src>_<split>_<NNNN>.jsonl`.

## Loader 동작

`audio_env_sound` 행은 `captions` 필드 존재 시 sound_caption pool 경로로 라우팅 (`TASK_PROMPTS["sound_caption"]`에서 prompt 샘플링, `captions` 리스트에서 random 1개 → assistant target). Legacy `labels`-기반 esc50 shard는 백워드 호환 유지 (현 v3 dir에는 없음).

`audio_emotion` 행은 native MCQA dispatch — `question` + `choices` 그대로 user prompt로 표시, `answer` letter 그대로 target.

## 라이선스 (요약)

| Source | License | 출처 |
|---|---|---|
| LibriSpeech (eval만) | CC BY 4.0 | OpenSLR |
| MLS | CC BY 4.0 | OpenSLR / FAIR |
| VoxPopuli | CC0 (transcripts), 다양 (audio) | FAIR |
| LibriTTS-R | CC BY 4.0 | Google |
| Clotho | Tampere Univ. dataset license (research only) | Zenodo |
| AudioCaps | MIT | github/cdjkim/audiocaps |
| MACS | CC BY 4.0 | Zenodo (TAU 2019 base) |
| AudioSet | CC BY 4.0 (annotations) / 원본 YouTube | Google |
| FSD50K | CC BY 4.0 | Zenodo |
| LAION-Audio-630K (Freesound/BBC/Epidemic/Audiostock) | LAION 공개 (clip 별 라이선스 상이; CC0/CC-BY 등) | LAION |
| IEMOCAP / MELD / DailyTalk / EmoV-DB / MUStARD++ / RAVDESS / CREMA-D | 각 academic-only EULA | – |

상업 배포 시 academic-only 클립 분리 필요.

## 빌드 재현

```bash
cd /mnt/ddn/users/jos/audiollm-trainer

# 1) LAION 추출 (~1h, 백그라운드)
python scripts/laion_audio_630k/extract_freesound.py    # ~30 min
python scripts/laion_audio_630k/extract_bbc.py          # ~5 min
python scripts/laion_audio_630k/extract_epidemic.py     # ~10 min
# Audiostock은 추출 불필요 (이미 mp3 개별 파일)

# 2) Manifest builders (sound captioning + classification → caption)
python scripts/manifest_builders/build_audioset.py            # ontology→cap, AudioSet bal_train
python scripts/manifest_builders/build_fsd50k.py              # ontology→cap, FSD50K dev
python scripts/manifest_builders/build_audiocaps.py           # parquet 추출 + caption
python scripts/manifest_builders/build_clotho.py              # dev+val 5 cap/clip
python scripts/manifest_builders/build_audiostock.py          # meta.csv → 1 cap
python scripts/manifest_builders/build_macs.py                # TAU2019 zip 추출 (오래 걸림)

# 3) Emotion MCQA (Set B: per-dataset native classes)
python scripts/manifest_builders/build_emotion_iemocap.py
python scripts/manifest_builders/build_emotion_meld.py
python scripts/manifest_builders/build_emotion_cremad.py
python scripts/manifest_builders/build_emotion_dailytalk.py
python scripts/manifest_builders/build_emotion_emovdb.py
python scripts/manifest_builders/build_emotion_ravdess.py
python scripts/manifest_builders/build_emotion_mustardpp.py

# 4) ASR conversion (sehyun nubes shard → v3 modality 필드 prepend)
python scripts/manifest_builders/convert_libri_mls_vox.py     # 128 shards 1:1 변환

# 5) 학습 시작
bash scripts/ASR/run_stage1_dac_vae_v3.sh
```

## 변경 이력

- **v3 init** (2026-05-01): 5-agent 병렬 빌드 (T1-T11), loader 확장 (T12), config (T13).
- 진행 + 결정 추적: [`docs/v3_progress_log.md`](../v3_progress_log.md).
