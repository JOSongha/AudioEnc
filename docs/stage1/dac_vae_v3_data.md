# Stage-1 DAC-VAE v3 — 학습 데이터 카탈로그

> Config: [`configs/ASR/stage1_dac_vae_v3.yaml`](../../configs/ASR/stage1_dac_vae_v3.yaml)
> Run script: [`scripts/ASR/run_stage1_dac_vae_v3.sh`](../../scripts/ASR/run_stage1_dac_vae_v3.sh)
> Manifest dir: `/mnt/tmp/datasets/manifests/v3/`
> Build log: [`docs/v3_progress_log.md`](../v3_progress_log.md)

## 개요

v2(libri+mls+vox 단독)에서 sound captioning 9 dataset + emotion MCQA 7 dataset 추가. 모달리티 3종을 단일 manifest dir에서 IterableDataset로 mix.

| Modality | Field schema | Loader path |
|---|---|---|
| `audio_asr` | `nubes_path`/`audio_path` + `text` | `omni_dataset.py:_build_prompt_targets:audio_asr` |
| `audio_env_sound` | `audio_path` + `captions:[...]` | `omni_dataset.py:_build_prompt_targets:audio_env_sound` (caption pool) |
| `audio_emotion` | `audio_path` + `question`/`choices`/`answer` | `omni_dataset.py:_build_prompt_targets:audio_emotion` (MCQA) |

## 데이터셋별 통계

### ASR (audio_asr)

| Source | rows | shards | 비고 |
|---|---:|---:|---|
| librispeech + mls + voxpopuli (combined) | 11,345,224 | 128 | nubes에서 스트리밍 (libri_mls_vox_*.jsonl) |

### Sound captioning (audio_env_sound)

| Source | split | rows | shards | Caption 출처 |
|---|---|---:|---:|---|
| laion_freesound | train_1 | 235,520 | 16 | tar 안 `<id>.json:text` (1-2 caps) |
| laion_freesound | train_2 | 178,607 | 12 | 〃 |
| laion_freesound | test | 46,014 | 4 | 〃 |
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
| fsd50k | dev | 40,966 | 3 | 〃 |
| macs | – | _pending_ | _pending_ | 3-annotator caps (list) |
| **subtotal** | | **~787,000+** | **~52** | |

### Emotion MCQA (audio_emotion)

7 datasets — pending Agent E completion. 라벨 캐논화: `{neutral, happy, sad, angry, fear, disgust, surprise}` (7-way MCQA).

| Source | rows | shards | 라벨 매핑 비고 |
|---|---:|---:|---|
| iemocap | _pending_ | _pending_ | – |
| meld | _pending_ | _pending_ | – |
| crema_d | _pending_ | _pending_ | – |
| dailytalk | _pending_ | _pending_ | – |
| emov_db | _pending_ | _pending_ | – |
| mustard | _pending_ | _pending_ | – |
| ravdess | _pending_ | _pending_ | calm→neutral 머지 후보 |

## 포맷 스펙 (manifest row)

```jsonc
// 1) Sound captioning
{"modality":"audio_env_sound","source":"<src>","audio_path":"...","captions":["..."]}

// 2) Speech ASR
{"modality":"audio_asr","source":"librispeech|mls|voxpopuli","audio_path|nubes_path":"...","text":"..."}

// 3) Emotion MCQA
{"modality":"audio_emotion","source":"<src>","audio_path":"...","question":"...","choices":["A. ...","B. ...","..."],"answer":"<letter>"}
```

Shard 크기: 15,000 rows. 파일명: `<src>_<split>_<NNNN>.jsonl`.

## Loader 동작

`audio_env_sound` 행은 `captions` 필드 존재 시 항상 sound_caption pool 경로로 라우팅 (`TASK_PROMPTS["sound_caption"]`에서 prompt 샘플링, `captions` 리스트에서 random 1개 → assistant target). Legacy `labels`-기반 fsd50k/esc50 shard는 백워드 호환 유지 ([`omni_dataset.py:282-307`](../../src/llamafactory/data/omni_dataset.py#L282-L307)).

## 라이선스 (요약)

| Source | License | 출처 |
|---|---|---|
| LibriSpeech | CC BY 4.0 | OpenSLR |
| MLS | CC BY 4.0 | OpenSLR / FAIR |
| VoxPopuli | CC0 (transcripts), 다양 (audio) | FAIR |
| Clotho | Tampere Univ. dataset license (research only) | Zenodo |
| AudioCaps | MIT | github/cdjkim/audiocaps |
| MACS | CC BY 4.0 | Zenodo |
| AudioSet | CC BY 4.0 (annotations) / 원본 YouTube | Google |
| FSD50K | CC BY 4.0 | Zenodo |
| LAION-Audio-630K (BBC/Freesound/Epidemic/Audiostock) | LAION 공개 (개별 클립 라이선스 상이) | LAION |
| IEMOCAP / MELD / CREMA-D / DailyTalk / EmoV-DB / MUStARD / RAVDESS | 각 데이터셋 이용 약관 (대부분 academic-only) | – |

상업 배포 시 LAION/IEMOCAP 등 academic-only 클립 분리 필요.

## 빌드 재현

```bash
cd /mnt/ddn/users/jos/audiollm-trainer
# 1) LAION 추출 (이미 완료)
python scripts/laion_audio_630k/extract_freesound.py
python scripts/laion_audio_630k/extract_bbc.py
python scripts/laion_audio_630k/extract_epidemic.py
# 2) Manifest builders
python scripts/v3_manifest/build_audioset.py
python scripts/v3_manifest/build_fsd50k.py
# (audiocaps/clotho/macs/audiostock/emotion: scripts/env_sound/, scripts/emo/ 참조)
# 3) ASR conversion
# (Agent D 산출물: libri_mls_vox_*.jsonl 128 shards 이미 v3 dir에 존재)
```

## 변경 이력

- v3 init: 2026-05-01 (이 문서). 전 buildplan + 진행 추적은 `docs/v3_progress_log.md`.
