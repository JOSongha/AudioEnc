# Stage-1 v3 (DAC-VAE) — 작업 단톡방

> **목적**: libri+mls+vox + sound (caption) + emotion (7) + LAION (4) + AudioSet + FSD50K + MACS 학습용 manifest 빌드
> **포맷 스펙**: 모든 row JSON, 모달리티 별 필드 통일 (§ 포맷 스펙 참조)
> **출력**: manifest = `/mnt/tmp/datasets/manifests/v3/`, 추출 audio = `/mnt/tmp/datasets/laion_extracted/<src>/`
> **DDN write 금지** (small repo files만)

---

## § 포맷 스펙

```jsonc
// Sound captioning (단일 또는 multi caption)
{"modality": "audio_env_sound", "source": "<src>", "audio_path": ".../x.flac", "captions": ["..."]}
//   src ∈ {clotho, audiocaps, macs, laion_freesound, laion_bbc, laion_epidemic, laion_audiostock, audioset, fsd50k}
//   audioset/fsd50k는 ontology→synthesized captions

// Speech ASR (libri/mls/vox)
{"modality": "audio_asr", "source": "librispeech|mls|voxpopuli", "audio_path"|"nubes_path": "...", "text": "..."}

// Emotion classification (MCQA)
{"modality": "audio_emotion", "source": "<src>", "audio_path": "...", "question": "...", "choices": ["A. happy", ...], "answer": "D"}
```

Shard 크기: 15000 rows/shard. 파일명: `<src>_<split>_<NNNN>.jsonl`.

---

## § Task 분배표

| ID | Task | 담당 | 상태 | Output |
|---|---|---|---|---|
| T1 | LAION Freesound 추출 + manifest | (직접 실행) | ✅ 완료 (2026-05-01 04:xx) | 460,141 clips, 32 shards, 607 GB FLAC |
| T2 | LAION BBC 추출 + manifest | (직접 실행) | ✅ 완료 (2026-05-01 ~04:30) | 31,936 clips, 2 shards |
| T3 | LAION Epidemic 추출 + manifest | (Agent A) | ✅ 완료 (2026-05-01 07:30) | 75,645 clips, 6 shards |
| T4 | LAION Audiostock manifest (추출 불필요) | (Agent D) | ✅ 완료 (2026-05-01 07:21) | 9,139 clips, 1 shard |
| T5 | AudioSet bal_train manifest (ontology→cap) | (Agent B) | ✅ 완료 (2026-05-01 07:19) | 18,683 clips, 2 shards |
| T6 | FSD50K dev manifest (ontology→cap) | (Agent B) | ✅ 완료 (2026-05-01 07:19) | 40,966 clips, 3 shards |
| T7 | AudioCaps train+val manifest (parquet 추출) | (Agent C) | ✅ 완료 (2026-05-01 07:25) | 45,623 clips, 5 shards |
| T8 | Clotho dev+val manifest (rebuild) | (Agent C) | ✅ 완료 (2026-05-01 07:25) | 4,881 clips, 2 shards |
| T9 | MACS 추출 + manifest | (Agent C) | 🔄 진행 (TAU2019 36GB DL 중) | (예상 ~3,930 clips) |
| T10 | libri/mls/vox conversion (modality 필드 추가) | (Agent D) | ✅ 완료 (2026-05-01 07:21) | 128 shards (11.35M rows) |
| T11 | Emotion 7 manifest (iemocap, meld, CREMA-D, DailyTalk, EmoV-DB, MUStARD, RAVDESS) | Agent E | ✅ 완료 (2026-05-01 07:32) | 61,547 rows / 7 sources, canonical 7-class fixed-order MCQA, 10 shards (MUStARD audio 1/1202만 존재) |
| T12 | Loader extension (omni_dataset.py + omni_dataset_whisper.py) | (직접/완료자) | ✅ 완료 (이미 수정됨) | captions 우선, labels 폴백 |
| T13 | stage1_dac_vae_v3.yaml + run_stage1_dac_vae_v3.sh | (직접/완료자) | ✅ 완료 (이미 작성됨) | configs/ASR/stage1_dac_vae_v3.yaml + scripts/ASR/run_stage1_dac_vae_v3.sh |
| T14 | docs/stage1/dac_vae_v3_data.md (학습 데이터 카탈로그) | (직접) | ✅ 골격 완료 (2026-05-01 07:30) | docs/stage1/dac_vae_v3_data.md (MACS/Emotion 통계는 빌드 완료 후 기입) |

---

## § 병렬화 계획 (agent dispatch 안)

병렬 dispatch 가능한 묶음 (각 agent는 self-contained, 다른 agent 출력 의존 안 함):

- **Agent A**: T3 (Epidemic 추출) — 가장 무거운 I/O, 단독
- **Agent B**: T5 + T6 (AudioSet + FSD50K, ontology 캡션 합성, 같은 로직)
- **Agent C**: T7 + T8 + T9 (AudioCaps + Clotho + MACS, sound captioning)
- **Agent D**: T4 + T10 (Audiostock manifest + libri/mls/vox conversion, 둘 다 가벼움)
- **Agent E**: T11 (Emotion 7개, 각 데이터셋 조사 필요)
- **Direct (me)**: T12 (loader 코드), T13 (yaml/sh), T14 (doc)

병렬 cap: 5 agent + 직접 작업.

---

## § 실시간 업데이트 로그

### [2026-05-01 04:00] 시작
- 전제: Freesound DL 완료 (608 GB), BBC/Epidemic/AudioCaps/Audiostock 다운 완료
- v3 = libri/mls/vox + sound + emotion + LAION 4 + AudioSet + FSD50K + MACS

### [2026-05-01 04:xx] T1 ✅ Freesound 추출 완료
- 460,141 clips → `/mnt/tmp/datasets/laion_extracted/freesound/<id>.flac`
- 32 manifest shards: `/mnt/tmp/datasets/manifests/v3/laion_freesound_*.jsonl`
- 캡션: tar 안 `<id>.json`의 `text` 필드 (1-2개), `tag`/`original_data` 무시
- 스크립트: `scripts/laion_audio_630k/extract_freesound.py`

### [2026-05-01 ~04:30] T2 ✅ BBC 추출 완료
- 31,936 clips (test 3194 + train 28742) → `/mnt/tmp/datasets/laion_extracted/bbc/`
- 2 manifest shards
- 캡션: 단일 `text` 필드
- 스크립트: `scripts/laion_audio_630k/extract_bbc.py`

### [2026-05-01 ~07:00] 단톡방 시스템 도입
- 사용자 요청으로 이 md 파일 생성 (지금까지 진행 사항 기록 + agent 분배)
- 다음: 사용자 승인 후 5개 agent 병렬 dispatch + loader 작업

### [07:15] T5+T6 (1차 빌드, deprecated)
- 초기 빌드: 2-caption 포맷 ("Audio of A, B." / "the sounds of A, ..., and Z.") + `labels` 필드 포함
- 그러나 § 포맷 스펙은 단일 캡션 "sound of ..." 포맷 — 07:19에 spec 준수로 재빌드 (아래 참조)

### [2026-05-01 07:19] T5 ✅ AudioSet bal_train manifest (Agent B, spec 준수 재빌드)
- 18,683 clips kept (drop 없음 — `human_labels` 비어있는 row 0, 누락 audio 0)
- ontology 632 entries 로드, 캡션 합성: lowercase + comma-sep, ≥3 라벨 시 "and" 직전 콤마
  - 예: `sound of speech, gush` (2 labels), `sound of goat, music, and speech` (3 labels)
- 단일 caption만 (spec 준수). `labels` 필드 제거 (spec 미정의).
- 2 shards: `audioset_bal_train_0000.jsonl` (15,000) + `audioset_bal_train_0001.jsonl` (3,683)
- 출력: `/mnt/tmp/datasets/manifests/v3/audioset_bal_train_*.jsonl`
- 스크립트: `scripts/v3_manifest/build_audioset.py`
  (audio는 prepare_manifest.py가 미리 추출해둔 `/mnt/tmp/datasets/env_sound/AudioSet/audio/<vid>.flac` 그대로 사용)
- 1차 빌드 stale shards는 스크립트가 시작 시 unlink 후 재작성

### [2026-05-01 07:25] T7+T8 ✅ AudioCaps + Clotho manifests (Agent C, T9 MACS 진행 중)
- **AudioCaps train+val** (T7): 45,623 unique audios → 5 shards
  - train: 45,178 rows → 45,178 unique audios (1 caption each, no grouping happened) → 4 shards
  - val: 2,223 rows → 445 unique audios (~5 captions each, grouped by `(youtube_id, start_time)`) → 1 shard
  - FLAC bytes 추출 → `/mnt/tmp/datasets/laion_extracted/audiocaps/<youtube_id>_<start_time>.flac`
  - test split 제외 (Stage-2 held-out)
- **Clotho dev+val** (T8): 4,881 clips → 2 shards
  - dev: 3,837 clips (3,839 CSV rows, 2 missing audio) × 5 captions → 1 shard
  - val: 1,044 clips (1,045 CSV rows, 1 missing audio) × 5 captions → 1 shard
  - evaluation split 제외 (Stage-2 held-out)
- **MACS** (T9): MACS_audio.tar.gz가 0 bytes (Zenodo 5114771 record에 audio 없음)
  - 실제 source = TAU Urban Acoustic Scenes 2019 Development (Zenodo 2589280, 21 zip × ~1.7GB ≈ 36GB)
  - target = airport/park/public_square 3,930 files only
  - DL→extract MACS targets only→delete zip 방식, 별도 백그라운드 진행 (완료 후 row 표 갱신)
- 스크립트: `scripts/v3_manifest/build_{audiocaps,clotho,macs}.py`

### [2026-05-01 07:19] T6 ✅ FSD50K dev manifest (Agent B, spec 준수 재빌드)
- 40,966 clips kept (dev.csv 전체 = train+val sub-split, drop 없음)
- AudioSet ontology 재사용 (FSD50K = subset). 라벨 정규화: `mid` 우선 lookup, fallback은 underscore→space
  - 예: `Electric_guitar` → ontology 표시명 `Electric guitar` → lowercased `electric guitar`
- 단일 caption만 (spec 준수). `labels` 필드 제거.
- 3 shards: `fsd50k_dev_0000.jsonl` (15,000) + `fsd50k_dev_0001.jsonl` (15,000) + `fsd50k_dev_0002.jsonl` (10,966)
- 출력: `/mnt/tmp/datasets/manifests/v3/fsd50k_dev_*.jsonl`
- 스크립트: `scripts/v3_manifest/build_fsd50k.py`

### [07:30] 🚨 Eval contamination 격리 + 중복 가능성 발견 (직접)

- 누군가(분배 외) `librispeech_*` (test_clean/test_other/validation_clean/validation_other) 4 shards을 `v3/`에 추가했음 → Stage-2 `eval_librispeech_wer.py` 평가 셋과 정확히 일치하는 leak 가능성. 즉시 `/mnt/tmp/datasets/manifests/v3_quarantine/`로 격리.
- 추가로 발견된 신규 shards (분배 spec 외):
  - `librispeech_train_clean_100|360|other_500` (19 shards, 281,241 rows) — vanilla LibriSpeech (local audio_path = `/mnt/ddn/omni_dataset/audio/librispeech/...`)
  - `mls_train_part_*` (248 shards, 3,706,608 rows) — MLS via local — `libri_mls_vox`의 MLS 10.8M 부분과 잠재 중복
- 결정 대기 중:
  - librispeech_train_* (281k) 유지/드롭
  - mls_train_part_* vs libri_mls_vox MLS 중 어느 쪽 사용

### [07:30] T3 ✅ LAION Epidemic 추출 완료 (Agent A)
- 75,645 rows → 6 shards
- 출력: `/mnt/tmp/datasets/laion_extracted/epidemic/<safe_index>.flac` + `/mnt/tmp/datasets/manifests/v3/laion_epidemic_*.jsonl`
- 스크립트: `scripts/laion_audio_630k/extract_epidemic.py`

### [07:21] T4 + T10 ✅ Audiostock manifest + libri/mls/vox 변환 (Agent D)
- **Audiostock** (T4): 9,139 clips → 1 shard (`laion_audiostock_0000.jsonl`)
  - meta.csv 10,001 rows → status!=ok 862 drop, missing audio 0, size<=1024B 0, empty caption 0
- **libri/mls/vox** (T10): 128 shards converted, 11,345,224 rows total
  - source breakdown: LibriSpeech 0, MLS 10,808,037, VoxPopuli 182,466, unknown 354,721
  - unknown_asr 354,721은 모두 LibriTTS-R (path = `en_LibriTTS_R_single/...`); 스펙대로 "librispeech" 문자열 매칭이 안 되어 unknown으로 분류됨. 이 데이터셋에 LibriSpeech 자체는 포함되지 않은 것으로 확인.
  - 1:1 shard mapping (no resharding); 기존 `nubes_path`/`text` 그대로 보존 + `modality`/`source` prepend
- 스크립트: `scripts/v3_manifest/build_audiostock.py` + `scripts/v3_manifest/convert_libri_mls_vox.py`

### [2026-05-01 07:30] T12+T13+T14 ✅ Direct 작업 완료
- **T12 loader**: `omni_dataset.py:282-307` + `omni_dataset_whisper.py:99-122` — `_build_prompt_targets()` 의 `audio_env_sound` 분기를 captions 우선·labels 백워드 호환으로 단순화. v3 9개 source (clotho/audiocaps/macs/laion_*/audioset/fsd50k) 모두 captions 라우팅 1줄로 통일.
- **T13 yaml/sh**: `configs/ASR/stage1_dac_vae_v3.yaml` + `scripts/ASR/run_stage1_dac_vae_v3.sh` — v2 카피 후 `omni_manifest=/mnt/tmp/datasets/manifests/v3`, run_name/output_dir → v3.
- **T14 doc**: `docs/stage1/dac_vae_v3_data.md` 골격 완료 (modality/dataset 표 + 라이선스 요약 + 빌드 재현 절차). MACS/Emotion 통계는 Agent C/E 완료 후 채워 넣기.

### [2026-05-01 07:30] 미해결 이슈
- **libri/mls/vox 중복**: `libri_mls_vox_*` (128 shards, nubes_path) + `mls_train_part_*` (248 shards, local audio_path) + `librispeech_train_*` (19 shards, local) 동시 존재. 격리 4 shards (eval set leak)는 v3_quarantine 처리 완료. **결정 필요**: 사용 manifest 한쪽으로 통일 — local audio_path 쪽 권장 (nubes 의존 제거).
- **Agent E (T11 emotion)**: ✅ 완료 (아래 [07:32] 로그 참조).
- **MACS DL**: 백그라운드 wget (PID 26182) 8/21 zip · 12GB/36GB · 1h55m 진행 중. 완료까지 대기.
- **tmux**: 미설치 — 사용자 직접 `sudo apt install tmux` 진행 중.

### [2026-05-01 07:32] T11 ✅ Emotion 7 manifest (Agent E)

- **단일 스크립트**: `scripts/emo/build_emotion_v3_manifest.py` (per-dataset 함수 구조)
- **Canonical 7-class (FIXED 순서, 모든 row 동일)**:
  `["A. happy","B. sad","C. angry","D. neutral","E. fear","F. disgust","G. surprise"]`
  question = `"What is the emotion expressed?"`
- **출력**: `/mnt/tmp/datasets/manifests/v3/emotion_<src>_<split>_<NNNN>.jsonl` (15k/shard)
  shard 10개 / 총 **61,547 clips**

| src         | location (root)                                                                                          | kept    | drop_label | miss_path | merges / notes |
|---|---|---|---|---|---|
| iemocap     | sehyun shards + `/mnt/ddn/kyudan/IEMOCAP/data/`                                                         | 10,013  | 26         | 0         | excited→happy, frustrated→angry; `other` (26) drop |
| meld        | `/mnt/tmp/datasets/emotion_raw/MELD/audio/{train,dev,test}/`                                            | 13,706  | 0          | 2         | joy→happy, sadness→sad, anger→angry; train+dev+test 별도 shard |
| crema_d     | `/mnt/ddn/shkim/.../dataset_emotion/Source/CREMA-D/AudioWAV/`                                           | 7,442   | 0          | 0         | 파일명 ANG/DIS/FEA/HAP/NEU/SAD → canonical 1:1 |
| dailytalk   | `/mnt/tmp/datasets/emotion_raw/DailyTalk/dailytalk/data/`                                               | 23,773  | 0          | 0         | "no emotion"→neutral; metadata.json keyed (dialog_id, utt_id) |
| emov_db     | `/mnt/tmp/datasets/emotion_raw/EmoV-DB/{bea,jenie,josh,sam}/<Emo>/`                                     | 5,172   | 1,721      | 0         | Amused→happy, Disgusted→disgust; **Sleepy(1,721) drop** (no canonical) |
| mustard     | `/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/videos/{augmented_utterance,final_context_videos}/`    | **1**   | 0          | 1,201     | CSV labeled 1,202 rows / disk has only 1 utterance video (`1_60_u.mp4`) — videos 미다운로드. Label=Excitement→happy. **MUStARD essentially missing.** |
| ravdess     | `/mnt/tmp/datasets/emotion_raw/RAVDESS/Actor_*/`                                                        | 1,440   | 0          | 0         | calm→neutral (merge per spec); fearful→fear, surprised→surprise |
| **TOTAL**   |                                                                                                          | **61,547** |          |           |  |

- **Canonical 7-class merges (cross-dataset summary)**:
  - calm (RAVDESS) → neutral
  - excited / Excitement / Amused → happy
  - frustrated / Frustration → angry
  - fearful → fear; surprised → surprise; sadness → sad; anger → angry; joy/Happiness → happy; disgusted → disgust
  - drop: IEMOCAP `other` (26), EmoV-DB `sleepy` (1,721), MUStARD `Ridicule` (CSV-only, no audio anyway)

- **Spot-check (1 row each)**:
  - iemocap:  `/mnt/ddn/kyudan/IEMOCAP/data/Ses01F_impro01_F000.wav` answer=D (neutral) ✓
  - meld:     `/mnt/tmp/.../MELD/audio/train/dia0_utt0.wav` answer=D (neutral) ✓
  - crema_d:  `/mnt/ddn/shkim/.../1001_DFA_ANG_XX.wav` answer=C (angry) ✓
  - dailytalk:`/mnt/tmp/.../DailyTalk/.../0/0_1_d0.wav` answer=D (neutral) ✓
  - emov_db:  `/mnt/tmp/.../EmoV-DB/bea/Amused/amused_1-15_0001.wav` answer=A (happy) ✓
  - mustard:  `/mnt/tmp/.../MUStARD_Plus_Plus/.../1_60_u.mp4` answer=A (happy=Excitement) ✓
  - ravdess:  `/mnt/tmp/.../RAVDESS/Actor_01/03-01-01-01-01-01-01.wav` answer=D (neutral; emotion idx=01) ✓

- **Missing/incomplete datasets** (not skipped — partials manifested):
  - **MUStARD**: 본격 사용 불가 (1 row만 존재, 나머지 1,201 rows 모두 audio missing). 다운로드 보강 필요.
  - **CREMA-D**: 원본 `emotion_raw/CREMA-D/AudioWAV` 비어 있음 → shkim의 mirror 사용으로 회수.
  - **IEMOCAP**: 라이선스 데이터, sehyun shards에 path가 sehyun cache로 박혀 있어 → kyudan IEMOCAP wav로 remap. 모두 존재 확인 (10,039 wav 중 10,013 mappable).
- **사이드 메모**: 기존 `scripts/v3_manifest/build_emotion_*.py` 스크립트들은 per-source labels + shuffled choices를 사용 — T11 spec과 충돌하므로 새 unified 스크립트로 대체. (기존 파일은 유지, 빌드 진입점은 `scripts/emo/build_emotion_v3_manifest.py`.)
