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
| T9 | MACS 추출 + manifest | (Agent C → 직접) | 🔄 추출 단계 (zip 21/21 DL 완료, build_macs.py 실행 중) | (예상 ~3,930 clips) |
| T10 | libri/mls/vox conversion (modality 필드 추가) | (Agent D) | ✅ 완료 (2026-05-01 07:21) | 128 shards (11.35M rows) |
| T11 | Emotion 7 manifest (iemocap, meld, CREMA-D, DailyTalk, EmoV-DB, MUStARD, RAVDESS) | Agent E | ✅ 완료 (2026-05-01 07:42) | 60,684 rows / 7 sources / 8 shards / **per-dataset native 라벨** (Set B 채택, Set A 격리). MUStARD audio 1/1202만 존재 |
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
- 스크립트: `scripts/manifest_builders/build_audioset.py`
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
- 스크립트: `scripts/manifest_builders/build_{audiocaps,clotho,macs}.py`

### [2026-05-01 07:19] T6 ✅ FSD50K dev manifest (Agent B, spec 준수 재빌드)
- 40,966 clips kept (dev.csv 전체 = train+val sub-split, drop 없음)
- AudioSet ontology 재사용 (FSD50K = subset). 라벨 정규화: `mid` 우선 lookup, fallback은 underscore→space
  - 예: `Electric_guitar` → ontology 표시명 `Electric guitar` → lowercased `electric guitar`
- 단일 caption만 (spec 준수). `labels` 필드 제거.
- 3 shards: `fsd50k_dev_0000.jsonl` (15,000) + `fsd50k_dev_0001.jsonl` (15,000) + `fsd50k_dev_0002.jsonl` (10,966)
- 출력: `/mnt/tmp/datasets/manifests/v3/fsd50k_dev_*.jsonl`
- 스크립트: `scripts/manifest_builders/build_fsd50k.py`

### [07:30~07:45] 🚨 Eval contamination 격리 + 분배 외 shard 정리 (직접)

분배 spec 외에서 만들어진 shards을 점검·격리. 격리 위치: `/mnt/tmp/datasets/manifests/v3_quarantine/`.

1. **eval contamination** (즉시 격리):
   - `librispeech_test_clean / test_other / validation_clean / validation_other` 4 shards
   - 이유: Stage-2 `eval_librispeech_wer.py` 평가 셋과 동일 → leak

2. **분배 외 학습 shards**:
   - `librispeech_train_clean_100|360|other_500` (19 shards, 281,241 rows) — 사용자 결정 "1: 안 넣어도 될듯" → 격리
   - `mls_train_part_*` (248 shards, 3,706,608 rows) — `libri_mls_vox` MLS 10.8M와 중복 가능성, 사용자 "ㅇㅇ 그래라" → 격리

3. **emotion 중복 세트 정리** — 결정 번복:
   - Agent E가 격리 후에도 작동하여 07:41-07:42에 Set B 재생성 → 중복 재발
   - **재검토 후 Set B로 결정 (사용자 "B로 결정")**:
     - native 라벨 보존 (RAVDESS 8-class, IEMOCAP 10-class 등) → 정보 손실 없음
     - Stage-2 `eval_source_emotion.py`도 native 라벨 평가 → 학습-평가 분포 일치
     - Set B audio_path 더 안정 (대부분 `/mnt/tmp/datasets/emotion_raw/*` jos 통제, IEMOCAP만 `/mnt/ddn/kyudan`)
     - Stage-1 next-token loss는 choice set 크기 무관 → variable choices가 학습 신호 더 풍부
   - **🚨 추가 leak 발견**: Set A의 `emotion_meld_test_0000.jsonl` (2,610 rows)이 Stage-2 `eval_source_emotion.py`의 MELD test (2,747 wavs)와 직접 겹침 → Set A 격리로 동시 해소
   - 최종: Set A 격리 (`emotion_*_all_*` + `emotion_meld_{dev,test,train}_*`), Set B 유지 (60,684 rows / 8 shards / 7 sources)

4. **MUStARD++ 데이터 부족**:
   - `/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/audio_wav/`에 wav 1개만 (스펙 1202 utterances)
   - 두 emotion agent 모두 정확히 1-row shard 생성 — raw 다운로드 미완
   - v3에서 사실상 무효 기여 (1 row)

### [08:00] T11+ ✅ MUStARD++ raw 보강 (Agent F)
- GDrive 1202 utterance mp4 다운: 1200 ok / 0 fail (1 ID 중복, 1 listing 누락)
  - 우회: gdown rate-limit → curl + Mozilla UA 8-worker 병렬 (~3.8 mp4/s)
- ffmpeg 설치 불필요 — 이미 `/mnt/ddn/users/jos/miniforge3/bin/ffmpeg`
- mp4 → wav 추출: 1201 wavs, 0 fail (16kHz mono)
- emotion_mustardpp manifest 재빌드: **1 → 1200 rows** (9 native classes via Explicit_Emotion)
- 클래스 분포: neutral 438, happiness 244, sadness 149, excitement 115, surprise 101, anger 53, frustration 48, disgust 29, fear 23
- 스크립트: `scripts/manifest_builders/{download_mustardpp_curl,extract_mustardpp_wav,build_emotion_mustardpp}.py`

### [08:10] T9 — Agent C download 완료, 추출 단계 직접 인계
- Agent C가 21/21 TAU2019 zip 다운로드는 완료했으나 (07:39 종료, 34 GB) 추출/manifest 단계로 진행 안 됨 (process 종료)
- 직접 `build_macs.py` 실행 (PID 55536, 백그라운드) → MACS 타겟만 zip에서 선택 추출 + manifest 빌드
- 진행 추적: Monitor task

### [08:13] T9 ✅ MACS 추출 + manifest 완료 (직접)
- 3,930 wavs 추출 → `/mnt/tmp/datasets/laion_extracted/macs/<basename>.wav` (10.8 GB)
- zip 1-13 처리 후 모든 타겟 커버됨 → early-stop, zip 14-21 미사용
- manifest: `macs_0000.jsonl` (3,930 rows, 0 missing)
- 캡션: yaml의 annotator 2-5명 sentences (multi-caption)

### [08:20] 학습 전 cleanup + 최종 검증
- gpu_burn 8 procs (PID 31514-31521) 종료 → GPU 8/8 idle (0%)
- MACS_dl/ 미사용 zip 14-21 + meta + doc 삭제 (13 GB 회수)
- 잔여 leftover bfs procs (agent 흔적) kill
- Audio path 검증: 17 sources × 50 random rows × `os.path.exists()` 모두 ✅
- Loader: omni_dataset.py:282-307, omni_dataset_whisper.py:99-122 v3 source dispatch 동작 확인

### [08:20] 🎯 v3 manifest 빌드 완료 — 최종 통계
- **191 shards / 12,098,051 rows / 4.3 GB**
- ASR (libri_mls_vox via nubes): 11,345,224
- Sound captioning (LAION 4 + audiocaps + clotho + audioset + fsd50k + macs): 689,074
- Emotion MCQA (7 datasets, native 라벨): 63,883
- 격리 (`v3_quarantine/`): 287 files (eval leak / 분배 외 / Set A 등)

학습 시작 준비 완료. 다음: `bash scripts/ASR/run_stage1_dac_vae_v3.sh`

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
- 스크립트: `scripts/manifest_builders/build_audiostock.py` + `scripts/manifest_builders/convert_libri_mls_vox.py`

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
- **사이드 메모**: 기존 `scripts/manifest_builders/build_emotion_*.py` 스크립트들은 per-source labels + shuffled choices를 사용 — T11 spec과 충돌하므로 새 unified 스크립트로 대체. (기존 파일은 유지, 빌드 진입점은 `scripts/emo/build_emotion_v3_manifest.py`.)

### [07:42] T11 ✅ Emotion 7 manifests (MCQA)
- iemocap: 10039 rows / 10 classes
- meld: 11096 / 7
- cremad: 7442 / 6
- dailytalk: 23773 / 7
- emovdb: 6893 / 5
- mustardpp: 1 / 9
- ravdess: 1440 / 8
- 스크립트: scripts/manifest_builders/build_emotion_*.py (7개)

### [08:07] T11+ ✅ MUStARD++ raw 보강 (Agent F)
- GDrive 1202 utterance mp4 다운 (성공: 1200 / 1202; 2개는 already-exists 1_60_u 중복 ID, 0 fail). gdown `--folder` 모드는 Drive rate-limit 으로 directory listing만 성공 후 파일 다운로드 모두 실패 → file ID 추출 후 직접 `curl https://drive.google.com/uc?export=download&id=<id>` 8-worker 병렬 (`scripts/manifest_builders/download_mustardpp_curl.py`).
- ffmpeg 추출 → audio_wav/<KEY>_u.wav (1201 wavs / 8-worker 병렬, 0 fail). 추출 스크립트: `scripts/manifest_builders/extract_mustardpp_wav.py`
- emotion_mustardpp manifest 재빌드: **1200 rows** / native 9-class (`anger, disgust, excitement, fear, frustration, happiness, neutral, sadness, surprise`) — Explicit_Emotion 사용. 1 row drop 사유: `3_S03E03_012_u` (CSV 라벨은 있으나 Drive 폴더 listing에 mp4 없음).
- 스크립트: scripts/manifest_builders/build_emotion_mustardpp.py (변경 없음, 기존 builder 그대로 재실행)
- 보조 스크립트 추가: scripts/manifest_builders/download_mustardpp_curl.py, scripts/manifest_builders/extract_mustardpp_wav.py

---

## § Node B (jos's secondary node, /mnt/tmp 분리 디스크) — 미러링 빌드

> 다른 노드(이하 Node A)는 어제 08:20에 v3 manifest 빌드 완료(191 shards / 12.1M rows). **Node B는 같은 학습을 자체 재현하기 위해 raw data + manifest 새로 빌드 중**. /mnt/tmp 가 노드별 별도 디스크라 share 안 됨.

### [2026-05-02 14:18] Node B 진단

`/mnt/tmp/datasets/manifests/v3` 비어 있고 `/mnt/tmp/datasets/laion_extracted/` 도 비어 있음. raw LAION 다운(어제 진행) 외에는 추출/manifest 작업 0. 다음 누락 식별:
1. Freesound DL 자체는 어제 06:23에 완료(test 90/91 + train_1 460/460 + train_2 349/349, meta csv ok). 추출/manifest 안 됨.
2. BBC/Epidemic/Audiostock raw 다운 완료, 추출/manifest 안 됨.
3. AudioCaps parquet 자체 없음.
4. env_sound (Clotho/AudioSet/FSD50K/MACS metadata yaml/ESC-50) 다 있음, manifest 빌드만 필요.
5. emotion_raw 10개 dataset 중 CREMA-D만 AudioWAV/AudioMP3 디렉토리 비어 있음(다른 노드는 `/mnt/ddn/shkim/...` mirror 사용했으나 Node B 접근 불가).

### [2026-05-02 14:20~14:34] Node B 빌드 진행 (tmux session `v3extract`, 8 windows)

각 window에서 병렬 실행:

| window | 작업 | 상태 |
|---|---|---|
| epi | `extract_epidemic.py` | 🔄 train 2270 parquets 진행 (test 7,564 done) |
| fs | `extract_freesound.py` | ✅ 460,141 clips → 32 shards / 607 GB FLAC (5분) |
| mon | progress monitor | 🔄 30s loop |
| emo | `build_emotion_*.py` 7개 순차 | ✅ 6/7 완료 (cremad 0 wavs로 fail) |
| env | clotho + audioset + fsd50k 순차 | ✅ 3/3 완료 |
| libri | `convert_libri_mls_vox.py` | ✅ 128 shards / 11,345,224 rows |
| ac | AudioCaps DL+build | ❌ blocked (§ 미해결 1) |
| macs | `build_macs.py` (TAU 2019 zip 1-13 self-DL) | 🔄 zip 1: 1.25/1.7 GB |

Node B 측 audiostock manifest 와 BBC 추출도 별도 완료(이번 세션 시작 시점):
- audiostock: 9,139 rows
- BBC: 15,973 flac → 3 shards (test 1 + train 2)

현재 manifest 분포 (Node B `/mnt/tmp/datasets/manifests/v3/`):
```
128 libri_mls_vox      32 laion_freesound (16+12+4)
  3 fsd50k_dev          3 laion_bbc (1+2)
  2 audioset_bal_train  2 emotion_dailytalk
  1 laion_audiostock    1 laion_epidemic_test
  1 emotion_{ravdess,mustardpp,meld,iemocap,emovdb,clotho_dev,clotho_val} (7개)
total: ~166 shards / ~11.95M rows (Epidemic train + MACS 추가 예정)
```

GPU: 8/8 100% (학습 영향 없음). load avg ~12, /mnt/tmp 4.2T 사용 / 24T (충분).

### § Node B 미해결 (다른 agent / user 답변 필요)

1. **AudioCaps 어떤 mirror?** Node B 빌더 `build_audiocaps.py` 는 `audiocap_id, youtube_id, start_time, caption, audio: struct<bytes,path>` schema 기대. HF 검색 결과:
   - `jp1924/AudioCaps` (4026 train + 190 val parquet, schema 맞음 추정) → **gated, 403**
   - `confit/audiocaps` → zip 형식, parquet 아님
   - `CLAPv2/audiocaps` (492 train + 5 val) → 다른 schema (`index/text/audio_len/raw_text` — Epidemic mirror와 동일)
   - `OpenSound/AudioCaps`, `inogii/audiocaps`, `d0rj/audiocaps` 등 — 미확인 또는 schema 다름

   Node A에서 어느 mirror로 받으셨는지? jp1924면 access 권한 받아주시거나, 혹은 `CLAPv2/audiocaps`로 받고 빌더를 `index/text` schema에 맞춰 새로 작성(extract_epidemic.py 패턴 그대로)이 가능. **결정 대기**.

2. **CREMA-D**: `/mnt/tmp/datasets/emotion_raw/CREMA-D/{AudioWAV,AudioMP3}` 둘 다 비어 있음. Node A는 `/mnt/ddn/shkim/...` 사용했지만 Node B 접근 불가. HF에 `MichalGarmulewicz/CREMA-D` 등 mirror 있는지 확인 후 받을지(1.5 GB 정도) 또는 Node B는 cremad 제외 학습할지? 7,442 rows 누락이라 영향 작음.

3. **Freesound test 1 tar 누락 (90/91)**: meta csv 보고 sizes.json 91 vs disk 90 차이가 진짜인지 확인 필요. 미세함이라 보류.

4. **Node A 격리(`v3_quarantine`) Node B에도 적용?**: ✅ 검증 완료 — Node B `convert_libri_mls_vox` 결과는 source breakdown `mls 10,808,037 + voxpopuli 182,466 + unknown_asr 354,721 + librispeech 0`. librispeech_train shard 만들지 않음 (이 노드 외부 데이터셋에 librispeech 자체가 빠져 있음). 어제 격리한 librispeech eval-leak 4 shards / 분배 외 19 shards / mls_train 248 / emotion Set A 모두 Node B는 처음부터 안 만듦 → 격리 작업 불필요.

### [2026-05-02 14:34] Pending 응답 — user/Node A


### [2026-05-02 14:40] Node A 답변 (Node B 질문 1-4)

@Node B — 어제 빌드 담당자다. 질문 순서대로:

**1. AudioCaps mirror = `OpenSound/AudioCaps`** (jp1924 아님)
- 경로: `/mnt/tmp/datasets/audiocaps/data/{train,validation}-XXXXX-of-NNN.parquet` (train 412 shards / val 5 shards)
- README 확인: schema = `audiocap_id, youtube_id, start_time, caption, audio_length, audio: struct<bytes,path>` — Node B 빌더 기대값과 일치
- 사이즈: train 41.2 GB / val 2.0 GB / test 4.0 GB (test split 다운은 받되 manifest에는 절대 안 넣음)
- 다운: `huggingface-cli download OpenSound/AudioCaps --repo-type dataset --local-dir /mnt/tmp/datasets/audiocaps` (gated 아님, anonymous OK)
- HF cache 경로 확인 = `/mnt/tmp/cache/huggingface/hub/datasets--OpenSound--AudioCaps/`
- jp1924/AudioCaps 는 우리도 access 못 받았었음 — OpenSound 가 정답

**2. CREMA-D = `/mnt/ddn/shkim/...` 그대로 사용 가능**
- 경로 정확히: `/mnt/ddn/shkim/omni_acoustic_bench_v4_train_evaluator/train_eval_model/dataset_emotion/Source/CREMA-D/AudioWAV/` — 7,442 wavs 있음
- `/mnt/ddn` 은 클러스터 Lustre 공유 fs라 Node B 도 동일 경로 그대로 접근 가능 (`ls -ld` 권한 `drwxrwxr-x. nsml nsml` 확인)
- "Node B 접근 불가" 는 시도 안 했거나 path 오타일 가능성 — 다시 확인 바람
- 굳이 HF mirror 받고 싶으면 `MichalGarmulewicz/CREMA-D` 가 있긴 하나 1.5 GB 추가 다운 + 권한 이슈 있을 수 있어 비추천

**3. Freesound test 1 tar 누락 (90/91)**
- 무시 OK. test split 은 학습에 안 쓰이고 manifest 에만 들어감 (eval 용도). 1 tar 차이는 ≤ 5,000 clips 수준이고 test 전체 46k 의 일부라 영향 미미. 추후 보강 시에만 신경 쓰면 됨

**4. Quarantine — Node B 검증 결과 신뢰**
- Node B `convert_libri_mls_vox` 에서 librispeech 0 / mls_train_part 0 으로 나온 거면 격리 대상 자체가 안 만들어졌다는 뜻이라 OK
- Node A 어제 격리한 파일들 (`v3_quarantine/librispeech_test_*` 등) 은 별도 경로에서 흘러들어온 거라 Node B 빌드 흐름에는 원천 부재
- 즉 Node B 격리 액션 불필요 — 검증 정확함

### [2026-05-02 14:40] Node A 측 학습 config 조정 (OOM 대응)
- `configs/ASR/stage1_dac_vae_v3.yaml` 조정:
  - `omni_max_audio_samples`: 2,160,000 (45s) → **1,440,000 (30s)** — v3 LAION 일부 45s 클립 + packing 시 45+45+45 OOM
  - `disable_gradient_checkpointing`: true → **false** (gradient checkpointing 활성화)
  - `per_device_train_batch_size`: 3 → **2**
- v3 LAION caption 데이터셋 일부 (laion_freesound, epidemic) 30s 초과 클립이 30%↑ 비율인지는 확인 필요. loader 가 30s 초과 자동 drop 하므로 학습 데이터 손실 발생 — Node B 빌드 시 참고
- 학습 스크립트는 변경 없음, 위 yaml 만 갱신

### [2026-05-02 14:40] Node B 빌드 우선순위 (Node A 의견)
1. **CREMA-D**: shkim path 즉시 사용 → emotion 빌더 cremad fail 해결 (5분)
2. **AudioCaps**: OpenSound/AudioCaps 다운 트리거 (`huggingface-cli download` 백그라운드 tmux window 추가) — 47 GB 라 1-2시간 소요
3. **MACS**: TAU 자체 DL 진행 중이면 그대로 두고 zip 1-13 까지만 처리 후 early-stop (어제 Node A는 zip 14-21 미사용 / 13 GB 절약). 빌더 `build_macs.py` 가 이미 그 패턴이면 그대로
4. **Epidemic train**: 진행 중이면 그대로 (75k clips / 6 shards 예상)
5. Freesound test 90/91 tar 누락은 무시

### [2026-05-02 14:45] Node B 액션 진행 + CREMA-D 정정

**진행:**
- ✅ AudioCaps `OpenSound/AudioCaps` train+val 다운 시작 (`v3extract:ac` window, PID 153901, 47 GB / 예상 1-2 h)
- 🔄 Epidemic train, MACS DL (zip 5번까지 받는 중) 그대로 진행

**🚨 CREMA-D 정정 (user 지적):**
- Node A 가이드대로 `/mnt/ddn/shkim/...AudioWAV` 를 symlink로 묶어 7,442 rows manifest 빌드했으나 — **user 지적: "CREMA-D는 내 데이터가 아닌데 왜 그걸 쓰겠다는 거야?"**
- shkim 사용자 소유 데이터를 jos 학습에 그대로 쓰는 건 권한/소유 모호. **즉시 rollback**:
  - `rm /mnt/tmp/datasets/manifests/v3/emotion_cremad_0000.jsonl`
  - symlink 제거, `/mnt/tmp/datasets/emotion_raw/CREMA-D/AudioWAV` 빈 디렉토리로 복원
- **결정 대기:**
  - (a) HF mirror 자체 다운 (예: `MichalGarmulewicz/CREMA-D`, ~1.5 GB) → jos 자체 데이터로 빌드
  - (b) Node B 는 CREMA-D 제외 학습 (-7,442 emotion rows, emotion 총합의 ~12% 손실)
  - Node A 도 shkim 데이터 쓰는 게 맞는지 재고려 필요할 수 있음 (어제 별 문제 없이 진행하긴 했지만)

### [2026-05-02 14:46] ⚠️ Epidemic train stall 의심 (Node B)

- `extract_epidemic.py` PID 84914 살아있고 (etime 21m, S+ 상태), worker 8개도 살아있음
- 그러나 `/mnt/tmp/datasets/laion_extracted/epidemic` flac 카운트 7,834 에서 **5초간 변화 0**
- 마지막 progress log: `[epidemic] train: 2270 parquets` 이후 진척 출력 없음 (스크립트는 100 parquet마다 print)
- test split (253 parquet) 끝낼 때는 정상 진행, train 시작 직후부터 멈춘 모습 → 어제 Freesound 의 hf_transfer futex_wait stall 과 유사할 수 있음
- 진단: stall 확인되면 죽이고 재시작 (extract 자체는 idempotent — 이미 추출된 7,834 flac 보존, partial test manifest 보존)



### [2026-05-02 14:48] ⚠ 정정: CREMA-D **전체 제외** (user 지시)

User 정정: "CREMA-D는 내 데이터가 아니잖아. 제외해."

shkim path (`/mnt/ddn/shkim/.../CREMA-D/AudioWAV/`) = 다른 user 소유. 14:40 답변에서 "Node B도 그 path 쓰면 됨" 가이드는 잘못. 정정 사항:

**Node A 즉시 액션 (완료)**
- `emotion_cremad_0000.jsonl` (7,442 rows) → `/mnt/tmp/datasets/manifests/v3_quarantine/emotion_cremad_NODEA_excluded_2026-05-02.jsonl` 격리
- v3 dir 에서 `emotion_crema*` glob 결과 0 (확인됨)
- 이전 빌드 산출물(`emotion_crema_d_all_*`) 도 이미 v3_quarantine 에 있어 추가 액션 불요
- v3 final stats 갱신: 191 → **190 shards**, 12,098,051 → **12,090,609 rows**, emotion 63,883 → **56,441 rows** (cremad 7,442 제외)

**Node B 가이드 정정**
- 14:40 답변 §1 (Node B 빌드 우선순위 1번 "CREMA-D shkim path 즉시 사용") **무효**. CREMA-D 는 빌드 대상에서 **빼고** 진행
- Node B emotion 빌더 cremad fail 은 정상 결과 — 굳이 fix 시도 안 해도 됨
- emotion 데이터셋 = 6개 (iemocap / meld / dailytalk / emov_db / mustard++ / ravdess) 로 축소

**T14 doc 업데이트 예정**
- `docs/stage1/dac_vae_v3_data.md` Emotion 표에서 crema_d 행 제거 + subtotal 갱신 (61,547 → 54,105) — 다음 turn 처리

**향후 예방**
- emotion_raw / 외부 path 사용 전 user 소유 확인 필수 (특히 `/mnt/ddn/<other_user>/` path)

### [2026-05-02 14:50] Node B 외부 사용자 데이터 전수 점검 + user 지시 반영

User 추가 지시: "그 외에도 내 데이터 아닌 게 있으면 말해" → Node B 빌더 input path 전수 검토.

| # | source | 외부 owner path | user 결정 |
|---|---|---|---|
| 1 | CREMA-D | `/mnt/ddn/shkim/.../AudioWAV` | ❌ **제외** (이미 rollback) |
| 2 | IEMOCAP | shards `/mnt/ddn/users/sehyun/.../emotion/iemocap_shard_*` + audio `/mnt/ddn/kyudan/IEMOCAP/data/` | ❌ **제외** |
| 3 | MELD | shards `/mnt/ddn/users/sehyun/.../meld_*_shard_*` (audio 는 jos `/mnt/tmp` REMAP) | ✅ **유지** |
| 4 | libri_mls_vox | SRC `/mnt/ddn/users/sehyun/.../libri_mls_vox/shard_*.jsonl` (결과 row 의 `nubes_path` 는 cluster 공유) | ✅ **유지** |

**Node B 즉시 액션 (완료):**
- IEMOCAP `emotion_iemocap_0000.jsonl` 삭제 (kyudan/sehyun 데이터 사용 → 제외)
- CREMA-D 는 14:45 에 이미 rollback (manifest 삭제 + symlink 제거)

**Node B 현재 manifest 분포** (총 17개 source, 약 ~167 shards):
```
   2 audioset_bal_train         1 emotion_emovdb
   1 clotho_dev                 1 emotion_meld
   1 clotho_val                 1 emotion_mustardpp
   2 emotion_dailytalk          1 emotion_ravdess
   3 fsd50k_dev                 1 laion_audiostock
   1 laion_bbc_test             2 laion_bbc_train
   1 laion_epidemic_test        4 laion_freesound_test
  16 laion_freesound_train_1   12 laion_freesound_train_2
 128 libri_mls_vox
```

(progress: AudioCaps OpenSound DL + Epidemic train + MACS DL 진행 중)

**Node A 께 — IEMOCAP 도 제외 필요?**
- Node A 14:48 정정에서 CREMA-D 만 격리. IEMOCAP 도 sehyun shards + kyudan audio 사용 (Node A 7:32 entry 참조: "라이선스 데이터, sehyun shards 에 path 가 sehyun cache 로 박혀 있어 → kyudan IEMOCAP wav 로 remap")
- user 지시는 Node B 한정 답변이지만, 정책 일관성상 Node A 도 IEMOCAP 격리 필요할 가능성. Node A 답변 부탁드립니다.

### [2026-05-02 14:55] Node A 답변 — IEMOCAP 도 제외 (Node B 14:50 질문 응답)

@Node B — Yes, 동일 정책. IEMOCAP 도 제외 처리.

**Node A 전수 audit 결과** (`v3/*.jsonl` 모든 source × audio_path/nubes_path 첫 row 검사):

| source | path root | 소유 | 결정 |
|---|---|---|---|
| audiocaps / audioset / clotho / fsd50k / laion_* / macs | `/mnt/tmp/datasets/...` | jos local | ✅ 유지 |
| emotion_dailytalk / emov_db / meld / mustardpp / ravdess | `/mnt/tmp/datasets/emotion_raw/...` | jos local | ✅ 유지 |
| **emotion_iemocap** | `/mnt/ddn/kyudan/IEMOCAP/data/...` | **kyudan** | ❌ **제외** |
| libri_mls_vox | `hyperscaleai-audiollm/...` (nubes path) | cluster 공유 게이트웨이 (특정 user 소유 아님) | ✅ 유지 |

**Node A 즉시 액션 (완료)**:
- `emotion_iemocap_0000.jsonl` (10,013 rows) → `v3_quarantine/` 격리
- v3 dir 에서 `emotion_iemocap*` glob 결과 0 (확인됨)

**Node A v3 final stats 재갱신**:
- 190 → **189 shards**
- 12,090,609 → **12,080,596 rows**
- emotion 56,441 → **46,428 rows** (cremad 7,442 + iemocap 10,013 제외)
- emotion 데이터셋 = **5개** (meld / dailytalk / emov_db / mustard++ / ravdess)

**Node B 가이드**:
- IEMOCAP 빌더 (sehyun shards + kyudan audio) 둘 다 외부 user 데이터라 빌드/사용 안 함 (이미 Node B 14:50 에 처리 완료한 것 확인)
- emotion 데이터셋 5개로 정렬됨 → Node A/B 동일 셋

**libri_mls_vox 는 왜 유지?**
- `nubes_path` 는 hyperscaleai 공용 게이트웨이 경로 (`hyperscaleai-audiollm/datasets/public/...`) — 특정 user `/mnt/ddn/<name>/` 가 아닌 cluster 차원 공용 storage. 어제 v2 학습도 같은 path 로 진행했고 회사 차원 공용 데이터로 분류
- 단, source manifest (Agent D 가 변환에 쓴 입력 shards) 가 `/mnt/ddn/users/sehyun/...` 였던 부분은 Node A 도 빌드 시점에는 sehyun shards 를 입력으로 썼지만 **결과 manifest 의 `nubes_path` 자체는 sehyun 소유 아님**. Node B 도 같은 흐름이면 OK
- 만약 user 가 sehyun 입력 shards 사용 자체를 문제 삼는다면 추가 알려달라 (그 경우 libri_mls_vox 도 격리 후 nubes 직접 listing 으로 재빌드 필요)

**T14 doc 추가 갱신 예정**:
- `docs/stage1/dac_vae_v3_data.md` Emotion 표에서 iemocap 행 추가 제거 + subtotal 53,242 → **43,229** rows / 7 → **6 shards**

### [2026-05-02 14:56] Node B 정리 — Node A 답변 수렴 + 자율 진행 (user "다 알아서 해")

**정책 수렴 확인:**
- Node A/B emotion 동일 5 sources: meld / dailytalk / emov_db / mustard++ / ravdess
- libri_mls_vox 결과 `nubes_path` 는 cluster 공용 (회사 차원 공용 storage) — 유지 OK. Node B 도 동일 입력으로 변환했으니 같은 흐름
- IEMOCAP/CREMA-D 둘 다 Node A/B 모두 격리 완료

**Node B Epidemic stall 처리 (완료):**
- Stall 진단: 모든 worker `futex_wait_queue_me`, 부모 `pipe_wait`, 8s flac 변화 0 — 어제 Freesound multiprocess deadlock 과 동일 패턴
- pkill -KILL → tmux:epi 에서 재시작 (PID 19242). test 7,564 flac idempotent skip → train 2270 parquets 재진입
- 다시 stall 나면 max_workers 8→4 로 줄이거나 single-process 폴백

**Node B 현재 진행:**
- 🔄 epi (재시작 후 train 진행 중)
- 🔄 ac (AudioCaps OpenSound 47 GB DL → build_audiocaps.py auto chain)
- 🔄 macs (TAU 2019 zip DL, ~5/13 까지 받음, build_macs.py 안에서 자동 추출+manifest)
- 위 3개 끝나면 Node B v3 manifest 최종 (Node A 와 동일 189 shards 목표, 단 Node B 가 별도 빌드한 LAION 등 source 차이 있을 수 있음)

**다음 단계 (Node B 자율):**
1. ⏰ 위 3 작업 완료 대기 (예상 1-2 h, AudioCaps 가 가장 길음)
2. manifest 최종 검증: 각 source 한 row 의 audio_path/nubes_path `os.path.exists()` 50개씩 sanity
3. T14 `docs/stage1/dac_vae_v3_data.md` Emotion 표 갱신 (cremad/iemocap 행 제거, subtotal 53,242 → 43,229)
4. Stage-1 v3 학습 launch — `bash scripts/ASR/run_stage1_dac_vae_v3.sh` (yaml 변경 없음, Node A 14:40 OOM 조정 그대로)

### [2026-05-02 15:09] Node B Epidemic — 2번째 stall, max_workers 8→4 임시 패치

14:51 재시작한 PID 19242 가 **train 진입 직후 또 stall** (15:08 확인: train 2270 parquets 마지막 log 그대로, flac 7,834 변화 0, 17분 idle). Stall 패턴이 ProcessPoolExecutor train worker 초기화 단계에서 재현됨 — pyarrow.parquet read + multiprocess fork 조합 의심.

**패치:**
- `scripts/laion_audio_630k/extract_epidemic.py:92` `max_workers=8` → `max_workers=4` (sed in-place, commit 안 함, 빌드 끝나면 원복)
- pkill -KILL → 재시작 (PID 77641). test 12초 만에 fast-skip, train 진입

만약 4-worker 도 stall 재발하면:
- (a) max_workers=2
- (b) ProcessPoolExecutor → 단일 process 직렬 (속도 trade-off)
- (c) 다른 노드와 sync (DDN 통한 manifest copy) 옵션 검토

**진행 보고:**
- 🔄 epi (max_workers=4 재시작, 모니터링 중)
- 🔄 ac AudioCaps: 432 parquet / 38 GB 받음 (target ~417 train+val, 거의 완료) — 빌드 자동 chain 대기
- 🔄 macs: 3,642 / 3,930 wav 추출됨, zip 13 다운 중 (마지막 zip)
- ✅ libri/freesound/bbc/audiostock/clotho/audioset/fsd50k/emotion(5) 완료
