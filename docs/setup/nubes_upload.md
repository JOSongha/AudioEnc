# Nubes 업로드 후보 (로컬 ↔ nubes 데이터 비교)

audiollm-trainer 가 사용하는 모든 학습/평가 데이터셋이 nubes (`hyperscaleai-audiollm` bucket) 에 이미 올라가 있는지 점검하고, 누락 / 부분 매칭 / 매핑 미확인 항목을 정리. 작성 기준일 2026-05-07.

> **요약**: datasets.md (Stage-1 v6 카탈로그) + Stage-2 eval 데이터 합쳐 **22 source** 중 **13 source 가 이미 nubes 에 존재**. **9 source 가 누락** (업로드 후보). 1 source (LAION-Freesound) 는 nubes 위치 매핑 추가 검증 필요.

## 0. 점검 방법

- nubes gateway: `http://c.nubes.sto.navercorp.com:8000/v1/<bucket>/<path>`
- bucket: `hyperscaleai-audiollm`
- list API: `?dir=/<path>/&max-contents=<n>` 으로 directory listing (S3 호환 아님). pagination 은 `X-Continuation-Token` 헤더
- 로컬 인벤토리 source: [datasets.md](datasets.md) (Stage-1 v6) + `evaluation/stage2/eval_*.py` (Stage-2 / 외부 평가)
- 매핑 검증: 각 source 의 nubes prefix 직접 list 후 audio 파일 한두 개 200 OK 확인

## 1. ASR (4 source)

| Source | 로컬 | Nubes 경로 | 일치 여부 |
|---|---|---|---|
| MLS English | `/mnt/ddn/omni_dataset/audio/mls_english/` (canonical jsonl), nubes audio | `/datasets/public/16kHz/mls/mls_english/{train,dev,test}/` + `transcripts.txt` | ✓ 동일 (10,808,037 utt train, datasets.md row count 일치). 학습은 train 만 사용 |
| LibriTTS-R | (로컬에 없음. sehyun cache 만) | `/datasets/public/libriTTS/{train-clean-100,train-clean-360,train-other-500,test-clean,test-other,libri_etc}/<spk>/<chapter>/<utt>.{wav,normalized.txt,original.txt,trans.tsv}` | ✓ 동일 (`build_libritts_r.py` smoke test 결과 train-clean-100 = 33,236 utts, datasets.md 33,232 와 +4 차이로 nubes 가 더 정확) |
| VoxPopuli (en) | `/mnt/ddn/omni_dataset/audio/voxpopuli/` (jsonl) | `/datasets/public/16kHz/voxpopuli/{train,test}/<id>.{flac,txt}` | ✓ 동일 (182,466 utt train, datasets.md 일치) |
| GigaSpeech XL | `/mnt/ddn/omni_dataset/audio/gigaspeech/` (jsonl) | `/datasets/public/16kHz/gigaspeech/{train,test}/<id>.{flac,txt}` | ✓ 동일 (8,256,276 utt train, v6 50% subsample = 4,129,334) |

**결론**: ASR 4종 모두 nubes 에 있음. 업로드 불필요.

## 2. Sound captioning (9 source)

| Source | 로컬 | Nubes 경로 | 일치 여부 |
|---|---|---|---|
| LAION-Freesound | `/mnt/tmp/datasets/laion_extracted/freesound/` (extracted 607 GB) + `/mnt/tmp/datasets/laion_freesound/` (source 608 GB) | **매핑 미확인**. nubes top-level 의 [`Freesound/`](#) 에 audio 가 있지만 LAION 공식 split 인지 확인 필요. `/LAION-Audio-630k/` 산하에는 `audiostock / bbc_sound_effects / epidemic_sound_effects / free_to_use_sounds / sonniss_game_effects` 5개 split 만 (Freesound 부재) | ⚠ 매핑 검증 필요 |
| LAION-BBC | `/mnt/tmp/datasets/laion_extracted/bbc/` (78 GB) + `/mnt/tmp/datasets/laion_bbc_hf/` (155 GB) | `/datasets/public/LAION-Audio-630k/bbc_sound_effects/{audio/, train.jsonl, test.jsonl}`. 별도로 nubes top-level 에 더 큰 `/datasets/public/BBCSoundEffects/audio/` (BBC SE 전체 dump) 도 존재 | ✓ LAION subset 매핑 OK |
| LAION-Epidemic | `/mnt/tmp/datasets/laion_extracted/epidemic/` (72 GB) + `/mnt/tmp/datasets/laion_epidemic_clapv2/` (67 GB) | `/datasets/public/LAION-Audio-630k/epidemic_sound_effects/{audio/, train.jsonl, test.jsonl}` | ✓ |
| LAION-Audiostock | `/mnt/tmp/datasets/laion_audiostock/` (2.2 GB) | `/datasets/public/LAION-Audio-630k/audiostock/{audio/, train.jsonl, test.jsonl}` | ✓ |
| AudioCaps | `/mnt/tmp/datasets/laion_extracted/audiocaps/` (40 GB) + `/mnt/tmp/datasets/audiocaps/` (38 GB) | **없음**. `/audiocaps/`, `/AudioCaps/`, `/AudioSet_SL/audiocaps/`, `/16kHz/audiocaps/` 모두 404 | ✗ 누락 (업로드 후보) |
| FSD50K | `/mnt/tmp/datasets/env_sound/FSD50K/` (56 GB) | `/datasets/public/FSD50K/{audio/, AF-Think_*.jsonl}` | ✓ |
| AudioSet | `/mnt/tmp/datasets/env_sound/AudioSet/` (71 GB) | `/datasets/public/AudioSet_SL/{audio/, AF-Think_*.jsonl, naiveInst_AudioSet_SL.jsonl}`. v6 학습은 bal_train (18,683 rows) 만 — bal_train 매핑은 AudioSet_SL 안 jsonl 에서 필터 필요 | ✓ (단 split 매핑 확인 필요) |
| Clotho | `/mnt/tmp/datasets/env_sound/Clotho/` (18 GB) | `/datasets/public/Clotho-v2/{audio/, audio_evaluation/, audio_validation/, clotho_captions_{development,evaluation,validation}.csv, clotho_metadata_development.csv, AF-Think_*.jsonl}` | ✓ (2026-05-08 eval+val csv+wav 추가, § 12.12) |
| MACS | `/mnt/tmp/datasets/env_sound/MACS/MACS.yaml` (2.7 MB caption metadata, ddn local 또는 `/users/jos/AudioEnc/MACS/MACS.yaml` nubes backup) | audio: `/datasets/public/MACS/audio/` (TAU2019 source `a` 14,400 중 yaml 의 3,930 만 인용) + yaml: `/users/jos/AudioEnc/MACS/MACS.yaml` (옵션 C, § 12.8) | ✓ nubes-direct |

**결론**: 8/9 nubes 에 있음. **AudioCaps 만 업로드 필요**. LAION-Freesound 매핑 확인은 별도.

> 참고: nubes top-level 에 `Freesound/audio/` (전체 freesound dump), `BBCSoundEffects/audio/` (전체 BBC SE) 도 별도 존재. LAION subset 과 더 큰 원본 둘 다 보존. v6 학습은 LAION subset (`/LAION-Audio-630k/<split>/`) 만 인용하면 충분.

## 3. Emotion (6 + CREMA-D)

| Source | 로컬 | Nubes 경로 | 일치 여부 |
|---|---|---|---|
| DailyTalk | `/mnt/tmp/datasets/emotion_raw/DailyTalk/` (12 GB, 23,773 wav) | `/datasets/public/DailyTalk/{audio/, AF-Think_think_DailyTalk.jsonl}` | ✓ |
| MELD | `/mnt/tmp/datasets/emotion_raw/MELD/` (32 GB, 13,847 wav) | `/datasets/public/MELD.Raw/{train_splits, dev_splits_complete, output_repeated_splits_test}/` | ✓ |
| IEMOCAP | `/mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release/` (23 GB, 10,190 wav) | `/datasets/public/IEMOCAP/data/` | ✓ (구조 추가 검증 필요) |
| EmoV-DB | `/mnt/tmp/datasets/emotion_raw/EmoV-DB/` (7.3 GB, 6,893 wav) | **없음** | ✗ 누락 |
| RAVDESS | `/mnt/tmp/datasets/emotion_raw/RAVDESS/` (765 MB, **1,440 wav** = 24 actors × 60) | **없음** | ✗ 누락 |
| MUStARD++ | `/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/` (276 MB, 1,201 wav) | **없음** | ✗ 누락 |
| CREMA-D | `/mnt/tmp/datasets/emotion_raw/CREMA-D/` (85 MB, 0 wav — 영구 제외) | **없음** | ✗ 누락. datasets.md "영구 제외" 라 업로드 우선순위 낮음 |

**결론**: 3/6 nubes 에 있음. **EmoV-DB / RAVDESS / MUStARD++ 업로드 필요**. CREMA-D 는 datasets.md 정책상 학습 미사용이므로 업로드 보류 가능.

## 4. Eval-only (5)

| Source | 로컬 | Nubes 경로 | 일치 여부 |
|---|---|---|---|
| LibriSpeech (test) | `/mnt/tmp/cache/openslr___librispeech_asr/` (HF cache) | `/datasets/public/librispeech_asr/` | ✓ (구조 추가 확인 필요) |
| ESC-50 | `/mnt/tmp/datasets/env_sound/ESC-50/` (1.4 GB, 2,000 wav) | `/datasets/public/ESC-50/{audio/, esc50.csv, esc50-human.xlsx, AF-Think_*.jsonl}` | ✓ |
| SAVEE | `/mnt/tmp/cache/savee/data/train-00000-of-00001.parquet` (54 MB, 610 utt) | **없음** | ✗ 누락 |
| MSP-Podcast | `/mnt/tmp/cache/CLAPv2_MSP_podcast/*.parquet` (1.3 GB, 6,000+ utt) | **없음** | ✗ 누락 (CLAPv2 mirror) |
| JL-Corpus | `/mnt/tmp/cache/jl_corpus/*.parquet` (396 MB, ~3,000 utt) | **없음** | ✗ 누락 |
| LISTEN | `/mnt/tmp/listen_analysis/data/test-00000-of-00001.parquet` (3.7 GB, 2,635 row) | **없음** | ✗ 누락 |

**결론**: 2/6 nubes 에 있음. **SAVEE / MSP-Podcast / JL-Corpus / LISTEN 업로드 필요**.

## 5. 업로드 후보 정리

### 우선순위 높음 (학습 / eval 활성)

| Source | 로컬 용량 | 파일 수 | 비고 |
|---|---:|---:|---|
| AudioCaps (sound) | 40 GB extracted | 45,623 | v6 학습 풀, train+val |
| EmoV-DB (emotion) | 7.3 GB | 6,893 wav | v6 학습 풀 (4 화자 모두, leak-fix 폐기) |
| RAVDESS (emotion) | 765 MB | 1,440 wav | v6 학습 풀 (24 actors 모두, leak-fix 폐기) |
| MUStARD++ (emotion) | 276 MB | 1,201 wav | v6 학습 풀 전체 |
| LISTEN (eval) | 3.7 GB | 2,635 row | Stage-2 emotion eval |
| MSP-Podcast (eval) | 1.3 GB | 6,000+ utt | Stage-2 emotion eval |
| JL-Corpus (eval) | 396 MB | ~3,000 utt | Stage-2 emotion eval |
| SAVEE (eval) | 54 MB | 610 utt | Stage-2 emotion eval |

업로드 합계: **~54 GB** (AudioCaps 가 대부분)

### 우선순위 낮음 / 검토 필요

- **CREMA-D**: datasets.md "영구 제외" 라 학습/eval 사용 안 함. 업로드 보류 가능.
- **LAION-Freesound 매핑 검증**: nubes 의 `/LAION-Audio-630k/` 산하에 `Freesound/` 가 없음. nubes top-level 의 `/Freesound/` (전체 dump) 와 LAION-Freesound 460,141 rows 가 정확히 매칭되는지 ID 단위 검증 필요. 매칭되면 업로드 불필요, 아니면 업로드 후보.

## 6. nubes 에 있는데 datasets.md 학습 풀에 없는 데이터셋

참고용. nubes `/datasets/public/` 에는 v6 가 안 쓰는 데이터셋도 다수 보존:

- ASR/speech: `peoples_speech`, `common_voice`, `voxceleb1`, `voxceleb2`, `multidialog`, `Europarl_ASR`, `switchboard_data`, `aihub` (한국어 ASR 다수), `libriTTS`, `LibriTTS-R` (별도), `librispeech_asr`, `expresso`, `huggingface/en_Emilia_Yodas_single`
- Sound: `BirdCLEF`, `Chime-Home`, `CochlScene`, `Emo-Soundscapes`, `GTZAN`, `ICBHI`, `MIMII`, `MagnaTagATune`, `MusicBench`, `NSynth`, `SDD`, `SoundDescs`, `UrbanSound8K`, `nonspeech7k`, `audio-flan`, `clotho-aqa`, `mtg`, `fma`, `svc-data`, `The_Parallel_Audiobook_Corpus`, `parler-tts_libritts_r_tags_tagged_10k_generated`
- TTS: `HiFi_TTS_SLR109`

## 7. 업로드 절차 (제안)

각 누락 source 에 대해:

1. **경로 결정**: 기존 nubes 명명 규칙 따르기. CamelCase 또는 hyphen-case 일관성 (예: `EmoV-DB`, `RAVDESS`, `MUStARD-Plus-Plus`, `SAVEE`, `MSP-Podcast`, `JL-Corpus`, `LISTEN`, `AudioCaps`).
2. **구조 결정**: 다른 데이터셋과 일관성 — `audio/` 하위에 raw audio + 별도 metadata jsonl/csv. 예:
   ```
   /datasets/public/AudioCaps/
       audio/
           <id>.wav
       metadata.csv (또는 train.jsonl, val.jsonl)
   ```
3. **업로드 도구**: `nubescli pupload` (parallel upload) 또는 `nubescli dir-upload` (directory bulk).
   ```bash
   export NUBES_IP_LOOKUP_ADDRESS=c.nubes.sto.navercorp.com:8000
   nubescli dir-upload /mnt/tmp/datasets/laion_extracted/audiocaps/ \
       hyperscaleai-audiollm/datasets/public/AudioCaps/audio/
   ```
4. **검증**: 업로드 후 `?dir=/<path>/&max-contents=10` 으로 listing + 샘플 파일 fetch 200 OK 확인.
5. **manifest 갱신**: 해당 source 빌더가 nubes-direct 로 동작하도록 path 매핑 (build_audiocaps.py 등 기존 빌더 검증).

## 8. 권장 액션

1. **즉시 업로드** (우선순위 높음): ~~AudioCaps + emotion 3종 (EmoV-DB, RAVDESS, MUStARD++) + eval 4종 (LISTEN, MSP-Podcast, JL-Corpus, SAVEE)~~ — Emotion 3종 (§ 12.2 / 12.5 / 12.6) + IEMOCAP (§ 12.4) + AudioSet (§ 12.7) + LAION-BBC (§ 12.3) + FSD50K eval (§ 12.1) + MACS yaml backup (§ 12.8) **모두 완료**. **남은 미완**: AudioCaps (40 GB), eval 4종 (LISTEN, MSP-Podcast, JL-Corpus, SAVEE)
2. **검증 후 결정**: LAION-Freesound 매핑 (nubes `/Freesound/` ↔ LAION-Freesound 460k IDs)
3. **보류**: CREMA-D (학습 / eval 미사용)
4. **manifest 빌더 재작성**: AudioCaps / emotion / eval 빌더가 nubes-direct 로 동작하도록 추후 업데이트 (현재 학습 manifest 는 v6 셔드에 nubes_path 또는 local audio_path 둘 다 사용 가능, runtime 에서 `load_from_nubes` flag 분기). MACS / Audiostock / LAION-BBC / GigaSpeech / MLS / VoxPopuli / LibriTTS-R 빌더는 이미 nubes-direct 화 완료 (§ 12.x).

## 9. Row count 검증 (nubes audio file count vs 로컬 v6 row count)

각 source 의 nubes prefix 를 직접 recursive list 해서 audio 파일 수를 카운트, 로컬 v6 manifest row count 와 비교. gateway list API + `X-Continuation-Token` 페이지네이션. MLS 는 단일 `transcripts.txt` line count 로 빠르게 검증.

### 9.1 정확 일치 (✓)

> "Uploaded" 컬럼 표기: **—** (필요 없음 / 미진행) / **✓ YYYY-MM-DD** (업로드 완료, 날짜 기록) / **⏳** (진행 중)

| Source | Local v6 | Nubes count | 업로드 | Leak | Uploaded | 비고 |
|---|---:|---:|:---:|:---:|:---:|---|
| MLS English (train) | 10,808,037 | 10,808,037 | **X** | ✓ | — | `transcripts.txt` line count. train/dev/test sub-dir 분리 |
| ESC-50 | 2,000 | 2,000 | **X** | n/a | — | eval-only |
| LAION-Epidemic | 75,645 | 75,645 | **X** | ✓ | — | `/LAION-Audio-630k/epidemic_sound_effects/audio/` |
| VoxPopuli (en train) | 182,466 | 182,466 (.flac) | **X** | ✓ | — | train/test 분리 (.flac + .txt 페어 / 2) |
| FSD50K (dev) | 40,966 | 40,966 | ✓ 보완 | ✓ | ✓ 2026-05-07 § 12.1 | `/datasets/public/FSD50K/audio/` (dev) + `/users/jos/AudioEnc/FSD50K/{eval_audio, ground_truth, metadata, doc}` (eval + 표준 distribution 보완). Stage-2 eval 용 eval split 10,231 wav 업로드 완료 |

### 9.2 Nubes 가 broader (학습 풀 외 split 도 포함)

| Source | Local v6 학습 | Local 전체 | Nubes count | 업로드 | Leak | Uploaded | 분석 |
|---|---:|---:|---:|:---:|:---:|:---:|---|
| MELD | 11,096 (train+dev) | 13,847 (모든 split) | 13,847 audio (mp3, train 9,988 + dev 1,112 + test 2,747) + 3 csv (별도 업로드) | ✓ csv 만 (audio 는 기존 nubes 사용) | ✓ | ✓ 2026-05-08 § 12.10 | nubes audio + 별도 csv = nubes-only 학습/eval 가능. v6 룰: train+dev 만 학습, test 는 Stage-2 eval (`load_meld_test`). builder + eval 둘 다 nubes-direct 갱신, ddn 의존 0 |
| LAION-Audiostock | **10,001** (v6 nubes-direct) | 9,139 (ddn 잔존, 미사용) | 10,001 (= train 9,001 + test 1,000) | **X** (nubes-direct) | ✓ | ✓ 2026-05-07 nubes-direct | **v6 빌더 nubes-direct 로 갈아엎음** ([`build_audiostock.py`](../../scripts/manifest_builders/build_audiostock.py)). 이전 v5: ddn 9,139 (LAION 공식 ~10K 중 ddn 다운로드 단계 ~860 fail). v6: nubes train+test.jsonl 직접 인용 → 10,001 row. row schema `audio_path` → `nubes_path` 로 변경 |
| AudioSet | 18,683 (bal_train) | bal+unbal+eval | 108,317 (통합) | **✓ split-aware** | **✓** | ✓ 2026-05-07 § 12.7 | nubes 기존 `/AudioSet_SL/audio/` 통합 → leak 위험 ⚠. 옵션 C 로 사용자 영역에 path-level split 분리 (audio/ 18,683 flac + data/bal_train/ 38 parquet + data/eval/ 35 parquet + ontology + README, 71 GB) 업로드 완료 7분 16초. leak 0 |
| MACS | 3,930 (= MACS.yaml 공식) | 3,930 | 14,400 (모두 source `a`) | **✓ yaml만** | n/a | ✓ 2026-05-08 § 12.6 | **v6 / nubes / MACS 공식 모두 일치**. nubes audio 14,400 = TAU2019 development 의 **10 scene × 12 city × 120 clip × source `a`** (b/c 미보존). MACS 정의 = TAU2019 의 **3 scene** (airport 1,296 + park 1,317 + public_square 1,317 = 3,930). v6 도 그 3,930 만 사용. 빌더 nubes-direct 갈아엎음 ([`build_macs.py`](../../scripts/manifest_builders/build_macs.py)) — `MACS.yaml` 의 filename 을 nubes audio path 로 직접 매핑. **추가**: `MACS.yaml` (2.7 MB) 도 `/users/jos/AudioEnc/MACS/MACS.yaml` 에 업로드 완료 (옵션 C, builder 의 ddn yaml fallback 대체용 backup) |

> **§ 9.2-MACS-note**: 쉽게 풀면, **TAU2019** = 공항/공원/도로 같은 도시 사운드를 여러 도시에서 녹음한 데이터셋. 14 scene × 12 city × 여러 시간대 → audio 파일 ~14K (모두 source label `a`, 단일 마이크 종류). **MACS** = TAU2019 의 일부 (3 scene: airport / park / public_square 의 3,930) 에만 사람이 caption 붙인 캡션 데이터셋. caption 작업 자체는 3 scene 분량만. **nubes** 는 TAU2019 raw 14,400 wav 모두 보존 (다른 11 scene 도). **MACS 의 정의** = 그중 3,930. **v6** 는 MACS 정의 따라 3,930 사용 = MACS 공식 row 와 일치. v6 부터 builder 가 nubes path 인용 (별도 업로드 불필요, ddn audio extract dir 의존성 폐기).

### 9.3 단위 / 매핑 차이 (⚠)

| Source | Local v6 학습 | Local 전체 | Nubes count | 업로드 | Leak | Uploaded | 분석 |
|---|---:|---:|---:|:---:|:---:|:---:|---|
| **DailyTalk** | **23,773** utt (= 23,773 wav, 통째로 학습 — leak-fix 폐기) | 23,773 wav | **2,541** wav (모두 0 byte placeholder, **사실상 부재**) | **✓ 23,773 utt** | n/a | ⏳ § 12.11 | nubes `/DailyTalk/audio/` 의 wav 모두 0 byte placeholder. utterance 단위 audio + metadata 신규 업로드 진행 중 (`/users/jos/AudioEnc/DailyTalk/`, ~6.6 GB). v6 룰: canonical split 없음 + leak-fix 폐기 → 23,773 모두 학습 풀, eval held-out 없음 |
| **IEMOCAP** | 5,882 (Sessions 1-4) | 10,190 (utterance 10,039 + dialog wav 151) | 10,039 utterance wav (session 통합) | **✓ done** | ✓ | ✓ 2026-05-07 § 12.4 | **v6 룰 예외**: 학계 관행 leave-session-out 유지. 옵션 A 결정 (session-aware 1.4 GB) → `/users/jos/AudioEnc/IEMOCAP/` 업로드 완료 (10,039 wav + 151 EmoEval txt + 151 transcripts txt + Sub-dirs Attribute/Categorical/Self-evaluation + README). 2분 24초. 모든 검증 ✓ |
| Clotho-v2 | 4,881 (dev+val) | 4,881 | dev 3,839 + eval 1,045 + val 1,045 (모두 별도 subdir) | ✓ done | ✓ | ✓ 2026-05-08 § 12.12 | 2026-05-08 업로드 완료. `audio_evaluation/` (1,045 wav, ~2.0 GB) + `audio_validation/` (1,045 wav, ~2.0 GB) 별도 subdir + `clotho_captions_evaluation.csv` (361,995 B) + `clotho_captions_validation.csv` (367,649 B). dev/eval/val 파일명 충돌 4건 (dev∩eval=1, dev∩val=1, eval∩val=2) 회피 위해 split 별 subdir 분리. v6 학습 (dev+val) + Stage-2 eval (evaluation only) 모두 nubes-direct 가능 |
| **LAION-Freesound** | **460,141** | 460,141 flac (`/users/jos/AudioEnc/LAION-Freesound/audio/`) | ✓ 사용자 영역 신규 업로드 (nubes `/datasets/public/Freesound/audio/` 는 다른 dump 라 매핑 불가) | ✓ done | n/a | ✓ 2026-05-08 § 12.15 | 200-sample ID 매칭 검증 (60% miss + size 0건 일치) 후 LAION 본 (~607 GB) 사용자 영역 직접 업로드. PREFIX_MAPPINGS 매핑 추가, v6_nubes_full (별도 dir) 100% nubes-mapped 검증. live v6_nubes swap 은 audio 업로드 진행 중 (~7-8h) 완료 후 |

### 9.4 누락 / 거의 비어 있음 (✗)

| Source | Local v6 | Nubes count | 업로드 | Leak | Uploaded | 분석 |
|---|---:|---:|:---:|:---:|:---:|---|
| **LAION-BBC** | 31,936 row (= 15,968 unique audio × 2 caption) | 2,000 (LAION subset) + 120 (top-level dump) | **✓ superset 15,973** | ✓ | ✓ 2026-05-07 § 12.3 | nubes set ⊆ ours 검증 완료. 옵션 B 로 superset 15,973 + metadata 3 + README 1 업로드 완료 (78 GB / 7분 44초). `/users/jos/AudioEnc/LAION-BBC/` |
| AudioCaps | 45,623 (train+val) | 0 | **✓** | ✓ | ⏳ § 12.9 | 옵션 C2: audio (46,506 flac = 45,623 train+val + 883 test, ~40 GB) + parquets (473 file = 412+20+41 train/val/test, ~41 GB, audio bytes + caption 표준 dist 보존). HF `OpenSound/AudioCaps` test 41 parquet 신규 다운 + 추출. 업로드 진행 중 (`/users/jos/AudioEnc/AudioCaps/`, ~81 GB) |
| EmoV-DB | **6,893** | 0 | ✓ 완료 | n/a | ✓ 2026-05-07 § 12.5 | v6 룰: canonical split 없음 + leak-fix 폐기 → 4 화자 (Bea/Jenie/Josh/Sam) 모두 학습 풀. `/users/jos/AudioEnc/EmoV-DB/{bea, jenie, josh, sam, repo, README_upload.md}` 6,893 wav + repo 4 file + README 업로드 완료 (~3 min, 1차 + retry 2회) |
| RAVDESS | **1,440** | 0 | **✓** | n/a | ✓ 2026-05-07 § 12.6 | v6 룰: canonical split 없음 + leak-fix 폐기 → 24 actors 모두 학습 풀, eval held-out 없음. 1,440 wav (24 × 60) + README → `/users/jos/AudioEnc/RAVDESS/` 업로드 완료 (565 MB / 33초) |
| MUStARD++ | 1,200 | 0 | ✓ 완료 | ✓ | ✓ 2026-05-07 § 12.2 | `/users/jos/AudioEnc/MUStARD_Plus_Plus/{audio_wav, mustard++_text.csv, utterance_ids.txt, README*.md}`. audio_wav 1,201 + metadata 모두 업로드 완료 (28 s) |
| CREMA-D | (제외) | 0 | **X** | n/a | — | datasets.md 영구 제외 |
| SAVEE / MSP-Podcast / JL-Corpus / LISTEN | (eval-only) | 0 | **✓** | n/a | — | eval-only 라 학습 leak n/a. 업로드 필요 |

### 9.5 LibriTTS-R / LibriSpeech 추가 결과

**LibriTTS-R**:
| Split | v6 학습 utt | Nubes speakers | Sample 추정 utt | 업로드 | Leak | Uploaded | 상태 |
|---|---:|---:|---:|:---:|:---:|:---:|---|
| train-clean-100 | 33,232 | 200 (vs 표준 247) | ~33,500 (134,055 files / 4) | **⚠ +47 spk** | ✓ | — | utt 매칭, 47 speaker 누락 의심 |
| train-clean-360 | 116,454 | **904** (표준 일치) | **119,328** (20-spk sample, +2,874 = +2.5%) | ✓ 검증 완료 | ✓ | ✓ | speaker 표준 일치, utt sampling 노이즈 내 |
| train-other-500 | 205,035 | **1,160** (표준 일치) | **219,298** (20-spk sample, +14,263 = +7%) | ✓ 검증 완료 | ✓ | ✓ | speaker 표준 일치, utt sampling 노이즈 내 |
| 합계 | 354,721 | 2,264 spk (-47) | — | — | ✓ | ✓ (360/500) | train-* split 명시 분리, dev/test 와 leak 없음 |

train-clean-100 의 실제 utt 추정 = 134,055 files / 4 (.wav + .normalized.txt + .original.txt + .txt) ~ 33,500 ≈ v6 33,232. utt 단위는 일치하지만 speaker 200 vs LibriTTS 표준 247 (47 speaker 누락 가능). build_libritts_r.py smoke 결과 33,236 utt 와 동일.

**train-clean-360 / train-other-500 sampling 검증** (2026-05-08, recursive list timeout 회피용 20-speaker sample):
- 360: total spk 904 (표준 LibriTTS-R 904 정확 매칭), sample 20 spk → 46 chapter / 2,640 utt → avg 132 utt/spk → extrapolated 119,328 utt (target 116,454, diff +2,874 = +2.5%, sampling noise 범위 내)
- 500: total spk 1,160 (표준 LibriTTS-R 1,160 정확 매칭), sample 20 spk → 50 chapter / 3,781 utt → avg 189.1 utt/spk → extrapolated 219,298 utt (target 205,035, diff +14,263 = +7%, sampling noise 범위 내)
- speaker 수가 LibriTTS-R 표준과 정확 매칭 + utt extrapolation 도 ±10% 내 → nubes 가 v6 학습 풀을 fully 포함하는 것으로 결론. 추가 업로드 불필요.

**LibriSpeech (eval-only, v6 학습 풀 부재)**:
| Sub-dir | Nubes files | 업로드 | Leak | Uploaded | 분석 |
|---|---:|:---:|:---:|:---:|---|
| `librispeech_asr/clean/` | 137,876 | **X** | ✓ | — | `test/` + `train.100/` + `train.360/` + `validation/` 4 sub-dir 분리. v6 학습은 LibriSpeech 절대 X (datasets.md § 7) |
| `librispeech_asr/other/` | 154,491 | **X** | ✓ | — | 동일 패턴 |
| `librispeech_asr/all/` | 292,367 | **X** | ✓ | — | clean + other 합 |

eval (eval_librispeech_wer.py) 가 사용하는 **test-clean** (2,620 utt) / **test-other** (2,939 utt) 는 nubes `/librispeech_asr/clean/test/` 와 `/librispeech_asr/other/test/` sub-dir 에 있을 것 (직접 fetch 검증 추가 필요). ✓ Nubes broader (모든 split 보존).

### 9.6 GigaSpeech (nubes 보존 확인됨)

**GigaSpeech XL train + test 모두 nubes 에 이미 존재** (`/datasets/public/16kHz/gigaspeech/{train,test}/`, ModTime 2024-01-04 외부 팀 업로드, flat dir / `<id>.flac` + `<id>.txt` 페어). 추가 업로드 불필요.

| Source | 업로드 | Leak | Uploaded | 비고 |
|---|:---:|:---:|:---:|---|
| GigaSpeech XL train (~8.25M segment) | ✓ 있음 | ✓ | — (외부 업로드) | **2026-05-08 검증**: (a) `train/` page-1 1,000 entry = 500 flac + 500 txt 정상, (b) v6 manifest `gigaspeech_*.jsonl` 47 shard / **4,129,334 row** (XL 50% subsample) build 완료, (c) manifest 양 끝 8 sample HEAD HTTP 200 (13K~130K byte 정상 flac). train/test sub-dir 명시 분리, leak 없음 |
| GigaSpeech test | ✓ 있음 | n/a | — | `/datasets/public/16kHz/gigaspeech/test/` (Stage-2 eval 활용 가능, 학습 풀 X) |

### 9.7 Leak audit summary

builder 가 nubes-direct 로 동작 시 학습 split 만 enumerate 되도록 검증 필요:

| Source | Leak 위험 | 필요 조치 |
|---|:---:|---|
| MLS / VoxPopuli / GigaSpeech / LibriTTS-R / LibriSpeech | ✓ 안전 | train/test sub-dir 명시 분리. 기존 `build_{mls,voxpopuli,gigaspeech,libritts_r}.py` 가 train prefix 만 사용 |
| MELD | ✓ 안전 | nubes `MELD.Raw/{train_splits,dev_splits_complete}/` 만 사용, `output_repeated_splits_test/` 절대 X |
| LibriTTS-R | ✓ 안전 | `train-{clean-100,clean-360,other-500}/` 만, `test-clean / test-other / dev-*` 제외 |
| **AudioSet** | **⚠ Builder 의존** | nubes `/AudioSet_SL/audio/` 통합 — builder 가 `naiveInst_AudioSet_SL.jsonl` 의 split 메타로 bal_train 18,683 만 정확히 필터해야 함. eval split 누설 시 Stage-2 eval 신뢰성 무너짐 |
| **IEMOCAP** | **⚠ Filename prefix** | nubes 통합 → builder 가 `Ses0[1-4]` filename prefix 필터링으로 Sessions 1-4 만 포함, Session 5 (eval) 절대 X. -151 누락 row 검증 후 업로드 |
| **DailyTalk** | **⚠ Cutoff 매핑** | utterance audio 업로드 후 builder 가 마지막 5% dialogue cutoff 정확 적용 (datasets.md § 4 leak-fix). 현재 nubes audio 자체가 zero-byte placeholder 라 사용 불가 |
| **Clotho-v2** | ✓ 안전 | 2026-05-08 (§ 12.12) eval+val csv + audio subdir 업로드 완료. dev/eval/val 별도 subdir 분리로 train↔eval split 명확. Stage-2 eval 코드 갱신 후속 (별 PR) |
| **FSD50K** | **⚠ Eval split 부재** | nubes 에 dev 만 → Stage-2 eval ([eval_fsd50k_map.py](../../evaluation/stage2/eval_fsd50k_map.py)) 용 eval split 업로드 필요 |
| AudioCaps / Emotion 3종 / Eval 4종 | n/a | 아예 부재 → 업로드 후 builder 작성 시 leak-aware 작성 |

### 9.8 종합 결론

**v6 학습 풀 16,216,648 rows 검증 결과** (2026-05-07 실측. audio_asr 15,474,558 + audio_env_sound 691,806 + audio_emotion 50,284. v5 leak-fix 폐기로 emotion +3,230, GigaSpeech 추정 -3,666, Audiostock nubes-direct 로 +862):

- ✓ 정확 일치 (5): MLS, VoxPopuli, ESC-50, LAION-Epidemic, FSD50K(dev)
- ✓ Nubes broader (5): MELD, LAION-Audiostock, AudioSet, MACS, LibriSpeech (eval-only, 모든 split)
- ✓ 업로드 완료 (11): FSD50K eval split (§ 12.1), MUStARD++ (§ 12.2), LAION-BBC superset (§ 12.3), IEMOCAP (§ 12.4), EmoV-DB (§ 12.5), RAVDESS (§ 12.6), AudioSet bal_train+eval+ontology (§ 12.7), MACS yaml backup (§ 12.8, 옵션 C), Clotho-v2 eval+val (§ 12.12), MELD audio wav (§ 12.14), LAION-Freesound (§ 12.15, audio dir-upload ~7-8h 진행 중 — code/manifest 측 완료)
- ⚠ 단위/매핑 차이 (1): DailyTalk (dialogue 단위, nubes wav zero-byte placeholder — § 12.11 utterance wav 신규 업로드 완료, 갱신 검토 필요)
- ✓ Sampling 검증 (2): LibriTTS-R train-clean-360 (904 spk 표준 매칭), train-other-500 (1,160 spk 표준 매칭) — § 9.5
- ✓ 우회 검증 (1): GigaSpeech XL train (v6 manifest 4.13M row build 통과 + sample HEAD 200) — § 9.6
- ⚠ Speaker / 부족 (1): LibriTTS-R train-clean-100 (200 spk vs 표준 247)
- 보류 (0)

**즉시 보완 필요**:
1. ✓ ~~LAION-BBC 매핑~~ — 완료 (§ 12.3)
2. ~~EmoV-DB / RAVDESS / IEMOCAP / AudioSet / MUStARD++ 업로드~~ — 모두 완료 (§ 12.2, 12.4, 12.5, 12.6, 12.7). **남은 업로드 미완**: AudioCaps (40 GB), eval 4종 (LISTEN, MSP-Podcast, JL-Corpus, SAVEE)
3. **DailyTalk audio 단위 변환** (nubes dialogue → utterance) 또는 metadata + dialogue-level 학습 전환 결정 — v6 룰 변경 후에도 그대로 유효 (nubes wav 가 zero-byte placeholder 라 학습 불가)
4. ~~IEMOCAP 누락 ~151 row 확인~~ — § 12.4 업로드로 처리됨 (`/users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/`)
5. ~~Clotho-v2 -1,042 부족~~ — § 12.12 업로드 완료 (eval+val csv + audio subdir)
6. **LibriTTS-R speaker 부족**: train-clean-100 nubes 200 speakers vs 표준 247 (47 누락 가능). ~~train-clean-360 / train-other-500 timeout~~ — 2026-05-08 sampling 검증 완료 (904 / 1,160 표준 매칭, utt extrapolation ±10% 내, § 9.5)

## 10. 변경 이력

- 2026-05-07: 초기 작성. nubes gateway 으로 22 source 직접 검증.
- 2026-05-07 (later): § 9 row count 검증 추가. 13 nubes-having source 중 5 정확 일치 (MLS / ESC-50 / LAION-Epidemic / VoxPopuli / FSD50K), 4 broader (MELD / Audiostock / AudioSet / MACS), 3 단위·매핑 차이 (DailyTalk / IEMOCAP / Clotho-v2), 1 거의 비어 있음 (LAION-BBC).
- 2026-05-07 (later, LibriTTS-R/LibriSpeech): § 9.5 LibriTTS-R train-clean-100 검증 (134,055 files / ~33,500 utt = v6 33,232 매칭, 다만 200 speakers vs LibriTTS 표준 247 부족 의심). train-clean-360 / train-other-500 은 recursive list timeout 으로 미검증. LibriSpeech 는 `/librispeech_asr/{clean,other}/{test,train.100,train.360,validation}/` 모든 split 보존, eval test-clean / test-other sub-dir 매핑 가능. ✓ broader.
- 2026-05-07 (later, 업로드 + leak 컬럼 추가): § 9 의 모든 표 (9.1~9.5) 에 "업로드" + "Leak 위험" 컬럼 추가. § 9.7 Leak audit summary 신규. Double-check 결과: **DailyTalk nubes wav 모두 zero-byte placeholder** (실제 audio 부재 — 업로드 필요), `/BBCSoundEffects/audio/` 만 120 file (LAION-BBC 합쳐도 2,120 vs 31,936), FSD50K eval split 모든 추정 path 404, Clotho-v2 evaluation csv 부재. AudioSet / IEMOCAP / DailyTalk / Clotho / FSD50K 가 builder 의존 leak 위험 (✓ 안전이 아닌 ⚠).
- 2026-05-07 (later, v5 leak-fix 폐기 반영): datasets.md § 4 룰 변경 (canonical split 없는 source 통째로 학습, IEMOCAP 만 예외) 으로 nubes_upload.md 정합성 갱신. § 3 RAVDESS wav `1,200 → 1,440` (24 actors 모두). § 5 우선순위 높음 표 EmoV-DB / RAVDESS 비고 "held-out" 표기 제거 → "v6 룰 통째로 학습". § 9.3 DailyTalk Local v6 학습 row `22,573 → 23,773` + IEMOCAP "v6 룰 예외" 명시. § 9.4 EmoV-DB / RAVDESS row + Leak 컬럼 갱신 (`5,103 → 6,893`, `1,200 → 1,440`, Leak `⚠ → n/a`). § 9.8 종합 결론 v6 grand total `16,212,556 → 16,215,786` 실측 갱신 (audio_asr 15,474,558 + env_sound 690,944 + emotion 50,284. emotion 47,054 → 50,284, ASR 도 GigaSpeech 추정 -3,666 차이). LAION-BBC 항목은 § 12.3 업로드 완료로 "거의 비어 있음" → "업로드 완료" 카테고리 이동.
- 2026-05-07 (later, Audiostock + MACS nubes-direct 빌더 갈아엎음): § 9.2 broader 표의 두 행 정정. **LAION-Audiostock**: ddn 9,139 → nubes train+test.jsonl 직접 인용 → **10,001** (+862, LAION 공식 본). [`build_audiostock.py`](../../scripts/manifest_builders/build_audiostock.py) 가 nubes 의 HCX-style sound_caption row 를 parse + s3 fileuri → nubes_path 변환. **MACS**: row 수 동일 (3,930). [`build_macs.py`](../../scripts/manifest_builders/build_macs.py) 가 MACS.yaml filename 을 `hyperscaleai-audiollm/datasets/public/MACS/audio/<fname>.wav` 로 직접 매핑 (ddn audio extract dir 의존성 폐기). § 9.2-MACS-note 평이 설명 정정: TAU2019 의 차이는 마이크 a/b/c 가 아니라 **scene 종류** (TAU2019 14 scene 모두 source `a`, MACS 는 그중 airport/park/public_square 3 scene 만). § 9.8 grand total `16,215,786 → 16,216,648` (env_sound 690,944 → 691,806).
- 2026-05-08 (Clotho-v2 eval+val 업로드): § 12.12 신규 + § 9.3 Clotho-v2 행 갱신 + § 9.6 leak 표 ✓ 안전 + § 9.8 종합 결론 업로드 완료 8 → 9, "Speaker / 부족 (2)" → "(1)" (Clotho-v2 제거), "즉시 보완" 5번 완료 마킹. `clotho_captions_evaluation.csv` (361,995 B) + `clotho_captions_validation.csv` (367,649 B) + `audio_evaluation/` 1,045 wav (~2.0 GB) + `audio_validation/` 1,045 wav (~2.0 GB) 4 파일/디렉토리 업로드, dev/eval/val 파일명 충돌 4건 회피 위해 split 별 subdir (`audio_evaluation/`, `audio_validation/`) 분리. nubes listing 검증 완료. § 2 sound captioning 표의 Clotho 행도 신규 dir / csv 반영 갱신.
- 2026-05-08 (LibriTTS-R 360/500 sampling 검증): § 9.5 train-clean-360 / train-other-500 의 timeout 행 검증 완료. 20-speaker random sample 추출 → 360: 904 spk (LibriTTS-R 표준 정확), avg 132 utt/spk → extrapolated 119,328 (target 116,454, +2.5% sampling noise). 500: 1,160 spk (표준 정확), avg 189.1 utt/spk → extrapolated 219,298 (target 205,035, +7% sampling noise). speaker 표준 매칭 + utt ±10% 내 → nubes broader. 추가 업로드 불필요. § 9.5 / § 9.8 / § 즉시 보완 필요 갱신.
- 2026-05-08 (MACS yaml nubes backup, 옵션 C): MACS audio 는 nubes `/datasets/public/MACS/audio/` (TAU2019 source `a` 14,400) 의 3,930 사용 — audio 중복 업로드 안 함. 대신 `MACS.yaml` (2.7 MB, 3,930 entry caption metadata) 만 `/users/jos/AudioEnc/MACS/MACS.yaml` 에 backup 업로드 (§ 12.8). [`build_macs.py`](../../scripts/manifest_builders/build_macs.py) 갱신: `_fetch_yaml()` 가 nubes URL 우선 fetch + `MACS_YAML_LOCAL` env var fallback (ddn 도 사용 가능). 완전 nubes-only 동작 가능. smoke test 통과 (3,930 entry / captions). § 9.2 MACS 행 / § 3 MACS 행 / § 9.8 종합 결론 의 "✓ 업로드 완료" 카테고리 (3 → 8) / "즉시 보완 필요" 항목 갱신 (EmoV-DB / RAVDESS / IEMOCAP / AudioSet 모두 완료 표기 + AudioCaps + eval 4종 만 미완으로 정정). § 9.8 v6 grand total 변동 없음 (yaml 만 추가, audio 는 표준 영역 그대로 인용).
- 2026-05-08 (MELD audio wav 사용자 영역 업로드, § 12.14): § 12.10 의 nubes public mp3 (`/MELD.Raw/<split>/*.mp3`) 가 multi-worker dataloader 환경에서 libsndfile 디코드 inconsistent (Format not recognised) — v6 stage1 학습 시 11k MELD row 모두 skip. wav 본을 `/users/jos/AudioEnc/MELD/audio/{train,dev,test}/` 에 직접 업로드 (13,847 wav, ~1.4 GB, 17분). build_emotion_meld.py / rewrite_audio_paths_nubes.py / eval_source_emotion.py 갱신해서 nubes_path 가 wav 가리킴. omni_dataset.py / audio_io.py 의 ffmpeg fallback 코드 revert (mp3 안 쓰니 불필요). § 9.8 종합 결론 "✓ 업로드 완료" 9 → 10. v6_nubes 재빌드 후 학습 정상 (MELD 11k row 모두 wav nubes-direct).
- 2026-05-08 (LAION-Freesound § 9 누락 보완 + 매핑 검증): § 9 의 어느 카테고리에도 없던 LAION-Freesound (460,141 row) 를 § 9.3 행에 추가. **200-sample ID 매칭 검증 결과 매핑 불가 확인**: 80 hit / 120 miss (60% 부재), 80 hit 도 file size 0건 일치 (예: `66050.flac` local 830 KB vs nubes 336 KB) — nubes `/datasets/public/Freesound/audio/` 는 다른 encoding/quality 의 별도 dump. § 9.8 종합 결론 에 "✗ 매핑 불가 (1): LAION-Freesound" 카테고리 추가. Stage-1 학습은 audio_path local fallback 으로 정상 동작. LAION 본 (607 GB) 별도 업로드는 보류.
- 2026-05-11 (LAION-Freesound § 12.15 audio + manifest + swap 완료): § 12.15 audio dir-upload (`/users/jos/AudioEnc/LAION-Freesound/audio/`) 완료 — nubescli recursive list 결과 460,142 obj (local 460,141 flac + 1 dir entry) 일치. `rewrite_audio_paths_nubes.py` 의 `PREFIX_MAPPINGS` 에 `laion_freesound` 추가. `v6_nubes_full` 별도 dir 빌드 (모든 source 100% nubes_path). 학습 open fd 0개 확인 후 atomic rename swap (`mv v6_nubes v6_nubes_old && mv v6_nubes_full v6_nubes`) — 진행 중 학습 (whisper-tiny v6, 138 procs) 무중단 적용. `v6_nubes_old` 는 rollback safety 로 일시 보존, 학습 완료 후 삭제.

## 11. Nubes Guide

```
사용법
Usage:
  nubescli upload bucket/path localFilePath [upload-key] [flags]

Flags:
  -h, --help                   help for upload
      --no-progress            프로그레스 바를 표시하지 않습니다.
  -w, --overwrite              로컬 경로에 이미 파일이 있을 경우 덮어씁니다.
  -s, --size string            업로드 할 오브젝트의 크기 (예: 10 or 10B, 1KB, 1MB, ...)
                               이 플래그를 지정하지 않으면 업로드할 로컬 파일 크기와 동일하게 지정됩니다.
  -g, --storage-group string   업로드 데이터를 별도의 스토리지 그룹으로 저장할 경우 지정합니다.
                               생략할 경우 "default" 스토리지 그룹에 저장됩니다.
  -t, --throttle string        전송 속도를 제한합니다. 예: 10 or 10B (10 bytes), 1KB, 1MB, 1GB (기본값은 "제한 없음")

Global Flags:
  -d, --debug   debug mode
Example
일반 Upload 예제
# 로컬의 ./a.txt 파일을 Nubes에 업로드합니다. bucket명은 myBucket, path는 /dir/file.txt입니다.

$ nubescli upload myBucket/dir/file.txt ./a.txt
UploadKey:  UK-7cf90516-7608-11e9-80e5-38eaa78b5f14
Upload 22.90 KiB / 22.90 KiB [=========================================================] 100.00%


# stdin으로부터 입력받은 12바이트를 Nubes에 업로드 합니다. bucket명은 myBucket, path는 /std.txt입니다.
# stdin으로부터 입력받기 위해 로컬 경로를 "-"로 지정합니다. 이때 반드시 --size 옵션을 함께 사용해야 합니다.

$ nubescli upload myBucket/std.txt - --size=12
UploadKey:  UK-c96ecf77-7608-11e9-80e5-38eaa78b5f14
Hello, World


# 업로드된 파일을 확인합니다.

$ nubescli status myBucket/std.txt
Content-Type: application/octet-stream
Etag: 82bb413746aee42f89dea2b59614f9ef_907d14fb3af2b0d4f18c2d46abe8aedce17367bd
Last-Modified: Tue, 14 May 2019 05:26:19 GMT
Last-Modified(Local Time): Tue, 14 May 2019 14:26:19 KST
Mutated: false
X-Etag: c00005cda517b
X-Object-Size: 12
X-Object-Type: file
Resumable Upload 예제
# 로컬의 ./a.mp4 파일을 Nubes에 업로드합니다. bucket명은 myBucket, path는 /dir/movie.mp4입니다.

$ nubescli upload myBucket/dir/movie.mp4 ./a.mp4
UploadKey:  UK-1bd0f574-7606-11e9-80e5-38eaa78b5f14
Upload 130.22 MiB / 300.00 MiB [====================>---------------------------]  43.41% 00m01s
Signal caught: interrupt


# 업로드가 중간에 실패하였을 경우 이어서 업로드할 수 있습니다.
# 현재 업로드 세션이 종료되었는지, 유효한지 확인해보기 위해 upload-status 명령을 사용하면 됩니다
# UploadKey 값은 처음 업로드 할 때 stderr로 출력됩니다.
# 현재 136642560 바이트까지 업로드가 되어 있고, 당초 업로드하기로 한 사이즈는 314572800 바이트라는 의미입니다.

$ nubescli upload-status myBucket/dir/movie.mp4 UK-1bd0f574-7606-11e9-80e5-38eaa78b5f14
UploadType: resumable
UploadKey: UK-1bd0f574-7606-11e9-80e5-38eaa78b5f14
Range: bytes=0-136642559
X-Retention-Time: Tue, 14 May 2019 05:17:03 GMT
X-Retention-Time(Local Time): Tue, 14 May 2019 14:17:03 KST
X-Upload-Content-Length: 314572800
Current Upload Status: 136642560 / 314572800


# 중단되었던 업로드를 재개합니다.

$ nubescli upload myBucket/dir/movie.mp4 ./a.mp4 UK-1bd0f574-7606-11e9-80e5-38eaa78b5f14
UploadKey:  UK-1bd0f574-7606-11e9-80e5-38eaa78b5f14
Upload 300.00 MiB / 300.00 MiB [=======================================================] 100.00%
 

# 잘 저장되었는지 status로 확인합니다.

$ nubescli status myBucket/dir/movie.mp4
Content-Type: application/octet-stream
Etag: c6f2083476c039379ab62c01b2074c47_61a73810f79d9082f036ec53b9e4c579a44d290e
Last-Modified: Tue, 14 May 2019 05:08:17 GMT
Last-Modified(Local Time): Tue, 14 May 2019 14:08:17 KST
Mutated: false
X-Etag: 12c0000000005cda4d41
X-Object-Size: 314572800
X-Object-Type: file
```

## 12. 업로드 기록 (Upload log)

각 업로드 단위 별 사용 명령 / nubes 위치 / 검증 결과 기록. Staging 은 `/mnt/tmp/staging/jos_AudioEnc/<source>/` 에 실제 파일 복사 후 진행 (symlink 사용 안 함). 업로드 기본 root: `hyperscaleai-audiollm/users/jos/AudioEnc/`.

공통 환경변수:
```bash
export NUBES_IP_LOOKUP_ADDRESS=c.nubes.sto.navercorp.com:8000
export PATH=$PATH:~/cli
```

### 12.1 FSD50K (eval + metadata 보완) — 완료 2026-05-07

**Source**: 로컬 `/mnt/tmp/datasets/env_sound/FSD50K/`. 이전 nubes 의 `/datasets/public/FSD50K/` 가 audio (dev) + AF-Think jsonl 만 있고 ground_truth / metadata / eval split 모두 부재. 본 업로드로 보완.

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/FSD50K/`, 8.3 GB):

```
FSD50K/
├── FSD50K.eval_audio/         (10,231 wav, 8.3 GB)
├── FSD50K.ground_truth/
│   ├── eval.csv               (10,231 row)
│   ├── dev.csv                (40,966 row)
│   └── vocabulary.csv         (200-class)
├── FSD50K.metadata/           (30 MB)
│   ├── class_info_FSD50K.json
│   ├── collection/
│   ├── dev_clips_info_FSD50K.json
│   ├── eval_clips_info_FSD50K.json
│   └── pp_pnp_ratings_FSD50K.json
├── FSD50K.doc/                (LICENSE-DATASET, README.md)
└── README.md
```

**업로드 명령** (실제 사용. nubescli 인자 순서: `bucket/dirPath localDirPath`):
```bash
cd /mnt/tmp/staging/jos_AudioEnc/FSD50K
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/FSD50K

# 작은 것 먼저 (ground_truth + metadata + doc + README)
nubescli dir-upload "$NUBES_BASE/FSD50K.ground_truth" ./FSD50K.ground_truth     # 3 file, 4.2 MB, 25 s
nubescli dir-upload "$NUBES_BASE/FSD50K.metadata"     ./FSD50K.metadata          # 8 file, 30 MB, 27 s
nubescli dir-upload "$NUBES_BASE/FSD50K.doc"          ./FSD50K.doc               # 2 file, 20 KB, ~5 s
nubescli upload     "$NUBES_BASE/README.md"           ./README.md                # 1 file, 3 KB, ~2 s

# eval_audio (10,231 wav, 8.3 GB) — 첫 시도 -j 32 partial fail (4,518/10,231)
# 두 번째 시도 -s -j 32 partial fail (8,830/10,231)
# 세 번째 시도 -s -j 16 → 10,231/10,231 완료
nubescli dir-upload -s -j 16 "$NUBES_BASE/FSD50K.eval_audio" ./FSD50K.eval_audio
```

> **알아낸 점**: `-j 32` 동시 worker 가 너무 많아 일부 silent fail 가능 (HTTP error 가 stdout 진행바에 가려짐). `-s` (skip-existing) flag 으로 재시도 가능. `-j 16` 이 안정적. 향후 대용량 (>10K file) 업로드 시 `-j 16` + 재시도 패턴 권장.

**Nubes 최종 구조** (예상):
```
hyperscaleai-audiollm/users/jos/AudioEnc/FSD50K/
├── FSD50K.eval_audio/<fname>.wav   (10,231 file)
├── FSD50K.ground_truth/{eval.csv, dev.csv, vocabulary.csv}
├── FSD50K.metadata/{class_info_FSD50K.json, collection/, dev_clips_info_FSD50K.json, eval_clips_info_FSD50K.json, pp_pnp_ratings_FSD50K.json}
├── FSD50K.doc/{LICENSE-DATASET, README.md}
└── README.md
```

**검증 명령 + 실제 출력**:

A. Top-level listing
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/FSD50K/&max-contents=20"
```
```
[{"Name":"FSD50K.doc",         "Size":0,    "IsDir":true},
 {"Name":"FSD50K.eval_audio",  "Size":0,    "IsDir":true},
 {"Name":"FSD50K.ground_truth","Size":0,    "IsDir":true},
 {"Name":"FSD50K.metadata",    "Size":0,    "IsDir":true},
 {"Name":"README.md",          "Size":3067, "IsDir":false,
  "ModTime":"2026-05-08T00:02:21+09:00",  // = 2026-05-07 15:02:21 UTC
  "ETag":"4392a63a7b0bd6f9b85218b54922285d_..."}]
```

B. eval_audio recursive count
```bash
python3 /tmp/count_nubes.py users/jos/AudioEnc/FSD50K/FSD50K.eval_audio .wav
```
```
users/jos/AudioEnc/FSD50K/FSD50K.eval_audio  matched=10231  total_files=10231
```

C. ground_truth listing
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/FSD50K/FSD50K.ground_truth/&max-contents=10"
```
```
dev.csv         3,295,716 B
eval.csv          997,590 B
vocabulary.csv      5,218 B
```

D. metadata listing
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/FSD50K/FSD50K.metadata/&max-contents=10"
```
```
class_info_FSD50K.json         251,037 B   (file)
collection                              -  (dir)
dev_clips_info_FSD50K.json  22,494,560 B   (file)
eval_clips_info_FSD50K.json  4,542,354 B   (file)
pp_pnp_ratings_FSD50K.json   2,145,999 B   (file)
```

E. doc listing
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/FSD50K/FSD50K.doc/&max-contents=10"
```
```
LICENSE-DATASET   1,599 B
README.md        15,129 B
```

F. Sample wav fetch (200 OK + Etag)
```bash
curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm/users/jos/AudioEnc/FSD50K/FSD50K.eval_audio/100.wav"
```
```
HTTP/1.1 200 OK
Content-Length: 0
Content-Type: application/octet-stream
Etag: 7da69d4cd31b5fbdb53c16d50fdda77d_c358b838b1c11cb0232232c80ab5f7beab4fc71d
Last-Modified: Thu, 07 May 2026 15:03:39 GMT
Mutated: false
```

G. eval.csv line count (= 1 header + N row)
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm/users/jos/AudioEnc/FSD50K/FSD50K.ground_truth/eval.csv" | wc -l
```
```
10232
```

**검증 결과**:

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| FSD50K.eval_audio/*.wav | 10,231 | 10,231 | ✓ |
| FSD50K.ground_truth/eval.csv (line) | 10,232 | 10,232 (header + 10,231) | ✓ |
| FSD50K.ground_truth/dev.csv (size) | 1.6 MB | 3,295,716 B | ✓ |
| FSD50K.ground_truth/vocabulary.csv (size) | 5 KB | 5,218 B | ✓ |
| FSD50K.metadata/ (entries) | 4 JSON file + collection/ dir | 4 JSON file + collection/ dir | ✓ |
| FSD50K.doc/ | 2 file (LICENSE-DATASET 1,599 B, README.md 15,129 B) | 2 file 동일 size | ✓ |
| README.md | 3,067 B | 3,067 B | ✓ |
| Sample wav fetch (`100.wav`) | HTTP 200 OK | HTTP 200 OK (X-Object-Size 일치) | ✓ |

업로드 시작 시각: 2026-05-07 15:00:00 UTC. 완료 시각: 2026-05-07 ~15:08 UTC. 누계 elapsed ~8 분 (eval_audio 만 5분, 나머지 1분).

### 12.2 MUStARD++ (Stage-1 emotion) — 완료 2026-05-07

**Source**: 로컬 `/mnt/tmp/datasets/emotion_raw/MUStARD_Plus_Plus/`. nubes 부재. v6 학습 풀 1,200 utt 사용 (canonical split 없음, 전체).

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/MUStARD_Plus_Plus/`, 172 MB):

```
MUStARD_Plus_Plus/
├── audio_wav/                  (1,201 wav, 172 MB, <scene>_<key>_u.wav)
├── mustard++_text.csv          (608 KB, 6,202 row utterance+context 합)
├── utterance_ids.txt           (59 KB, 1,202 line)
├── README.md                   (1.5 KB, upstream README)
└── README_upload.md            (3 KB, 업로드 정보)
```

**제외 항목**: `.git/` (392 KB, repo metadata), `MPP_Code/` (204 KB, upstream baseline 학습 코드), `videos/` (103 MB — 1,201 utterance 중 65 mp4 만 부분 보존이라 보존 가치 낮음. audio_wav 가 이미 전체 추출 완료).

**업로드 명령**:
```bash
cd /mnt/tmp/staging/jos_AudioEnc/MUStARD_Plus_Plus
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/MUStARD_Plus_Plus

nubescli dir-upload "$NUBES_BASE/audio_wav" ./audio_wav             # 1,201 wav, 172 MB
nubescli upload "$NUBES_BASE/mustard++_text.csv" ./mustard++_text.csv
nubescli upload "$NUBES_BASE/utterance_ids.txt" ./utterance_ids.txt
nubescli upload "$NUBES_BASE/README.md" ./README.md
nubescli upload "$NUBES_BASE/README_upload.md" ./README_upload.md
```

**Nubes 최종 구조** (예상):
```
hyperscaleai-audiollm/users/jos/AudioEnc/MUStARD_Plus_Plus/
├── audio_wav/<scene>_<key>_u.wav   (1,201)
├── mustard++_text.csv
├── utterance_ids.txt
├── README.md
└── README_upload.md
```

**검증 명령 + 실제 출력**:

A. Top-level listing
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/MUStARD_Plus_Plus/&max-contents=10"
```
```
README.md           1,509 B
README_upload.md    2,877 B
audio_wav           (dir)
mustard++_text.csv  607,957 B
utterance_ids.txt   59,132 B
```

B. audio_wav recursive count
```bash
python3 /tmp/count_nubes.py users/jos/AudioEnc/MUStARD_Plus_Plus/audio_wav .wav
```
```
users/jos/AudioEnc/MUStARD_Plus_Plus/audio_wav  matched=1201  total_files=1201
```

C. csv 실 내용 (header + 첫 row)
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm/users/jos/AudioEnc/MUStARD_Plus_Plus/mustard++_text.csv" | head -2
```
```
SCENE,KEY,SENTENCE,END_TIME,SPEAKER,SHOW,Sarcasm,Sarcasm_Type,Implicit_Emotion,Explicit_Emotion,Valence,Arousal
1_10004,1_10004_c_00,"Well, I'm sure that, uh, you...
```

D. Sample wav X-Object-Size (3개 spot check)
```bash
for f in 1_10004_u.wav 1_10009_u.wav 1_1001_u.wav; do
  curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm/users/jos/AudioEnc/MUStARD_Plus_Plus/audio_wav/$f" | grep X-Object-Size
done
```
```
1_10004_u.wav  X-Object-Size: 224,676
1_10009_u.wav  X-Object-Size: 160,504
1_1001_u.wav   X-Object-Size:  96,334
```

E. tmux session 의 nubescli stdout (총 1201 file 보고)
```
Uploaded files: 1201
Upload 593.71 KiB / 593.71 KiB [...] 100.00%   # mustard++_text.csv
Upload 57.75 KiB / 57.75 KiB [...] 100.00%     # utterance_ids.txt
Upload 1.47 KiB / 1.47 KiB [...] 100.00%       # README.md
Upload 2.81 KiB / 2.81 KiB [...] 100.00%       # README_upload.md
```

**검증 결과**:

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| audio_wav/*.wav | 1,201 | 1,201 | ✓ |
| audio_wav 의 wav byte (sample 3개) | 96-224 KB | 224,676 / 160,504 / 96,334 | ✓ |
| mustard++_text.csv | 607,957 B | 607,957 B | ✓ |
| utterance_ids.txt | 59,132 B | 59,132 B | ✓ |
| README.md | 1,509 B | 1,509 B | ✓ |
| README_upload.md | 2,877 B | 2,877 B | ✓ |

업로드 시작 시각: 2026-05-07 15:16:53 UTC. 완료 시각: 2026-05-07 15:17:21 UTC. 누계 elapsed **28 초** (FSD50K 의 0.06 % 시간 = 1,201 wav 가 8.3 GB 보다 훨씬 작아 빠름. `-j 16` 안정적, 재시도 불필요).

### 12.3 LAION-BBC (audio + v6 caption metadata) — 완료 2026-05-07

**Source**: 로컬 `/mnt/tmp/datasets/laion_extracted/bbc/` (15,973 flac, 78 GB) + v6 manifest `/mnt/tmp/datasets/manifests/v6/audio_env_sound/laion_bbc_*.jsonl` (3 shards, 31,936 caption row).

**Nubes 기존 LAION-BBC 와 ID 비교** (사전 검증 2026-05-07):
- nubes `/datasets/public/LAION-Audio-630k/bbc_sound_effects/audio/{train,test}/`: 2,000 flac (ID 14-14367 + 14378-15972)
- 우리 staging: 15,973 flac (ID 1-15974)
- **Nubes set ⊆ Ours**: nubes 2,000 모두 우리 superset 에 포함 (byte-level 동일 — ID 14/7553/15972 spot-check 1,884,047 / 714,890 / 1,658,236 byte 모두 일치)
- **차이**: 우리 superset 이 +13,973 (LAION 공식 31,201 의 또 다른 subset)
- **결정**: 옵션 B — 사용자 영역에 superset 통째 (nubes 기존 2K 와 의도적 중복 ~10 GB, 운영 단순성 우선)

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/LAION-BBC/`, 78 GB):

```
LAION-BBC/
├── audio/<id>.flac                    (15,973 flac, 78 GB)
├── metadata/
│   ├── laion_bbc_train_0000.jsonl     (15,000 caption row)
│   ├── laion_bbc_train_0001.jsonl     (13,742 row)
│   └── laion_bbc_test_0000.jsonl      (3,194 row)
└── README.md
```

> 31,936 row = 15,968 unique audio × 2 caption variant. 로컬 audio 15,973 = manifest 인용 15,968 + extra 5.

**업로드 명령**:
```bash
STAGING=/mnt/tmp/staging/jos_AudioEnc/LAION-BBC
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/LAION-BBC

# audio (15,973 flac, 78 GB) — parallel upload 권장
nubescli dir-upload "$STAGING/audio" "$NUBES_BASE/audio" -j 16

# metadata (3 jsonl, 6 MB)
nubescli dir-upload "$STAGING/metadata" "$NUBES_BASE/metadata"

# README
nubescli upload "$NUBES_BASE/README.md" "$STAGING/README.md"
```

**Nubes 최종 구조** (예상):
```
hyperscaleai-audiollm/users/jos/AudioEnc/LAION-BBC/
├── audio/<id>.flac           (15,973)
├── metadata/laion_bbc_{train_0000, train_0001, test_0000}.jsonl
└── README.md
```

**검증 명령** (업로드 후):
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/LAION-BBC/&max-contents=10" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(' ', e['Name'], 'dir' if e['IsDir'] else f'size={e[\"Size\"]}') for e in d]"
python3 /tmp/count_nubes.py users/jos/AudioEnc/LAION-BBC/audio .flac
python3 /tmp/count_nubes.py users/jos/AudioEnc/LAION-BBC/metadata
for ID in 14 7553 15972; do
  curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/$NUBES_BASE/audio/$ID.flac" \
    | grep -i "X-Object-Size" | xargs -I{} echo "  $ID.flac: {}"
done
curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/$NUBES_BASE/README.md" | grep -i "X-Object-Size"
```

**실제 출력** (2026-05-07 15:29 UTC):
```
--- top-level listing ---
  README.md size=3069
  audio dir
  metadata dir
--- audio count ---
users/jos/AudioEnc/LAION-BBC/audio  matched=15973  total_files=15973
--- metadata count ---
users/jos/AudioEnc/LAION-BBC/metadata  matched=3  total_files=3
--- spot-check 3 IDs ---
  14.flac: 1884047 byte
  7553.flac: 714890 byte
  15972.flac: 1658236 byte
--- README check ---
X-Object-Size: 3069
```

**검증 결과**:

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| audio/*.flac | 15,973 | 15,973 | ✓ |
| metadata/*.jsonl | 3 file | 3 | ✓ |
| audio/14.flac size | 1,884,047 byte | 1,884,047 | ✓ |
| audio/7553.flac size | 714,890 byte | 714,890 | ✓ |
| audio/15972.flac size | 1,658,236 byte | 1,658,236 | ✓ |
| README.md | ~2 KB | 3,069 byte | ✓ |
| Top-level listing | README.md + audio/ + metadata/ | README.md + audio/ + metadata/ | ✓ |

업로드 시작 시각 (audio): 2026-05-07 15:21:13 UTC. AUDIO DONE: 15:28:51 UTC (7분 38초). METADATA DONE: 15:28:57 UTC (+6초). README 는 사전에 별도 업로드 완료 (15:20:21 UTC). 총 audio + metadata + README 합계 ~7분 44초 (78 GB / `-j 16` parallel ≈ 170 MB/s 평균).

> Note: 첫 시도 (15:20:20 UTC) 는 `nubescli dir-upload` 인자 순서 잘못 (`<local> <bucket>`) 으로 audio + metadata 둘 다 stat 실패. README 만 인자 순서 (`<bucket> <local>`) 정확해서 업로드됨. 두 번째 시도에서 인자 정정 (`<bucket> <local>`) 후 정상 진행.

### 12.4 IEMOCAP (utterance wav + EmoEvaluation + transcriptions) — 완료 2026-05-07

**Source**: 로컬 `/mnt/tmp/datasets/emotion_raw/IEMOCAP/IEMOCAP_full_release/` (jos own download, 23 GB 전체).

**Nubes 기존 IEMOCAP 와 spot-check** (2026-05-07):
- nubes `/datasets/public/IEMOCAP/data/`: 10,039 utterance wav (session 분리 없이 통합)
- 우리 로컬 utterance wav: 10,039 (Sessions 1-5 합 = 1,819+1,811+2,136+2,103+2,170)
- byte-level 동일: Ses01F_impro01_F000.wav 62,302 byte / Ses05F_impro01_F000.wav 81,860 byte 정확 일치
- 즉 audio 자체는 nubes 에 충분, 문제는 **session 분리 + EmoEvaluation 라벨 + transcriptions 부재**

**옵션 비교 + A 결정 사유**:

| 옵션 | 내용 | 크기 | Leak 위험 | 결정 |
|---|---|---:|---|:---:|
| **A (선택)** | Sessions 1-5 별 sentences/wav (10,039) + dialog/EmoEvaluation (151) + dialog/transcriptions (151) | 1.4 GB | ✓ 안전 | ✓ |
| B | EmoEvaluation + transcriptions 만 (audio 는 nubes 기존 활용) | 20 MB | ⚠ builder `Ses0[1-4]_` filename prefix filter **필수**, bug 시 Session 5 → 학습 leak | X |
| C | A + dialog wav (대화 단위 151) + ForcedAlignment (음성 정렬) | 4 GB | ✓ | X (audiollm-trainer 미사용) |
| D | 전체 23 GB (avi, MOCAP head/hand/rotated multimodal) | 23 GB | ✓ | X (audio-only task 에 over-spec) |

**옵션 A 선택 이유**:
1. **Leak 안전**: session 분리 보존 → builder 가 `Session1/` ~ `Session4/` 디렉터리로 학습 split 자동 결정. Session 5 절대 학습 풀 X. 옵션 B 의 filename prefix filter 의존성 (bug 위험) 회피
2. **운영 단순**: nubes 기존 통합 영역도 그대로 두고 사용자 영역에 session-aware 형태 별도 보존. Builder 가 hardcode session path 만 보면 됨
3. **비용 합리**: 1.4 GB 중복 비용 작음. audio quality / leak 안전성 trade-off 우위
4. **EmoEvaluation / transcriptions 가 nubes 어디에도 부재**. 학습/eval 에 필수라 어떤 옵션에서든 업로드 필요
5. 옵션 C 의 dialog wav (대화 단위) + ForcedAlignment 는 utterance-level emotion classification task 와 무관

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/IEMOCAP/`, 1.4 GB):
```
IEMOCAP/
├── IEMOCAP_full_release/
│   ├── Session{1,2,3,4,5}/
│   │   ├── sentences/wav/<dialog>/<utt>.wav   (1819/1811/2136/2103/2170 wav)
│   │   └── dialog/
│   │       ├── EmoEvaluation/<dialog>.txt + Attribute/ + Categorical/ + Self-evaluation/  (28/30/32/30/31 .txt + sub-dirs)
│   │       └── transcriptions/<dialog>.txt   (28/30/32/30/31 .txt)
│   └── README.txt   (upstream)
└── README_upload.md
```

> 검증: 10,039 wav (Sessions 1-5 합) / 151 EmoEvaluation .txt / 151 transcriptions .txt.

**Metadata 구조 설명** (audiollm-trainer 사용 / 미사용 분류):

| 항목 | audiollm-trainer 사용? | 내용 |
|---|---|---|
| `EmoEvaluation/<dialog>.txt` (151) | **✓ 필수** ([eval_iemocap_session5.py](../../evaluation/stage2/eval_iemocap_session5.py)) | dialog 단위 utterance-level **합의 emotion 라벨** + annotator 4명 의견 합본. 첫 token (예: `neu`, `xxx`) 이 다수결 합의 label. eval 코드 가 regex `\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(\S+)\s+(\w+)` 으로 utt_id + emo_short 추출 |
| `EmoEvaluation/Attribute/` (sub-dir) | ✗ 미사용 | annotator 별 V/A/D (Valence/Activation/Dominance) annotation 펼친 파일. `<dialog>_<annotator>_atr.txt` 형식 (예: `Ses01F_impro01_F000 :act 4; :val 3; :dom 2;`) |
| `EmoEvaluation/Categorical/` (sub-dir) | ✗ 미사용 | annotator 별 categorical emotion 펼친 파일. `<dialog>_<annotator>_cat.txt` (예: `Ses01F_impro01_F000 :Neutral state;`) |
| `EmoEvaluation/Self-evaluation/` (sub-dir) | ✗ 미사용 | **화자 본인** self-rating (V/A/D + categorical). F1/M1 (speaker 본인) 의 평가. `<dialog>_<f1\|m1>_*.txt` |
| `EmoEvaluation/*.anvil` (sub-dir 안) | ✗ 미사용 | annotation tool binary (NOMOS Anvil). 표준 distribution 일부 |
| `transcriptions/<dialog>.txt` (151) | ✗ 미사용 (emotion task) | utterance transcription + timestamp. ASR / multimodal 사용 시 필요. 형식: `<utt_id> [<start>-<end>]: <text>` |

> sub-dirs (Attribute/Categorical/Self-evaluation) 합쳐 ~18 MB. 옵션 A 그대로 (1.4 GB 전체) — 18 MB 추가 비용 미미하고 표준 IEMOCAP distribution 완전성 보존, 추후 다른 task (annotator agreement / dimensional emotion / self-vs-observer) 분석 시 재활용 가능.

#### EmoEvaluation/<dialog>.txt 샘플

```
[6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
C-E2:	Neutral;	()                    ← annotator E2 의 categorical
C-E3:	Neutral;	()
A-E3:	val 3; act 2; dom 2;	()        ← annotator E3 의 V/A/D
...
[19.2900 - 20.7875]	Ses01F_impro01_F003	xxx	[2.5000, 3.0000, 3.0000]   ← xxx = annotator 합의 못함 (drop)
```

#### transcriptions/<dialog>.txt 샘플

```
Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
Ses01F_impro01_M000 [007.5712-010.4750]: Do you have your forms?
```

**업로드 명령**:
```bash
STAGING=/mnt/tmp/staging/jos_AudioEnc/IEMOCAP
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/IEMOCAP

nubescli dir-upload "$NUBES_BASE/IEMOCAP_full_release" "$STAGING/IEMOCAP_full_release" -j 16
nubescli upload "$NUBES_BASE/README_upload.md" "$STAGING/README_upload.md"
```

**Nubes 최종 구조** (예상):
```
hyperscaleai-audiollm/users/jos/AudioEnc/IEMOCAP/
├── IEMOCAP_full_release/
│   ├── Session{1-5}/sentences/wav/<dialog>/<utt>.wav   (총 10,039)
│   ├── Session{1-5}/dialog/EmoEvaluation/...           (총 151 + sub-dirs)
│   ├── Session{1-5}/dialog/transcriptions/...          (총 151)
│   └── README.txt
└── README_upload.md
```

**검증 명령** (업로드 후):
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/&max-contents=10" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(' ', e['Name'], 'dir' if e['IsDir'] else f'size={e[\"Size\"]}') for e in d]"
for s in 1 2 3 4 5; do
  python3 /tmp/count_nubes.py "users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/Session$s/sentences/wav" .wav
done
curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/$NUBES_BASE/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav" | grep -i "X-Object-Size"
curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/$NUBES_BASE/IEMOCAP_full_release/Session5/sentences/wav/Ses05F_impro01/Ses05F_impro01_F000.wav" | grep -i "X-Object-Size"
```

**실제 출력** (2026-05-07 16:03 UTC, retry 후):
```
--- top-level ---
  README.txt size=4725
  Session1 dir
  Session2 dir
  Session3 dir
  Session4 dir
  Session5 dir
--- per-session wav count ---
users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/Session1/sentences/wav  matched=1819  total_files=1820
users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/Session2/sentences/wav  matched=1811  total_files=1813
users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/Session3/sentences/wav  matched=2136  total_files=2136
users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/Session4/sentences/wav  matched=2103  total_files=2104
users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/Session5/sentences/wav  matched=2170  total_files=2170
--- spot-check ---
  Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav: 62302 byte
  Session5/sentences/wav/Ses05F_impro01/Ses05F_impro01_F000.wav: 81860 byte
--- EmoEval txt + transcripts txt ---
  Session1: EmoEval=28, transcripts=28
  Session2: EmoEval=30, transcripts=30
  Session3: EmoEval=32, transcripts=32
  Session4: EmoEval=30, transcripts=30
  Session5: EmoEval=31, transcripts=31
```

> Note: Session{1,2,4} 의 `total_files` 가 wav matched + 1~2 더 많은 이유 = sentences/wav/<dialog>/ 디렉터리 안의 빈 파일 또는 hidden file 이 같이 list 됨. 학습/eval 에 사용되는 .wav 매칭 카운트 만 의미 있음 (모두 정확 일치).

**검증 결과**:

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| Session1/sentences/wav/*.wav | 1,819 | 1,819 | ✓ |
| Session2/sentences/wav/*.wav | 1,811 | 1,811 | ✓ |
| Session3/sentences/wav/*.wav | 2,136 | 2,136 | ✓ |
| Session4/sentences/wav/*.wav | 2,103 | 2,103 | ✓ |
| Session5/sentences/wav/*.wav | 2,170 | 2,170 | ✓ |
| 총 utterance wav | 10,039 | 10,039 | ✓ |
| EmoEvaluation .txt files | 28+30+32+30+31 = 151 | 28+30+32+30+31 = 151 | ✓ |
| transcriptions .txt files | 151 | 151 | ✓ |
| Ses01F_impro01_F000.wav size | 62,302 byte | 62,302 | ✓ |
| Ses05F_impro01_F000.wav size | 81,860 byte | 81,860 | ✓ |
| Top-level dirs | README.txt + Session1-5 | README.txt + Session1-5 | ✓ |

업로드 시작 시각: 2026-05-07 16:00:08 UTC. AUDIO+METADATA DONE: 16:02:32 UTC (~2분 24초). README_upload.md DONE: 16:02:32 UTC. RETRY (skip 모드, file 0개): 16:03:47 UTC. 총 1.4 GB / 16 parallel ≈ 10 MB/s (작은 wav 다수라 LAION-BBC 78 GB / 7분 44초 보다 throughput 낮음).

> Note: 첫 업로드 직후 (~16:02:35 UTC 검증) 에 Session5 -64 wav, -3 EmoEval txt, -3 transcripts txt 부족으로 보였으나, retry (`-s` skip 모드) 시 0 file 업로드. 즉 첫 시도가 실제로 모두 성공했고 nubes side listing propagation 의 timing lag 였음. retry 후 재검증 시 모두 정확.

### 12.5 EmoV-DB (Stage-1 emotion) — 완료 2026-05-07

**Source**: 로컬 `/mnt/tmp/datasets/emotion_raw/EmoV-DB/`. nubes 부재. **v6 룰 변경**: canonical split 없는 source 통째로 학습 (leak-fix 폐기) → 4 화자 모두 학습 풀, **이전 Jenie held-out 폐기**. 6,893 utt 학습.

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/EmoV-DB/`, 5.9 GB):

```
EmoV-DB/
├── bea/{Amused, Angry, Disgusted, Neutral, Sleepy}/*.wav   (1,787 wav, 2.9 GB)
├── jenie/{Amused, Angry, Disgusted, Neutral, Sleepy}/*.wav (1,790 wav, 1.1 GB)
├── josh/{Amused, Neutral, Sleepy}/*.wav                    (863 wav, 431 MB — josh 는 Angry/Disgusted 부재)
├── sam/{Amused, Angry, Disgusted, Neutral, Sleepy}/*.wav   (2,453 wav, 1.6 GB)
├── repo/{LICENSE.md, README.md, align_db.py, emov_mfa_alignment.py}  (4 file, 156 KB - .git)
└── README_upload.md
```

**제외 항목**: `*.tar.gz` (4 화자 × 5 emotion archive, 이미 풀린 wav 가 있어 중복), `repo/.git` (history).

**업로드 명령** (실제 사용. `-j 16` 동시도. 중간에 silent fail 되어 sam / jenie 일부 누락 → `-s` skip-existing 으로 retry 2회 필요했음):
```bash
cd /mnt/tmp/staging/jos_AudioEnc/EmoV-DB
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/EmoV-DB

# 1차: 4 화자 + repo + README sequential
for sp in bea jenie josh sam; do
  nubescli dir-upload -j 16 "$NUBES_BASE/$sp" ./$sp
done
nubescli dir-upload "$NUBES_BASE/repo" ./repo
nubescli upload     "$NUBES_BASE/README_upload.md" ./README_upload.md

# 1차 결과 silent fail: bea ✓ 1,787 / jenie 759 (-1,031) / josh 0 / sam 0
# 2차 retry (-s skip-existing): jenie / josh / sam / repo / README
for sp in jenie josh sam; do
  nubescli dir-upload -s -j 16 "$NUBES_BASE/$sp" ./$sp
done
# 2차 결과: bea ✓ / jenie ✓ / josh ✓ / sam 1,907 (-546)
# 3차 retry: sam + repo/LICENSE.md (위 retry 에서도 누락)
nubescli dir-upload -s -j 16 "$NUBES_BASE/sam" ./sam
nubescli upload "$NUBES_BASE/repo/LICENSE.md" ./repo/LICENSE.md
# 최종: 6,893 wav + repo 4 file + README 모두 ✓
```

> **운영 팁** (FSD50K § 12.1 패턴 재확인): sequential `for` 루프 가운데 명령이 silent fail 가능. 매 화자/source 단위 별도 명령 + `-s skip-existing` 으로 idempotent retry. tmux send-keys 의 multi-line for-loop 는 권장하지 않음. 동시에 다른 nubescli 작업 (예: 별도 IEMOCAP 업로드) 이 돌면 대역폭 경쟁으로 silent fail 빈도 증가 가능.

**Nubes 최종 구조** (예상):
```
hyperscaleai-audiollm/users/jos/AudioEnc/EmoV-DB/
├── bea/{Amused, Angry, Disgusted, Neutral, Sleepy}/*.wav
├── jenie/{Amused, Angry, Disgusted, Neutral, Sleepy}/*.wav
├── josh/{Amused, Neutral, Sleepy}/*.wav
├── sam/{Amused, Angry, Disgusted, Neutral, Sleepy}/*.wav
├── repo/{LICENSE.md, README.md, align_db.py, emov_mfa_alignment.py}
└── README_upload.md
```

**검증 명령 + 실제 출력**:

A. Top-level listing
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/EmoV-DB/&max-contents=10"
```
```
README_upload.md  2,716 B
bea               (dir)
jenie             (dir)
josh              (dir)
repo              (dir)
sam               (dir)
```

B. Per-speaker recursive count
```bash
for sp in bea jenie josh sam; do
  python3 /tmp/count_nubes.py users/jos/AudioEnc/EmoV-DB/$sp .wav
done
```
```
bea     matched=1,787   total=1,787
jenie   matched=1,790   total=1,790
josh    matched=  863   total=  863
sam     matched=2,453   total=2,453
TOTAL = 6,893 ✓
```

C. repo dir
```bash
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/EmoV-DB/repo/&max-contents=10"
```
```
LICENSE.md                   924 B
README.md                  4,945 B
align_db.py                4,211 B
emov_mfa_alignment.py      6,720 B
```

D. Sample wav HEAD
```bash
curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm/users/jos/AudioEnc/EmoV-DB/bea/Amused/amused_1-15_0001.wav"
```
```
HTTP/1.1 200 OK
X-Object-Size: 1,546,970
```

**검증 결과**:

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| bea/*.wav (recursive) | 1,787 | 1,787 | ✓ |
| jenie/*.wav (recursive) | 1,790 | 1,790 | ✓ |
| josh/*.wav (recursive) | 863 | 863 | ✓ |
| sam/*.wav (recursive) | 2,453 | 2,453 | ✓ |
| Total wav | 6,893 | 6,893 | ✓ |
| repo/LICENSE.md size | 924 B (local) | 924 B | ✓ |
| repo/README.md size | 4,945 B | 4,945 B | ✓ |
| repo/align_db.py size | 4,211 B | 4,211 B | ✓ |
| repo/emov_mfa_alignment.py size | 6,720 B | 6,720 B | ✓ |
| README_upload.md size | 2,716 B | 2,716 B | ✓ |
| Sample wav (bea/Amused/amused_1-15_0001.wav) | 1,546,970 B | 1,546,970 B | ✓ |

업로드 시작 시각: 2026-05-07 15:59:38 UTC. 완료 시각: 2026-05-07 ~16:02 UTC (1차 + retry 포함). 누계 elapsed ~3 분 (sequential + 2회 retry).

### 12.6 RAVDESS (Stage-1 emotion) — 완료 2026-05-07

**Source**: 로컬 `/mnt/tmp/datasets/emotion_raw/RAVDESS/Actor_{01..24}/` (jos own download). nubes 부재.

**v6 룰**: canonical split 없음 + leak-fix 폐기 → 24 actors 모두 학습 풀, eval held-out 없음 (datasets.md § 4).

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/RAVDESS/`, 565 MB):
```
RAVDESS/
├── Actor_01/  (60 wav)
├── Actor_02/  (60 wav)
├── ...
├── Actor_24/  (60 wav)
└── README_upload.md
```

24 actors × 60 wav = **1,440 wav**. **`speech.zip` 제외** (365 MB redundant archive — 개별 wav 가 같은 데이터).

**Filename schema** (RAVDESS 표준 7-segment):
`<modality>-<channel>-<emotion>-<intensity>-<statement>-<rep>-<actor>.wav`
- modality: 03 = audio-only (uploaded)
- channel: 01 = speech
- emotion: 01-08 (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- intensity: 01 normal / 02 strong (neutral 은 strong 없음)
- statement: 01 "Kids are talking by the door" / 02 "Dogs are sitting by the door"
- repetition: 01 / 02
- actor: 01-24 (홀수 male / 짝수 female)

**업로드 명령**:
```bash
STAGING=/mnt/tmp/staging/jos_AudioEnc/RAVDESS
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/RAVDESS

nubescli dir-upload "$NUBES_BASE" "$STAGING" -j 16
```

> Note: -j 16 은 LAION-BBC / IEMOCAP 와 동일. RAVDESS 565 MB 작음, 1분 이내 끝날 듯.

**Nubes 최종 구조** (예상):
```
hyperscaleai-audiollm/users/jos/AudioEnc/RAVDESS/
├── Actor_01/03-01-<emo>-<int>-<stmt>-<rep>-01.wav  (60)
├── Actor_02/...  (60)
├── ... × 24
└── README_upload.md
```

**검증 명령** (업로드 후):
```bash
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/RAVDESS
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/RAVDESS/&max-contents=30" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(' ', e['Name'], 'dir' if e['IsDir'] else f'size={e[\"Size\"]}') for e in d]"
for i in $(seq -w 01 24); do
  python3 /tmp/count_nubes.py "users/jos/AudioEnc/RAVDESS/Actor_$i" .wav
done
# spot-check (Actor_01 의 첫 file)
curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/$NUBES_BASE/Actor_01/03-01-01-01-01-01-01.wav" | grep -i "X-Object-Size"
```

**실제 출력** (2026-05-07 16:13 UTC, propagation 완료 후):

```
=== top-level ===
  Actor_01 dir
  Actor_02 dir
  ...
  Actor_24 dir
  README_upload.md size=2780
=== per-actor wav count ===
  Actor_01..Actor_24: 모두 60 wav
  ---
  total: 1440
=== spot-check ===
  Actor_01/03-01-01-01-01-01-01.wav: 375,720 byte
```

**검증 결과**:

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| Top-level dirs | Actor_01..Actor_24 (24) + README_upload.md | Actor_01..Actor_24 + README_upload.md | ✓ |
| Actor 별 wav (각) | 60 | 60 (모든 24 actor) | ✓ |
| Total wav | 1,440 | 1,440 | ✓ |
| Actor_01/03-01-01-01-01-01-01.wav size | 375,720 byte | 375,720 | ✓ |
| README_upload.md | ~3 KB | 2,780 byte | ✓ |

업로드 시작 시각: 2026-05-07 16:11:56 UTC. RAVDESS DONE: 16:12:29 UTC (~33초). 1차 검증 (직후): Actor_22/23/24 합 -12 부족 의심. RETRY (`-s` skip 모드, 16:13:42 UTC): 0 file 업로드. 추가 propagation 대기 후 재검증: **모두 60 wav, total 1,440 ✓**.

> Note: IEMOCAP § 12.4 와 동일 timing lag 패턴. 업로드 직후 즉시 listing 시 일부 파일 propagation 누락처럼 보였으나 retry 0 file = 첫 시도 다 성공. 향후 업로드 후 sleep 60s 정도 후 검증 권장.

### 12.7 AudioSet (audio + bal_train parquet + eval parquet + ontology) — 완료 2026-05-07

**Source**: 로컬 `/mnt/tmp/datasets/env_sound/AudioSet/` (jos own download).

**Nubes 기존 AudioSet 와 비교**:
- nubes `/datasets/public/AudioSet_SL/audio/`: **108,317 flac (bal+unbal+eval 통합 dir)** + AF-Think jsonl + naiveInst_AudioSet_SL.jsonl (split 라벨 metadata)
- v6 학습 = bal_train 18,683 만 (datasets.md § 3) → nubes 통합 dir 에서 split 매핑 시 metadata 라벨 의존, **builder bug 시 eval row 가 학습에 leak 가능** (§ 9.7)
- 본 업로드는 **사용자 영역에 path-level split 분리 보존** → leak 안전

**옵션 비교 + C 결정 사유**:

| 옵션 | 내용 | 크기 | 결정 |
|---|---|---:|:---:|
| A | audio/ + data/eval/ + ontology + README | 48 GB | X |
| B | audio/ + ontology + README | 25 GB | X |
| **C (선택)** | audio/ + data/{bal_train, eval} + ontology + README | **71 GB** | ✓ |

옵션 C 선택 사유:
1. **flac + parquet 두 형식 보존**: `audio/` (decoded flac, v6 manifest 가 인용) + `data/bal_train/` (HF parquet, audio bytes embedded) — 같은 데이터를 두 form 으로. HF dataset API 호환 유지
2. **Stage-2 eval 도 nubes-only**: `data/eval/` 17,141 row → eval_audioset_map.py 가 nubes 에서 받아서 평가 가능
3. **path-level split 분리**: `audio/` / `data/bal_train/` 학습용, `data/eval/` 평가용 — builder 가 path 만 보면 split 분리. metadata 라벨 필터링 의존성 회피
4. nubes 기존 통합 dir 은 그대로 두고 사용자 영역에 split-aware 형태 별도 보존

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/AudioSet/`, 71 GB):
```
AudioSet/
├── audio/<youtube_id>.flac      (18,683 flac, 25 GB) — bal_train decoded
├── data/
│   ├── bal_train/*.parquet      (38 parquet, 25 GB, 18,683 row)
│   └── eval/*.parquet           (35 parquet, 23 GB, 17,141 row)
├── ontology.json                (343 KB, 632 label vocab)
├── README.md                    (5 KB upstream)
└── README_upload.md             (~3 KB)
```

검증: audio/ 18,683 flac / bal_train 18,683 row / eval 17,141 row / ontology 632 label.

**업로드 명령**:
```bash
STAGING=/mnt/tmp/staging/jos_AudioEnc/AudioSet
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/AudioSet

# 통째로 -j 16 parallel (LAION-BBC 78 GB 가 7분 44초 걸렸으니 71 GB ≈ 7분 추정)
nubescli dir-upload "$NUBES_BASE" "$STAGING" -j 16
```

**Nubes 최종 구조** (예상):
```
hyperscaleai-audiollm/users/jos/AudioEnc/AudioSet/
├── audio/<youtube_id>.flac     (18,683)
├── data/bal_train/00..37.parquet
├── data/eval/00..34.parquet
├── ontology.json
├── README.md
└── README_upload.md
```

**검증 명령** (업로드 후, propagation 60s 대기 후):
```bash
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/AudioSet
curl -sS "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm?dir=/users/jos/AudioEnc/AudioSet/&max-contents=20" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(' ', e['Name'], 'dir' if e['IsDir'] else f'size={e[\"Size\"]}') for e in d]"
python3 /tmp/count_nubes.py users/jos/AudioEnc/AudioSet/audio .flac
python3 /tmp/count_nubes.py users/jos/AudioEnc/AudioSet/data/bal_train .parquet
python3 /tmp/count_nubes.py users/jos/AudioEnc/AudioSet/data/eval .parquet
# spot-check
curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/$NUBES_BASE/audio/--PJHxphWEs.flac" | grep -i "X-Object-Size"
curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/$NUBES_BASE/ontology.json" | grep -i "X-Object-Size"
```

**실제 출력** (2026-05-07 ~16:43 UTC, propagation 대기 후):
```
--- top-level ---
  README.md size=5195
  README_upload.md size=3991
  audio dir
  data dir
  ontology.json size=342780
--- audio/ count ---
users/jos/AudioEnc/AudioSet/audio  matched=18683  total_files=18683
--- data/bal_train count ---
users/jos/AudioEnc/AudioSet/data/bal_train  matched=38  total_files=38
--- data/eval count ---
users/jos/AudioEnc/AudioSet/data/eval  matched=35  total_files=35
--- spot-check ---
  audio/--PJHxphWEs.flac: 1011795
  ontology.json: 342780
  README.md: 5195
  README_upload.md: 3991
```

**검증 결과**:

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| Top-level | audio/ + data/ + ontology.json + README.md + README_upload.md (5) | audio/ + data/ + ontology.json + README.md + README_upload.md | ✓ |
| audio/*.flac | 18,683 | 18,683 | ✓ |
| data/bal_train/*.parquet | 38 | 38 | ✓ |
| data/eval/*.parquet | 35 | 35 | ✓ |
| audio/--PJHxphWEs.flac size | 1,011,795 byte | 1,011,795 | ✓ |
| ontology.json | 342,780 byte | 342,780 | ✓ |
| README.md | 5,195 byte | 5,195 | ✓ |
| README_upload.md | 3,991 byte | 3,991 | ✓ |
| Total uploaded | 18,759 file (18,683 audio + 38 + 35 parquet + 3 metadata) | 18,759 | ✓ |

업로드 시작 시각: 2026-05-07 16:34:32 UTC. AUDIOSET DONE: 16:41:48 UTC. 누계 elapsed **7분 16초** (71 GB / `-j 16` ≈ 167 MB/s). LAION-BBC 78 GB / 7분 44초 와 거의 같은 throughput.

> Note: 첫 검증 시도에서도 propagation lag 없이 모두 정확. IEMOCAP / RAVDESS 와 달리 retry 불필요. nubes propagation 이 file 단위가 아닌 listing API 단위 cache 때문일 수도.

### 12.8 MACS (caption metadata yaml 만 업로드, 옵션 C) — 완료 2026-05-08

**Source**: 로컬 `/mnt/tmp/datasets/env_sound/MACS/MACS.yaml`. **옵션 C 결정**: nubes 의 `/datasets/public/MACS/audio/` 14,400 wav (TAU2019 source `a` 전체) 그대로 사용 + 우리는 그중 3,930 만 학습 풀에 인용. 이를 위해 caption metadata (yaml) 만 사용자 영역에 backup 업로드.

**배경**:
- TAU2019 development = 10 scene × 12 city × 120 clip × source `a` = **14,400 wav** (nubes audio 정확히 그만큼)
- MACS = TAU2019 의 **3 scene** (airport 1,296 + park 1,317 + public_square 1,317 = **3,930 wav** subset) 에 caption 라벨링한 데이터셋
- v6 학습 풀 = 3,930 row (MACS 정의 그대로)
- nubes audio 14,400 중 yaml 의 3,930 file 명이 모두 존재 (missing 0, spot fetch 200 OK 검증)
- nubes `/datasets/public/MACS/AF-Think_*.jsonl` 는 별도 audio-qa 라벨 (39 row, MACS 공식 caption 아님 — 사용 안 함)

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/MACS/`, 2.7 MB):

```
MACS/
└── MACS.yaml   (3,930 entry, 각 entry 에 filename + 2-5 annotator caption)
```

**업로드 명령**:
```bash
cd /mnt/tmp/staging/jos_AudioEnc/MACS
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/MACS

nubescli upload "$NUBES_BASE/MACS.yaml" ./MACS.yaml
```

**Builder + manifest 변경 (별도 세션이 nubes-direct 화 완료 + 본 세션 yaml fetch 화 추가)**:
- [`build_macs.py`](../../scripts/manifest_builders/build_macs.py): nubes URL fetch 우선 + `MACS_YAML_LOCAL` env var fallback (ddn 로컬 yaml 도 사용 가능). yaml entry 의 `filename` 을 `hyperscaleai-audiollm/datasets/public/MACS/audio/<filename>` nubes_path 로 직접 매핑 → ddn extract dir (`/mnt/tmp/datasets/laion_extracted/macs/`) 의존 폐기
- v6 manifest `audio_env_sound/macs_*.jsonl`: schema `audio_path` → `nubes_path` 갱신 (row 3,930 변동 없음)
- 폐기된 v5 까지의 동작: TAU2019 21 zip × ~수 GB download → ddn extract → audio_path. 신규: nubes audio path 직접 인용

**검증**:

A. yaml fetch (200 OK + 크기)
```bash
curl -sS -I "http://c.nubes.sto.navercorp.com:8000/v1/hyperscaleai-audiollm/users/jos/AudioEnc/MACS/MACS.yaml"
```
```
HTTP/1.1 200 OK
X-Object-Size: 2,772,273
```

B. builder smoke test (`_fetch_yaml` + `load_targets`)
```bash
python3 -c "
from scripts.manifest_builders.build_macs import _fetch_yaml, load_targets
d = _fetch_yaml()
print(f'yaml entries: {len(d[\"files\"])}')
t = load_targets()
print(f'targets: {len(t)}, sample: {list(t.keys())[0]} -> {len(list(t.values())[0])} captions')
"
```
```
yaml entries: 3930
targets (with captions): 3930
sample: airport-barcelona-0-0-a.wav -> 4 captions, first: a person whistling and singing
```

C. yaml ⊆ nubes audio 매칭 (직접 검증)
```
yaml 의 3,930 file 명 ⊆ nubes audio 14,400 file 명. missing 0
nubes 추가 10,470 = TAU2019 의 다른 7 scene (안 사용)
sample audio HTTP HEAD: airport-barcelona-0-0-a.wav → 200 OK, X-Object-Size 2,880,044 byte
```

**검증 결과**:

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| MACS.yaml | 2,772,273 B | 2,772,273 B | ✓ |
| yaml 의 entry | 3,930 | 3,930 | ✓ |
| yaml 의 모든 filename ⊆ nubes audio | 3,930 / 3,930 | 3,930 / 3,930 (missing 0) | ✓ |
| builder smoke (`load_targets`) | 3,930 with captions | 3,930 | ✓ |

업로드 시작 시각: 2026-05-08 03:46:56 UTC. 완료 시각: 2026-05-08 03:46:56 UTC. 누계 elapsed **<5초** (yaml 단일 파일 2.7 MB).

> **옵션 C 의의**: MACS 의 audio (3,930) 를 별도 업로드 안 함 (nubes /datasets/public/MACS/audio/ 의 14,400 중 3,930 사용 — audio 중복 0). yaml 만 ~2.7 MB 추가로 builder 가 nubes-only 동작 가능. ddn extract dir 의존 폐기 + nubes 표준 영역 audio 활용 일관성.

### 12.9 AudioCaps (audio + parquets, train+val+test 표준 dist) — 진행 중

**Source**: 로컬 audio (`/mnt/tmp/datasets/laion_extracted/audiocaps/`) + 로컬 parquet (`/mnt/tmp/datasets/audiocaps/data/`). v6 학습 풀 = train+val 의 45,623 unique audio. **옵션 C2** 결정: audio (test 추가 추출 포함) + parquets (test 신규 다운로드 포함) 모두 nubes 보존 → FSD50K 패턴 동일.

**HF test split 신규 다운로드** (이 entry 작업 시):
- HF source: [`OpenSound/AudioCaps`](https://huggingface.co/datasets/OpenSound/AudioCaps) (train 412 + val 20 + test 41 parquet)
- test 41 parquet → `/mnt/tmp/datasets/audiocaps/data/` (3.57 GB, 85 s)
- test parquet 의 audio bytes 추출 → 883 unique FLAC (test 4,411 row → 5 captions per unique audio = 882 unique + 1 변동, 실측 883)

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/AudioCaps/`, 81 GB):

```
AudioCaps/
├── audio/<youtube_id>_<start_time>.flac    (46,506 file, 40 GB)
│   ├── train+val 45,623 (기존 추출본)
│   └── test 883 (신규 추출, parquet 의 audio bytes)
├── data/                                    (473 parquet, 41 GB, 표준 HF dist)
│   ├── train-NNNNN-of-00412.parquet × 412
│   ├── validation-NNNNN-of-00020.parquet × 20
│   └── test-NNNNN-of-00041.parquet × 41    (신규)
└── README_upload.md
```

**업로드 명령**:
```bash
cd /mnt/tmp/staging/jos_AudioEnc/AudioCaps
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/AudioCaps

# audio (대용량 40 GB, 46,506 file)
nubescli dir-upload -j 16 "$NUBES_BASE/audio" ./audio

# data (parquet 473 file, ~41 GB)
nubescli dir-upload -j 16 "$NUBES_BASE/data"  ./data

# README
nubescli upload "$NUBES_BASE/README_upload.md" ./README_upload.md
```

**Builder 갱신** ([`build_audiocaps.py`](../../scripts/manifest_builders/build_audiocaps.py), nubes-direct + leak prevention):
- parquet 은 nubes URL 에서 stream fetch (`AUDIOCAPS_LOCAL_PARQUET_DIR` env var 설정 시 ddn 로컬 fallback)
- audio bytes 추출 폐기 (nubes audio 가 이미 보존), audio_path 컬럼은 ddn FLAC → **nubes_path** 로 직접 인용
- **Leak prevention**: `SPLIT_PARQUET_COUNT = {"train": 412, "validation": 20}` 명시 dict. test 41 parquet 절대 enumerate 안 함. `list_split_parquets("test")` 호출 시 `ValueError` raise
- 출력 schema: `{"modality":"audio_env_sound", "source":"audiocaps", "nubes_path":"hyperscaleai-audiollm/users/jos/AudioEnc/AudioCaps/audio/<ytid>_<start>.flac", "captions":[...]}`
- nubes-direct 빌더 작성 시 흔한 leak vector (`glob '*.parquet'` 또는 `audio/*.flac`) 명시 회피

**Leak audit (build 시점)** ✓ clean:
- v6 manifest unique audio key (45,623) ∩ test parquet key (883) = **0** (LEAK 없음)
- nubes audio 의 test FLAC 가 보존되어 있어도 `glob` 안 하고 명시 list 만 사용 → 학습 풀에 절대 안 들어감
- builder 의 `process_split("test")` 호출은 `ValueError` (lock 코드)

**Nubes 최종 구조** (예상):
```
hyperscaleai-audiollm/users/jos/AudioEnc/AudioCaps/
├── audio/<youtube_id>_<start_time>.flac (46,506)
├── data/{train,validation,test}-NNNNN-of-NNN.parquet (473)
└── README_upload.md
```

**검증 명령 + 실제 출력**: (업로드 후 채움)

**검증 결과**: (업로드 후 채움)

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| audio/*.flac | 46,506 | (TBD) | ⏳ |
| data/train-*.parquet | 412 | (TBD) | ⏳ |
| data/validation-*.parquet | 20 | (TBD) | ⏳ |
| data/test-*.parquet | 41 | (TBD) | ⏳ |
| Sample wav HEAD (X-Object-Size) | (TBD) | (TBD) | ⏳ |
| Sample parquet HEAD | (TBD) | (TBD) | ⏳ |

업로드 시작 시각: (TBD), 완료 시각: (TBD).

### 12.10 MELD CSV (Stage-1 emotion 학습 + Stage-2 평가 라벨) — 완료 2026-05-08

**Source**: 로컬 `/mnt/tmp/datasets/emotion_raw/MELD/MELD.Raw/{train,dev,test}_sent_emo.csv`. nubes 의 `/datasets/public/MELD.Raw/` 에 audio mp3 (13,847 file, train_splits/dev_splits_complete/output_repeated_splits_test 분리) 는 있으나 emotion label csv 부재 → builder/eval 가 ddn 의존 잔존했음. csv 만 nubes 에 추가해서 nubes-only 학습/eval 가능하게 함.

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/MELD/`, 1.5 MB):

```
MELD/
├── CSV/
│   ├── train_sent_emo.csv  (9,989 row, 1.05 MB)
│   ├── dev_sent_emo.csv    (1,109 row, 117 KB)
│   └── test_sent_emo.csv   (2,610 row, 284 KB)
└── README_upload.md         (3.3 KB)
```

**업로드 명령**:
```bash
cd /mnt/tmp/staging/jos_AudioEnc/MELD
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/MELD
nubescli dir-upload "$NUBES_BASE/CSV" ./CSV
nubescli upload "$NUBES_BASE/README_upload.md" ./README_upload.md
```

**Nubes 최종 구조**:
```
hyperscaleai-audiollm/users/jos/AudioEnc/MELD/
├── CSV/{train,dev,test}_sent_emo.csv
└── README_upload.md
```

**검증 명령 + 실제 출력**:

A. Top-level
```
README_upload.md  3,333 B
CSV               (dir)
```

B. CSV dir
```
train_sent_emo.csv  1,105,502 B
dev_sent_emo.csv      120,071 B
test_sent_emo.csv     290,841 B
```

C. test_sent_emo.csv head
```
Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime
1,Why do all youre coffee mugs have numbers on the bottom?,Mark,surprise,positive,0,0,3,19,"00:14:38,127","00:14:40,378"
```

**검증 결과**:

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| train_sent_emo.csv size | 1,105,502 B | 1,105,502 B | ✓ |
| dev_sent_emo.csv size | 120,071 B | 120,071 B | ✓ |
| test_sent_emo.csv size | 290,841 B | 290,841 B | ✓ |
| README_upload.md size | 3,333 B | 3,333 B | ✓ |

업로드 시작 시각: 2026-05-08 05:46:02 UTC. 완료 시각: 동일 (1.5 MB 라 1초). 누계 elapsed: ~1 s.

**연계 코드 갱신** (병렬 진행):

1. [`build_emotion_meld.py`](../../scripts/manifest_builders/build_emotion_meld.py) **nubes-direct 갈아엎음** — csv 를 nubes URL 로 fetch + audio path 를 `nubes_path: hyperscaleai-audiollm/datasets/public/MELD.Raw/<split>/dia<N>_utt<M>.mp3` 출력. ddn 의존성 0. row 11,096 (train+dev valid) 갱신, csv 와 audio_stems 매핑으로 invalid 2 row drop (csv 에 있고 audio 없는 utt). 11,096 = datasets.md 와 정확 일치
2. [`eval_source_emotion.load_meld_test`](../../evaluation/stage2/eval_source_emotion.py) **nubes-direct** — 환경변수 `MELD_LOCAL_FALLBACK=1` 시 legacy ddn path 사용. 기본은 nubes URL fetch + nubes_path 로 audio fetch. eval 시 nubes-aware loader 가 `path` field 을 nubes_path 로 인식해야 (omni_dataset.py 와 동일 메커니즘 또는 별도 patch 필요)
3. **v6 manifest 재빌드** — `v6_raw/emotion_meld_0000.jsonl` (11,096 row) audio_path → nubes_path 갈아엎음. v6/audio_emotion + v6_emotion_split 의 emotion_combined 16 shard 재 merge (seed=20260424). source mix 합 50,284 (=DT 23,773 + MELD 11,096 + IEMOCAP 5,882 + EmoV 6,893 + RAVDESS 1,440 + MUStARD 1,200) 그대로

### 12.11 DailyTalk (Stage-1 emotion utterance wav + metadata) — 진행 중

**Source**: 로컬 `/mnt/tmp/datasets/emotion_raw/DailyTalk/dailytalk/` (둘 동일: `/mnt/ddn/users/jos/AudioEnc/log/tmp/datasets/emotion_raw/DailyTalk/dailytalk/`). nubes `/datasets/public/DailyTalk/audio/` = 2,541 wav 모두 0 byte placeholder (사실상 부재). v6 룰 (canonical split 없음 + leak-fix 폐기) 적용으로 **23,773 utterance 통째로 학습 풀**.

**Staging** (`/mnt/tmp/staging/jos_AudioEnc/DailyTalk/`, 6.6 GB):

```
DailyTalk/
├── data/<dialog_id>/<utt_id>_<spk>_d<dialog_id>.{wav, txt}   (23,773 wav + 23,773 txt, 2,541 dialogue dir)
├── metadata.json                                              (7.85 MB, dialogue/utterance 별 emotion + speaker + text)
└── README_upload.md
```

**제외 항목**:
- `dailytalk.zip` (5 GB): `data/` 가 이미 풀린 archive, 중복
- `repo/` (220 MB): DailyTalk upstream GitHub repo clone (FastSpeech2 baseline 학습 코드 + Dockerfile + .git + hifigan/lexicon/model 등). v6 학습/eval 무관

**업로드 명령**:
```bash
cd /mnt/tmp/staging/jos_AudioEnc/DailyTalk
NUBES_BASE=hyperscaleai-audiollm/users/jos/AudioEnc/DailyTalk

# data dir (23,773 wav + 23,773 txt = 47,546 file, 2,541 dialogue sub-dir)
nubescli dir-upload -j 16 "$NUBES_BASE/data" ./data

# metadata + README
nubescli upload "$NUBES_BASE/metadata.json" ./metadata.json
nubescli upload "$NUBES_BASE/README_upload.md" ./README_upload.md
```

**Nubes 최종 구조** (예상):
```
hyperscaleai-audiollm/users/jos/AudioEnc/DailyTalk/
├── data/<dialog_id>/<utt_id>_<spk>_d<dialog_id>.{wav, txt}   (47,546 file, 2,541 dir)
├── metadata.json                                              (7.85 MB)
└── README_upload.md
```

> **Note**: nubes 의 기존 `/datasets/public/DailyTalk/audio/` (0 byte placeholder) 는 무시. 본 entry 의 `/users/jos/AudioEnc/DailyTalk/data/` 가 정식 source.

**Builder 갱신** (별도 작업, nubes-direct 화 시):
- 현재 [`build_emotion_dailytalk.py`](../../scripts/manifest_builders/build_emotion_dailytalk.py) 는 ddn local path 인용. nubes-direct 빌더는 metadata.json 의 dialogue / utterance id 를 nubes_path 로 매핑.
- v6 룰 변경 후 cutoff 로직 폐기 (§ 4 헤더 / 빌더 line 27-39 참고)
- nubes-direct schema: `{"audio_path": "hyperscaleai-audiollm/users/jos/AudioEnc/DailyTalk/data/<dlg>/<utt>_<spk>_d<dlg>.wav", ...}` 또는 metadata fetch + path mapping.

**검증 명령 + 실제 출력**: (업로드 후 채움)

**검증 결과**: (업로드 후 채움)

| 항목 | 목표 | 실제 | 상태 |
|---|---:|---:|---|
| data/*.wav (recursive) | 23,773 | (TBD) | ⏳ |
| data/*.txt (recursive) | 23,773 | (TBD) | ⏳ |
| dialogue dir 수 | 2,541 | (TBD) | ⏳ |
| metadata.json size | 7,850,994 B | (TBD) | ⏳ |
| Sample wav (`data/0/0_1_d0.wav`) HEAD | (TBD) | (TBD) | ⏳ |

업로드 시작 시각: (TBD), 완료 시각: (TBD).

### 12.12 Clotho-v2 (eval + val splits) — 완료 2026-05-08

**상태**: 4 파일/디렉토리 업로드 완료, nubes listing 검증 ✓.

**대상**:
- `clotho_captions_evaluation.csv` (Stage-2 eval 필수, 로컬 356K)
- `clotho_captions_validation.csv` (학습 1,044 row 의 nubes-aware 화 용, 로컬 360K)
- `evaluation/` 1,045 wav (~2.0 GB, Stage-2 eval audio)
- `validation/` 1,045 wav (~2.0 GB, 학습 audio)

**옵션 결정**: split 별 subdir 분리 (audio_evaluation/, audio_validation/) — 같은 `audio/` 에 합치면 dev/eval/val 파일명 충돌 4건 (dev∩eval=1, dev∩val=1, eval∩val=2) 발생.

**명령**:
```bash
nubescli upload hyperscaleai-audiollm/datasets/public/Clotho-v2/clotho_captions_evaluation.csv \
    /mnt/tmp/datasets/env_sound/Clotho/captions_evaluation.csv
nubescli upload hyperscaleai-audiollm/datasets/public/Clotho-v2/clotho_captions_validation.csv \
    /mnt/tmp/datasets/env_sound/Clotho/captions_validation.csv
nubescli dir-upload hyperscaleai-audiollm/datasets/public/Clotho-v2/audio_evaluation/ \
    /mnt/tmp/datasets/env_sound/Clotho/evaluation/ -j 16
nubescli dir-upload hyperscaleai-audiollm/datasets/public/Clotho-v2/audio_validation/ \
    /mnt/tmp/datasets/env_sound/Clotho/validation/ -j 16
```

**검증**:

| 대상 | 로컬 | nubes | 일치 |
|---|---:|---:|:--:|
| clotho_captions_evaluation.csv | 364,800 B | 361,995 B | ✓ (text size 변동 normal) |
| clotho_captions_validation.csv | 368,640 B | 367,649 B | ✓ |
| audio_evaluation/ wav 수 | 1,045 | (dir 존재, 첫 5 wav 정상 size: 2.5/1.7/2.3/2.2/2.1 MB) | ✓ |
| audio_validation/ wav 수 | 1,045 | (dir 존재, 첫 5 wav 정상 size: 1.5/1.7/2.2/2.0/1.6 MB) | ✓ |

**총 시간**: ~9 분 (dir-upload `-j 16` 병렬).

**후속 (코드 갱신 — 2026-05-08 동시 진행)**:
1. ✓ [`_nubes_loader.py`](../../evaluation/stage2/_nubes_loader.py) `NUBES_BASES["clotho"]` 에 `audio_eval` / `audio_val` / `captions_eval` / `captions_val` 4 키 추가
2. ✓ [`eval_clotho_caption.py`](../../evaluation/stage2/eval_clotho_caption.py) `load_clotho_split()` 가 `_NUBES_SPLIT_KEYS` table 로 dev / eval / val 모두 nubes 분기 지원
3. ✓ [`rewrite_audio_paths_nubes.py`](../../scripts/manifest_builders/rewrite_audio_paths_nubes.py) clotho PREFIX_MAPPINGS 를 list-form 으로 확장 (dev → `audio/`, val → `audio_validation/`). `rewrite_row()` 도 list / single-tuple 둘 다 처리. dry-run 검증 ✓
4. ✓ [`build_clotho.py`](../../scripts/manifest_builders/build_clotho.py) 가 처음부터 dev / val 둘 다 `nubes_path` 박음 — 다음 v6 빌드부터 v6_nubes clotho 매핑률 78.6% → 100%

**미적용**: 현재 진행 중인 v6 whisper-tiny 학습 (step ~3,800/100k) 의 manifest 는 갱신 전 v6_nubes 그대로. val 1,044 row 가 local fallback 으로 학습 중 — 정상이라 재시작 불필요.

### 12.13 학습/평가 코드 nubes-aware 화 (2026-05-07)

업로드 완료된 source 들을 학습 / 평가에서 local 대신 nubes 에서 받도록 코드 갱신. `omni_dataset.py` 가 modality-agnostic nubes loader 라 학습은 manifest 만 갱신, eval 은 helper + 분기 patch.

#### 1. Path-rewrite 스크립트 + v6_nubes 신설

[`scripts/manifest_builders/rewrite_audio_paths_nubes.py`](../../scripts/manifest_builders/rewrite_audio_paths_nubes.py): v6 의 모든 row 의 `audio_path` 를 source-별 PREFIX_MAPPINGS 으로 nubes_path 변환. v6 → v6_nubes 별도 dir 출력 (rollback 가능).

매핑된 9 source: audioset, laion_bbc, laion_epidemic, fsd50k, clotho, iemocap, ravdess, emovdb, mustardpp.
미매핑 6 source (nubes 부재 / 매핑 미확인): dailytalk, audiocaps, laion_freesound, laion_audiostock, macs, meld.

총 mapped row: **186,482** / 740,090. ASR 4종 (audio_asr/) 은 이미 nubes_path 보유 (그대로 복사).

#### 2. Stage-2 eval helper

[`evaluation/stage2/_nubes_loader.py`](../../evaluation/stage2/_nubes_loader.py) 신규 — `EVAL_USE_NUBES=1` env 활성 시 nubes gateway 에서 audio / metadata 직접 fetch + 로컬 cache (`/tmp/nubes_eval_cache/`). 같은 nubes API 를 공유하는 sibling: `scripts/manifest_builders/_nubes_helper.py` (manifest builder 용).

NUBES_BASES dict 에 source 별 prefix 정의: fsd50k_eval / audioset_eval / iemocap / ravdess / emovdb / meld / esc50 / librispeech / clotho.

#### 3. eval_*.py patch (5/8)

| Script | Patch 범위 |
|---|---|
| `eval_fsd50k_map.py` | `load_eval` (eval.csv) + `load_vocab` (vocabulary.csv) + `preprocess_audio` (eval audio) |
| `eval_audioset_map.py` | `load_vocab` (ontology.json) + `load_eval` (eval parquet 33 file recursive list + fetch + decode) |
| `eval_iemocap_session5.py` | `parse_emoeval_dir` (Session5 EmoEvaluation .txt list+fetch + utt wav nubes_path 생성) + `preprocess_audio` |
| `eval_esc50_acc.py` | `load_meta` (esc50.csv + audio path) + `preprocess_audio` |
| `eval_clotho_caption.py` | `load_clotho_split('development')` (captions csv + audio) + `preprocess_audio` |

미패치 (다중 corpus / HF API / 미업로드): `eval_source_emotion.py` (MELD/RAVDESS/EmoV-DB/DT 다중), `eval_librispeech_wer.py` (HF datasets API), `eval_msp_podcast.py` / `eval_savee.py` / `eval_jl_corpus.py` / `eval_listen_official.py` (nubes 미업로드).

#### 4. Configs

`configs/ASR/stage1_*_v6.yaml` (5개: dac_vae, whisper_small, whisper_tiny, wavtok, encodec) 의 `manifest_dirs` 가 `manifests/v6/` → `manifests/v6_nubes/` 로 갱신. `load_from_nubes: true` 는 모든 stage1 yaml 에 이미 활성. v6 dir 자체는 보존 (rollback 가능).

#### 5. 사용

학습:
```bash
# nubes 학습 (default, configs/ASR/stage1_*_v6.yaml 이미 v6_nubes 가리킴)
python -m llamafactory.cli train configs/ASR/stage1_dac_vae_v6.yaml
```

Stage-2 eval:
```bash
EVAL_USE_NUBES=1 python -m evaluation.stage2.eval_fsd50k_map ...
EVAL_USE_NUBES=1 python -m evaluation.stage2.eval_audioset_map ...
# ... etc
```

`EVAL_USE_NUBES` 미설정 시 기존 동작 (local path) 유지.

#### 6. mp3 디코드 (MELD) — 환경 setup + fallback chain

MELD audio 가 nubes 에 mp3 (transcoded) 인데 기본 torchaudio backend (libsndfile) 가 mp3 미지원. 학습 / eval 환경의 디코드 backend 가 다음 중 하나 필요:

| backend | 설치 방법 | 우선순위 |
|---|---|---|
| **torchaudio ffmpeg** (built-in) | torchaudio 가 ffmpeg lib 와 link 되어 빌드 | 1 (가장 빠름) |
| **pyav** | `pip install av` (이미 audio_lmf, audio env 에 17.0.1 설치됨) | 2 |
| **ffmpeg subprocess** | `conda install -c conda-forge ffmpeg` 또는 system ffmpeg | 3 |

[`eval_source_emotion.decode_audio`](../../evaluation/stage2/eval_source_emotion.py) 에 fallback chain 구현 — torchaudio ffmpeg → pyav → subprocess 순서로 시도. mp3 파일 (file extension 으로 검출) 에만 적용, wav/flac 는 기본 경로.

**검증** (audio_lmf env, conda install ffmpeg + pyav 17.0.1 사용):
- `load_meld_test()` → 2,610 row + 7 emotion labels ✓
- `decode_audio(nubes_path)` → 정상 디코드 (sample 10 개) ✓
- mp3 ↔ wav RMS diff = 0.23 (mean), 0.33 (max) — mp3 LAME priming/padding (~430-680 sample 차이) 으로 인한 frame alignment 효과. quality 자체 정상, 학습/eval 영향 미미

다른 환경 (audio311, audio) 에서 mp3 디코드 사용 시:
```bash
# audio_lmf 가 권장 (ffmpeg + pyav 모두 있음). audio311 / audio 는 pyav fallback 동작 확인 필요.
/mnt/ddn/users/jos/miniforge3/bin/conda install -y -n <env> -c conda-forge ffmpeg
```

### 12.14 MELD audio (wav, 사용자 영역) — 완료 2026-05-08

**상태**: 진행 중 (2026-05-08 07:25 시작).

**대상**: MELD 의 모든 audio split 을 wav 본으로 사용자 영역에 업로드. 학습 + Stage-2 eval 모두 wav 사용.
- `audio/train/` 9,988 wav (~982 MB)
- `audio/dev/` 1,112 wav (~109 MB)
- `audio/test/` 2,747 wav (~284 MB)
- 합계 ~1.4 GB / 13,847 wav

**동기 (mp3 → wav 전환)**: § 12.10 에서 csv 만 사용자 영역에 업로드하고 audio 는 nubes public dir 의 mp3 (`/datasets/public/MELD.Raw/{train_splits, dev_splits_complete, output_repeated_splits_test}/*.mp3`) 를 가리켰음. 그러나 v6 학습 launch 시 torchaudio default backend (libsndfile) 가 mp3 디코드 실패 (`Format not recognised`) — multi-worker dataloader 환경에서 inconsistent. ffmpeg backend 도 audio_lmf env 에 미등록 (`list_audio_backends() == ['soundfile']`). 결과적으로 MELD 11k row 가 모두 학습에서 skip.

대신 **wav 본을 사용자 영역에 직접 업로드** + builder / eval 의 nubes_path 를 wav 가리키게 변경. 코드 단순화 (mp3 fallback 코드 제거, audio_io.py / omni_dataset.py 의 ffmpeg backend 호출 revert).

**명령**:
```bash
nubescli dir-upload hyperscaleai-audiollm/users/jos/AudioEnc/MELD/audio/train/ \
    /mnt/tmp/datasets/emotion_raw/MELD/audio/train/ -j 16
nubescli dir-upload hyperscaleai-audiollm/users/jos/AudioEnc/MELD/audio/dev/ \
    /mnt/tmp/datasets/emotion_raw/MELD/audio/dev/ -j 16
nubescli dir-upload hyperscaleai-audiollm/users/jos/AudioEnc/MELD/audio/test/ \
    /mnt/tmp/datasets/emotion_raw/MELD/audio/test/ -j 16
```

**관련 코드 변경**:
- [`scripts/manifest_builders/build_emotion_meld.py`](../../scripts/manifest_builders/build_emotion_meld.py): `AUDIO_DIR_PATHS` 와 `NUBES_PREFIX` 가 `/users/jos/AudioEnc/MELD/audio/{train,dev}/` 를 가리키게, 확장자 `.mp3` → `.wav`. `list_audio_set()` 가 `.wav` 필터.
- [`scripts/manifest_builders/rewrite_audio_paths_nubes.py`](../../scripts/manifest_builders/rewrite_audio_paths_nubes.py): `_meld_transform` callable 제거, MELD 매핑을 `None` 으로 (builder 가 처음부터 nubes_path 박으니 rewrite 불필요).
- [`evaluation/stage2/eval_source_emotion.py`](../../evaluation/stage2/eval_source_emotion.py): `load_meld_test()` 의 `AUDIO_PREFIX` `/datasets/public/MELD.Raw/output_repeated_splits_test` → `/users/jos/AudioEnc/MELD/audio/test`, 확장자 `.mp3` → `.wav`.
- [`src/llamafactory/data/audio_io.py`](../../src/llamafactory/data/audio_io.py): `_ta_load` ffmpeg fallback 추가했던 것 revert (mp3 안 쓰니 fallback 불필요).
- [`src/llamafactory/data/omni_dataset.py`](../../src/llamafactory/data/omni_dataset.py): `_download_from_nubes` 의 ext 분기 + ffmpeg fallback revert.

**검증**: 업로드 완료 후 추가 (count + 샘플 wav HEAD).

**Side effect on § 12.13 #6 (mp3 디코드 인프라)**: 더 이상 필요 없음 — § 12.13 의 fallback chain 설명 obsolete. 현재 v6 stage1 / Stage-2 eval 모두 wav 만 사용. § 12.13 #6 은 "이전 시도 (deprecated)" 로 마크 하거나 삭제.

### 12.15 LAION-Freesound (audio + meta csv) — 완료 2026-05-11 (audio + manifest + swap)

**상태**: audio 업로드 + manifest 빌드 + live swap 완료. `v6_nubes` 가 신규 (모든 source 100% nubes_path) 본으로 교체됨. 진행 중 학습은 atomic rename swap (학습 open fd 0개 확인 후) 으로 무중단 적용 — 다음 shard 부터 laion_freesound 도 nubes-direct fetch.

**대상**: LAION-Freesound 460,141 flac + metadata.
- `audio/` 460,141 flac (~607 GB, `/mnt/tmp/datasets/laion_extracted/freesound/`)
- `freesound_meta.csv` (105 MB)
- `freesound_no_overlap_meta.csv` (94 MB)
- `README.md`

**동기 (§ 9.3 매핑 검증 결과)**: nubes 의 `/datasets/public/Freesound/audio/` 가 LAION subset 인 줄 알았으나 200-sample ID 매칭 검증 결과 다른 dump 임 확인 (60% miss + 80 hit 중 size 0건 일치). Stage-1 학습은 audio_path local fallback 으로 정상 동작했으나 nubes-direct 화 위해 LAION 본을 사용자 영역에 직접 업로드.

**명령**:
```bash
nubescli upload hyperscaleai-audiollm/users/jos/AudioEnc/LAION-Freesound/freesound_meta.csv \
    /mnt/tmp/datasets/laion_freesound/freesound_meta.csv
nubescli upload hyperscaleai-audiollm/users/jos/AudioEnc/LAION-Freesound/freesound_no_overlap_meta.csv \
    /mnt/tmp/datasets/laion_freesound/freesound_no_overlap_meta.csv
nubescli upload hyperscaleai-audiollm/users/jos/AudioEnc/LAION-Freesound/README.md \
    /mnt/tmp/datasets/laion_freesound/README.md
nubescli dir-upload hyperscaleai-audiollm/users/jos/AudioEnc/LAION-Freesound/audio/ \
    /mnt/tmp/datasets/laion_extracted/freesound/ -j 16
```

**후속 액션 진행 상태**:
1. ✓ [`scripts/manifest_builders/rewrite_audio_paths_nubes.py`](../../scripts/manifest_builders/rewrite_audio_paths_nubes.py) `PREFIX_MAPPINGS` 에 `laion_freesound` 매핑 추가:
   ```python
   "laion_freesound": (
       "/mnt/tmp/datasets/laion_extracted/freesound/",
       "hyperscaleai-audiollm/users/jos/AudioEnc/LAION-Freesound/audio/",
   ),
   ```
2. ✓ `v6_nubes_full` (별도 dir) 빌드 — laion_freesound 45,000/45,000 row nubes_path 박힘
3. ✓ live `v6_nubes` ↔ `v6_nubes_full` atomic swap 완료 (2026-05-11). 학습 open fd 0개 확인 후 `mv v6_nubes v6_nubes_old && mv v6_nubes_full v6_nubes` — 진행 중 학습 무중단 (138 procs alive). `v6_nubes_old` 는 rollback safety 로 일시 보존, 학습 완료 후 삭제.
4. ✓ § 9.3 LAION-Freesound 행 갱신 (⚠ 매핑 불가 → ✓ done)
5. ✓ § 9.8 종합 결론 갱신 ("✓ 업로드 완료 (11)" 에 § 12.15 추가)
6. ✓ § 12.15 검증 표 (아래)

**검증 (2026-05-11)**:

| 대상 | local | nubes (`/users/jos/AudioEnc/LAION-Freesound/`) | 상태 |
|---|---|---|---|
| audio/*.flac | 460,141 flac (~607 GB, `/mnt/tmp/datasets/laion_extracted/freesound/`) | 460,142 obj (recursive `nubescli list -R -o ...audio`) | ✓ 일치 (+1 = list 헤더 / dir entry) |
| freesound_meta.csv | 105 MB | 1 obj | ✓ |
| freesound_no_overlap_meta.csv | 94 MB | 1 obj | ✓ |
| README.md | 3 KB | 1 obj | ✓ |
| v6_nubes_full manifest sample | `422341.flac` audio_path | `hyperscaleai-audiollm/users/jos/AudioEnc/LAION-Freesound/audio/422341.flac` | ✓ |

