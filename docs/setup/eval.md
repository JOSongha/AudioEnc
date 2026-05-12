# Stage-2 / 외부 eval 카탈로그

> [datasets.md](datasets.md) 의 Test 칼럼 = T 인 source 와 매칭되는 eval 스크립트를 정리. eval 코드 위치는 [`evaluation/audio/`](../../evaluation/audio/), 공통 nubes helper 는 [`_nubes_loader.py`](../../evaluation/audio/_nubes_loader.py).

## 1. Test=T source ↔ eval 스크립트

| Modality | Source | Test 라벨 (datasets.md) | eval 스크립트 | 데이터 fetch 경로 | 비고 |
|---|---|---|---|---|---|
| ASR | LibriTTS-R | T (LibriSpeech Test-clean/other) | [`eval_librispeech_wer.py`](../../evaluation/audio/eval_librispeech_wer.py) | nubes audio `/datasets/public/librispeech_asr/{clean,other}/test/*.wav` (2025-12, 외부 팀 본) + transcripts `/users/jos/AudioEnc/LibriSpeech/test_{clean,other}.jsonl` | nubes-only (2026-05-12) |
| ASR | GigaSpeech | T (GigaSpeech test) | [`eval_asr_external.py --datasets gigaspeech`](../../evaluation/audio/eval_asr_external.py) | nubes `/datasets/public/16kHz/gigaspeech/test/<id>.{flac,txt}` (2024-01-04, 외부 팀 본) | `_NUBES_LOADERS` dispatcher 추가 (2026-05-12). HF fallback 은 OOD-tag (mls / voxpopuli / commonvoice) 에 한해 잔존 |
| Sound | AudioCaps | T | [`eval_audiocaps_caption.py`](../../evaluation/audio/eval_audiocaps_caption.py) | nubes `/users/jos/AudioEnc/AudioCaps/data/test-*.parquet` (41 parquet, 4,411 row → 883 unique audio × 5 caps) — audio bytes 가 parquet 안 embedded FLAC | 2026-05-12 신규 작성. Stage-2 LISTEN-mix contamination 무관 |
| Sound | FSD50K | T (eval split) | [`eval_fsd50k_map.py`](../../evaluation/audio/eval_fsd50k_map.py) | nubes `/users/jos/AudioEnc/FSD50K/{FSD50K.eval_audio, FSD50K.ground_truth}/` | nubes-only (local fallback 제거됨, 2026-05-12) |
| Sound | AudioSet | T (eval split) | [`eval_audioset_map.py`](../../evaluation/audio/eval_audioset_map.py) | nubes `/users/jos/AudioEnc/AudioSet/{data/eval/, ontology.json}` | nubes-only |
| Sound | Clotho | T (eval split) | [`eval_clotho_caption.py`](../../evaluation/audio/eval_clotho_caption.py) | nubes `/datasets/public/Clotho-v2/{audio_evaluation/, clotho_captions_evaluation.csv}` | nubes-only, dev/val/eval 모두 지원 |
| Emotion | MELD | T (test split) | [`eval_source_emotion.py --corpora meld`](../../evaluation/audio/eval_source_emotion.py) | nubes `/users/jos/AudioEnc/MELD/{CSV/test_sent_emo.csv, audio/test/}` | nubes-direct (`load_meld_test()` 인라인) |
| Emotion | IEMOCAP | T (Leave-session-out, 4-class) | [`eval_iemocap_session5.py`](../../evaluation/audio/eval_iemocap_session5.py) | nubes `/users/jos/AudioEnc/IEMOCAP/IEMOCAP_full_release/Session5/` | nubes-only. 학습 10-class native, eval 4-class (ang/hap/exc→hap/neu/sad), Session 5 2,170 → xxx 제외 1,650 → 4-class 1,241 utt 채점 |

## 2. 누락 / 미흡 항목 — 2026-05-12 해소

| 항목 | 원 상태 | 처리 |
|---|---|---|
| AudioCaps eval | 부재 | 신규 [`eval_audiocaps_caption.py`](../../evaluation/audio/eval_audiocaps_caption.py) 작성. test 41 parquet 을 nubes 에서 stream + `(youtube_id, start_time)` groupby → 5 caption / unique audio (4,411 → 883). corpus BLEU-1/4 + 옵션 CIDEr / METEOR / ROUGE / SPICE (pycocoevalcap) — `eval_clotho_caption.py` 의 BLEU + scoring 패턴 그대로 |
| LibriSpeech eval HF 의존 | `cache_dir=/mnt/tmp/cache` HF gate | [`eval_librispeech_wer.py`](../../evaluation/audio/eval_librispeech_wer.py) 의 `load_split()` 을 nubes audio + 사용자 영역 jsonl transcript 페어로 재작성 (linter 적용). HF datasets 의존 0 |
| GigaSpeech eval HF 의존 | 동일 | [`eval_asr_external.py`](../../evaluation/audio/eval_asr_external.py) 에 `_NUBES_LOADERS` dispatcher 추가. `--datasets gigaspeech / librispeech_clean / librispeech_other` 호출 시 nubes paired-file (`<id>.flac` + `<id>.txt`) 직접 fetch. mls / voxpopuli / commonvoice 는 그대로 HF (Stage-2 OOD tag 라 nubes 미업로드) |

각 변경은 [_nubes_loader.py](../../evaluation/audio/_nubes_loader.py) 의 `fetch_nubes_audio_tensor` / `fetch_nubes_text` / `list_nubes_dir` 만 사용 — 공통 캐시 (`EVAL_NUBES_CACHE`) 자동 활용.

## 2.5. Trajectory wrapper 의 sub-sampling 상태 (2026-05-12)

[run_all_evals.sh](../../evaluation/audio/run_all_evals.sh) 의 `TASKS` 배열은 trajectory tractability 를 위해 일부 task 에 sub-sampling / drop 적용:

| Task | sample | full set | 비율 | 사유 |
|---|---:|---:|---:|---|
| fsd50k_map_seq | **1,000** | 10,231 | 10 % | sequence-mode 가 200 label × N row 로 forward 폭발, 1,000 만으로 trajectory direction 충분 |
| asr_external (gigaspeech) | **500** (script default) | ~40,000 | 1.3 % | OOD WER trajectory 용 sample 충분 |
| ~~audioset_map_seq~~ | **DROPPED** | — | 0 % | 0.054 sps × 1000 sample × 632 label = ~5 hr/ckpt, FLAC-parquet decode overhead. greedy F1/Jaccard 로 trajectory 대체 |
| 나머지 8 task | full | full | 100 % | librispeech / source_emotion / audiocaps / clotho / audioset_greedy / fsd50k_greedy / iemocap_s5 모두 full |

→ 최종 ckpt 에서 leaderboard-comparable mAP 가 필요하면 audioset_map_seq + fsd50k_map_seq 를 full-set 으로 별도 rerun. RERUN 노트: [eval_trajectory/RERUN_AUDIOSET_SEQ.md](../../../../../mnt/fr20tb/wbl_residency/jos/Qwen3.5_dac_vae_v6_Stage1_jos/Qwen3.5AE-ASR-Stage1-dac-vae-v6/eval_trajectory/RERUN_AUDIOSET_SEQ.md).

### 2.5.1. Full-set rerun 시 단일 task 의 multi-GPU 데이터 병렬 (제안)

audioset_map_seq full (17,141 row × 632 label) 은 1 GPU 에 5+ 시간. trajectory wrapper 는 task-level 풀 (한 task = 한 GPU) 이라 8 GPU 가 있어도 가속 안 됨. 최종 ckpt rerun 에서는 **row-shard 데이터 병렬** 이 자연스러움:

| 옵션 | 구조 | 구현 비용 | 가속 |
|---|---|---|---|
| **A. Row-shard** (권장) | `--row-shard k/N` 인자 추가. 각 GPU 가 `rows[k::N]` 처리 → `predictions_shard_k.jsonl` 출력. 마지막에 merge + sklearn `average_precision_score` 한 번 | 30~50 줄 (eval_audioset_map.py 의 `evaluate_one` 에 slicing + 파일 분리, merge helper) | ~N × (N=8 → 5hr → ~40min) |
| B. Label-shard | label 차원 분할 (각 GPU 가 label subset score). 모델 한 번 로드 / 라벨 forward 만 쪼갬 | 더 복잡 (scoring loop 깊은 곳 수정 + per-sample 결과 모음) | ~N × (compute 분배 동일) |
| C. Row × label 2D | 위 두 개 결합 | 과도 | ~N × |

row-shard 가 깔끔한 이유:
- 행 간 독립 (per-row teacher-forced score 끼리 dependency 0)
- 모델 N copy 메모리 부담 = N × ~10 GB / A100 80 GB → 8 GPU 다 들어감
- 코드 변경 최소: argparse 하나 + row slice 한 줄 + 출력 파일 이름 변경 + 한 줄짜리 merge 스크립트
- 셔딩 단위가 audio 라 audio cache (`EVAL_NUBES_CACHE`) 재활용 가능 (이미 모든 row 캐시됨)

호출 패턴 (gpu 0-7):
```bash
for k in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$k python -m evaluation.audio.eval_audioset_map \
        --ckpt-root $RUN --out-root $OUT_AUDIOSET_FULL \
        --ckpts 100000 --score-mode sequence \
        --row-shard $k/8 --batch-size 8 --label-batch-size 100 &
done
wait
python -m evaluation.audio.eval_audioset_map_merge $OUT_AUDIOSET_FULL/checkpoint-100000
```

label-batch-size 50 → 100/200 추가 bump 으로 단일-GPU throughput 도 2-4x 가능 (A100 80 GB 여유). 그 결과 8 × 2 = ~16x 가속 → 5 hr → ~20 min.

fsd50k_map_seq 도 동일 패턴 적용 가능. 동시에 multiple GPU 차지하니 trajectory wrapper 의 task-pool 과 동시 실행은 안 됨 — final ckpt 의 별도 rerun 으로 사용.

## 3. 실행 환경 변수

| Var | 기본값 | 의미 |
|---|---|---|
| `EVAL_NUBES_CACHE` | `/mnt/tmp/nubes_eval_cache` | nubes 게이트웨이 fetch 결과 로컬 캐시. 동일 ckpt 여러 번 돌릴 때 효과적 |
| `EVAL_MAX_GPUS` | (모든 visible GPU) | [`run_all_evals.sh`](../../evaluation/audio/run_all_evals.sh) 의 GPU 슬롯 풀 크기 제한 (예: 4 → 8개 중 절반만). `CUDA_VISIBLE_DEVICES` 가 우선, 미설정 시 `nvidia-smi` 로 자동 감지 |

## 4. 통합 실행 예시

### 4a. 한 ckpt 의 모든 task 를 GPU 풀에 병렬 dispatch (권장)

```bash
# 13 task × 8 GPU 슬롯 풀 (FIFO semaphore). 긴 task (audioset/fsd50k sequence-mode,
# asr_external, librispeech) 가 먼저 슬롯 차지 → 마지막 wave 균형. nvidia-smi 자동 감지.
bash evaluation/audio/run_all_evals.sh \
    /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17/checkpoint-12000 \
    /mnt/tmp/results/.../evals \
    /mnt/tmp/s2_init_42k
# EVAL_MAX_GPUS=4 로 풀 절반만 쓰기 가능. summary 는 <out-root>/summary.txt.
```

### 4b. 개별 task 직접 호출

```bash
# Stage-2 LoRA ckpt 들에 대해 모든 Test=T eval 한번에 (LISTEN/외부 ASR 별도)
CKPT_ROOT=/mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17
BASE=/mnt/tmp/s2_init_42k
OUT=${CKPT_ROOT}/evals
STEPS=2000,5000,12000

# Sound captioning / multi-label
python -m evaluation.audio.eval_fsd50k_map        --ckpt-root $CKPT_ROOT --out-root $OUT/eval_fsd50k     --base-model $BASE --ckpts $STEPS --batch-size 4
python -m evaluation.audio.eval_audioset_map      --ckpt-root $CKPT_ROOT --out-root $OUT/eval_audioset   --base-model $BASE --ckpts $STEPS --batch-size 8
python -m evaluation.audio.eval_clotho_caption    --ckpt-root $CKPT_ROOT --out-root $OUT/eval_clotho     --base-model $BASE --ckpts $STEPS --batch-size 4
python -m evaluation.audio.eval_audiocaps_caption --ckpt-root $CKPT_ROOT --out-root $OUT/eval_audiocaps  --base-model $BASE --ckpts $STEPS --batch-size 4

# Emotion
python -m evaluation.audio.eval_source_emotion       --ckpt-root $CKPT_ROOT --out-root $OUT/eval_meld     --base-model $BASE --ckpts $STEPS --corpora meld --batch-size 4
python -m evaluation.audio.eval_iemocap_session5     --ckpt-root $CKPT_ROOT --out-root $OUT/eval_iemocap5 --base-model $BASE --ckpts $STEPS --batch-size 4

# ASR (nubes-direct)
python -m evaluation.audio.eval_librispeech_wer  --ckpt-root $CKPT_ROOT --out-root $OUT/eval_libri   --base-model $BASE --ckpts $STEPS --split test.clean --batch-size 4
python -m evaluation.audio.eval_asr_external     --ckpt-root $CKPT_ROOT --out-root $OUT/eval_asr_ood --base-model $BASE --ckpts $STEPS --datasets gigaspeech --max-samples 500 --batch-size 8
```

## 5. Leak audit 요약

datasets.md § 7 의 leak audit 표 그대로 — Test=T source 의 eval split 이 학습 풀과 모두 disjoint:

- LibriSpeech test ↔ LibriTTS-R train.\* (speaker/text 일부 공유, split 자체는 disjoint, datasets.md § 2 LibriTTS-R 행 참고)
- GigaSpeech test ↔ XL train 50% subsample (canonical 분리)
- FSD50K eval ↔ dev (canonical 분리)
- AudioSet eval ↔ bal_train (eval / unbal_train 제외)
- Clotho eval ↔ dev+val (canonical 분리)
- MELD test ↔ train+dev (canonical 분리)
- IEMOCAP Session 5 ↔ Sessions 1-4 (Leave-session-out)
- AudioCaps test ↔ train+val (canonical 분리, builder 의 `SPLIT_PARQUET_COUNT` dict 가 train+validation 만 enumerate — test parquet 호출 시 `ValueError` 로 lock)

Stage-2 LISTEN-mix contamination caveat ([`eval_source_emotion.py:11-18`](../../evaluation/audio/eval_source_emotion.py#L11-L18)): Stage-2 ckpt 평가 시 MELD-test / MOSEI-test 가 LISTEN-train 에 흡수돼 들어가 있을 수 있음 — Stage-1 v6 ckpt 는 영향 없음.

## 변경 이력

- 2026-05-12: 작성. datasets.md Test=T row × evaluation/audio/eval_*.py 매핑 표 + 누락 항목 (AudioCaps eval 부재, LibriSpeech/GigaSpeech nubes 미활용) 정리. § 3 / § 4 / § 5 추가.
- 2026-05-12 (later): § 2 항목 3개 해소. (1) [`eval_audiocaps_caption.py`](../../evaluation/audio/eval_audiocaps_caption.py) 신규 작성 — test 41 parquet stream + caption groupby + BLEU/CIDEr. (2) [`eval_librispeech_wer.py`](../../evaluation/audio/eval_librispeech_wer.py) `load_split()` nubes-only 재작성. (3) [`eval_asr_external.py`](../../evaluation/audio/eval_asr_external.py) 의 `gigaspeech / librispeech_clean / librispeech_other` 태그를 `_NUBES_LOADERS` dispatcher 경유 nubes-direct 로 라우팅. § 1 표 갱신, § 2 결과 표로 교체, § 4 실행 예시에 audiocaps 추가, § 5 AudioCaps leak audit 행 보강.
- 2026-05-12 (run_all_evals 병렬화): [`run_all_evals.sh`](../../evaluation/audio/run_all_evals.sh) 를 FIFO-세마포어 GPU 풀로 재작성. `nvidia-smi` 자동 감지 (또는 `CUDA_VISIBLE_DEVICES`), `EVAL_MAX_GPUS` 로 cap 가능. 13 task (audiocaps 추가) 를 longest-first 로 dispatch — 슬로우 task (audioset/fsd50k sequence-mode, asr_external) 가 먼저 슬롯 차지해 마지막 wave 균형. 8 task × 0.3 s dry-run 으로 1차 wave (gpu 0-7) + 2차 wave (gpu 0-3) dynamic 분배 검증. § 3 `EVAL_MAX_GPUS` 추가, § 4a 권장 호출 예 추가.
