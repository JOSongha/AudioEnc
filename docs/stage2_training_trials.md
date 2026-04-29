# Stage 2 — training trials log

다양한 데이터 mix / sampling 전략으로 돌린 Stage-2 LoRA 학습 trial 기록.
Eval 결과의 정식 기록은 [`stage2_eval_harness.md`](stage2_eval_harness.md)에,
이 문서는 **trial 설계 의도와 결정 근거**, 비교 결과 요약을 담는다.

각 trial은 별도 run dir + run_name 으로 구분된다.

---

## Trial v1 — fixed-mix concat manifest (2026-04-23 ~ 2026-04-25)

### 설정

- **Run name**: `Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17`
- **Output dir**: `/mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17/`
- **Manifest**: `stage2_combined_shards/` (118 588 rows, 단일 셔플)
- **Builder**: [`scripts/emo/build_combined_manifest.py`](../scripts/emo/build_combined_manifest.py)
- **Config**: [`configs/qwen3_5ae-asr/stage2.yaml`](../configs/qwen3_5ae-asr/stage2.yaml)
- **Mix per epoch** (모든 modality 매 epoch 전체 = pool 그대로):

| modality | pool size | rows / epoch | row % | tokens / row (avg) | tokens / epoch | token % |
|---|---:|---:|---:|---:|---:|---:|
| audio_asr | 17 150 | 17 150 | 14.46 % | 458 (audio 400 + text 58) | 7 854 k | 29.6 % |
| audio_emotion | 39 919 | 39 919 | 33.66 % | 137 (audio 77 + text 60) | 5 469 k | 20.6 % |
| audio_env_sound | 41 000 | 41 000 | 34.57 % | 252 (audio 209 + text 42) | 10 332 k | 38.9 % |
| text | 20 519 | 20 519 | 17.30 % | 141 (text only) | 2 893 k | 10.9 % |
| **합계** | **118 588** | **118 588** | 100 % | — | **26 549 k** | 100 % |

  토큰 추정 방식: 텍스트 부분은 S2 tokenizer로 직접 측정, 오디오는
  `t_audio = num_samples / 1920` (48 kHz hop 1920 = 25 fps) 평균 사용
  (asr 50개 / emotion 50개 / env 50개 샘플링).
- **Effective epochs (50k step × eff batch 24 / pool size)**: ~10 epochs/modality

### 의도

LISTEN_full 사용 포기 후 source corpora 직접 학습으로 전환. 소스 코퍼스의
held-out split을 evaluation에 쓰고, train split을 균등 mix. ASR은 Stage 1
14 % 정도만 anchor로 유지, env-sound/text는 새 capability 추가.

### 결과 요약 (ckpt-12000 기준)

| benchmark | metric | value | 평가 |
|---|---|---:|---|
| LibriSpeech test-clean | WER | 5.40 % (4k가 minimum 4.98 %) | S1 7.28 % 대비 −1.88 pp 개선 |
| LibriSpeech test-other | WER | 19.76 % | 2k 18.68 % 대비 +1.08 pp **회귀** |
| MELD test | acc/F1 | 0.497 / 0.230 | F1 단조 상승, acc 평탄 |
| DailyTalk holdout | acc/F1 | 0.833 / 0.376 (11k F1=0.425 peak) | acc 4k 포화, F1 11k 정점 |
| EmoV-DB Jenie | acc/F1 | 0.729 / 0.661 (11k 0.754 peak) | 11k 정점 후 12k 회귀 |
| RAVDESS actors 21-24 | acc/F1 | 0.512 / 0.501 | 거의 monotone |
| ESC-50 5-fold | acc | 79.6 % | 학습 25 %에서 Qwen-Audio 인접 |
| FSD50K eval | F1-mi/ma | 0.398 / 0.270 | 단조 상승 |
| Clotho eval | BLEU-1/4 | 0.459 / 0.081 | DCASE 2020 baseline 통과 |
| Text retention (mean of 6) | acc | 0.906 | 단조 상승 (overfit 부인) |

### v1에서 발견된 문제 / 학습

1. **Text SFT 빠른 메모라이즈**: text loss 7.49 → 0.034 by step 12 000.
   20 k pool × 17 % mix × 50 k step × 24 batch = 모든 row 평균 **10× 반복**.
   6개 좁은 MCQA 코퍼스라 메모라이즈에 충분.
   - 다행히 held-out 정확도는 단조 상승 (overfit이 아닌 in-distribution fit
     로 판명됨).
   - LISTEN 1_text WA는 ckpt-1k → ckpt-2k에서 −0.7 pp 하락 (out-of-distribution).
2. **ASR test-other regression**: Stage 2 학습 데이터가 클린 위주 → noise
   robustness가 학습 후반에 악화 (+1.08 pp).
3. **Late-training 회귀**: DailyTalk / EmoV-DB가 ckpt-11k에서 peak 후
   ckpt-12k에서 −2 ~ −5 pp 하락. ckpt-11000이 cross-task Pareto-optimum.
4. **모든 모달이 동일하게 ~10 epoch** — auxiliary task에 너무 많은 반복.
   variability 부족 → memorization 위험 + diminishing returns.

### Status

- 2026-04-25 ~05:00 UTC: ckpt-25000에서 SIGTERM으로 정지 (요청 따름).
- 모든 ckpt (1k-25k) 보존, eval은 ckpt-13k~25k 일부 추가 완료.

---

## Trial v2 — per-epoch random sub-sampling (2026-04-25 시작 예정)

### 설정

- **Run name**: `Qwen3.5AE-Stage2v2-emoFull-asr033-env05-txt03`
- **Output dir**: `/mnt/tmp/results/Qwen3.5AE-Stage2v2-emoFull-asr033-env05-txt03/` (예정)
- **Manifest**: `stage2_combined_shards_eprandom/` (1 840 380 rows = 20 pseudo-epochs × 92 019 rows)
- **Builder**: [`scripts/emo/build_epoch_random_manifest.py`](../scripts/emo/build_epoch_random_manifest.py)
- **Config**: [`configs/qwen3_5ae-asr/stage2_v2.yaml`](../configs/qwen3_5ae-asr/stage2_v2.yaml)
- **Launcher**: [`configs/qwen3_5ae-asr/run_v2_8gpu.sh`](../configs/qwen3_5ae-asr/run_v2_8gpu.sh) (single-node 8-GPU)
- **Per-epoch fractions** (final values):
  - emotion **1.0** (39 919 / 39 919) — 메인 task, variability 불필요
  - asr **0.33** (13 200 / 40 000) — 풀은 17 k → 40 k로 확장했으나 token 비중
    조절 위해 매 epoch 1/3만 사용
  - env-sound **0.5** (32 744 / 65 488) — AudioSet 추가로 풀 확장 (41 k → 65 k),
    매 epoch 절반만 random subsample
  - text **0.3** (6 156 / 20 519) — v1 메모라이즈 가장 심했음. 가장 강한 cut

- **Mix per pseudo-epoch**:

| modality | pool size (Δ vs v1) | rows / epoch | frac | row % (Δ vs v1) | tokens / row | tokens / epoch | token % (Δ vs v1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| audio_asr | **40 000** (+22 850) | 13 200 | **0.33** | 14.34 % (−0.1) | 458 | 6 046 k | 29.3 % (−0.3) |
| audio_emotion | 39 919 | 39 919 | 1.0 | 43.38 % (+9.7) | 137 | 5 469 k | 26.5 % (+5.9) |
| audio_env_sound | **65 488** (+24 488) | **32 744** | **0.5** | 35.58 % (+1.0) | 252 | 8 251 k | **40.0 % (+1.1)** |
| text | 20 519 | **6 156** | **0.3** | **6.69 % (−10.6)** | 141 | **868 k** | **4.2 % (−6.7)** |
| **합계** | **165 926** (+47 338) | **92 019** | — | 100 % | — | **20 634 k** | 100 % |

**Pool 변경점 (v1 → v2)**:
- ASR 17 150 → **40 000** (Bernoulli 재샘플; 21 M superset에서 다시 추출).
  pool 자체는 키웠지만 per-epoch 사용은 **frac=0.33으로 1/3만** — token 점유율
  v1과 비슷하게 유지하면서 rows의 다양성 확보.
- env-sound 41 000 → **65 488** (FSD50K 40 966 + Clotho 3 839 + ESC-50 2 000 +
  **AudioSet bal_train 18 683** 추가). AudioSet은 HuggingFace `agkphysics/AudioSet`
  bal_train 22 k 중 human_labels 비어있는 행 제외 후 18 683 row.
- emotion 39 919 / text 20 519: 풀 크기 변경 없음.

**Per-epoch rows 변경점**:
- asr 17 150 → 13 200 (풀은 2.3×지만 frac으로 1/3만 사용)
- env 41 000 → 32 744 (풀 1.6×지만 frac 0.5로 −20 %)
- text 20 519 → 6 156 (frac 0.3, −70 %)
- emotion 동일

**Token mix 시사점 (v1 vs 현재 v2)**:
- ASR token % v1 32 % → **v2 29 %** (풀 확장이 frac 1/3으로 거의 상쇄됨)
- env token % v1 39 % → **v2 40 %** (env 비중 거의 동일)
- emotion token % v1 21 % → **v2 27 %** (+5.9 pp, 메인 task 강화)
- text token % v1 11 % → **v2 4 %** (가장 큰 cut, overfit 압력 약화 의도)

  Token 수는 v1과 같은 추정 방식. v1 → v2 변화량이 row %보다 token %가
  낮은 이유 = text가 짧기 때문 (avg 141 토큰 vs 다른 모달 137-458).
  **Text 절대량은 2.89 M → 0.87 M (−70 %)** — row 60 % cut 보다 token cut이
  약간 더 큼.

- **Effective row repetition over 50k×24 = 1.2M rows seen**:
  - emotion ~12× (v1: 10×)
  - asr ~12× (v1: 10×)
  - env ~12× (v1: 10×, similar — but each pseudo-epoch the row order is
    different due to per-epoch independent random shuffle, breaking the
    deterministic ordering of v1's once-shuffled manifest)
  - text ~4× (v1: 10×, **−60 %**)

### 의도

1. **Text overfit 약화** — 매 pseudo-epoch마다 6 k의 다른 random subset → 같은
   row의 반복 빈도 60 % 감소. v1처럼 train loss → 0.03 수준 메모라이즈는
   덜 일어날 것으로 기대.
2. **Env-sound 확장 + 다양성** — pool 41 k → **65 k** (AudioSet bal_train 18.7 k
   추가) 하면서 매 epoch 절반만 random sample. ESC-50가 v1 ckpt-12 k 79.6 %
   에서 plateau 진입 — AudioSet의 새 acoustic 분포가 plateau 깰 가능성.
3. **ASR pool 확장 + token 비중 조절** — pool 17 k → **40 k** (5-corpus
   superset에서 다시 추출), 단 매 epoch는 0.33만 사용해서 token 비중을 v1과
   비슷한 29 % 수준 유지. row variety 2.3 ×.
4. **Emotion 비중 강화** — pool 변화 없지만 다른 task가 cut되면서 자연스럽게
   row mix 33.7 → 43.4 %, token mix 21 → 27 %. v1 후반 emotion regression
   (DailyTalk/EmoV ckpt-12 k 회귀)의 한 원인이 비중 부족이었을 수 있음.

### Fraction 결정 이력 (사용자 요청 반영)

| 회차 | env-frac | asr-frac | text-frac | 사용자 요청 / 근거 |
|---|---:|---:|---:|---|
| 1차 | 0.5 | 1.0 | 0.3 | 초기 설계 |
| 2차 | 0.7 | 1.0 | 0.3 | "env 좀 더 늘려" |
| 3차 | 1.0 | 1.0 | 0.3 | "env-sound rows/epoch을 emotion과 비슷하게" |
| 4차 | 1.0 | 1.0 | 0.3 | (ASR 풀 17 k → 40 k 확장 + AudioSet 18.7 k 추가) |
| **최종** | **0.5** | **0.33** | **0.3** | "ASR을 1/3으로 줄여도" — 토큰 비중 56 % → 29 %로 균형 회복; env는 풀 65 k에서 절반만 |

**최종 비율 근거** (final):
- **emotion = 1.0**: 메인 task. 변경 없음.
- **asr = 0.33**: pool 40 k이지만 매 epoch 13 200 row만. ASR transcript가
  길어서 (per-row 458 토큰) 풀 확장 + 1.0 적용 시 token 비중 56 %까지 치솟음.
  0.33 적용 시 v1과 비슷한 ~29 %로 회복.
- **env = 0.5**: pool 65 k의 절반 = 32 744 row/epoch. AudioSet 18.7 k 추가로
  새 acoustic 분포 학습 + per-epoch 다른 32 k random sample → variability.
- **text = 0.3**: v1에서 overfit 가장 심했음. 풀 20 k에서 6 156만 → 매
  pseudo-epoch 다른 6 k 봄. row repetition 10 × → 4 ×.

이 fraction은 **추가 trial로 검증/조정 가능**한 hyperparameter. v3에서 다른
조합 (예: text=0.5, env=0.7) 시도 가능.

### 추가 변경 — ASR-only noise augmentation (2026-04-25)

v1: encoder 내부에서 모든 audio row에 latent-space Gaussian noise 적용
(`x̃ = √k·ε + √(1-k)·x`, k ~ U(0, 0.1)).

**v2: ASR row(modality_id=0)에만 노이즈 적용.** 비-ASR (emotion / env-sound)
는 prosodic / spectral 디테일이 중요해서 latent noise가 오히려 학습 신호를
망칠 수 있다는 가설.

**구현** (4 파일 수정):
- `audio_encoder.py` — `forward(audio_values, modality_ids=None)` 시그니처
  추가. `noise_aug_asr_only` config flag와 환경변수
  `AUDIO_NOISE_AUG_ASR_ONLY` 모두 지원. flag=True + modality_ids 제공
  시에만 ASR-gated, 그 외엔 legacy 전체 적용.
- `modeling_qwen3_5AE.py` — outer/inner forward에 `audio_modality_ids`
  threading. `prepare_inputs_for_generation`도 같이.
- `omni_dataset.py` — `process_samples`에서 audio별 modality id 수집
  (`all_audio_modality_ids`). packer에도 동일 propagation.
- `collator.py` (omni_dataset 내 collator class) — flatten해서
  `audio_modality_ids: torch.LongTensor[N_audio]` 형태로 batch에 추가.
- `config.json` — `audio_config.noise_aug_asr_only: true` 추가.

**옵션 토글**:
- `config.audio_config.noise_aug_asr_only`:
  - `true` (default for v2) → ASR-only
  - `false` → 전체 적용 (v1 legacy)
- env `AUDIO_NOISE_AUG_ASR_ONLY=0/1` (config 우회)
- env `AUDIO_NOISE_AUG=0/1` (master toggle, 노이즈 자체 on/off)

이 옵션으로 ASR-only vs 전체 비교 실험 가능 — 다음 trial에서
ablation 후보.

### 변경하지 않은 항목

v1과 동일:
- model_name_or_path = `/mnt/tmp/s2_init_42k`
- LoRA: rank 16, alpha 32, dropout 0.05, target q/k/v/o_proj
- additional_target = `audio_encoder.projector` (full-trainable)
- LR 1e-5, warmup 1000, weight_decay 0.01
- max_steps 50000
- per-device batch 3, grad accum 1, eff batch 24 (8 GPU)
- packing + neat_packing
- max_audio_samples 1.6M (33 s @ 48 kHz)

### 미정 / 시작 시점

- **Auto-launch watchdog (`b3hev4hx6`)** 작동 중: 60s polling, 모든 eval 프로세스
  종료 시 `run_v2_8gpu.sh` 실행
- 시작 후 **약 22 시간** 예상 소요 (50 k step, v1과 동일 step rate)

### 평가 계획

- 동일 evaluation harness ([`stage2_eval_harness.md`](stage2_eval_harness.md) §2)
- 같은 sparse trajectory: 1k, 2k, 5k, 10k, 12k, 15k, 18k, 22k, 25k, 30k, 40k, 50k
  (per-task)
- v1 vs v2 직접 비교용 핵심 지표:
  - **Text retention 6-bench mean**: v2가 v1보다 더 부드러운 단조 상승해야
    (text overfit 완화 검증)
  - **LISTEN 1_text WA** (held-out emotion-from-transcript): v2에서 음의 trend
    나오는지 — 나오면 v2가 의도대로 작동
  - **EmoV-DB Jenie peak ckpt**: v1은 11k peak. v2에서 peak 위치 늦춰지거나
    plateau 확장되면 emotion-strengthen 의도 검증
  - **Env-sound (ESC-50, FSD50K)**: v2가 v1과 동등 이상이어야 — env 비중 줄였지만
    variability 더 줘서 generalization 약하지 않음을 입증

### 비교 결과 (학습 진행 후 추가 예정)

| benchmark / metric | v1 ckpt-12k | v1 ckpt-25k | **v2 ckpt-12k** | **v2 ckpt-25k** | 비교 |
|---|---:|---:|---:|---:|---|
| ASR test-clean WER | 5.40 % | TBD | TBD | TBD | |
| ASR test-other WER | 19.76 % | TBD | TBD | TBD | |
| MELD F1 | 0.230 | TBD | TBD | TBD | |
| DailyTalk F1 | 0.376 | TBD | TBD | TBD | |
| EmoV-DB acc | 0.729 | TBD | TBD | TBD | |
| RAVDESS acc | 0.512 | TBD | TBD | TBD | |
| ESC-50 acc | 0.796 | TBD | TBD | TBD | |
| Clotho BLEU-4 | 0.081 | TBD | TBD | TBD | |
| FSD50K F1-micro | 0.398 | TBD | TBD | TBD | |
| Text retention mean | 0.906 | TBD | TBD | TBD | |

(v1 ckpt-25k까지의 evaluation은 별도 후속 sweep으로 진행 중 — `stage2_eval_harness.md`
에 추가 trajectory 반영 예정.)

---

## Trial v3-whisper — encoder swap (small + tiny, 2026-04-27 시작 예정)

### 동기

v1/v2 모두 **DAC-VAE 48kHz 인코더**를 audio backbone으로 사용. v2 결과
(11/n: 모든 task에서 v1 대비 우세 — `stage2_eval_harness.md` §14)는 *데이터
mix*가 v1→v2의 단일 변경이었으므로 *audio backbone* 자체는 비교 대상에서
빠져 있음. **인코더를 바꾸면 어디까지 오를 수 있는가**가 본 trial의 질문.

후보:
- **Whisper-small.en** (768 hidden, ~88 M params encoder + projector)
- **Whisper-tiny.en** (384 hidden, ~39 M params encoder + projector)

각각 audio_encoder 로 사용하는 Stage-2 SFT를 v2와 동일 mix·동일 hyperparam
으로 돌림 → encoder-controlled 비교.

### Stage-1 출발점 (sehyun 빌드)

| variant | Stage-1 final ckpt | step | loss | LibriSpeech test-clean WER |
|---|---|---:|---:|---:|
| Whisper-small | `external/ckpts/Qwen3.5_whisper_small_Stage1/.../checkpoint-13000` | 13_000 | 0.146 | **2.58 %** |
| Whisper-tiny  | `external/ckpts/Qwen3.5_whisper_tiny_Stage1/.../checkpoint-13000`  | 13_000 | 0.185 | **3.57 %** |
| (DAC, 비교용) `s2_init_42k` baseline | step 42 000 | 42_000 | — | 7.28 % |

→ Whisper 변형이 Stage-1 시점에서 이미 **DAC 대비 2-3× 더 낮은 WER**.
audio understanding의 모든 high-level task가 Stage-2에서 어떻게 따라오는지가
관심 포인트.

### 설정

- **Run names**:
  - small: `Qwen3.5AE-Stage2-whisper-small-emoFull-asr033-env05-txt03`
  - tiny:  `Qwen3.5AE-Stage2-whisper-tiny-emoFull-asr033-env05-txt03`
- **Configs**:
  - [`configs/qwen3_5ae-asr/stage2_whisper_small.yaml`](../configs/qwen3_5ae-asr/stage2_whisper_small.yaml)
  - [`configs/qwen3_5ae-asr/stage2_whisper_tiny.yaml`](../configs/qwen3_5ae-asr/stage2_whisper_tiny.yaml)
- **Launch scripts**:
  - [`run_whisper_small_8gpu.sh`](../configs/qwen3_5ae-asr/run_whisper_small_8gpu.sh)
  - [`run_whisper_tiny_8gpu.sh`](../configs/qwen3_5ae-asr/run_whisper_tiny_8gpu.sh)
- **Manifest**: v2 그대로 (`stage2_combined_shards_eprandom`, 1.84 M rows, 20 epochs, emo 43% / asr 14% / env 36% / text 7%).
- **Mix 비율**: v2와 동일 (`emoFull / asr033 / env05 / txt03` per-epoch fractions).

### v2 (DAC) 와의 차이 — yaml 레벨

순수 학습 hyperparam은 100 % 동일. 차이는 인코더 결정의 직접 결과만:

| 필드 | v2 (DAC) | whisper-small | whisper-tiny |
|---|---|---|---|
| `model_name_or_path` | `/mnt/tmp/s2_init_42k` | Whisper-small Stage-1 final | Whisper-tiny Stage-1 final |
| `omni_max_audio_samples` | 1_600_000 (33s @ 48kHz) | 480_000 (30s @ 16kHz) | 480_000 |
| `run_name` | `Stage2v2-emoFull-...` | `Stage2-whisper-small-...` | `Stage2-whisper-tiny-...` |

나머지 동일: LoRA `rank=16, alpha=32, dropout=0.05, target=q/k/v/o_proj`,
`additional_target=audio_encoder.projector` (projector full-trainable),
deepspeed `ds_z2_no_fused.json`, fa2 + liger + bf16, lr 1e-5, batch 3,
gradient_accumulation 1, max_steps 50_000, warmup 1000, save_steps 1000,
manifest 동일.

### 인프라 — 자동 라우팅

추가 코드 변경 불필요. 기존 dispatcher가 model `audio_config` 의
`whisper_model_id` 필드를 읽어 자동 분기:

| 컴포넌트 | DAC 경로 | Whisper 경로 | 분기 위치 |
|---|---|---|---|
| Per-row processor | `create_omni_processor` | `create_omni_processor_whisper` | [`data/loader.py:638`](../src/llamafactory/data/loader.py#L638) |
| Collator | `OmniCollator` (waveform pad+stack) | `WhisperOmniCollator` (mel `[N,80,3000]` stack) | [`train/omni/workflow.py:79`](../src/llamafactory/train/omni/workflow.py#L79) |
| Audio I/O | DAC packed-token preload | `audio_io.load_audio_chunk` (auto-resample 16kHz) | [`data/audio_io.py`](../src/llamafactory/data/audio_io.py) |
| Feature extraction | (DAC token sequence) | `whisper_features.extract_mel` ([80, 3000] log-mel via `WhisperFeatureExtractor`) | [`data/whisper_features.py`](../src/llamafactory/data/whisper_features.py) |

⇒ Whisper용 manifest 별도 빌드 불필요. v2 manifest 의 `path` 필드(어떤 sr이든) 가 16kHz 로 자동 resample → mono → log-mel 변환되어 collator 에 [N, 80, 3000] 로 stack.

### 운영 plan

1. **Phase 0 (완료)**: 인프라 검증 — auto-routing, audio_io resample, mel
   shape contract 모두 확인.
2. **Phase 1 (완료)**: yaml 4개 (config × 2 + launch × 2) 작성.
3. **Phase 3**: smoke test — small variant 200-step 짧은 학습으로 forward
   / backward / save / load 검증.
4. **Phase 4**: 8-GPU 풀런. **small 먼저 단독** → 끝나면 tiny 후속 launch.
   동시 실행 안 함 (8-GPU 리소스 한 번에 한 trial 만).
5. **Phase 5**: 학습 종료(또는 50k 도달) 후 v1/v2 와 동일한 25-task eval
   harness 적용. 비교 표는 `stage2_eval_harness.md` §15 (예정) 에 기재.

### 예상 timeline

- v2 reference: 31_283 step 까지 7.5h × 8-GPU = ~31h 추정 (하지만 실측은
  user-SIGTERM 으로 중단 — 50k 풀 추정 ~38h).
- Whisper-small: encoder forward 약간 무거움(88M vs DAC 비교 미상) →
  ~10-15 % overhead 가정 → **50k 약 42-45h**.
- Whisper-tiny: encoder 작아서 v2 와 비슷 또는 살짝 빠름 → **50k 약
  35-40h**.
- 합계 small + tiny 시퀀셜: 약 **3.5-4 일** (smoke + 본런).

### 비교 가설

1. **ASR**: Whisper-small > Whisper-tiny > DAC. Stage-1 WER (2.58 / 3.57 /
   ~4.98 v1 best @ ckpt-4k) 와 같은 순서로 Stage-2 후 ASR 도 정렬될 가능성.
2. **Sound classification (ESC-50, FSD50K)**: Whisper 가 음성 특화이므로
   non-speech 에서 DAC 보다 약할 수 있음. v2 ESC-50 99.1 % 를 따라잡을 수
   있는지가 큰 관전 포인트.
3. **Emotion**: Whisper 가 음성 prosody를 보존하므로 DAC 보다 우세 가능. 단
   Whisper.en 은 영어 전용 음성 모델이라 prosody 표현 폭이 좁을 수 있음.
4. **Captioning (Clotho)**: Whisper 의 음성 편향 때문에 환경음 captioning
   에서 DAC 보다 떨어질 가능성.
5. **Tiny vs small**: tiny 가 small 의 70-90 % 정도 성능에 capacity gap
   확인.

### 결과 (TBD — 학습 종료 후 채움)

| metric | v2 (DAC) ckpt-21k | whisper-small | whisper-tiny |
|---|---:|---:|---:|
| WER clean      | 0.0520 | TBD | TBD |
| WER other      | 0.1843 | TBD | TBD |
| FSD50K mAP-μ   | 0.3033 | TBD | TBD |
| ESC-50 acc     | 0.9600 | TBD | TBD |
| Clotho BLEU-4  | 0.0920 | TBD | TBD |
| Clotho CIDEr   | 0.1465 | TBD | TBD |
| Text retention | 0.9061 | TBD | TBD |
| LISTEN MCQA acc| 0.2641 | TBD | TBD |
| MELD F1        | 0.2775 | TBD | TBD |
| DailyTalk F1   | 0.3929 | TBD | TBD |
| EmoV acc       | 0.7721 | TBD | TBD |
| RAVDESS acc    | 0.5583 | TBD | TBD |

(v2 column 은 cross-task Pareto-best ckpt-21k 의 값. Whisper run 의
best-ckpt 도 동일 방식으로 산출 후 채움.)

---

## Possible v3+ candidates (future trials, not yet committed)

후속 trial 후보 — v2 결과 본 후 결정:

1. **v3a: text-pool expansion (Open-Orca / OpenHermes / Magpie 추가)**
   - text 풀을 6 small benchmarks (20 k)에서 ~500 k로 확장
   - text=1.0이어도 row repetition 1× 미만 → memorize 불가
   - 단점: storage / pre-tokenize cost 증가
   - v2가 text overfit 거의 해결하면 불필요

2. **v3b: text=0.1 더 강하게 cut**
   - v2 text=0.3에서도 부족하면
   - per-pseudo-epoch text exposure ≈ 2 k → repetition 약 1.2×
   - 단점: text task 학습이 너무 가벼워질 수 있음

3. **v3c: env=1.0 + ESC-50 oversample**
   - ESC-50가 FSD50K 대비 풀이 작음 (2 k vs 41 k)
   - env-sound 내부 sub-mix를 조정해서 ESC-50을 더 자주 보도록
   - 별도 builder 수정 필요

4. ~~**v3d: noise-aug ASR**~~ — **v2에 이미 포함됨** (latent-space Gaussian
   noise를 ASR row에만 적용하도록 toggling). test-other regression이 v2에서
   해결되는지 결과 보면서 SpecAugment / waveform-additive-noise는 v3+
   후속 옵션으로 보류.

5. **v3e: rationale synthesis**
   - emotion 데이터에 GPT-4 / Llama-3-70B로 rationale 생성
   - 현재 letter-only target → letter+rationale (20-50 token)
   - [`stage2_design.md`](stage2_design.md) §3.1, §6.2 참조
   - 가장 큰 architectural change지만 macro-F1 가장 크게 끌어올릴 가능성

6. **v3f: emotion mix를 더 키움**
   - v2가 emotion 43% — 더 키워서 50%+ 시도
   - v1 ckpt-11k peak가 v2에서 사라지지 않으면 시도

---

## Trial 운영 메모

- **각 trial은 별도 output dir + run_name** — 결과 비교 용이
- v1 ckpt 모두 보존 (size 큰 deepspeed step 폴더 포함). v3 시작 전 archive 검토
- evaluation harness는 trial-agnostic — `--ckpt-root` 만 바꾸면 동일 protocol
- 각 trial 시작·종료 시점, 파라미터 변경, 발견된 이슈는 이 문서에 누적
