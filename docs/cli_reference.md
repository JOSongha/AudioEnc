# CLI 레퍼런스 — train_pipeline_override.py / run.sh

작성: 2026-04-08

---

## 사용법

```bash
bash run.sh --encoder <encoder> [옵션...]
```

`run.sh`는 `accelerate launch train_pipeline_override.py`의 래퍼.  
`--gpus`와 `--encoder`는 run.sh가 처리하고, 나머지는 train_pipeline_override.py로 전달됨.

---

## 인자 목록

### 필수

| 인자 | 설명 |
|---|---|
| `--encoder` | 사용할 오디오 인코더. `encodec` \| `dac` \| `fb_dacvae` \| `mimi_acoustic` \| `mimi_semantic` |

---

### 모델

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--llm` | config의 `llm_model` (`Qwen/Qwen3.5-2B`) | LLM 모델명. 예: `Qwen/Qwen3.5-4B` |
| `--cache-dir` | config의 `model_cache_dir` (`/mnt/tmp/cache/hf`) | HuggingFace 모델 캐시 경로 |

---

### GPU / 분산

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--gpus` | `8` | GPU 수 (run.sh 전용, `accelerate --num_processes`로 전달됨) |

---

### 데이터셋

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--datasets` | `ls100,ls360,ls500,mls,gs,vp` | 사용할 데이터셋 (쉼표 구분). 가능한 값: `ls100`, `ls360`, `ls500`, `mls`, `gs`, `vp` |
| `--estimated-hours` | `--datasets` 기반 자동 계산 | 데이터셋 총 시간(h) 수동 지정. `max_steps` 계산에 사용 |
| `--word-aug` | `False` | Word-level ASR augmentation 활성화. 사전 생성된 word alignment Arrow 필요 |

---

### 학습 스케줄

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--stage` | `all` | 실행할 스테이지. `all` (1→2 순서) \| `1` \| `2` |
| `--stage1-epochs` | config의 `stage1_epochs` (`1`) | Stage 1 epoch 수 |
| `--stage2-epochs` | config의 `stage2_epochs` (`2`) | Stage 2 epoch 수. encoder별 오버라이드 있음 (dac, mimi_semantic은 8) |
| `--max-steps` | `estimated_hours` 기반 자동 계산 | 학습 최대 step 수. 지정 시 epoch 계산 무시 |
| `--eval-steps` | config의 `eval_steps` (`500`) | WER + val_loss 평가 주기 (steps) |
| `--save-steps` | config의 `save_steps` (`5000`) | 체크포인트 저장 주기 (steps) |
| `--resume` | None | 재개할 체크포인트 경로. `"latest"` 지정 시 최근 체크포인트 자동 탐색 |

---

### 시퀀스 패킹

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--cutoff-len` | config의 `packing_cutoff_len` (`16384`) | 하나의 packed bin 최대 토큰 수. cutoff_len을 초과하는 샘플은 드롭됨 (절단 없음). 클수록 GPU utilization↑, 메모리↑ |

> **중요**: `cutoff_len=10`이고 bin에 8 토큰이 차 있을 때 다음 아이템이 4 토큰이면, 그 아이템은 현재 bin에 들어가지 않고 다음 bin으로 넘어감. 아이템은 절대 mid-truncate 되지 않는다.

**`packing_bucket_size` vs `cutoff_len` 차이**

| | `cutoff_len` (= `packing_cutoff_len`) | `packing_bucket_size` |
|---|---|---|
| 역할 | bin 하나의 크기 | greedy knapsack의 입력 풀 크기 |
| 기본값 | `16384` | `200` (config의 `packing_bucket_size`) |
| 영향 | 모델이 받는 시퀀스 길이, GPU util | 패킹 품질 (클수록 padding↓, fill ratio↑) |

흐름: `packing_bucket_size=200`개 샘플 → greedy knapsack → `cutoff_len=16384`짜리 bin들 생성.
`packing_bucket_size`가 클수록 knapsack이 더 좋은 조합을 찾아 padding이 줄지만, 처리 단위가 커진다.

---

### 최적화

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--attn-impl` | `flash_attention_2` | Attention 구현체. `eager` \| `sdpa` \| `flash_attention_2` |
| `--liger` / `--no-liger` | `--liger` (ON) | Liger fused kernel (RoPE, RMSNorm, SwiGLU, CE loss). 비활성화: `--no-liger` |
| `--fsdp` / `--no-fsdp` | `--fsdp` (ON) | Stage 2 FSDP 활성화 (full_shard, encoder excluded). 비활성화: `--no-fsdp` |
| `--fsdp-stage1` / `--no-fsdp-stage1` | `--no-fsdp-stage1` (OFF) | Stage 1 FSDP 활성화. ON 시 LLM 메모리 1/8 절감 (GPU당 ~4.6GB), forward all-gather 오버헤드 추가 |

---

### 실험 추적

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--wandb-mode` | `online` | WandB 모드. `online` \| `offline` \| `disabled` |

---

## 예시

```bash
# 전체 데이터셋, 모든 최적화 ON (기본)
bash run.sh --encoder fb_dacvae

# word-aug 포함 전체 학습
bash run.sh --encoder fb_dacvae --word-aug

# 기능 테스트 (2 steps, WandB 비활성화)
bash run.sh --encoder fb_dacvae \
    --wandb-mode disabled --max-steps 2 --datasets ls100 --stage all

# SDPA + FSDP 없음 (디버그)
bash run.sh --encoder fb_dacvae \
    --attn-impl sdpa --no-fsdp --no-liger \
    --wandb-mode disabled --max-steps 2 --datasets ls100

# Stage 1만, cutoff_len 1024, 4 GPU
bash run.sh --encoder encodec --gpus 4 --stage 1 --cutoff-len 1024

# Stage 2 재개
bash run.sh --encoder fb_dacvae --stage 2 \
    --resume /mnt/tmp/cache/hf/fb_dacvae/s1_outputs_0408_1200
```

---

## 인코더 설정 — hop, fps, samples_per_token

`ENCODER_REGISTRY`(config.py)에 인코더별로 `hop`, `tgt_sr`, `proj_strides`가 정의된다.  
이 세 값이 모델이 받는 토큰 수와 학습 속도를 결정한다.

### 용어 정의

| 필드 | 의미 |
|---|---|
| `tgt_sr` | 인코더가 기대하는 샘플레이트 (Hz). 16kHz 입력은 내부에서 리샘플링됨 |
| `hop` | `tgt_sr` 기준 인코더 hop 크기 (samples). `tgt_sr / hop` = encoder fps |
| `proj_strides` | projector Conv1d stride 목록. `[2, 2]` → 총 ×4 다운샘플 |
| `fps` (encoder) | `tgt_sr / hop`. 인코더 출력 프레임레이트 |
| `fps` (projector) | `fps_enc / prod(proj_strides)`. LLM이 받는 오디오 토큰 프레임레이트 |
| `samples_per_token` | 16kHz 샘플 몇 개가 LLM 토큰 1개에 대응하는지. `hop × (16000 / tgt_sr) × prod(proj_strides)` |
| `tokens_per_10s` | 10초 오디오 → LLM 토큰 수. `16000 × 10 / samples_per_token` |

### 인코더별 수치

| encoder | `tgt_sr` | `hop` | `fps_enc` | `proj_strides` | `fps_proj` | `samples_per_token` | 10초 토큰 |
|---|---|---|---|---|---|---|---|
| `encodec` | 24000 | 320 | 75 | [2, 2] | 18.75 | ~683 | ~234 |
| `dac` | 44100 | 512 | ~86 | [2, 2] | ~21.5 | ~743 | ~215 |
| `fb_dacvae` | 44100 | 512 | ~86 | [2, 2] | ~21.5 | ~743 | ~215 |
| `mimi_acoustic` | 24000 | 960 | 25 | [2, 2] | 6.25 | ~2048 | ~78 |
| `mimi_semantic` | 24000 | 960 | 25 | [2] | 12.5 | ~1024 | ~156 |

> `samples_per_token` = `hop × (16000 / tgt_sr) × prod(proj_strides)`  
> encodec 예시: `320 × (16000/24000) × 4 = 853.3` → 10초: `160000 / 853.3 ≈ 188`  
> *(표 수치는 부동소수 반올림으로 README 표와 약간 차이 날 수 있음)*

### packing과의 관계

`cutoff_len=16384`일 때 bin 하나에 들어가는 **오디오 토큰 수**는 `tokens_per_10s`에 비례.  
mimi_acoustic(~78 tokens/10s)은 bin 하나에 더 많은 클립이 들어가고,  
dac/fb_dacvae(~215 tokens/10s)는 상대적으로 적은 클립이 들어간다.

> 실측 (fb_dacvae, 8×A100-80GB): cutoff 2048→13GB/24%util, 8192→35GB/82%util, 16384→64GB/~95%util (~55s/step, ~44h wall).

---

## 설정 우선순위

```
CLI 인자 > config.py의 TRAIN_CONFIG > 함수 기본값 (cfg.get(..., fallback))
```

`--cutoff-len`, `--eval-steps`, `--save-steps`는 지정하지 않으면 config.py 값을 사용.  
`--attn-impl`은 지정하지 않으면 config.py의 `attn_implementation`을 사용.

---

## 체크포인트 경로

```
/mnt/tmp/cache/hf/{encoder}/
  s1_outputs_{run_id}/
    best_s1_proj.pt        ← WER 기준 best projector (Stage 2 로딩용)
    s1_proj.pt             ← Stage 1 완료 후 최종 projector
    s1_proj_step{N}.pt     ← eval_steps마다 저장된 intermediate projector
    checkpoint-{N}/        ← Trainer step 체크포인트
  s2_outputs_{run_id}/
    checkpoint-{N}/        ← FSDP sharded 체크포인트 (save_total_limit=3)
```

`run_id` = `MMDD_HHMM` (실행 시작 시각, 예: `0408_1430`)

Stage 1 조기 종료: `kill -USR1 $(cat /mnt/tmp/cache/train.pid)`
