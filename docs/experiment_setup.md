# Experiment Setup

## Overview

Audio Encoder 종류에 따른 ASR 성능 비교 실험.
공통 백본(Qwen3.5-4B Base)과 데이터를 고정하고, audio encoder만 교체하여 비교.

---

## Data

| 데이터셋 | Split | 규모 |
|---|---|---|
| LibriSpeech | train-clean-100 | ~100h |
| LibriSpeech | train-clean-360 | ~360h |
| LibriSpeech | train-other-500 | ~500h |
| MLS English (`parler-tts/mls_eng_10k`) | train (랜덤 샘플링, seed=42) | ~10,016h |
| **학습 합계** | | **~10,976h** |
| LibriSpeech | dev-clean | 검증 |

- 전처리: 16kHz mono, 최대 **20초** truncate
- MLS 소스: `parler-tts/mls_eng_10k` (HuggingFace datasets, decode=False + torchaudio)
- `mls_num_samples = 4,050,000` → 실제 데이터셋 크기 ~2,420k 샘플로 min 제한 (avg ~14.9s)
- 총 학습 샘플 수: ~2,701k / epoch (84416 steps × 32 batch)

---

## Backbone LLM

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3.5-4B` (Base) |
| 파라미터 | **~4.0B** |
| hidden_size | 2,560 |
| 학습 방식 | Stage 1: frozen / Stage 2: LoRA (r=16) |
| LoRA 학습 파라미터 | ~12M (q/k/v/o_proj × 36 layers) |
| dtype | bf16 |
| 프롬프트 포맷 | `"Audio:\n"` + audio embeds + `"\nTranscript:\n"` + transcript |

> Instruct 버전(`Qwen3.5-4B-Instruct`)과의 차이: config에서 `llm_type: "instruct"`로 전환 가능.
> Instruct는 ChatML 포맷(`<|im_start|>system...`) 사용.

---

## Audio Encoders

모든 encoder는 **frozen** (학습 중 가중치 고정).

### Acoustic Encoders

오디오의 저수준 음향 특징을 추출. 음색·음질에 민감.

#### 1. DAC (Descript Audio Codec)

| 항목 | 값 |
|---|---|
| 모델 | `descript/dac-44khz` |
| 전체 파라미터 | ~74M |
| 사용 부분 | encoder only (pre-RVQ) |
| Encoder 파라미터 | **~25M** |
| out_dim | 1,024 |
| 입력 sr | 44kHz (내부 16→44kHz 리샘플) |
| 출력 fps | ~86 fps (hop=512 @ 44kHz) |

#### 2. EnCodec

| 항목 | 값 |
|---|---|
| 모델 | `facebook/encodec-24khz` |
| 전체 파라미터 | ~45M |
| 사용 부분 | encoder only (pre-RVQ) |
| Encoder 파라미터 | **~14M** |
| out_dim | 128 |
| 입력 sr | 24kHz (내부 16→24kHz 리샘플) |
| 출력 fps | 75 fps (hop=320 @ 24kHz) |

---

### Semantic Encoder (구현 예정)

오디오의 언어적·음소적 표현 추출. ASR에 유리.

#### 3. Whisper Tiny

| 항목 | 값 |
|---|---|
| 모델 | `openai/whisper-tiny` |
| 전체 파라미터 | 39M |
| 사용 부분 | encoder only |
| Encoder 파라미터 | **~15M** |
| out_dim | 384 |
| 출력 fps | ~50 fps |

> **현재 미구현.** `encoders/whisper.py` 추가 필요.

---

### Hybrid Encoders

Acoustic encoder(저수준) + Semantic transformer(고수준)를 순차 통과.

#### 4. Mimi — Semantic (acoustic + encoder_transformer)

| 항목 | 값 |
|---|---|
| 모델 | `kyutai/mimi` |
| 사용 부분 | encoder + encoder_transformer |
| 전체 파라미터 | **~87M** |
| out_dim | 512 |
| 입력 sr | 24kHz (내부 16→24kHz 리샘플) |
| 출력 fps | 25 fps (hop=960 @ 24kHz) |

#### 5. Mimi — Acoustic only

| 항목 | 값 |
|---|---|
| 모델 | `kyutai/mimi` |
| 사용 부분 | encoder only (encoder_transformer 제외) |
| Encoder 파라미터 | **~20M** |
| out_dim | 512 |
| 출력 fps | 25 fps |

---

## Projector

각 encoder의 출력을 LLM hidden_size(2560)로 매핑하는 학습 가능 모듈.

```
Conv1d(in_dim → 2560, k=5, s=2) + GELU
Conv1d(2560   → 2560, k=5, s=2) + GELU   ← mimi_semantic은 stride-2 1회만
Conv1d(2560   → 2560, k=1)
LayerNorm(2560)
```

| Encoder | in_dim | Projector 파라미터 | 총 stride | 10초 후 토큰 수 |
|---|---|---|---|---|
| DAC | 1,024 | **~52M** | ×4 | ~215 |
| EnCodec | 128 | **~41M** | ×4 | ~188 |
| Whisper tiny | 384 | **~44M** | ×4 | ~125 |
| Mimi semantic | 512 | **~46M** | ×2 | ~125 |
| Mimi acoustic | 512 | **~46M** | ×4 | ~63 |

---

## 최종 모델 구성 요약

| 실험 | Encoder | Encoder 크기 | Projector | LLM (LoRA) | 총 파라미터 | 학습 파라미터 (Stage 2) |
|---|---|---|---|---|---|---|
| dac | DAC acoustic | ~25M | ~52M | 4.0B | **~4.08B** | ~64M (projector + LoRA) |
| encodec | EnCodec acoustic | ~14M | ~41M | 4.0B | **~4.06B** | ~53M |
| mimi_semantic | Mimi hybrid | ~87M | ~46M | 4.0B | **~4.13B** | ~58M |
| mimi_acoustic | Mimi acoustic | ~20M | ~46M | 4.0B | **~4.07B** | ~58M |
| whisper (예정) | Whisper semantic | ~15M | ~44M | 4.0B | **~4.06B** | ~56M |

> 학습 파라미터: projector 전체 + LoRA (~12M). Encoder와 LLM 나머지는 frozen.

---

## Training Pipeline (2-Stage)

```
Stage 1: Projector Alignment
  - LLM frozen, projector만 학습
  - AdamW, lr=5e-5, 3 epochs, cosine schedule
  - batch_size=4 per GPU × 8 GPU × grad_accum=4 → 실효 배치=128

Stage 2: LoRA Fine-tuning
  - LLM에 LoRA(r=16) 적용, projector도 계속 학습
  - AdamW8bit (bitsandbytes), lr=2e-5, 16 epochs
  - batch_size=2 per GPU × 8 GPU × grad_accum=4 → 실효 배치=64
  - val_loss 기준 best checkpoint 저장
```

---

## Stage 1 조기 종료

학습 중 Stage 1을 예정 epoch 전에 끊고 Stage 2로 넘어가려면:

```bash
kill -USR1 $(cat /mnt/tmp/cache/hf/train.pid)
```

- 시그널을 받으면 **현재 epoch를 완료한 뒤** Stage 2로 진입 (mid-epoch 중단 없음)
- train.pid는 Stage 1 시작 시 rank 0 프로세스가 자동 생성, Stage 1 종료 시 삭제
- Stage 1이 이미 skip된 경우(projector 파일 존재)에는 pid 파일이 생성되지 않음

---

## Checkpoint 경로

기본 `model_cache_dir = /mnt/tmp/cache/hf`

| 단계 | 경로 | 형식 |
|---|---|---|
| Stage 1 skip 판별용 | `{cache_dir}/s1_proj_{enc}.pt` | `.pt` (state_dict) |
| Stage 1 best (기록용) | `{cache_dir}/s1_proj_{enc}_ep{N}_step{N}_best.pt` | `.pt` |
| Stage 1 step 저장 | `{cache_dir}/s1_proj_{enc}_ep{N}_step{N}.pt` | `.pt` |
| Stage 2 best | `{cache_dir}/best_{enc}_ckpt_ep{N}_step{N}/` | safetensors (accelerate) |
| Stage 2 step 저장 | `{cache_dir}/step{N}_{enc}_ckpt/` | safetensors |
| Stage 2 final | `{cache_dir}/final_{enc}_ckpt_ep{N}_step{N}/` | safetensors |

- `ep{N}`: 저장 시점의 epoch 번호, `step{N}`: 저장 시점의 global step
- Stage 1 best는 `s1_proj_{enc}.pt`(고정 이름)와 `_ep{N}_step{N}_best.pt`(기록용) 둘 다 저장됨
  - 고정 이름은 Stage 2 로딩 및 Stage 1 skip 판별에 사용
  - 기록용은 어느 epoch/step에서 best였는지 확인용

**예시 (dac_vae, 2번째 epoch에서 best, step=1200일 때)**
```
/mnt/tmp/cache/hf/
├── s1_proj_dac_vae.pt                          ← Stage 1 skip 판별 / Stage 2 로딩용
├── s1_proj_dac_vae_ep2_step1200_best.pt        ← Stage 1 best 기록용
├── s1_proj_dac_vae_ep1_step600.pt              ← step 저장 예시
├── best_dac_vae_ckpt_ep3_step5000/             ← Stage 2 val_loss 최선 시점
│   └── model.safetensors
├── step4000_dac_vae_ckpt/                      ← Stage 2 step 저장 예시
└── final_dac_vae_ckpt_ep16_step21000/          ← Stage 2 종료 시
    └── model.safetensors
```
