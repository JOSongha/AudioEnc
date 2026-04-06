# Sequence Packing + Flash Attention 2 + Liger Kernel + FSDP

## 개요

학습 속도·메모리 효율화를 위한 4가지 최적화를 opt-in 플래그로 추가.  
기존 `train.py` 플래그 없이 실행 시 동작은 완전히 보존됨.

---

## 설치 요구사항

| 패키지 | 버전 | 용도 | 상태 |
|--------|------|------|------|
| `torch` | 2.8.0 | 기본 | 설치됨 |
| `transformers` | 5.4.0 | LLM | 설치됨 |
| `accelerate` | 1.13.0 | DDP/FSDP | 설치됨 |
| `flash-attn` | 2.8.3 | Flash Attention 2 | 설치됨 |
| `peft` | 0.18.1 | LoRA | 설치됨 |
| `bitsandbytes` | 0.49.2 | 8-bit AdamW | 설치됨 |
| `liger-kernel` | ≥0.3 | Fused Qwen2 kernels | **미설치 → `pip install liger-kernel`** |

flash_linear_attention-0.4.2
---

## 변경 전: 기존 구조

### 데이터 파이프라인

```
Dataset.__getitem__()
  └─ 반환: (waveform: Tensor[T], transcript: str)

collate_fn_factory(tokenizer)
  └─ 배치 내 최장 waveform 기준 zero-padding
  └─ 반환: (audio_padded: [B, T_max], audio_lengths: [B], transcript_ids: [B, L])

DataLoader
  └─ DynamicBatchSampler: max_batch_tokens 기준 greedy packing (인덱스만 묶음)
  └─ num_workers: Stage1=1, Stage2=8
```

**문제**: 배치 내 가장 긴 샘플 기준 패딩 → 최대 40% 토큰이 낭비.

### 모델 forward

```
model.forward(audio, audio_lengths, transcript_input_ids)
  ├─ encoder(audio) → (feats, enc_mask)          # 오디오 인코딩 (항상 fp32)
  ├─ projector(feats) + proj_norm → audio_embeds  # (B, T_proj, llm_dim)
  ├─ embed(prompt_p1_ids) → p1_embeds             # register_buffer에서
  ├─ embed(prompt_p2_ids) → p2_embeds
  ├─ embed(transcript_ids) → text_embeds
  └─ cat([p1, audio, p2, text]) → LLM → CE loss
```

**문제**: 시퀀스를 forward 시점에 구성하므로 전처리 단계에서 packing 불가.

### 학습 루프 병목

| 위치 | 문제 | 빈도 |
|------|------|------|
| `torch.distributed.broadcast(step_tensor)` | 불필요한 NCCL 통신 (step 카운터 동기화) | 매 gradient step |
| `outputs.loss.item()` | GPU→CPU sync | 매 gradient step |
| `mixed_precision="no"` | Accelerate autocast 비활성, 수동 bf16 | 항상 |
| `num_workers=1` (Stage 1) | DataLoader 병목 | 항상 |
| `wandb.log()` | 매 step 호출 | 매 gradient step |

---

## 변경 후: 새 구조

### 실행 방법

```bash
# Flash Attention 2만
torchrun --nproc_per_node=8 train.py --encoder fb_dacvae --flash-attn

# Sequence Packing + FA2
torchrun --nproc_per_node=8 train.py --encoder fb_dacvae --packing --flash-attn

# 전체 조합
torchrun --nproc_per_node=8 train.py --encoder fb_dacvae \
    --packing --flash-attn --liger --cutoff-len 2048

# FSDP (4B 모델)
torchrun --nproc_per_node=8 train.py --encoder fb_dacvae \
    --llm Qwen/Qwen3.5-4B --packing --flash-attn --fsdp
```

### 새 데이터 파이프라인 (`--packing` 시)

```
build_processor(tokenizer, cfg)
  └─ (waveform, transcript) →
       input_ids:  [p1_tokens] + [image_pad × t_audio] + [p2_tokens] + [text_tokens] + [EOS]
       labels:     [IGNORE × (p1+t_audio+p2)] + [text_tokens] + [EOS]
       audio_features: waveform (Tensor[S])
       audio_lengths:  t_audio (int)

SequencePacker(cutoff_len, neat_packing=True)
  └─ greedy knapsack (bisect 기반): 여러 샘플을 cutoff_len 안에 이어붙임
  └─ attention_mask: 샘플 인덱스(1,2,3...) 부여 → 경계 인식
  └─ 패딩을 cutoff_len까지

PackedCollator(attn_implementation)
  ├─ [FA2 경로]
  │   └─ 패딩 제거 → (1, sum_nonpad)
  │   └─ position_ids: 샘플마다 0부터 리셋 (FA2 varlen 경계 인식)
  └─ [eager/sdpa 경로]
      └─ 4D block-diagonal causal mask 생성

audio_features: (N_audio, 1, S_max)  ← 배치 내 모든 오디오 flatten
audio_lengths:  (N_audio,)
```

**효과**: 패딩 없이 cutoff_len을 꽉 채움 → 실질적 배치 크기 증가.

### 새 모델 forward (`input_ids=` 전달 시)

```
model._forward_packed(input_ids, labels, audio_features, audio_lengths, attention_mask, position_ids)
  ├─ _get_audio_embeds(audio_features, audio_lengths)
  │   └─ encoder → projector → proj_norm → audio_embeds: (N, T_proj, llm_dim)
  ├─ embed(input_ids) → inputs_embeds: (B, T, llm_dim)
  ├─ inputs_embeds[input_ids == image_pad_id] = audio_embeds.flatten(0,1)
  │   └─ placeholder 위치를 audio embedding으로 교체
  └─ LLM(inputs_embeds, attention_mask, position_ids, labels) → CE loss
```

**기존 forward 완전 보존**: `audio=` 인자로 호출 시 기존 경로(`_forward_legacy`) 사용.

---

## 각 최적화 설계 결정

### 1. Sequence Packing

**Audio placeholder token**: `<|image_pad|>` (id=151655) 재사용.  
Qwen2.5 base 모델에서 이 토큰은 vision 모델(Qwen2.5-VL) 용 슬롯으로 예약되었으나  
base/instruct 학습 데이터에 등장하지 않아 embedding이 사실상 비어있음 (미학습 슬롯).  
forward 시 항상 audio embedding으로 덮어써지므로 초기값 무관.

```python
# 검증: 세 visual pad 토큰 모두 동일한 norm, 일반 토큰의 1/3 수준
# image_pad: norm=0.3008, diff_from_mean=0.1445
# the:       norm=0.3652, diff_from_mean=0.4160
```

새 special token 추가 불필요 → `resize_token_embeddings()` 불필요.

**t_audio 계산**:
```python
t_audio = int(num_samples / cfg["samples_per_token"])
# samples_per_token = hop × (16000/tgt_sr) × prod(proj_strides)  ← config에서 계산
```
encoder+projector 실제 출력 길이와 정확히 일치해야 함.  
±1 frame 불일치 방어 코드 포함 (`_forward_packed` 내 assert/clamp).

**neat_packing**: attention_mask 값으로 샘플 경계를 인코딩 (0=pad, 1=sample1, 2=sample2, ...).  
FA2 없이도 `prepare_4d_causal_attention_mask`로 block-diagonal mask 생성 가능.

### 2. Flash Attention 2

```python
# model.py: from_pretrained에 한 줄 추가
attn_implementation=cfg.get("attn_implementation", "eager")
```

Qwen2.5-2B (Qwen2 아키텍처, GQA: 14 heads / 2 KV heads) — FA2 완전 호환.  
Sequence packing + FA2: position_ids를 샘플마다 0부터 리셋하면  
FA2가 position discontinuity를 경계로 인식 → cross-sample attention 자동 차단.

### 3. Liger Kernel

```python
# model.py: LLM 로드 직후, gradient_checkpointing_enable() 이전에 적용
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
apply_liger_kernel_to_qwen2(rope=True, rms_norm=True, swiglu=True,
                             fused_linear_cross_entropy=True)
```

`fused_linear_cross_entropy=True`: LM head 출력 logit 텐서를 materialize 없이  
chunk 단위로 CE loss 계산 → vocab=151,665 × seq_len × batch 크기의 메모리 절감.

**주의**: `fused_linear_cross_entropy=True` 사용 시 `out.loss` 직접 계산으로 대체됨.  
`eos_weight` 커스텀 loss (기존 `_forward_legacy`에서 사용) 와 충돌 가능 →  
packed 경로에서만 Liger CE 활성화, legacy 경로에서는 비활성.

### 4. FSDP

```python
# Accelerate FullyShardedDataParallelPlugin 사용
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(...),
)
accelerator = Accelerator(..., fsdp_plugin=fsdp_plugin)
```

**LoRA + FSDP**: `use_orig_params=True` 필수.  
8×A100 80GB + Qwen3.5-2B 기준으로도 **FSDP 적용** — sequence packing + FA2 + Liger 조합 시 메모리 여유가 줄어들기 때문.  
FSDP 없는 DDP는 이 조합에서 권장하지 않음.

---

## 학습 루프 병목 수정 (검증1 완료 후 적용)

| 병목 | 수정 내용 |
|------|---------|
| `torch.distributed.broadcast(step_tensor)` per step | 제거: 각 rank 독립 카운팅 (`global_step += 1`) |
| `loss.item()` per step | `loss.detach()` 누적 후 sync_gradients 시점에만 `.item()` |
| `mixed_precision="no"` | FA2 사용 시 `"bf16"` 로 변경 |
| `num_workers=1` (Stage 1) | → 4, `persistent_workers=True`, `prefetch_factor=2` |
| `wandb.log()` 매 step | `log_every=10` steps마다 로깅 |
| `train_stage1.py` | 위 변경사항 동일하게 적용 |

---

## 검증 계획

### 검증1: 기존 vs 새 최적화 플래그 (병목 수정 전)

```bash
# baseline
torchrun --nproc_per_node=8 train.py --encoder fb_dacvae \
    --datasets ls100 --ls-samples 5000 --wandb-mode offline

# 검증1a: --flash-attn
# 검증1b: --packing --flash-attn
# 검증1c: --packing --flash-attn --liger --fsdp   ← 주력 비교 (FSDP 필수)
# 검증1d: --packing --flash-attn --liger --fsdp --cutoff-len 4096   ← 실운용 설정
```

### 검증2: 병목 수정 후

```bash
# 검증2: 병목 수정 + 실운용 플래그
torchrun --nproc_per_node=8 train.py --encoder fb_dacvae \
    --datasets ls100 --ls-samples 5000 --packing --flash-attn --liger --fsdp \
    --cutoff-len 4096 --wandb-mode offline
```