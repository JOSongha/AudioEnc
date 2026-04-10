# Sequence Packing + Flash Attention 2 + Liger Kernel + FSDP

## 개요

학습 속도·메모리 효율화를 위한 4가지 최적화.  
`train_pipeline_override.py` 기준: **Sequence Packing·Flash Attention 2는 기본값 ON**.  
Liger Kernel(`--liger`)·FSDP(`--fsdp`)는 CLI 플래그로 활성화.

> **참고**: 이 문서의 `--packing`, `--flash-attn` 예시는 설계 초안 기준이며,  
> 실제 `train_pipeline_override.py`에서는 플래그 없이도 두 기능이 기본 활성화됨.  
> FA2 비활성화가 필요한 경우 `--attn-impl sdpa` 또는 `--attn-impl eager` 사용.

---

## 설치 요구사항

| 패키지 | 버전 | 용도 | 상태 |
|--------|------|------|------|
| `torch` | 2.5.1+cu124 | 기본 | 설치됨 |
| `torchaudio` | 2.5.1 | 오디오 로드 | 설치됨 |
| `transformers` | 5.5.0 | LLM | 설치됨 |
| `accelerate` | 1.12.0 | DDP/FSDP | 설치됨 |
| `flash-attn` | 2.8.3 | Flash Attention 2 | 설치됨 |
| `peft` | 0.18.0 | LoRA | 설치됨 |
| `bitsandbytes` | 0.49.0 | 8-bit AdamW | 설치됨 |
| `liger-kernel` | 0.7.0 | Fused Qwen2 kernels | 설치됨 |
| `datasets` | 3.6.0 | HuggingFace 데이터셋 | 설치됨 |

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

**EOS / PAD 토큰** (Qwen3.5-2B):

| 토큰 | 문자열 | ID |
|------|--------|----|
| EOS  | `<\|im_end\|>` | 151643 → (Qwen3.5) 248046 |
| PAD  | `<\|endoftext\|>` | 248044 |

EOS ≠ PAD. `model.py`에서 `pad_token is None`이면 EOS로 대체하는 분기가 있으나,  
Qwen3.5-2B는 PAD가 이미 설정되어 있으므로 **실행되지 않음** — 두 토큰은 항상 다름.

**PAD 발생 위치 및 마스킹**:
- `collate_fn`: 배치 내 최장 샘플 기준 right-padding (pad_id=248044)
- `eos_first=False` (기본): `tgt_labels[:, :-1]`의 PAD만 -100 → 마지막 열(EOS) 무조건 보존
- `eos_first=True`: PAD 위치 전부 -100 → EOS ≠ PAD이므로 EOS는 위치에 무관하게 자동 보존

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

`train_pipeline_override.py` 기준 (packing·FA2는 기본 활성화):

```bash
# 기본 (packing + FA2 자동 활성화)
accelerate launch train_pipeline_override.py --encoder fb_dacvae

# Liger + FSDP 추가 (권장 실운용 설정)
accelerate launch train_pipeline_override.py --encoder fb_dacvae \
    --liger --fsdp

# cutoff 길이 변경
accelerate launch train_pipeline_override.py --encoder fb_dacvae \
    --liger --fsdp --cutoff-len 4096

# FA2 비활성화 (SDPA로 fallback)
accelerate launch train_pipeline_override.py --encoder fb_dacvae \
    --attn-impl sdpa

# FSDP + 4B 모델
accelerate launch train_pipeline_override.py --encoder fb_dacvae \
    --llm Qwen/Qwen3.5-4B --liger --fsdp
```

> **flash_attn LD_PRELOAD 필요** (GLIBCXX_3.4.29 + GLIBC_2.32):  
> `LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/glibc_compat.so" accelerate launch ...`  
> 자세한 내용은 `docs/train_pipeline_errors.md` §6.7–6.8 참조.

### 새 데이터 파이프라인 (항상 활성화)

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

OmniCollator(attn_implementation)
  ├─ [FA2 경로]
  │   └─ 패딩 제거 → (1, sum_nonpad)
  │   └─ position_ids: 샘플마다 0부터 리셋 (FA2 varlen 경계 인식)
  └─ [eager/sdpa 경로]
      └─ 4D block-diagonal causal mask 생성

audio_features: (N_audio, 1, S_max)  ← 배치 내 모든 오디오 flatten
audio_lengths:  (N_audio,)
```

**PAD 처리 비교**:

| 설정 | PAD 존재 여부 |
|------|--------------|
| packing 없음 (구버전 `train.py`) | 배치 내 최대 길이까지 PAD (최대 40% 낭비) |
| packing only (`--attn-impl sdpa`) | 각 pack 끝에 소량 PAD (cutoff_len까지 채움) |
| packing + FA2 (기본) | **PAD 완전 제거** → `(1, sum_nonpad)` shape |

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

#### 한 줄 요약

> LLM 내부 연산(RoPE, LayerNorm, FFN, loss)을 Triton으로 구현한 **fused kernel 라이브러리**.  
> **메모리를 덜 쓰고 HBM 왕복이 줄어서 빠르다.**

---

#### 왜 필요한가 — "fused"의 의미

PyTorch 연산은 기본적으로 **연산 하나마다 GPU 메모리(HBM)를 한 번씩 읽고 쓴다**.

```
일반 PyTorch (RMSNorm 예시):
  HBM에서 x 읽기 → x² 계산 → 결과 HBM에 저장
  HBM에서 다시 읽기 → mean 계산 → HBM에 저장
  HBM에서 다시 읽기 → x/rms 계산 → HBM에 저장
  HBM에서 다시 읽기 → weight 곱 → HBM에 저장   ← 총 4번 왕복

Liger Fused (RMSNorm):
  HBM에서 x 읽기 → 한 번에 x², mean, rms, weight 곱 전부 계산 → HBM에 저장   ← 1번 왕복
```

GPU는 연산(FLOP)보다 메모리 읽기/쓰기(bandwidth)가 병목인 경우가 많다.  
Liger는 이 병목을 "묶어서 한 번만 읽고 쓰기"로 줄인다.

---

#### 사용 방법

```python
# model.py: LLM 로드 직후, gradient_checkpointing_enable() 이전
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
apply_liger_kernel_to_qwen2(rope=True, rms_norm=True, swiglu=True,
                             fused_linear_cross_entropy=True)
```

이 한 줄이 Qwen2 모델의 모듈들을 Liger 구현으로 **in-place 교체**한다.  
모델 구조 변경 없이 동작하며, 학습 결과에 영향을 주지 않는다.

---

#### 교체되는 모듈 4가지

| 옵션 | 교체 대상 | 효과 |
|------|---------|------|
| `rope=True` | `apply_rotary_pos_emb` (Q, K에 위치정보 적용) | HBM 왕복 횟수 감소 |
| `rms_norm=True` | `Qwen2RMSNorm` (각 layer 앞뒤 정규화) | HBM 왕복 4→1회 |
| `swiglu=True` | `Qwen2MLP` 내 SiLU gate 연산 | 중간 activation 텐서 1개 제거 |
| `fused_linear_cross_entropy=True` | `lm_head` + `CrossEntropyLoss` | **logit 텐서 ~5 GB 절감** ← 가장 중요 |

---

#### fused_linear_cross_entropy가 핵심인 이유

일반적으로 LLM loss 계산은 이렇게 된다:

```
hidden_states (16384 tokens, 2048 dim)
  → lm_head (선형 변환)
  → logits: (16384 tokens, 151665 vocab) 텐서 생성  ← 이게 문제
  → CrossEntropy(logits, labels)
```

`logits` 텐서의 크기: 16384 × 151665 × 2 bytes(bf16) ≈ **4.97 GB**  
이 텐서가 순전파 + 역전파 동안 GPU 메모리에 상주한다.

Liger는 이 텐서를 **절대 통째로 만들지 않는다**:

```
hidden_states를 작은 chunk (예: 1024 tokens)로 나눔
  각 chunk:
    lm_head 적용 → 작은 logit (1024 × 151665) 임시 생성
    CE loss 계산 → gradient 저장 → 임시 logit 즉시 삭제
  → 5 GB 짜리 텐서가 메모리에 올라가지 않음
```

**주의**: 이 옵션은 HuggingFace의 `out.loss` 계산 경로를 교체한다.  
커스텀 loss(`_forward_legacy`의 `eos_weight`)와 충돌하므로  
**packed 경로(`_forward_packed`)에서만 활성화**, legacy 경로에서는 비활성.

---

#### 구현 이슈: qwen3 vs qwen3_5 아키텍처 불일치

`liger-kernel 0.7.0`에는 `apply_liger_kernel_to_qwen3_5`가 없다.  
기존 코드는 ImportError 시 `apply_liger_kernel_to_qwen3`로 fallback했는데,  
이 함수는 `transformers.models.qwen3.*` 클래스를 패치하지만  
Qwen3.5-2B는 `transformers.models.qwen3_5.*` 클래스를 사용한다 → **패치가 silent no-op**.

결과적으로 "Liger ON"과 "Liger OFF" 실험 모두 실질적으로 Liger 없이 실행되었다.

**수정**: `train_pipeline_override.py`에서 fallback 대신 `modeling_qwen3_5` 모듈을 직접 패치.  
주요 호환성 차이점:

| 항목 | Qwen3 | Qwen3.5 | 처리 방법 |
|------|-------|---------|---------|
| RMSNorm 수식 | `output * weight` (init: ones) | `output * (1 + weight)` (init: zeros) | `LigerRMSNorm(offset=1.0, init_fn="zeros")` 서브클래스 |
| MLP 시그니처 | `__init__(self, config)` | `__init__(self, config, intermediate_size)` | config를 복사해 `intermediate_size` 주입하는 래퍼 |
| Fused CE | `Qwen3ForCausalLM.forward` 교체 | `Qwen3_5ForCausalLM.forward` 교체 | `_qwen3_lce_forward` 직접 할당 |

---

#### 실측 비교 (2026-04-10, 8×A100-80GB, fb_dacvae, cutoff_len=16384)

| 설정 | slow step | fast step | 비고 |
|------|-----------|-----------|------|
| Liger OFF (no-liger 베이스라인) | ~109 s | ~25 s | DataLoader 대기 / compute |
| Liger "ON" (버그 전, 실제 no-op) | ~109 s | ~25 s | qwen3_5 패치 안 됨 |
| Liger ON (버그 수정 후) | ~108 s | ~27 s | 실제 패치 적용, 차이 없음 |

**결론**: Liger 적용 후에도 스텝 시간이 거의 변하지 않는다.

**근본 원인 (2026-04-10 추가 분석)**:  
DataLoader 병목과 별개로, Liger가 최적화하는 연산(RoPE, RMSNorm, SwiGLU, CE loss)은 **memory-bound element-wise 연산**으로 전체 compute의 5~10%에 불과하다. 학습 step의 대부분은 Liger가 건드리지 않는 **196개 matmul** (Q/K/V/O + gate/up/down × 28 레이어)과 **attention 연산**이 지배한다. 5~10% 구간을 2배 빠르게 해도 전체 step은 2.5~5% 개선에 그쳐 측정 노이즈 범위.

> **업데이트 (2026-04-10)**: Offline Pre-Packing (§10 in `dataloader_trials.md`)으로 DataLoader 병목 해소.
> Pre-packed Arrow 모드에서는 CPU packing이 없으므로 Liger가 compute 구간에서 효과를 발휘할 수 있다.
> 단, gradient_checkpointing이 여전히 ON (cutoff_len=16384에서 필수 — OFF 시 OOM)이므로
> Liger의 메모리 절감이 배치 크기 증가로 이어지지는 않음. 순수 compute 속도 개선만 기대.

> **더 큰 속도 개선 후보**: `flash-linear-attention` 미설치로 Qwen3.5의 linear attention 레이어가
> 순수 PyTorch fallback으로 동작 중. fla 설치 시 fused CUDA kernel 사용 → Liger보다 훨씬 큰 속도 개선 기대.
> 설치 방법 및 상세: `dataloader_trials.md` §9b 참조.
> Pre-packed 모드에서의 Liger 효과 재측정 예정.

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

## 검증 결과 (완료: 2026-04-08)

아래 4가지 설정을 `train_pipeline_override.py`로 검증 완료.  
각 테스트: `--max-steps 2 --datasets ls100 --stage all` (Stage 1→2 전체 통과 확인).

| 테스트 | 설정 | 결과 |
|--------|------|------|
| Test 1 | `--attn-impl sdpa` (packing ON, FA2 OFF) | ✅ PASS |
| Test 2 | FA2 기본값 (packing+FA2 ON) | ✅ PASS |
| Test 3 | FA2 + `--liger` | ✅ PASS |
| Test 4 | FA2 + `--liger` + `--fsdp` | ✅ PASS (`train_loss: 4.504`) |

```bash
# Test 4 실행 명령 (검증 기준)
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/glibc_compat.so" \
accelerate launch --num_processes 2 train_pipeline_override.py \
    --encoder fb_dacvae --attn-impl flash_attention_2 --liger --fsdp \
    --wandb-mode disabled --max-steps 2 --datasets ls100 --stage all
```