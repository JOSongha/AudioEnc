# Stage 1 → Stage 2 Weight Change Analysis

Stage 2 fine-tune가 어느 layer에 어느 정도의 변화를 만들었는지 정량 측정.
Encoder representation 한계 vs LLM/projector adaptation 어디에 capacity가 쓰였는지 진단.

**스크립트**: `scripts/analysis/compare_stage1_stage2_weights.py`
**결과 PNG**: `/mnt/tmp/results/{run}_weight_change_heatmap.png`

---

## 1. 측정 정의

각 weight tensor에 대해 두 가지 지표:

```
rel_L2_diff (%) = ||W_s2 - W_s1||_F / ||W_s1||_F × 100
cos_sim         = <W_s1, W_s2> / (||W_s1|| ||W_s2||)        # Frobenius cos sim, float64
1 − cos_sim     = directional drift (방향 변화량)
```

- **L2 magnitude**: weight 절대 변화량
- **Cosine drift**: 방향 변화량 (단순 scaling은 cos=1이라 안 잡힘)
- 두 지표를 함께 봐야 "scale change vs direction change" 구분 가능

분류 (관행):
| 비교군 | rel L2 |
|---|---|
| 방치된 frozen layer (numerical noise) | < 0.1% |
| 보수적 LoRA fine-tune | 1~10% |
| Aggressive full fine-tune | 10~30% |
| Continued pretraining (대규모) | 20%+ |

**주의:** rel L2 % 자체가 downstream 영향력을 보장하지 않음.
LayerNorm은 절대값 작아도 영향 큼; LoRA 적용 안 된 24개 linear-attn layer는 변화 0%.
**상대 비교용** 지표.

---

## 2. 비교 setup

| Stage 2 ckpt | Base (Stage 1) | LoRA r/α | Trained for |
|---|---|---|---|
| WavTok-40-unify lora64 (ckpt-12000, full mix) | `Qwen3.5_wavtok_40_unify_Stage1/checkpoint-100000` | 64 / 128 | asr+emo+env+text mix |
| DAC-VAE lora16 (s2_best `dacvae_v2/ckpt-15000`) | `Qwen3.5AE-4B-dacvae_ASR-Stage1` (step 42k) | 16 / 32 | asr+emo+env+text mix |
| Whisper-tiny lora16 (s2_best ckpt-20000) | `Qwen3.5_whisper_tiny_Stage1/checkpoint-13000` | 16 / 32 | asr+emo+env+text mix |
| Whisper-small lora16 (s2_best ckpt-12000) | `Qwen3.5_whisper_small_Stage1/checkpoint-13000` | 16 / 32 | asr+emo+env+text mix |

공통:
- `freeze_vision_tower: true` — encoder frozen 유지
- `additional_target: audio_encoder.projector` — projector full fine-tune
- LoRA target: q/k/v/o_proj of full-attn LLM layers (qwen3.5의 hybrid attn에서 8개 layer)

---

## 3. 결과 요약

### 3.1 Encoder (frozen 검증)

직접 검증: DeepSpeed `param_shapes`에 추적되는 trainable param 142개 중 encoder 0개,
adapter_model.safetensors에도 encoder weight 0개. **0% 변화 확정**.

### 3.2 Projector (full fine-tune, 4 layer Llama-style)

#### Standalone params (rel L2 % / 1−cos)
| 파라미터 | WavTok lora64 | DAC-VAE lora16 | Whisper-tiny lora16 | Whisper-small lora16 |
|---|---|---|---|---|
| `input_proj.weight` (audio_dim → 512) | **1.59%** / 0.00013 | 5.45% / 0.00148 | 6.33% / 0.00199 | **6.53%** / 0.00213 |
| `output_proj.weight` (512 → LLM hidden 2560) | 5.49% / 0.00151 | 7.90% / 0.00312 | **9.90%** / 0.00491 | 7.87% / 0.00310 |
| `final_norm.weight` | 0.43% / 0.00001 | 0.50% / 0.00001 | 0.40% / 0.00001 | 0.32% / 0.00000 |

> Encoder hidden dim 차이로 input_proj shape 다름: WavTok 512×512 / DAC 128→512 / Whisper-tiny 384→512 / Whisper-small 768→512.
> rel L2 %는 dim에 무관하게 정규화된 metric.

#### Per-layer aggregate (4 transformer-style layer)
| 그룹 | WavTok lora64 (avg rel L2 %) | Whisper-tiny lora16 |
|---|---|---|
| self_attn.q_proj | 5.85 | (heatmap 참조) |
| self_attn.k_proj | 5.65 | |
| self_attn.v_proj | 2.98 | |
| self_attn.o_proj | 4.18 | |
| mlp.gate/up/down_proj | 5.13 / 5.16 / 5.39 | |
| input_layernorm | 0.26 | |
| post_attention_layernorm | 0.24 | |

### 3.3 LLM (LoRA)

8개 full-attention layer (3, 7, 11, 15, 19, 23, 27, 31)에만 LoRA 적용.
나머지 24개 linear-attention layer는 변화 0% (LoRA 미적용).

#### Per-projection 평균 (32개 module 집계)

| Run | r / α | q_proj | k_proj | v_proj | o_proj | **overall** |
|---|---|---|---|---|---|---|
| WavTok-40 lora64 | 64 / 128 | 5.16% | 3.53% | 2.86% | 2.81% | **3.59%** |
| DAC-VAE lora16 | 16 / 32 | 4.48% | 2.82% | 2.27% | 2.15% | **2.93%** |
| Whisper-tiny lora16 | 16 / 32 | 3.68% | 2.58% | 2.10% | 2.11% | **2.62%** |
| Whisper-small lora16 | 16 / 32 | 3.02% | 2.08% | 1.71% | 1.71% | **2.13%** |

#### Layer-wise (4 projection 평균, layer index별 rel L2 %)

| Run | L3 | L7 | L11 | L15 | L19 | L23 | L27 | L31 |
|---|---|---|---|---|---|---|---|---|
| WavTok-40 lora64 | 3.12 | 3.46 | 3.35 | 3.43 | 3.58 | 3.83 | 3.59 | **4.35** |
| DAC-VAE lora16 | 2.53 | 2.85 | 2.77 | 2.69 | 2.71 | 3.29 | 3.05 | **3.56** |
| Whisper-tiny lora16 | 2.17 | 2.37 | 2.29 | 2.27 | 2.46 | 2.95 | 2.89 | **3.54** |
| Whisper-small lora16 | 1.71 | 1.90 | 1.78 | 1.77 | 2.10 | 2.50 | 2.50 | **2.79** |

#### Cos sim range
| Run | min ~ max |
|---|---|
| WavTok lora64 | 0.99792 ~ 0.99972 |
| DAC-VAE lora16 | 0.99831 ~ 0.99982 |
| Whisper-tiny lora16 | 0.99867 ~ 0.99984 |
| Whisper-small lora16 | 0.99916 ~ 0.99989 |

> LoRA rank가 다르면 ΔW 표현력 자체가 달라지므로 (r=64 vs r=16),
> rank-controlled 비교는 **DAC-VAE vs Whisper-tiny** (둘 다 r=16)에서만 가능.

---

## 4. 핵심 관찰

### 4.1 Encoder이 정말 frozen인지 직접 검증
인라인 추론("adapter에 없으니 변화 없음")만으로는 부족.
DeepSpeed `param_shapes` (학습 중 추적된 trainable param 목록) 직접 검사 → encoder 0개 확인.
**0% 변화 확정** (gradient 한 번도 받지 않음).

### 4.2 Projector adaptation 패턴
- **q_proj > k_proj > o_proj > v_proj**: query/key projection이 fine-tune capacity의 대부분 흡수 → "어떤 audio token에 attend할지" 학습이 핵심
- **MLP 균등 변화 (5%대)**: feature space 일반 조정
- **LayerNorm 거의 안 변함 (0.2~0.4%)**: 정규화는 안정 — 정상 신호

### 4.3 Encoder별 projector 변화 격차

input_proj 변화: **WavTok 1.59% < DAC 5.45% < Whisper-tiny 6.33% ≈ Whisper-small 6.53%**
output_proj 변화: **WavTok 5.49% < DAC 7.90% ≈ Whisper-small 7.87% < Whisper-tiny 9.90%**

해석 후보:
1. Whisper-tiny가 가장 많이 흔들림 — 작은 ASR-편향 encoder feature를 multi-task용으로 reshape하는 데 가장 큰 변화 필요
2. Whisper-small은 더 큰 encoder라 feature가 풍부 → output_proj는 tiny보다 적게 흔들림 (richer feature → less reshape)
3. WavTok은 reconstruction codec인데도 가장 적게 흔들림 — stage 1을 100k step (DAC 42k, Whisper 13k 대비 2~7배)으로 학습해서 projector가 이미 깊이 수렴된 영향 가능성
4. DAC-VAE는 모든 지표에서 중간

### 4.4 Encoder feature 풍부함 ↔ 필요한 LLM adaptation 양

r=16 fair comparison (rank 통제):

| Run | input_proj | output_proj | LLM 평균 |
|---|---|---|---|
| DAC-VAE lora16 | 5.45% | 7.90% | **2.93%** |
| Whisper-tiny lora16 | 6.33% | 9.90% | **2.62%** |
| Whisper-small lora16 | 6.53% | 7.87% | **2.13%** |
| WavTok lora64 (참고) | 1.59% | 5.49% | 3.59% (rank 64로 인플레) |

핵심 패턴 (r=16 셋):
- **LLM 평균: DAC > Whisper-tiny > Whisper-small** (2.93 > 2.62 > 2.13)
- Encoder가 클수록/feature가 풍부할수록 LLM 적응량 ↓
- Whisper-small은 stage 1 ASR WER 2.58%로 가장 좋은 acoustic representation → stage 2에서 LLM 쪽 보정 가장 적게 필요

**Projector ↔ LLM의 부분적 trade-off**:
- Whisper-tiny vs Whisper-small: tiny가 output_proj 9.9% + LLM 2.62%, small이 output_proj 7.9% + LLM 2.13% — small이 양쪽 모두 적게 변함 → "encoder 클수록 전체 adaptation 줄어듦"이 더 강한 효과
- DAC vs Whisper-small: DAC가 LLM 2.93% (더 큼), 둘 다 projector ~7.9% 비슷 → encoder pretraining 목적(reconstruction vs ASR) 차이가 LLM 보정량에 반영
- WavTok 예외 사례: projector 변화는 가장 적은데 LLM은 r=64에 큰 변화 — stage 1 projector가 이미 saturated되어서 stage 2 budget이 LLM으로 흘러갔을 가능성

> "Projector ↔ LLM 단순 반비례"라기보다, **"encoder 표현력이 부족할수록 어딘가에서 더 많이 보정해야 함"** 으로 정리. 어디로 흘러가는지는 stage 1 학습 깊이와 LoRA rank capacity에 따라 분배.

### 4.5 LoRA q_proj 편향이 일반 패턴
LLM/projector 공통, 그리고 **4개 encoder 모두에서 일관됨**:
- **q_proj > k_proj > v_proj ≈ o_proj** (모든 ckpt)
- **deeper layer일수록 큰 변화** (모든 ckpt에서 L31이 max, L3 대비 1.4~1.6배)
- audio-conditioned task adaptation은 query/key 학습 중심 — "어떤 audio token에 attend할지"가 핵심
- 후반부 layer는 multi-modal output 생성에 더 기여

### 4.6 변화 규모는 "보수적 fine-tune" 범위
- 모든 변화 ≤ 10% (cos > 0.995)
- LR 1e-5 × 12~20k step의 결과로 자연스러운 규모
- pretrained representation을 망가뜨리지 않는 안전한 영역

---

## 5. 후속 검증 아이디어

| 가설 | 검증 방법 |
|---|---|
| Projector 변화량 ≈ encoder feature가 LLM space에 안 맞아서 보정 중 | Stage 1 final → Stage 1 100k 비교: projector 자체가 stage 1에서 얼마나 흔들렸는지 비교 |
| q_proj 편향이 일반 패턴인지 audio-task 특수인지 | text-only LoRA fine-tune의 q/k/v/o 변화 비교 |
| Projector capacity 충분한지 | larger projector (예: 6-layer or transformer projector)로 학습해서 변화량 차이 비교 |
| Encoder unfreeze 시 어디부터 흔들리는지 | encoder LoRA 추가하여 SEANet/Whisper encoder의 layer-wise 변화 측정 |

---

## 6. 재현 방법

```bash
python scripts/analysis/compare_stage1_stage2_weights.py \
    --stage1 external/ckpts/.../checkpoint-NNNN \
    --stage2 /path/to/stage2/checkpoint-NNNN \
    --out    /path/to/heatmap.png \
    --title-suffix " (run name)"
```

자동 출력:
1. encoder frozen 검증 (Stage 2 adapter에 encoder weight 0개)
2. projector standalone params (input/output_proj, final_norm) L2% + cosine
3. LoRA target layer 자동 감지 (adapter_config의 `target_modules` 파싱)
4. 4-panel heatmap (projector/LLM × L2/cosine drift, float64 정밀도)

---

## 7. 참조 자료

- 분석 스크립트: `scripts/analysis/compare_stage1_stage2_weights.py`
- WavTok lora64 stage2: `Qwen3.5AE-wavtok-Stage2-lora64/.../checkpoint-12000`
- DAC-VAE lora16 best: `external/ckpts/s2_best_ckpts/dacvae_v2/checkpoint-15000`
- Whisper-tiny lora16 best: `external/ckpts/s2_best_ckpts/whisper_tiny/checkpoint-20000`
- Whisper-small lora16 best: `external/ckpts/s2_best_ckpts/whisper_small/checkpoint-12000`
- 학습 로그: `Audio_Results/ASR_Libri_whisper_normalization/wavtok_stage2_lora64_full_mix/`
- Linear probing 결과 (encoder 비교): `experiments/audio_encoder_probe/` *(별도 분석)*
- 결과 PNG: `/mnt/tmp/results/{wavtok,dacvae,whisper_tiny,whisper_small}_stage2_weight_change_heatmap.png`
