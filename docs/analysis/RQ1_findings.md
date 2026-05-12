# RQ1 — Layer-wise CKA Distance on IEMOCAP (Findings)

> 방법론: [RQ1_layer_distance.md](RQ1_layer_distance.md)
> 산출물: `experiments/audio_encoder_probe/{cka, figs, embeds_layers}/`

---

## 개요

- **목적**: 4 Audio LLM family 의 layer-wise representation 이 어디서/어떻게 변하는지, family 간 차이가 어디서 흡수되는지
- **데이터**: IEMOCAP 4-class 균형 샘플 2000 utt (class 별 500)
- **Metric**: Linear CKA (mean + last pooling)
- **범위**: Stage 1 ckpt 만 (**LLM frozen 확증됨** — §관찰 4)

## 분석 대상

| Family | Audio Encoder | enc fps | enc 출력 dim |
|---|---|---|---|
| whisper_tiny | Whisper-tiny.en (4 Transformer) | 50 | 384 |
| whisper_small | Whisper-small.en (12 Transformer) | 50 | 768 |
| dacvae | DAC CNN + VAE bottleneck | 25 | 128 (post-VAE z) |
| wavtok_40_unify | SEANet (pre-VQ z_e) | 40 | 512 |

**공통**: Projector (`input_proj` → 4 × `LlamaDecoderLayer` 512-d → `output_proj` 2560-d) + Qwen3.5-4B (32 layers, 24 `linear_attn` + 8 `full_attn` at [3,7,11,15,19,23,27,31])

## 추출 레이어 (15 / family = 5 enc + 5 proj + 5 llm)

| 위치 | whisper_tiny | whisper_small | dacvae | wavtok |
|---|---|---|---|---|
| enc_0 | embed (384) | embed (768) | conv_in (64) | conv_in (32) |
| enc_1 | L0 (384) | L2 (768) | block_0 s=2 (128) | stage_0 s=2 (64) |
| enc_2 | L1 (384) | L5 (768) | block_1 s=8 (256) | stage_1 s=4 (128) |
| enc_3 | L2 (384) | L8 (768) ⚠️ outlier | block_3 s=12 (1024) | stage_3 s=8 (512) |
| enc_4 (→proj 입력) | last_hidden_state (384) | last_hidden_state (768) | post-VAE z (128) | conv_out (512) |
| proj_0..3 | LlamaDecoderLayer × 4 | (512) | | |
| proj_out | output_proj | (2560) | | |
| llm_0,8,15,23,31 | Qwen3.5 subsample (llm_15,23,31 = full_attn) | (2560) | | |

---

## 핵심 관찰

### 관찰 1: Cross-family CKA trajectory (mean of 6 pairs, mean pooling)

| Position | CKA | Δ |
|---|---|---|
| enc_0 | 0.18 | (start) |
| enc_4 | 0.42 | 점진 증가 |
| **proj_0** | **0.55** | **+0.13 ← single biggest jump** |
| proj_3 | 0.44 | 다시 발산 |
| proj_out | 0.48 | +0.04 |
| llm_31 | 0.46 | LLM 동안 거의 변화 없음 |

### 관찰 2: `proj_out ↔ llm_0` family 별 (within-family)

| Family | CKA | rel L2 변형 | cosine |
|---|---|---|---|
| whisper_tiny | 0.9999 | 15.0% | 0.989 |
| whisper_small | 0.9998 | 12.9% | 0.992 |
| dacvae | 1.0000 | 22.6% | 0.978 |
| **wavtok** | **0.9838** | **29.4%** | **0.957** |

→ wavtok 만 layer 0 변형이 **2 배 강함**

### 관찰 3: Mean vs Last (llm_31 cross-family 평균)

| Pooling | CKA |
|---|---|
| mean | 0.46 |
| last | 0.17 |

→ **3 배 차이**. last pooling 이 family signature 보존

### 관찰 4: Frozen LLM 확증

- 4 family Stage 1 ALM 의 LLM weight (`mlp.*`, `linear_attn.*`, `embed_tokens`, `norm`) 가 **bit-exact identical**
- → Stage 1 은 projector 만 학습, LLM 완전 frozen

### 관찰 5: 기타 정량 (95% bootstrap CI 동반, permutation p < 0.001)

| 항목 | CKA |
|---|---|
| DACVAE block_3 (1024-d) ↔ post-VAE z (128-d) | **0.998** |
| dacvae enc_4 ↔ wavtok enc_4 | **0.97** |
| whisper × CNN enc_0 (cross-family) | **~0.00** |
| whisper_small L8 (enc_3) ↔ neighbors | 0.17–0.26 (다른 enc pair 0.85+) |

---

## 해석

- **Cross-family unification 의 핵심은 projector input_proj**: 차원 정렬 (32~1024-d → 512-d) 자체가 main 메커니즘. 이후 projector transformer 는 다시 family-specific 정보 회복.
- **Layer 0 변화 family-dependent**: frozen LLM 이라도 projector 가 만드는 audio embedding 분포 차이 → wavtok 만 2× 강한 반응. 두 가능성:
  - (A) wavtok audio_embed 가 text token embed 와 더 align → 정상 LLM 반응
  - (B) wavtok audio_embed 가 OOD → wild 반응
  - 현 데이터로 구분 불가 — layer 0 δ 직접 측정 필요.
- **CKA = 1 ≠ 동일**: centered sample-sample Gram matrix 가 같으면 element 값이 달라도 1.0. proj_out ↔ llm_0 은 sample 간 상대 관계가 보존되는 변환.
- **Mean vs Last**: mean 은 시간축 평균으로 family signature 희석, last (causal accumulation) 은 보존 → inference generation pivot 차이.
- **acoustic vs semantic 분리 본질적**: CNN family 간은 후반에 수렴 (dacvae ↔ wavtok enc_4 = 0.97), Whisper × CNN 은 pipeline 끝까지 별개 manifold.
- **DACVAE VAE 무손실**: CNN block_3 의 intrinsic dim ≤ 128.

---

## 가설 평가

| 가설 (방법론 §1) | 결과 | 근거 |
|---|---|---|
| H1.1: enc→proj 가장 sharp | ⚠️ 부분 | cross-family +0.13 jump (가장 큼), within-family smooth |
| H1.2: proj→llm 큰 점프 | ❌ 반증 | proj_out↔llm_0 ≈ 1.0 |
| H1.3: LLM 후반 plateau | ✅ 확증 | llm_8..23 인접 0.95+ |
| H2.1: enc 단 family 차이 최대 | ✅ 확증 | sem×ac enc_0 ≈ 0.00 |
| H2.2: projector 통과 후 차이 감소 | ✅ 확증 | enc_4→proj_0 cross-CKA 0.42→0.55 |
| H2.3: LLM 후반 family 차이 최소 | ❌ 반증 | llm_31 cross-CKA ≈ proj_out cross-CKA |
| H2.4: acoustic vs semantic 본질적 분리 | ✅ 강하게 확증 | 모든 위치에서 sem×ac < sem×sem |

### 새로 발견한 패턴 (가설에 없던 것)

- 🆕 wavtok 의 frozen LLM 반응이 다른 family 보다 2× 강함 (A/B 미해소)
- 🆕 Mean vs Last pooling 의 divergence (LLM 후반 mean=0.46, last=0.17)
- 🆕 DACVAE VAE 가 무손실 차원 정리 (block_3↔z CKA=0.998)
- 🆕 Whisper-small layer 8 anomaly
- 🆕 Projector `input_proj` 단계가 cross-family alignment 의 single 최대 jump

---

## 비판적 재평가 — 어떤 발견이 실제로 robust 한가

각 finding 을 frozen LLM / causal attribution / variance artifact / overclaim 관점에서 재검토:

| Finding | 진짜 의미 있는가? | 이유 |
|---|---|---|
| 1. LLM 후반 family 차이 감소 반증 (H2.3) | ❌ 의미 약함 | LLM frozen → audio 학습 안 했으니 통합 못 하는 게 당연. 가설 자체가 frozen 에 적용 불가능했음. Stage 2 비교 전까진 결론 불가 |
| 2. `input_proj` 가 unification 의 main | ⚠️ 인과 귀속 추측 | `proj_0` hook = Linear + 첫 LlamaDecoderLayer **둘 다** 통과한 결과. Linear 단독 vs Linear+Layer 분리 측정 안 됨 |
| 3. LLM 첫 layer = residual identity | ⚠️ 부정확 | (a) wavtok=0.984 ≠ 1.000 — 모든 family 동일 아님. (b) "audio→텍스트 변환 안 함" 도 frozen 효과 |
| 4. Mean vs Last divergence | ⚠️ Variance artifact 가능 | last 는 단일 frame (N=1), mean 은 ~수백 frame 평균. CKA 차이가 family signature 인지 통계적 noise 인지 구분 안 됨. **random frame baseline 필요** |
| 5. DACVAE VAE 무손실 | ✅ Robust | block_3 (1024-d) ↔ post-VAE z (128-d) CKA=0.998 명확. 단 "intrinsic dim ≤ 128" 은 IEMOCAP 도메인 한정 해석 |
| 6. CNN encoder universal convergence | ⚠️ N=2 overclaim | dacvae↔wavtok enc_4=0.97 이지만 N=2. 후반 layer 에서만 수렴 (enc_2 까지는 ~0.4). "두 모델 후반 수렴" 이 정확 |
| 7. Whisper-small L8 anomaly | ✅ 패턴 진짜, 해석 추측 | L8↔L5=0.26, L8↔L11=0.20, L5↔L11=0.91. 패턴은 명확. "specialist layer" 메커니즘은 미검증 — 12 layer 전부 추출 + probing 필요 |

### 진짜 robust 한 결과 (2 개)

- ✅ **#5 DACVAE VAE 무손실**: 실험 결과 명확, 해석은 도메인-specific 한정
- ✅ **#7 Whisper-small L8 outlier**: 패턴 통계적으로 명확, 메커니즘 해석은 후속

### 추가 실험으로 검증 필요 (3 개)

- 🟡 **#2 input_proj 단독 효과**: `input_proj` 출력만 별도 추출하면 측정 가능
- 🟡 **#4 last vs mean**: random single frame 을 baseline 으로 두면 측정 가능
- 🟡 **#6 CNN convergence**: 더 많은 acoustic codec family 추가 필요

### Stage 2 비교 없이는 결론 불가 (2 개)

- 🔴 **#1 H2.3 반증**: frozen LLM 효과인지 진짜 LLM 특성인지 결정 불가
- 🔴 **#3 LLM residual identity**: 같은 이유

### 신뢰성 sumary

| Tier | 발견 |
|---|---|
| 🟢 Robust (Stage 2 와 무관) | VAE 무손실, L8 anomaly |
| 🟡 Methodologically improvable | input_proj 분리, last frame baseline, CNN N 늘리기 |
| 🔴 Frozen LLM 제약 | LLM 관련 모든 결론 |

---

## 후속 작업 (우선순위)

1. 🔥 **Stage 2 ckpt 비교** — H2.3 가 진짜 반증인지, frozen 특수효과인지 결정적 판정
2. **wavtok (A)/(B) 판정** — layer 0 δ 측정 (audio_pad vs text token activations) → text-aligned vs OOD 구분
3. **whisper-small L8 specialization 검증** — 12 layer 전부 추출, attention pattern / probing
4. **last token signature → downstream 상관** — emotion classification / WER 와 last-pooled llm_31 관계
5. **Projector input_proj 단독 효과** — Linear 만 통과 vs Linear+LlamaDecoderLayer 분리 측정

---

## 산출물 / 코드

| Stage | 파일 |
|---|---|
| Manifest | `experiments/audio_encoder_probe/{build_balanced_manifest.py, manifests/iemocap_4class_balanced500.csv}` |
| Extraction | `extract_layers.py` → `embeds_layers/{family}/*.npz` (8000 파일, 1.2 GB) |
| CKA | `compute_cka.py` → `cka/*.npy` (52 파일, 40 within + 10 cross + 2 metadata) |
| Plot | `plot_rq1.py` → `figs/*.png` (54 plots, 14 main + 40 emotion-conditioned appendix) |
| Statistical | `cka_robust.py` → `cka/axis2_ci.json` (48 결과 bootstrap CI + permutation p) |
| Tests | `tests/experiments/test_{balanced_manifest, cka, extract_layers}.py` (34 tests, 모두 통과) |
