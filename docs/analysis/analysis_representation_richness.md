# Analysis — Representation Richness across Encoder Families

> ALM (Audio Language Model) 의 audio encoder family 에 따라 임베딩이 보존하는 정보 구조가 어떻게 달라지는지 측정한다. Stage 1 학습된 4 개 encoder family (Whisper-tiny, Whisper-small, EnCodec-24k, WavTokenizer-40) 의 표현 공간을 비교 대상으로 한다.

---

## 1. Hypothesis

본 분석은 두 갈래의 가설을 차례로 검증한다.

### (A) Disentanglement / Information-preservation 가설 — 우선 검증

**Semantic ALM** (Whisper 계열, ASR loss 로 사전학습된 encoder 기반)
- speaker variance 가 표현에서 제거됨
- semantic clustering 만 남음
- 같은 transcript 의 임베딩이 화자에 무관하게 한 점으로 수렴 (within-class collapse)
- 다른 transcript 끼리는 분리되지만, acoustic 변이 축은 무너져 있음

**Acoustic ALM** (EnCodec / WavTokenizer 계열, 재합성 loss 기반 encoder)
- speaker + semantic 둘 다 유지됨
- 임베딩 geometry 가 더 풍부 (richer)
- 같은 transcript 라도 화자별로 분리되어 있고, semantic 축도 함께 분리

> **핵심 메시지.** Acoustic encoder 기반 ALM 은 information-preserving 하면서도 disentanglable 한 표현을 만들어 둔다. 즉 downstream LLM 이 필요할 때 화자/acoustic 단서를 끌어 쓸 수 있다.

> **Layer-wise 관점.** 위 대비는 encoder 출력에서 가장 또렷할 것으로 예상되며, projector 와 LLM 을 거치며 변화 양상이 어떻게 흐르는지 (acoustic 축이 보존되는지 vs 점진적으로 collapse 되는지) 가 본 분석의 두 번째 관전 포인트다.

### (B) Spectrum / capacity 가설 — 후속 검증

표현 공간의 **용량 자체** (effective dimensionality, spectrum flatness) 가 family 간에 다르다는 가설. (A) 가 disentanglement 라는 *질적* 성질을 본다면, (B) 는 임베딩이 분포해 있는 manifold 의 *유효 차원* 을 본다. (A) 결론을 본 뒤 같은 데이터에 대해 후속으로 측정한다.

---

## 2. 비교 대상 (ALM Stack, Layer-wise)

비교 단위는 encoder 가 아니라 **ALM (encoder + projector + LLM) 의 각 layer 임베딩** 이다. 4 개 family 모두 RVQ/VQ 를 통과시키지 않고 encoder 의 **continuous latent** 를 projector 입력으로 쓴다 (e.g. EnCodec 은 pre-RVQ `z_e`).

### 2.1 ALM family 구성

모든 family 는 동일한 LLM backbone (Qwen3.5-4B, hidden 2560, 32 layers) 과 동일한 projector hidden (512, 4 layers) 을 공유한다. 차이는 encoder 와 그 출력 차원뿐.

| Family | Stage 1 config | Encoder loss type | Encoder 출력 | Stage 1 ckpt |
|---|---|---|---|---|
| Whisper-tiny | `configs/ASR/stage1_whisper_tiny.yaml` | ASR (sequence-to-text) | 384-d continuous | `external/ckpts/Qwen3.5_whisper_tiny_Stage1` |
| Whisper-small | `configs/ASR/stage1_whisper_small.yaml` | ASR (sequence-to-text) | 768-d continuous | `external/ckpts/Qwen3.5_whisper_small_Stage1` |
| EnCodec-24k | `configs/ASR/stage1_encodec_24k.yaml` | reconstruction (pre-RVQ tap) | 128-d continuous @ 75 fps | `external/ckpts/Qwen3.5_encodec_24k_Stage1` |
| WavTokenizer-40 | `configs/ASR/stage1_wavtok_40_unify.yaml` | reconstruction (pre-VQ tap) | 512-d continuous @ 40 fps | `external/ckpts/Qwen3.5_wavtok_40_unify_Stage1` |

> Stage 2 학습 후에는 동일 분석을 ckpt 만 갈아끼워 재실행한다. 본 문서의 분석은 **Stage 1 ckpt** 에서 수행 (projector 만 학습된 상태, encoder/LLM frozen).

### 2.2 추출 layer

family 당 **1 (encoder out) + 4 (projector L1–L4) + 32 (LLM L1–L32) = 37 layers**.

| Stage | Layer 정의 | 차원 | 비고 |
|---|---|---|---|
| Encoder | `z_e` (Whisper) / pre-quantization latent (EnCodec, WavTok) | family 마다 다름 (384/768/128/512) | projector 입력 직전 |
| Projector | `LlamaDecoderLayer` × 4 의 hidden_states | 512 (4 family 공통) | causal stack, RoPE 사용 |
| LLM | Qwen3.5-4B `transformer.layers[i]` 의 hidden_states (i = 0..31) | 2560 (4 family 공통) | audio span 위치만 추출 |

> 차원이 stage·family 마다 다르므로 metric 은 **차원 비의존적** 형태만 비교 (cosine sim, normalized rank, probe accuracy). Effective rank 류는 `r_eff / d` 로 정규화 보고.

### 2.3 LLM 입력 형식

LLM hidden 추출 시 **audio-only 입력** 으로 통일. 즉 학습 시의 prompt/transcript 토큰을 붙이지 않고 audio embedding 시퀀스만 LLM 에 forward 시킨다 (template 의 `<audio_start>` / `<audio_end>` BOS-류 토큰만 붙음). 이렇게 하면:

- 모든 utterance 가 동일 길이 prefix/suffix 만 가져 비교가 깔끔하다.
- 학습 시의 specific prompt 가 LLM hidden 에 미친 영향을 분석 결과에서 분리할 수 있다.
- audio span hidden states 는 audio_embeds 가 들어간 token position 들로 정의.

---

## 3. 데이터셋 설계

핵심 원칙: **같은 transcript × 다른 speaker** 그리드 데이터로 acoustic / semantic 축을 직교화한다. 가능하면 **같은 화자 × 다른 transcript** 축도 함께 갖춰 4-cell ablation matrix 를 구성한다.

### Tier 1 — 우선 진행

**CMU-ARCTIC** (single-tier baseline)
- 7~18 명의 화자가 1,132 개 phonetically balanced 문장을 모두 평행 녹음.
- 깨끗한 스튜디오 녹음, prosody/감정 변이 적음 → 순수한 화자 변이만 분리됨.
- 분석 baseline 으로 가장 깔끔. 이 데이터에서 결론이 안 나오면 다른 데이터에서도 안 나올 가능성 높음.

### Tier 2 — Acoustic 변이 확장

**VCTK** (스케일 + accent 변이)
- 110 명 화자, Rainbow Passage + Harvard sentences 가 평행 부분.
- 화자 N 이 큼 → 통계적 안정성 ↑.
- accent 변이가 추가되어 acoustic 축이 확장됨.

**DAPS** (channel / environment 변이)
- 20 명 화자가 5 개 장문 passage 를 clean (스튜디오) + device-produced (iPhone, ipad, conference room 등) 으로 녹음.
- 같은 화자 × 같은 transcript × 다른 acoustic condition → recording channel 변이를 isolation 해서 볼 수 있음.
- "acoustic encoder 가 semantic 외 어떤 acoustic 정보를 보존하느냐" 를 보는 데 가장 직접적.

**EXPRESSO** (prosody / style 변이)
- 4 명 화자가 동일 텍스트를 다양한 style/emotion 으로 읽음.
- 같은 화자 × 같은 transcript × 다른 prosody → prosody 축 isolation.
- speaker 와 prosody 를 분리해서 보고 싶을 때.

### Tier 3 — 추가 후보 (선택)

| 데이터셋 | 장점 |
|---|---|
| TIMIT (SA1/SA2) | 630 화자가 동일 2 문장 → 화자 분산 통계 N 큼 |
| L2-ARCTIC | ARCTIC 평행 + L2 (non-native) accent 변이 |
| EARS | 107 화자, reading + emotional, 고품질 최신 |
| Common Voice (filtered) | 같은 sentence id 의 다중 녹음, 매우 큰 N (필터링 비용 있음) |

> Tier 1 → Tier 2 → 필요 시 Tier 3 순서로 확장한다.

---

## 4. Analysis (A) — Disentanglement Metrics

### 4.1 Notation

- $e_{t,s} \in \mathbb{R}^d$ : transcript $t$, speaker $s$ 의 utterance 를 encoder $f$ 로 임베딩 후 시간축 평균 풀링한 벡터.
- $\mathcal{T}$ : transcript 집합, $\mathcal{S}$ : speaker 집합.
- $\text{sim}(\cdot, \cdot)$ : cosine similarity (L2-normalize 후 내적).

### 4.2 Pairwise similarity

| 양 | 정의 | 직관 |
|---|---|---|
| **Content invariance** | $C = \mathbb{E}_{t, s_1 \neq s_2}\, \text{sim}(e_{t,s_1}, e_{t,s_2})$ | 같은 내용, 다른 화자 끼리 얼마나 가까운가 |
| **Speaker invariance** | $S = \mathbb{E}_{s, t_1 \neq t_2}\, \text{sim}(e_{t_1,s}, e_{t_2,s})$ | 같은 화자, 다른 내용 끼리 얼마나 가까운가 |
| **Cross** | $X = \mathbb{E}_{t_1 \neq t_2, s_1 \neq s_2}\, \text{sim}(e_{t_1,s_1}, e_{t_2,s_2})$ | 둘 다 다를 때 baseline |
| **Content-leaning Δ** | $\Delta_C = C - X$ | 커질수록 semantic 축이 분리되어 있음 |
| **Speaker-leaning Δ** | $\Delta_S = S - X$ | 커질수록 화자 축이 분리되어 있음 |

가설 (A) 의 예측 (encoder 출력 기준):
- **Whisper 계열**: $\Delta_C \gg \Delta_S$ (content 만 분리)
- **EnCodec / WavTok 계열**: $\Delta_C, \Delta_S$ 모두 유의미하게 양수 (둘 다 분리, 즉 disentangled)

Layer 진행에 따른 추가 예측:
- 모든 family 에서 LLM 깊은 layer 로 갈수록 $\Delta_S \to 0$ 경향 (LLM 은 transcript 예측에 unrelated 한 화자 정보를 점차 제거할 가능성).
- 다만 *어느 layer 까지 $\Delta_S$ 가 유지되느냐* 가 family 별로 다를 것 — Acoustic 계열이 더 깊은 layer 까지 유지되면 가설 강화.

### 4.3 Linear probing

같은 임베딩 위에 두 개의 logistic regression probe 를 학습.

- **Content probe**: 임베딩 → transcript-id 분류 (Top-1 accuracy)
- **Speaker probe**: 임베딩 → speaker-id 분류 (Top-1 accuracy)

가설 (A) 의 예측:
- Whisper: content probe 높음, speaker probe 낮음
- EnCodec / WavTok: 둘 다 높음

> Probe 는 train/test split 을 transcript 기준이 아닌 utterance 기준으로 split (같은 transcript 의 다른 utterance 가 train/test 양쪽에 존재해도 무방, speaker probe 는 화자 기준 hold-out 도 함께 보고).

### 4.4 Disentanglement score

[**MIG (Mutual Information Gap)**](https://arxiv.org/abs/1802.05983) 류는 임베딩 차원이 명확한 latent factor 에 매핑된다는 가정이 강하므로, 본 분석에서는 좀 더 가벼운 두 지표를 사용한다.

- **Probe gap**: `content_acc − speaker_acc`. 양수면 semantic 편향, 0 근처면 disentangled.
- **Subspace separation** (CKA-based): 임베딩 행렬을 content-grouped block 과 speaker-grouped block 으로 정렬한 뒤, 각 그룹 내부 평균 vs 그룹 간 분산 비를 계산. (Fisher-discriminant analog)

---

## 5. Analysis (B) — Spectrum / Capacity Metrics

(A) 결론 도출 후 같은 임베딩 셋에 대해 측정.

| Metric | 정의 | 비고 |
|---|---|---|
| **Effective rank (entropy)** | $r_\text{eff}(\Sigma) = \exp\bigl(-\sum_i p_i \log p_i\bigr),\ p_i = \sigma_i / \sum_j \sigma_j$ | spectrum flatness; rank 가 높을수록 dimension 이 골고루 사용됨 |
| **RankMe** [Garrido+2023] | 위와 동일하지만 self-supervised 평가용으로 제안된 형식 | 보고 시 차원으로 정규화 (`r_eff / d`) |
| **Participation ratio** | $\text{PR}(\Sigma) = \dfrac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}$ | low-rank 일수록 작아짐 |
| **α-ReQ** [Ghosh+2022] | Power-law fit $\sigma_i \propto i^{-\alpha}$ 의 $\alpha$ | scaling exponent; 작을수록 풍부 |

가설 (B) 의 예측: Acoustic encoder 의 effective rank > Semantic encoder. 단, Whisper 의 차원 (384/768) 이 EnCodec (128) 보다 크므로 raw rank 가 아닌 정규화 형태로 비교한다.

---

## 6. Pipeline

```
1) Embedding extraction (layer-wise)
   - 입력: (audio_path, transcript_id, speaker_id)
   - 각 ALM family 별로 ckpt 로드 후 audio 만 forward (LLM 입력 형식: §2.3 audio-only)
   - 추출 layer (family 당 37 개):
     * encoder out                : (B, T_enc, d_enc)
     * projector L1, L2, L3, L4   : (B, T_enc, 512)
     * LLM L1, ..., L32           : (B, T_audio_span, 2560)
       - LLM hidden 은 audio_embeds 가 들어간 token position 만 slicing
   - 30s 미만 utterance 만 사용 (Whisper 30s padding 영향 통제)
   - frame-level tensor 와 풀링된 utterance-level vector 모두 저장
     * frame: float16 (B, T, d)  → spectrum (B) 분석용
     * utt-mean: float32 (B, d)  → invariance/probe (A) 분석용
     * utt-last: float32 (B, d)  → causal stack ablation 용 (projector & LLM 만)
   - padding mask 동봉 (특히 Whisper)

2) Pairwise similarity (Analysis A)
   - utterance 별 (transcript_id, speaker_id) 메타와 함께 sim matrix 산출
   - Content / Speaker / Cross invariance 계산
   - 모든 layer × 모든 family × {mean, last} 풀링 으로 격자 보고

3) Linear probe (Analysis A)
   - 80/20 utterance split, sklearn LogisticRegression (L2)
   - speaker probe 는 화자 hold-out split 추가 보고 (OOD 화자에서도 동작?)
   - layer × family 격자, mean 풀링 메인 / last 풀링 ablation

4) Spectrum (Analysis B)
   - frame-level tensor 를 utterance 축으로 concat → (Σ_b T_b, d) 행렬
   - centering 후 SVD → spectrum
   - effective rank (entropy form), PR, α 계산. r_eff/d 정규화 보고
   - utterance-mean 행렬 (N_utt, d) 의 spectrum 도 별도 측정 (둘이 다른 양임을 명시)

5) Stage 2 ASR WER 와의 correlation (선택, Stage 2 ckpt 갱신 후)
   - 4 family 의 metric 값 (Δ_C, Δ_S, probe gap, r_eff) 을 x 축, downstream WER 을 y 축
   - 4 점 회귀라 통계 검증은 제한적이지만 *방향성* 은 보임
```

각 단계별 산출물은 `experiments/representation_richness/{dataset}/{family}/{layer}/...` 아래에 저장.

---

## 7. 풀링 전략

Family·layer 간 공정 비교에 가장 민감한 부분. **Mean over valid frames 가 메인**, **last frame 이 ablation**.

### 7.1 결정된 사항

- **Main pooling — Mean over valid frames** (모든 stage 공통 적용):
  - Encoder: padding mask 로 valid frames 만 평균. Whisper 의 30s zero-pad 는 mask 처리 필수.
  - Projector / LLM: causal stack 이므로 valid audio span 의 frame 만 평균. text/special token 위치 제외.
- **Ablation pooling — Last valid frame** (projector & LLM 에 한정):
  - Causal stack 은 마지막 frame 에 정보가 응축되는 경향이 있음. *모델이 실제로 사용하는* 표현 관점.
  - Encoder 단계는 non-causal (Whisper, EnCodec, WavTok 모두 양방향) 이므로 last frame ablation 비적용.
- **LLM 입력 형식 — Audio-only** (§2.3):
  - 학습 시 prompt/transcript 를 빼고 audio embedding 시퀀스만 LLM 에 forward.
  - audio span hidden states 만 풀링 대상으로 사용.
- **Normalization**:
  - cosine sim / probe: L2-normalize 후 사용.
  - spectrum (Analysis B): centering 만 적용, normalize 안 함.

### 7.2 도입하지 않는 것

- CLS-style 토큰, attention pooling — family 마다 사전학습 정의가 달라 공정성 깨짐.
- DTW alignment — 평행 utterance 비교의 정확도는 올라가나 비용·해석 복잡도 ↑. 후속 ablation 후보로만 둠.
- Multi-segment concat / multi-stat (mean+std+max) — 차원 늘어남 + 해석 부담. 지금 단계에서 도입 X.

---

## 8. 열린 질문 / 결정 필요

1. **EXPRESSO 의 prosody 축**: speaker 와 별도 factor 로 취급할지, acoustic 변이의 일부로 묶을지.
2. **Common Voice 도입 여부**: scale 은 매력적이나 sentence id 별 화자 수가 들쭉날쭉. Tier 1/2 결과 본 후 결정.
3. **Stage 2 ckpt 분석 시점**: Stage 2 학습이 LLM/projector 에 미치는 영향을 ablation 으로 보고 싶다면, 같은 dataset 에서 Stage 1 vs Stage 2 ckpt 비교 plot 를 추가.
4. **Mid-layer ablation**: 본 분석은 모든 layer 를 보지만, 만약 시간 부족 시 우선순위는 (encoder out, projector L4, LLM L0/L8/L16/L24/L31) 정도로 sparse subset.

---

## 9. 작업 순서 (running plan)

Phase 1 (CMU-ARCTIC 7 화자) 의 실행 plan 은 별도 문서로 분리:

- [Plan 1 — Download](plans/plan_repr_richness_1_download.md)
- [Plan 2 — Extraction](plans/plan_repr_richness_2_extraction.md)
- [Plan 3 — Analysis](plans/plan_repr_richness_3_analysis.md)

이후 후속 phase:

4. VCTK + DAPS 임베딩 → Analysis A 재현 (Plan 4a, 별도 문서)
5. EXPRESSO → prosody 축 추가 (Plan 4b)
6. Stage 2 ckpt 학습 완료 후 동일 분석 재실행 → §8.3 비교 plot (Plan 4c)
7. Stage 2 WER correlation 정리
