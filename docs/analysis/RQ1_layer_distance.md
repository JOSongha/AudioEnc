# RQ1 — Layer-wise Representation Distance across Audio-LLM Family

> Stage 1 학습된 4 개 ALM family (Whisper-tiny, Whisper-small, DACVAE, WavTokenizer-40) 의 표현이 **layer 진행에 따라 어떻게 변화하는지**, 그리고 **family 간 동일 위치 layer 가 얼마나 다른지** 를 IEMOCAP 4-class 데이터에 대해 CKA 로 측정한다.

본 분석은 [Representation Richness 분석](analysis_representation_richness.md) 과 동일 ALM stack · 동일 추출 파이프라인을 공유한다. 차이는 데이터셋 (IEMOCAP) 과 메트릭 (CKA) 뿐.

---

## 1. Hypothesis

두 가지 *축* 의 가설을 동시에 검증한다.

### (1) Within-ALM 축 — pipeline 내부 representation drift

가설:
- **encoder → projector 경계** 에서 representation 이 가장 급격히 변한다 (서로 다른 학습 목표를 거치는 첫 지점).
- projector 4 layer 내부는 점진적이며, projector → LLM 첫 layer 는 다시 한 번 큰 점프 (LLM token-space 로의 정렬).
- LLM 후반 layer 에서는 인접 layer 끼리 CKA 가 높아지며 plateau (residual stack 의 전형적 패턴).

> **시각화**: 모델별 42×42 CKA heatmap. 위 가설은 두 개의 어두운 “경계선” (enc/proj 사이, proj/llm 사이) + 우하단 LLM 영역의 매끈한 그라데이션으로 나타날 것.

### (2) Between-ALM 축 — family 간 동일 position 차이

가설 (Acoustic vs Semantic 대비 — Repr Richness §1(A) 와 일관):
- **encoder out**: 가장 큰 family 차이 (학습 목표가 직접 작용한 지점). Whisper × Whisper 는 높고, Whisper × DACVAE/WavTok 은 낮을 것.
- **projector 통과 후**: 차이가 줄어듦 (모든 family 가 같은 LLM 의 token-space 에 맞춰지는 압력).
- **LLM 깊은 layer**: family 차이가 가장 작아지며 모두 “Qwen3.5 audio-input 표현” 으로 수렴. 단 *어느 layer 에서 수렴하느냐* 가 family 별로 다를 것.

> **시각화**: position × (family-pair) heatmap (가로축 42 position, 세로축 6 pair). 위 가설은 좌측 (enc) 이 어둡고 우측 (LLM 후반) 으로 갈수록 밝아지는 그라데이션.

> 가설이 빗나가면 (예: LLM 후반에서도 family 차이가 유지) → ALM 이 family-specific signature 를 끝까지 보존한다는 강한 증거가 됨.

---

## 2. 비교 대상 (ALM Stack, Layer-wise)

비교 단위는 encoder 가 아니라 **ALM (encoder + projector + LLM) 의 각 layer 임베딩**. 4 family 모두 RVQ/VQ 를 통과시키지 않고 encoder 의 **continuous latent** 를 projector 입력으로 쓴다.

### 2.1 Family 구성

모든 family 는 동일 LLM (Qwen3.5-4B, 32 layers, 2560-d) 과 동일 projector (4 Llama layers, 512-d) 를 공유한다. 차이는 encoder 와 그 출력 차원 / fps.

| Family | Encoder loss | Encoder 출력 | Stage 1 ckpt |
|---|---|---|---|
| Whisper-tiny | ASR (seq2text) | 384-d @ 50 fps | `external/ckpts/Qwen3.5_whisper_tiny_Stage1/.../checkpoint-13000` |
| Whisper-small | ASR (seq2text) | 768-d @ 50 fps | `external/ckpts/Qwen3.5_whisper_small_Stage1/.../checkpoint-13000` |
| DACVAE | reconstruction (post-VAE z) | 128-d @ 25 fps | `external/ckpts/Qwen3.5AE-4B-dacvae_ASR-Stage1` |
| WavTokenizer-40 | reconstruction (pre-VQ z_e) | 512-d @ 40 fps | `external/models/Qwen3.5AE-4B-wavtok-40-unify` |

### 2.2 추출 layer (family 당 15 개, 5 + 5 + 5 대칭)

| Stage | Layer 수 | Layer 정의 | 차원 |
|---|---|---|---|
| Encoder | 5 | family 별 §2.3 (균등 샘플) | family 마다 다름 |
| Projector | 5 | `proj_0..3` (LlamaDecoderLayer × 4 의 hidden_states) + `proj_out` (output_proj 출력, LLM 입력 직전) | 512 / 2560 |
| LLM | 5 | `llm_0, llm_8, llm_15, llm_23, llm_31` (32 layer 균등 5 점 샘플), audio_pad 위치만 | 2560 |

**설계 메모**:
- 인접 LLM layer 끼리 CKA 가 0.9+ 로 redundant 한 점을 반영해 33 → 5 점으로 subsample. 5 점이면 enc / proj / llm 모두 동일 해상도로 비교 가능.
- `llm_embed` 는 제외 — `inputs_embeds=audio_embeds` 로 forward 하면 `hidden_states[0] == audio_embeds == proj_out` 이므로 중복.
- `proj_out` 은 projector 의 `output_proj` (Linear 512→2560) 통과 후 값. `proj_3` (512-d) 와 `proj_out` (2560-d) 사이의 변화가 "projector → LLM token-space" 의 마지막 변환을 보여줌.
- CKA 가 차원-비의존이라 family 간 enc 차원 차이는 문제 안 됨.
- Heatmap 크기: 15×15 (가독성 ↑). Pair 수: 15·16/2 = 120 per (family, pooling, emotion).

### 2.3 Encoder 5-layer 매핑

원칙:
- **`enc_0`** = encoder 첫 stage 출력 (early representation).
- **`enc_4`** = **각 family 의 projector 가 실제로 받는 텐서** (= ALM 의 정의된 encoder output).
- `enc_1` / `enc_2` / `enc_3` = 그 사이 균등 샘플.

| 포지션 | whisper_tiny | whisper_small | dacvae | wavtok |
|--------|-------------|---------------|--------|--------|
| `enc_0` (early) | embed (384-d)        | embed (768-d)        | conv_in (64-d)             | conv_in (32-d) |
| `enc_1`         | hs[1] after L0       | hs[3] after L2       | block_0 stride=2 (128-d)   | stage_0 stride=2 (64-d) |
| `enc_2` (mid)   | hs[2] after L1       | hs[6] after L5       | block_1 stride=8 (256-d)   | stage_1 stride=4 (128-d) |
| `enc_3`         | hs[3] after L2       | hs[9] after L8       | block_3 stride=12 (1024-d) | stage_3 stride=8 (512-d) |
| `enc_4` (→proj) | **last_hidden_state** (384-d) | **last_hidden_state** (768-d) | **post-VAE z** (128-d) | **conv_out** (512-d) |

각 family 별 `enc_4` 정의:
- **whisper**: `WhisperEncoder` 의 `last_hidden_state` = `layer_norm(hidden_states[-1])`. 단순 마지막 transformer layer 출력이 아니라 **final layer norm 까지 적용된 값** 이 projector 에 들어감.
- **dacvae**: `DACVAE.encode()` 의 return = **VAE 샘플링 거친 stochastic z** (`mean + softplus(scale)·ε`). 분석 시 `ε` seed fix 필수 (또는 `mean` 만 deterministic 으로 ablation).
- **wavtok**: SEANet `self.model(x)` 의 마지막 element = `conv_out` (final `SConv1d(512→512, k=7)`). stage_3 다음에 SLSTM + ELU + conv_out 이 더 있음에 주의.

추출 노트:
- whisper enc 는 normal forward 에서 `last_hidden_state` 만 사용하므로, 중간 layer (hs[1..]) 는 `output_hidden_states=True` 로 별도 호출.
- dacvae stochastic 처리: utterance 마다 `torch.manual_seed(stable_hash(utt_id))` 호출 후 forward → utterance 의 z 가 처리 순서 / subset / family 와 무관하게 항상 동일. `stable_hash` 는 결정론적 (e.g., `int.from_bytes(hashlib.sha1(utt_id.encode()).digest()[:4], 'big')`) 사용 (Python 의 `hash()` 는 PYTHONHASHSEED 영향 받으므로 금지).

### 2.4 LLM 입력 형식 — Inference 표준 ChatML 사용

**inference 와 동일한 ChatML prompt** 를 사용한다 (Repr Richness §2.3 의 audio-only 와는 다름).

근거: `evaluation/stage2/_loader.py:431` 의 `build_prompt_ids` 와 `evaluation/stage2/eval_iemocap_session5.py` 의 IEMOCAP eval prompt 를 그대로 따른다.

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|>[audio_pad]*t_audio<|audio_end|>What is the emotion expressed?
A. angry
B. happy
C. neutral
D. sad
Answer with the letter.<|im_end|>
<|im_start|>assistant
```

audio_embeds 주입 위치 (`modeling_qwen3_5AE.py:1050`):
```python
audio_mask = input_ids == self.config.audio_pad_token_id
inputs_embeds[audio_mask] = audio_embeds[valid_audio_mask]
```

LLM hidden 풀링은 **`audio_pad_id` 위치 (T_audio 개) 만** 대상 — `<|audio_start|>`, `<|audio_end|>`, system / user / suffix 토큰은 모두 제외.

> **Causal masking 의 영향**: LLM 이 causal 이므로 audio span hidden state 는 system + user prefix + `<|audio_start|>` 만 attend 한다. suffix question (`What is the emotion...`) 은 audio 뒤에 위치하므로 audio span hidden 에 직접 영향 없음. 즉 audio 표현은 prompt prefix 에만 conditioned.

---

## 3. 데이터셋 — IEMOCAP 4-class

| 항목 | 값 |
|---|---|
| Manifest | `experiments/audio_encoder_probe/manifests/iemocap_4class.csv` |
| Class | angry / happy / sad / neutral |
| 화자 | 10 명 (Session 1–5, M+F) |
| 전체 발화 수 | 6,877 (happy 2632 / neutral 1726 / angry 1269 / sad 1250) |
| 길이 | 평균 4.5 s, 최대 30 s |

### 3.1 균형 샘플링

**Class 당 500 utterance 균등 샘플링 → 총 2,000 utterance** 를 분석에 사용.

이유:
- class 불균형 (happy 2632 vs sad 1250) 으로 인한 CKA bias 제거
- emotion-conditioned 분석 (§8.1) 시 class 마다 동일 N 으로 비교 가능
- CKA 는 N=500 이면 충분히 안정 (gram matrix 4 MB, 모든 분석 빠르게)

샘플링 규칙:
- seed = 42 로 fix (재현성)
- **같은 2000 utterance set 을 4 family 모두 공통 사용** (Axis 2 의 family-pair CKA 가 의미를 가지려면 동일 utterance 에 대한 임베딩이어야 함)
- 모든 layer · 모든 pooling 도 동일 set
- emotion-conditioned 분석 (§8.1) 시 이 set 안에서 emotion 별 500 utterance 부분집합 사용 (즉 emotion subset 도 family 간 공통)
- 샘플링은 manifest 단계에서 1 회 수행 → `iemocap_4class_balanced500.csv` 로 고정 → 모든 추출 스크립트가 이 manifest 만 읽음

> **중요**: family 마다 다른 utterance 를 쓰면 Axis 2 cross-family CKA 가 의미 없어진다 (CKA 는 동일 N 개 sample 에 대한 두 representation matrix 의 alignment 측정). 샘플링 분리는 절대 금지.

> CKA 계산은 utterance 단위 풀링 vector (N=2000 또는 emotion 별 N=500, D) 를 쓴다.

### 3.2 Audio length / padding 처리

**Padding mask 필수** — utterance 길이가 다르므로 batch 처리 시 zero-pad 가 들어가고, mean pool 시 valid frame 만 평균해야 함.

| 위치 | valid frame 정의 |
|---|---|
| Whisper mel input | feature_extractor 가 30 s 까지 zero-pad → valid mel = `min(audio_samples / 320, 1500)` (50 fps) |
| Whisper enc / DACVAE / WavTok encoder 출력 | `valid_t = ceil(audio_samples / hop_length)` |
| projector 입력 | encoder 출력의 valid_t 그대로 |
| LLM audio span | `t_audio` = projector 출력 frame 수 = encoder valid_t (audio_pad_id 토큰 수와 동일) |

**fps 차이는 inherent — last frame 은 그대로 last 만 사용**:
| Family | fps | 5 s utt → token 수 |
|---|---|---|
| Whisper | 50 | 250 |
| DACVAE | 25 | 125 |
| WavTok | 40 | 200 |

family 마다 마지막 frame 이 표현하는 시간 granularity 가 다르지만 (20 ms vs 40 ms), 분석에서는 보정하지 않고 **그대로 last 사용**. inference 의 last frame 이 실제로 무엇이든 그대로 본다는 의도.

**Min length filter**:
- 0.5 s 미만 utterance 는 DACVAE 25 fps 기준 ≤ 12 frame → noisy → 제외
- 균형 샘플링 (§3.1) 시 사전에 1 s 이상만 후보로 제한

---

## 4. Metric — CKA

차원이 family·layer 마다 다르므로 **차원-비의존** 인 CKA 를 메인으로 쓴다.

### 4.1 Notation

- $X \in \mathbb{R}^{N \times d_X}$: layer $X$ 에서 추출한 N 개 utterance embedding (mean-pooled).
- $K = X X^\top \in \mathbb{R}^{N \times N}$: linear gram matrix.
- $H = I_N - \tfrac{1}{N}\mathbf{1}\mathbf{1}^\top$: centering matrix.

### 4.2 Linear CKA (메인)

$$
\mathrm{HSIC}(K, L) = \frac{1}{(N-1)^2} \mathrm{tr}(KHLH)
$$

$$
\mathrm{CKA}(X, Y) = \frac{\mathrm{HSIC}(K_X, K_Y)}{\sqrt{\mathrm{HSIC}(K_X, K_X)\, \mathrm{HSIC}(K_Y, K_Y)}}
$$

Linear CKA 는 X, Y 의 차원이 달라도 계산 가능. 값은 $[0, 1]$.

### 4.3 RBF CKA (ablation)

비선형 구조까지 보고 싶을 때. RBF kernel $k(x, y) = \exp(-\|x-y\|^2 / 2\sigma^2)$, $\sigma$ 는 median heuristic. 메인 결론은 linear CKA 로 보고하고 RBF 는 부록으로 첨부.

### 4.4 구현 디테일

- **Centering 필수**: `K_c = H K H` 또는 `X_c = X - X.mean(0); K = X_c X_c^T`. 안 하면 결과 무의미.
- **Numerical stability**: bf16 입력은 CKA 계산 전 float32 cast. 분모가 0 근처면 NaN — degenerate case (e.g. all-zero feature) 방어.
- **Gram matrix caching**: layer 별 K_X 계산은 1 회만, 이후 모든 pair 의 HSIC 에서 재사용. 캐싱 안 하면 ~60 배 느려짐.
- **계산량 견적**:
  - Pair 수: 15×16/2 = 120 (Axis 1) + 6×15 = 90 (Axis 2) = 210 per (family, pooling, emotion)
  - 총 ≈ 4 × 2 × 5 × 210 = 8,400 CKA scalar
  - N=2000 gram matrix 의 HSIC ≈ ms 단위 → 전체 1–3 분

### 4.5 왜 CKA 인가 (다른 후보 대비)

| 메트릭 | 차원 무관 | 직관 | 이번 분석에 적합? |
|---|---|---|---|
| Cosine sim of centroids | ❌ (D 같아야) | 같은 dim 내 angular | within-family 만 |
| L2 / Euclidean | ❌ | scale 의존 | 비추 |
| **CKA (linear)** | ✅ | feature-space alignment | ★ 메인 |
| RSA (Spearman of dist matrix) | ✅ | rank-based geometry | sub-ablation |

---

## 5. 두 분석 축

### 5.1 Axis 1 — Within-ALM (family 1 개 × layer × layer)

family 별 **15 × 15 CKA matrix**.

```
rows / cols : [enc_0..4, proj_0..3, proj_out, llm_0, llm_8, llm_15, llm_23, llm_31]
cell (i, j) : CKA(X_i, X_j)
```

산출물:
- 4 family × 1 heatmap = 4 heatmap (각 15×15)
- enc / proj / llm 경계선 overlay (3-그룹 구분)
- 대각선 인접 ($|i - j| = 1$) 의 평균 CKA → "smoothness" 1-D plot

### 5.2 Axis 2 — Between-ALM (family-pair × layer position)

같은 위치 layer 끼리 CKA. family 4 개 → $\binom{4}{2} = 6$ pair.

| Pair | 해석 |
|---|---|
| `whisper_tiny ↔ whisper_small` | 구조 동일, scale 차이 |
| `whisper_tiny ↔ dacvae` | semantic vs acoustic, 작은 모델 |
| `whisper_tiny ↔ wavtok` | semantic vs acoustic |
| `whisper_small ↔ dacvae` | semantic vs acoustic, 큰 semantic |
| `whisper_small ↔ wavtok` | semantic vs acoustic |
| `dacvae ↔ wavtok` | acoustic 끼리 (다른 codec) |

산출물:
- **(6 pair) × (15 position) heatmap** 1 장
- 또는 position 을 x 축으로 한 6 line plot
- enc / proj / llm 경계 vertical line overlay

---

## 6. Pipeline

```
1) Embedding extraction (layer-wise)
   - 입력: §3.1 의 균형 manifest (2000 utt)
   - 각 family ckpt 로드 → §2.4 ChatML prompt 로 forward
   - 추출: 15 layer (5 enc + 5 proj + 5 llm)
     * encoder: family 별 hook 위치 (§2.3)
     * projector: hook on layers[0..3] + output_proj
     * LLM: language_model(inputs_embeds=full_chatml_inputs_embeds, output_hidden_states=True)
                → audio_pad 위치 hidden 만 slicing → 5 점 (llm_0, 8, 15, 23, 31) 만 저장
   - utterance 별 두 풀링 산출 (§7):
       * mean over valid frames (모든 15 layer)
       * last valid frame (proj/llm 10 layer 만, encoder 5 layer 는 non-causal 이라 미적용)
     → (D,) float32 each
   - 저장: embeds_layers/{family}/{utt_id}.npz
            keys: enc_{0..4}_mean,
                  proj_{0..3}_{mean,last}, proj_out_{mean,last},
                  llm_{0,8,15,23,31}_{mean,last}

2) (N, D) 행렬 빌드
   - family · layer 마다 모든 utterance 를 stack → (N, D_layer)
   - centering 후 저장 (재계산 비용 절감)

3) CKA 계산 (mean / last × overall / emotion-conditioned)
   - 입력 set: §3.1 의 2000 utterance (class 별 500)
   - Axis 1: family 별 15×15 matrix
       * mean: 15 layer 모두 (N=2000 또는 emotion 별 500)
       * last: enc 5 개는 mean 값으로 채우고 proj/llm 10 개는 last 값
   - Axis 2: 6 pair × 15 position = 90 scalar
   - 각 위 산출을 (overall N=2000) + (4 emotion × N=500) = 5 변형 으로 반복
   - linear CKA 메인, RBF CKA 부록

4) 시각화
   - Axis 1: 4 family × 2 풀링 × 5 (overall + 4 emotion) = 40 heatmap (각 15×15)
       * 본문엔 overall 4 family × 2 풀링 = 8 heatmap, 부록에 emotion 별
   - Axis 2: 2 풀링 × 5 (6×15 heatmap + per-pair line plot)
   - smoothness curve (인접 layer CKA): family overlay 1 plot per (풀링 × emotion)
```

산출물 경로: `experiments/audio_encoder_probe/rq1_layer_distance/`
- `manifests/iemocap_4class_balanced500.csv`
- `embeds_layers/{family}/{utt_id}.npz`
- `cka/{family}_within_{mean,last}_{overall,angry,happy,sad,neutral}.npy`
- `cka/cross_position_{mean,last}_{overall,angry,happy,sad,neutral}.npy`
- `figs/within_{family}_{mean,last}_{...}.png`, `cross_position_{...}.png`, `smoothness_{...}.png`

---

## 7. 풀링 전략

본 분석은 **두 가지 풀링을 모두 메인** 으로 산출한다. 답하려는 질문이 다름.

| 풀링 | 답하는 질문 | 적용 범위 |
|---|---|---|
| **Mean** over valid frames | layer 의 representation 구조는 어떻게 생겼나? (structural) | 15 layer 모두 (5 enc + 5 proj + 5 llm) |
| **Last** valid frame | 모델이 generation 에 실제로 쓰는 표현은? (behavioral) | 10 layer 만 (5 proj + 5 llm) |

이유:
- **Mean** 은 layer 전체의 manifold 를 안정적으로 특성화 → CKA 의 통계적 신뢰성 ↑.
- **Last** 는 causal stack 의 마지막 audio position hidden state. inference 시 첫 generated token 의 logit 이 정확히 이 position 에서 산출됨 → "모델이 실제로 사용하는" 표현.
- Encoder (Whisper / DACVAE / WavTok) 는 모두 **non-causal (양방향)** 이므로 last frame 이 특별한 의미 없음 → mean 만 적용.

실무 처리:
- valid frames 식별: padding mask 로 (Whisper 30 s zero-pad, batch padding 영향 통제).
- LLM `last` = audio span 의 마지막 token position hidden state. (audio_embeds 의 마지막 frame 이 LLM 에 들어간 위치).
- L2-normalize 안 함 (CKA 는 centering 만 사용).

산출물:
- `embeds_layers/{family}/{utt_id}.npz` 에 두 풀링 모두 저장
  - keys: `enc_{0..4}_mean`, `proj_{0..3}_{mean,last}`, `proj_out_{mean,last}`, `llm_{0,8,15,23,31}_{mean,last}`
- Axis 1 / Axis 2 모두 mean / last 두 종류 heatmap 산출.

---

## 8. Sanity check / baseline

분석 시작 전 / 결과 신뢰성 검증용으로 다음을 산출.

| 검증 | 기대값 | 실패 시 의미 |
|---|---|---|
| **CKA 자기-유사도**: `CKA(X_i, X_i)` | = 1.000 (정확히) | CKA 구현 버그 |
| **Random baseline**: 같은 N 개 random Gaussian (N, D) 두 set 의 CKA | 0.01–0.05 | 우리 분석 CKA 의 floor 기준점 |
| **Stage 1 encoder frozen 검증**: ALM 의 whisper enc layer i 표현 vs HuggingFace `whisper-tiny.en` 같은 layer 의 표현 | CKA = 1.0 (encoder frozen 이므로 동일해야 함) | Stage 1 학습 중 encoder 가 의도치 않게 update 됨, 또는 dtype/precision 문제 |
| **DACVAE z 재현성**: 같은 utterance 두 번 forward 했을 때 z (post-VAE) 의 self-CKA | 1.0 (seed fix 후) / < 1.0 (seed fix 안 함) | DACVAE enc_4 의 self-CKA floor 가 됨; seed 정책 검증 |
| **DACVAE / WavTok encoder frozen 검증**: ALM encoder vs 원본 ckpt encoder | CKA = 1.0 | 위와 동일 |

→ 결과는 본문 §부록 A 에 1 페이지로 정리.

---

## 9. Statistical robustness

핵심 결론에만 적용 (전체에 적용하면 비용 큼):

### 9.1 Bootstrap 신뢰구간

- N=2000 utterance 에서 with-replacement 1000 회 resample
- 각 sample 에서 CKA 계산 → 분포 → 5/95 percentile
- 적용 대상: Axis 2 의 핵심 layer (enc_4, proj_3, llm_15, llm_31) × 6 pair × 2 pooling
- → 24 개 결과에 CI 부착

### 9.2 Permutation test

- Null: "두 family 의 layer i 표현은 random 한 alignment 이상의 관련 없음"
- utterance index 를 family A 만 셔플 → CKA 분포 → p-value
- 적용 대상: 위 24 개 결과
- p < 0.05 면 "유의미하게 random 보다 비슷" 으로 보고

---

## 10. Pre-registered 결과 해석 (확증편향 방지)

분석 돌리기 전 결과 패턴별 해석을 미리 정해둔다.

### Within-ALM (Axis 1) heatmap 패턴

| 패턴 | 해석 |
|---|---|
| enc → proj 경계가 sharp (off-diagonal CKA 급락) | projector 가 LLM token-space 로 강한 변환 |
| enc → proj 경계가 부드러움 | projector 가 encoder 표현을 거의 보존 |
| proj → llm 경계가 sharp | LLM 의 첫 layer 가 audio embed 를 textual representation 으로 강하게 재구성 |
| LLM 후반 (llm_24..31) 의 plateau | 표현이 깊이에 따라 수렴 (전형적 transformer pattern) |
| LLM 후반에 다시 하락 | LLM 이 답변 생성을 위한 specific representation 으로 분기 |

### Between-ALM (Axis 2) per-position 패턴

| 패턴 | 해석 (Acoustic vs Semantic 가설 §1) |
|---|---|
| 모든 position 에서 family pair 모두 CKA → 1.0 | family signature 가 LLM 까지 거의 흡수됨 (가설 약화) |
| enc 에서 차이 → llm 후반에서 모두 수렴 | family 가 다른 audio 를 보지만 LLM 이 통일 (메인 가설) |
| enc 에서 차이 → llm 후반에도 유지 | encoder 종류가 pipeline 끝까지 영향 (강한 RQ 결론) |
| Whisper×Whisper > Whisper×Acoustic 모든 position | family 분기 명확 (가설 강화) |

### mean vs last 비교

| 패턴 | 해석 |
|---|---|
| mean ≈ last 모든 곳에서 | causal stack 이 사실상 bidirectional 처럼 작동 (또는 짧은 utterance 효과) |
| 차이 큼 (특히 LLM 후반) | last frame 이 audio summary 로 응축, mean 은 distributional |
| enc 에서 mean ≠ last 가 클 가능성 (encoder 는 non-causal 이라 sanity 위배) | bug — encoder 는 mean 만 적용해야 |

---

## 11. Infrastructure 견적

| 항목 | 견적 |
|---|---|
| **저장**: 2000 utt × 4 family × 15 layer × ~1700 dim avg × 4 byte (float32, mean+last) | ~ 1.6 GB |
| **GPU memory** (extraction): LLM (Qwen3.5-4B, bf16) + 33 hidden states × (1, ~250, 2560) bf16 ≈ 8 GB + 40 MB/utt 일시적 | A100 40 GB 충분, batch=1 권장 (저장은 5 점만) |
| **실행 시간** (extraction): 2000 utt × 0.5–2 s/utt = 30 min – 1 h per family, 4 family 총 2–4 h | 단일 A100 |
| **CKA 계산**: 8,400 scalar × ms 단위 (gram caching 가정) | 1–3 min |
| **시각화**: matplotlib heatmap 산출 | 즉시 |

병렬화 가능 부분: family 4 개를 4 GPU 에 분산 → 시간 ¼.

---

## 12. 열린 질문 / 결정 필요

1. **Emotion-conditioned CKA** → **본 분석에 포함** (overall + 4 emotion 모두 산출).
   - overall: §3.1 의 2000 utterance 전체 (class 균형 보장된 상태)
   - emotion 별: 같은 set 안에서 class 마다 500 utterance 부분집합
   - 산출물: family 4 × pooling 2 × (overall + 4 emotion) 5 = **40 개 within-model heatmap**, axis 2 도 동일 배수
   - 해석 포인트: 같은 layer 라도 emotion 별 representation drift 패턴이 다를 수 있음 (예: arousal 높은 angry/happy 가 enc→proj 에서 변화 적고, neutral 은 큼).
2. ~~Stage 1 vs Stage 2 ckpt 비교~~ → **본 분석은 Stage 1 ckpt 만 다룬다.** Stage 2 비교는 후속 작업 범위 밖.
3. **frame-level RSA**: utterance 평균 대신 frame-level (T_b, D) 행렬에 대한 RSA. 비용 크지만 temporal granularity 정보 보존.
4. ~~DACVAE post-VAE z 의 stochasticity 처리~~ → **결정됨** (§2.3):
   - enc_4 = post-VAE z (projector 입력).
   - seed = `stable_hash(utt_id)` per-utterance fix (Option B). 처리 순서 / subset / family 와 무관하게 동일 utterance 의 z 가 항상 동일.
   - `stable_hash` = `int.from_bytes(hashlib.sha1(utt_id.encode()).digest()[:4], 'big')` (Python `hash()` 는 `PYTHONHASHSEED` 영향으로 사용 금지).
   - mean-deterministic ablation (`ε=0`) 은 v1 산출물에서 제외, 필요시 후속.
5. **Family 확장 (EnCodec-24k)**: 동일 인프라에 `external/ckpts/Qwen3.5_encodec_24k_Stage1` 가 있어 5 family 까지 확장 가능.
   - acoustic family 가 2 → 3 으로 늘어 acoustic-vs-acoustic CKA (DACVAE × WavTok × EnCodec) 비교 가능
   - Axis 2 pair 수: 6 → 10 (산출물 1.7 배)
   - 본 RQ1 v1 에서는 4 family 로 시작, EnCodec 은 v2 후속
6. **enc 차원 mismatch (Axis 2)**: enc_0 의 dim 이 family 별로 32–768 로 매우 다름. CKA 가 dim-invariant 라 계산은 가능하지만 "같은 position" 의 의미는 proj/llm (같은 dim) 보다 약함. 결과 보고 시 명시.
7. **CKA 외 보조 지표**: Axis 1 의 layer drift 를 1-D 로 요약하는 "smoothness curve" (인접 layer 평균 CKA) 외에, layer-wise probe accuracy (emotion 4-class) 도 같이 보고할지. 후속 RQ2 와 겹치므로 RQ1 에서는 제외.

---

## 13. Implementation plan

협업 원칙 (점진적 진행 / TDD / 단순성 / 기존 코드 학습) 에 맞춰 5 단계로 분할.
각 단계는 **목표 → 산출물 → 성공 기준 → 테스트 케이스 → 의존성** 명세 후 진행.
단계 종료 시 점진 커밋. 단계 간 컴파일 / smoke test / lint 통과 유지.

### Stage 0 — 기존 코드 패턴 학습 (구현 전)

**목표**: 4 가지 유사 기능 (audio loading, encoder forward, layer extraction, manifest 처리) 의 기존 구현을 분석해 재사용 패턴 결정.

| 분석 대상 | 위치 | 재사용 포인트 |
|---|---|---|
| Audio dataset / collate | `experiments/audio_encoder_probe/extract_encoder_only.py` `AudioDataset`, `_collate` | sampling rate / hop_length / max_samples 처리, batch padding |
| Encoder loading / forward | 같은 파일 `load_encoder`, `encode_batch` | family 별 model 경로 / dtype / hook 위치 |
| ChatML prompt 빌드 | `evaluation/stage2/_loader.py` `build_prompt_ids`, `left_pad_batch` | audio_pad 위치 / suffix / padding |
| IEMOCAP eval flow | `evaluation/stage2/eval_iemocap_session5.py` | label map / audio_pad_token_id / generate flow |

**산출물**: 본 §13 의 각 stage 가 어떤 함수를 import/재사용하는지 명시 (아래에 표기).

**성공 기준**: 새 코드에서 위 4 가지 기능을 zero-duplication 으로 호출. 새 abstraction 도입 시 명시적 정당화.

---

### Stage 1 — Manifest 균형 샘플링

**목표**: §3.1 의 균형 manifest (`iemocap_4class_balanced500.csv`) 생성 + audio_path 를 새 위치로 갱신.

**산출물**:
- `experiments/audio_encoder_probe/manifests/iemocap_4class_balanced500.csv` (2000 행)
- 스크립트: `experiments/audio_encoder_probe/build_balanced_manifest.py` (≤ 50 줄)

**성공 기준**:
- class 별 정확히 500 utterance (angry / happy / sad / neutral)
- 1 s 미만 utterance 제외 (§3.2 min length filter)
- audio_path = `/mnt/tmp/datasets/emotion_raw/IEMOCAP/data/{utt_id}.wav` (모든 파일 존재 확인)
- seed=42 fix → 재실행 시 동일 결과

**테스트 케이스** (`tests/test_balanced_manifest.py`):
1. `len(df) == 2000`
2. `df['emotion'].value_counts()` 모든 class = 500
3. `df['duration_s'].min() >= 1.0`
4. 모든 audio_path 가 실제 존재 (`Path.exists()`)
5. seed=42 두 번 실행 시 utt_id 집합 동일 (결정론)

**재사용**: 원본 `iemocap_4class.csv` 의 컬럼 그대로 유지.

**예상 소요**: 30 분.

---

### Stage 2 — Layer-wise embedding extraction

**목표**: 4 family × 2000 utt × 15 layer (mean+last) 임베딩 추출.

**산출물**:
- `experiments/audio_encoder_probe/extract_layers.py` (≤ 300 줄)
- 출력: `experiments/audio_encoder_probe/embeds_layers/{family}/{utt_id}.npz`
  - keys: `enc_{0..4}_mean`, `proj_{0..3}_{mean,last}`, `proj_out_{mean,last}`, `llm_{0,8,15,23,31}_{mean,last}`
- LLM 입력: §2.4 의 ChatML prompt (`build_prompt_ids` 재사용)
- LLM forward 는 `output_hidden_states=True` 로 33 hidden 모두 받지만, 저장은 5 점 (0, 8, 15, 23, 31) 만

**성공 기준**:
- 모든 family / utterance 에서 NaN / Inf 없음
- npz 파일 크기 일정 (utterance 마다 layer 수 동일)
- 실행 중 GPU OOM / 모델 로드 실패 없음
- 동일 utterance 두 번 실행 시 (DACVAE 제외) bit-exact 동일

**테스트 케이스** (`tests/test_extract_layers.py` — smoke test 5 utt):
1. 단일 family / 단일 utterance 추출 시 keys 가 expected set 과 일치
2. 모든 layer 의 (D,) shape 가 §2.3 의 dim 표와 일치
3. enc_4 의 dim = projector input dim (whisper_tiny=384, dacvae=128, ...)
4. mean / last 가 일치하는 케이스 = utterance length 1 frame 일 때만 (boundary)
5. encoder frozen sanity (Stage 8 §C3): ALM enc_4 vs HuggingFace pretrained 의 같은 layer CKA = 1.0 ± ε

**재사용**:
- `AudioDataset`, `_collate` from `extract_encoder_only.py`
- `build_prompt_ids` from `_loader.py` (LLM input 빌드)
- ENCODER_CFG (sample rate / hop_length) from `extract_encoder_only.py`

**구현 노트**:
- 각 family 마다 hook 위치 다름 (whisper: `encoder.layers[i]` + `last_hidden_state`, dacvae: `audio_encoder.encoder.encoder.block[i]` + `quantizer._vae_sample` 후, wavtok: `audio_encoder.encoder.model[i]`, projector: `projector.layers[i]`, LLM: `language_model(inputs_embeds, output_hidden_states=True).hidden_states`)
- DACVAE seed: utterance hash 기반 fix (§12.4)
- batch_size=1 (메모리 안정성)
- 1 family 씩 순차 실행 (메모리 reset)

**예상 소요**: family 당 30 min – 1 h, 총 2-4 h on 1×A100.

---

### Stage 3 — CKA computation

**목표**: §5 의 Axis 1 / Axis 2 CKA matrix 산출 (mean/last × overall+4emotion).

**산출물**:
- `experiments/audio_encoder_probe/compute_cka.py` (≤ 200 줄)
- 출력:
  - `cka/{family}_within_{mean,last}_{overall,angry,happy,sad,neutral}.npy` — 15×15
  - `cka/cross_position_{mean,last}_{overall,angry,happy,sad,neutral}.npy` — 6×15

**성공 기준**:
- 모든 CKA 값 ∈ [0, 1]
- 대각선 = 1.000 (자기-유사도 sanity, §8 baseline)
- random Gaussian baseline < 0.05 (sanity)
- gram matrix 캐싱으로 §4.4 견적 (5-15 min) 내 완료

**테스트 케이스** (`tests/test_cka.py`):
1. `cka(X, X) == 1.0` for any X
2. `cka(X, Y) == cka(Y, X)` (대칭)
3. random Gaussian (N=2000, D=128) 두 set: cka < 0.1
4. (N, D1) vs (N, D2) where D1 ≠ D2 → 정상 작동
5. centered vs uncentered 입력 동일 결과 (centering 내부 처리)

**재사용**:
- numpy / scipy 만 사용 (별도 라이브러리 도입 X)
- `embeds_layers/` 디렉토리에서 layer 별 (N, D) stack 후 계산

**구현 노트**:
- linear CKA = `HSIC(X X^T, Y Y^T) / sqrt(HSIC · HSIC)` (centering 포함)
- gram matrix 는 layer 별 1 회 계산 → 모든 pair 에서 재사용
- bf16 → float32 cast for stability

**예상 소요**: 5-15 분.

---

### Stage 4 — Visualization

**목표**: heatmap / line plot 산출. 본문용 + 부록용 분리.

**산출물**:
- `experiments/audio_encoder_probe/plot_rq1.py` (≤ 200 줄)
- 출력 (`figs/`):
  - `within_{family}_{mean,last}.png` — 42×42 heatmap, enc/proj/llm 경계 line + colorbar (본문)
  - `cross_position_{mean,last}.png` — 6×42 heatmap (본문)
  - `smoothness_{mean,last}.png` — 인접 layer CKA 1-D plot, 4 family overlay (본문)
  - `within_{family}_{mean,last}_{emotion}.png` (부록 — emotion-conditioned)

**성공 기준**:
- 모든 heatmap 동일 colormap / 동일 vmin/vmax (family 간 비교 가능)
- 축 label 에 layer 이름 (enc_0, enc_1, ..., proj_0, ..., llm_0, ..., llm_31)
- enc/proj/llm 경계 vertical/horizontal line 그어짐
- PNG 300 dpi

**테스트 케이스** (`tests/test_plot_rq1.py` — 시각 검증은 수동):
1. 입력 npy shape mismatch 시 명시적 에러 (fast-fail)
2. 빈 디렉토리에서 실행 시 명확한 에러 메시지
3. 모든 출력 파일이 expected 경로에 생성

**재사용**: matplotlib 만, seaborn / plotly 도입 X.

**예상 소요**: 1-2 h.

---

### Stage 5 — Sanity check + Statistical robustness

**목표**: §8 (sanity check) + §9 (bootstrap CI / permutation test) 산출 → 결과 신뢰성 확보.

**산출물**:
- `experiments/audio_encoder_probe/cka_robust.py` (GPU 구현, torch CUDA)
- 출력:
  - `cka/axis2_ci.json` — 48 핵심 결과 (6 pair × 4 position × 2 pooling) 의 5/95 percentile + permutation p-value

**Sanity check 분리 처리** (별도 산출 파일 없이 다른 곳에서 검증):
- self-similarity, random baseline, 차원 invariance : `tests/experiments/test_cka.py`
- DACVAE post-VAE z 재현성 : `tests/experiments/test_extract_layers.py::test_dacvae_reproducible_z`
- encoder frozen 검증 : Stage 1 ckpt 의 encoder 가 학습 안 됨이라 trivially OK (별도 검증 X)

**구현 메모**:
- numpy fancy indexing (`K[np.ix_(idx, idx)]`) 가 N=2000 에서 너무 느림 (1000 iter × 48 결과 ≈ 50 분).
- → GPU torch (`tensor.index_select(0, idx).index_select(1, idx)`) 로 재구현. 약 1-3 분.

**성공 기준**:
- 모든 CKA 값 ∈ [0, 1]
- bootstrap CI 의 mean ≈ point estimate
- permutation p-value ≤ 0.05 (강한 alignment 시) 또는 > 0.05 (random baseline 수준)
- 핵심 48 결과에 CI 부착 → 본문 plot 에 errorbar 추가 가능

**테스트 케이스**:
1. bootstrap 1000 회 결과의 mean ≈ point estimate
2. permutation 1000 회의 null 분포가 정규분포 가까움
3. seed fix 시 CI 재현

**예상 소요**: 1-2 h.

---

### Stage 분기 / 의존성

```
Stage 0 (학습)
   ↓
Stage 1 (manifest)
   ↓
Stage 2 (extraction)  ──┬─→ Stage 3 (CKA)  ──→ Stage 4 (plot)
                        └─→ Stage 5 (robustness)  ──→ Stage 4 (plot 갱신)
```

각 stage 는 **이전 stage 의 출력 / 테스트 통과** 가 전제.
3 회 실패 시 (§협업 원칙) 시도 내역 기록 → 대안 2-3 탐색 → 본 doc 의 §12 (열린 질문) 에 결정 추가 후 재진행.

### 후속 (RQ1 v1 종료 후)

- Repr Richness 결과와 cross-reference (Δ_C / Δ_S 가 큰 family 가 LLM 후반 CKA 차이도 큰가?)
- EnCodec family 추가 (§12.5)
- frame-level RSA (§12.3)

---

## 부록: 구현 노트

- Whisper projector `for layer in self.layers: hidden_states = layer(...)` 는 `LlamaDecoderLayer` 가 tuple 을 반환할 때 `hidden_states` 가 tuple 이 됨 → hook 에서 `out[0]` 로 처리.
- LLM hidden 추출 시 `model.model.language_model(inputs_embeds=audio_embeds, output_hidden_states=True)` 사용. audio span position slicing 은 audio_embeds 의 token 수와 동일 (전 구간이 audio span).
- DACVAE encoder 는 `model.model.audio_encoder.encoder.encoder.block` (Sequential), WavTokenizer 는 `model.model.audio_encoder.encoder.model` (Sequential) — hook 위치 다름.
- CKA 는 N 이 클수록 분산 안정 → IEMOCAP 5500 은 충분. 부족하면 RBF CKA 의 σ 선택이 noisier 해질 뿐.
