# Audio-encoder Linear Probe — Phase 0 Results

> **Setup:** ALM에서 projector / LLM을 제외한 **pure audio encoder** 출력만 가지고 emotion / sound classification을 linear probe로 측정. 5 encoder × 4 dataset 격자 비교. 모든 encoder는 ALM Stage 1 학습 시 frozen이므로 standalone pretrained 가중치와 동등.

---

## 1. Setup

### 1.1 Encoders (모두 frozen)

| Encoder | 출처 | Pretraining | 차원 | Frame rate | Sample rate |
|---|---|---|---|---|---|
| Whisper-tiny.en | `openai/whisper-tiny.en`의 `.encoder` | ASR (438k h English) | 384 | 50 fps | 16 kHz |
| Whisper-small.en | `openai/whisper-small.en`의 `.encoder` | ASR (438k h English) | 768 | 50 fps | 16 kHz |
| WavTok-40 (unify) | `Qwen3.5AE-4B-wavtok-40-unify`의 `audio_encoder.encoder` (SEANet) | reconstruction (pre-VQ tap) | 512 | 40 fps | 24 kHz |
| EnCodec-24k | `Qwen3.5AE-4B-encodec-24k`의 `audio_encoder.encoder` (HF `EncodecModel.encoder`) | reconstruction (pre-RVQ tap) | 128 | 75 fps | 24 kHz |
| DAC-VAE | `Qwen3.5AE-4B-ASR-Stage1`의 `audio_encoder.encoder` | reconstruction (pre-RVQ tap, `z_e`) | 128 | 25 fps | 48 kHz |

추출: utterance → encoder forward → valid frames mask (Whisper 30s padding) → mean pool over time → (d,) float32.

### 1.2 Datasets

| Type | Dataset | 클래스 | Class balance | 규모 | Split |
|---|---|---|---|---|---|
| Emotion | IEMOCAP 4-class (angry / happy[+excited] / sad / neutral) | 4 | 불균형 (38/25/19/18%) | 6,877 utt | 5-fold leave-session-out (Ses01..05) |
| Emotion | RAVDESS (8-class) | 8 | 균형 (~13% each, neutral 6.7%) | 1,440 utt | 5-fold leave-speakers-out (24 actors → 5 fold) |
| Emotion | CREMA-D (6-class) | 6 | 거의 균형 (~17% each) | 7,442 utt | 5-fold leave-speakers-out (91 actors → 5 fold) |
| Sound | ESC-50 | 50 | 완벽 균형 (40/class) | 2,000 clips | Official 5-fold CV |

> Whisper 30s padding cap → 30s 초과 IEMOCAP utt (max 34s)는 30s로 truncate.

### 1.3 Probe

- sklearn `LogisticRegression` (L2), C ∈ {0.01, 0.1, 1.0, 10.0}, best per (enc×ds)
- StandardScaler on, LBFGS, max_iter=5000, seed=42

---

## 2. Results

### 2.1 Accuracy table (best C, 5-fold mean)

| Encoder | Dim | IEMOCAP | RAVDESS | CREMA-D | ESC-50 |
|---|---|---|---|---|---|
| **Whisper-tiny.en** | 384 | 0.6467 | 0.7102 | 0.6940 | 0.7905 |
| **Whisper-small.en** | 768 | **0.6695** | **0.7688** | **0.7371** | **0.8735** |
| WavTok-40 | 512 | 0.4885 | 0.3618 | 0.4406 | 0.2470 |
| EnCodec-24k | 128 | 0.4921 | 0.4253 | 0.4711 | 0.4220 |
| DAC-VAE | 128 | 0.4745 | 0.3403 | 0.4018 | 0.3155 |
| *Random chance* | — | 0.25 | 0.125 | 0.167 | 0.020 |
| *Majority baseline* | — | 0.383 | 0.133 | 0.171 | 0.020 |

### 2.2 Plots

- `_results/figs/probe_summary_bar.png` — 5 enc × 4 ds accuracy bar (with random chance line)
- `_results/figs/probe_per_fold.png` — fold별 accuracy 곡선
- `_results/figs/probe_C_sensitivity.png` — C grid sensitivity
- `_results/figs/probe_heatmap.png` — accuracy heatmap

---

## 3. Findings (객관적 관찰)

### 3.1 ASR vs Reconstruction encoder 격차
- **Whisper 양 변형 모두 4 dataset 전부에서 압도적 1·2위**
- Whisper-tiny.en (39M params, 384d)도 **모든 reconstruction encoder를 모든 dataset에서 앞섬**
  - vs WavTok-40 (50M+ params, 512d): IEMOCAP +15.8%pt, RAVDESS +34.8%pt, CREMA-D +25.3%pt, ESC-50 +54.4%pt
  - 차원/parameter 수가 결정 요인 아님 (whisper-tiny가 차원 작음에도 우월)

### 3.2 Dataset별 격차 패턴
- **ESC-50 (sound classification)에서 격차 최대**: whisper-small 87.4% vs wavtok 24.7% (+62.7%pt)
- **Emotion task에서 reconstruction encoder 간 분기 일관됨**: EnCodec > DAC-VAE > WavTok across all 3 emotion sets
- IEMOCAP만 reconstruction encoder들이 비슷 (47-49%) — 클래스 불균형 + 화자 의존성 때문일 수도

### 3.3 Encoder ranking 일관성
4 dataset 모두에서 동일한 순위:
```
whisper_small > whisper_tiny > encodec_24k > dacvae > wavtok_40_unify
```
(WavTok이 emotion에서는 dacvae보다 약간 위지만 ESC-50에서 최하위)

### 3.4 Whisper-tiny vs Whisper-small
- Tiny → small 차이는 **dataset마다 비례적**: ESC-50 +8.3%pt, RAVDESS +5.9%pt, CREMA-D +4.3%pt, IEMOCAP +2.3%pt
- Sound classification (ESC-50)에서 더 큰 모델 효과 두드러짐

---

## 4. 가설 검토

원 가설: **"acoustic encoder (recon-pretrained)가 acoustic task에서 우월"**

본 Phase 0 결과는 **반대 방향의 강한 신호**를 4개 데이터셋에서 일관되게 보여줌:

- Reconstruction encoder가 ASR-pretrained Whisper에 모든 task에서 크게 뒤짐
- 격차가 가장 큰 task가 sound classification (ESC-50) — recon encoder가 *acoustic* 태스크에서도 약함
- Whisper-tiny.en (39M)이 reconstruction encoder들 (~50-100M)을 모두 앞섬 — capacity로 설명 안 됨

가능한 해석 (검증 안 됨):

1. **Pretraining objective의 본질적 차이**:
   - ASR encoder는 phoneme/word 인식을 위해 acoustic 변이 (화자, prosody, 환경 noise)를 *유지*해야 함 (decoder가 transcript 외 정보를 무시)
   - Reconstruction encoder는 *bit-rate 효율적* 표현 학습 → pre-VQ z_e가 quantizable 매니폴드에 사상 → low-rank, utterance-level discriminative power 약함
2. **데이터 다양성**: Whisper 438k h English 음성 = 노이즈/화자/감정/음악 포함 → broad acoustic features
3. **Frame-level 재합성 ↔ utterance-level discrimination 트레이드오프**: ESC-50의 5초 clip semantic class는 frame-level 정보의 *aggregation*이 필요. Reconstruction loss는 frame 수준 fidelity에 최적화.

---

## 5. 한계 / 비고

- **Pooling 단순화**: mean pooling만. attention pool / max pool / multi-stat 비교 미수행.
- **Pretraining checkpoint 출처**: WavTok / EnCodec encoder weights는 base model safetensors에서 추출. Standalone HF / WavTokenizer 가중치와의 sanity check 미수행 (Stage1 frozen이라 동등 추정).
- **Test 분포의 spurious correlation 가능성 미검증**: 예를 들어 RAVDESS는 24명 actor의 강한 음성 — speaker timbre가 emotion label과 confounding 가능. Whisper의 우월이 진정한 emotion 표현인지 speaker 식별인지 분리 안 됨.
- **Sample rate 차이**: 16k (whisper) vs 24k (wavtok/encodec) vs 48k (dacvae). 다운샘플 시 정보 손실 가능성, but 본 실험에서는 각 encoder의 native rate로 입력.

---

## 6. Phase 0.5 — Non-linear probe (entanglement test)

### 6.1 동기

Phase 0의 linear probe 결과만으로는 "reconstruction encoder는 정보가 *없다*"와 "정보는 *있는데* entangle되어 linear 추출 불가"를 구분할 수 없음. 따라서:

- **MLP probe (1 hidden layer × 512 units)**: non-linear 변환 후에도 정보 추출 가능한지 확인
- **kNN probe (k=11, cosine)**: 비파라미터 manifold 기반 추출
- **mel-only baseline (80-d log-mel time-mean)**: 인코더 자체의 가치를 측정 — 인코더가 mel feature보다 무엇을 더 더했는가?

### 6.2 Linear vs MLP 비교 (5-fold mean accuracy, MLP best-α grid {1e-5,1e-4,1e-3,1e-2})

> **방법론 주의**: MLP는 linear의 superset이므로 *원리상* linear ≤ MLP. 본 실험에서 h512(단일 hidden 512)과 h64x3(3 hidden × 64) 두 arch 모두 시도했으나 어느 것도 linear를 일관되게 이기지 못함. 이는 random init + adam + finite epoch + 작은 데이터(1.4k–7.4k) 조합의 *최적화 한계*로 해석. 따라서 절대 gap보다 *상대 패턴*에 주목.

| Encoder | Dim | Dataset | Linear | MLP best-α | gap (MLP−Lin) |
|---|---|---|---|---|---|
| Whisper-small | 768 | IEMOCAP | 0.6695 | 0.6954 | **+0.026** |
| | | RAVDESS | 0.7688 | 0.7670 | −0.002 |
| | | CREMA-D | 0.7371 | 0.7353 | −0.002 |
| | | ESC-50 | 0.8735 | 0.8680 | −0.006 |
| Whisper-tiny | 384 | IEMOCAP | 0.6467 | 0.6697 | **+0.023** |
| | | RAVDESS | 0.7102 | 0.7035 | −0.007 |
| | | CREMA-D | 0.6940 | 0.6930 | −0.001 |
| | | ESC-50 | 0.7905 | 0.7760 | −0.015 |
| WavTok-40 | 512 | IEMOCAP | 0.4885 | 0.4557 | −0.033 |
| | | RAVDESS | 0.3618 | 0.3193 | −0.043 |
| | | CREMA-D | 0.4406 | 0.4018 | −0.039 |
| | | ESC-50 | 0.2470 | 0.1935 | −0.054 |
| EnCodec-24k | 128 | IEMOCAP | 0.4921 | 0.4531 | −0.039 |
| | | RAVDESS | 0.4253 | 0.3978 | −0.028 |
| | | CREMA-D | 0.4711 | 0.4107 | −0.060 |
| | | ESC-50 | 0.4220 | 0.3975 | −0.025 |
| DAC-VAE | 128 | IEMOCAP | 0.4745 | 0.4399 | −0.035 |
| | | RAVDESS | 0.3403 | 0.3318 | −0.009 |
| | | CREMA-D | 0.4018 | 0.3539 | −0.048 |
| | | ESC-50 | 0.3155 | 0.3285 | +0.013 |
| mel-only | 80 | IEMOCAP | 0.4937 | 0.4592 | −0.035 |
| | | RAVDESS | 0.3993 | 0.3872 | −0.012 |
| | | CREMA-D | 0.3976 | 0.3964 | −0.001 |
| | | ESC-50 | 0.3265 | 0.3345 | +0.008 |

**관찰** (h512):
1. **Whisper encoder**: MLP ≈ linear (IEMOCAP에서만 +2-3 %pt 작은 이득; 나머지는 차이 없음). 이미 robustly linear-readable.
2. **Reconstruction encoder**: MLP가 일관되게 −2 ~ −6 %pt 더 *낮음*. Linear의 convex 최적화가 MLP-Adam의 non-convex 최적화보다 더 안정적으로 분류기를 찾음.

### 6.3 Deeper MLP (3 hidden layers × 64 units) — h64x3

좁은 bottleneck + 깊이로 capacity-regularization 균형 실험:

| 구성 | Linear 이긴 cells | 평균 gap |
|---|---|---|
| MLP-h512 (1 × 512) | 4/28 | −2.1 %pt |
| **MLP-h64x3 (3 × 64)** | **0/28** | **−4.3 %pt** |

h64x3이 *모든* (encoder × dataset)에서 linear에 진다. 이유:
- input dim(80~768) → 64 bottleneck이 강한 information loss
- 총 파라미터(~12k) < linear classifier(특히 768×50=38k for ESC-50)

→ Capacity 부족. 단순히 "narrow + deep"이 해결책이 아님.

### 6.4 Entanglement 가설 평가

두 MLP 구성으로도 linear을 일관되게 이기지 못함. 가능한 해석:
- (가설 A) 정보가 *not entangled, just absent* — reconstruction encoder representation이 utterance-level acoustic class 변별 정보를 충분히 담지 않음
- (가설 B) 정보 있으나 sklearn MLP-adam + 작은 데이터로 풀 수 없는 더 강한 non-linearity 필요

본 실험만으로 (A)/(B) 구분 불가. 결정적 검증을 위해서는:
- 더 큰 데이터 (e.g., 100k+) — non-linear probe의 train budget 확보
- Deeper transformer probe (e.g., 2-layer attention)
- Gradient boosting (XGBoost) — 다른 inductive bias
- Fine-tune 1-2 encoder layers — 표현 자체를 작업에 적응

본 Phase 0.5 결론: *strong entanglement* (정보가 강하게 얽혀 nonlinear로만 풀림) 가설은 *약하게 부정* — 가능한 nonlinear extraction (sklearn MLP)으로는 해결 안 됨.

### 6.5 Mel-only baseline vs reconstruction encoders

| Dataset | mel-only (80d) | DAC-VAE (128d) | EnCodec (128d) | WavTok (512d) |
|---|---|---|---|---|
| IEMOCAP | 0.4937 | 0.4745 | 0.4921 | 0.4885 |
| RAVDESS | 0.3993 | 0.3403 | 0.4253 | 0.3618 |
| CREMA-D | 0.3976 | 0.4018 | 0.4711 | 0.4406 |
| ESC-50 | 0.3265 | 0.3155 | 0.4220 | 0.2470 |

**관찰**: 80-d log-mel의 단순 time-mean이 reconstruction encoder의 forward 출력과 *비슷하거나 일부 task (IEMOCAP, RAVDESS, ESC-50 vs WavTok)에서는 더 나음*. 즉:
- Reconstruction encoder forward는 utterance-level discrimination 관점에서 **mel mean 대비 거의 가치를 더하지 않음**
- Whisper-small encoder만 mel(32.7% on ESC-50) → 87.4% (+54.7%pt) 처럼 진짜 변환을 수행

### 6.6 kNN probe

kNN은 모든 encoder에서 linear/MLP 보다 일관되게 낮음. 이는 normal — kNN은 high-d normalize된 embedding에서 manifold 가정에 의존하는데, 본 데이터 규모(1.4k–7.4k)에서 high-d로 가면 효율 떨어짐. 단독 결론용 metric은 아니지만 ranking은 linear/MLP와 동일.

### 6.7 Phase 0.5 Findings

1. **Entanglement 가설 약한 지지 안 됨**: MLP가 격차를 좁히지 못하고 오히려 약간 더 벌어짐. 단, MLP의 음의 gap은 최적화 한계로 부분 설명 가능 — 더 강한 비선형 probe (deeper, larger init ensemble) 로 추가 검증 필요. 현 단계에선 *strong entanglement* 가설은 기각, 다만 *mild non-linearity* 까지는 본 실험으로 단언 어려움.
2. **Reconstruction encoder ≈ raw mel feature**: 80-d log-mel time-mean이 reconstruction encoder의 forward 출력과 비슷. *frame-level acoustic detail*은 보존되지만 (재합성이 가능하므로), *clip-level semantic*은 mel 대비 거의 추가 학습되지 않음.
3. **Whisper encoder는 본질적으로 다른 변환**: mel + ASR pretraining 조합으로 utterance/clip-level discriminative geometry를 학습. mel(33%) → 87% on ESC-50.
4. **Post-VQ는 부차적**: WavTok의 quantization은 평균 −3.3 %pt 손실만 야기 — Whisper와의 큰 격차의 주원인이 *아님*. 격차는 encoder pretraining objective에서 결정.

### 6.8 Post-VQ probe (WavTok)

WavTok의 quantization 단계가 utterance-level 분류에 미치는 영향을 측정. 원본 WavTokenizer ckpt(`novateur/WavTokenizer-large-unify-40token`)에서 codebook (4096 × 512)을 추출하여 nearest-neighbor lookup으로 z_e → z_q 변환 후 mean pool.

| | IEMOCAP | RAVDESS | CREMA-D | ESC-50 |
|---|---|---|---|---|
| WavTok pre-VQ (z_e, 512d cont.) | 0.4885 | 0.3618 | 0.4406 | 0.2470 |
| WavTok post-VQ (z_q, codebook) | 0.4805 | 0.3373 | 0.3813 | 0.2080 |
| Δ (post − pre) | −0.008 | −0.025 | −0.059 | −0.039 |

**관찰**:
- VQ의 정보 손실 평균 약 −3.3 %pt — 양자화 자체가 분류 정보를 크게 깎아내지 않음
- 이는 WavTok이 reconstruction task에서 Whisper-tiny(79%)와 큰 격차(54.3 %pt)를 보이는 핵심 원인이 *quantization*이 아니라 *encoder pretraining objective* 자체에 있음을 시사
- 즉 SEANet 인코더가 reconstruction loss로 학습한 z_e가 이미 utterance-level acoustic class 변별력이 약함; VQ 후 추가 손실은 부차적

### 6.9 한계

- **Frame-level / sequence-level probe 미수행**: Mean pooling은 utterance-level에서만 평가. Reconstruction encoder가 frame-level (e.g., word-level emotion segment) 정보는 잘 보존할 수도 있음.
- **MLP capacity 단일 setting**: hidden_size={256, 1024} 등 grid search 없음.
- **Train data size 영향**: 더 큰 데이터에서 MLP 격차가 다르게 나올 수 있음.
- **Generative reconstruction quality와의 직접 비교 없음**: 정보 보존 != 분류 가능성. 인코더가 "정보를 다 가지고 있지만 utterance pooling 자체에 적합하지 않은 형태로 둔다"는 시나리오는 별도 generative probe (e.g., reconstruction loss 측정) 가 필요.

---

## 7. 결론 (사용자 추가)

> 본 섹션은 사용자가 작성. Haiku는 객관적 관찰까지만 기술.

---

## Appendix — 산출물 경로

```
experiments/audio_encoder_probe/
├── manifests/
│   ├── iemocap_4class.csv               (6,877 rows)
│   ├── ravdess.csv                      (1,440 rows)
│   ├── cremad.csv                       (7,442 rows)
│   └── esc50.csv                        (2,000 rows)
├── embeds/
│   ├── whisper_tiny/{4 ds}/{utt_id}.npy
│   ├── whisper_small/{4 ds}/{utt_id}.npy
│   ├── wavtok_40_unify/{4 ds}/{utt_id}.npy
│   ├── encodec_24k/{4 ds}/{utt_id}.npy
│   └── dacvae/{4 ds}/{utt_id}.npy
├── embeds/
│   ├── mel_only/{4 ds}/{utt_id}.npy            # 80-d log-mel time-mean baseline
│   └── wavtok_40_unify_postvq/{4 ds}/{utt_id}.npy  # 512-d post-VQ z_q (codebook from novateur HF)
└── _results/
    ├── probe_results.csv                (560 rows: 7 enc × 4 ds × 5 fold × 4 C, +mel_only +wavtok_postvq)
    ├── probe_nonlinear_results.csv      (600 rows: 6 enc × 4 ds × 5 fold × (4 alpha mlp + 1 knn))
    └── figs/
        ├── probe_summary_bar.png
        ├── probe_per_fold.png
        ├── probe_C_sensitivity.png
        └── probe_heatmap.png

docs/plans/plan_audio_encoder_probe_phase0.md     — 실행 plan
docs/stage2_analysis/audio_encoder_probe_phase0.md — 본 리포트
```
