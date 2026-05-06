# Plan 3 — Analysis (Representation Richness, Phase 1)

> 실행자: **Haiku**. Plan 2 산출물 (37 layers × **3 family** × ~7920 utt 임베딩) 으로 disentanglement / spectrum 메트릭 산출 + 리포트 작성. encodec-24k 는 ckpt 부재로 본 plan 에서 제외.

---

## 목표

CMU-ARCTIC 7 화자 셋에서:

1. **Analysis (A) Disentanglement** — content/speaker invariance, linear probe, probe gap
2. **Analysis (B) Spectrum** — effective rank, participation ratio, α-ReQ
3. 두 분석 결과를 **family × layer 격자** 로 정리하고 plot
4. `docs/stage2_analysis/representation_richness_phase1.md` 리포트 작성

---

## 선행 조건

```bash
# Plan 2 산출물 검증
ls experiments/representation_richness/cmu_arctic_7/whisper_tiny/encoder_out/ | wc -l  # ~7920

# 분석 의존성
python -c "import numpy, scipy, sklearn, matplotlib, pandas, seaborn; print('ok')"
```

---

## 절대 금지 / 주의사항

- ❌ Plan 2 의 npz 를 수정/삭제 금지. 분석은 read-only.
- ❌ 분석 도중 임의의 추가 metric 도입 금지. 본 plan 에 정의된 것만.
- ❌ Stage 2 ckpt 로 재추출하거나 다른 데이터셋 (VCTK, DAPS) 으로 확장 금지 — 본 plan 은 Phase 1 (CMU-ARCTIC 7 화자) 만.
- ⚠️ Probe 학습 시 train/test split 시드 고정 (e.g. 42).
- ⚠️ Spectrum 계산은 메모리 큼 — frame 행렬을 stream 으로 covariance 누적하거나, 차원이 높은 LLM layer (2560-d) 는 모든 frame 을 한 번에 메모리 올리지 말 것.
- ⚠️ Plot 은 family × layer 격자가 핵심. legend / axis label 영문, 한글 폰트 의존성 X.
- ⚠️ 결과 csv 는 모두 `experiments/representation_richness/cmu_arctic_7/_results/` 아래.

---

## TaskCreate 트래킹

```
1. 임베딩 로딩 유틸 (load_embeds.py)
2. Analysis A — content/speaker invariance
3. Analysis A — linear probe
4. Analysis A — 결과 정리 + plot
5. Analysis B — spectrum (frame-level)
6. Analysis B — spectrum (utt-mean)
7. Analysis B — 결과 정리 + plot
8. 리포트 작성 (representation_richness_phase1.md)
```

---

## Phase 3.1 — 임베딩 로딩 유틸

**파일**: `experiments/representation_richness/load_embeds.py`

```python
"""
Lazy loader for layer-wise embeddings.

Usage:
    from load_embeds import load_layer
    arr_mean, arr_last, frames, meta = load_layer(
        family="whisper_tiny",
        layer="projector_L4",
        manifest_csv="/mnt/tmp/cache/cmu_arctic_7/manifest.csv",
        load_frames=False,  # spectrum 분석 시에만 True
    )
    # arr_mean: (N, d) float32 — utterance-level mean pool
    # arr_last: (N, d) float32 — utterance-level last frame (encoder_out 은 None)
    # frames: list[(T_b, d) float16] — 길이 N, 가변 시간축
    # meta: pd.DataFrame with [utt_id, speaker_id, transcript_id, duration_s]
```

매니페스트와 npz 디렉토리를 join. utt_id 가 매니페스트엔 있는데 npz 가 없는 경우 (Plan 2 에서 일부 skip 된 utt) 는 빠진 채로 반환 + 경고.

---

## Phase 3.2 — Analysis (A) Disentanglement

### 3.2.A — Pairwise similarity invariance

**파일**: `experiments/representation_richness/analysis_a_invariance.py`

Family × layer × pooling (mean | last) 격자 순회. 각 cell 에서:

1. utterance-level vector 행렬 `E ∈ R^(N, d)` 로드
2. L2-normalize 후 sim matrix 의 각 entry 평균을 (transcript 같음? speaker 같음?) 4 분면으로 분류:
   - **Same content, diff speaker (SC-DS)** → 평균 = $C$
   - **Diff content, same speaker (DC-SS)** → 평균 = $S$
   - **Diff content, diff speaker (DC-DS)** → 평균 = $X$
   - **Same content, same speaker (SC-SS)** → 자기자신 외 평균 (보통 N=0 or 1, skip)
3. $\Delta_C = C - X$, $\Delta_S = S - X$ 계산

> **메모리/시간 주의**: 7920² ≈ 6.3e7 페어 → loop 으로 돌리지 말고 numpy / einsum 으로 sim matrix 한 번에. 차원 d=2560 일 때 sim matrix (7920, 7920) float32 = 250 MB — OK. RAM 으로 충분.

저장:

```
_results/analysis_a_invariance.csv
  columns: family, layer, pooling, C, S, X, delta_C, delta_S, n_pairs_C, n_pairs_S, n_pairs_X
```

### 3.2.B — Linear probe

**파일**: `analysis_a_probe.py`

각 cell 에서 두 logistic regression:

- **Content probe**: 입력 = utterance-mean vector, 라벨 = transcript_id (1132 클래스). class 가 너무 많으면 frequency top-K (e.g. 200 transcripts 이상 출현 빈도) 만 사용.
  - CMU-ARCTIC 의 경우 transcript_id 별로 7 utt → top-K 필터 후에도 여전히 적은 샘플. **adaptation**: transcript_id 가 아니라 *transcript text 의 phoneme/word level token ID* 를 라벨로 쓰는 게 분류 task 로 더 의미 있음. 일단 본 plan 에서는 transcript_id 그대로 두되 **train/test 를 utterance random split (80/20, seed=42)** 으로 하여 같은 transcript 의 다른 utterance 가 train/test 양쪽에 들어가도록 함.
- **Speaker probe**: 라벨 = speaker_id (7 클래스). 동일 split.

추가 OOD 측정 (선택):

- Speaker leave-one-out: 1 화자 hold-out, 6 화자로 train. 7 폴드 평균.

저장:

```
_results/analysis_a_probe.csv
  columns: family, layer, pooling, content_acc, speaker_acc, content_acc_loo, speaker_acc_loo
```

### 3.2.C — 결과 정리 + plot

**파일**: `plot_analysis_a.py` → `_results/figs/`:

1. `delta_C_vs_layer.png`: x=layer (encoder_out, projector_L1..4, llm_L01..L32), y=$\Delta_C$, family 별 line
2. `delta_S_vs_layer.png`: 동일 format, y=$\Delta_S$
3. `probe_gap_vs_layer.png`: y=content_acc − speaker_acc, family 별 line
4. `pool_compare.png`: mean vs last 풀링 비교 (projector & LLM 만)

각 figure 의 title 에 dataset = "CMU-ARCTIC 7-speaker", pooling 명시.

---

## Phase 3.3 — Analysis (B) Spectrum

### 3.3.A — Frame-level spectrum (메인)

**파일**: `analysis_b_spectrum.py`

각 cell 에서:

1. 모든 utterance 의 frame-level 행렬을 시간 축으로 concat → `F ∈ R^(M, d)` where M ≤ 7920 × 256
2. centering (column mean 빼기). normalize 안 함.
3. covariance `Σ = F^T F / (M-1)`, $d \times d$ 행렬
4. SVD or eigendecomposition → 고윳값 $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_d$
5. 메트릭:
   - $r_\text{eff} = \exp\bigl(-\sum p_i \log p_i\bigr),\ p_i = \sigma_i/\sum \sigma_j$
   - $\text{PR} = (\sum \sigma_i)^2 / \sum \sigma_i^2$
   - $\alpha$ from log-log fit of $\sigma_i$ vs $i$ (i ∈ [10, d/2] 범위 — 양 끝 noisy 영역 제외)
6. $r_\text{eff}/d$ 로 정규화 보고

> **메모리**: M 이 200 만 정도, d 가 LLM 의 2560 일 때 F 가 ~10 GB 메모리 필요. **streaming covariance** 로:
> ```python
> Sigma = np.zeros((d, d), dtype=np.float64)
> n = 0
> for utt_frames in iterate_npz_frames(...):  # one utt at a time
>     X = utt_frames - column_mean  # (T_b, d)
>     Sigma += X.T @ X
>     n += X.shape[0]
> Sigma /= (n - 1)
> ```
> column_mean 은 1-pass 누적 후 2-pass 로 분리하거나, Welford 알고리즘 사용.

저장:

```
_results/analysis_b_spectrum_frame.csv
  columns: family, layer, d, n_frames, r_eff, r_eff_norm, PR, alpha
```

### 3.3.B — Utterance-mean spectrum (보조)

같은 메트릭을 utt-mean 행렬 `E ∈ R^(N, d)` 에 적용. N=7920 이므로 메모리 부담 없음. utt-level 의 spread 가 frame-level 과 다른 양임을 명시.

저장: `_results/analysis_b_spectrum_utt.csv`

### 3.3.C — Plot

`plot_analysis_b.py`:

1. `r_eff_vs_layer.png`: x=layer, y=r_eff_norm, family line. frame / utt 두 panel
2. `PR_vs_layer.png`: 동일 format
3. `spectrum_curves.png`: layer 몇 개 (encoder_out, projector_L4, llm_L01, llm_L16, llm_L32) 에 대해 log-log 스펙트럼 곡선 4 family overlay

---

## Phase 3.4 — 리포트 작성

**파일**: `docs/stage2_analysis/representation_richness_phase1.md` (신규)

구조:

```markdown
# Representation Richness — Phase 1 Results (CMU-ARCTIC 7-speaker, Stage 1 ckpt)

## 1. Setup
- 데이터셋, family, ckpt step, 풀링, 추출 layer 수 요약
- 각 family 별 utt 수 (skip 발생 여부 포함)

## 2. Analysis A — Disentanglement
### 2.1 Δ_C, Δ_S layer-wise plot
### 2.2 Probe gap layer-wise plot
### 2.3 Mean vs Last pooling
### 2.4 Findings (가설 검증)
- 가설 (A) 의 예측 (acoustic family 가 더 큰 Δ_S, 더 깊은 layer 까지 유지) 이 보이는가?
- Findings 는 plot 에 기반하여 *기술* 만. over-claim 금지.

## 3. Analysis B — Spectrum
### 3.1 Effective rank layer-wise
### 3.2 Spectrum curves at selected layers
### 3.3 Findings

## 4. Cross-analysis
- (A) 의 Δ_S 와 (B) 의 r_eff 가 family 간 상관 (3 점 산점도, 통계 검증 제한적)
- 한계 / 다음 단계 (Phase 2 = VCTK / DAPS / EXPRESSO)
```

> 리포트는 사실/관찰만. *결론 문장은 사용자 검토 후 추가.* Haiku 가 임의로 "가설 입증됨/기각됨" 같은 결론 쓰지 말 것. 데이터 / plot reference 와 한 줄 객관적 관찰까지.

---

## Phase 3.5 — 최종 산출물 점검

```bash
ls experiments/representation_richness/cmu_arctic_7/_results/
# 기대:
#   analysis_a_invariance.csv
#   analysis_a_probe.csv
#   analysis_b_spectrum_frame.csv
#   analysis_b_spectrum_utt.csv
#   figs/
#     delta_C_vs_layer.png
#     delta_S_vs_layer.png
#     probe_gap_vs_layer.png
#     pool_compare.png
#     r_eff_vs_layer.png
#     PR_vs_layer.png
#     spectrum_curves.png

ls docs/stage2_analysis/representation_richness_phase1.md
```

---

## 완료 조건

- [ ] 모든 csv 파일이 family × layer × pooling 격자를 채움
- [ ] 7 plot 모두 생성
- [ ] 리포트 markdown 작성 완료, 객관적 관찰 위주 (결론은 사용자가 추가)
- [ ] encodec-24k 부재가 리포트 §1 에 명시됨

---

## Plan 4 (후속) 후보

이 plan 종료 후 사용자 결정에 따라:

- **Plan 4a — VCTK / DAPS 확장**: 같은 추출/분석 파이프라인 재실행
- **Plan 4b — EXPRESSO prosody 축**
- **Plan 4c — Stage 2 ckpt 재분석**: Stage 2 학습 완료 후 동일 파이프라인 재실행하여 Stage 1 ↔ 2 비교

각 Plan 4* 는 별도 plan 문서로 작성.
