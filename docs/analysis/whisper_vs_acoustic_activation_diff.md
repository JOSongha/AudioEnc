# Whisper vs Acoustic Encoder — Activation Difference Analysis

**작성일**: 2026-05-06
**상태**: Phase A 완료 — Phase B (frame-level) 진입 준비
**대상 데이터**: NSynth (train_30k, 27218 utt, 11 instrument families)
**비교 대상 (1차)**: `whisper_small` × `dacvae`
**확장 (Phase B/D)**: `wavtok_40_unify` 합류 가능

---

## 1. 분석 동기 (교수님 피드백)

> 내부 활성화 비교 시, Whisper 가 acoustic 보다 더 정확하다면 *음은 비슷하지만 의미가 다른 경우* 를 Whisper 가 더 잘 구분하기 때문일 것. 음과 의미가 1:1 매핑되는 경우는 비슷할 것으로 예상됨.
>
> → Whisper 의 성능이 더 높게 나타나는 지점의 활성화 차이를 분석하면, *언제 / 왜* Whisper 가 더 잘 작동하는지 이해할 수 있음.
>
> → 그 정보를 acoustic 모델에 LoRA 등으로 추가해도 비슷한 성능을 낸다는 점 (경량화 효과) 을 논문에 어필.

이 두 메시지를 다음 두 질문으로 분리:
- **Q1 (먼저)**: 언제 / 왜 Whisper 가 acoustic 보다 잘 하는가? (observational diagnostic)
- **Q2 (그 다음)**: 그 추가 정보를 acoustic encoder 위에 K-dim 으로 옮길 수 있는가? (LoRA-style sufficiency)

이 문서는 **Q1** 의 plan. Q2 는 별도 문서로 후속.

---

## 2. 분석의 사전 결정 사항

| 결정 사항 | 선택 | 이유 |
|---|---|---|
| **데이터셋** | NSynth (train_30k) 만 | speech 가 없어서 prof hypothesis 의 "음소-의미" 변형이 *instrument family* 단위로 깨끗하게 검증됨 |
| **비교 family (1차)** | whisper_small × dacvae | 모든 데이터 (utt-mean + frame-level + per-sample prediction) 이 이미 확보됨 |
| **확장 family** | wavtok_40_unify | Phase B (frame-level 비교) 합류 — utt-mean 추출 필요 |
| **Frame rate 정렬 (Q1)** | (c) 각자 native 유지 | whisper 50Hz / dac 25Hz / wavtok 40Hz — 비교는 sec 단위 통계 또는 probe-projected space 에서 |
| **차원 정렬** | 직접 정렬 안 함 | 768 vs 128 vs 512. 비교는 (1) class-prob trajectory (probe 투영) 또는 (2) sample-pair distance 공간에서 |

### 왜 mean-pool 하면 안 되는가
linear probing 은 utt-mean 으로 했지만, 이번 분석은 *언제* (utt 안 어느 frame 에서) 활성화가 갈리는지 봐야 prof hypothesis 의 mechanism 이 잡힘. utt-mean 은 그 시간 정보를 잃음 → Phase B 부터는 frame-level 필수.

(단, Phase A 의 sample 식별 + meta 분포는 utt-mean 으로 충분 — 이미 확보된 데이터로 즉시 시작 가능)

---

## 3. 활성화 차이의 operational 정의

dim 과 frame rate 가 다르므로, "활성화 차이" 를 *직접* 측정할 수 없음. 의미 있는 투영은 셋:

| Layer | 어디서 비교 | 답하는 질문 |
|---|---|---|
| **L1. Behavioral** | per-frame class-prob trajectory `[T_e, 11]` (probe 투영) | **WHEN** — utt 안 어느 frame 에서 활성화가 갈리는가 |
| **L2. Geometric** | sample-pair cos distance 공간 (각자 normalize) | **WHICH** — 어떤 sample-pair 에서 분리도 차이가 발생 |
| **L3. Mechanistic** | cross-encoder linear residual $r = W - \hat M D$ | **WHAT** extra info, **HOW** much (LoRA 가능한가) |

**Q1 (언제/왜)** 은 주로 **L1 + L2** 가 답함. **L3 (= Q2)** 는 후속 문서.

---

## 4. 진단 pipeline 큰 그림

```
Phase A.  WHO   — 어느 sample 에서 Whisper가 이기는가 (sample 식별 + 속성 분포)
   ↓                                                  [utt-mean, 데이터 확보 완료]
Phase B.  WHEN  — 그 sample 의 utt 안 어느 frame 에서 활성화가 갈리는가
   ↓                                                  [frame-level, 데이터 확보 완료]
Phase C.  WHY   — 그 frame 의 audio 가 무엇을 담고 있는가 (음향적 의미해석)
   ↓                                                  [audio-aligned inspection]
Phase D.  HOW   — 그 정보를 acoustic 위에 K-dim 으로 옮길 수 있는가 (= Q2, 별도 문서)
```

---

## 5. Phase A. WHO wins — 즉시 시작

### 5.1 목표
NSynth 27218 utt 안에서 *Whisper-only correct* set $S_{W^+A^-}$ 를 식별하고, 그 set 의 속성 분포를 control set 들과 비교. 이걸로 "Whisper 가 *어떤 종류* 의 sample 에서 강한가" 의 1차 답을 도출.

### 5.2 사전 데이터 (모두 확보)

| 자원 | 경로 | 상태 |
|---|---|---|
| whisper_small utt-mean | `experiments/audio_encoder_probe/embeds/whisper_small/nsynth_train_30k/*.npy` | ✓ 27218 |
| dacvae utt-mean | `experiments/audio_encoder_probe/embeds/dacvae/nsynth_train_30k/*.npy` | ✓ 27218 |
| whisper_small frame-level | `experiments/audio_encoder_probe/embeds_frames/whisper_small/nsynth_train_30k/*.npz` | ✓ 27218 (Phase B 용) |
| dacvae frame-level | `experiments/audio_encoder_probe/embeds_frames/dacvae/nsynth_train_30k/*.npz` | ✓ 27218 (Phase B 용) |
| Combined predictions | `experiments/audio_encoder_probe/_results/predictions/combined_whisper_small_dacvae_nsynth_train_30k.csv` | ✓ |

**Combined CSV schema**:
```
utt_id, audio_path, true_label, fold,
whisper_small_pred, whisper_small_proba, whisper_small_correct,
dacvae_pred, dacvae_proba, dacvae_correct
```

### 5.3 NSynth meta — utt_id 파싱으로 즉시 join

NSynth json 다운로드 필요 없음. utt_id 형식:
```
bass_electronic_034-061-100
└─┬─┘ └────┬────┘ └┬┘ └┬┘ └┬┘
family  source    id pitch velocity
```
- `instrument_family` ∈ {bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth_lead, vocal}
- `instrument_source` ∈ {acoustic, electronic, synthetic}
- `pitch` (MIDI 0–127)
- `velocity` (25, 50, 75, 100, 127)

(NSynth `qualities` tag — bright/dark/percussive/long_release/... — 는 json 필요. 1차 분석에선 제외, 흥미롭게 나오면 후속 join.)

### 5.4 Sample set 정의

2-way (whisper_small / dacvae) 에서 4 set 으로 분할:

| Set | 정의 | 의미 |
|---|---|---|
| $S_{W^+A^+}$ | W ✓ AND D ✓ | 둘 다 맞춤 (control, 쉬운 sample) |
| $S_{W^+A^-}$ | W ✓ AND D ✗ | **타겟**: Whisper 만 맞춘 sample |
| $S_{W^-A^+}$ | W ✗ AND D ✓ | acoustic 만 맞춤 (희귀할 것, sanity) |
| $S_{W^-A^-}$ | W ✗ AND D ✗ | 둘 다 틀림 (hard sample) |

각 set 의 size 자체가 hypothesis 강도 indicator:
- $|S_{W^+A^-}|$ 가 NSynth 의 30~40% → prof 해석이 강하게 작용
- $|S_{W^+A^-}| \approx |S_{W^-A^+}|$ → 우연 분류, hypothesis 약함
- $|S_{W^-A^+}|$ 가 의미 있는 크기 → "DAC 만 맞춘 sample" 도 별도 분석 가치 (acoustic 만의 강점 영역)

### 5.5 Phase A 분석 항목 (deliverable 5 figure)

#### F-A1. 4 set size
$S_{W^+A^+}$, $S_{W^+A^-}$, $S_{W^-A^+}$, $S_{W^-A^-}$ 의 sample 수 — bar/pie. set imbalance 파악.

#### F-A2. DAC confusion on $S_{W^+A^-}$
DAC 가 그 sample 들에서 *무엇으로* 라벨을 잘못 예측했는가. 11×11 confusion matrix (행 = true family, 열 = DAC pred family).
→ "Whisper-only 셋에서 DAC 는 bass→guitar (X%), organ→keyboard (Y%) 로 헷갈린다" — prof 의 "음 비슷·의미 다름" 의 *구체적 family pair* 정체.

#### F-A3. NSynth meta 분포 비교
$S_{W^+A^-}$ vs $S_{W^+A^+}$ vs $S_{W^-A^-}$ 에서:
- pitch histogram (MIDI bin)
- velocity bar (5 levels)
- instrument_source bar (acoustic/electronic/synthetic)

각 attribute 에서 chi-square / KS 검정으로 over/under-represented 영역 식별.
→ "Whisper-only 셋은 *electronic source* 와 *low pitch (MIDI 24–48)* 에 통계적으로 유의하게 몰림" 같은 형태의 답.

#### F-A4. Whisper pred_proba 분포
4 set 별 violin plot. $S_{W^+A^-}$ 에서 Whisper 가 *확신* (proba > 0.9) 으로 맞춘 비율 vs *marginal* (0.4–0.6) 비율 → hypothesis 신호 강도 sanity check.

#### F-A5. Confused-pair audio 예시 HTML
F-A2 에서 식별한 top-3 confused family pair 별로 $S_{W^+A^-}$ sample 들을 listed HTML viewer (audio 들어볼 수 있게). Phase B 에서 frame-level 로 들여다 볼 sample 후보 풀 선별용.
→ 청각적으로 정말 "음 비슷한가?" 직관 sanity check.

### 5.6 Phase A 결과로 알아내는 것

- $S_{W^+A^-}$ 의 *공간* (어떤 family pair, pitch range, source 에서 Whisper 가 강한가) 정의
- Phase B 가 frame-level 로 들여다 볼 *prototypical* sample 풀 선별 (전체 27k 다 안 봐도 됨)
- 만약 $S_{W^+A^-}$ 가 *random uniform* → prof hypothesis weak. *특정 패턴* → strong.

### 5.7 산출물

- 분석 스크립트: `experiments/audio_encoder_probe/phase_a_who_wins.py` (작성 예정)
- 결과: `experiments/audio_encoder_probe/_results/phase_a/`
  - `set_sizes.csv`, `dac_confusion_W+A-.csv`
  - `meta_distribution.csv` (set × attribute × value × count)
  - figures `F-A{1..4}.png`
  - `confused_pair_examples.html` (F-A5)
- meta-augmented manifest: `experiments/audio_encoder_probe/manifests/nsynth_train_30k_meta.csv`

---

## 6. Phase B. WHEN — frame-level activation 비교 (Phase A 후)

### 6.1 입력
- Phase A 에서 식별된 $S_{W^+A^-}$ 의 prototypical sample 풀 (수십~수백 utt)
- 이미 확보된 frame-level embedding: whisper_small `[T, 768]` (50Hz), dacvae `[T, 128]` (25Hz)

### 6.2 Frame-level linear probe
각 encoder, train fold 의 frame embedding 으로 logreg 학습 (5-fold CV).
- 입력: $h_t \in \mathbb{R}^{d_e}$ (모든 frame)
- 출력: $P_e(t, c) \in [0,1]^{11}$ — frame 별 class prob.
- frame 라벨 = 그 frame 이 속한 utt 의 family label.

### 6.3 Per-utt trajectory 추출
$S_{W^+A^-}$ 의 각 utt 마다:
- $P_W \in \mathbb{R}^{T_W \times 11}$ (Whisper, 50Hz)
- $P_D \in \mathbb{R}^{T_D \times 11}$ (DAC, 25Hz)

### 6.4 비교 metric (시간 native, sec 단위 비교)

| Metric | 정의 | 가설 |
|---|---|---|
| **Truth-prob gap** | $\Delta(t) = P_W(t, c^*) - P_D(t, c^*)$ | $S_{W^+A^-}$ 에서 *큰 양수*, 시간 분포 |
| **Argmax-time** | $t^*_e = \arg\max_t P_e(t, c^*)$ (sec) | Whisper 의 결정 시간 위치 |
| **Confidence sharpness** | $P_e(t^*_e, c^*) - \text{mean}_t P_e(t, c^*)$ | Whisper 가 더 sharp 한 confidence spike |
| **Frame agreement ratio** | $\frac{1}{T} \sum_t \mathbb{1}[\arg\max_c P_W(t,c) = \arg\max_c P_D(t,c)]$ | $S_{W^+A^-}$ 에서 낮음, $S_{W^+A^+}$ 에서 높음 |

### 6.5 시각화
- 2-panel timeline (utt 별): top = Whisper class-prob heatmap `[time × 11]`, bottom = DAC heatmap. 정답 class row 강조. 시간축 sec.
- aggregate: $S_{W^+A^-}$ vs $S_{W^+A^+}$ vs $S_{W^-A^-}$ 에서 metric 분포 violin/CDF 비교.

---

## 7. Phase C. WHY — audio-aligned semantic 해석

### 7.1 Decisive frame 분석
Phase B 에서 식별된 *결정 frame* ($t^*_W$) 위치에서:
- waveform / spectrogram 영역 inspection (onset transient? sustain harmonic? release?)
- 같은 시간 위치에서 DAC 의 prob entropy / norm — DAC 는 그 frame 에서 *무엇을* 안 보고 있는가

### 7.2 NSynth meta correlation
Phase A 의 meta 와 Phase B 의 frame metric 결합:
- "Whisper 가 winning 하는 sample 은 *attack 이 sharp* 한 sound 에 몰림" (velocity high + decisive frame at onset 의 결합 신호)
- "long_release quality sample 에서 Whisper 는 release 후반부에서 결정" (qualities tag 활용 가능 시)

### 7.3 결과 형태
> "Whisper 는 NSynth 에서 *low-pitch electronic source 의 onset transient* 를 결정 단서로 쓴다. DAC 는 그 frame 에서 평탄한 활성화를 보이며 (instrument family pair) 를 헷갈린다."

이 정도가 Q1 (언제/왜) 의 closing statement.

---

## 8. Phase D — Q2 (별도 문서)

Phase A–C 끝나면, 추가 정보의 *압축 가능성* 을 검증:
- Cross-encoder linear residual $r = W - \hat M D$
- Probe(r) accuracy on $S_{W^+A^-}$
- Sufficiency curve: $D \oplus r_{\text{top-K}}$ probe 의 K vs accuracy
- prof 의 "LoRA 로 추가 가능" 의 정량적 백킹

상세는 별도 문서 `whisper_vs_acoustic_residual_lora_sufficiency.md` (작성 예정).

---

## 9. 진행 순서 요약

- [x] 데이터 확보 (Phase A1–A2, Phase B 입력) — 모두 확보됨
- [x] **Phase A**: WHO wins — 분석 스크립트 작성 + figure 5개 산출 ✓ (2026-05-06)
  - W+A- = 12300/27218 (45.2%) — hypothesis 강함
  - top confusion: keyboard→mallet 501, reed→string 453, guitar→mallet 443
  - source χ²=440.7 p≈2e-96, pitch_bin χ²=46.4 p≈7e-8 (significant); velocity p=0.16 (not sig)
  - Whisper proba on W+A-: median=0.925 — confident, not marginal
- [ ] **Phase B**: WHEN — frame-level probe + trajectory 비교 ← *다음 작업*
- [ ] Phase C: WHY — audio-aligned semantic 해석
- [ ] Phase D: HOW — residual / LoRA sufficiency (= Q2)

## 10. 확장 옵션

- **wavtok_40_unify 합류**: Phase B 에서 합류 권장 — utt-mean + frame-level 추출 약 30분~1시간 GPU. 합류 시점은 Phase A 결과 본 후 결정.
- **NSynth qualities tag**: Phase C 에서 의미해석 풍부하게 하고 싶으면 NSynth json join.
- **다른 dataset 으로 확장**: NSynth 에서 hypothesis 입증되면 ESC50 / RAVDESS 로 확대 — speech 도 아니지만 paralinguistic emotion 까지 일반화 검증.
