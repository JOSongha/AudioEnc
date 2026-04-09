# EOS Token Loss 분석 보고서

분석 대상: Stage 1 projector checkpoint (fb_dacvae + CTC)

- 참고 파일: `eos_loss_debug_fb_dacvae_ctc_*.csv`
- `pred_token`: free generation 출력

---

## 분석 방법론

### 신뢰 가능한 지표

| 지표 | 신뢰성 | 의미 |
|---|---|---|
| `tf_loss` at GT EOS | ✅ | GT context가 주어졌을 때 EOS를 얼마나 잘 예측하는가 |
| `pred_eos_pos` vs `gt_eos_pos` | ✅ | Free generation에서 EOS를 언제 출력하는가 |
| `free_loss` at GT EOS | ❌ | 루프 artifact로 오염됨 (사용 안 함) |

### `free_loss` at EOS를 신뢰할 수 없는 이유

Free generation 루프는 EOS 예측 시 멈추지 않는다.
모델이 step k에서 EOS를 예측하면 그 EOS가 context에 쌓이고, 이후 step들도 EOS를 계속 예측하는 경향이 생긴다.
이는 "EOS 남발"이 아니라 루프 설계의 artifact다.
GT EOS 위치에서의 `free_loss`는 모델이 EOS를 일찍 예측했는지 여부에 따라 trivially 낮아지거나 무관한 값이 된다.

---

## 실험 설정

| 실험 | 디렉토리 | EOS 가중 방식 |
|---|---|---|
| **baseline** | `debug_fb_dacvae_ctc` | EOS 가중 없음 |
| **eos_upweight** | `debug_fb_dacvae_ctc_0401_2023` | EOS loss 상시 upweighting (×3.0) |
| **eos_decay** | `debug_fb_dacvae_ctc_0402_1156` | EOS loss → 1.0으로 linear decay |

---

## 실험 1 (예비): 5샘플, max_tokens=50

### 배경

초기 분석은 5샘플, 최대 50토큰으로 수행됐다. GT EOS가 50토큰 이내에 등장한 샘플은 각 2개뿐이었다.

### tf_loss at GT EOS (n=2)

| 실험 | sample 0 | sample 4 | 평균 |
|---|---|---|---|
| baseline | 12.38 | 9.94 | **11.16** |
| eos_upweight | 0.60 | 1.21 | **0.91** |
| eos_decay | 7.38 | 1.18 | **4.28** |

### pred_eos_pos vs gt_eos_pos (5샘플)

| 실험 | sample 0 (gt=48) | sample 4 (gt=46) |
|---|---|---|
| baseline | pred=25, diff=**-23** | pred=34, diff=**-12** |
| eos_upweight | pred=34, diff=**-14** | pred=**없음** |
| eos_decay | pred=46, diff=**-2** | pred=45, diff=**-1** |

### 5샘플 분석의 한계

- `tf_loss` 비교 대상이 2개뿐 — 통계적 신뢰성 낮음
- eos_upweight의 tf_loss=0.91은 인상적이나 N이 너무 작음
- EOS 타이밍 지표에서는 eos_decay가 명확히 우수 (diff=-1~2)

---

## 실험 2 (본실험): 50샘플, max_tokens=100

LibriSpeech train-clean-100에서 첫 50개 샘플. 전 샘플에서 GT EOS가 100토큰 이내 등장 (실제 최장 70토큰).

### 1. tf_loss 요약 (n=50, 전체 EOS 토큰)

| 실험 | EOS tf_loss | non-EOS tf_loss |
|---|---|---|
| baseline | **2.0821** | 7.1930 |
| eos_upweight | 2.2645 | 7.2783 |
| eos_decay | 2.3688 | 7.5259 |

- 세 조건 모두 EOS tf_loss 차이가 작음 (2.08 ~ 2.37)
- 5샘플에서 보인 eos_upweight의 압도적 우위(0.91)는 재현되지 않음 — **5샘플 결과는 샘플 선택 편향**
- non-EOS tf_loss도 같은 순서로 증가 → 실험 간 전반적인 loss 수준 차이 반영

### 2. EOS 생성 타이밍 (pred_eos_pos vs gt_eos_pos)

| 실험 | pred 있음 / 50 | missed | false alarm | avg diff (pred-gt) | \|diff\|≤3 비율 |
|---|---|---|---|---|---|
| baseline | 40 | **10** | 0 | **-15.70** | 10/40 (25.0%) |
| eos_upweight | 42 | 8 | 0 | -13.12 | 10/42 (23.8%) |
| eos_decay | **44** | **6** | 0 | **-11.18** | **12/44 (27.3%)** |

### diff 분포 (pred - gt)

| 구간 | baseline | eos_upweight | eos_decay |
|---|---|---|---|
| early >5스텝 | 28 | 29 | **32** |
| early 1~5스텝 | 4 | 3 | 2 |
| exact (0) | **8** | **10** | **10** |
| late 1~5스텝 | 0 | 0 | 0 |
| late >5스텝 | 0 | 0 | 0 |

---

## 분석

### baseline

- EOS tf_loss는 3개 중 가장 낮지만 (2.08), 실제 EOS 생성은 가장 나쁨
- 50샘플 중 10개에서 EOS를 아예 생성 못함 (missed)
- 예측한 40개 중에서도 평균 15.7스텝 조기 종료

### eos_upweight

- EOS tf_loss 소폭 상승 (2.26), 타이밍은 baseline보다 약간 개선
- missed 8개, avg diff -13.12
- 5샘플에서 보인 tf_loss 우위는 대표성 부족이었던 것으로 결론

### eos_decay

- EOS tf_loss가 가장 높지만 (2.37), 차이는 small
- **EOS 생성 비율이 가장 높음**: 44/50 (missed 6개)
- **타이밍이 가장 정확**: avg diff -11.18, |diff|≤3 비율 27.3%
- 그러나 절대적으로는 여전히 11스텝 조기 종료 — 완전한 해결은 아님

---

## 핵심 관찰

```
tf_loss가 낮다 ≠ EOS를 올바른 위치에서 생성한다
```

| 측면 | baseline | eos_upweight | eos_decay |
|---|---|---|---|
| EOS 예측 능력 (tf_loss) | 가장 낮음 (2.08) | 중간 (2.26) | 가장 높음 (2.37) |
| EOS 생성 비율 | 나쁨 (40/50) | 중간 (42/50) | **좋음 (44/50)** |
| EOS 타이밍 정확도 | 나쁨 (-15.7) | 중간 (-13.1) | **좋음 (-11.2)** |
| false alarm | 없음 | 없음 | 없음 |

EOS upweighting은 EOS를 더 자주, 더 적절한 위치에 생성하도록 유도하지만,
tf_loss 지표만으로는 이 효과가 잘 드러나지 않는다.
**타이밍 지표(pred_eos_pos - gt_eos_pos)가 EOS 품질 평가에 더 적합한 지표.**

---

## 미해결 문제

- **전반적인 조기 종료**: 모든 조건에서 avg diff -11 ~ -16스텝. EOS upweighting만으로는 부족.
  - 원인 후보: projector가 오디오 길이 정보를 충분히 인코딩하지 못함
  - Stage 1 한계일 가능성 (LLM frozen → EOS 위치 추론 능력 제한)
- **tf_loss와 타이밍 지표의 역전**: EOS upweighting이 tf_loss를 오히려 올리는 경향
  - 가능한 설명: upweighting이 EOS 자체보다 전체 분포에 영향을 줌

---

## 결론

- eos_upweight의 5샘플 tf_loss=0.91은 **샘플 편향**으로, 50샘플에서 재현되지 않음
- **eos_decay가 EOS 생성 행동(타이밍 + 비율) 측면에서 일관되게 우수**
- 그러나 전체적으로 Stage 1 모델의 EOS 위치 정확도는 아직 낮음 (-11스텝)
- 새 EOS 토큰(`<|asr_eos|>`) 실험은 eos_decay 전략과 병행하는 것이 가장 유의미함
- 향후 Stage 2 (LLM fine-tuning)에서 EOS 타이밍이 얼마나 개선되는지 추적 필요
