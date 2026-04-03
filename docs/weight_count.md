# 파라미터 수 계산

> 기준 모델: `Qwen/Qwen3.5-2B` (hidden_size = **2048**)

---

## Projector 구조 복습

```python
# proj_strides = [2, 2] (encodec / dac / fb_dacvae / mimi_acoustic)
Conv1d(out_dim, 2048, kernel_size=5)   # 1st strided
GELU()
Conv1d(2048,    2048, kernel_size=5)   # 2nd strided
GELU()
Conv1d(2048,    2048, kernel_size=1)   # channel mix (stride=1)
LayerNorm(2048)

# proj_strides = [2] (mimi_semantic)
Conv1d(out_dim, 2048, kernel_size=5)   # 1st strided
GELU()
Conv1d(2048,    2048, kernel_size=1)   # channel mix (stride=1)
LayerNorm(2048)
```

Conv1d 파라미터 수 = `in_ch × out_ch × kernel_size + out_ch` (bias 포함)  
LayerNorm 파라미터 수 = `dim × 2` (weight + bias)

---

## Projector 파라미터 수 (encoder별)

### 공통 레이어 (out_dim 무관)

| 레이어 | 계산 | 파라미터 |
|---|---|---|
| Conv1d(2048→2048, k=5) | 2048×2048×5 + 2048 | **20,973,568** |
| Conv1d(2048→2048, k=1) | 2048×2048×1 + 2048 | **4,196,352** |
| LayerNorm(2048) | 2048×2 | **4,096** |

### 1st Conv1d (out_dim별)

| encoder | out_dim | 계산 | 파라미터 |
|---|---|---|---|
| encodec | 128 | 128×2048×5 + 2048 | 1,312,768 |
| dac | 1024 | 1024×2048×5 + 2048 | 10,487,808 |
| fb_dacvae | 8 | 8×2048×5 + 2048 | 83,968 |
| mimi_acoustic | 512 | 512×2048×5 + 2048 | 5,244,928 |
| mimi_semantic | 512 | 512×2048×5 + 2048 | 5,244,928 |

### 총합 (projector + proj_norm)

| encoder | proj_strides | 합계 |
|---|---|---|
| encodec | [2, 2] | 1,312,768 + 20,973,568 + 4,196,352 + 4,096 = **26,486,784 (~26.5M)** |
| dac | [2, 2] | 10,487,808 + 20,973,568 + 4,196,352 + 4,096 = **35,661,824 (~35.7M)** |
| fb_dacvae | [2, 2] | 83,968 + 20,973,568 + 4,196,352 + 4,096 = **25,257,984 (~25.3M)** |
| mimi_acoustic | [2, 2] | 5,244,928 + 20,973,568 + 4,196,352 + 4,096 = **30,418,944 (~30.4M)** |
| mimi_semantic | [2] | 5,244,928 + 4,196,352 + 4,096 = **9,445,376 (~9.4M)** |

mimi_semantic이 절반 이하인 이유: stride-2 Conv1d(k=5)가 1개뿐 (20.97M 레이어 없음).

---

## CTC Head (--debug c)

```python
nn.Linear(2048, 28)   # blank=0, a-z=1-26, space=27
```

| 항목 | 계산 | 파라미터 |
|---|---|---|
| weight | 2048 × 28 | 57,344 |
| bias | 28 | 28 |
| **합계** | | **57,372** |

dtype: `float32` (`ctc_head.float()` — model.py `init_ctc_head()`)  
메모리: 57,372 × 4 bytes = **229,488 bytes (~0.22 MB)**

### Projector 대비 CTC head 비중

| encoder | projector | CTC head | 비중 |
|---|---|---|---|
| mimi_semantic | 9.4M | 57K | **0.61%** |
| encodec | 26.5M | 57K | **0.22%** |
| fb_dacvae | 25.3M | 57K | **0.23%** |
| mimi_acoustic | 30.4M | 57K | **0.19%** |
| dac | 35.7M | 57K | **0.16%** |

---

## Stage 1 학습 파라미터 요약

Stage 1에서 LLM은 frozen. 학습 대상: projector + proj_norm (+ CTC head if `--debug c`).

| encoder | without CTC | with CTC (`--debug c`) | 차이 |
|---|---|---|---|
| encodec | 26.5M | 26.5M + 57K | +0.22% |
| dac | 35.7M | 35.7M + 57K | +0.16% |
| fb_dacvae | 25.3M | 25.3M + 57K | +0.23% |
| mimi_acoustic | 30.4M | 30.4M + 57K | +0.19% |
| mimi_semantic | 9.4M | 9.4M + 57K | +0.61% |
