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

---

## DACVAE (`facebook/dacvae-watermarked`) 파라미터 수

> 측정: `DACVAE.load("facebook/dacvae-watermarked")` 후 `sum(p.numel() for p in ...)`
> 전체 frozen — AudioEnc에서는 `self.dacvae.encode()` (encoder + quantizer.in_proj + reparam) 경로만 사용.

### Loaded config

| 항목 | 값 |
|---|---|
| `sample_rate` | 48,000 |
| `hop_length` | 1,920 (encoder_rates product) |
| `encoder_rates` | `[2, 8, 10, 12]` |
| `decoder_rates` | `[12, 10, 8, 2]` |
| `encoder_dim` | 64 |
| `latent_dim` | 1024 |
| `decoder_dim` | 1536 |
| `quantizer.codebook_dim` | 128 |
| encoder fps | 48000 / 1920 = **25** |

※ `fb_dacvae.py`의 주석 `codebook_dim # e.g. 8`은 오래된 값. 실제 pretrained는 **128**.
※ CLAUDE.md encoder registry의 `fps ~86` 도 잘못된 표기 — 실제는 25 fps (mimi와 동일).

### 전체 합계

| 구성 | 파라미터 | 비율 |
|---|---|---|
| **DACVAE 전체** | **107,671,171 (~107.7M)** | 100% |
| encoder | 27,288,704 (~27.3M) | 25.3% |
| quantizer (VAEBottleneck) | 395,776 (~0.40M) | 0.37% |
| decoder (all) | 79,986,691 (~80.0M) | 74.3% |
| └ decoder.model (upsampler) | 70,658,208 (~70.7M) | 65.6% |
| └ decoder.wm_model (Watermarker) | 9,328,483 (~9.3M) | 8.7% |

### Inference-path 분해

| 경로 | 구성 | 파라미터 |
|---|---|---|
| encode (AudioEnc가 쓰는 경로) | encoder + `quantizer.in_proj` | **27,551,360 (~27.6M)** |
| decode (AudioEnc에서 미사용) | `quantizer.out_proj` + decoder | 80,119,811 (~80.1M) |

AudioEnc 학습 시 DACVAE은 frozen이므로 grad 메모리는 0.
순수 forward 파라미터 메모리: 27.6M × 2 bytes (bf16) = **~55 MB** (FSDP 환경 기준).

### 세부 분해

**encoder.block**

| idx | 모듈 | 파라미터 |
|---|---|---|
| 0 | `NormConv1d` (1→64, k=7) | 576 |
| 1 | `EncoderBlock` (stride=2) | 132,544 |
| 2 | `EncoderBlock` (stride=8) | 920,448 |
| 3 | `EncoderBlock` (stride=10) | 4,200,192 |
| 4 | `EncoderBlock` (stride=12) | 18,886,144 |
| 5 | `Snake1d(1024)` | 1,024 |
| 6 | `NormConv1d` (1024→1024, k=3) | 3,147,776 |
| | **total** | **27,288,704** |

**quantizer (`VAEBottleneck`, latent=1024 → codebook_dim=128)**

| sub | 파라미터 |
|---|---|
| `in_proj` (1024 → 2×128) | 262,656 |
| `out_proj` (128 → 1024) | 133,120 |
| **total** | **395,776** |

**decoder.model** (latent → audio)

| idx | 모듈 | 파라미터 |
|---|---|---|
| 0 | `NormConv1d` (128→1536) | 11,013,120 |
| 1 | `DecoderBlock` (stride=12) | 46,942,976 |
| 2 | `DecoderBlock` (stride=10) | 10,167,680 |
| 3 | `DecoderBlock` (stride=8) | 2,216,640 |
| 4 | `DecoderBlock` (stride=2) | 317,792 |
| | **total** | **70,658,208** |

**decoder.wm_model** (Watermarker — AudioEnc 경로에서는 미사용)

| sub | 파라미터 |
|---|---|
| `encoder_block` | 4,662,402 |
| `msg_processor` | 4,096 |
| `decoder_block` | 4,661,985 |
| **total** | **9,328,483** |

### 재현 스크립트

```python
import sys
sys.path.insert(0, "AudioEnc/dacvae")
from dacvae import DACVAE

m = DACVAE.load("facebook/dacvae-watermarked").eval()
total = sum(p.numel() for p in m.parameters())
print(f"total: {total:,}")  # 107,671,171
```
