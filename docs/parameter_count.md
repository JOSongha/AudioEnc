# Qwen3AE / Qwen3.5AE — Audio 파라미터 수

> 대상: `/mnt/fr20tb/audiollm/sanghyuk/Qwen3AE-4B`, `/mnt/fr20tb/audiollm/sanghyuk/Qwen3.5AE-4B`
> 측정 방법: `audio_encoder.py` 의 `AudioEncoder(AudioConfig)` 를 `config.json` 의 `audio_config` 그대로 인스턴스화 후, `sum(p.numel() for p in module.parameters())`.
> 두 variant 의 audio 쪽 구성은 동일 — DACVAE + AudioProjector 파라미터 수는 완전히 같음.

---

## 요약

**"로드된 파라미터 수" 와 "실제 forward 에 쓰이는 파라미터 수" 가 다름** — `audio_encoder.py` 가 DACVAE 전체(encoder+quantizer+decoder+watermarker) 를 인스턴스화 하지만, forward 경로는 `DACVAE.encode()` 만 호출하기 때문.

### A. 로드(=GPU 메모리 점유) 기준

| 구성 | 파라미터 |
|---|---|
| **AudioEncoder 전체** | **125,806,146 (~125.8M)** |
| └ `self.encoder` (DACVAE, 전체 로드, frozen) | 107,648,066 (~107.6M) |
| └ `self.projector` (AudioProjector, Stage1 학습 대상) | **18,158,080 (~18.2M)** |

`model.safetensors` 에도 DACVAE 255개 key 전부 실려있음 (decoder 162, encoder+quantizer 93). 체크포인트가 "watermarked DACVAE full model" 인 걸 그대로 박제한 결과.

### B. 실제 forward 에 쓰이는 파라미터 기준

| 구성 | 파라미터 |
|---|---|
| **Active 총합** | **45,709,440 (~45.7M)** |
| └ DACVAE encode path (encoder + `quantizer.in_proj`) | 27,551,360 (~27.6M) |
| └ AudioProjector | 18,158,080 (~18.2M) |

나머지 **80,096,706 (~80.1M)** 은 로드만 되고 한 번도 안 불리는 dead weight:
- `decoder.model` (upsampler, 70.66M)
- `decoder.wm_model` (watermarker, 9.33M)
- `quantizer.out_proj` (133K)

`AudioEncoder.forward()` 는 `self.encoder.encode(...)` 만 호출 →
→ `DACVAE.encode()` 는 `encoder(x) → quantizer.in_proj(z).chunk → _vae_sample` 로 끝 →
→ decoder/out_proj/watermarker 는 영원히 dead path.

### Projector 대조

Projector 18,158,080 은 PDF / notion 문서의 **18,158,080 (18M)** 표기와 정확히 일치.

### AudioEncoder = DACVAE + AudioProjector 인 이유

`audio_encoder.py` 의 `AudioEncoder.__init__`:

```python
self.encoder = DACVAE(...)       # from dacvae import DACVAE  — 전체 생성
# weight_norm 제거 (accelerate meta tensor 호환용)
for m in self.encoder.modules():
    try: nn.utils.remove_weight_norm(m)
    except ValueError: pass
self.projector = AudioProjector(config)
```

※ DACVAE 원본(`facebook/dacvae-watermarked` 로드시) 파라미터 107,671,171 대비 **23,105 감소** = `weight_norm` 의 `weight_g` scalar 파라미터 제거분 (실제 가중치에는 영향 없음, `weight_g` 를 `weight` 자체로 흡수).

### 숫자 어떻게 인용할지

- **모델 사이즈/체크포인트 크기** 문맥 → **A (125.8M)** — 저장/로드 되는 실제 물리량
- **연산량/FLOPs/메모리 효율** 문맥 → **B (45.7M)** — forward 에 관여하는 실질 크기
- **"audio encoder 가 몇 M이냐?"** 같은 모호한 경우 → 둘 다 병기, dead weight 80M 이유 한 줄 덧붙이는 게 안전

---

## DACVAE (frozen)

### Loaded config (config.json `audio_config`)

| key | 값 |
|---|---|
| `dac_sample_rate` | 48,000 |
| `dac_encoder_rates` | `[2, 8, 10, 12]` (hop = 1,920) |
| `dac_latent_dim` | 1,024 |
| `dac_codebook_dim` | 128 (= `audio_hidden_size`, projector 입력 차원) |
| encoder fps | 48000 / 1920 = **25** |

### Breakdown (DACVAE 원본, weight_norm 포함)

| 구성 | 파라미터 | 비율 |
|---|---|---|
| **total** | **107,671,171** | 100% |
| encoder | 27,288,704 | 25.3% |
| quantizer (VAEBottleneck) | 395,776 | 0.4% |
| decoder (all) | 79,986,691 | 74.3% |
| └ decoder.model (upsampler) | 70,658,208 | 65.6% |
| └ decoder.wm_model (Watermarker) | 9,328,483 | 8.7% |

### Inference-path 파라미터

AudioEncoder forward 는 DACVAE 의 `encode()` 만 호출 (decoder 경로 미사용).

| 경로 | 구성 | 파라미터 |
|---|---|---|
| **encode** (실제 활성) | encoder + `quantizer.in_proj` | **27,551,360 (~27.6M)** |
| decode (미사용) | `quantizer.out_proj` + decoder | 80,119,811 (~80.1M) |

DACVAE 는 Stage1/Stage2 모두 frozen — grad memory 0.
forward 에 관여하는 27.55M 만 실제로 audio waveform → latent 변환에 기여.

### encoder.block sub-layers

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

### quantizer (VAEBottleneck)

| sub | 파라미터 |
|---|---|
| `in_proj` (1024 → 2 × 128) | 262,656 |
| `out_proj` (128 → 1024) | 133,120 |
| **total** | **395,776** |

### decoder.model

| idx | 모듈 | 파라미터 |
|---|---|---|
| 0 | `NormConv1d` (128→1536) | 11,013,120 |
| 1 | `DecoderBlock` (stride=12) | 46,942,976 |
| 2 | `DecoderBlock` (stride=10) | 10,167,680 |
| 3 | `DecoderBlock` (stride=8) | 2,216,640 |
| 4 | `DecoderBlock` (stride=2) | 317,792 |
| | **total** | **70,658,208** |

### decoder.wm_model (Watermarker — 미사용 경로)

| sub | 파라미터 |
|---|---|
| `encoder_block` | 4,662,402 |
| `msg_processor` | 4,096 |
| `decoder_block` | 4,661,985 |
| **total** | **9,328,483** |

---

## AudioProjector (Stage1 학습 대상)

### Config (`audio_config`)

| key | 값 |
|---|---|
| `audio_hidden_size` | 128 (= DACVAE `codebook_dim`) |
| `adapter_hidden_size` | 512 |
| `num_adapter_layers` | 4 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 8 |
| `intermediate_size` | 2,048 |
| `llm_embed_size` | 2,560 |
| `head_dim` | 64 |

### 구성 (4-layer causal Llama decoder adapter)

```python
class AudioProjector(nn.Module):
    self.input_proj   = nn.Linear(128, 512, bias=False)             #   65,536
    self.layers       = nn.ModuleList([LlamaDecoderLayer(...) x 4]) # 16,781,312
    self.final_norm   = LlamaRMSNorm(512)                           #      512
    self.output_proj  = nn.Linear(512, 2560, bias=False)            #  1,310,720
```

### Breakdown

| 모듈 | 파라미터 | 비율 |
|---|---|---|
| **total** | **18,158,080** | 100% |
| `input_proj` (128 → 512) | 65,536 | 0.36% |
| `layers` (4 × LlamaDecoderLayer) | 16,781,312 | 92.4% |
| `final_norm` (RMSNorm 512) | 512 | < 0.01% |
| `output_proj` (512 → 2560) | 1,310,720 | 7.22% |

Layer 1개당 내부 구조 (512 hidden, 8 heads × 64 head_dim, 2048 intermediate, attention_bias=false):

| 서브모듈 | 계산 | 파라미터 |
|---|---|---|
| q_proj | 512 × 512 | 262,144 |
| k_proj | 512 × 512 | 262,144 |
| v_proj | 512 × 512 | 262,144 |
| o_proj | 512 × 512 | 262,144 |
| gate_proj | 512 × 2048 | 1,048,576 |
| up_proj | 512 × 2048 | 1,048,576 |
| down_proj | 2048 × 512 | 1,048,576 |
| input_layernorm | RMSNorm 512 | 512 |
| post_attention_layernorm | RMSNorm 512 | 512 |
| **layer total** | | **4,195,328** |

4 layer × 4,195,328 = **16,781,312** ✓

---

## Stage1 학습 파라미터

Stage1 workflow 는 `audio_encoder.projector` 를 제외한 모든 파라미터 freeze.

| 분류 | 파라미터 | Grad memory |
|---|---|---|
| Qwen3.5-4B LLM backbone | ~4B | frozen (0) |
| DACVAE encoder | 107.6M | frozen (0) |
| **AudioProjector** (학습 대상) | **18,158,080** | bf16 weight + fp32 grad + optimizer state |

---

## 재현 스크립트

```python
import sys, json, importlib.util
sys.path.insert(0, "AudioEnc/dacvae")   # dacvae 패키지 경로

MODEL_DIR = "/mnt/fr20tb/audiollm/sanghyuk/Qwen3.5AE-4B"
sys.path.insert(0, MODEL_DIR)

spec = importlib.util.spec_from_file_location("audio_encoder", f"{MODEL_DIR}/audio_encoder.py")
ae = importlib.util.module_from_spec(spec); spec.loader.exec_module(ae)

with open(f"{MODEL_DIR}/config.json") as f:
    audio_cfg = json.load(f)["audio_config"]

class AC: pass
ac = AC()
for k, v in audio_cfg.items(): setattr(ac, k, v)
ac.rope_scaling = None
ac._attn_implementation = "eager"

m = ae.AudioEncoder(ac)
total = sum(p.numel() for p in m.parameters())
proj  = sum(p.numel() for p in m.projector.parameters())
enc   = sum(p.numel() for p in m.encoder.parameters())
print(f"total: {total:,}   encoder: {enc:,}   projector: {proj:,}")
# total: 125,806,146   encoder: 107,648,066   projector: 18,158,080
```

---

## Audio encoder 비교 (대안 후보)

> Qwen3.5AE 아키텍처에서 DACVAE 를 다른 audio encoder 로 교체할 때 참고용.
> 모든 수치는 **encoder 만** 기준 (RVQ/decoder/watermarker 등 downstream 모듈 제외).
> 측정: `sum(p.numel() for p in model.encoder.parameters())`.

| 모델 | Encoder only | 원 모델 전체 | encoder 구성 | 출력 (dim / fps) | 입력 SR |
|---|---|---|---|---|---|
| **DACVAE** (`facebook/dacvae-watermarked`) | **27,551,360** (~27.55M) | 107,671,171 | Snake1d conv stack + VAE bottleneck in_proj | 128 / 25 | 48k |
| EnCodec 24k (`facebook/encodec_24khz`) | **7,425,792** (~7.43M) | 14,851,810 | SEANet conv stack (pre-RVQ) | 128 / 75 | 24k |
| EnCodec 48k (`facebook/encodec_48khz`) | **7,428,336** (~7.43M) | 14,856,294 | SEANet conv stack (pre-RVQ, stereo) | 128 / 150 | 48k |
| Mimi acoustic (`kyutai/mimi`, encoder only) | **12,628,256** (~12.63M) | 79,308,609 | SEANet conv stack (encoder_transformer 미통과) | 512 / 25 | 24k |
| Mimi semantic (`kyutai/mimi`, encoder + enc_transformer) | **37,818,656** (~37.82M) | 79,308,609 | SEANet conv + 8-layer transformer (downsample 스킵) | 512 / 25 | 24k |
| Whisper-tiny.en (`openai/whisper-tiny.en`) | **8,208,384** (~8.21M) | 37,760,256 | 4 layers, d=384, 6 heads, ffn=1536 | 384 / 50 | 16k |
| Whisper-base.en (`openai/whisper-base.en`) | **20,590,592** (~20.59M) | 72,593,408 | 6 layers, d=512, 8 heads, ffn=2048 | 512 / 50 | 16k |
| Whisper-small.en (`openai/whisper-small.en`) | **88,154,112** (~88.15M) | 241,734,144 | 12 layers, d=768, 12 heads, ffn=3072 | 768 / 50 | 16k |

※ Mimi 는 `encoder → downsample (1.05M) → encoder_transformer → quantizer` 가 원본 파이프라인. AudioEnc 는 `downsample` 을 건너뛰고 encoder_transformer 에 바로 넣음 (25 fps 유지 목적, Mimi 원 quantizer 는 12.5 fps). 원본대로 쓰면 encoder + downsample + encoder_transformer = **38,867,232 (~38.87M)**.

### Projector 영향

`AudioProjector` 는 4-layer LlamaDecoder adapter (hidden=512, 8 heads, ffn=2048) + 양끝 Linear. Encoder 교체 시 달라지는 건 **`input_proj` 한 레이어뿐**:

```python
self.input_proj  = nn.Linear(audio_hidden_size, 512, bias=False)   # encoder 에 따라 변동
self.layers      = [LlamaDecoderLayer() × 4]                         # 고정 16,781,312
self.final_norm  = LlamaRMSNorm(512)                                  # 고정 512
self.output_proj = nn.Linear(512, 2560, bias=False)                   # 고정 1,310,720 (LLM hidden 2560)
```

→ projector 고정 부분 = **18,092,544**
→ `input_proj` = `audio_hidden_size × 512`

### 종합 비교 (encoder + projector, Stage1 학습 규모 포함)

| 모델 | Enc output dim / fps | Encoder (frozen) | input_proj | Projector 전체 | **Active AudioEncoder** | **Stage1 학습 param** |
|---|---|---|---|---|---|---|
| **DACVAE** (현재) | 128 / 25 | 27,551,360 | 65,536 | 18,158,080 | **45,709,440 (~45.7M)** | **18,158,080** |
| EnCodec 24k | 128 / 75 | 7,425,792 | 65,536 | 18,158,080 | 25,583,872 (~25.6M) | 18,158,080 |
| EnCodec 48k | 128 / 150 | 7,428,336 | 65,536 | 18,158,080 | 25,586,416 (~25.6M) | 18,158,080 |
| Mimi acoustic | 512 / 25 | 12,628,256 | 262,144 | 18,354,688 | 30,982,944 (~30.98M) | 18,354,688 |
| Mimi semantic | 512 / 25 | 37,818,656 | 262,144 | 18,354,688 | 56,173,344 (~56.17M) | 18,354,688 |
| Whisper-tiny.en | 384 / 50 | 8,208,384 | 196,608 | 18,289,152 | 26,497,536 (~26.5M) | 18,289,152 |
| Whisper-base.en | 512 / 50 | 20,590,592 | 262,144 | 18,354,688 | 38,945,280 (~38.9M) | 18,354,688 |
| Whisper-small.en | 768 / 50 | 88,154,112 | 393,216 | 18,485,760 | 106,639,872 (~106.6M) | 18,485,760 |

### DACVAE 대비 비교

| 모델 | Active AudioEncoder | vs 현재 45.7M | Stage1 param | vs 현재 18.16M |
|---|---|---|---|---|
| EnCodec 24k/48k | 25.6M | **1.79× 감소** | 18.16M | 동일 |
| Whisper-tiny.en | 26.5M | **1.73× 감소** | 18.29M | +0.72% |
| Mimi acoustic | 30.98M | 1.47× 감소 | 18.35M | +1.09% |
| Whisper-base.en | 38.9M | 1.17× 감소 | 18.35M | +1.09% |
| Mimi semantic | 56.17M | 1.23× 증가 | 18.35M | +1.09% |
| Whisper-small.en | 106.6M | 2.33× 증가 | 18.49M | +1.80% |

→ Projector 쪽 변동은 encoder 선택 무관하게 **±2% 이하**. Stage1 memory/optimizer cost 는 실질 동일.
→ 차이 대부분은 **frozen encoder** 쪽 (forward latency + GPU 상주 메모리).

### 재현 스크립트

**Whisper**:
```python
from transformers import WhisperModel
for repo in ["openai/whisper-tiny.en", "openai/whisper-base.en", "openai/whisper-small.en"]:
    m = WhisperModel.from_pretrained(repo)
    print(repo, sum(p.numel() for p in m.encoder.parameters()))
```

**EnCodec** (encoder = pre-RVQ continuous representation):
```python
from transformers import EncodecModel
for repo in ["facebook/encodec_24khz", "facebook/encodec_48khz"]:
    m = EncodecModel.from_pretrained(repo)
    print(repo, sum(p.numel() for p in m.encoder.parameters()))
# encoder = raw waveform → 128-dim 연속 표현 (RVQ 통과 전)
# HF impl 의 quantizer 는 codebook 을 nn.Buffer 로 갖기 때문에 parameters() 에는 안 잡힘 (0 반환)
```

**Mimi** (acoustic = encoder only, semantic = encoder + encoder_transformer):
```python
from transformers import MimiModel
m = MimiModel.from_pretrained("kyutai/mimi")
acoustic = sum(p.numel() for p in m.encoder.parameters())
semantic = acoustic + sum(p.numel() for p in m.encoder_transformer.parameters())
full     = semantic + sum(p.numel() for p in m.downsample.parameters())  # 원본 파이프라인
print(f"acoustic(enc): {acoustic:,}  semantic(enc+trans): {semantic:,}  full(+downsample): {full:,}")
# acoustic: 12,628,256   semantic: 37,818,656   full: 38,867,232
```
