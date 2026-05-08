# WavTokenizer (SEANetEncoder) + Qwen3.5AE-4B — Stage1 학습 세팅

> 목표: 현재 `Qwen3.5AE-4B` 아키텍처에서 audio encoder 만 **DACVAE → WavTokenizer SEANetEncoder** 으로 교체, Stage1 (projector-only) 학습.
> LLM, AudioProjector 구조는 그대로 유지. Encoder output dim 만 128 → 512 로 바뀜.
> ※ WavTokenizer SEANetEncoder 는 codec reconstruction 목표로 학습된 모델 — **ASR semantic** (Whisper) 과 **acoustic codec** (DAC) 의 중간 지점. 둘의 trade-off 연구 목적.

---

## 1. 아키텍처 비교

### 현재 (DACVAE)

```
[48kHz wav] → DACVAE.encode()
            → (B, T, 128)  @ 25 fps  (hop=1920 samples)
            → AudioProjector (input_proj: 128→512, 4x LlamaDecoderLayer, output_proj: 512→2560)
            → (B, T, 2560)  → Qwen3.5-4B
```

### Whisper-small.en (참고)

```
[16kHz wav] → WhisperFeatureExtractor (mel)
            → WhisperEncoder  (12 layers, d_model=768)
            → last_hidden_state (B, 1500, 768)  @ 50 fps
            → AudioProjector (768→512→2560)
            → (B, T_audio, 2560)  → Qwen3.5-4B
```

### 변경 (WavTokenizer SEANetEncoder)

```
[24kHz wav] → SEANetEncoder (frozen, pre-VQ z_e tap)
            → (B, 512, T)  @ 40 fps  (hop=600 samples)
            → transpose → (B, T, 512)
            → AudioProjector (input_proj: 512→512, 4x LlamaDecoderLayer, output_proj: 512→2560)
            → (B, T_audio, 2560)  → Qwen3.5-4B
```

---

## 2. 파라미터 수 비교

| 구성 | DACVAE | Whisper-small.en | WavTokenizer-40 | WavTokenizer-75 | 비고 |
|------|--------|------------------|-----------------|-----------------|------|
| Audio encoder (active) | 27.55M | 88.15M | **8.30M** | **8.80M** | 75fps는 ratios 변경으로 소폭 증가 |
| Tap point | VAE latent `z̃` (post-quant) | `last_hidden_state` | pre-VQ `z_e` | pre-VQ `z_e` | 동일 |
| Sample rate | 48kHz | 16kHz | **24kHz** | **24kHz** | 동일 |
| Frame rate (fps) | 25 | 50 | **40** | **75** | hop_length = sample_rate / fps |
| 30s sequence length | 750 frames | 1500 frames | **1200 frames** | **2250 frames** | 75fps는 Whisper의 1.5배 |
| **Stage1 학습 파라미터 (projector)** | 18.16M | 18.49M | **18.43M** (512→512) | **18.35M** (512→512) | 거의 동일 |

---

## 3. Tap point 결정 — Pre-VQ z_e

### 왜 z_e (pre-VQ)?

WavTokenizer 는 다단계 처리:
1. **SEANetEncoder** → output `z_e` ∈ ℝ^(512×T)
2. **VQ bottleneck** → codes (discrete, 8-bit)
3. **SEANetDecoder** → reconstruction

**Pre-VQ z_e 선택 이유**:
- **정보 보존**: post-VQ codes 는 1/8 압축 (128 codebook, 8-bit), 정보 손실 큼
- **Whisper `last_hidden_state` 와 parallel**: Whisper 도 최종 layer 의 연속 출력을 사용 (unbottlenecked)
- **공정한 비교**: DAC 의 "128-dim VAE latent" 대비 DAC 와 동일한 "연속 표현" 축으로 정렬
- **State dict 간결함**: 없음 (encoder 동결, weight_norm 제거만 필요)

### 대안 고려 (향후)

**Post-VQ z_q (codes)**: `[B, 8, T]` (8개 codebook, 각 frame)
- 잠재공간이 이산 → 별도 embedding layer 필요
- 한 줄 변경 (config 플래그): `use_post_vq=True` 시 `encoder.encode_infer()` 호출로 전환 가능
- 향후 ablation 목적으로 A/B 테스트 가능

---

## 4. 체크포인트 선택 — `wavtokenizer_large_unify_600_24k`

### 왜 이 체크포인트?

**WavTokenizer 는 여러 변형 배포** (frame rate, 데이터 도메인):

| 변형 | Frame rate (fps) | Hop (samples @ 24k) | 데이터 도메인 | 학습량 | 비고 |
|------|------------------|---------------------|--------------|--------|------|
| large-unify-40token | **40** | **600** | Speech + Audio + Music (General) | 150k h | **구현 완료** |
| large-speech-75token | **75** | **320** | Speech + Audio + Music | **80k h** | **구현 완료** |
| small-600-24k-4096 | 40 | 600 | LibriTTS (600h) | 600h | 향후 ablation |

> **주의 — HF 이름 오해 소지**: `novateur/WavTokenizer-large-speech-75token` 이름의 "speech"는 speech-only를 의미하지 **않는다**.
> GitHub README 표(WavTokenizer-large-320-24k-4096 행) 의 Domain 컬럼은 **"Speech, Audio, Music"** 으로 명시되어 있으며,
> 이는 40fps unify와 동일한 멀티도메인 데이터로 학습된 것. 다만 학습량이 150k h → **80k h 로 적음**.
> HuggingFace 모델 카드에는 학습 데이터 상세가 없으므로, GitHub README가 권위 있는 출처.
>
> **Ablation 설계 시사점**: 40fps vs 75fps 비교에서 domain confound는 없지만, **데이터 규모 차이(150k vs 80k h)** 는 confound로 남음.

**선택 기준**:

1. **Frame rate 정렬 (40 fps)**
   - DAC: 25 fps (hop=1920@48k), Whisper: 50 fps (hop=320@16k), WavTok: 40 fps
   - 40 fps 는 DAC(25) 와 Whisper(50) 의 중간
   - **중요**: 30s @ 40fps = 1200 frames, 이는 Whisper(1500) 과 유사한 LLM token budget

2. **"Unify" 데이터 도메인**
   - General 150k hours (SUPERB benchmark 평가 기준)
   - "Acoustic encoder" 프레이밍 → 도메인 agnostic
   - vs "speech-only" (LibriTTS만)

3. **체크포인트 가용성**
   - 공식 배포됨: [jishengpeng/WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
   - 다운로드 위치: `/mnt/tmp/hf_cache/wavtokenizer/wavtokenizer_large_unify_600_24k.ckpt` (1.7 GB, PyTorch Lightning) — *예시 경로. 본 노드(jos)에는 미존재; 사용 시 직접 download.*

> **상태 (2026-04-26)**: WavTok-40-unify 구현 완료. safetensors 생성, Stage1 학습 중 (100k steps, step ~400+).
> **상태 (2026-05-08)**: WavTok-75-speech 구현 완료. `novateur/WavTokenizer-large-speech-75token` 다운로드 → `convert_to_wavtok75.py` 실행 → safetensors 생성. yaml/script 준비 완료, 학습 미시작.

---

## 5. Architecture Verified Facts

### SEANetEncoder 스펙

| 항목 | WavTok-40-unify | WavTok-75-speech |
|------|-----------------|------------------|
| Input shape | `[B, 1, S]` (mono waveform) @ 24kHz | 동일 |
| Output shape | `[B, 512, T]` where T = S // **600** | `[B, 512, T]` where T = S // **320** |
| Input channels | 1 (mono) | 동일 |
| Feature dimension (z_e) | **512** | **512** |
| n_filters (base) | 32 | 32 |
| Downsampling ratios | **[6, 5, 5, 4]** → total 600× | **[8, 5, 4, 2]** → total 320× |
| LSTM layers | 2 | 2 |
| Weight parametrization | `weight_norm` (weight_g + weight_v) | 동일 |
| **Active parameters** | **8,301,760 (~8.30M)** | **8,797,568 (~8.80M)** |
| **Encoder keys in ckpt** | 62 keys | 62 keys |
| Checkpoint | `wavtokenizer_large_unify_600_24k.ckpt` | `wavtokenizer_large_speech_320_v2.ckpt` |

### Weight normalization 처리

WavTokenizer checkpoint 는 `weight_norm` 으로 저장:
```
feature_extractor.encodec.encoder.conv_pre_g  [1, 32]     ← scalar per channel
feature_extractor.encodec.encoder.conv_pre_v  [32, 1, 15] ← weight matrix
```

**변환 공식**: `weight = weight_g * weight_v / ||weight_v||_{L2,dims≠0}`

Convert script 에서 자동 병합 후, model state_dict 에는 plain `weight` 로만 저장. 이렇게 하면:
- Accelerate meta-tensor 호환성 (weight_norm 은 문제 일으킬 수 있음)
- State dict 간결 (g, v 쌍 대신 단일 weight)

---

## 6. 데이터 처리

### Sample rate 변환

Stage1 manifest (LibriTTS-R + MLS + VoxPopuli) 의 원본 sample rates:
- LibriTTS-R: **24kHz** → WavTok native, 변환 불필요 ✓
- MLS: **16kHz** → dataloader 에서 24kHz resample 필요
- VoxPopuli: **16kHz** → 동상

기존 `src/llamafactory/data/audio_io.py:load_audio_chunk()` 는 이미 configurable `target_sr` 지원.

### Audio token 개수 계산

```python
audio_len_samples = waveform.shape[-1]  # 24kHz, seconds × 24000
t_audio = audio_len_samples // 600      # hop_length = 600
```

YAML 에서 설정:
```yaml
omni_sample_rate: 24000
omni_hop_length: 600
omni_max_audio_samples: 1080000  # 45s @ 24kHz (24000 × 45)
```

### 데이터 라우팅

Convert script 에서 `config.whisper_model_id` 필드를 **생략** (DAC 도 생략).
→ `src/llamafactory/train/omni/workflow.py:80` 의 조건:
```python
if hasattr(config.audio_config, 'whisper_model_id') and config.audio_config.whisper_model_id:
    # Whisper branch
else:
    # DAC / WavTok branch (둘 다 raw waveform)
```

이미 존재하는 `create_omni_processor()` 가 DAC waveform 처리를 하는데, 이미 `omni_sample_rate`/`omni_hop_length` 를 사용하므로 **인프라 코드 변경 없음** ✓

---

## 7. Convert script (`convert_to_wavtok.py`)

기본 흐름:
1. Base Qwen3.5AE-4B 로드 (LLM weight source)
2. WavTok AudioConfig 로 새 모델 인스턴스화 (SEANetEncoder random init)
3. LLM weight 로드 (audio_encoder.* 제외)
4. WavTokenizer checkpoint 로드 → weight_norm 병합 → SEANetEncoder 에 로드
5. Projector 는 random init 유지 (Stage1 에서 학습)
6. save_pretrained → safetensors

**관련 코드 파일 필요**:
- `wavtokenizer_modules/` (seanet.py, conv.py, norm.py, lstm.py, __init__.py)
- `configuration_qwen3_5AE.py` (WavTok-specific `AudioConfig`)
- `audio_encoder.py` (SEANetEncoder integration)
- `modeling_qwen3_5AE.py` (복사, encoder-agnostic)
- `tokenization_*.py`, 토크나이저 aux files (복사)

---

## 8. Stage1 yaml 설정

### WavTok-40-unify

```yaml
### model
model_name_or_path: /path/to/Qwen3.5AE-4B-wavtok-40-unify
stage: omni  # freezes all except audio_encoder.projector

### dataset (WavTok-specific)
omni_sample_rate: 24000
omni_hop_length: 600
omni_max_audio_samples: 1080000  # 45s @ 24kHz → 1800 tokens

### train
per_device_train_batch_size: 4
learning_rate: 2.0e-4
max_steps: 100000
warmup_steps: 1000
cutoff_len: 3584
```

### WavTok-75-speech

```yaml
### model
model_name_or_path: /path/to/Qwen3.5AE-4B-wavtok-75-speech
stage: omni

### dataset (75fps-specific)
omni_sample_rate: 24000
omni_hop_length: 320           # 75 fps
omni_max_audio_samples: 576000 # 24s @ 24kHz → 1800 tokens (75fps × 24s = 40fps × 45s)

### train
per_device_train_batch_size: 4
learning_rate: 2.0e-4
max_steps: 100000
warmup_steps: 1000
cutoff_len: 3584
```

> **Token budget 유지**: 75fps × 24s = 1800 tokens = 40fps × 45s. `omni_max_audio_samples` 를 576000(24s)으로 줄여 LLM sequence length 동일하게 유지.

---

## 9. 비교 표: DAC vs Whisper vs WavTok @ 30s

| 항목 | DAC | Whisper-small | WavTok-40-unify | WavTok-75-speech | 분석 |
|------|-----|----------------|-----------------|------------------|------|
| **Encoder** | DACVAE encoder | Whisper encoder (12-layer) | SEANetEncoder | SEANetEncoder | codec, ASR, hybrid |
| **Encoder params** | 27.55M | 88.15M | **8.30M** | **8.80M** | 75fps는 소폭 증가 |
| **Objective** | Reconstruction (VAE) | Speech recognition | Reconstruction (codec) | Reconstruction (codec) | codec vs semantic |
| **Sample rate** | 48kHz | 16kHz | **24kHz** | **24kHz** | 동일 |
| **Frame rate** | 25 fps | 50 fps | **40 fps** | **75 fps** | 2배 차이 |
| **Tap point** | post-VQ z̃ (128-dim) | last_hidden_state (768-dim) | pre-VQ z_e (512-dim) | pre-VQ z_e (512-dim) | 동일 |
| **Downsampling ratios** | — | — | [6,5,5,4] hop=600 | [8,5,4,2] hop=320 | 아키텍처 차이 |
| **30s → frames** | 750 | 1500 | **1200** | **2250** | 75fps: Whisper의 1.5배 |
| **Max clip (token budget=1800)** | — | — | 45s | **24s** | 75fps: 짧은 클립 처리 |
| **Projector params** | 18.16M | 18.49M | **18.43M** (512→512) | **18.35M** (512→512) | 거의 동일 |
| **Stage1 training** | 기존 | 기존 + yaml 변경 | 학습 중 (~step 400) | **yaml/script 완료, 미시작** | |

---

## 10. Verification Checklist

### Convert script smoke test

```bash
python external/models/Qwen3.5AE-4B-wavtok-40-unify/convert_to_wavtok.py \
    --source external/models/Qwen3.5AE-4B \
    --wavtok-ckpt /mnt/tmp/hf_cache/wavtokenizer/wavtokenizer_large_unify_600_24k.ckpt \
    --dry-run
```

**확인 사항**:
- 0 missing non-audio keys (LLM 모두 로드됨)
- 62 encoder keys 병합 완료
- Param breakdown: ~4B LLM + 8.30M encoder + ~18M projector

### Sanity check script

```bash
python external/models/Qwen3.5AE-4B-wavtok-40-unify/sanity_check.py
```

**확인 사항**:
- [2/3] SEANetEncoder weights 와 WavTok checkpoint 일치 (rtol=1e-5)
- [3/3] Projector init non-trivial (std > 1e-3)

### Forward shape assertion

```python
from transformers import AutoModelForCausalLM
import torch

m = AutoModelForCausalLM.from_pretrained(
    'external/models/Qwen3.5AE-4B-wavtok-40-unify',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()

audio = torch.randn(1, 1, 72000, dtype=torch.bfloat16, device='cuda')  # 3s @ 24kHz
out = m.model.audio_encoder(audio)
assert out.shape == (1, 120, 2560), f"Expected (1, 120, 2560), got {out.shape}"
# 72000 / 600 = 120 frames; LLM hidden = 2560
```

### Stage1 dry-run

```bash
llamafactory-cli train configs/qwen3_5ae-asr/stage1_wavtok_40_unify.yaml \
    --max_steps 5 --logging_steps 1
```

**확인 사항**:
- 첫 step loss 유한 (NaN/Inf 없음)
- `audio_pad_token` 개수와 `num_samples // 600` 일치

---

## 11. 향후 Ablation 로드맵

### 1순위: Small speech-only variant (같은 frame rate, 다른 데이터)

```
WavTokenizer-small-600-24k-4096
- Frame rate: 40 fps (동일, hop=600)
- Data: LibriTTS only (speech-only, codec 관점)
- Purpose: "Objective" (reconstruction vs ASR) vs "Data domain" confound 제거
- Expected: WavTok-40-unify 대비 음성 타스크에서 더 specialized
```

**Stage1 실행**: 같은 yaml, model_name_or_path 만 변경.

### 2순위: Large speech-75token variant (다른 frame rate, 같은 데이터) — **구현 완료**

```
WavTokenizer-large-speech-75token (novateur/WavTokenizer-large-speech-75token)
- Frame rate: 75 fps (hop=320 @ 24kHz)
- Ratios: [8, 5, 4, 2] (WavTok-40의 [6,5,5,4]와 다름)
- 30s → 2250 frames (vs unify 의 1200)
- Purpose: Frame rate / LLM sequence length 영향 분석
- Token budget 유지: omni_max_audio_samples=576000 (24s) → 75×24=1800 tokens
```

**구현 내역**:
- `Qwen3.5AE-4B-wavtok-75-speech/` 모델 디렉터리 생성
- `convert_to_wavtok75.py` → safetensors 생성 완료 (encoder 8.80M, projector 18.35M)
- `configs/ASR/stage1_wavtok_75_speech.yaml` 생성
- `scripts/ASR/run_stage1_wavtok75.sh` 생성
- **학습 미시작** (WavTok-40 먼저 진행 중)

### 3순위: Post-VQ z_q codes (다른 tap point)

```
- Tap: SEANetEncoder → VQ bottleneck → z_q codes [B, 8, T] (8 codebooks)
- Embedding layer 추가 필요
- Purpose: Information bottleneck 효과 연구
```

---

## 12. 오픈 이슈

### OOM 위험

WavTok (8.30M) 은 DAC(27.55M) 보다 가볍고 Whisper(88.15M) 보다 훨씬 가벼워 OOM 위험은 **낮음**.

혹시 발생 시:
1. `per_device_train_batch_size: 3 → 2`
2. `omni_max_audio_samples: 1080000 → 720000` (30s 절감)

### Resampling 오버헤드

MLS + VoxPopuli (16kHz) 를 24kHz 로 resample 해야 함.
- DataLoader 병목 우려 가능 (실시간 resample)
- 해결: `torchaudio.functional.resample()` 는 빠르지만, 만약 커진다면:
  - Manifest 전처리 단계에서 미리 24kHz 로 저장
  - 또는 WavTok-16k variant 찾기 (현재 배포 없음)

---

## 13. 작업 체크리스트

### 코드 구현 — WavTok-40-unify

- [x] `wavtokenizer_modules/` 다운로드 + `__init__.py` 수정 (transformer import 제거)
- [x] `configuration_qwen3_5AE.py` (WavTok fields, fps=40, ratios=[6,5,5,4])
- [x] `audio_encoder.py` (SEANetEncoder integration)
- [x] `convert_to_wavtok.py` (weight_norm 병합 포함)
- [x] `sanity_check.py` (encoder weight 검증)
- [x] `configs/ASR/stage1_wavtok_40_unify.yaml`
- [x] `scripts/ASR/run_stage1_wavtok.sh`

### 코드 구현 — WavTok-75-speech

- [x] `Qwen3.5AE-4B-wavtok-75-speech/configuration_qwen3_5AE.py` (fps=75, ratios=[8,5,4,2], hop=320)
- [x] `Qwen3.5AE-4B-wavtok-75-speech/convert_to_wavtok75.py`
- [x] safetensors 생성 완료 (encoder 8.80M, projector 18.35M, LM 4.21B)
- [x] `configs/ASR/stage1_wavtok_75_speech.yaml`
- [x] `scripts/ASR/run_stage1_wavtok75.sh`

### Verification — WavTok-40-unify

- [x] Convert → safetensors 생성 완료
- [x] Forward shape 확인 (72000 // 600 = 120 frames → projector (1, 120, 2560))
- [x] Stage1 학습 진행 중 (~step 400)
- [ ] Loss curve 수렴 확인 (100k steps)

### Verification — WavTok-75-speech

- [x] `convert_to_wavtok75.py` 실행 성공 (strict=True, 62 encoder keys)
- [ ] Forward shape assertion (S // 320 = T frames)
- [ ] Stage1 dry-run (5 steps)
- [ ] 본 학습 시작

### 본 학습 (100k steps)

- [ ] WavTok-40: WandB 모니터링, checkpoint 1000 step 마다, loss 수렴 확인
- [ ] WavTok-75: 학습 시작 후 동일 모니터링

---

## 14. Reference

- **WavTokenizer repo**: https://github.com/jishengpeng/WavTokenizer
- **Paper**: "WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Models"
- **Checkpoint (40fps)**: `/mnt/tmp/hf_cache/wavtokenizer/wavtokenizer_large_unify_600_24k.ckpt` (1.7 GB, PyTorch Lightning)
- **Checkpoint (75fps)**: `/mnt/tmp/hf_cache/wavtokenizer/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_v2.ckpt` (HF: `novateur/WavTokenizer-large-speech-75token`)
- **Model dir (40fps)**: `external/models/Qwen3.5AE-4B-wavtok-40-unify/`
- **Model dir (75fps)**: `external/models/Qwen3.5AE-4B-wavtok-75-speech/`
- **DAC reference**: Existing `Qwen3.5AE-4B` variant (DAC encoder)
- **Whisper reference**: Existing `Qwen3.5AE-4B-whisper-tiny/small` variants
