# EnCodec-24k Stage1: ASR-Domain Audio Encoder Integration

## 목표 / 아키텍처 비교

Qwen3.5AE-4B는 현재 3개의 오디오 인코더 변형을 지원합니다:
1. **DAC-VAE** (48kHz, 48fps) — 음성 용 DAC
2. **WavTokenizer-40** (24kHz, 40fps) — 음성 + 음악 사전학습
3. **Whisper-tiny/small** (16kHz, 50fps Mel) — 다중언어 음성 인식

EnCodec-24k 추가로 4번째 변형 계획:
4. **EnCodec-24k** (24kHz, 75fps) — Meta/HF 표준 SEANet+RVQ 오디오 코덱

| 항목 | DAC-VAE | WavTok-40 | Whisper-tiny | **EnCodec-24k** |
|---|---|---|---|---|
| **샘플링 레이트** | 48kHz | 24kHz | 16kHz | **24kHz** |
| **fps (75s @ SR)** | 48fps | 40fps | 50fps | **75fps** |
| **인코더 출력 dim** | 512 | 512 | 384 | **128** |
| **프레임 수 @ 45s** | 1,125 | 1,800 | ~1,500 | **3,375** |
| **hop_length** | 1000 | 600 | ~320 | **320** |
| **인코더 가중치** | 18M | 70.8M | - | **7.43M** |
| **훈련 도메인** | 음성 | 음성+음악 | 다중언어 음성 | **음성 60%, 음악 40%** |
| **구현 패턴** | 외부 import | 인라인 | 외부 API | **표준 HF import** |

**배치 설정**:
```
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
effective_batch = 2 × 2 × 8GPU = 32
```
(WavTok-40는 `batch=4, grad_accum=1, effective=32` — EnCodec는 메모리 상황 동일 유지)

---

## 체크포인트 선택 — `facebook/encodec_24khz` (24k vs 48k 결정)

EnCodec은 2가지 표준 프리트레인 버전 제공:
- `facebook/encodec_24khz` (24 kHz mono)
- `facebook/encodec_48khz` (48 kHz stereo)

### 비교표

| 항목 | **24kHz (선택)** | 48kHz (제외) |
|---|---|---|
| **FPS (45s @ SR)** | 75fps | 150fps |
| **45s 프레임 수** | 3,375 | 6,750 |
| **채널** | Mono (✓) | Stereo (❌ 불일치) |
| **훈련 도메인** | 음성 >60% (DNS, CommonVoice, AudioSet, FSD50K, Jamendo) | **음악 전용 (Jamendo)** |
| **인코더 파라미터** | 7.43M | 7.43M |
| **Hidden dim** | 128 | 128 |

### 결론: 24kHz 단일 선택

**48kHz 제외 이유**:
1. **FPS 초과**: 45s @ 48kHz = 6,750 frames. `cutoff_len=3584`에 비해 1.9배 → packing 무효, 긴 발화 심하게 자름.
2. **ASR 도메인 미스매치**: 48kHz는 음악 중심 학습 (Jamendo). ASR은 음성 데이터셋에서 성능 저하.
3. **Stereo 처리 부담**: 학습 입력 형식이 모노(mono)인데, stereo 인코더의 stereo→mono 변환은 정보손실 또는 추가 처리.

---

## 공정 비교 축 — Audio Time(45s) vs LLM Frames

인코더 비교 시 **공정한 기준축**을 정의하는 것이 핵심입니다.

### Axis 1: Audio Time 매칭 (45초)

**원칙**: 같은 wall-clock duration (45s)의 오디오를 각 인코더에 입력.  
**결과**: LLM seq length는 각 인코더의 native fps에 따라 달라짐.

```
Input:  45s waveform @ SR (같음)
        ↓
Encoder (fps마다 다름)
        ↓
Output: [B, T, hidden_size]  where T varies (45s × fps)
```

| variant | SR | fps | 45s frames |
|---|---|---|---|
| DAC-VAE | 48k | 48 | 1,125 |
| WavTok-40 | 24k | 40 | 1,800 |
| Whisper | 16k | 50 | ~1,500 |
| **EnCodec-24k** | **24k** | **75** | **3,375** |

**이유**: 인코더의 본질적 직무는 "고정된 오디오 → LLM-usable representation"입니다.  
**Axis 1에서 비교하면**: 같은 정보량(45s 오디오)을 어떻게 LLM-친화적으로 변환하는가를 평가합니다.

### Axis 2: LLM Frame 매칭 (토큰 예산)

**원칙**: 같은 LLM frame budget (e.g., 3584 cutoff_len 사용)에서 각 인코더가 처리할 수 있는 오디오 길이.

```
cutoff_len = 3584 (고정)
            ↓
Encoder fps 따라 다르게 변환
            ↓
Input audio duration 다름 (fps 높을수록 짧음)
```

**예시**: cutoff 3584일 때 처리 가능 오디오
- DAC-48fps: 3584 / 48 ≈ 74.7초
- WavTok-40fps: 3584 / 40 = 89.6초
- **EnCodec-75fps: 3584 / 75 ≈ 47.8초**

### 왜 Axis 1 선택?

1. **인코더 평가의 본질**: 인코더는 "오디오 품질 → 표현력"을 설계합니다. Axis 2는 사실상 "인코더 + 강제 다운샘플(또는 프레임 드롭)"을 평가하는 것입니다. 인코더 자체 성능 평가가 아님.

2. **fps는 설계 선택**: 75fps vs 40fps는 "trade-off (시간 해상도 vs 메모리)"입니다. 이를 패널티로 매기는 것은 부당합니다.

3. **배포 현실**: 실제 사용 시나리오에서:
   - 사용자: "30초 음성을 인식해줘"
   - 인코더: 자신의 fps로 프레임 생성 (고정)
   - LLM: 그 frame seq를 처리
   - **결과**: fps가 높으면 seq 길고, 낮으면 짧음. 이는 인코더의 cost-quality tradeoff이지, 페널티 아님.

4. **Axis 2의 의미**: "동일 LLM 비용(토큰)으로 얼마나 많은 정보를 압축할 수 있는가" = **압축 효율** 질문. ASR 성능 평가와는 다른 차원.

**결론**: **Axis 1 (45s audio time 매칭)이 ASR 인코더 품질 비교의 기본축입니다.**

---

## `omni_max_audio_samples` & `cutoff_len` 결정

### 현재 4개 variant의 컨벤션

| variant | SR | omni_max_audio_samples | 최대 시간 | 최대 프레임 | cutoff_len |
|---|---|---|---|---|---|
| DAC-VAE | 48k | 2,160,000 | 45s | 1,125 | 3,584 |
| WavTok-40 | 24k | 1,080,000 | 45s | 1,800 | 3,584 |
| Whisper | 16k | 480,000 | 30s | ~1,500 | 3,584 |
| **EnCodec-24k** | **24k** | **1,080,000** | **45s** | **3,375** | **6,144** |

### EnCodec-24k 특수성

**문제**: WavTok-40과 동일하게 `omni_max_audio_samples=1,080,000` (45s @ 24kHz)로 설정하면,  
EnCodec-24k는 75fps이므로 **3,375 frames 발생**.

`cutoff_len=3584`일 때:
```
text_budget = 3584 - 3375 = 209 tokens
```

이는:
1. **Packing 무효화**: 발화마다 3375/3584 ≈ 94% 점유 → 패킹 밀도 ~1.0 (낭비)
2. **긴 발화 절단**: 텍스트 예산 209 토큰은 매우 적음. 일반 발화 (100+ 토큰)도 절단됨.

### 해결책: `cutoff_len=6144` (EnCodec-24k 전용)

```
audio_frame_budget = 3375 (45s @ 75fps)
text_token_budget = 6144 - 3375 = 2,769
```

**이점**:
- **Packing 회복**: 여러 발화를 하나의 예제로 묶을 수 있음. 예상 packing density ≈ 1.7 samples/pack (WavTok-40의 ~1.9에 비해 약간 낮지만 합리적).
- **Text 여유**: 2,769 토큰 예산으로 대부분 발화 완전 포함.
- **모델 동작 영향 없음**: cutoff_len은 packing constraint일 뿐, LLM의 position embedding 한도 (8192) 내에 충분함.

**메모리 보정**:
- WavTok-40: `per_device_batch=4, grad_accum=1` → effective batch 32
- EnCodec-24k: `per_device_batch=2, grad_accum=2` → effective batch 32
  
  cutoff_len 1.7배 증가(3584→6144)를 `batch 2배 감소 × grad_accum 2배 증가`로 보정.  
  Activation memory ≈ 동일 유지.

---

## Training Fairness 고려: Step당 본 Audio Sample 수

실제 훈련에서는 step 수와 batch 설정 차이로 인해 인코더마다 본 데이터 양이 다릅니다.

| variant | packing_density (이론) | step당 samples (8GPU) | 예상 |
|---|---|---|---|
| DAC | ~2.9 | 93.8 | 높음 |
| WavTok-40 | ~1.9 | 60.5 | 중간 |
| Whisper | ~2.2 | 71.7 | 중간 |
| **EnCodec-24k** | **~1.7** | **54.4** | **낮음** |

**현재 설정**: 모든 variant `max_steps=100,000` (동일).  
**결과**: EnCodec은 step당 본 samples이 적어서, 전체 epoch 수 실질적으로 다름.

**향후 개선** (현재는 불가):
- max_steps 조정: EnCodec-24k는 ~120K로 늘려서 본 data 통일.
- Evaluation 리포트: 각 variant의 "total audio samples seen" 함께 기록.

---

## Encoder 코드 통합 방식

### 3가지 패턴

1. **DAC 패턴** (외부 import):
   ```python
   from dacvae import DACVAE
   encoder = DACVAE.from_pretrained(...)
   ```
   - 외부 패키지 필요.
   - config에 `dacvae_ckpt_path` 같은 runtime 경로 필드.

2. **WavTok 패턴** (인라인):
   ```python
   # audio_encoder.py 내에 SEANetEncoder 전체 포함
   # 외부 import 없음. config에만 `wavtok_*` 하이퍼파라미터.
   ```
   - 외부 패키지 불필요.
   - 모든 코드가 repo에 자체 포함.
   - state_dict에 `weight_norm` (_g, _v) 쌍 → merge 필요.

3. **EnCodec 패턴** (표준 HF import) ← **우리 선택**:
   ```python
   from transformers import EncodecModel, EncodecConfig
   encodec_cfg = EncodecConfig(...)
   full = EncodecModel(encodec_cfg)
   encoder = full.encoder
   ```
   - `transformers`는 표준 라이브러리 (이미 의존성).
   - 인라인 불필요.
   - **하지만 중요**: HF EncodecEncoder는 **modern parametrization API** 사용.
     - WavTok: `torch.nn.utils.weight_norm` (deprecated) → state_dict에 `_g`, `_v` keys.
     - EnCodec: `torch.nn.utils.parametrize.register_parametrization` (modern) → state_dict에 `parametrizations.weight.original0/1` keys.
   - **Strip 방식 차이**:
     ```python
     # WavTok (deprecated)
     nn.utils.remove_weight_norm(module)  # ❌ EnCodec에는 작동 안 함
     
     # EnCodec (modern, 정답)
     from torch.nn.utils.parametrize import remove_parametrizations, is_parametrized
     for module in encoder.modules():
         if is_parametrized(module, "weight"):
             remove_parametrizations(module, "weight", leave_parametrized=True)
     ```

---

## Convert Flow

`convert_to_encodec.py` 워크플로우:

**단계 1-4**: WavTok와 동일 (LLM 가중치 로드, DAC 키 드롭, 상태 로드).

**단계 5** (EnCodec 고유):
```python
# Lightning 체크포인트 로드 → 키 prefix strip → weight_norm merge (WavTok)
# 대신:

# HF에서 직접 로드
hf_encodec = EncodecModel.from_pretrained("facebook/encodec_24khz")

# 파라미터화 제거 (modern API)
for module in hf_encodec.encoder.modules():
    if is_parametrized(module, "weight"):
        remove_parametrizations(module, "weight", leave_parametrized=True)

# 상태 로드 (매우 간단)
encoder_state = hf_encodec.encoder.state_dict()
ae_model.model.audio_encoder.encoder.load_state_dict(encoder_state, strict=True)
```

**단계 6**: `save_pretrained`, 보조 파일 복사, config 패치 (WavTok과 동일).

---

## Parameter Count 검증

`docs/parameter_count.md` 예상 행:

| Variant | Encoder | Projector | Total Active | Frozen |
|---|---|---|---|---|
| DAC-VAE | 18.0M | 18.2M | 36.2M | 3.0B (LLM) |
| WavTok-40 | 70.8M | 18.2M | 89.0M | 3.0B |
| Whisper | - | 18.2M | 18.2M | 3.0B + Whisper encoder |
| **EnCodec-24k** | **7.43M** | **18.1M** | **25.6M** | **3.0B** |

실제 convert 실행 로그에서 출력되는 파라미터 수와 위 표 일치 확인.

---

## 단계별 검증 (Stage1 훈련 전)

convert script 실행 후, Stage1 훈련 시작 전:

### 1. Encoder 출력 형태 확인

```bash
conda activate audiollm
python -c "
from transformers import AutoModelForCausalLM
import torch

m = AutoModelForCausalLM.from_pretrained(
    'external/models/Qwen3.5AE-4B-encodec-24k',
    trust_remote_code=True, torch_dtype=torch.bfloat16
).cuda()
m.eval()

# 30s @ 24kHz = 720,000 samples
wav = torch.randn(2, 1, 720000, dtype=torch.bfloat16).cuda()
with torch.no_grad():
    embeds = m.model.audio_encoder(wav)
print(f'Input shape: {wav.shape}')
print(f'Output shape: {embeds.shape}')
print(f'Expected: [2, 2250, 2560]')  # 2250 = 30s × 75fps
"
```

**예상 출력**:
```
Input shape: torch.Size([2, 1, 720000])
Output shape: torch.Size([2, 2250, 2560])
Expected: [2, 2250, 2560]
```

### 2. Encoder 가중치 일치 확인

`facebook/encodec_24khz` 로드 후 가중치가 모델에 올바르게 로드됐는지 확인:
```bash
python -c "
from transformers import AutoModelForCausalLM, EncodecModel
import torch

# EnCodec-24k 모델
enc_model = AutoModelForCausalLM.from_pretrained(
    'external/models/Qwen3.5AE-4B-encodec-24k',
    trust_remote_code=True, torch_dtype=torch.bfloat16
).cuda()

# 원본 HF EnCodec-24k
hf_enc = EncodecModel.from_pretrained('facebook/encodec_24khz')

# 비교 샘플
enc_keys = set(enc_model.model.audio_encoder.encoder.state_dict().keys())
hf_keys = set(hf_enc.encoder.state_dict().keys())

print(f'Keys in converted model: {len(enc_keys)}')
print(f'Keys in original HF: {len(hf_keys)}')
print(f'Same keys: {enc_keys == hf_keys}')
"
```

### 3. Single-step Dry-run Training

메모리 및 그래디언트 흐름 확인:

```bash
# 한 GPU에서 2 step만 실행
python -m torch.distributed.launch --nproc_per_node=1 \
  src/llamafactory/cli.py train \
  configs/ASR/stage1_encodec_24k.yaml \
  --override max_steps=2 \
  --override per_device_train_batch_size=2
```

**확인 사항**:
- ✅ OOM 없음 (batch=2 × seq=6144 메모리 적절)
- ✅ Encoder 가중치 no grad (frozen)
- ✅ Projector 그래디언트 흐름 (trainable)
- ✅ Loss finite (nan/inf 없음)

### 4. Full Launch (검증 완료 후)

```bash
bash scripts/ASR/run_stage1_encodec.sh
```

---

## 추가 메모

- **Packing 이론**: cutoff_len 증액으로 여러 발화 조합이 가능하지만, 실제 packing 밀도는 데이터셋의 발화 길이 분포에 의존합니다.
- **Step당 sample 수 불일치**: 현재 100K step으로 모두 동일하지만, 향후 재훈련 시 EnCodec 전용으로 더 많은 step을 고려할 수 있습니다.
- **ASR Evaluation**: Stage1 훈련 완료 후 Stage2 또는 평가 시 모든 variant에 대해 동일 조건의 WER 벤치마크를 실행합니다.
