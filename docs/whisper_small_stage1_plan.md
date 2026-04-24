# Whisper-small.en + Qwen3.5-4B — Stage1 학습 세팅

> 목표: 현재 `Qwen3.5AE-4B` 아키텍처에서 audio encoder 만 **DACVAE → `openai/whisper-small.en`** 으로 교체, Stage1 (projector-only) 학습.
> LLM, AudioProjector 구조는 그대로 유지. Encoder output dim 만 128 → 768 로 바뀜.
> ※ Whisper-small 은 DACVAE 보다 encoder 자체가 **큼** (88M vs 27M) — "가볍게 가자" 는 방향이 아니라, acoustic-semantic 양쪽 표현력 있는 ASR-pretrained encoder 를 갖고 오는 목적.

---

## 1. 아키텍처 비교

### 현재 (DACVAE)

```
[48kHz wav] → DACVAE.encode()
            → (B, T, 128)  @ 25 fps  (hop=1920 samples)
            → AudioProjector (input_proj: 128→512, 4x LlamaDecoderLayer, output_proj: 512→2560)
            → (B, T, 2560)  → Qwen3.5-4B
```

### 변경 후 (Whisper-small.en encoder)

```
[16kHz wav] → WhisperFeatureExtractor (CPU, dataloader) → log-mel (B, 80, 3000)
            → WhisperEncoder  (12 layers, d_model=768, 2x Conv1d stride=2)
            → last_hidden_state (B, 1500, 768)  @ 50 fps  (30s 고정 padding)
            → AudioProjector (input_proj: 768→512, 4x LlamaDecoderLayer, output_proj: 512→2560)
            → (B, T_audio, 2560)  → Qwen3.5-4B
```

※ `WhisperFeatureExtractor` 는 학습 파라미터 0 개의 DSP 전처리 (mel filterbank + log). 모델의 layer 아님.
※ Tap point (Whisper 의 어느 layer 를 projector 입력으로 쓰는가) 는 §6 참고.

---

## 2. 파라미터 수 비교

### 모델 기준 (ASR feature extractor 용도)

| 모델                 | 원 모델 전체           | Encoder only (실제 사용분) | 비고                                                                                           |
| -------------------- | ---------------------- | -------------------------- | ---------------------------------------------------------------------------------------------- |
| Whisper-small.en     | 241,734,144 (~241.73M) | **88,154,112 (~88.15M)**   | seq2seq — decoder 153.58M 은 ASR feature 에 불필요 (token embed 39.83M 포함)                  |
| DACVAE (watermarked) | 107,671,171 (~107.67M) | **27,551,360 (~27.55M)**   | encode-path = encoder + quantizer.in_proj. 나머지 80.1M (decoder + watermarker) 은 dead weight |

→ Encoder only 기준 **DACVAE 27.55M → Whisper-small 88.15M (약 3.2× 증가)**.

### Qwen3.5AE AudioEncoder 통합 기준

| 구성                           | 현재 (DACVAE)         | 변경 (Whisper-small.en)                                   |
| ------------------------------ | --------------------- | --------------------------------------------------------- |
| Audio encoder loaded           | 107,648,066 (~107.6M) | **88,154,112 (~88.15M)**                                  |
| Audio encoder active (forward) | 27,551,360 (~27.6M)   | **88,154,112 (~88.15M)**                                  |
| Dead weight (decoder etc.)     | 80,096,706 (~80.1M)   | 0                                                         |
| AudioProjector                 | 18,158,080            | **~18,485,760** (input_proj 65,536→393,216, 나머지 동일) |
| **Stage1 학습 파라미터**       | 18,158,080            | **~18,485,760** (+1.80%)                                 |

※ AudioProjector 는 `audio_hidden_size` 만 128 → 768 로 바뀌므로 `input_proj` 파라미터만 65,536 → 393,216 (+328K). 나머지 4-layer LlamaDecoder + output_proj 는 전부 동일.
※ Whisper-small.en encoder 88.15M: 12 layers × d_model=768 × 12 heads, ffn=3072.
※ HF 표기 "whisper-small 244M" 은 encoder+decoder 합계 (실측 241.73M).

---

## 3. Dataset (Stage1)

**이 repo 내 manifest 사용** — DACVAE Stage1 과 동일한 LibriTTS-R + MLS + VoxPopuli 조합.

| Code name            | Train durations (hrs) | 샘플레이트 (원본)                                                   |
| -------------------- | --------------------- | ------------------------------------------------------------------- |
| en_LibriTTS_R_single | 552.42                | **24k** → dataloader 에서 16k resample 필요                        |
| en_MLS_single        | 45,000.00             | 16k                                                                 |
| en_VoxPopuli_single  | 522.00                | 16k                                                                 |
| **TOTAL**            | **~46,074** hrs       |                                                                     |

**Manifest 경로**:

```
/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/libri_mls_vox
```

shard 포맷 `shard_NNNNN.jsonl`. DACVAE Stage1 학습에서도 이미 사용 중.

---

## 4. Whisper-small.en 기본 스펙

| 항목                           | 값                                                                |
| ------------------------------ | ----------------------------------------------------------------- |
| HF repo                        | `openai/whisper-small.en`                                         |
| 입력 sample rate               | **16,000** Hz                                                     |
| FeatureExtractor hop           | 10 ms (mel 100 fps)                                               |
| Encoder Conv1d stride          | 2 × 2 → **50 fps** output                                        |
| 고정 입력 길이                 | 30 초 (3000 mel frames → 1500 encoder frames)                    |
| 짧은 입력 처리                 | zero-pad to 30s (HF default)                                      |
| 긴 입력 처리                   | **manifest 전처리 단계에서 30s chunk 로 split** (§13.1)          |
| d_model                        | 768                                                               |
| encoder layers                 | 12                                                                |
| attention heads                | 12                                                                |
| feed-forward dim               | 3072                                                              |
| encoder params                 | **88,154,112 (~88.15M)** (`WhisperModel.encoder.parameters()` 합) |
| decoder params (미사용)        | 153,580,032 (~153.58M)                                            |

### `small.en` vs multilingual `small` — **`.en` 선택**

이번 Stage1 manifest 는 영어 전용 (LibriTTS-R + MLS-en + VoxPopuli-en). `whisper-small.en` 이 영어 전용 학습으로 WER 더 낮음 (LibriSpeech test-clean 3.1 vs multilingual small 3.4).

---

## 5. 데이터 파이프라인 변경점

현재 `src/llamafactory/data/omni_dataset.py` — raw waveform 을 `audio_features = [N, 1, samples]` 로 collate 후 모델 forward 시 `AudioEncoder.encode()` 내부에서 waveform → latent.

### Option A (확정) — DataLoader 에서 mel 추출

`audio_features` 계약을 **raw waveform → log-mel spectrogram 으로 변경**.

- `src/llamafactory/data/omni_dataset_whisper.py` 신규:
  ```python
  from transformers import WhisperFeatureExtractor
  fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")

  sample_rate = 16000
  # 24k (LibriTTS-R) → 16k resample 필요
  if orig_sr != 16000:
      wav = torchaudio.functional.resample(wav, orig_sr, 16000)

  # 30s 초과 처리: manifest 단계에서 이미 ≤30s 로 split 되어 있음 (§13.1)
  # Safety check 만 — manifest split 못 한 경우 대비
  if len(wav) > 30 * 16000:
      wav = wav[: 30 * 16000]  # 안전장치 (정상 manifest 면 도달 안 함)

  # Manifest 에 audio_start / audio_end 있으면 해당 구간만 load
  # if "audio_start" in entry:
  #     wav, _ = torchaudio.load(path, frame_offset=int(entry["audio_start"]*sr),
  #                               num_frames=int((entry["audio_end"]-entry["audio_start"])*sr))

  # Whisper mel: (1, 80, 3000) fp32
  mel = fe(wav, sampling_rate=16000, return_tensors="pt").input_features[0]

  # audio placeholder token 수 (실제 길이 기준, padding 구간 제외)
  t_audio = min(1500, math.ceil(len(wav) / 320))  # 50 fps
  ```

- `src/llamafactory/data/mm_plugin_whisper.py` 신규:
  - `audio_features` 가 `[N, 80, 3000]` 텐서로 collate 됨. 모든 샘플이 (80, 3000) 고정이라 padding 불필요, 단순 stack.

- 모델 forward 시 `audio_features` 가 이미 mel → `AudioEncoder` 가 바로 Whisper encoder 에 투입.

**dtype 경계**: `WhisperFeatureExtractor` 는 numpy/fp32 출력. Collator 단계에서 fp32 tensor 로 전달, 모델 forward 진입시 autocast bf16 (§9 참고).

### Option B (기각) — Encoder 내부 mel 추출

기존 `audio_features = [N, 1, samples]` 인터페이스 유지하고 `AudioEncoder.forward` 내부에서 mel 변환. Batch 내 길이 다른 waveform padding 을 모델 레벨에서 처리해야 하고, FeatureExtractor 가 numpy 기반이라 CPU-GPU 왕복 발생. 선택 안 함.

---

## 6. Tap point — Whisper 어느 layer 를 쓰는가

### Baseline (Stage1 초기 실행): `encoder.last_hidden_state`

```python
hidden = self.encoder(input_features).last_hidden_state
# [B, 1500, 768] — 12-layer transformer + final LayerNorm 통과 후
```

**선택 이유**:
- Whisper 의 ASR decoder 가 attend 하는 지점 → 이미 "LM 이 읽기 좋은 형태" 로 정렬됨
- 추가 학습 파라미터 0 (final LN 까지 포함된 표준 output)
- HF `WhisperModel.encoder(...)` 기본 반환값, 코드 간결
- Stage1 이 projector-only 제약이라 **단일 layer tap 이 합리적**

### 후속 탐색 로드맵 (주 경로 아님)

Baseline Stage1 수렴 후 개선 여지 보이면 시도:

- **중간 layer tap (e.g. layer 6~8)**
  - `encoder(..., output_hidden_states=True).hidden_states[k]` 사용
  - SUPERB / Layer-wise probing 연구: ASR 관련 phonetic 정보는 **중간 layer (40~70% depth) 에서 peak** 보고
  - 최종 layer 는 이미 decoder 용 semantic bias 가 있어, 오히려 덜 가공된 중간 representation 이 projector 에 유리할 수 있음
  - 각 layer 별로 Stage1 재학습 → ASR eval 비교

- **초반 layer tap (conv stem 직후 또는 layer 2~3)**
  - 더 원음에 가까운 표현. ASR 이전 phonetic/paralinguistic 정보 풍부
  - 이번 Stage1 목표 (ASR) 엔 부적합하나, 감정/화자 태스크 대상일 때 의미

- **Weighted sum of layers (learnable)**
  - `α_0 * h_0 + ... + α_12 * h_12`, `α` 는 softmax-normalized scalar 12~13 개
  - SUPERB 표준 방식. 추가 파라미터 최소 (<100)
  - "Projector 구조 고정" 제약을 사실상 위반하지 않음 (projector 앞에 weighted-sum module 하나 추가)
  - Stage1 레시피 거의 그대로 쓸 수 있어 도입 비용 낮음

판단 기준: baseline Stage1 수렴 loss / ASR eval WER 보고 결정.

---

## 7. `audio_encoder_whisper.py` 설계

```python
# external/models/Qwen3.5AE-4B-whisper-small/audio_encoder.py
import torch
import torch.nn as nn
from transformers import WhisperModel
from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding,
)


class AudioProjector(nn.Module):
    """기존 DACVAE 버전과 동일. input_proj 의 in_features 만 config 에서 결정됨."""
    def __init__(self, config):
        super().__init__()
        # ... (기존 audio_encoder.py 의 AudioProjector 코드 그대로)
        # input_proj: Linear(config.audio_hidden_size=768 → config.adapter_hidden_size=512)


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Whisper encoder 를 HF 에서 pretrained 로 로드
        #    convert script 실행 시점에 HF cache 에서 한 번 download → safetensors 에 박힘
        whisper = WhisperModel.from_pretrained(
            config.whisper_model_id,         # "openai/whisper-small.en"
            torch_dtype=torch.bfloat16,
        )
        self.encoder = whisper.encoder
        del whisper  # decoder 버림

        # 2. Freeze (Stage1)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.audio_dim = config.audio_hidden_size  # 768

        # 3. Projector (기존과 동일 구조)
        self.projector = AudioProjector(config)

    def forward(self, audio_features, use_cache=False):
        """
        audio_features: [N, 80, 3000] log-mel (dataloader 에서 사전 추출)
        """
        with torch.no_grad():  # Encoder frozen
            hidden = self.encoder(audio_features).last_hidden_state  # [N, 1500, 768]

        # Projector 통과 (DAC 버전과 동일 로직, transpose 불필요 — 이미 [N, T, C])
        audio_embeds, _ = self.projector(hidden, use_cache=use_cache)
        return audio_embeds
```

### 주요 차이 (기존 DAC 버전 대비)

| 항목 | DAC 버전 | Whisper 버전 |
|---|---|---|
| Encoder 로드 | `DACVAE(**dac_kwargs)` (random init, weight 는 safetensors 에서 주입) | `WhisperModel.from_pretrained(HF id)` (init 시점에 pretrained 로드) |
| `remove_weight_norm` 루프 | 필요 | 불필요 (Whisper 는 weight_norm 미사용) |
| Input dim transpose | `[B, C, T] → [B, T, C]` | 불필요 (Whisper 출력이 이미 `[B, T, C]`) |
| Noise augmentation 블록 | 있음 (현재 꺼져있음) | 제거 |
| `audio_features` 입력 shape | `[N, 1, samples]` | `[N, 80, 3000]` |

---

## 8. Checkpoint 빌드 — `convert_qwen3_5_to_qwen3_5AE_whisper.py`

기존 `convert_qwen3_5_to_qwen3_5AE.py` 복제 후 수정. 흐름:

1. Base `Qwen/Qwen3.5-4B` 로드 (또는 로컬 경로)
2. `Qwen3_5AETextConfig` 는 그대로 base 에서 복제
3. `AudioConfig` 를 Whisper 용으로 변경 (`dac_*` 제거, `whisper_*` 추가, `audio_hidden_size=768`)
4. `Qwen3_5AEForConditionalGeneration(ae_config)` instantiate
   - 이때 `AudioEncoder.__init__` 실행 → `WhisperModel.from_pretrained("openai/whisper-small.en")` 호출 → pretrained Whisper encoder weight 가 모델 안에 로드됨
5. Base LLM state_dict 로드 → `model.X` → `model.language_model.X` remap → `load_state_dict(strict=False)`
6. `save_pretrained(output_dir)` → 전체 state_dict 저장:
   - `model.language_model.*` — base Qwen3.5 에서 가져온 값
   - `model.audio_encoder.encoder.*` — HF 에서 가져온 Whisper pretrained 값 ✅
   - `model.audio_encoder.projector.*` — random init (Stage1 에서 학습됨)
   - `lm_head.*` — tied 또는 base 값
7. 코드 파일 (`audio_encoder.py` whisper 버전, `modeling_qwen3_5AE.py`, `configuration_qwen3_5AE.py`, tokenizer) 복사
8. `config.json` 의 `auto_map` + `audio_config` + `dtype` 채움

### 실행

```bash
python external/models/Qwen3.5AE-4B-whisper-small/convert_qwen3_5_to_qwen3_5AE_whisper.py \
    --base-id Qwen/Qwen3.5-4B \
    --output-dir external/models/Qwen3.5AE-4B-whisper-small \
    --dtype bfloat16
```

한 번 실행 후 `external/models/Qwen3.5AE-4B-whisper-small/` 에 모든 파일 완성. 이후 학습/추론은 이 dir 를 `model_name_or_path` 로 사용. HF 네트워크 접근 불필요.

### 결과 dir 구조

```
external/models/Qwen3.5AE-4B-whisper-small/
├── config.json                       (신규, whisper audio_config)
├── audio_encoder.py                  (신규, whisper 버전)
├── configuration_qwen3_5AE.py        (소폭 수정: AudioConfig 에 whisper 필드)
├── modeling_qwen3_5AE.py             (기존 복사)
├── tokenization_qwen3_5AE.py         (기존 복사)
├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, added_tokens.json, chat_template.jinja
├── generation_config.json
├── model-00001-of-00002.safetensors  ← LLM + Whisper pretrained + random projector
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
└── convert_qwen3_5_to_qwen3_5AE_whisper.py   (재현성 위해 복사)
```

디스크 ~8GB (LLM bf16 8GB + Whisper 0.18GB + projector 37MB).

### DAC 관련 키 처리

Whisper 버전 `AudioEncoder` 는 DAC 모듈이 아예 없음. state_dict 에 `dac_encoder.*` 류 키 없음 → `_keys_to_ignore_on_load_unexpected` 같은 hack 불필요.

---

## 9. `config.json` audio_config 변경

```diff
  "audio_config": {
    "adapter_hidden_size": 512,
-   "audio_hidden_size": 128,
+   "audio_hidden_size": 768,
-   "dac_codebook_dim": 128,
-   "dac_codebook_size": 1024,
-   "dac_decoder_dim": 1536,
-   "dac_decoder_rates": [12, 10, 8, 2],
-   "dac_encoder_dim": 64,
-   "dac_encoder_rates": [2, 8, 10, 12],
-   "dac_latent_dim": 1024,
-   "dac_n_codebooks": 16,
-   "dac_sample_rate": 48000,
+   "whisper_model_id": "openai/whisper-small.en",
+   "whisper_sample_rate": 16000,
+   "whisper_fps": 50,
    "head_dim": 64,
    "intermediate_size": 2048,
    "num_adapter_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    ...
  }
```

`configuration_qwen3_5AE.py` 의 `AudioConfig` 클래스에서 `dac_*` 필드 제거 + `whisper_*` 필드 추가.

---

## 10. Stage1 training config (yaml)

> **원칙**: 현재 DACVAE Stage1 yaml 에서 **Whisper 때문에 반드시 바뀌어야 하는 값만 수정**. 나머지는 전부 동일 (bs, max_steps, lr, scheduler, deepspeed, packing 옵션 등).

### DACVAE vs Whisper-small yaml 차이

| 키                                              | DACVAE (기존)                                              | Whisper-small                                                                                  |
| ----------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `model_name_or_path`                            | `external/models/Qwen3.5AE-4B`                             | `external/models/Qwen3.5AE-4B-whisper-small`                                                   |
| `run_name`                                      | `Qwen3.5AE-ASR-Stage1-libri_mls_vox`                       | `Qwen3.5_whisper_small_Stage1`                                                                 |
| `omni_max_audio_samples`                        | `2160000` (45s @ 48kHz)                                    | `480000` (30s @ 16kHz)                                                                         |
| (`omni_dataset_whisper.py` 내부) `sample_rate` | 48000                                                      | 16000                                                                                          |
| (`omni_dataset_whisper.py` 내부) `hop_length`  | 1920                                                       | 320 (50fps 기준)                                                                              |
| `output_dir`                                    | `external/ckpts/Qwen3.5AE-4B-ASR-Stage1`                   | `external/ckpts/Qwen3.5_whisper_small_Stage1`                                                  |

**동일 유지**: `flash_attn: fa2`, `enable_liger_kernel: true`, `stage: omni`, `deepspeed: ds_z2_config.json`, `omni_manifest` 경로, `omni_packing_bucket_size: 128`, `cutoff_len: 3584`, `packing/neat_packing`, `preprocessing_num_workers: 16`, `per_device_train_batch_size: 3`, `gradient_accumulation_steps: 1`, `learning_rate: 2.0e-4`, `weight_decay: 0.01`, `max_steps: 100000`, `lr_scheduler_type: warmup_stable_decay`, `warmup_steps: 1000`, `bf16: true`.

### 최종 yaml (`configs/qwen3_5ae-asr/stage1_projector_whisper_small.yaml`)

```yaml
### model
model_name_or_path: /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/models/Qwen3.5AE-4B-whisper-small
flash_attn: fa2
enable_liger_kernel: true
disable_gradient_checkpointing: false
trust_remote_code: true

### method
stage: omni  # omni workflow freezes all params except audio_encoder.projector
do_train: true
do_eval: false
finetuning_type: full
deepspeed: examples/deepspeed/ds_z2_config.json
compute_accuracy: false
report_to: [tensorboard, wandb]

### dataset
dataset: speechx_v9  # unused when omni_manifest is set
omni_manifest: /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/libri_mls_vox
omni_packing_bucket_size: 128
omni_max_audio_samples: 480000  # 30s @ 16kHz (Whisper 입력 상한)
template: omni
cutoff_len: 3584
overwrite_cache: false
preprocessing_num_workers: 16
streaming: true
packing: true
neat_packing: true
preprocessing_batch_size: 2
predict_with_generate: false
dataloader_pin_memory: true
dataloader_num_workers: 4
dataloader_prefetch_factor: 4
dataloader_persistent_workers: false
accelerator_config:
  split_batches: false
  dispatch_batches: false
  non_blocking: true

### output
run_name: Qwen3.5_whisper_small_Stage1
output_dir: /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/ckpts/Qwen3.5_whisper_small_Stage1
logging_dir: /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/ckpts/Qwen3.5_whisper_small_Stage1
logging_steps: 5
save_steps: 1000
save_total_limit: 8
plot_loss: false
overwrite_output_dir: true

### train
per_device_train_batch_size: 3
gradient_accumulation_steps: 1
learning_rate: 2.0e-4
weight_decay: 0.01
max_steps: 100000
lr_scheduler_type: warmup_stable_decay
warmup_steps: 1000
bf16: true
ddp_timeout: 180000000

### eval
eval_dataset: speechx_v9  # unused
```

### Launch

기존 `test_run.sh` 에서 `CONFIG` 경로만 이 yaml 로 바꿔서 실행:
```bash
llamafactory-cli train configs/qwen3_5ae-asr/stage1_projector_whisper_small.yaml
```

---

## 11. BF16 정밀도 전략

Whisper 는 원래 fp32 로 pretraining 되어 bf16 로 돌릴 때 주의 필요.

### Risk

1. **LayerNorm 정밀도** — Whisper encoder 의 모든 LN 을 bf16 로 계산하면 variance 누적 오차. Activation 분산 큰 구간에서 NaN/Inf 위험.
2. **Softmax underflow** — Attention softmax 입력이 bf16 이면 큰 음수 logit 에서 `exp → 0`. Flash-attn 2 사용 시 내부 fp32 softmax 로 자동 안전.
3. **FeatureExtractor fp32 → model bf16 경계** — `WhisperFeatureExtractor` numpy fp32 출력. `mm_plugin_whisper` 또는 모델 forward 진입 시 `.to(bfloat16)` 캐스팅 지점 명시.
4. **Positional embedding** — Sinusoidal pos embed 가 fp32 로 생성 후 bf16 cast. 미세한 rounding, 실질 영향 작음.
5. **Frozen encoder 라 backward 정밀도 이슈 없음** — forward 만 관리.

### 전략: `torch.autocast` 로 LN 만 fp32 유지

```python
# audio_encoder.py 의 forward
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        hidden = self.encoder(audio_features).last_hidden_state
```

Autocast default policy 가 LN 을 fp32 로 유지함. 단순 `.to(bfloat16)` 통째 캐스팅보다 안전.

### Smoke run 에서 확인할 것

- First 100 step loss 에 NaN/Inf 없는지
- Projector gradient norm 이 안정적인지 (폭발/소실 없음)
- Forward output activation 평균/분산 로그 (첫 step, 100 step, 1000 step)

NaN 발생 시 mitigation:
1. Whisper encoder 전체를 fp32 로 cast (`.float()`) — 속도 손해 30% 정도
2. Attention mask 의 `-inf` 를 `-1e4` 로 교체 (일부 fp16/bf16 환경에서 필요)
3. `flash_attn` → `sdpa` 로 내려서 재현성 확보 후 디버그

---

## 12. 메모리/속도 영향 예측

| 항목                      | DACVAE                     | Whisper-small                                    | 예상 영향                        |
| ------------------------- | -------------------------- | ------------------------------------------------ | -------------------------------- |
| Encoder forward params    | 27.55M                     | 88.15M                                           | **forward latency 약 3× 증가**  |
| Encoder activation memory | 작음 (conv, 25fps, 128-dim)| 큼 (12 transformer layers × 1500 × 768 × bf16) | **activation memory 수 GB 증가** |
| Projector grad memory     | 18.16M                     | 18.49M                                           | ≈ 동일                          |
| LLM forward (frozen)      | 4B                         | 4B                                               | 동일                             |
| Audio seq tokens (25→50fps) | 1× (25 tokens/s)         | **2× (50 tokens/s)**                             | **LLM seq 길이 2배, LLM forward 비용 증가** |

- LLM seq 길이가 2배로 늘면 attention 계산이 2~4× (linear layer 는 linear 하나, full-attention 은 quadratic). `cutoff_len=3584` 는 기존 DAC 에 맞춰져 있으니 Whisper 에서도 재검토.
- `per_device_train_batch_size` 를 DACVAE 기준 3 에서 그대로 두고 smoke run. OOM 시 2 로 내리고 `gradient_accumulation_steps` 를 1 → 2 로 올려 effective batch 유지.
- FSDP/ZeRO-2 하에서 encoder 도 frozen 이면 parameter shard 이득은 있지만 activation memory 는 여전히 비용.

---

## 13. 열린 이슈

### 1. 30s 초과 오디오 처리 — **Manifest 전처리 단계 Split (확정)**

Whisper 는 입력을 30s 로 zero-pad 해서 항상 `(B, 1500, 768)` encoder 출력. 30s 초과 오디오는 **manifest 전처리 단계에서 30s chunk 로 사전 분할** 하는 것을 기본 전략으로 채택.

**이유**:
- Runtime truncate 는 label mismatch 위험 (30s 초과 구간의 transcript 가 그대로 남아 audio 와 align 안 됨)
- Runtime filter 는 데이터 손실 큼 (Stage2 데이터엔 긴 오디오 많음)
- Whisper 원 학습 방식과 동일 (Whisper 도 긴 오디오는 30s chunk 로 잘라 학습)
- Stage1/Stage2 양쪽에 동일 파이프라인 적용 가능
- Manifest 단계에서 한 번만 처리하면 이후 dataloader 에서 고려할 것 없음 (모든 샘플이 이미 ≤30s)

**Manifest split 전략**:

```python
# scripts 신규: scripts/split_long_audio_30s.py (예시)
#
# 입력: 기존 manifest (shard_NNNNN.jsonl), 각 엔트리에 {audio_path, transcript, ...}
# 출력: 새 manifest, 30s 초과 엔트리가 여러 ≤30s 엔트리로 분할됨
#
# 분할 알고리즘:
# 1. audio_len = duration(audio_path)  # ffprobe or torchaudio
# 2. if audio_len <= 30.0:
#        emit entry as-is
#    else:
#        n_chunks = ceil(audio_len / 30.0)
#        for i in range(n_chunks):
#            start = i * 30.0
#            end = min(start + 30.0, audio_len)
#            # Transcript 정렬:
#            #  - 단어별 timestamp 가 manifest 에 있으면 해당 구간만 추출
#            #  - 없으면 force-aligner (wav2vec2 CTC etc.) 로 사전 align
#            #  - fallback: 시간 비율 기반 단순 분할 (quality 낮음, 마지막 수단)
#            chunk_text = extract_text_in_range(entry.transcript, start, end)
#            emit {
#                "audio_path": entry.audio_path,
#                "audio_start": start,       # 새 필드: dataloader 에서 이 구간만 load
#                "audio_end":   end,
#                "transcript":  chunk_text,
#                ...
#            }
```

**Dataloader 쪽 변경**: `audio_start` / `audio_end` 필드가 있으면 `torchaudio.load` 시 `frame_offset`, `num_frames` 로 해당 구간만 read. 없으면 (Stage1 기존 manifest) 전체 load.

**Transcript 정렬 품질**:
- LibriTTS-R, MLS 는 **utterance 단위** 라 원래 대부분 <20s. 긴 것도 하나의 문장/발화 단위라 분할 시 1~2 chunk 로 충분
- VoxPopuli 는 의회 speech 라 수분 단위 긴 파일 존재 가능 → force-aligner 필요성 있음
- Stage2 데이터 (팟캐스트/강의 등 긴 오디오) 는 단어별 timestamp 포함된 manifest 를 준비하거나 force-aligner 전처리 단계 필수

**Stage1 의 실제 영향**:
- LibriTTS-R + MLS utterance manifest 에 30s 초과 거의 없음 → split 스크립트 돌려도 변화 미미
- VoxPopuli 에서 일부만 분할됨
- 그럼에도 동일 파이프라인 확립해두면 Stage2 때 재사용 가능

**작업 분리**: Manifest split 스크립트는 이 학습 파이프라인과 독립된 별도 스크립트. `scripts/` 또는 `external/datasets/` 쪽에 배치. Stage1 smoke run 단계에선 기존 manifest 통계 찍어보고 split 전후 차이 파악.

### 2. Padding 영역 slice

실제 음성 길이가 짧으면 encoder 출력 `[B, 1500, 768]` 중 뒤쪽이 padding. `audio_pad_token` 개수가 실제 길이 기준이라면 projector 입력도 `[:, :t_audio, :]` 로 slice 해야 정렬 일치.

```python
t_audio = math.ceil(num_samples / 320)   # 50fps 기준
valid_hidden = hidden[:, :t_audio, :]     # [B, t_audio, 768]
```

smoke run 에서 `input_ids` 내 `audio_pad_token` 개수 vs projector output 길이 일치 여부 로그 찍기.

### 3. OOM 위험

Whisper encoder activation memory 증가 + LLM seq 2× 로 OOM 가능성.

OOM 시 조정 순서:
1. `per_device_train_batch_size: 3 → 2`, `gradient_accumulation_steps: 1 → 2`
2. `omni_packing_bucket_size: 128 → 64`
3. `cutoff_len: 3584 → 2048` 검토 (packing 에 영향)

### 4. HF 네트워크 접근

`convert_qwen3_5_to_qwen3_5AE_whisper.py` 실행 시 `WhisperModel.from_pretrained("openai/whisper-small.en")` 에서 HF hub 접근. 한 번 받고 나면 HF cache 에 저장 + safetensors 에 박히므로 이후 학습에선 네트워크 불필요.

노드가 네트워크 격리된 환경이면, 사전에 `huggingface-cli download openai/whisper-small.en` 으로 다운받아 `HF_HOME` 에 둔 상태에서 convert 실행.

---

## 14. 작업 체크리스트

### 준비 — 코드/체크포인트

- [ ] `external/models/Qwen3.5AE-4B-whisper-small/` dir 생성
- [ ] `audio_encoder.py` (whisper 버전) 작성 → 새 dir 에 배치
- [ ] `configuration_qwen3_5AE.py` 의 `AudioConfig` 에 whisper 필드 추가 (`whisper_model_id`, `whisper_sample_rate`, `whisper_fps`, `audio_hidden_size` 기본값 768). DAC 필드는 제거 또는 optional 로 유지 (backward compat 원하면).
- [ ] `convert_qwen3_5_to_qwen3_5AE_whisper.py` 작성 — 기존 convert 복제 후 AudioConfig 부분만 수정
- [ ] convert script 실행 → 새 dir 에 `model-*.safetensors`, `config.json`, tokenizer 등 생성 확인
- [ ] 생성된 dir 에서 `AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)` 로드 테스트 (Whisper encoder weight 가 제대로 박혔는지 `encoder.layers[0].self_attn.q_proj.weight.sum()` 등으로 확인)

### 데이터 파이프라인

- [ ] **Manifest split 스크립트 (`scripts/split_long_audio_30s.py`)** — 30s 초과 엔트리를 여러 ≤30s chunk 로 분할 (§13.1). Stage1 manifest 에선 영향 미미하나 Stage2 재사용 목적으로 작성.
- [ ] Stage1 manifest 에 split 적용 (통계: split 전후 entry 수 변화 확인)
- [ ] `src/llamafactory/data/omni_dataset_whisper.py` 신규 — Whisper mel 추출 + 16k resample + (`audio_start`/`audio_end` 있으면 구간 load) + audio_pad_token 개수 계산 (`ceil(num_samples/320)`)
- [ ] `src/llamafactory/data/mm_plugin_whisper.py` 신규 — `audio_features = [N, 80, 3000]` 대응
- [ ] `src/llamafactory/train/omni/workflow.py` freeze 로직 점검 — `audio_encoder.projector` 만 unfreeze 하는 기존 규칙이 whisper 버전에서도 그대로 동작 (encoder 는 `audio_encoder.encoder` 경로에 있음)

### yaml / launch

- [ ] `configs/qwen3_5ae-asr/stage1_projector_whisper_small.yaml` 작성 (§10 최종본)
- [ ] `test_run.sh` 에서 CONFIG 경로만 바꿔 실행 또는 별도 run script

### Smoke run

- [ ] 1~2 step 돌려서 loss/grad 정상 + OOM 여부 + Whisper encoder output mean/std 로그
- [ ] `audio_pad_token` 개수 vs projector output 길이 일치 확인
- [ ] NaN 없는지, loss 하락 시작하는지

### 본 학습

- [ ] Stage1 full run (100k steps)
- [ ] WandB 모니터링 — loss curve, projector grad norm, audio/text token ratio
- [ ] 1000 step 마다 checkpoint 저장 (`external/ckpts/Qwen3.5_whisper_small_Stage1/`)

---

## 15. Stage2 로의 diff (짧은 섹션)

Stage1 → Stage2 는 **"projector 외에 LLM 도 학습"** 이 본질적 차이. Encoder 는 그대로 frozen.

### Freeze 구성 변경

`src/llamafactory/train/omni/workflow.py` 에서 Stage2 에서는 `audio_encoder.projector` + `language_model.*` 모두 `requires_grad=True` 로. Encoder 는 여전히 frozen.

### yaml diff (Stage1 대비)

| 키                                | Stage1             | Stage2 (제안)                                                                |
| --------------------------------- | ------------------ | ---------------------------------------------------------------------------- |
| `model_name_or_path`              | whisper_small dir  | `external/ckpts/Qwen3.5_whisper_small_Stage1` (Stage1 output)                |
| `run_name`                        | `..._Stage1`       | `..._Stage2` 또는 `..._Stage2_mix` (데이터 구성에 따라)                     |
| `learning_rate`                   | `2.0e-4`           | 보통 `1e-5` ~ `5e-5` (LLM 보호)                                             |
| `per_device_train_batch_size`     | `3`                | 보통 `1`~`2` (LLM grad memory 증가로 OOM 빈도 ↑)                           |
| `gradient_accumulation_steps`     | `1`                | 늘려서 effective batch 유지                                                  |
| `disable_gradient_checkpointing`  | `false` (off=활성) | Stage2 도 동일하게 grad checkpoint 활성 권장                                 |
| `max_steps` or `num_train_epochs` | `100000`           | 데이터/수렴에 따라 재결정                                                    |
| Dataset 구성                      | ASR-only           | **TBD** — ASR + QA/Instruction mix (이 브랜치 `stage2-mix` 의도). 이번 문서 범위 밖 |

### 체크포인트 로드

Stage1 output dir (`external/ckpts/Qwen3.5_whisper_small_Stage1/`) 은 학습 완료 시 HF 표준 포맷으로 자동 저장됨:
- `model-*.safetensors` (LLM + Whisper encoder + **학습된 projector**)
- `config.json`, `modeling_*.py`, `audio_encoder.py`, tokenizer

Stage2 는 이걸 `model_name_or_path` 로 지정하면 자동 로드. Whisper encoder weight 는 Stage1 에서 frozen 이었지만 safetensors 에 그대로 저장되어 있어 Stage2 에서도 이어짐.

### Stage2 전용 결정 사항 (문서 범위 밖, 별도 세션에서)

- Dataset mix 비율 (ASR vs QA vs instruction)
- Multi-turn 대화 템플릿
- Freeze 대상 세부 (LN 만 학습? Embedding freeze?)
- Learning rate schedule (linear decay vs cosine)
- **30s chunk split 파이프라인 재사용** — Stage2 데이터엔 30s 초과 오디오 많음. Stage1 에서 확립한 split 스크립트를 Stage2 manifest 에도 적용. 이땐 단어별 timestamp / force-aligner 품질이 결과에 중요.

이번 Stage1 문서는 여기까지.
