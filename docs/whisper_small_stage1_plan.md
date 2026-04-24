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
[16kHz wav] → WhisperFeatureExtractor → log-mel (B, 80, 3000)
            → WhisperEncoder  (12 layers, d_model=768, 2x Conv1d stride=2)
            → (B, 1500, 768)  @ 50 fps  (30s 고정 padding — §5 참고)
            → AudioProjector (input_proj: 768→512, 4x LlamaDecoderLayer, output_proj: 512→2560)
            → (B, T_audio, 2560)  → Qwen3.5-4B
```

---

## 2. 파라미터 수 비교

### 모델 기준 (ASR feature extractor 용도)


| 모델                 | 원 모델 전체           | Encoder only (실제 사용분) | 비고                                                                                           |
| -------------------- | ---------------------- | -------------------------- | ---------------------------------------------------------------------------------------------- |
| Whisper-small.en     | 241,734,144 (~241.73M) | **88,154,112 (~88.15M)**   | seq2seq — decoder 153.58M 은 ASR feature 에 불필요 (token embed 39.83M 포함)                  |
| DACVAE (watermarked) | 107,671,171 (~107.67M) | **27,551,360 (~27.55M)**   | encode-path = encoder + quantizer.in_proj. 나머지 80.1M (decoder + watermarker) 은 dead weight |

→ Encoder only 기준 **DACVAE 27.55M → Whisper-small 88.15M (약 3.2× 증가)**.
→ 로드 기준 **DACVAE 107.67M → Whisper-small 88.15M (약 1.22× 감소)** (Whisper decoder 를 로드 안 하는 전제).

### Qwen3.5AE AudioEncoder 통합 기준


| 구성                           | 현재 (DACVAE)         | 변경 (Whisper-small.en)                                   |
| ------------------------------ | --------------------- | --------------------------------------------------------- |
| Audio encoder loaded           | 107,648,066 (~107.6M) | **88,154,112 (~88.15M)**                                  |
| Audio encoder active (forward) | 27,551,360 (~27.6M)   | **88,154,112 (~88.15M)**                                  |
| Dead weight (decoder etc.)     | 80,096,706 (~80.1M)   | 0                                                         |
| AudioProjector                 | 18,158,080            | **~18,485,760** (input_proj 65,536→393,216, 나머지 동일) |
| **Stage1 학습 파라미터**       | 18,158,080            | **~18,485,760** (+1.80%)                                  |

※ AudioProjector 는 `audio_hidden_size` 만 128 → 768 로 바뀌므로 `input_proj` 파라미터만 65,536 → 393,216 (+328K). 나머지 4-layer LlamaDecoder + output_proj 는 전부 동일.
※ Whisper-small.en encoder 88.15M: 12 layers × d_model=768 × 12 heads, ffn=3072.
구성: conv1 (80→768) 185K + conv2 (768→768, stride=2) 1.77M + embed_positions (1500×768) 1.15M + 12 layers 84.97M + final LN 1.5K.
※ HF 표기 "whisper-small 244M" 은 encoder+decoder 합계 (실측 241.73M). decoder 153.58M 중 ~40M 이 token embedding.
※ 전체 로드 크기 `125.8M → 106.64M` 으로 약 1.18× 감소. 다만 forward 실효 크기는 `45.7M → 106.64M` 로 **2.33× 증가**.

---

## 3. Dataset (Stage1)

**DACVAE Stage1 (현재 돌고 있는 것) 과 동일한 manifest 공유** — 별도 데이터 작업 없음.


| Code name            | Train durations (hrs) | 샘플레이트 (원본)                                                   |
| -------------------- | --------------------- | ------------------------------------------------------------------- |
| en_LibriTTS_R_single | 552.42                | (LibriTTS-R, 기본 24k → dataloader 내부에서 target SR 로 resample) |
| en_MLS_single        | 45,000.00             | 16k                                                                 |
| en_VoxPopuli_single  | 522.00                | 16k                                                                 |
| **TOTAL**            | **~46,074** hrs       |                                                                     |

**Manifest 경로** (DACVAE Stage1 와 공유):

```
/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/datasets/libri_mls_vox_shuffled_128
```

(128 shards, 총 11,345,224 엔트리 — `/mnt/fr20tb/audiollm/sanghyuk/datasets/qwen3_5_dacvae_asr_shuffled_128` 에서 CommonVoice + GigaSpeech 를 제외한 필터링 결과. 필터 스크립트: `filter_libri_mls_vox.py` 같은 디렉터리 내.)

**Libri 선택 확정 — LibriTTS-R**:

- 기존 manifest 에 있는 `en_LibriTTS_R_single` (552h) 그대로 사용.
- LibriSpeech 960h 는 nubes 에 없어서 추가 업로드 필요 → 이번 Stage1 에서는 제외. 추후 필요시 별도 작업.

---

## 4. Whisper-small.en 기본 스펙


| 항목                           | 값                                                                |
| ------------------------------ | ----------------------------------------------------------------- |
| HF repo                        | `openai/whisper-small.en`                                         |
| 입력 sample rate               | **16,000** Hz                                                     |
| FeatureExtractor hop           | 10 ms (mel 100 fps)                                               |
| Encoder Conv1d stride          | 2 × 2 →**50 fps** output                                        |
| 고정 입력 길이                 | 30 초 (3000 mel frames → 1500 encoder frames)                    |
| 짧은 입력 처리                 | zero-pad to 30s (HF default)                                      |
| 긴 입력 처리                   | truncate to 30s or chunking (Whisper-style long-form)             |
| d_model                        | 768                                                               |
| encoder layers                 | 12                                                                |
| attention heads                | 12                                                                |
| feed-forward dim               | 3072                                                              |
| encoder params                 | **88,154,112 (~88.15M)** (`WhisperModel.encoder.parameters()` 합) |
| decoder params (미사용)        | 153,580,032 (~153.58M)                                            |
| Whisper 전체 (encoder+decoder) | 241,734,144 (~241.73M) — HF 표의 "244M" 은 반올림                |

### Whisper-tiny 와의 비교 (참고)


| 항목                                        | tiny.en | base.en | **small.en** |
| ------------------------------------------- | ------- | ------- | ------------ |
| encoder params                              | 8.21M   | 20.59M  | **88.15M**   |
| d_model                                     | 384     | 512     | **768**      |
| layers                                      | 4       | 6       | **12**       |
| heads                                       | 6       | 8       | **12**       |
| ffn                                         | 1536    | 2048    | **3072**     |
| ASR WER (LibriSpeech test-clean, zero-shot) | 5.6     | 4.2     | **3.1**      |

출처: Whisper 논문 Table (LibriSpeech.test-clean, zero-shot).

→ small 은 tiny 대비 약 10.7× 크지만 WER 이 5.6 → 3.1 (거의 절반) 로 유의미하게 좋음. Projector-only Stage1 학습 결과에서도 encoder 품질이 곧 상한선이 됨.

---

## 5. 데이터 파이프라인 변경점

현재 `src/llamafactory/data/omni_dataset.py` — raw waveform 을 그대로 `audio_features` 에 담아 model 까지 전달.

```python
# 현재
sample_rate = 48000
hop_length  = 1920           # DACVAE hop
t_audio     = num_samples // hop_length
```

Whisper 로 바꿀 때 옵션 두 가지:

### Option A. DataLoader 에서 mel 추출 (선호)

- 장점: GPU forward 경로 단순, `audio_features` 가 일정한 mel tensor (B, 80, 3000) 로 통일 → collator 깔끔.
- 단점: CPU preprocessing 비용 증가 (10ms hop mel 계산).

```python
# omni_dataset.py (변경 구상)
from transformers import WhisperFeatureExtractor
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")

sample_rate = 16000
# Whisper 는 30초 고정 입력 → t_audio = 1500 (padded) or ceil(num_samples/320) (trimmed)
# audio_pad_token 수는 실제 음성 길이 기준 (padding 구간은 LLM 이 볼 필요 없음)
t_audio     = min(1500, math.ceil(num_samples / 320))   # 50 fps 기준
mel = fe(waveform, sampling_rate=16000).input_features   # (1, 80, 3000)
```

Encoder 의 attention mask 는 t_audio 만큼만 valid → projector 입력 전에 `[:t_audio]` 로 slice.

### Option B. raw waveform 유지 + encoder 내부에서 mel 추출

- 장점: 기존 `audio_features = waveform` 인터페이스 유지, dataset 코드 변경 최소.
- 단점: encoder module 이 FeatureExtractor 까지 포함, batch forward 시 padding 이중 처리 번거로움.

**결정**: Option A (mel 사전 추출). Collator / packing 도 (B, 80, 3000) 텐서 concat 만 하면 되므로 단순.

---

## 6. `audio_encoder.py` 변경 구상

```python
from transformers import WhisperModel

class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(
            config.whisper_model_id,               # "openai/whisper-small.en"
            torch_dtype=torch.bfloat16,
        ).encoder
        for p in self.encoder.parameters():
            p.requires_grad = False                # Stage1 frozen
        self.audio_dim = config.audio_hidden_size  # 768
        self.projector = AudioProjector(config)

    def forward(self, input_features, audio_token_counts=None, use_cache=False):
        # input_features: (B, 80, 3000) precomputed mel
        with torch.no_grad():
            hidden = self.encoder(input_features).last_hidden_state   # (B, 1500, 768)
        # Slice to actual audio length (projector 입력이 shorter 하면 LLM 쪽도 pad_token 수와 맞아야 함)
        audio_embeds, _ = self.projector(hidden, use_cache=use_cache)
        return audio_embeds
```

- DACVAE 관련 import, `remove_weight_norm` 루프 전부 제거.
- Gaussian noise 블록 (`if self.training and False:`) 은 제거 or 그대로 dead-code 유지 — 현재도 꺼져있어서 기능엔 영향 없음.

---

## 7. `config.json` audio_config 변경

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
    ...
  }
```

새 모델 체크포인트 디렉터리 하나 만들 필요 있음 — `/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/Qwen3.5AE-4B-whisper-small/` 에:

- 수정된 `config.json`
- 수정된 `audio_encoder.py`
- 기존 `modeling_qwen3_5AE.py`, `configuration_qwen3_5AE.py` 복사 (or 소폭 수정)
- LLM safetensors 만 복사, audio_encoder 부분은 랜덤 init (Stage1 에서 projector 학습)
- Whisper encoder 는 `from_pretrained` 로 HF 에서 런타임 로드

---

## 8. Stage1 training config (yaml)

> **원칙**: 현재 돌고 있는 DACVAE `stage1_projector.yaml` 에서 **Whisper 때문에 반드시 바뀌어야 하는 값만 수정**. 나머지는 전부 동일 (bs, max_steps, lr, scheduler, deepspeed, packing 옵션 등).

### DACVAE vs Whisper-small yaml 차이 (변경 최소화)


| 키                                             | 현재 DACVAE (돌고 있는 값)                   | Whisper-small                                                               | 변경 이유             |
| ---------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------- | --------------------- |
| `model_name_or_path`                           | `/mnt/fr20tb/audiollm/sanghyuk/Qwen3.5AE-4B` | `/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/Qwen3.5AE-4B-whisper-small` | 새 체크포인트 dir     |
| `run_name`                                     | `Qwen3.5AE-ASR-Stage1-libri_mls_vox`         | `Qwen3.5AE-WhisperSmall-ASR-Stage1-libri_mls_vox`                           | wandb 분리            |
| `omni_max_audio_samples`                       | `2160000` (45s @ 48kHz)                      | `480000` (30s @ 16kHz)                                                      | Whisper 입력 상한 30s |
| (`omni_dataset_whisper.py` 내부) `sample_rate` | 48000                                        | 16000                                                                       | Whisper 입력 SR       |
| (`omni_dataset_whisper.py` 내부) `hop_length`  | 1920                                         | 320                                                                         | Whisper 50 fps        |

**동일 유지**: `flash_attn: fa2`, `enable_liger_kernel: true`, `stage: omni`, `deepspeed: examples/deepspeed/ds_z2_config.json`, `omni_manifest` (libri_mls_vox_shuffled_128 공유), `load_from_nubes: true`, `omni_packing_bucket_size: 128`, `cutoff_len: 3584`, `packing/neat_packing`, `preprocessing_num_workers: 16`, `preprocessing_batch_size: 2`, `per_device_train_batch_size: 3`, `gradient_accumulation_steps: 1`, `learning_rate: 2.0e-4`, `weight_decay: 0.01`, `max_steps: 100000`, `lr_scheduler_type: warmup_stable_decay`, `warmup_steps: 1000`, `bf16: true`, `ddp_timeout: 180000000`, `output_dir: /mnt/tmp/results`, save/logging config 전부.

### 최종 yaml (`configs/qwen3_5ae-asr/stage1_projector_whisper_small.yaml`)

```yaml
### model
model_name_or_path: /mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/Qwen3.5AE-4B-whisper-small
flash_attn: fa2
enable_liger_kernel: true
disable_gradient_checkpointing: false
trust_remote_code: true

### method
stage: omni  # omni workflow freezes all params except audio_encoder.projector
do_train: true
do_eval: false
finetuning_type: full
deepspeed: /mnt/fr20tb/wbl_residency/jos/ddn/audiollm-trainer/examples/deepspeed/ds_z2_config.json
compute_accuracy: false
report_to: [tensorboard, wandb]

### dataset
dataset: speechx_v9  # unused when omni_manifest is set
omni_manifest: /mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/datasets/libri_mls_vox_shuffled_128
load_from_nubes: true
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
run_name: Qwen3.5AE-WhisperSmall-ASR-Stage1-libri_mls_vox
output_dir: /mnt/tmp/results
logging_dir: /mnt/tmp/results
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

### Launch script (`run_nsml_whisper_small.sh`)

DACVAE 의 `run_nsml.sh` 그대로 복사해서 `CONFIG` 경로만 바꾸는 버전:

```bash
#!/usr/bin/env bash
set -eo pipefail

source /mnt/fr20tb/wbl_residency/jos/ddn/miniforge3/bin/activate audio_lmf
export LD_PRELOAD="/mnt/fr20tb/wbl_residency/jos/ddn/miniforge3/envs/audio_lmf/lib/glibc_compat.so${LD_PRELOAD:+:$LD_PRELOAD}"

export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-qwen3_5ae-asr}"
export WANDB_API_KEY="${WANDB_API_KEY:?set via env or run_nsml.sh}"  # 평문 embed 피함
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-/mnt/fr20tb/wbl_residency/jos/ddn/audiollm-trainer/configs/qwen3_5ae-asr/stage1_projector_whisper_small.yaml}"

FORCE_TORCHRUN=1 \
NNODES="${NSML_WORLD_SIZE}" \
NODE_RANK="${NSML_RANK}" \
MASTER_ADDR="${NSML_HOST_RANK0}" \
MASTER_PORT=21267 \
llamafactory-cli train "${CONFIG}"
```

---

## 9. 메모리/속도 영향 예측

Whisper-small 은 DACVAE 보다 encoder 가 3.2× 크므로 학습 스텝당 영향이 있음:


| 항목                      | DACVAE                      | Whisper-small                                     | 예상 영향                        |
| ------------------------- | --------------------------- | ------------------------------------------------- | -------------------------------- |
| Encoder forward params    | 27.55M                      | 88.15M                                            | **forward latency 약 3× 증가**  |
| Encoder activation memory | 작음 (conv, 25fps, 128-dim) | 큼 (12 transformer layers × 1500 × 768 × bf16) | **activation memory 수 GB 증가** |
| Projector grad memory     | 18.16M                      | 18.49M                                            | ≈ 동일                          |
| LLM forward (frozen)      | 4B                          | 4B                                                | 동일                             |

- `per_device_train_batch_size` 를 DACVAE 기준 4 에서 **2~3 으로 조정** 필요할 수 있음 (OOM 여부 smoke run 으로 확인).
- `gradient_accumulation_steps` 를 늘려서 effective batch size 유지.
- FSDP 쓸 경우 encoder 도 frozen 이라도 shard 되어 activation checkpoint 이득은 projector 쪽에만.

---

## 10. 구현 결정 사항 (2026-04-22)


| # | 결정 내용                                        | 선택                                                                                                                                         |
| - | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Libri 소스                                        | **LibriTTS-R 확정** (552h) — LibriSpeech 960h 는 nubes 에 없어 이번엔 제외                                                                  |
| 2 | 새 체크포인트 디렉터리 위치                      | `/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/Qwen3.5AE-4B-whisper-small/`                                                                 |
| 3 | Shared code 수정 방식                            | **분기/새 파일** 생성 (DACVAE 학습 regression 방지) — 기존 `omni_dataset.py`, `mm_plugin.py` 는 건드리지 않고 `*_whisper.py` 변형을 따로 둠 |
| 4 | Mel 추출 전략                                    | **Option A (streaming 중 실시간 mel)** — 디스크 부담 없음, 구현 단순. 학습 속도 병목 확인되면 추후 사전 변환 전략으로 전환                  |
| 5 | Manifest                                         | **DACVAE Stage1 와 공유** (`/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/datasets/libri_mls_vox_shuffled_128`, 11,345,224 엔트리)          |
| 6 | yaml / launch script                             | **DACVAE 의 `stage1_projector.yaml` / `run_nsml.sh` 에서 Whisper 때문에 반드시 바뀌는 값만 수정** — §8 표 참조                              |

### LLM safetensors 처리 전략

새 dir 에 LLM weight (~8GB) 전체 복사하는 건 디스크 낭비. 두 가지 대안:

- **A. Symbolic link**: 새 dir 에서 sanghyuk 의 safetensors 에 심볼릭 링크. 장점 — 디스크 0 추가. 단점 — sanghyuk 경로 이동/삭제 시 깨짐.
- **B. `modeling_qwen3_5AE.py` 에서 LLM 만 원 경로에서 로드**: `audio_encoder` 만 random init, LLM 은 `from_pretrained(sanghyuk_path, subfolder="text_model")` 식. 구현 가장 깔끔하나 modeling 코드 약간 수정 필요.
- C. 그냥 복사 (가장 안전, 디스크 8GB 먹음).

→ 기본은 **A (심볼릭 링크)** 로 시작, 문제 생기면 C 로 폴백.

---

## 11. 작업 체크리스트

### 완료 (DACVAE Stage1 작업과 공유)

- [x] Libri 소스 확정 → LibriTTS-R
- [x] Manifest 확보 → `libri_mls_vox_shuffled_128` (11,345,224 엔트리, DACVAE Stage1 과 공유)

### 코드/체크포인트 준비 (즉시 시작 가능)

- [ ] 체크포인트 dir 생성: `/mnt/fr20tb/wbl_residency/jos/AudioEnc/log/tmp/Qwen3.5AE-4B-whisper-small/`
- [ ] **기존 파일 그대로 유지 (심볼릭 링크)** — sanghyuk `Qwen3.5AE-4B/` 에서 원본 변형 없이:
  - LLM weight: `model-*.safetensors`, `model.safetensors.index.json`
  - Tokenizer: `tokenizer*`, `vocab.json`, `merges.txt`, `chat_template.jinja`
  - 기존 코드 파일: `modeling_qwen3_5AE.py`, `tokenization_qwen3_5AE.py`, `configuration_qwen3_5AE.py`
- [ ] **신규 파일 생성** (DACVAE 쪽 자산 건드리지 않는 방향):
  - `audio_encoder_whisper.py` — Whisper 버전 AudioEncoder 클래스 (기존 `audio_encoder.py` 는 원본 그대로 심볼릭 링크로 두거나, 없애도 무방 — import 참조만 `audio_encoder_whisper` 로 바뀜)
  - `configuration_audio_whisper.py` (필요시) — `AudioConfig` 의 whisper 버전. 기존 `Qwen3_5AEAudioConfig` 는 건드리지 않음
  - `config.json` — 새로 작성. `audio_config` 블록에 `dac_*` 없이 `whisper_model_id`, `whisper_sample_rate`, `whisper_fps`, `audio_hidden_size: 768` 만. `auto_map` 의 AudioConfig 항목을 신규 whisper config 클래스로 지정
- [ ] 새 data pipeline 파일 (분기, DACVAE 영향 없음):
  - `src/llamafactory/data/omni_dataset_whisper.py` — Option A 실시간 mel 추출 (`sample_rate=16000`, `hop_length=320`)
  - `src/llamafactory/data/mm_plugin_whisper.py` — `audio_features` (B, 80, 3000) mel 텐서 대응
- [ ] Stage1 freeze 로직 — `train/omni/workflow.py:63-65` 가 이미 `audio_encoder.projector` 만 unfreeze 라 Whisper 도 `audio_encoder.encoder` 이면 자동 freeze. 코드 수정 불필요, 네이밍만 맞추면 됨.
- [ ] Stage1 yaml: `configs/qwen3_5ae-asr/stage1_projector_whisper_small.yaml` (§8 최종본)
- [ ] Launch script: `configs/qwen3_5ae-asr/run_nsml_whisper_small.sh` (§8 최종본)

### 실행 (DACVAE Stage1 완료 후 or 병행)

- [ ] Smoke run — 1~2 step 돌려서 loss/grad 정상 + OOM 여부 확인 (DACVAE 기준 `bs=3` 그대로 시도)
- [ ] OOM 나면 `per_device_train_batch_size: 3 → 2` + `gradient_accumulation_steps: 1 → 2` (effective batch 유지)
- [ ] 본 학습 (tmux `train` 세션 or 별도 세션, liger + FusedAdam ON)

---

## 12. 열린 이슈

### 해결됨

- ~~LibriTTS-R vs LibriSpeech 선택~~ → **LibriTTS-R 확정** (nubes 에 LibriSpeech 없음).
- ~~Manifest 생성 툴링~~ → `filter_libri_mls_vox.py` 작성 + 실행 완료. 결과 128 shards, 11,345,224 엔트리 @ `AudioEnc/log/tmp/datasets/libri_mls_vox_shuffled_128`.

### 남은 이슈

1. **Whisper encoder mask** — Whisper 는 입력을 30s 로 zero-pad 해서 항상 `(B, 1500, 768)` encoder 출력. 실제 음성 길이가 짧으면 padding 영역도 encoder hidden state 에 포함됨.
   - 영향: projector 입력이 padding 까지 포함되면, `audio_pad_token` 개수 (실제 길이 기준으로 `omni_dataset_whisper.py` 에서 결정) 와 안 맞아서 LLM 입력 정렬이 깨짐.
   - 처리: encoder 출력 후 `t_audio = ceil(num_samples / 320)` 으로 slice 해서 projector 에 전달. `audio_pad_token` 개수도 같은 공식으로 맞춤 (§5 Option A 에 명시).
   - 확인 포인트: smoke run 에서 `input_ids` 내 `audio_pad_token` 개수 vs projector output 길이 일치 여부 로그 찍기.

2. **FP32/BF16 일관성** — Whisper encoder 자체는 bf16 OK 지만 mel 추출 (`WhisperFeatureExtractor`) 은 numpy/fp32. dtype 전이 지점:
   - dataloader (fp32 mel) → collator 에서 `(B, 80, 3000)` tensor → 모델 forward 에서 `.to(bfloat16)` 캐스팅 or `torch.autocast`.
   - Whisper encoder 내부 LayerNorm 은 fp32 가 안전 — 필요시 `torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)` 로 감싸서 LN 만 fp32 유지.
   - DACVAE 에서는 autocast disable 하고 fp32 계산하는 패턴이었음 (`torch.autocast(device_type="cuda", enabled=False)`). Whisper 는 반대로 bf16 우세가 맞음 — smoke run 에서 loss NaN 안 뜨는지 먼저 확인.

3. **OOM 위험** — Whisper-small encoder activation memory (12 layers × 1500 tokens × 768 × bf16 = 약 27MB / 샘플 × packing × grad-free = 수백 MB) 가 DACVAE 대비 크게 증가. DACVAE Stage1 에서 이미 step 91 에서 한 번 OOM 발생했던 이력 있음 (setup_log §6.5).
   - 현재 DACVAE yaml 은 `per_device_train_batch_size: 3` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 로 돌아감.
   - Whisper 로 바꾸면 first smoke run 에서 OOM 확인 필요. OOM 시 조정 순서:
     1. `per_device_train_batch_size: 3 → 2`, `gradient_accumulation_steps: 1 → 2` (effective batch 유지)
     2. `omni_packing_bucket_size: 128 → 64` (bucket size 작게)
     3. 그래도 OOM 이면 `cutoff_len: 3584 → 2048` 축소 검토.

4. **Config 처리 방침 — "신규 파일 생성"** (기존 수정 X)

   기존 `configuration_qwen3_5AE.py` / `audio_encoder.py` 는 **원본 그대로 유지** (심볼릭 링크). Whisper 용 설정은 아래처럼 **새 파일로** 만듦:

   - `audio_encoder_whisper.py` — 새 파일. Whisper 버전 `AudioEncoder` 클래스 정의. `from dacvae import DACVAE` 같은 기존 import 없이 `from transformers import WhisperModel` 로 시작.
   - `configuration_audio_whisper.py` — (필요한 경우) 새 `AudioConfig` 클래스. `dac_*` 없이 `whisper_*` 키만. 기존 `Qwen3_5AEAudioConfig` 는 그대로 원본 참조.
   - `config.json` — 새 dir 안에 새로 작성. 내용:
     ```json
     "audio_config": {
       "adapter_hidden_size": 512,
       "audio_hidden_size": 768,
       "whisper_model_id": "openai/whisper-small.en",
       "whisper_sample_rate": 16000,
       "whisper_fps": 50,
       ... (nondac keys 유지: head_dim, intermediate_size, llm_embed_size, num_adapter_layers, num_attention_heads 등)
     }
     ```
     `auto_map` 에서 audio-쪽 클래스 참조를 새 파일/새 클래스로 연결.

   **장점**:
   - 기존 DACVAE 쪽 코드/config 는 바이트 한 개도 안 건드림 → regression 제로
   - sanghyuk 원본 경로의 파일을 바꾸지 않아서, 만약 sanghyuk 가 그 경로에 뭔가 업데이트해도 충돌 없음 (심볼릭 링크는 항상 최신 참조)
   - 두 구성을 동시 유지 가능 (디버깅 시 DACVAE 로 돌아가기 쉬움)

   **유일한 주의점**: `modeling_qwen3_5AE.py` 내부에서 audio_encoder 를 어떤 경로로 import 하는지 확인. 만약 `from .audio_encoder import AudioEncoder` 로 hard-coded 되어있으면 `modeling_qwen3_5AE_whisper.py` 라는 wrapper 를 하나 더 만들거나, 새 dir 의 `audio_encoder.py` 를 whisper 버전으로 덮어 쓰기 (이 경우엔 sanghyuk 원본 audio_encoder.py 는 심볼릭 링크 **하지 않고**, 새 dir 의 `audio_encoder.py` = 새로 작성한 whisper 버전).

   → 실제 파일 건드려 보기 전에 `modeling_qwen3_5AE.py` 의 audio_encoder import 경로만 한 번 grep 해서 확정.

5. **Checkpoint resume / mid-run 호환성** — DACVAE 로 돌고 있는 run 의 중간 체크포인트는 Whisper 모델과 shape 이 안 맞음. Whisper Stage1 은 반드시 처음부터 새로 시작 (resume 아님). `overwrite_output_dir: true` 가 설정돼있어 충돌 없음, 다만 `output_dir` 을 `run_name` 포함 새 경로로 두거나 따로 관리 필요.
