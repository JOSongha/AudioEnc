# train_pipeline 수정 로그

**작성일: 2026-04-08**


---
- [train\_pipeline 수정 로그](#train_pipeline-수정-로그)
  - [0. 패키지 버전](#0-패키지-버전)
  - [1. 오디오 로딩: torchcodec → soundfile](#1-오디오-로딩-torchcodec--soundfile)
    - [에러 로그](#에러-로그)
    - [원인](#원인)
    - [수정](#수정)
  - [2. 모델 아키텍처 / Forward / Prompt Template](#2-모델-아키텍처--forward--prompt-template)
    - [Forward 시그니처](#forward-시그니처)
    - [Prompt / Token Template](#prompt--token-template)
    - [모델안에서](#모델안에서)
  - [3. 데이터 파이프라인](#3-데이터-파이프라인)
    - [전체 구조 변화](#전체-구조-변화)
    - [Import 정리 (`_torchcodec.py` → `_override.py`)](#import-정리-_torchcodecpy--_overridepy)
    - [Config 의존성 정리](#config-의존성-정리)
  - [4. 학습 루프 / 실행 방식](#4-학습-루프--실행-방식)
  - [5. 평가 (Validation / WER)](#5-평가-validation--wer)
  - [6. 런타임 실행 오류 (실행 테스트 2026-04-08)](#6-런타임-실행-오류-실행-테스트-2026-04-08)
    - [6.11 word-aug: CUDA OOM (메모리 단편화)](#611-word-aug-cuda-oom-메모리-단편화)
    - [6.12 word-aug: 300× 속도 저하 (혼합 길이 패딩 폭발)](#612-word-aug-300-속도-저하-혼합-길이-패딩-폭발)
    - [6.13 word-aug (precomputed): `_project_precomputed` pad-stack OOM](#613-word-aug-precomputed-_project_precomputed-pad-stack-oom)
    - [6.14 run.sh에 `--precomputed-dir` 누락 → raw audio fallback](#614-runsh-에-precomputed-dir-누락--raw-audio-fallback)
    - [6.15 `evaluate_val_loss` raw waveform을 precomputed collator에 전달](#615-evaluate_val_loss-raw-waveform을-precomputed-collator에-전달)
    - [6.16 §6.4 `_save` override regression + save_strategy/save_safetensors 우회](#616-64-_save-override-regression--save_strategysave_safetensors-우회)
    - [6.17 NCCL timeout: rank 간 bin 개수 불균형 (split 8 deadlock)](#617-nccl-timeout-rank-간-bin-개수-불균형-split-8-deadlock)
    - [6.18 wandb 내장 WandbCallback split 경계 finish → 로그 단절](#618-wandb-내장-wandbcallback-split-경계-finish--로그-단절)


## 0. 패키지 버전 
**torch 2.11 with CUDA 12.8 support**
conda create -n venv_torch211 python=3.11 -y && conda activate venv_torch211
conda install -c conda-forge "ffmpeg<8" -y
conda install -c nvidia cuda-toolkit=12.8

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install datasets transformers huggingface PyYAML tqdm einops -q
pip install torchcodec --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"

pip install accelerate transformers wandb einops git+https://github.com/descriptinc/audiotools flash-linear-attention liger-kernel peft
pip install transformers --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation

## 1. 오디오 로딩: torchcodec → soundfile

### 에러 로그

```
RuntimeError: Could not load libtorchcodec. Likely causes:
      1. FFmpeg is not properly installed in your environment.
      2. The PyTorch version (2.10.0+cu126) is not compatible with ...

Traceback:
  File ".../torchcodec/_core/ops.py", line 38, in <module>
    load_torchcodec_shared_libraries()
  RuntimeError: Could not load libtorchcodec.
```

`torchaudio.load()` 경유 시에도 동일 에러:

```
File ".../torchaudio/_torchcodec.py", line 246, in save_with_torchcodec
    from torchcodec.encoders import AudioEncoder
RuntimeError: Could not load libtorchcodec.
```

백엔드 우회 시도 시:

```
set_audio_backend error: module 'torchaudio' has no attribute 'set_audio_backend'
```

### 원인

- `datasets 4.x`부터 오디오 디코딩 백엔드가 `soundfile` → `torchcodec + FFmpeg`로 변경됨
- `torchaudio 2.10`도 모든 I/O를 torchcodec으로 라우팅 → `torchaudio.load()`도 torchcodec을 import
- `torchcodec==0.10.0`이 `.so.13` soname 요구하는데 CUDA 12.6 환경에는 `.so.12`만 존재  
  (`libcudart.so.13`, `libnvrtc.so.13`, `libnppicc.so.13` 미존재)
- `torchaudio.set_audio_backend()` API는 torchaudio 2.x에서 삭제됨

참고: [Torch ↔ TorchCodec compatibility table](https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec) · [HF Discussion](https://discuss.huggingface.co/t/issue-with-torchcodec-when-fine-tuning-whisper-asr-model/169315)

### 수정

**방법 A (권장): `soundfile`로 직접 로딩** — `_override.py`에 적용

```python
import soundfile as sf

def _load_audio(audio_obj) -> tuple[torch.Tensor, int]:
    """Load audio using soundfile (no torchcodec dependency)."""
    if "bytes" in audio_obj and audio_obj["bytes"] is not None:
        data, sr = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32", always_2d=True)
    elif "path" in audio_obj and audio_obj["path"] is not None:
        data, sr = sf.read(audio_obj["path"], dtype="float32", always_2d=True)
    else:
        raise ValueError("audio_obj has neither 'bytes' nor 'path'")
    # soundfile: (frames, channels) → torchaudio convention: (channels, frames)
    return torch.from_numpy(data.T), sr
```

- `StaticEvalDataset.__getitem__`, `create_processor` 내 `torchaudio.load()` 4곳 교체
- `soundfile.read(dtype="float32")`: [-1.0, 1.0] 정규화 — torchaudio 동작과 동일
- FLAC, WAV, OGG 지원 (LibriSpeech/MLS/GigaSpeech 포맷)

**방법 B: datasets 버전 고정**

```bash
pip install "datasets[audio]==3.6.0"  # soundfile 백엔드 사용하는 마지막 버전
```

**방법 C: CUDA symlink** (torchcodec 유지하면서 버전 불일치 우회)

```bash
ln -s libnvrtc.so.12   libnvrtc.so.13
ln -s libnppicc.so.12  libnppicc.so.13
ln -s libcudart.so.12  libcudart.so.13
```

| 항목 | `_torchcodec.py` (원본) | `_override.py` (수정) |
|------|------------------------|----------------------|
| 로딩 함수 | `torchaudio.load()` 직접 호출 | `_load_audio()` 헬퍼 (`soundfile.read()`) |
| 로딩 위치 (main 기준) | `dataset.py` collate_fn 내부 | `_load_audio()` 헬퍼로 분리 |
| 의존 라이브러리 | torchaudio → torchcodec → FFmpeg | soundfile (순수 C 바인딩, FFmpeg 불필요) |

---

## 2. 모델 아키텍처 / Forward / Prompt Template

`main:train.py + model.py`와 `_override.py`의 가장 큰 구조적 차이.

### Forward 시그니처

**`main:model.py`** — raw audio를 받아 모델 내부에서 prompt 조합:

```python
def forward(self, audio, audio_lengths=None, transcript_input_ids=None, ...):
    # p1("Audio:\n") + audio_embeds + p2("\nTranscript:\n") + transcript 를 내부에서 concat
    p1_embeds = embed(self.prompt_p1_ids).expand(B, -1, -1)  # register_buffer
    p2_embeds = embed(self.prompt_p2_ids).expand(B, -1, -1)
    inputs_embeds = torch.cat([p1_embeds, audio_embeds, p2_embeds, tgt_embeds], dim=1)
```

**`_override.py`** — 이미 조합된 `input_ids`를 받아 audio placeholder를 audio embedding으로 치환:

```python
def forward(self, input_ids, labels, audio_features, audio_lengths,
            attention_mask=None, position_ids=None, num_items_in_batch=None, ...):
    # input_ids에 audio_pad_token_id(151655) placeholder가 삽입된 채로 들어옴
    audio_pad_mask = (input_ids == self.audio_pad_token_id)
    inputs_embeds[audio_pad_mask] = audio_flat.to(inputs_embeds.dtype)
```

### Prompt / Token Template

| 항목 | `main:model.py` | `_override.py` |
|------|----------------|---------------|
| 구조 | `"Audio:\n"` + audio_embeds + `"\nTranscript:\n"` + text | `[audio_pad × t_audio]` + `<\|audio_correspond\|>` + text + EOS |
| audio 삽입 방식 | forward 내에서 embed concat | `create_processor`가 `input_ids`에 placeholder 삽입 → forward에서 치환 |
| audio placeholder token | 없음 (embed를 직접 concat) | `audio_pad_token_id = 151655` (Qwen2.5 `<\|image_pad\|>` 재활용) |
| 경계 토큰 | p1/p2 문자열을 `register_buffer`로 저장 | `<\|audio_correspond\|>` special token 추가 + `resize_token_embeddings` |

### 모델안에서

| 항목 | `main:train.py` | `_override.py` |
|------|----------------|---------------|
| `AudioQwen` 위치 | 별도 `model.py` | 파일 내 직접 정의 (자기 완결형) |
| Liger kernel 적용 | 별도 처리 필요 | `AudioQwen.__init__` 내 모델 로드 직전 자동 적용 |
| Stage별 encoder forward | 단일 `_get_audio_embeds` | `_get_audio_embeds_s1` (DDP batch) / `_get_audio_embeds_s2` (FSDP chunked) 분리 |

---

## 3. 데이터 파이프라인

### 전체 구조 변화

| 항목 | `main:train.py` | `_override.py` |
|------|----------------|---------------|
| 데이터셋 타입 | map-style (`build_datasets`, 로컬 다운로드) | HF Streaming (`load_dataset(streaming=True)`) |
| 샤딩 시점 | Sampler 단계 (처리 후) | Load 직후 — `ds.skip(rank).take_every(world_size)` |
| 배치 구성 | `DynamicBatchSampler` (max_batch_tokens 상한, greedy) | `create_packer` → `build_multi_dataset_streaming_pipeline` |
| Collate | `collate_fn_factory(tokenizer, max_text_len)` | `OmniCollator` (4D block-diagonal mask + position_ids 생성) |
| Trainer | 수동 루프 (`optimizer.step` / `scheduler.step`) | HF `Trainer` / `StreamingShardedTrainer` |

**`StreamingShardedTrainer`** 가 필요한 이유: HF Trainer는 DDP 환경에서 자동으로 `DistributedSampler`를 붙이는데, streaming dataset은 이미 최상단에서 샤딩됨 → 이중 샤딩 방지.

### Import 정리 (`_torchcodec.py` → `_override.py`)

| 제거된 import | 이유 |
|--------------|------|
| `DistributedSampler` | `StreamingShardedTrainer`로 대체, 직접 미사용 |
| `import datasets` (bare) | `from datasets import ...` 로 충분 |
| `DataLoader` 중복 import | 두 번 선언됨 |
| `build_datasets`, `get_dataset_lengths`, `PackedDataset`, `PackedCollator`, `build_packed_processor` | streaming 파이프라인으로 교체, 미사용 |
| `evaluate_wer`, `_compute_wer`, `_edit_distance` | 파일 내 인라인으로 전환 |

### Config 의존성 정리

| 항목 | `_torchcodec.py` (원본) | `_override.py` (수정) |
|------|------------------------|----------------------|
| 설정 소스 | `from config import get_config` | `TRAIN_CONFIG` dict 파일 내 직접 정의 |
| `audio_pad_token_id` | 없음 | `TRAIN_CONFIG`에 `151655` 추가 |
| `sample_rate` | `get_config()` 내부에만 존재 | `TRAIN_CONFIG["sample_rate"] = 16000` 명시 |
| `WerCallback`, `evaluate_wer` | `from train_pipeline import ...` | 파일 내 인라인 |

---

## 4. 학습 루프 / 실행 방식

| 항목 | `main:train.py` | `_override.py` |
|------|----------------|---------------|
| 실행 커맨드 | `torchrun --nproc_per_node=8 train.py` | `accelerate launch --num_processes=8 --mixed_precision=bf16` |
| 분산 초기화 | `InitProcessGroupKwargs(timeout=7200s)` 수동 | Accelerate 자동 관리 |
| NCCL 설정 | `TORCH_NCCL_BLOCKING_WAIT`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`, `NCCL_DEBUG` 수동 | 없음 |
| Stage 1 LR | cosine_schedule | LR 고정 |
| LR Scheduler | `get_cosine_schedule_with_warmup` 수동 생성 | `TrainingArguments(lr_scheduler_type, warmup_ratio)` |
| Steps 계산 | `len(dataloader) × epochs` | `calculate_max_steps(estimated_hours, ...)` (streaming → hours 기반 추산) |
| Stage 1 early stop | `kill -USR1 <pid>` → `SIGUSR1` handler → `_stop_stage1 = True` | 없음 |
| Stage 선택 플래그 | `--stage2-only` (bool) | `--stage {all, s1, s2}` |
| 모델 저장 | - | 저장 ckpt에 runid 붙여서 overwrite 방지 |

---

## 5. 평가 (Validation / WER)

| 항목 | `main:train.py` | `_override.py` |
|------|----------------|---------------|
| 평가 지표 | CE Loss (`run_validation`) | WER (`evaluate_wer` + greedy decoding) |
| 평가 데이터 | val split 전체 | LibriSpeech dev-clean 300샘플 (`StaticEvalDataset`) |
| 평가 호출 방식 | step마다 직접 호출 | `WerCallback` (HF `TrainerCallback`) |
| best 모델 저장 | `best_{encoder}_ckpt_*/` (전체 체크포인트) | `best_s1_proj.pt` (projector 가중치만) |

**`WerCallback` 수정 이력**: 미사용 import 정리 작업 중 `WerCallback`이 의존하는 심볼이 함께 삭제되어 `NameError` 발생. 삭제된 항목:
- `from torch.distributed.fsdp import FullyShardedDataParallel as FSDP` → `WerCallback.on_step_end` 내 `FSDP.summon_full_params` 호출
- `from transformers.trainer_callback import TrainerCallback` → `WerCallback`의 base class
- `evaluate_wer`, `_compute_wer`, `_edit_distance` → import 제거 후 파일 내 인라인으로 복원

---

## 6. 런타임 실행 오류 (실행 테스트 2026-04-08)

`train_pipeline_override.py` 첫 실행 시 발견된 버그 목록. 패키지 버전 충돌 2건 + 모델 구현 버그 4건.

### 6.1 transformers 버전: Qwen3.5 아키텍처 미지원

```
KeyError: 'qwen3_5'
ValueError: The following model type is not supported: qwen3_5.
  Supported types: ...
```

**원인**: 설치된 transformers 4.57.6은 Qwen3.5 아키텍처(`qwen3_5`) 등록 이전 버전.

**수정**:
```bash
pip install --upgrade transformers   # → 5.5.0
```

---

### 6.2 datasets 버전: `IterableDataset.shard()` 미존재

```
AttributeError: 'IterableDataset' object has no attribute 'shard'
  File "train_pipeline_override.py", line 918, in build_multi_dataset_streaming_pipeline
    ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)
```

**원인**: `IterableDataset.shard()`는 datasets 3.0에서 추가됨. 설치된 2.21.0에는 없음.  
datasets 4.x는 torchcodec 의존성으로 인해 사용 불가 (섹션 1 참조).

**수정**:
```bash
pip install "datasets>=3.0,<4.0"   # → 3.6.0
```

---

### 6.3 인코더 dtype 충돌: encoder Conv1d float32 vs bf16 bias

```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
  File ".../encoders/dacvae.py", in forward
    x = self.encoder(audio_in.float(), ...)
```

**원인**: `freeze_llm()`과 `apply_lora()` 두 곳에서 `self.encoder.to(dtype=torch.bfloat16)` 호출.  
`fb_dacvae` 인코더 내부는 `autocast(enabled=False)` + `audio_in.float()`로 항상 float32 입력을 강제하는데, Conv1d bias가 bf16으로 캐스팅되어 dtype mismatch 발생.

**수정**: 두 메서드에서 encoder dtype 캐스팅 제거.

```python
# freeze_llm() — 제거
- self.encoder.to(dtype=torch.bfloat16)

# apply_lora() — 제거
- self.encoder.to(dtype=torch.bfloat16)
```

**주의**: `fb_dacvae`처럼 내부적으로 float32를 강제하는 encoder는 bf16 환경에서도 encoder 전체를 float32로 유지해야 한다. encoder를 bf16으로 캐스팅하면 안 됨.

---

### 6.4 Qwen3.5 Weight Tying + safetensors 직렬화 오류

```
RuntimeError: Some tensors share memory, this will lead to duplicate memory on disk:
  [{'llm.lm_head.weight', 'llm.model.embed_tokens.weight'}]
```

**원인**: Qwen3.5는 `lm_head.weight`와 `embed_tokens.weight`가 동일 메모리를 공유(weight tying).  
safetensors는 공유 메모리 텐서 직렬화를 거부함.

**실패한 우회 시도**: `TrainingArguments(save_safetensors=False)` → transformers 5.5.0에서 해당 인자 제거됨 (→ 6.5 참조).

**수정**: `StreamingShardedTrainer._save` 오버라이드 — state_dict 수집 후 data_ptr 중복 탐지 및 클론:

```python
def _save(self, output_dir=None, state_dict=None):
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if state_dict is None:
        state_dict = self.model.state_dict()

    # data_ptr 중복 탐지: 나중에 나온 쪽을 클론해 공유 해제
    seen_ptrs: dict = {}
    for k in list(state_dict.keys()):
        ptr = state_dict[k].data_ptr()
        if ptr in seen_ptrs:
            state_dict[k] = state_dict[k].clone()
        else:
            seen_ptrs[ptr] = k

    super()._save(output_dir, state_dict=state_dict)
```

이 방식은 LoRA 래핑 여부(Stage 1 vs Stage 2)에 무관하게 동작한다.  
Stage 2(LoRA 적용 후)에서는 경로가 `llm.base_model.model.lm_head.weight` / `llm.base_model.model.model.embed_tokens.weight`로 바뀌지만, data_ptr 비교로 탐지하므로 경로 무관.

**저장 검증 결과** (Stage 2 체크포인트 `/mnt/tmp/cache/hf/fb_dacvae/s2_outputs_0408_1711`):

```python
from safetensors import safe_open
import torch

with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    keys = [k for k in f.keys() if "lm_head" in k or "embed_tokens" in k]
    tensors = {k: f.get_tensor(k) for k in keys}

# 결과:
# 값 동일: True   ← weight tying이 학습 중 올바르게 유지됨
# 메모리 공유: False ← 파일에는 독립 복사본으로 저장 (safetensors 직렬화 성공)
```

로드 시 weight tying 복원: `AutoModelForCausalLM.from_pretrained()`이 `config.json`의  
`tie_word_embeddings=True`를 읽고 `tie_weights()` 호출 → 자동으로 공유 메모리 복원됨.

---

### 6.5 `TrainingArguments.save_safetensors` 인자 제거됨

```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'save_safetensors'
```

**원인**: transformers 5.5.0에서 `save_safetensors` 파라미터 제거.

**수정**: `TrainingArguments(save_safetensors=False, ...)` → `save_safetensors` 인자 삭제.  
safetensors 저장 우회는 `_save` 오버라이드(6.4)로 처리.

---

### 6.6 `Trainer.save_model(safe_serialization=False)` 인자 제거됨

```
TypeError: Trainer.save_model() got an unexpected keyword argument 'safe_serialization'
```

**원인**: transformers 5.5.0에서 `save_model(safe_serialization=...)` 파라미터 제거.  
Stage 2 종료 후 `trainer.save_model(s2_output_dir, safe_serialization=False)` 호출이 실패.

**수정**: `safe_serialization` 인자 제거.

```python
# 수정 전
trainer.save_model(s2_output_dir, safe_serialization=False)

# 수정 후
trainer.save_model(s2_output_dir)
```

safetensors 공유 메모리 문제는 `_save` 오버라이드(6.4)에서 처리하므로 인자 불필요.

---

### 6.7 flash_attn GLIBC_2.32 미지원 (Ubuntu 20.04 시스템)

```
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
  (required by flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so)
```

**환경**: 시스템 GLIBC 2.31 (Ubuntu 20.04), flash_attn 2.8.3

**원인**: PyPI에서 배포되는 flash_attn 2.8.3 prebuilt wheel은 GLIBC_2.32 이상을 빌드 타깃으로 컴파일됨.  
구체적으로 `__libc_single_threaded@GLIBC_2.32` 심볼 하나만 사용하며, 이는 GLIBC 2.32에서 추가된 스레딩 최적화 힌트 변수다.  
Ubuntu 20.04의 시스템 GLIBC는 2.31이므로 이 심볼을 제공하지 못해 동적 링커가 로드를 거부한다.

**시도했으나 실패한 방법들**:

1. **소스 빌드** (`--no-binary flash-attn`): 시스템 GCC 9.4.0 + CUDA 11.8 NVCC로 소스 빌드했으나 결과 바이너리도 동일하게 GLIBC_2.32 요구. 빌드 툴체인(CUDA toolkit 또는 링커 설정)이 GLIBC_2.32 심볼을 끌어들임.

2. **LD_PRELOAD (버전 태그 없음)**: `__libc_single_threaded = 0`만 정의한 compat .so를 preload해도 실패. 동적 링커가 `libc.so.6`에서 특정 버전 `GLIBC_2.32`을 확인하므로 다른 이름의 라이브러리로 우회 불가.

3. **LD_PRELOAD (버전 태그 포함)**: `--version-script`로 `GLIBC_2.32 { __libc_single_threaded; };`를 정의해도 실패. VERNEED 항목이 `filename: libc.so.6`을 명시하므로 다른 .so 파일의 버전 태그는 검사 대상에서 제외됨.

4. **문자열만 패치** (`GLIBC_2.32` → `GLIBC_2.17`): ELF VERNEED 섹션의 문자열 테이블만 바꾸면 동적 링커의 해시 검사에서 실패. VERNEED Vernaux 구조체의 `vna_hash` 필드가 `GLIBC_2.32`의 ELF 해시값(0x069691b2)으로 남아 있어 이름과 해시가 불일치.

**수정 (최종)**:

두 단계로 flash_attn 바이너리를 수정한다.

**단계 1**: `patchelf --clear-symbol-version`으로 `.dynsym`의 심볼 버전 제거 (unversioned로 만들어 LD_PRELOAD로 제공 가능하게 함)

```bash
# flash_attn site-packages 디렉터리로 이동 후
# 백업
cp flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so{,.bak}
# 심볼 버전 제거
patchelf --clear-symbol-version __libc_single_threaded \
  flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so
```

**단계 2**: VERNEED의 문자열 + 해시를 함께 패치 (GLIBC_2.32 → GLIBC_2.17, 시스템 libc가 제공하는 버전으로 교체)

```python
import struct

def elf_hash(name):
    h = 0
    for c in name.encode('ascii'):
        h = ((h << 4) + c) & 0xffffffff
        g = h & 0xf0000000
        if g:
            h = (h ^ (g >> 24)) & 0xffffffff
        h = (h & ~g) & 0xffffffff
    return h

# GLIBC_2.32 hash: 0x069691b2, GLIBC_2.17 hash: 0x06969197
SO = ".../flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so"
with open(SO, 'rb') as f:
    data = bytearray(f.read())
# 1) 문자열 교체 (null 포함, 동일 길이 10바이트)
data = bytearray(bytes(data).replace(b'GLIBC_2.32\x00', b'GLIBC_2.17\x00'))
# 2) 해시 교체 (LE uint32)
old_hash = struct.pack('<I', 0x069691b2)
new_hash = struct.pack('<I', 0x06969197)
data = bytearray(bytes(data).replace(old_hash, new_hash))
with open(SO, 'wb') as f:
    f.write(data)
```

**단계 3**: unversioned `__libc_single_threaded` 심볼을 LD_PRELOAD로 제공

```bash
# GLIBC 2.31에는 __libc_single_threaded가 없으므로 직접 정의
printf 'int __libc_single_threaded = 0;\n' > /tmp/glibc_compat.c
gcc -shared -fPIC -o /tmp/glibc_compat.so /tmp/glibc_compat.c
```

**실행 시 항상 LD_PRELOAD 필요**:

```bash
LD_PRELOAD=/tmp/glibc_compat.so accelerate launch --num_processes N \
  train_pipeline_override.py --attn-impl flash_attention_2 ...
```

**검증**:
```bash
# VERNEED에 GLIBC_2.32 없어야 함
objdump -p flash_attn_2_cuda.so | grep GLIBC
# __libc_single_threaded가 unversioned인지 확인
readelf -W --dyn-syms flash_attn_2_cuda.so | grep __libc_single_threaded
```

**주의사항**:
- 이 패치는 재설치 시 초기화되므로 flash_attn 업그레이드/재설치 후 재적용 필요
- `/tmp/glibc_compat.so`는 재부팅 시 삭제되므로 영구 경로에 보관 권장  
- `__libc_single_threaded = 0`은 "항상 멀티스레드 모드"로 설정 — libc의 최적화 힌트가 꺼지는 것이므로 정확성에는 영향 없음

---

### 6.8 flash_attn GLIBCXX_3.4.29 미지원 (GCC 9 libstdc++)

6.7 패치 후 다음 에러 연속 발생:

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
  (required by flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so)
```

**원인**: GLIBCXX_3.4.29는 GCC 12 / libstdc++ 12에서 추가됨.  
Ubuntu 20.04 기본 GCC 9의 시스템 libstdc++는 최대 GLIBCXX_3.4.28을 제공.

**수정**: GCC 12+ libstdc++가 포함된 conda 환경의 라이브러리를 LD_PRELOAD로 사전 로드.

```bash
# 확인: GLIBCXX_3.4.29 이상이 있는 libstdc++ 경로
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep "GLIBCXX_3.4.29"

# 최종 실행 명령 (GLIBC + GLIBCXX 두 compat 모두 preload)
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/glibc_compat.so" \
  accelerate launch --num_processes N \
  train_pipeline_override.py --liger --fsdp ...
```

**glibc_compat.so 위치**: `$CONDA_PREFIX/lib/glibc_compat.so`에 영구 보관.  
재부팅 후에도 별도 재생성 불필요. (생성 방법은 §6.7 참조)

**flash_attn_2_cuda.so 패치 필요**: flash_attn을 재설치하면 GLIBC_2.32 VERNEED가 다시 생김.  
재설치 후 §6.7의 패치 절차를 재적용할 것.

---

### 6.9 MLS / VoxPopuli OGG-Opus: soundfile 실패 → torchcodec fallback

**증상**: MLS 또는 VoxPopuli 데이터셋 로드 시 다음과 같은 오류 또는 torchcodec 관련 import 에러 발생.

```
RuntimeError: soundfile: Error opening .../audio.opus: File contains data in an unknown format
# 또는
ModuleNotFoundError: No module named 'torchcodec'
```

**원인**:
- MLS (Multilingual LibriSpeech) 와 VoxPopuli는 오디오를 **OGG-Opus 포맷**으로 저장함.
- HuggingFace `datasets`는 오디오 디코딩 시 `soundfile`을 기본 백엔드로 시도하나, soundfile은 OGG-Opus를 지원하지 않음.
- 디코딩 실패 시 HF datasets가 `torchcodec`을 fallback으로 시도. 이 환경에 torchcodec이 없으면 import 에러 발생.

**해결책**: HF datasets의 오디오 자동 디코딩을 우회하고, `torchaudio.load()`로 직접 디코딩.

```python
# dataset.py: Audio(decode=False)로 raw bytes만 수신
ds = load_dataset(...).cast_column("audio", Audio(decode=False))

# train_pipeline_override.py: _load_audio()가 torchaudio로 직접 디코딩
def _load_audio(audio_obj):
    if "bytes" in audio_obj and audio_obj["bytes"] is not None:
        waveform, sr = torchaudio.load(io.BytesIO(audio_obj["bytes"]))
    elif "path" in audio_obj and audio_obj["path"] is not None:
        waveform, sr = torchaudio.load(audio_obj["path"])
    ...
```

**`HF_DATASETS_AUDIO_BACKEND` 환경변수에 대하여**:  
remote에서 `os.environ.setdefault("HF_DATASETS_AUDIO_BACKEND", "soundfile")`이 추가되었으나,  
soundfile은 OGG-Opus를 지원하지 않으므로 이 값은 잘못됨.  
`decode=False`가 HF 디코딩을 우회하므로 실제로는 무시되지만, `"torchaudio"`로 수정하여 의미를 명확히 함.

---

### 6.10 FSDP + PEFT LoRA: mixed-dtype 오류

**증상**: Stage 2 FSDP 학습 시 다음 오류 발생.

```
ValueError: Must flatten tensors with uniform dtype but got torch.float32 and torch.bfloat16
  File "torch/distributed/fsdp/_flat_param.py", in _validate_tensors_to_flatten
```

**원인**:
- `get_peft_model()`은 LoRA adapter 파라미터 (`lora_A.weight`, `lora_B.weight`)를 기본 dtype (fp32)으로 초기화함.
- 기반 LLM은 bfloat16으로 로드되어 있으므로 동일 module 내 dtype이 혼재함.
- FSDP는 각 shard 단위 내 파라미터 dtype 균일성을 강제함 → `_validate_tensors_to_flatten` 에서 ValueError.

**시도된 부분 해결책**:
- `self.llm = self.llm.to(torch.bfloat16)` in `apply_lora()`: LoRA 초기화 직후 캐스트.
- `model = model.to(torch.bfloat16)` in `run_stage2()`: 전체 모델 캐스트.

그러나 위 두 방법만으로는 여전히 실패. `model.to()`가 PeftModel 내부의 adapter weight를 건너뛰거나,  
이후 `load_state_dict(proj_state)` 호출이 fp32 파라미터를 재삽입할 가능성이 있음.

**최종 해결책**: `load_state_dict` 이후, Trainer 생성 직전에 명시적 per-param 캐스트 추가.

```python
model.apply_lora()
# ... load_state_dict(proj_state) ...

# FSDP shard dtype 균일성 보장: load_state_dict 이후 잔존 fp32 파라미터 강제 캐스트
_fp32_params = [(n, p.dtype) for n, p in model.named_parameters()
                if p.is_floating_point() and p.dtype != torch.bfloat16]
if _fp32_params:
    logger.info(f"Casting {len(_fp32_params)} non-bf16 param(s) to bf16 for FSDP: ...")
    for param in model.parameters():
        if param.is_floating_point() and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
```

`model.to()` 대신 `param.data = param.data.to(...)` 직접 수정을 사용하는 이유:  
`nn.Module.to()`는 일부 커스텀 모듈 (PeftModel, Liger 패치 모듈 등)에서 adapter weight를  
건너뛸 수 있음. `param.data` 직접 수정은 어떠한 커스텀 `.to()` 오버라이드도 우회하여  
반드시 모든 파라미터를 캐스트함.

---

### 6.11 word-aug: CUDA OOM (메모리 단편화)

**발생 시점**: `--word-aug` 플래그로 학습 시작, Step 2 backward 중 GPU 4에서 OOM.

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 14.38 GiB.
GPU 4 has a total capacity of 79.33 GiB of which 12.64 GiB is free.
Process 110386 has 66.63 GiB memory in use. Of the allocated memory 43.49 GiB
is allocated by PyTorch, and 21.50 GiB is reserved by PyTorch but unallocated.
Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

**원인**: word-aug로 한 bin에 단어 클립이 많아지면 backward 시 대형 연속 블록을 할당해야 하는데,  
21.5 GiB가 예약되어 있지만 단편화로 인해 연속된 14.38 GiB 블록 확보 불가.

**수정**: `run.sh`에 CUDA 메모리 확장 세그먼트 설정 추가.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

OOM은 해소되지만 근본 원인(메모리 압박)은 §6.12에서 별도 처리.

---

### 6.12 word-aug: 300× 속도 저하 (혼합 길이 패딩 폭발)

**증상**: word-aug 없이 ~1 s/it이던 Step 속도가 word-aug ON 시 ~300 s/it으로 저하.  
GPU 4 utilization 26%, 나머지 7개 GPU는 100% (NCCL spin-wait).

**원인**: `OmniCollator`가 bin 내 모든 오디오 클립을 `max(clip_length)`로 패딩한다.

word-aug processor는 원본 발화(full utterance)와 단어 단위 서브클립(word clip)을 **같은 패킹 풀**에 섞어 넣는다.
그 결과 하나의 packed bin에 긴 발화와 짧은 단어 클립이 공존한다.

```
bin 예시 (cutoff_len=2048 LLM tokens):
  발화 1개:   10s  = 160,000 samples (@ 16kHz)
  단어 클립 100개:  0.5s =   8,000 samples 각

collator → audio_features: (101, 1, 160,000)
실제 유효 데이터: 1×160,000 + 100×8,000 = 960,000 samples
패딩 포함 데이터: 101×160,000 = 16,160,000 samples
효율: 6%  (94%가 패딩 0)
```

이 텐서가 fb_dacvae 인코더로 전달되면 44kHz 리샘플 후 `(101, 1, 440,000)` = 178 MB.  
인코더 forward + backward 비용이 ~20× 증가 → GPU 4 메모리 소진 → 300× 속도 저하.

**근본 수정 방향**: `_get_audio_embeds_batched` 대신 `_get_audio_embeds_sequential` 사용.  
클립 하나씩 실제 길이로 인코더에 통과시키므로 교차 패딩 낭비 없음.

현재 라우팅 (`_get_audio_embeds`):
```python
def _get_audio_embeds(self, audio, audio_lengths=None):
    if self._cfg.get("use_fsdp", False):   # FSDP만 sequential
        return self._get_audio_embeds_sequential(audio, audio_lengths)
    return self._get_audio_embeds_batched(audio, audio_lengths)  # word-aug 시 비효율
```

**수정 방향**: `use_fsdp` 조건 제거하거나 `N_audio > 임계값` 시 sequential로 전환.  
Stage 1 DDP에서도 sequential을 쓸 경우 약간의 루프 오버헤드가 있지만,  
word-aug 없이도 클립 10~20개 × ~10ms/클립 = 100~200ms로 전체 step의 일부에 불과.

**상태**: 미수정 (베이스라인 확인 후 적용 예정).

---

### 6.13 word-aug (precomputed): `_project_precomputed` pad-stack OOM

**발생 시점**: §20 offline word-aug pre-pack 로 생성한 mixed shard 로 학습 중 Step 38 (epoch 0.35) 에서 rank2 OOM.

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 17.47 GiB.
GPU 2 has a total capacity of 79.33 GiB of which 3.74 GiB is free.
  File "train_pipeline_override.py", line 484, in forward
    audio_embeds = self._project_precomputed(...)
  File "train_pipeline_override.py", line 390, in _project_precomputed
    embeds = self.proj_norm(proj.transpose(1, 2))   # (N, T_proj_max, llm_dim)
```

steady-state 50~52 GB / GPU 로 돌던 rank 가 특정 bin 에서 LN 한방 할당 17 GB 요청 → OOM.

**원인**: §6.12 와 같은 구조적 문제가 **precomputed 경로에도** 존재. raw audio 경로의 `_get_audio_embeds_batched` 대응으로, precomputed 경로의 `_project_precomputed` 가 bin 안의 모든 clip 을 `(N, T_enc_max, out_dim)` 로 zero-pad 해서 한 번에 projector+LN 에 먹임.

word-aug 이전에는 bin 안 clip 이 whole utterance 만 있어서 길이가 균질 → pad 낭비 미미. word-aug 이후에는:
- bin 당 clip 수 `N` 이 100+ 로 증가 (word clip 이 짧아 dense packing)
- whole utterance 1개만 섞여도 `T_enc_max` 가 그 clip 기준으로 잡힘
- 짧은 word clip 수십~수백 개가 동일 길이로 pad → `(N, T_enc_max, C)` 텐서 폭발
- `(N, T_proj_max, llm_dim)` LayerNorm 입력 = 수십 GB

**수정** ([train_pipeline_override.py:379-403](../train_pipeline_override.py#L379-L403)): §6.12 해법과 동일하게 clip 단위 loop 전환 — pad-stack 자체를 제거.

```python
def _project_precomputed(self, enc_feats, enc_feat_lengths):
    proj_dtype = self.projector[0].weight.dtype
    llm_dtype  = self.llm.get_input_embeddings().weight.dtype
    clips = []
    for i in range(enc_feats.shape[0]):
        T_enc_valid = int(enc_feat_lengths[i].item())
        if T_enc_valid == 0:
            continue
        feats_i = enc_feats[i:i+1, :T_enc_valid, :].to(proj_dtype)
        proj_i  = self.projector(feats_i.transpose(1, 2))
        emb_i   = self.proj_norm(proj_i.transpose(1, 2)).squeeze(0)
        clips.append(emb_i.to(llm_dtype))
    return torch.cat(clips, dim=0).unsqueeze(0)
```

핵심:
- clip 하나씩 **유효 길이로만** projector+LN 통과 → peak 메모리 = 가장 긴 clip 1개 기준
- 출력 shape `(1, total_T_proj, llm_dim)` 불변 → LLM forward 쪽 변경 0
- packed bin 구조/schema 불변 → pre-packed Arrow 재생성 불필요

**속도 영향 (예상)**: projector 는 Conv1d(k=5)×3 + LN 으로 연산량이 LLM forward 의 <1%. Python loop + kernel launch overhead 도 bin 당 ~5 ms 수준. 34 s/step 대비 <1% slowdown.

**차이 — §6.11 / §6.12 / §6.13**:
- §6.11: backward 단편화 회피 (env var)
- §6.12: raw audio `_get_audio_embeds_batched` 의 N_audio 폭증 시 encoder forward 비용 (sequential 전환 제안만, 미적용)
- §6.13: **precomputed** `_project_precomputed` 의 mixed-length pad-stack → 본 수정으로 적용

**상태**: 수정 완료. 재시작 후 검증 중.

### 6.14 run.sh 에 `--precomputed-dir` 누락 → raw audio fallback

**증상**: §6.13 OOM 수정 후 재시작 시:
```
ValueError: The train_dataset does not implement __len__, max_steps has to be specified.
```

**원인**: `run.sh` 의 `accelerate launch` 커맨드가 `--precomputed-dir` 을 누락. `args.precomputed_dir=None` → main() 이 raw audio 경로로 분기 → `build_multi_dataset_streaming_pipeline` 이 `IterableDataset` 반환 → `__len__` 없음 → §19 의 `num_train_epochs` 자동 step 계산 실패.

**수정**: [run.sh](../run.sh) 의 accelerate launch 블록에 하드코딩:
```bash
accelerate launch \
    --num_processes "$GPUS" \
    train_pipeline_override.py \
    --encoder "$ENCODER" \
    --datasets ls100,ls360,ls500,mls,gs,vp \
    --precomputed-dir /mnt/ddn/users/jos/precomputed \
    --liger --fsdp \
    "${EXTRA_ARGS[@]}"
```

우리 작업 흐름에선 precomputed 모드 외엔 거의 안 쓰므로 기본 경로로 하드코딩. raw audio 모드 필요 시 EXTRA_ARGS 의 `--precomputed-dir ""` 로 오버라이드 가능 (argparse last-wins).

---

### 6.15 `evaluate_val_loss` raw waveform을 precomputed collator에 전달

**발생 시점**: §6.13 fix 후 재시작, split 1/8 의 step 100 에서 `WerCallback.on_step_end` 첫 trigger.

```
ValueError: cannot reshape array of size 24960 into shape (48, 128)
  File "train_pipeline_override.py", line 1049, in __call__
    arr = np.asarray(flat_feat, dtype=np.float32).reshape(t_enc, out_dim)
  File "train_pipeline_override.py", line 1745, in evaluate_val_loss
    batch = collator(padded)
```

**원인**: [`evaluate_val_loss`](../train_pipeline_override.py#L1684) 은 `StaticEvalDataset` 의 `(waveform_1d, text)` 쌍을 받아 sample 을 구성할 때
```python
samples.append({
    ...
    "audio_features": [waveform],     # raw 1-D float tensor
    "audio_lengths":  [t_audio],      # waveform.shape[0] // hop
})
```
로 raw waveform 을 그대로 `audio_features` 에 넣는다.

그러나 precomputed 모드에서 `collator = OmniCollator(enc_out_dim=model.encoder.out_dim)` 이라 [`__call__`](../train_pipeline_override.py#L1041) 이 `self.enc_out_dim > 0` 경로로 진입 → `flat_feat` 을 `(t_enc, out_dim)` 로 reshape 시도 → raw waveform 텐서 사이즈와 불일치.

**왜 지금까지 숨어있었나**: 이전 run 들은 `eval_steps=500`. split 당 max 110 step 이라 `state.global_step` 이 500 에 도달하기 전에 run_stage1 이 끝나 다음 split 로 넘어감. 즉 callback 의 `on_step_end` 에서 `state.global_step % eval_every != 0` 조건이 항상 True → eval 경로 자체가 트리거된 적 없음. `eval_steps` 를 100 으로 내리면서 (§20) 첫 노출.

**수정** ([train_pipeline_override.py:1790](../train_pipeline_override.py#L1790)):
```python
# evaluate_val_loss 는 raw waveform 기반 — precomputed collator 와 호환 안 됨.
# precomputed 모드에선 건너뛰고 WER 만 사용.
if self.cfg.get("precomputed_dir"):
    val_loss = float("nan")
else:
    val_loss = evaluate_val_loss(...)
```

`evaluate_wer` 는 `_greedy_batch` 를 통해 내부에서 encoder 를 직접 돌리므로 collator schema 불일치 문제 없음. val_loss 는 로그용 보조 지표였으므로 NaN 으로 두어도 학습 흐름에 영향 없음.

**근본 해결 (추후)**: `evaluate_val_loss` 내부에서 raw waveform 을 encoder 로 돌려 pre-encoded feature 로 변환한 뒤 collator 에 전달하는 방식. 지금은 precomputed 모드 우선이라 건너뜀.

**갱신 (2026-04-13)**: §24 (`docs/dataloader_trials.md`) 에서 `evaluate_val_loss_precomputed` 신규 함수 추가. encoder 를 eval 시점에 직접 호출하여 precomputed 모드에서도 실제 val_loss 계산 가능. `nan` fallback 제거.

---

### 6.16 §6.4 `_save` override regression + save_strategy/save_safetensors 우회

**배경**: §6.4 에서 `StreamingShardedTrainer._save` 를 override 해 `data_ptr` 중복 탐지 + clone 으로 tied embedding safetensors 저장 크래시를 우회했었음. 그러나 2026-04-13 시점 `train_pipeline_override.py` 에서는 이 override 가 **사라져있음** — 현재 `StreamingShardedTrainer` 는 `get_train_dataloader` 만 override. regression 또는 리팩터 과정에서 누락됨.

**재현**: Stage 1 bs=12 / save_steps=100 run, step 100 에서 즉시 발생.
```
RuntimeError: Some tensors share memory, this will lead to duplicate memory on disk:
  [{'llm.lm_head.weight', 'llm.model.embed_tokens.weight'}]
  File "trainer.py", line 3823, in _save
    safetensors.torch.save_file(...)
  File "trainer.py", line 2088, in _maybe_log_save_evaluate
    self._save_checkpoint(model, trial)
```

**원인**: Qwen3.5 의 tied embedding 이 `lm_head.weight` 와 `embed_tokens.weight` 를 같은 storage 로 공유. safetensors 가 shared storage 를 거부.

**수정 (근본 fix 대신 우회)**:

- **Stage 1** ([train_pipeline_override.py:1956](../train_pipeline_override.py#L1956)): `TrainingArguments(save_strategy="no")`. Stage 1 은 projector 만 학습이라 full-model snapshot 이 불필요. WerCallback 이 eval 시점에 projector state 만 torch.save 로 저장.

- **Stage 2** ([train_pipeline_override.py:2152-2154](../train_pipeline_override.py#L2152)): `TrainingArguments(save_safetensors=False)` 추가. Stage 2 는 LoRA adapter + projector 를 저장해야 함. safetensors 대신 torch pickle (.bin) 로 저장하면 shared storage 문제 없음. `.bin` 은 pickle 기반이라 object graph 의 shared reference 를 그대로 보존 가능.

**§6.4 복원은 하지 않은 이유**: 현재 `save_strategy="no"` / `save_safetensors=False` 우회가 목적 (WER 기반 best projector 저장 + Stage 2 full checkpoint) 에 충분. `_save` override 복원은 더 깔끔하지만 `TrainingArguments.save_safetensors` 인자가 transformers 5.5+ 에서 제거된 건지 확인 필요 (§6.5 참조). 현재 환경에서 `save_safetensors=False` 가 동작함을 확인했으므로 그쪽으로 고정.

**관련 문서**: [dataloader_trials.md §22](dataloader_trials.md#22-hf-trainer-save_strategy--no-stage-1-save_safetensorsfalse-stage-2)

---

### 6.17 NCCL timeout: rank 간 bin 개수 불균형 (split 8 deadlock)

**발생 시점**: Stage 1 bs=12 / num_data_splits=8 run, split 8 (마지막 split) step 29 → 30 transition 에서 collective deadlock.

**로그**:
```
[rank 7]:[E413 16:51:14] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=3952, OpType=BROADCAST, NumelIn=64, NumelOut=64,
           Timeout(ms)=600000) ran for 600066 ms before timing out.
[rank 4/2/5]:[E413 16:51:42] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=3953, OpType=ALLREDUCE, NumelIn=4198400, NumelOut=4198400,
           Timeout(ms)=600000) ran for 600002~600079 ms before timing out.
```

**증거** — split 8 로드 시 rank 별 bin 수:
```
[rank 6] 1374 bins   ← 최소
[rank 7] 1395 bins
[rank 5] 1697 bins
[rank 3] 1661 bins
[rank 1] 1796 bins
[rank 2] 1802 bins
[rank 0] 1822 bins
[rank 4] 1851 bins   ← 최대 (477 bin 차)
```

**원인 체인**:
1. `pack_arrow.py pack_rank_mixed` 는 **byte 기준** shard rotation 사용 (`SHARD_MAX_BYTES = 20 GB`). rank 별 bin 평균 size 가 조금씩 달라서 20 GB 를 채우는 데 필요한 bin 수 편차가 생김.
2. 처음 7 개 shard 는 정확히 20 GB 근처에서 rotate → 비교적 균일한 bin 수. 그러나 **마지막 shard 는 residual** — rank 마다 서로 다른 크기.
3. 학습 시 `num_data_splits=8` + `shard-level split` 설정이므로 각 split 은 각 rank 의 shard idx 하나씩 로드. split 8 이 하필 최악의 shard (마지막 residual) 를 뽑음.
4. `StreamingShardedTrainer` 가 DistributedSampler 우회 설계라 **각 rank 의 DataLoader 는 자기 `len(dataset)` 기준으로 독립적으로 step 수 결정**.
5. rank 6 (1374 bin) 은 `ceil(1374/12) = 115` iteration 로 종료. rank 4 (1851 bin) 은 155 iteration 목표.
6. rank 6/7 이 먼저 train loop 빠져나가 collective 참여 안 함 → rank 4 가 다음 allreduce 에서 영원히 기다림 → 600 초 watchdog timeout → crash.

**수정 (2단계 방어)**:

1. **Runtime safety** ([train_pipeline_override.py:1408-1429](../train_pipeline_override.py#L1408)): `build_precomputed_pipeline` mixed 분기에서 shard 를 로드한 직후 전 rank 의 bin 수 최소값을 `torch.distributed.all_reduce(MIN)` 으로 구해서 truncate.
   ```python
   local_n = torch.tensor([_table.num_rows], device=accelerator.device)
   dist.all_reduce(local_n, op=dist.ReduceOp.MIN)
   min_n = int(local_n.item())
   if _table.num_rows > min_n:
       _table = _table.slice(0, min_n)
   ```
   → 모든 rank 가 동일한 `min_n` 으로 학습, collective 불일치 원천 차단.

2. **Offline rebalance** ([precompute/pack_arrow.py](../precompute/pack_arrow.py)): 새 함수 `rebalance_mixed_shards(base_dir, cutoff_len, num_ranks)` — 기존 packed shard 에 소급 rebalance. 모든 rank 의 총 bin 수 스캔 → 최소값 산출 → 각 rank 의 shard 뒤쪽부터 bin 삭제 (전체 shard 삭제 or 마지막 shard 자르기).
   - CLI: `python precompute/pack_arrow.py --encoder X --rebalance` (rebalance 모드에선 `--rank` optional)
   - UX: `run_pack.sh --mixed` 가 packing 완료 직후 자동 rebalance 호출 → 사용자 개입 불필요

**관련 문서**: [dataloader_trials.md §26](dataloader_trials.md#26-nccl-timeout-과-rank-간-bin-개수-불균형)

---

### 6.18 wandb 내장 WandbCallback split 경계 finish → 로그 단절

**증상**: num_data_splits=8 학습에서 split 1 은 `Qwen3.5-2b-ASR-fb_dacvae` project 의 올바른 run 에 기록. split 2 부터 **자동으로 `huggingface` 기본 project 에 "pleasant-dew-1" 같은 랜덤 이름** 으로 떨어짐.

**원인**: HF Trainer 내장 `WandbCallback.on_train_end` 가 split 1 끝에서 `wandb.finish()` 호출 → `wandb.run = None`. split 2 의 새 Trainer 가 report_to="wandb" 로 재진입 → WandbCallback 이 `wandb.init()` 새로 호출. 이때 `WANDB_PROJECT` 환경변수가 없어서 HF Trainer 가 기본값 `"huggingface"` 를 사용 + 랜덤 run name.

더 심각한 문제: `WANDB_PROJECT` 를 고정해도 **wandb 내부 step counter 는 monotonic 증가만 허용**. split 2 가 step 0 로 재시작하면 wandb 가 이전 run 의 step 100+ 보다 낮은 값을 거부 → 차트에 split 2+ 가 안 그려짐.

**수정** ([train_pipeline_override.py](../train_pipeline_override.py)):

1. **새 callback `CumulativeWandbCallback`** — `on_log` 후크에서 `wandb.log(payload)` 호출 (step 인자 **생략** → wandb 내부 auto-increment 로 단조 증가 보장)
   ```python
   class CumulativeWandbCallback(TrainerCallback):
       def on_log(self, args, state, control, logs=None, **kwargs):
           if logs is None or wandb.run is None:
               return
           payload = {("train/" + k if not k.startswith(("train/","eval/","val/")) else k): v
                      for k, v in logs.items() if isinstance(v, (int, float))}
           if payload:
               wandb.log(payload)
   ```
2. **Stage 1 / Stage 2 TrainingArguments `report_to="none"`** — HF Trainer 내장 WandbCallback 완전 disable
3. **`run_stage1` / `run_stage2` 끝의 `wandb.finish()` 제거** — wandb run 이 split 경계를 넘어 살아있어야 함
4. **main() 끝에서만 `wandb.finish()` 호출** — 모든 split 루프 완료 후 단일 종료
5. **Env 안전망** — main() 시작 시 `os.environ["WANDB_PROJECT"] = cfg["project_name"]`, `WANDB_RUN_ID`, `WANDB_RESUME="allow"` 셋팅

**검증**: split 1 이 `fb_dacvae_2b_S1_0413_0739` 로 시작 → split 2 에서 `Resuming run fb_dacvae_2b_S1_0413_0739` 로그 확인. 하나의 wandb run 에 연속 기록.

**관련 문서**: [dataloader_trials.md §23](dataloader_trials.md#23-cumulative-wandb-logging-num_data_splits1-에서-split-경계-wandb-run-연속성)

---

### 6.19 Stage 2 `_greedy_batch` dtype mismatch (fp32 audio_embeds → bf16 Qwen3.5 linear)

**증상**: Stage 2 첫 `eval_steps=35` 트리거 → `WerCallback.on_step_end` → `evaluate_wer` → `_greedy_batch` → `model.llm.generate` 호출 중 rank 0 crash.

```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
  at transformers/models/qwen3_5/modeling_qwen3_5.py:442 — self.in_proj_qkv(hidden_states)
```

`linear_attn.in_proj_qkv.weight` 는 bf16 이지만 입력 `hidden_states` 가 fp32 → F.linear 에서 dtype 불일치.

**원인**: `_greedy_batch` 는 `_get_audio_embeds` 가 반환한 `audio_embeds` 를 그대로 `torch.cat([audio_embeds, corr_embeds])` → `model.llm.generate(inputs_embeds=...)` 로 넘김. `_get_audio_embeds_sequential` 내부에서 `proj_dtype = self.projector[0].weight.dtype` 로 출력 dtype 을 결정하는데, **FSDP `FlatParameter` 하에서 이 attribute 가 fp32 로 보이는 경로** 가 존재. 이유는 명확히 추적 안 됨 — FSDP 내부 master-weight 처리 / LoRA wrapping / `use_orig_params=True` 상호작용 추정.

결과적으로 audio_embeds 가 fp32 로 빠져나와 LLM forward 전체가 fp32 activation 으로 진입, bf16 weight 에 F.linear → mismatch.

**수정** ([train_pipeline_override.py:1662-1678](../train_pipeline_override.py#L1662-L1678)):

```python
audio_embeds = model._get_audio_embeds(audio_pad, audio_lengths)  # (B, T_proj, D)

tokenizer = model.tokenizer
embed     = model.llm.get_input_embeddings()

# Qwen3.5 linear_attn.in_proj_qkv 가 bf16 weight → 입력 activation 도 bf16 이어야.
# FSDP FlatParameter 하에서 proj_dtype 이 fp32 로 보이는 케이스 방어 (§6.19).
audio_embeds = audio_embeds.to(torch.bfloat16)
corr_embeds  = embed(corr_tensor).expand(B, -1, -1).to(torch.bfloat16)
inputs_embeds = torch.cat([audio_embeds, corr_embeds], dim=1).to(torch.bfloat16)
```

명시적 bf16 cast 3 곳 추가. `embed.weight.dtype` 읽는 건 같은 FSDP 경로 이슈로 안전하지 않아 `torch.bfloat16` 리터럴 사용.

**교훈**:
- FSDP 내부에서 `module.weight.dtype` 조회는 **신뢰하지 말 것**. LoRA wrapping + `use_orig_params` + FSDP 조합에서 기대와 다른 dtype 반환 가능
- Stage 2 eval 같은 "rank 0 부분 forward" 경로는 **항상 LLM 입력 dtype 을 명시 cast** 해서 방어

**관련 문서**: [arXiv §28](../arXiv/docs/stage2_fsdp_nccl_resolution_apr14.md#section-28) — Stage 2 `_greedy_batch` dtype mismatch

---

### 6.20 Stage 2 `WerCallback` FSDP in-loop eval NCCL deadlock (완전 비활성화로 우회)

**증상**: §6.19 수정 후 재시작 → 다시 `eval_steps=35` trigger 에서 **NCCL timeout 600s** 두 번 연속 발생.

**첫 번째 timeout (v5)** — `FSDP.summon_full_params(model, writeback=False, rank0_only=True)` 사용:
```
[rank0]: stuck at _ALLGATHER_BASE (SeqNum=7300)
[rank1-7]: stuck at ALLREDUCE(1,1) (SeqNum=7300)
```
`rank0_only=True` 는 collective gather 후 non-zero rank 는 param 을 즉시 discard. rank 0 가 context 안에서 `generate()` 를 호출 → FSDP hook 이 추가 allgather 시도 → 다른 rank 는 이미 param 버린 상태라 재진입 불가 → **데드락**.

**두 번째 timeout (v6)** — `rank0_only=False, recurse=True` 로 변경:
```
[rank0]: stuck at _ALLGATHER_BASE (summon_full_params 자체의 gather)
[rank1-7]: stuck at ALLREDUCE(1,1) (barrier 급)
```
이번엔 rank 0 만 `summon_full_params` 에 진입. 나머지 7 rank 는 그 앞의 tiny ALLREDUCE (아마 `wait_for_everyone()` 의 barrier) 에 갇힘. 즉 **HF Trainer 가 callback 을 rank 별로 다르게 부르거나, `model` kwarg 경로가 분기**. 정확한 원인 추적 실패.

배경: FSDP + LoRA + 커스텀 in-callback eval 조합이 HF Trainer 내부 수명주기와 충돌. `rank0_only` 스위치로는 해결 불가.

**최종 수정** ([train_pipeline_override.py:1905-1915](../train_pipeline_override.py#L1905-L1915)):

`WerCallback.on_step_end` 진입 시 FSDP 활성 여부 체크 → 즉시 return. Stage 2 in-loop WER/val_loss eval **완전 비활성화**.

```python
def on_step_end(self, args, state, control, model=None, **kwargs):
    if state.global_step > 0 and state.global_step % self.eval_every != 0:
        return

    # Stage 2 FSDP + LoRA + 커스텀 in-loop eval 조합에서 NCCL deadlock 반복 (§6.20).
    # rank0 만 summon_full_params 에 진입하고 나머지는 barrier 에 남는 경로 문제.
    # WER/val_loss 는 HF Trainer 의 save_steps 체크포인트에서 offline 계산.
    if self.cfg.get("use_fsdp", False):
        return
    ...
```

**대체 경로**:
- HF Trainer `save_strategy="steps" save_steps=35` 로 체크포인트는 정상 저장 (tied embedding 은 `StreamingShardedTrainer._save` override 가 처리 — §6.4, §6.16)
- WER/val_loss 는 학습 종료 후 저장된 체크포인트에서 **offline 계산** 으로 전환 (별도 스크립트 TBD)

**영향**:
- Stage 2 training loop 복구, 검증 지표는 후처리로 이동
- Stage 1 (DDP) 은 영향 없음 — 기존 WerCallback 경로 그대로 사용

**교훈**:
- FSDP in-loop eval 은 `summon_full_params` + rank-0-only 패턴으로 구현 금지. 전체 rank 참여 forward 또는 완전 분리된 eval worker 가 필요
- "FSDP + PEFT + 커스텀 TrainerCallback" 조합은 HF Trainer 내부 가정 (DDP 기반 rank-0 특권 작업) 과 충돌

**관련 문서**: [arXiv §29](../arXiv/docs/stage2_fsdp_nccl_resolution_apr14.md#section-29) — Stage 2 FSDP in-loop eval NCCL deadlock → WerCallback 완전 비활성화
