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
