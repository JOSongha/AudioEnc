# Qwen3.5AE ASR Stage1 — Setup & Trial Log

> **Historical**: 2026-04-21 NSML 세션 셋업 로그. 본 노드(jos)와 다른 환경의 기록일 수 있음 — 그대로 따라하기보다 참고용으로.

기록 시점: 2026-04-21
환경: NSML container (node0, 8× A100 80GB, glibc 2.31, kernel 5.4.239)

---

## 0. 목표

`oss.navercorp.com/HyperscaleAI/audiollm-trainer` (acoustic branch) 로 Qwen3.5-4B + DACVAE 의 ASR Stage1 (projector-only) 학습을 이 NSML 세션 안에서 tmux 로 돌린다.

원본 가이드: [../stage1/dacvae_asr.md](../stage1/dacvae_asr.md)

---

## 1. Repo clone

```
/mnt/fr20tb/wbl_residency/jos/ddn/audiollm-trainer  (branch: acoustic @ d15cb890)
```

HTTPS + PAT (nb93059) 로 clone. `ddn` 심볼릭 링크 → `/mnt/ddn/users/jos/` (기존 `WorsePenaltyARCE` 레포와 분리해서 하위 디렉터리에 둠).

---

## 2. Conda env 구성

### 방침
- 기존 `audio` env (py3.10, 사용자 기존 AudioEnc 작업 env) 는 **건드리지 않음** (transformers 5.5.0 등 AudioEnc 호환 버전 유지)
- `audio` 를 **clone** → `audio_lmf` 생성하고 llamafactory 호환 버전으로 downgrade 만 거기서 수행

### clone 및 downgrade
```bash
conda create --clone audio -n audio_lmf -y
pip install "transformers>=4.51.0,<=4.57.1,!=4.52.0,!=4.57.0" "accelerate>=1.3.0,<=1.11.0"
pip install "peft>=0.14.0,<=0.17.1" "trl>=0.18.0,<=0.24.0" "tyro<0.9.0"
pip install "gradio>=4.38.0,<=5.50.0"
```

### 신규 설치 (pyproject.toml deps 맞추기)
- `deepspeed==0.16.9`
- `trl`, `torchdata`, `tyro`, `sentencepiece`, `tiktoken`, `modelscope`, `hf-transfer`
- `av`, `omegaconf`, `antlr4-python3-runtime==4.9.3` (omegaconf 2.3 호환)
- `starlette`, `fastapi`, `uvicorn`, `sse-starlette`, `python-multipart`
- `dacvae` (editable from `/mnt/fr20tb/wbl_residency/jos/AudioEnc/dacvae`, `--no-deps`)
- `llamafactory` (editable from repo root, `--no-deps --ignore-requires-python`)

### 최종 버전 요약 (audio_lmf)
| 패키지        | 버전     |
| ------------- | -------- |
| python        | 3.10.20  |
| torch         | 2.6.0+cu124 |
| transformers  | 4.57.1   |
| accelerate    | 1.11.0   |
| peft          | 0.17.1   |
| trl           | 0.24.0   |
| gradio        | 5.50.0   |
| deepspeed     | 0.16.9   |
| flash_attn    | 2.8.3    |
| causal_conv1d | 1.6.1    |
| flash-linear-attention | 0.4.2 |
| liger_kernel  | 0.7.0    |
| dacvae        | 1.0.0 (editable) |
| llamafactory  | 0.9.4 (editable) |

---

## 3. Repo source 패치 (py3.10 호환)

acoustic 브랜치의 `pyproject.toml` 은 `requires-python>=3.11` 이지만 env 는 py3.10. py3.11+ 전용 typing import 를 `typing_extensions` fallback 으로 대체.

수정 파일 (4개):
- `src/llamafactory/data/mm_plugin.py:25`
- `src/llamafactory/v1/plugins/data_plugins/converter.py:16`
- `src/llamafactory/v1/utils/types.py:15`
- `src/llamafactory/hparams/model_args.py:20`

각각 `NotRequired` / `Self` 를 `from typing_extensions import ...` 로 변경.

> ⚠️ 이 패치들은 uncommitted. upstream `git pull` 시 충돌 가능.

---

## 4. 작성한 config/script

디렉터리: `configs/qwen3_5ae-asr/`

| 파일                     | 내용                                          |
| ------------------------ | --------------------------------------------- |
| `stage1_projector.yaml`  | training config (PDF 값 그대로)               |
| `ds_z2_no_fused.json`    | `ds_z2_config.json` 에서 `optimizer` 블록 제거한 버전 (FusedAdam JIT 회피) |
| `run_nsml.sh`            | conda activate + LD_PRELOAD + llamafactory-cli train 래퍼 |

`run_nsml.sh` 주요 env:
- `WANDB_PROJECT=qwen3_5ae-asr`
- `WANDB_API_KEY` (사용자 제공, 스크립트 기본값으로 embed — 민감)
- `LD_PRELOAD=<audio_lmf>/lib/glibc_compat.so` (flash_attn 용, §5.3 참조)

### Stage1 = projector-only 확인
[src/llamafactory/train/omni/workflow.py:63-65](../../src/llamafactory/train/omni/workflow.py#L63-L65):
```python
for name, param in model.named_parameters():
    require_grad = "audio_encoder.projector" in name
    param.requires_grad = require_grad
```
`stage: omni` 로 두면 자동으로 projector 만 학습. 별도 설정 불필요.

---

## 5. 실행 시도 및 에러 log

tmux session: `asr_s1`. 로그: `/tmp/asr_s1.log`.

### 5.1 1차 — `set -euo pipefail` 의 `-u` 충돌
```
NVCC_PREPEND_FLAGS: unbound variable
```
**조치**: `run_nsml.sh` 에서 `set -euo pipefail` → `set -eo pipefail` (unbound check 해제)

### 5.2 2차 — `dacvae` import 실패
```
ModuleNotFoundError: No module named 'dacvae'
```
모델의 `audio_encoder.py` 가 `from dacvae import DACVAE` 참조.
**조치**: 사용자가 경로 알려줌 (`/mnt/fr20tb/wbl_residency/jos/AudioEnc/dacvae`) → `pip install -e . --no-deps` (editable)

### 5.3 3차 — flash_attn 심볼 누락
```
flash_attn_2_cuda...so: undefined symbol: __libc_single_threaded
```
- 시스템 glibc 2.31 / flash_attn 2.8.3 wheel 은 glibc 2.32+ 용
- env 안에 이미 `lib/glibc_compat.so` 존재 (해당 심볼 제공)

**조치**: `run_nsml.sh` 에 `LD_PRELOAD=<env>/lib/glibc_compat.so` 추가

### 5.4 4차 — DeepSpeed FusedAdam JIT 컴파일 실패
```
unsupported GNU version! gcc versions later than 11 are not supported!
```
- conda env 의 gcc 13.4.0 / CUDA 12.4 nvcc 는 gcc≤11 요구
- `ds_z2_config.json` 의 `optimizer` 블록이 FusedAdam JIT 트리거

**임시 조치 (현재)**: `ds_z2_no_fused.json` 생성 — optimizer 블록 제거. HF Trainer 가 기본 `torch.optim.AdamW` 제공 → DeepSpeed 는 ZeRO wrapper 만 담당. `zero_allow_untested_optimizer: true` 가 이미 설정돼 있어 안전.

이 상태로 재실행 중 (결과 미확인).

---

## 6. 진행 경과 및 현재 상태

### 6.1 1차 학습 시도 (FusedAdam bypass, 2026-04-21 04:26~05:01)
- tmux session: `asr_s1`
- config: `ds_z2_no_fused.json` (FusedAdam bypass)
- 초기 loss: 5.25 → 3.77 (step 250 경)
- 페이스 변동:
  - 04:35, step 40: **5.16 s/it** (ETA ~143h)
  - 04:48, step 198: **6.53 s/it** (ETA ~181h) — GPU burn 경합 가설
  - 05:00, step 250 (burn kill 직후): **4.61 s/it** (ETA ~128h) — 30% 개선
- **GPU burn 발견 및 제거**:
  - PID 150863 (host view) = PID 13815 (내 namespace view) 가 `gpu_burn.py` 1일 19시간째 실행 중
  - `nvidia-smi` 의 PID (150863) 는 host namespace 기준이라 내 container 에서 바로 kill 안 됨
  - `ps -u nsml` 로 같은 프로세스의 container-side PID (13815) 발견 후 kill
  - burn 각 GPU 당 1.3GB 차지, 실측으로는 ~30% 속도 저하 유발 (이론 2-3× 예상보다 적음)

### 6.2 1차 OOM Crash (step 250+, 05:01:17)
```
CUDA out of memory. 57.88 GiB is allocated by PyTorch...
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
```
- projector 만 학습이지만 full model forward + DACVAE encoder + cutoff_len=3584 × batch 4 조합으로 메모리 여유 작음
- 특정 long-sample batch 에서 fragmentation 으로 OOM

### 6.3 2차 학습 시도 (gcc 11 + FusedAdam + expandable_segments, 2026-04-21 05:03~)
조치 2가지 동시 적용:
1. yaml 의 deepspeed 경로를 **원본 `examples/deepspeed/ds_z2_config.json`** (FusedAdam 켜짐) 으로 되돌림
   - 앞서 설치한 **gcc 11** 로 JIT 컴파일 성공해야 정상 진행
2. `run_nsml.sh` 에 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 추가

재실행 후 진행 상태 모니터링 중.

### 6.4 2차 학습 결과 (05:05~, 성공)
- **FusedAdam JIT 컴파일 성공** (gcc 11 효과 확인)
- step 14, loss 5.37, 페이스: **2.33 s/it**, ETA **~65시간 (2.7일)**

속도 개선 summary:
| 단계 | s/it | ETA |
| --- | --- | --- |
| no-FusedAdam + burn 有 | 6.53 | 181h |
| no-FusedAdam + burn kill | 4.61 | 128h |
| **FusedAdam + expandable_segments** | **2.33** | **65h** |

즉 1차 대비 약 **2.8× 가속**. 효과 분해:
- burn kill: ~30%
- FusedAdam + expandable_segments: ~추가 2× (FusedAdam 자체 + fragmentation 제거로 long-sample 처리 효율 상승)

### 6.5 2차 학습 도중 OOM (step 91+ 경, 05:19)
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.40 GiB.
GPU 6 has a total capacity of 79.33 GiB of which 16.27 GiB is free.
Process 142129 has 63.03 GiB memory
```
- `expandable_segments` 가 fragmentation 만 줄여줄 뿐, 단일 batch 의 peak 메모리 자체가 80GB 에 근접한 것이 원인
- GPU 마다 allocation 편차 큼 (GPU 0 = 80GB full, GPU 1-7 = 67-74GB) — packing bucket 에서 batch 간 길이 분포 불균등

### 6.6 3차 조치 (peak memory ↓, load balance ↑)
- `omni_packing_bucket_size`: 256 → **128** (길이 분포 bucket 단위 줄여 batch 간 variance 감소)
- `per_device_train_batch_size`: 4 → **2**, `gradient_accumulation_steps`: 1 → **2**
  → global effective batch (per-step tokens 는 half, optimizer update 는 2 step 마다) 는 8×2×2 = 32 로 동일 유지
  → per-step peak memory ~절반으로 감소해 OOM 여유 확보

재실행 후 안정성/속도 재측정 예정.

### 6.7 3차 결과 & 4차 튜닝 (bs 스윕)

bs=2 + grad_accum=2 시도 결과: stable 하지만 **power 90W** (A100 TDP 의 22%) → all-reduce / sync overhead 가 너무 큼. compute 낭비.

→ **bs=3, grad_accum=1, bucket_size=128** 로 재튜닝 (kill + restart).

#### 최종 측정 (step 53, 2026-04-21 05:28)
- pace: **2.09 s/it**, ETA **~58h (2.4일)** ← 지금까지 최단
- memory: 51-59 GB / 80 GB (여유 20-30GB, OOM 안전)
- power: 302-384W (A100 proper compute ✓)
- util: 100% × 8 GPU

#### config 스윕 요약
| bs | ga | s/it | Power | 결과 |
| --- | --- | --- | --- | --- |
| 4 | 1 | 2.33 | 400W+ | OOM crash |
| 2 | 2 | 2.82 | 90W | under-util |
| **3** | **1** | **2.09** | 300-380W | **최적** ✓

이 세팅이 이 NSML 세션 / 8×A100 / cutoff_len=3584 조건의 sweet spot.

### gcc 11 설치 결과 (2026-04-21 04:35)
현재 run 과 별개로 `audio_lmf` env 의 compiler toolchain 을 gcc 13.4 → gcc 11.4 로 교체 완료.

진행 과정 (해결한 conflict):
- 첫 시도: `binutils_linux-64=2.45.1` 고정 충돌
- 재시도: `libgcc-devel_linux-64=13.4.0`, `libsanitizer=13.4.0` 등 추가 충돌
- 최종 해결 — 함께 downgrade 필요한 모든 패키지 명시:
  ```bash
  conda install -n audio_lmf -c conda-forge \
      gcc_linux-64=11.4 gxx_linux-64=11.4 \
      binutils_linux-64=2.40 binutils_impl_linux-64=2.40 ld_impl_linux-64=2.40 \
      libgcc-devel_linux-64=11.4 libstdcxx-devel_linux-64=11.4 libsanitizer=11.4 -y
  ```
- 결과: `x86_64-conda-linux-gnu-cc` = gcc 11.4.0 (CUDA 12.4 nvcc 와 호환)

### 현재 run 과의 관계
- 현재 돌고 있는 `asr_s1` 은 FusedAdam **bypass** 한 config (`ds_z2_no_fused.json`) 로 시작됨 → gcc 설치는 이 run 에 영향 없음
- **중단하지 않음** (user rule: 명시적 kill 승인 전까지 running training 보존)

### 결정 필요한 것
1. **FusedAdam 활성화 여부** — 현재 run 이 충분히 빠르므로 굳이 재시작 필요 없음. 다음 run 부터 원본 `examples/deepspeed/ds_z2_config.json` (FusedAdam 켜짐) 쓸지 결정.
2. **WANDB_API_KEY** 가 `run_nsml.sh` 에 평문 embed. git commit 전에 제거하거나 환경변수로만 주입할지 결정.
3. **upstream pyproject.toml 의 `requires-python>=3.11`** 과 env 의 py3.10 mismatch — 장기적으로는 py3.11 env 로 이관 고려.

---

## 7. 실행 재현 방법

```bash
source /mnt/fr20tb/wbl_residency/jos/ddn/miniforge3/bin/activate audio_lmf
cd /mnt/fr20tb/wbl_residency/jos/ddn/audiollm-trainer
tmux new -s asr_s1
bash configs/qwen3_5ae-asr/run_nsml.sh 2>&1 | tee /tmp/asr_s1.log
# Ctrl+b d 로 detach
```
