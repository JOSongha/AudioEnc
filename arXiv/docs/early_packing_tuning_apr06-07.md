# Early Packing / Bucket / Cutoff_len Tuning (Archive, 2026-04-06 ~ 04-11)

이 파일은 `dataloader_trials.md` 에서 2026-04-14 에 분리된 archive 입니다.
주제: 초기 packing/bucket/cutoff_len tuning, n_shards/num_workers 실험, Stage 1 FSDP 검증(무효), Liger Kernel 효과 측정, flash-linear-attention 환경 구성, offline pre-packing 구현.
원문 섹션: §1 ~ §10 (dataloader_trials.md 구 "핵심 관찰" 블록 + §9/§9b/§9c/§10)

이 섹션들은 모두 2026-04-06 ~ 04-11 사이에 resolved 되었으며, 현재 config 값 (packing_cutoff_len=16384, packing_bucket_size=200, process_batch_size=32) 으로 확정되었습니다.

---

## §1. 가장 큰 속도 향상 요인: raw audio → precomputed 전환 {#section-1}

*(archived from dataloader_trials.md §1, 2026-04-08~09)*

- raw audio 모드: 인코더(fb_dacvae)가 학습 중 실시간으로 GPU에서 동작 → 100~350 s/it
- precomputed 모드: 인코더 출력을 Arrow 파일에 저장해두고 로드만 함 → ~13 s/it
- **×10~25 속도 향상**

---

## §2. n_shards 효과: 미미함 (GPU-bound 상태) {#section-2}

*(archived from dataloader_trials.md §2, 2026-04-10)*

- 1-shard vs 4-shard vs 8-shard 모두 ~12.7~13.4 s/it
- precomputed 모드에서 dataloader 전처리가 충분히 빠름 → GPU 연산이 실질적 병목

---

## §3. num_workers: HF가 `min(num_workers, dataset.num_shards)`로 cap {#section-3}

*(archived from dataloader_trials.md §3, 2026-04-10)*

```
num_workers=8, n_shards=4 → "Too many dataloader workers: 8 (max=4)" → 실제 4 workers
num_workers=8, n_shards=8 → 경고 없음 → 실제 8 workers
```

---

## §4. `max_batch_tokens`는 training에 영향 없음 {#section-4}

*(archived from dataloader_trials.md §4, 2026-04-10)*

- `train_pipeline_override.py`에서 `max_eval_tokens = cfg["max_batch_tokens"]`로만 사용됨 (WER eval 전용)
- training 배치 크기는 `packing_cutoff_len` × `per_device_train_batch_size`로 결정됨
- `max_batch_tokens=2600→8000` 변경 후에도 GPU mem/util 변화 없었음 (확인됨)

---

## §5. `packing_cutoff_len`이 GPU utilization의 핵심 레버 {#section-5}

*(archived from dataloader_trials.md §5, 2026-04-10)*

- training: `per_device_train_batch_size=1` (Stage 1), `=2` (Stage 2)
- 각 step에서 GPU가 처리하는 토큰 수 = `packing_cutoff_len × per_device_train_batch_size`
- cutoff를 키울수록 step당 compute↑, GPU util↑, step 수↓ → wall time 단축

| packing_cutoff_len | s/it | GPU util | GPU mem | total steps | wall time | 비고 |
|--------------------|------|----------|---------|-------------|-----------|------|
| 2048 | 13s | ~24% | 13GB | 23025 | ~83h | — |
| 4096 | 23s | ~46% | 20GB | 11513 | ~74h | — |
| 8192 | 38s | ~82% | 35GB | 5757 | ~61h | — |
| 16384 | ~55s | ~95% | ~64GB | 2879 | **~44h** | **최적** (step8 steady-state 기준) |

> 초반 step(1~5)은 JIT 컴파일/warmup으로 ~109s처럼 보이지만, step8+ steady-state는 ~55s.
> Flash Attention 2 compute O(T²)이지만 step 수가 절반 → 16384가 실질 wall time 더 짧음.
> **현재 최적**: ~16384 (메모리 여유 있으면 더 올릴 여지도 있음)

---

## §6. Stage 1 FSDP 검증 결과 (2026-04-10, 무효) {#section-6}

*(archived from dataloader_trials.md §6, 2026-04-10)*

`--fsdp-stage1` 플래그 활성화:

| 실험 | 측정 step | s/step (EMA) | GPU 메모리 |
|------|-----------|-------------|----------|
| 1차 테스트 (25 step) | step 12~16 | ~115s | ~66GB/GPU |
| 2차 재현 (풀런, 2879 step) | step 201 | **~119s** | ~66GB/GPU |

DDP baseline 대비:

| 지표 | DDP (baseline) | FSDP Stage 1 |
|------|---------------|-------------|
| s/step | ~55s | **~119s (2.2배 느림)** |
| GPU 메모리 | ~64GB/GPU | ~66GB/GPU (절감 없음) |

**원인 분석**:

### 이유 1: 메모리 절감 없음 — activation이 메모리를 지배

FSDP의 파라미터 샤딩 효과:
```
Qwen3.5-2B in bf16 = ~4 GB
FSDP 8-GPU 분산 → 500 MB/GPU  ← 이론상 3.5 GB/GPU 절감
```

그런데 cutoff_len=16384인 packed sequence를 처리하면 각 레이어의 attention/FFN 중간 activation이 파라미터보다 훨씬 크다. 28개 레이어 × 16384 토큰의 activation 합산이 파라미터 절감분(3.5 GB)을 압도한다.

실측 결과: 64 GB → 66 GB로 오히려 증가. **파라미터가 메모리 병목이 아니라 activation이 병목**이므로 FSDP 파라미터 샤딩은 효과가 없다.

### 이유 2: forward마다 레이어별 all-gather 통신 발생

DDP는 각 GPU가 전체 파라미터를 보유하므로 forward 중 통신이 없다. FSDP는 파라미터를 분산 저장하므로 연산 전 매번 복원해야 한다:

```
[DDP forward]
  Layer 1 → Layer 2 → ... → Layer 28
  통신: 없음

[FSDP full_shard forward]
  all-gather(Layer1) → Layer 1 연산 → free
  all-gather(Layer2) → Layer 2 연산 → free
  ...
  all-gather(Layer28) → Layer 28 연산 → free
  통신: 28회 all-gather
```

backward도 레이어별 all-gather + reduce-scatter가 반복된다.

### 이유 3 (결정타): `gradient_checkpointing` + FSDP `full_shard` 충돌

`gradient_checkpointing`은 forward 시 activation을 저장하지 않고 backward 때 재계산한다. 재계산에는 해당 레이어의 파라미터가 필요한데, FSDP는 이미 free했으므로 **재계산 시 또 all-gather가 발생**한다:

```
[FSDP + gradient_checkpointing backward]
  Layer N:
    all-gather(파라미터 복원)          ← 1번째 all-gather
    activation 재계산 → all-gather 재발생  ← 2번째 all-gather (중복)
    gradient 계산
    reduce-scatter(gradient 분산)
    free
  → 레이어당 all-gather 2회
```

HF Trainer 경고:
```
When using FSDP full shard, instead of using `gradient_checkpointing` in TrainingArguments,
please use `activation_checkpointing` in `fsdp_config`. The former introduces a redundant
AllGather operation in backward pass.
```

올바른 방법은 `TrainingArguments(gradient_checkpointing=True)` 대신 `fsdp_config` 안에 `activation_checkpointing: true`를 쓰는 것이다. Stage 1 FSDP 시도 당시 이 설정이 누락되어 all-gather가 2배로 발생했다.

> **참고**: `fsdp_config.activation_checkpointing`으로 전환하면 이유 3의 중복 all-gather는 해소되지만, 이유 1(activation이 메모리 지배)과 이유 2(레이어별 all-gather 통신)는 여전히 유효하다. 또한 `gradient_checkpointing` 자체를 끄는 것도 불가능하다 — cutoff_len=16384에서 checkpointing=False 시 GPU 메모리 79.3GB를 전부 소진하고 OOM이 발생한다 ([§12](#section-12) 참조). 따라서 Stage 1 FSDP는 설정을 개선하더라도 DDP 대비 이점이 없다.

### 왜 Stage 2는 FSDP가 유효한가

Stage 2는 LoRA 파라미터의 optimizer state(Adam m, v)가 추가된다. float32 Adam이면 LoRA 파라미터 × 3배 메모리가 필요하다. 이게 각 GPU에 full copy로 있으면 OOM에 가까워지므로, FSDP로 optimizer state까지 분산하는 것이 의미 있다.

| 항목 | Stage 1 | Stage 2 |
|------|---------|---------|
| 학습 파라미터 | Projector만 (수 MB) | LoRA + Projector |
| Optimizer state | 무시 가능 | LoRA × 3 (유의미) |
| 메모리 병목 | activation | activation + optimizer state |
| FSDP 이점 | 없음 | optimizer state 분산 |
| 결론 | DDP 최적 | FSDP 최적 |

**결론**: Stage 1은 DDP가 최적. `--fsdp-stage1` 사용하지 않음.

> 코드는 유지 (`--fsdp-stage1` 플래그 존재). Stage 2 기본값은 FSDP 유지 (`--fsdp`).

---

## §7. DDP All-Reduce 병목 (Stage 1 한정) {#section-7}

*(archived from dataloader_trials.md §7, 2026-04-10)*

**Stage 1**: LLM frozen, projector만 학습 → DDP (gradient full sync)
**Stage 2**: LoRA + projector 학습 → **FSDP** (`--fsdp` 플래그, `run.sh` 기본값)

Stage 1 DDP 관찰 패턴 (packing_cutoff_len=8192, ~28s/step):
```
[forward+backward × 4 micro-step = ~22s] → [NCCL AllReduce + optimizer = ~6s]
                GPU util ~82-100%                   GPU util ~5-32%
```
- `gradient_accumulation_steps=4` → 4 micro-step마다 all-reduce 1회
- duty cycle ≈ 22/28 ≈ **79%** (Stage 1 전용 병목)
- Stage 2는 FSDP로 gradient sharding → all-reduce 통신량 1/8로 감소

**현재 상태**: Stage 2 FSDP는 이미 `run.sh` 기본값으로 활성화됨.
Stage 1 all-reduce는 projector만 학습하므로 gradient 크기가 작아 실질 overhead 제한적.

---

## §8. Sequence Packing — Greedy Knapsack 알고리즘 {#section-8}

*(archived from dataloader_trials.md §8, 2026-04-10)*

### 개념: 왜 packing이 필요한가

각 샘플(오디오+전사)은 길이가 다르다. 한 샘플씩 GPU에 넣으면 `cutoff_len` 슬롯의 대부분이 padding으로 낭비된다. 여러 샘플을 하나의 슬롯에 꽉 채워 GPU compute를 최대화하는 것이 sequence packing이다.

```
[  샘플A(4000토큰)  |  샘플C(5000토큰)  |  샘플B(3000토큰) | 샘플D(2000토큰) | pad(2384) ]
└──────────────────────────── cutoff_len=16384 ────────────────────────────────────────────┘
```

### Greedy Knapsack 동작 (배낭 채우기)

배낭(bin) 용량 = `cutoff_len`. 물건들(샘플들)을 최대한 꽉 채우는 문제.

`create_packer()` → `pack_samples()` 흐름:

1. **정렬**: bucket 내 샘플을 토큰 길이 오름차순으로 정렬 (bisect 탐색을 위해 필요)
2. **Greedy bin-filling**:
   ```
   bin 열기, remaining = 16384

   bisect로 remaining 이하 가장 큰 샘플 탐색 → O(log n)
   → 발견: bin에 추가, remaining 갱신
   → 없음: bin 닫고 다음 bin 열기

   반복
   ```
3. **패딩**: 각 bin을 `cutoff_len`에 맞게 우측 패딩 (`pad_token_id`)
4. **attention_mask**: 서브시퀀스별 1, 2, 3, ... (block-diagonal masking용), 패딩=0

"Greedy"인 이유: 현재 남은 공간에 들어갈 수 있는 가장 큰 샘플부터 탐욕적으로 집어넣는다. 최적 해(NP-hard)가 아니지만, 실용적으로 충분한 충전율을 얻는다.

### `packing_bucket_size`와 GPU utilization의 관계

packer는 전체 데이터셋을 한꺼번에 보지 않는다. HuggingFace `dataset.map(batched=True, batch_size=N)`이 `N`개씩 잘라서 packer에 전달한다. 이 `N`이 `packing_bucket_size`다.

```
스트리밍 데이터 → [ bucket N개 ] → packer(greedy 탐색) → bins → GPU
```

**bucket이 작을 때 (200)**:
- 탐색 풀이 좁음 → 길이 다양성 낮음 → 남은 공간을 못 채워 padding 증가
- CPU 처리량 적음 → GPU가 다음 batch를 빠르게 받음 → GPU starving 없음
- 실측: GPU util ~95%, ~55s/step (cutoff=16384)

**bucket이 클 때 (1000)**:
- 탐색 풀이 넓음 → fill ratio 이론상↑
- 하지만 1000개를 CPU에서 한꺼번에 정렬+bisect 반복 → CPU 처리 시간↑
- GPU가 다음 batch 대기하며 유휴 → **GPU util ~27%로 급락**
- 실측: 역효과. GPU util 오히려 감소

**핵심 trade-off**:

| bucket_size | fill ratio | CPU 처리 시간 | GPU util | s/step | 결과 |
|-------------|-----------|--------------|---------|--------|------|
| 200 | ~85~90% | 짧음 | ~95% | ~55s | 현재 최적 |
| 400 | ~90~93%? | 중간 | — | ~95s (step8) | 역효과 |
| 1000 | ~95%+ | 김 → GPU 굶김 | ~27% | ~117s (step4) | 역효과 |

`packing_bucket_size`는 fill ratio(품질)와 CPU latency(GPU 굶김) 사이의 trade-off다. precomputed 모드에서 오디오 디코딩이 없어 CPU가 빠르더라도, bucket이 너무 크면 packer 자체가 새 병목이 된다.

---

## §9. Liger Kernel 효과 측정 (2026-04-10) {#section-9}

*(archived from dataloader_trials.md §9, 2026-04-10)*

**배경**: Liger는 Qwen3.5-2B 내부 연산(RoPE, RMSNorm, SwiGLU, fused CE)을 Triton으로 fuse한다.
측정 당시 `process_batch_size=64`가 설정에 남아 있어 두 실험 모두 DataLoader 병목(108s/26s 교대) 하에서 진행됨.

| 실험 | slow step | fast step |
|------|-----------|-----------|
| Liger OFF | ~109 s | ~25 s |
| Liger ON (올바른 qwen3_5 패치) | ~108 s | ~26 s |

**결론**: Liger 적용 여부와 무관하게 스텝 시간이 동일. 원인:
- compute(fast step)는 ~26 s, DataLoader 대기(slow step)는 ~108 s
- Liger는 compute 구간만 최적화하므로 전체 시간 변화 없음
- **DataLoader stall이 해소되기 전까지는 Liger 효과를 측정할 수 없음**

**process_batch_size=64 방치 이슈**: 64 trial 이후 32로 되돌리지 않아 후속 실험 전부가 영향받음.
`config.py`에서 32로 복구 완료 (2026-04-10). 다음 실행부터 정상 적용.

**Liger가 효과 없는 근본 원인 (2026-04-10 분석)**:

Liger가 최적화하는 연산(RoPE, RMSNorm, SwiGLU, CE loss)은 **memory-bound element-wise 연산**으로, 전체 compute의 5~10%에 불과하다. 학습 step의 대부분은 Liger가 건드리지 않는 **linear projection matmul** (Q/K/V/O + gate/up/down MLP = 레이어당 7개 × 28 레이어 = 196개)과 **attention 연산**이 지배한다. gradient_checkpointing으로 forward가 2회 실행되는 것도 matmul 2회를 의미한다.

즉, Liger가 최적화하는 5~10% 구간을 2배 빠르게 해도 전체 step 시간은 2.5~5% 개선에 그치며, 이는 측정 노이즈 범위이다.

---

## §9b. flash-linear-attention 미설치 문제 (2026-04-10, 발견) {#section-9b}

*(archived from dataloader_trials.md §9b, 2026-04-10)*

**증상**: 학습 시작 시 반복 출력:
```
The fast path is not available because one of the required library is not installed.
Falling back to torch implementation.
```

**원인**: Qwen3.5는 hybrid 아키텍처로, 일부 레이어가 **gated delta rule linear attention**을 사용한다. `flash-linear-attention` (fla) 패키지가 설치되어 있으면 fused CUDA kernel으로 실행되지만, 미설치 시 순수 PyTorch loop fallback으로 동작한다.

**영향**: Liger가 최적화하는 RoPE/RMSNorm/SwiGLU는 전체 compute의 5~10%이지만, linear attention은 전체 attention 연산의 일부를 차지한다. fused kernel vs torch fallback의 속도 차이는 상당할 수 있으며, **Liger보다 훨씬 큰 속도 영향**이 예상된다.

**필요 패키지**:
- `causal-conv1d` (fla 의존성)
- `flash-linear-attention`

**설치 시 주의**: 시스템 nvcc (11.8)와 PyTorch CUDA (12.4) 버전 불일치로 빌드 실패 가능. conda 환경의 CUDA toolkit (12.4+)을 사용하도록 `CUDA_HOME` 설정 필요.
```bash
CUDA_HOME=/mnt/ddn/users/jos/miniforge3/envs/audio \
PATH=/mnt/ddn/users/jos/miniforge3/envs/audio/bin:$PATH \
pip install causal-conv1d flash-linear-attention --no-build-isolation
```

---

## §9c. 환경 변경 로그 — fla + torch + CUDA 호환성 (2026-04-11) {#section-9c}

*(archived from dataloader_trials.md §9c, 2026-04-11)*

fla 설치를 위해 시스템 및 conda 환경에 가해진 변경 사항 기록.

### 변경 전 환경 (baseline)

| 항목 | 버전 |
|------|------|
| torch | 2.5.1+cu124 |
| torchaudio | 2.5.1+cu124 |
| triton | 3.1.0 |
| CUDA driver | 535.129.03 (CUDA 12.2) |
| 시스템 nvcc | 11.8 |
| conda nvcc | 없음 |
| fla | 미설치 |
| causal-conv1d | 미설치 |

### 변경 내용 (시간순)

1. **conda cuda-toolkit 설치** (`conda install -n audio -c nvidia cuda-toolkit=12.4`)
   - nvcc 12.9가 conda env에 설치됨 (시스템 nvcc 11.8과 별도)
   - 목적: causal-conv1d CUDA 커널 빌드 시 nvcc 버전 불일치 해결

2. **causal-conv1d 1.6.1 빌드 설치**
   - `CUDA_HOME` + `CPLUS_INCLUDE_PATH` 설정 필요 (`cuda_runtime_api.h` 경로)
   ```bash
   CUDA_HOME=/mnt/ddn/users/jos/miniforge3/envs/audio \
   CPLUS_INCLUDE_PATH=.../targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH \
   PATH=/mnt/ddn/users/jos/miniforge3/envs/audio/bin:$PATH \
   pip install causal-conv1d --no-build-isolation
   ```

3. **flash-linear-attention 0.4.2 설치** (pure Python wheel, 빌드 불필요)

4. **CUDA toolkit 12.4 시스템 설치** (`apt install cuda-toolkit-12-4`)
   - 목적: torch 2.6.0의 CUDA runtime 12.4 요구 충족 (driver 535는 12.2까지만 지원했으나, toolkit 설치로 해결)
   - `/mnt/fr20tb/wbl_residency/jos/setup.sh` 참조

5. **torch 2.5.1 → 2.6.0+cu124 업그레이드**
   ```bash
   pip install torch torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu124
   ```
   - triton 3.1.0 → 3.2.0 자동 업그레이드 (torch 2.6.0 의존성)
   - transformers의 "triton 3.2.0 미만" 경고 해소

6. **nvidia-nccl-cu12 강제 재설치** (`pip install nvidia-nccl-cu12==2.21.5 --force-reinstall --no-deps`)
   - torch 2.11.0 시도 시 NCCL 2.28.9+cuda13.0으로 교체되었으나, 롤백 시 pip 메타데이터만 되돌아가고 실제 바이너리는 2.28.9 잔류
   - `strings libnccl.so.2 | grep "NCCL version"`으로 실제 버전 확인 후 강제 재설치

7. **nvidia-cudnn-cu12 9.1.0.70 재설치**
   - `pip install nvidia-cudnn-cu12 --upgrade`가 9.20.0.48 설치 → torch 2.6.0 요구 버전(9.1.0.70)과 불일치 → `CUDNN_STATUS_NOT_INITIALIZED`
   - 정확한 버전 지정 필요: `pip install nvidia-cudnn-cu12==9.1.0.70`

8. **fla 0.4.2 → 0.3.2 다운그레이드**
   - 0.4.2: loss=780→nan, grad_norm=nan (수치 발산)
   - 0.3.2: 역시 loss=0, grad_norm=nan (수치 발산은 fla 버전 무관, torch 2.6.0과의 근본 비호환 가능성)

9. **transformers 5.5.0 패치** (`import_utils.py:796`)
   - fla 0.3.2에 `__version__` 속성 없음 → `is_flash_linear_attention_available()`에서 `'N/A'` 파싱 실패
   - `InvalidVersion: 'N/A'` → 모든 모델 로드 시 crash (liger 유무 무관)
   - `None`/`'N/A'`/빈 문자열 → `return False` 처리로 패치

10. **liger-kernel 0.7.0 + torch 2.6.0 호환 문제**
    - liger가 qwen3_5 모델을 import할 때 transformers의 fla 체크가 트리거됨
    - 위 패치(9번)로 해결. liger 자체 업그레이드는 0.7.0이 최신

11. **dataloader_num_workers 1 → 0** (precomputed 모드)
    - cutoff_len=65536에서 worker=1 시 Bus error (shared memory)
    - pre-packed 모드는 데이터가 이미 메모리에 있어 worker 불필요
    - worker=0: 메인 프로세스에서 직접 로드, Bus error 해결, 오버헤드 제거

### 변경 후 환경 (현재)

| 항목 | 버전 |
|------|------|
| torch | **2.6.0+cu124** |
| torchaudio | **2.6.0+cu124** |
| triton | **3.2.0** |
| CUDA driver | 535.129.03 (변경 없음) |
| CUDA toolkit | **12.4** (apt 설치) |
| conda nvcc | **12.9** |
| fla-core | **0.4.2** (0.3.2는 fla.ops/modules 미지원) |
| flash-linear-attention | **0.4.2** |
| causal-conv1d | **1.6.1** |
| nvidia-nccl-cu12 | 2.21.5 |
| nvidia-cudnn-cu12 | 9.1.0.70 |

### 삽질 과정에서 확인된 비호환 조합

| torch | triton | fla | 결과 |
|-------|--------|-----|------|
| 2.5.1 | 3.1.0 | 0.4.2 | fla import 에러 (`STAGE` 미지원) |
| 2.5.1 | 3.6.0 | 0.4.2 | loss=nan, grad_norm=nan (torch-triton 비호환) |
| 2.5.1 | 3.2.0 | — | `AttrsDescriptor` dataclass 에러 |
| 2.5.1 | 3.1.0 | 0.3.2 | import 성공 (학습 OOM으로 미확인) |
| 2.11.0+cu130 | 3.6.0 | 0.4.2 | CUDA driver 부족 (535 < cu130 요구) |
| 2.6.0+cu124 | 3.2.0 | 0.4.2 | NCCL 2.28.9 잔류 → driver 에러 (강제 재설치로 해결) |
| 2.6.0+cu124 | 3.2.0 | 0.4.2 | NCCL·cuDNN 수정 후에도 **loss=780, grad_norm=nan** → 수치 발산 |
| 2.6.0+cu124 | 3.2.0 | 0.3.2 | **loss=0, grad_norm=nan** → 역시 수치 발산 |
| 2.6.0+cu124 | 3.2.0 | 제거 | transformers `InvalidVersion: 'N/A'` (fla 버전 체크 버그) |
| 2.6.0+cu124 | 3.2.0 | 0.3.2 + transformers 패치 | `--no-liger`만 정상. liger ON → nan (RoPE 원인) |
| **2.6.0+cu124** | **3.2.0** | **0.3.2 + transformers 패치 + partial RoPE fix** | **liger 4개 모두 정상** (loss=6.965, grad_norm=11.19) |

### transformers 5.5.0 fla 버전 체크 버그

`transformers/utils/import_utils.py`의 `is_flash_linear_attention_available()`가 fla 미설치 또는 `__version__` 미정의 시 `'N/A'`를 `packaging.version.parse()`에 넘겨 `InvalidVersion` 에러 발생.

fla 0.3.2는 `__version__` 속성이 없고, fla를 완전 제거해도 `_is_package_available("fla")`가 잔여 메타데이터로 인해 `True`를 반환하면서 `'N/A'` 버전 문자열을 파싱 시도.

**패치** (`import_utils.py:796`):
```python
@lru_cache
def is_flash_linear_attention_available():
    is_available, fla_version = _is_package_available("fla", return_version=True)
    if not is_available or fla_version in (None, "N/A", ""):
        return False
    try:
        return is_torch_cuda_available() and version.parse(fla_version) >= version.parse("0.2.2")
    except Exception:
        return False
```

> 패치 파일: `/mnt/ddn/users/jos/miniforge3/envs/audio/lib/python3.10/site-packages/transformers/utils/import_utils.py:796`
> 주의: transformers 업데이트 시 이 패치가 덮어씌워짐. 업데이트 후 재적용 필요.

### ~~fla + torch 2.6.0 수치 발산 문제~~ → liger RoPE + Qwen3.5 partial rotary 비호환 (해결됨)

**증상**: loss=nan/0, grad_norm=nan. fla와 무관하게 **liger가 켜져 있으면 발생**, `--no-liger`면 정상.

**원인 특정 과정**:
1. fla 0.4.2 + liger → nan → "fla 문제?" 의심
2. fla 0.3.2 + liger → nan → "fla 버전 무관"
3. fla 0.3.2 + `--no-liger` → **정상** (loss=5.787) → "liger가 원인"
4. liger에서 fused CE만 끔 → nan → "CE 아님"
5. liger에서 RoPE만 켬 → nan → "**RoPE가 범인**"
6. liger에서 RoPE만 끔 (RMSNorm+SwiGLU+CE) → **정상** (loss=6.817) → 확정

**root cause**: **Qwen3.5의 `partial_rotary_factor=0.25`**.

Qwen3.5는 head_dim=256 중 앞쪽 64차원만 RoPE 적용 (partial rotary). cos/sin shape = `(bsz, seq_len, 64)`.

liger의 Triton RoPE kernel은 `cos_offsets = tl.arange(0, pad_hd // 2)` → `pad_hd = head_dim = 256` → **128개**를 cos에서 읽으려 함. 실제 cos는 64개뿐 → **엉뚱한 메모리 읽기** → 잘못된 값 (실측 max diff = 10.3 vs 원본).

```
liger 기대: cos shape = (*, head_dim // 2) = (*, 128)
실제:       cos shape = (*, rope_dim)       = (*, 64)     ← partial_rotary_factor=0.25
```

> torch 2.5.1에서도 동일하게 잘못된 값이었지만, triton 3.1.0에서는 우연히 nan까지 도달하지 않았을 가능성.

**해결**: partial rotary 대응 래퍼. 앞쪽 `rope_dim` 차원만 liger Triton kernel에 넘기고 나머지는 pass-through.

```python
def _partial_liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    rope_dim = cos.shape[-1]  # 64
    q_rope, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    k_rope, k_pass = k[..., :rope_dim], k[..., rope_dim:]
    q_rope, k_rope = LigerRopeFunction.apply(
        q_rope.contiguous(), k_rope.contiguous(), cos, sin, position_ids, unsqueeze_dim)
    return torch.cat([q_rope, q_pass], dim=-1), torch.cat([k_rope, k_pass], dim=-1)
```

**검증**: loss=6.965, grad_norm=11.19 (4개 패치 모두 활성화)

### 교훈

- `pip install --upgrade`로 torch를 올렸다 내리면 nvidia-* 패키지의 **pip 메타데이터와 실제 바이너리가 불일치**할 수 있다. `strings`로 `.so` 파일의 실제 버전을 확인하고, `--force-reinstall --no-deps`로 교정.
- nvidia-cudnn-cu12는 `--upgrade`하면 최신(9.20)이 설치되지만 torch 2.6.0은 **정확히 9.1.0.70**을 요구한다. 버전 고정 필수.
- torch 버전을 올릴 때는 triton·nccl·cudnn·cuda driver를 **모두 함께** 맞춰야 한다. 하나라도 빠지면 런타임에 터진다.
- fla 0.3.2는 `__version__` 속성이 없어 transformers의 버전 체크가 실패한다. 패치 필수.
- fla를 `pip uninstall`해도 transformers가 잔여 메타데이터를 감지할 수 있다. 완전 제거가 어려우므로 패치가 더 안전.

---

## §10. Offline Pre-Packing (2026-04-10, 구현 완료) {#section-10}

*(archived from dataloader_trials.md §10, 2026-04-10)*

**문제**: DataLoader worker가 학습 중에 processor_fn + packer_fn을 실행하기 때문에 CPU 병목 발생.
packer_fn은 200개 샘플을 모아 greedy knapsack을 돌리는데, 이 처리 시간 동안 GPU가 대기한다.
결과적으로 ~26s compute / ~108s DataLoader stall 교대 패턴이 반복됨.

**해결책**: packing을 학습 시간에서 precompute 시간으로 이동.

```
[기존]
per-sample Arrow → (학습 중) processor_fn → packer_fn → GPU

[변경 후]
per-sample Arrow → (사전에) processor_fn → packer_fn → packed Arrow
                   (학습 중) packed Arrow 읽기 → GPU
```

**출력 경로**: `{precomputed_dir}/{encoder}/{dataset}/packed_{cutoff_len}/rank{N}.arrow`

**스키마** (packed Arrow):

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `input_ids` | list<int32> | 길이 = cutoff_len |
| `labels` | list<int32> | 길이 = cutoff_len |
| `attention_mask` | list<int32> | 길이 = cutoff_len (서브시퀀스 ID) |
| `audio_features` | list<list<float32>> | bin 내 클립별 flat feature |
| `audio_lengths` | list<int32> | bin 내 클립별 T_enc |

**실행**:
```bash
# 8 rank 병렬 (GPU 불필요)
bash precompute/run_pack.sh --encoder fb_dacvae

# 특정 데이터셋만
bash precompute/run_pack.sh --encoder fb_dacvae --datasets ls100,ls360
```

**학습 코드 변동**: `build_precomputed_pipeline`이 `packed_{cutoff_len}/rank{N}.arrow` 존재 여부를
자동 감지. 있으면 processor/packer map 건너뜀. 없으면 기존 on-the-fly 동작 유지 (하위 호환).

---

**이후 진행**: [mixed_packing_evolution_apr09-10.md](./mixed_packing_evolution_apr09-10.md) (RAM 로딩 OOM → shard split → shuffle/numpy/compile pool / max_steps 폐기)
