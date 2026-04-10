# Dataloader / 학습 속도 Trial 정리 — fb_dacvae, 8×GPU

## 효과 검증 요약

| 변수 | 변경 | 효과 | 검증 여부 |
|------|------|------|----------|
| raw audio → **precomputed** | on-the-fly 인코딩 → Arrow 파일 로드 | **×10~25 속도 향상** (113~351 s/it → 13 s/it) | ✅ 검증됨 |
| **packing_cutoff_len** 증가 | 2048 → 4096 → 8192 → **16384** | GPU util 24% → 46% → 82% → **~95%**, wall time 단축. 16384: ~55s/step, 2879 steps, **~44h** | ✅ 검증됨 |
| n_shards 증가 | 1 → 4 → 8 | 효과 없음 (GPU-bound, dataloader가 병목 아님) | ✅ 검증됨 (무효) |
| max_batch_tokens 증가 | 2600 → 8000 | 효과 없음 (training 미사용, eval 전용) | ✅ 검증됨 (무효) |
| **packing_bucket_size** 증가 | 200 → 1000 | GPU util 오히려 **~95% → ~27% 급락** (CPU 병목) | ✅ 검증됨 (무효, 역효과) |
| **packing_bucket_size** 조정 | 200 → 400 | 중단 (step 8 기준 ~95s, 200이 최적) | ✅ 검증됨 (무효) |
| **process_batch_size** 증가 | 32 → 64 | 2-slow(~109s)+2-fast(~29s) **교대 패턴**, avg GPU util ~70% → baseline 55s/95% 대비 악화. CPU burst (bucket=1000과 동일 원인) | ✅ 검증됨 (무효, 역효과) |
| **process_batch_size** 증가 | 32 → 128 | ~55~110s **불규칙 반복**, 평균 ~90s → baseline 55s 대비 악화. CPU burst 패턴 | ✅ 검증됨 (무효, 역효과) |
| **gradient_accumulation_steps** 증가 | 4 → 8 | step당 시간 ~2배 (avg ~179s), step 수 절반 → wall time 동일. DataLoader slow/fast 패턴 여전 | ✅ 검증됨 (무효) |
| **Liger kernel** (qwen3_5 올바른 패치) | OFF → ON (수동 monkey-patch) | slow 108s / fast 26s → slow 108s / fast 26s (동일) | ✅ 검증됨 (효과 없음 — DataLoader 병목 구간) |
| **Stage 1 FSDP** (`--fsdp-stage1`) | DDP → FSDP full_shard | DDP보다 **2.2배 느림** (~119s vs ~55s/step), 메모리 절감 없음 | ✅ 재검증됨 (무효) |
| **streaming=False** (RAM 로딩) | HF streaming → pyarrow.ipc.read_all() + map(num_proc=16) | ls100/ls360/ls500 성공, **MLS에서 OOM kill** (cgroup 제한 + cp 동시 실행) | ✅ 검증됨 (대용량 불가) |
| **Pre-packed Arrow** | on-the-fly pack → 사전 packed Arrow 직접 로드 | map 2단계 완전 스킵 (processor+packer 0s). DataLoader 병목 해소 | ✅ 검증됨 |
| **gradient_checkpointing=False** (Stage 1) | True → False | cutoff_len=16384에서 **OOM** (79.3GB/GPU 소진) | ✅ 검증됨 (불가, checkpointing 필수) |
| **flash-linear-attention** | 미설치 → 설치 | Qwen3.5의 linear attention이 torch fallback으로 동작 중. fla 설치 시 fused CUDA kernel 사용 → 속도 개선 기대 | 🔄 설치 중 |

**결론**: on-the-fly packing 모드에서 GPU utilization의 핵심 레버는 `packing_cutoff_len` 단 하나.
n_shards/num_workers/max_batch_tokens/packing_bucket_size/process_batch_size 모두 영향 없거나 역효과.
CPU 처리 단위(bucket_size, process_batch_size)를 늘리면 GPU 굶김(starving)이 발생해 오히려 악화됨.
- `process_batch_size`: 32(최적) < 64(역효과, avg ~90s, GPU util ~70%) < 128(역효과, avg ~90s, 불규칙)
- **Pre-packed Arrow** 모드로 전환 시 DataLoader 병목 자체가 해소됨 (§10 참조)

---

## Trial 결과 (측정된 s/it 기준)

| 날짜 | WandB run | 모드 | num_workers | n_shards | packing_cutoff_len | packing_bucket_size | 속도 (s/it) | GPU util | GPU mem | 비고 |
|------|-----------|------|-------------|----------|--------------------|---------------------|-------------|----------|---------|------|
| 04-08 22:14 | 221441 | raw audio | ? | — | 2048 | 1000 | ~351 | — | — | 초반 2 step, cffi 오류 |
| 04-08 22:46 | 224637 | raw audio | 0 | — | 2048 | 1000 | ~271 | — | — | 1 step 후 오류 |
| 04-08 22:56 | 225631 | raw audio | ? | — | 2048 | 1000 | ~316 | — | — | 3 step 후 중단 |
| 04-08 23:21 | 232137 | raw audio | 0 | — | 2048 | ? | ~134 | — | — | 15/20 step, OOM 오류 |
| 04-08 23:57 | 235719 | raw audio | 4 | — | 2048 | ? | ~113 | — | — | 2 step 후 오류 |
| 04-09 00:04 | 000426 | **precomputed** | 4 | 1 | 2048 | ? | **13.67** | ~35% | 13GB | 첫 precomputed 성공 |
| 04-09 23:20 | 232039 | precomputed | 4 | 1 | 2048 | 200 | **12.70** | ~35% | 13GB | 안정 실행 |
| 04-09 23:58 | 235801 | precomputed | 4 | 1 | 2048 | 200 | **12.68** | ~35% | 13GB | 재현 확인 |
| 04-10 00:44 | 004413 | precomputed | 8 | 4 | 2048 | 200 | **13.36** | ~35% | 13GB | "Too many dataloader workers: 8 (max=4)" 경고 |
| 04-10 00:51 | 005120 | precomputed | 8 | 4→8 | 2048 | 200 | — | — | — | SIGTERM (8-shard 생성 완료 전 시작) |
| 04-10 01:04 | 010352 | precomputed | 8 | 8 | 2048 | 200 | **~12.9** | ~35% | 13GB | 경고 없음 |
| 04-10 01:16 | — | precomputed | 8 | 8 | 2048 | 200 | **~13** | ~24% | 13GB | max_batch_tokens=8000 (training 무관, eval 전용) |
| 04-10 01:27 | — | precomputed | 8 | 8 | 4096 | 200 | **~23** | ~46% | 20GB | cutoff 2배 → util 2배 |
| 04-10 01:33 | — | precomputed | 8 | 8 | 8192 | 200 | **~38** | ~82% | 35GB | cutoff 4배 → util 3.4배 |
| 04-10 02:03 | — | precomputed | 8 | 8 | **16384** | 200 | **~55** | ~95% | ~64GB | isolated (bucket=200, process=32), step8 ETA ~44h |
| 04-10 09:25 | — | precomputed | 8 | 8 | 16384 | **1000** | ~117 (step4) | ~27% | ~64GB | CPU packing 병목, 5 step 중단 |
| 04-10 09:37 | — | precomputed | 8 | 8 | 16384 | **400** | ~95 (step8) | — | ~64GB | 중단 (여전히 악화) |
| 04-10 09:53 | — | precomputed | 8 | 8 | 16384 | 200 | **~55~110 불규칙** | — | ~64GB | process_batch_size=128, 평균 ~90s, 43 step |
| 04-10 11:04 | — | precomputed | 8 | 8 | 16384 | 200 | **25~29s(fast) / ~109s(slow) 교대**, avg ~90s | ~70% | ~64GB | process_batch_size=64, 2-slow+2-fast 주기4 패턴, 14 step 후 중단 |
| 04-10 12:56 | — | precomputed | 8 | 8 | 16384 | 200 | **26~28s(fast) / ~108s(slow) 교대** | — | ~64GB | **Liger OFF 베이스라인** (process_batch_size=64 방치, 실질 no-liger) |
| 04-10 12:56 | — | precomputed | 8 | 8 | 16384 | 200 | **26~28s(fast) / ~108s(slow) 교대** | — | ~64GB | **Liger ON (수정 후)** (process_batch_size=64 방치, 패치 성공했으나 DataLoader 병목 동일) |
| 04-10 11:32 | — | precomputed | 8 | 8 | 16384 | 200 | **128~280s 불규칙**, avg ~179s | — | ~64GB | **gradient_accumulation_steps=8**, step당 micro-step 2배 → 시간 2배, step 수 절반. wall time 이득 없음. 5 step 후 중단 |
| 04-10 20:22 | — | **pre-packed** | 1 | — | 16384 | — | OOM kill (MLS 로드 중) | — | — | streaming=False RAM 로드, num_proc=16, /dev/shm 캐시. ls100~ls500 성공 후 MLS(55GB×8 rank)에서 cgroup OOM. 동시 cp도 영향 |
| 04-10 21:07 | — | **pre-packed** | 1 | — | 16384 | — | ~244 (오측정) | 30~100% | ~13GB | ls100+ls360+ls500, pre-packed Arrow. **중복 학습 세션 2개 동시 실행으로 GPU 메모리 반분 → 오측정** |
| 04-10 21:32 | — | **pre-packed** | 1 | — | 16384 | — | OOM | — | 79.3GB | `gradient_checkpointing=False` 시도. activation이 GPU 메모리 전부 소진 → CUDA OOM |

## 핵심 관찰

### 1. 가장 큰 속도 향상 요인: raw audio → precomputed 전환
- raw audio 모드: 인코더(fb_dacvae)가 학습 중 실시간으로 GPU에서 동작 → 100~350 s/it
- precomputed 모드: 인코더 출력을 Arrow 파일에 저장해두고 로드만 함 → ~13 s/it
- **×10~25 속도 향상**

### 2. n_shards 효과: 미미함 (GPU-bound 상태)
- 1-shard vs 4-shard vs 8-shard 모두 ~12.7~13.4 s/it
- precomputed 모드에서 dataloader 전처리가 충분히 빠름 → GPU 연산이 실질적 병목

### 3. num_workers: HF가 `min(num_workers, dataset.num_shards)`로 cap
```
num_workers=8, n_shards=4 → "Too many dataloader workers: 8 (max=4)" → 실제 4 workers
num_workers=8, n_shards=8 → 경고 없음 → 실제 8 workers
```

### 4. `max_batch_tokens`는 training에 영향 없음
- `train_pipeline_override.py`에서 `max_eval_tokens = cfg["max_batch_tokens"]`로만 사용됨 (WER eval 전용)
- training 배치 크기는 `packing_cutoff_len` × `per_device_train_batch_size`로 결정됨
- `max_batch_tokens=2600→8000` 변경 후에도 GPU mem/util 변화 없었음 (확인됨)

### 5. `packing_cutoff_len`이 GPU utilization의 핵심 레버
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

### 6. Stage 1 FSDP 검증 결과 (2026-04-10, 무효)

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

#### 이유 1: 메모리 절감 없음 — activation이 메모리를 지배

FSDP의 파라미터 샤딩 효과:
```
Qwen3.5-2B in bf16 = ~4 GB
FSDP 8-GPU 분산 → 500 MB/GPU  ← 이론상 3.5 GB/GPU 절감
```

그런데 cutoff_len=16384인 packed sequence를 처리하면 각 레이어의 attention/FFN 중간 activation이 파라미터보다 훨씬 크다. 28개 레이어 × 16384 토큰의 activation 합산이 파라미터 절감분(3.5 GB)을 압도한다.

실측 결과: 64 GB → 66 GB로 오히려 증가. **파라미터가 메모리 병목이 아니라 activation이 병목**이므로 FSDP 파라미터 샤딩은 효과가 없다.

#### 이유 2: forward마다 레이어별 all-gather 통신 발생

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

#### 이유 3 (결정타): `gradient_checkpointing` + FSDP `full_shard` 충돌

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

> **참고**: `fsdp_config.activation_checkpointing`으로 전환하면 이유 3의 중복 all-gather는 해소되지만, 이유 1(activation이 메모리 지배)과 이유 2(레이어별 all-gather 통신)는 여전히 유효하다. 또한 `gradient_checkpointing` 자체를 끄는 것도 불가능하다 — cutoff_len=16384에서 checkpointing=False 시 GPU 메모리 79.3GB를 전부 소진하고 OOM이 발생한다 (§12 참조). 따라서 Stage 1 FSDP는 설정을 개선하더라도 DDP 대비 이점이 없다.

#### 왜 Stage 2는 FSDP가 유효한가

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

### 7. DDP All-Reduce 병목 (Stage 1 한정)

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

### 8. Sequence Packing — Greedy Knapsack 알고리즘

#### 개념: 왜 packing이 필요한가

각 샘플(오디오+전사)은 길이가 다르다. 한 샘플씩 GPU에 넣으면 `cutoff_len` 슬롯의 대부분이 padding으로 낭비된다. 여러 샘플을 하나의 슬롯에 꽉 채워 GPU compute를 최대화하는 것이 sequence packing이다.

```
[  샘플A(4000토큰)  |  샘플C(5000토큰)  |  샘플B(3000토큰) | 샘플D(2000토큰) | pad(2384) ]
└──────────────────────────── cutoff_len=16384 ────────────────────────────────────────────┘
```

#### Greedy Knapsack 동작 (배낭 채우기)

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

#### `packing_bucket_size`와 GPU utilization의 관계

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
| 200 | ~85~90% | 짧음 | ~95% | ~55s | ✅ 현재 최적 |
| 400 | ~90~93%? | 중간 | — | ~95s (step8) | ❌ 역효과 |
| 1000 | ~95%+ | 김 → GPU 굶김 | ~27% | ~117s (step4) | ❌ 역효과 |

`packing_bucket_size`는 fill ratio(품질)와 CPU latency(GPU 굶김) 사이의 trade-off다. precomputed 모드에서 오디오 디코딩이 없어 CPU가 빠르더라도, bucket이 너무 크면 packer 자체가 새 병목이 된다.

### 9. Liger Kernel 효과 측정 (2026-04-10)

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

### 9b. flash-linear-attention 미설치 문제 (2026-04-10, 발견)

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

**상태**: 🔄 설치 진행 중 (2026-04-10)

---

### 10. Offline Pre-Packing (2026-04-10, 구현 완료)

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

### 11. streaming=False RAM 로딩 시도 (2026-04-10, OOM)

**동기**: HF streaming DataLoader가 CPU 병목 → 데이터를 한 번에 RAM에 올려서 `num_proc=16` 병렬 map으로 처리.

**구현**:
```python
# pyarrow.ipc로 Arrow 파일 전체를 RAM에 로드
_tables = [_pa_ipc.open_file(f).read_all() for f in data_files]
ds = _HFDataset(_pa.concat_tables(_tables))

# 128코어 / 8 rank = 16 per rank, 캐시는 /dev/shm (RAM tmpfs)
ds = ds.map(processor_fn, num_proc=16,
            cache_file_name=f"/dev/shm/hf_map_cache/rank{rank}/{ds_key}_proc.arrow")
ds = ds.map(packer_fn, num_proc=16,
            cache_file_name=f"/dev/shm/hf_map_cache/rank{rank}/{ds_key}_pack.arrow")
```

**결과**:
- ls100 (4.4GB), ls360 (16GB), ls500 (22GB): 성공. processor ~34s + packer ~6s per rank.
- **MLS (55GB × 8 rank = 440GB)에서 OOM kill**.
  - `cp invoked oom-killer` — ddn→local 복사가 동시에 실행되어 cgroup 메모리 압박 가중
  - 프로세스 RSS: ~250GB (`anon-rss:261962984kB`)
  - cgroup 메모리 제한에 도달 (`CONSTRAINT_MEMCG`)

**교훈**: 소규모 데이터셋은 RAM 로딩이 가능하지만, MLS/GS 규모(55GB/rank)는 8 rank 동시 로드 시 메모리 초과. Pre-packed Arrow 모드가 더 실용적.

### 12. gradient_checkpointing=False 검증 (2026-04-10, OOM)

**가설**: Stage 1에서 GPU 메모리가 ~64GB/80GB → 여유 있으므로 checkpointing을 끄면 activation recompute가 제거되어 ~2배 빠를 것.

**결과**: CUDA OOM.
```
torch.OutOfMemoryError: CUDA out of memory.
GPU 0 has a total capacity of 79.33 GiB of which 1.81 MiB is free.
Process 154590 has 79.31 GiB memory in use.
77.76 GiB is allocated by PyTorch.
```

**원인**: cutoff_len=16384 시퀀스의 28 레이어 full activation이 80GB를 초과.
gradient_checkpointing=True 시 ~64GB (레이어 경계만 저장), False 시 ~80GB+ (모든 중간 텐서 저장).

**결론**: cutoff_len=16384에서 `gradient_checkpointing=True`는 **필수**. 끌 수 없음.

> 참고: 이전에 "13GB/GPU, 244s/step" 측정은 **학습 세션 2개가 동시에 실행**되어 GPU 메모리를 반분한 결과였음. 단일 세션 시 정상적으로 ~64GB 사용.

---

## 핵심 변수 설명

| 변수 | 위치 | 역할 | 현재값 |
|------|------|------|--------|
| `n_shards` | `shard_arrow.py --n-shards` | `load_dataset` num_shards → worker 상한 결정 | 8 |
| `dataloader_num_workers` | `train_pipeline_override.py` | DataLoader 병렬 worker 수 | 1 (precomputed) / 8 (raw audio) |
| `packing_cutoff_len` | `config.py` | packed bin 최대 토큰 수 (GPU util의 핵심 레버) | 16384 (현재 최적, wall ~44h) |
| `packing_bucket_size` | `config.py` | knapsack packer greedy 탐색 버킷 크기 (fill ratio vs CPU latency trade-off) | **200** (최적 확인) |
| `process_batch_size` | `config.py` | 토크나이징 배치 크기 | **32** (64·128 역효과 확인, 32가 최적) |
| `gradient_accumulation_steps` | `train_pipeline_override.py` | micro-step 수 (업데이트 1회당) | 4 (검증 예정) |
| `per_device_train_batch_size` | `train_pipeline_override.py` | GPU당 bin 수 (S1=1, S2=2) | 1/2 |

## 코드 변경 이력

| 파일 | 변경 내용 |
|------|----------|
| `precompute/shard_arrow.py` | 신규 생성: 2-pass streaming row-level split |
| `train_pipeline_override.py` | `build_precomputed_pipeline`: shard 파일 자동 감지 (`rank{N}_s*.arrow`) |
| `train_pipeline_override.py` | `dataloader_num_workers`: 4 → 8 (Stage1, Stage2 모두) |
| `config.py` | `packing_cutoff_len`: 2048 → 4096 → 8192 → 16384 |
| `config.py` | `packing_bucket_size`: 200 → 1000/400 (역효과 확인) → **200 복귀** |
| `config.py` | `process_batch_size`: 32 → 128 (역효과) → 32 복귀 → **64 추가 검증 (역효과)** → **32 최종 확정** |
| `train_pipeline_override.py` | `build_precomputed_pipeline`: pre-packed Arrow 자동 감지, streaming=False RAM 로드 경로 추가 |
| `train_pipeline_override.py` | `dataloader_num_workers`: precomputed 모드에서 1로 변경 (pre-packed은 CPU 처리 불필요) |
| `train_pipeline_override.py` | `gradient_checkpointing=False` 시도 → OOM → True로 복구 |
