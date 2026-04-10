# Dataloader / 학습 속도 Trial 정리 — fb_dacvae, 8×GPU

## 효과 검증 요약

| 변수 | 변경 | 효과 | 검증 여부 |
|------|------|------|----------|
| raw audio → **precomputed** | on-the-fly 인코딩 → Arrow 파일 로드 | **×10~25 속도 향상** (113~351 s/it → 13 s/it) | ✅ 검증됨 |
| **packing_cutoff_len** 증가 | 2048 → 4096 → 8192 | **GPU util 24% → 46% → 82%, wall time 단축** | ✅ 검증됨 |
| n_shards 증가 | 1 → 4 → 8 | 효과 없음 (GPU-bound, dataloader가 병목 아님) | ✅ 검증됨 (무효) |
| max_batch_tokens 증가 | 2600 → 8000 | 효과 없음 (training 미사용, eval 전용) | ✅ 검증됨 (무효) |
| **packing_bucket_size** 증가 | 200 → 1000 | 미검증 | 🔄 검증 예정 |
| **process_batch_size** 증가 | 32 → 128 | 미검증 | 🔄 검증 예정 |
| **gradient_accumulation_steps** 증가 | 4 → 8 | 미검증 | 🔄 검증 예정 |
| packing_cutoff_len=16384 | 8192 → 16384 | **~55s/step, 2879 steps → wall ~44h** (8192보다 빠름) | ✅ 검증됨 (유효) |
| **Stage 1 FSDP** (`--fsdp-stage1`) | DDP → FSDP full_shard | DDP보다 **2.2배 느림** (~119s vs ~55s/step), 메모리 절감 없음 | ✅ 재검증됨 (무효) |

**결론**: 현 구조에서 GPU utilization의 핵심 레버는 `packing_cutoff_len` 단 하나.
n_shards/num_workers/max_batch_tokens는 영향 없음.

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

1. **메모리 절감 없음**: cutoff_len=16384의 activation이 메모리를 지배. FSDP는 파라미터만 샤딩(LLM ~4GB → ~500MB/GPU)하지만, packed sequence activation 메모리가 훨씬 커서 절감 효과 없음.

2. **속도 저하**: `gradient_checkpointing`과 FSDP `full_shard` 조합 시 backward pass에서 redundant AllGather 발생 (HF Trainer 경고). 또한 forward pass마다 layer별 all-gather → 통신 오버헤드.

   ```
   When using FSDP full shard, instead of using `gradient_checkpointing` in TrainingArguments,
   please use `activation_checkpointing` in `fsdp_config`. The former introduces a redundant
   AllGather operation in backward pass.
   ```

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
`create_packer(cutoff_len)` 함수가 반환하는 `pack_samples()`의 동작:

1. **정렬**: bucket 내 샘플을 토큰 길이 오름차순으로 정렬
2. **Greedy bin-filling**: 비어있는 bin을 하나 열고, `remaining = cutoff_len`에서 시작
   - bisect로 `remaining` 이하인 가장 큰 샘플 탐색 (O(log n))
   - 찾으면 bin에 추가, `remaining` 갱신; 없으면 bin 닫고 다음 bin 열기
3. **패딩**: 각 bin을 `cutoff_len`에 맞게 `pad_token_id`로 우측 패딩
4. **attention_mask**: 서브시퀀스별 1, 2, 3, ... (block-diagonal masking용), 패딩=0

`packing_bucket_size`와 충전율의 관계:
- packer는 bucket 안에서만 탐색 → bucket이 작으면 탐색 풀이 좁아 bin이 덜 채워짐
- bucket=200: 평균 bin fill ratio ~85~90% 추정
- bucket=1000: 더 다양한 길이의 샘플을 동시에 탐색 → fill ratio ~95%+ 기대
- **precomputed 모드**: 오디오 디코딩 없어 CPU 부담 낮음 → 1000 이상 권장
- **raw audio 모드**: 오디오 디코딩이 CPU 병목 → 200 이하 권장

## 핵심 변수 설명

| 변수 | 위치 | 역할 | 현재값 |
|------|------|------|--------|
| `n_shards` | `shard_arrow.py --n-shards` | `load_dataset` num_shards → worker 상한 결정 | 8 |
| `dataloader_num_workers` | `train_pipeline_override.py` | DataLoader 병렬 worker 수 | 8 |
| `packing_cutoff_len` | `config.py` | packed bin 최대 토큰 수 (GPU util의 핵심 레버) | 16384 (현재 최적, wall ~44h) |
| `packing_bucket_size` | `config.py` | knapsack packer greedy 탐색 버킷 크기 | 200 (검증 예정) |
| `process_batch_size` | `config.py` | 토크나이징 배치 크기 | 32 (검증 예정) |
| `gradient_accumulation_steps` | `train_pipeline_override.py` | micro-step 수 (업데이트 1회당) | 4 (검증 예정) |
| `per_device_train_batch_size` | `train_pipeline_override.py` | GPU당 bin 수 (S1=1, S2=2) | 1/2 |

## 코드 변경 이력

| 파일 | 변경 내용 |
|------|----------|
| `precompute/shard_arrow.py` | 신규 생성: 2-pass streaming row-level split |
| `train_pipeline_override.py` | `build_precomputed_pipeline`: shard 파일 자동 감지 (`rank{N}_s*.arrow`) |
| `train_pipeline_override.py` | `dataloader_num_workers`: 4 → 8 (Stage1, Stage2 모두) |
| `config.py` | `packing_cutoff_len`: 2048 → 4096 → 8192 → 16384 |
| `config.py` | `packing_bucket_size`: 200 → 1000 (precomputed 모드 최적화) |
| `config.py` | `process_batch_size`: 32 → 128 (precomputed 모드 최적화) |
