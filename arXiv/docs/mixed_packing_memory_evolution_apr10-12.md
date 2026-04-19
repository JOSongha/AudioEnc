# Mixed Packing Memory Evolution (2026-04-10 ~ 04-12) (Archive)

이 파일은 `dataloader_trials.md` 에서 2026-04-14 에 분리된 archive 입니다.
주제: streaming RAM 로딩 시도 OOM → Data splits → shard 단위 split → shuffle 축소/제거 → HF Dataset numpy format → TORCH_WARM_POOL → max_steps 계산 변천 (rollback) → num_train_epochs + map-style Dataset 최종.
원문 섹션: §11 ~ §19 (num_data_splits 도입, mixed packing 진화, Stage 1/2 max_steps 최종)

---

## §11. streaming=False RAM 로딩 시도 (2026-04-10, OOM) {#section-11}

*(archived from dataloader_trials.md §11, 2026-04-14)*

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

## §13. Data Splits — 대용량 데이터셋 OOM 해결 (2026-04-11, 구현) {#section-13}

*(archived from dataloader_trials.md §13, 2026-04-14)*

#### 문제

Pre-packed Arrow 파일을 `read_all()`로 RAM에 올리는 구조에서, 대용량 데이터셋이 cgroup 메모리를 초과한다.

| 데이터셋 | packed 크기/rank | × 8 rank |
|----------|-----------------|----------|
| ls100 | 646MB | 5GB |
| ls360 | 2.0GB | 16GB |
| ls500 | 2.7GB | 22GB |
| vp | 3.1GB | 25GB |
| mls | 55GB | **440GB** |
| gs | ~55GB | **~440GB** |
| **합계** | | **~948GB** |

시스템 RAM 2TB이지만 cgroup 제한이 있어 ~500GB 이상 사용 시 OOM kill 발생. 실측: mls+gs+ls+vp 동시 로드 시 SIGKILL (3회 반복 확인).

#### 검토한 대안들

**1. rank 기반 분할 (기각)**

아이디어: epoch 1에서 rank 0~3 파일만 로드, epoch 2에서 rank 4~7.
```
epoch 1: GPU 0→rank0, GPU 1→rank1, ..., GPU 4→rank0, GPU 5→rank1, ...
epoch 2: GPU 0→rank4, GPU 1→rank5, ..., GPU 4→rank4, GPU 5→rank5, ...
```

**기각 이유**: 8개 GPU 프로세스가 각각 독립적으로 `read_all()`을 호출하므로, GPU 0과 GPU 4가 같은 rank0.arrow를 로드해도 **각 프로세스에 별도 복사본**이 생긴다. 메모리 = 8 × 55GB = 440GB (절약 없음).

mmap으로 물리 페이지를 공유하면 해결 가능하지만, packed Arrow의 가변 길이 컬럼(Sequence)은 pyarrow mmap에서 zero-copy가 보장되지 않아 효과 불확실.

**2. 순차 CPT (가능하지만 수동)**

별도 학습 실행을 이어붙이는 방식:
```bash
# 1차: 소규모 데이터
bash run.sh --encoder fb_dacvae --datasets ls100,ls360,ls500,vp
# 2차: checkpoint에서 resume + mls
bash run.sh --encoder fb_dacvae --datasets mls --resume <checkpoint_path>
```

코드 수정 불필요하지만 실행을 수동으로 관리해야 한다.

**3. bins 기반 분할 (채택)**

각 rank 파일 내의 bins(packed 시퀀스)을 N등분, split마다 1/N만 로드.
```
rank0.arrow: 2598 bins
  split 0: bins 0~1298    (27.5GB)  → 로드, 학습, 해제
  split 1: bins 1299~2597 (27.5GB)  → 로드, 학습, 해제
```

**채택 이유**: 각 GPU 프로세스가 자기 rank 파일만 읽되 일부분만 로드하므로, 프로세스 간 공유 없이도 메모리가 정확히 1/N로 줄어든다. 전체 데이터는 N회 split에 걸쳐 전부 소비된다.

#### 메모리 비교

| 설정 | 메모리 (전체 데이터셋) | 결과 |
|------|----------------------|------|
| num_data_splits=1 (기본) | ~948GB | ❌ OOM |
| num_data_splits=2 | ~474GB | ✅ 가능 |
| num_data_splits=4 | ~237GB | ✅ 여유 |

#### 구현 상세

**config.py**:
```python
"num_data_splits": 2,  # 1=전체 로드(기본), 2=절반씩, 4=1/4씩
```

**build_precomputed_pipeline** (새 파라미터):
```python
def build_precomputed_pipeline(..., data_split=0, num_data_splits=1):
    # pre-packed 로드 시:
    _table = _pa_ipc.open_file(str(packed_path)).read_all()
    if num_data_splits > 1:
        n = _table.num_rows
        chunk = n // num_data_splits
        start = data_split * chunk
        end = n if data_split == num_data_splits - 1 else start + chunk
        _table = _table.slice(start, end - start)  # 해당 split만 유지
```

주의: `read_all()` 시점에 전체 파일이 잠시 메모리에 올라간 후 `slice`로 잘린다. 피크 메모리는 전체 파일 크기이므로, 매우 큰 단일 파일(>100GB/rank)에서는 여전히 OOM 가능. 현재 mls 55GB/rank는 피크 시 문제없음.

**main() 학습 루프**:
```python
for data_split in range(num_data_splits):
    # 이번 split의 데이터 로드
    train_dataset = build_precomputed_pipeline(
        ..., data_split=data_split, num_data_splits=num_data_splits)
    # 학습
    run_stage1(... train_dataset=train_dataset ...)
    # 메모리 해제
    del train_dataset
    gc.collect()
```

Stage 1, Stage 2 모두 동일한 split 루프 적용.

#### trade-off 및 주의사항

1. **데이터 순서 편향**: greedy knapsack packer가 길이순 정렬하므로, split 0에 긴 시퀀스, split 1에 짧은 시퀀스가 몰릴 수 있음. 필요 시 `pack_arrow.py`에서 패킹 전 셔플 추가.

2. **Optimizer state 리셋**: split마다 Trainer가 새로 생성되므로 optimizer state(momentum, adaptive lr)가 초기화됨. learning rate scheduler도 리셋. 추후 checkpoint resume 로직으로 보완 가능.

3. **피크 메모리**: `read_all()` → `slice()` 과정에서 전체 파일이 순간적으로 메모리에 올라감. 파일 단위 split이 아닌 bins 단위 split이므로 이 피크를 피하려면 record batch 단위 읽기가 필요하나, 현재 파일 크기(55GB)에서는 문제없음.

4. **num_data_splits와 학습 step 수**: 각 split의 데이터가 1/N이므로 split당 step 수도 1/N. 전체 step 수는 동일 (N splits × 1/N steps = 원래 steps). wall time도 거의 동일.

## §12. gradient_checkpointing=False 검증 (2026-04-10, OOM) {#section-12}

*(archived from dataloader_trials.md §12, 2026-04-14)*

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

## §13. mixed packed: shard 단위 split (2026-04-12) {#section-13}

*(archived from dataloader_trials.md §13, 2026-04-14)*

**배경**: §11의 `num_data_splits`는 bins(row) 단위 슬라이스 — `read_all()`로 전체 파일을 RAM에 올린 뒤 `_table.slice(start, end)`로 잘랐다. 단일 packed 파일(rank당 ~55 GB) 시점에서는 피크가 감내 가능했지만, **mixed packing(`--mixed`)이 도입되면서 rank당 5 shard × ~22 GB = ~110 GB로 늘어났고**, 5 shard 전부를 `read_all()` → `concat_tables` 하는 순간 노드 RAM이 8 × 110 = ~880 GB까지 치솟아 OOM 가능성이 생겼다.

**원인**: [train_pipeline_override.py:1344-1356](../train_pipeline_override.py#L1344-L1356) 의 mixed 분기는 split 값과 무관하게 모든 shard를 먼저 메모리로 읽고, 그 다음에 bins 인덱스로 자르는 구조였다. iteration 시점 메모리는 1/N로 줄었지만 **peak load는 여전히 100%**.

```python
# 기존 (bins-level slice)
_tables = [_pa_ipc.open_file(str(f)).read_all() for f in mixed_shards]   # ← 여기서 100%
_table  = _pa.concat_tables(_tables)
if num_data_splits > 1:
    _table = _table.slice(start, end - start)                             # ← 너무 늦음
```

**수정**: shard 자체가 ~5개로 분할되어 있으므로, slice 단위를 **bins → shard**로 올린다.

```python
# 수정 (shard-level partition)
if num_data_splits > 1:
    n_sh  = len(mixed_shards)
    chunk = math.ceil(n_sh / num_data_splits)
    start = data_split * chunk
    end   = min(start + chunk, n_sh)
    sel   = mixed_shards[start:end]
else:
    sel = mixed_shards
_tables = [_pa_ipc.open_file(str(f)).read_all() for f in sel]            # ← 1/N만 로드
_table  = _pa.concat_tables(_tables) if len(_tables) > 1 else _tables[0]
# bins-level slice 분기 삭제 (shard로 이미 분할됨)
```

**메모리 (5 shard × ~22 GB / rank, 8 ranks 동일 노드)**

| `num_data_splits` | shard / split | rank당 peak | 노드 peak |
|---|---|---|---|
| 1 | 5 | ~110 GB | ~880 GB ❌ |
| 2 | 3 / 2 | ~66 GB | ~528 GB ⚠️ |
| 4 | 2 / 2 / 1 / — | ~44 GB | ~352 GB ✅ |
| **5** | 1 each | **~22 GB** | **~176 GB ✅✅** |

shard 수와 동일한 `num_data_splits=5`가 가장 균형 잡힘 (split마다 정확히 1 shard 로드).

**Trade-off**:
- (§11.1) 데이터 순서 편향 — pack_arrow.py mixed 모드는 패킹 전 셔플하므로, shard 간 분포는 거의 균등.
- (§11.2) Optimizer state는 split마다 리셋 (기존과 동일).
- shard 수가 num_data_splits로 나뉘지 않으면 마지막 split이 더 작아짐 (예: 5/2 = [3, 2]).

## §14. shuffle buffer_size 축소 (2026-04-12, OOM) {#section-14}

*(archived from dataloader_trials.md §14, 2026-04-14)*

**증상**: §13 적용 후 첫 실행 — `Mixed pre-packed (in-memory): 1/5 shard(s)` 로그 정상, Stage 1 진입, `0/2879 [00:00<?, ?it/s]` 까지 갔다가 약 32분 후 rank 0 SIGKILL (exitcode -9). dmesg `Memory cgroup out of memory: ... anon-rss:208 GB`. 한 프로세스가 208 GB → 8 ranks × 208 = ~1.66 TB → cgroup 1.5 TB 초과.

**원인**: [train_pipeline_override.py:1378-1379](../train_pipeline_override.py#L1378-L1379) 의 `ds.shuffle(buffer_size=10000)`.

`buffer_size`는 IterableDataset이 셔플 풀로 들고 있는 item 개수다. **bin 1개의 메모리 크기는 cutoff_len에 비례**하는데:

| `packing_cutoff_len` | shard 22 GB 기준 bin 수 (rank당) | bin 평균 크기 |
|---|---|---|
| 2048 (legacy default) | ~12000 | ~1.8 MB |
| 16384 (현재) | 1500 | **~14 MB** |

`buffer_size=10000`은 cutoff 2048 시절 ~18 GB / proc 정도였지만, cutoff 16384에서는 **~140 GB / proc**로 폭증. 거기에 underlying `_table` 22 GB + Trainer/Accelerate/intermediate = ~200 GB / proc.

**수정**: shard 1개당 bins가 1500 뿐이므로 10000 buffer 자체가 과도. 1000으로 축소 (사실상 거의 전체 셔플과 동일하면서 메모리 1/10).

```python
# train_pipeline_override.py mixed 분기 + per-dataset 분기 둘 다
ds = ds.shuffle(buffer_size=1000, seed=...)
```

**메모리 (예상, num_data_splits=5, mixed 1 shard / split)**

| 항목 | 크기 (proc당) |
|---|---|
| `_table` (read_all) | ~22 GB |
| shuffle buffer (1000 × 14 MB) | ~14 GB |
| Trainer / Accelerate / inter | ~5 GB |
| **합계** | **~41 GB / proc** |
| × 8 ranks | **~328 GB** ✅ (cgroup 1.5 TB 여유) |

**교훈**: cutoff_len 변경 시 shuffle `buffer_size`를 함께 재검토할 것. bin 크기 × buffer 가 proc당 실효 메모리 풋프린트.

## §15. HF Dataset numpy format (2026-04-12) {#section-15}

*(archived from dataloader_trials.md §15, 2026-04-14)*

**증상**: §13/§14 적용 후, 학습은 시작됐지만 `0/2879` 에서 멈추고 RAM 이 5.5 GB/min/proc 페이스로 선형 증가. 8 ranks × ~85 GB RSS, HWM ~129 GB. GPU util 0% 유지. 첫 step 까지 도달 못하고 cgroup 한계 근접.

**원인**: `_HFDataset(_table).to_iterable_dataset()` 가 매 `__next__` 마다 PyArrow 행 → Python dict 로 디코딩. 특히 `audio_features: list<list<float32>>` 컬럼이 nested Python list 로 풀리면서 polynomial blow-up.

검증 (rank0_s0.arrow, 1 bin):
```
list mode  : audio_features = list of 200 list of ~62720 Python float
            ≈ 12.5M Python float 객체 × 24 byte = ~300 MB / bin
numpy mode : audio_features = ndarray(200, dtype=object), 각각 ndarray(62720,) float32
            ≈ 200 × 62720 × 4 byte = ~50 MB / bin
```

**6배 차이**. shuffle buffer 1000 × 300 MB = 300 GB / proc (불가) vs 1000 × 50 MB = 50 GB / proc (감내).

**수정**: [train_pipeline_override.py:1357](../train_pipeline_override.py#L1357) (mixed) + [:1399](../train_pipeline_override.py#L1399) (per-dataset).
```python
ds = _HFDataset(_table).with_format("numpy").to_iterable_dataset()
#                       ^^^^^^^^^^^^^^^^^^^^^
```

OmniCollator는 이미 numpy 호환 ([train_pipeline_override.py:1043-1045](../train_pipeline_override.py#L1043-L1045)): `np.asarray(flat_feat, dtype=np.float32)`이 zero-copy로 처리.

추가 이점: Python loop 디코드 (CPU bound) → C-level buffer 복사. 디코딩 자체도 빨라짐.

## §16. TORCH_WARM_POOL=0 (2026-04-12, compile worker pool eager spawn 차단) {#section-16}

*(archived from dataloader_trials.md §16, 2026-04-14)*

**증상**: §15 fix 후에도 학습 시작 전에 8 rank 모두 ~85 GB RSS 잡고 멈춤. `ps -ef` 로 보니 각 rank 마다 `python torch/_inductor/compile_worker/__main__.py --workers=32 --parent={rank_pid}` 가 32개씩 떠 있음. **8 ranks × 32 workers = 256 compile worker subprocess**.

**원인**: [torch/_inductor/async_compile.py:318-329](file:///mnt/ddn/users/jos/miniforge3/envs/audio/lib/python3.10/site-packages/torch/_inductor/async_compile.py#L318-L329) 가 module import 시점에 `AsyncCompile.warm_pool()` 을 **eager 호출**:

```python
if (
    os.environ.get("TORCH_TNT_IN_USE", "0") == "1"
    or os.environ.get("TORCH_WARM_POOL", "1") != "1"
    or not has_triton_package()
    or config.is_fbcode()
):
    pass
else:
    AsyncCompile.warm_pool()   # ← module import 시 호출
```

`TORCH_WARM_POOL` 기본값이 `"1"`이라 디폴트로 동작. transformers / accelerate / fla 등이 `torch._inductor` 를 transitively import 하면서 자동 발동.

우리 코드에 `torch.compile()` 호출은 **한 줄도 없음** (`TrainingArguments.torch_compile=False`, wandb config로 확인). 그런데도 compile pool이 eagerly 떠서 메모리만 잡고 있음.

**메모리 비용**:
- worker subprocess 1개 ≈ Python 인터프리터 + torch lib ≈ **500 MB ~ 1 GB**
- 256 workers × 750 MB ≈ **192 GB CPU RAM** (학습 시작 전에 이미)

**수정**: `run.sh`에 환경변수 추가.
```bash
export TORCH_WARM_POOL=0
```

→ pool eager spawn 차단. 진짜 `torch.compile()` 호출되면 lazy spawn (우리 케이스엔 일어날 일 없음).

**연결**: §15 의 RAM 선형 증가는 (a) 데이터 디코드 + (b) compile worker pool 두 원인이 합쳐진 거였음. §15 가 (a) 를 6x 줄이고 §16 가 (b) 를 0 으로 만듦. 둘 다 적용해야 안정.

## §17. mixed 분기 shuffle 제거 (2026-04-12, unbounded RAM 누수) {#section-17}

*(archived from dataloader_trials.md §17, 2026-04-14)*

**증상**: §15+§16 적용 후, compile worker 0개 / numpy decode 효과 확인됐으나 학습 시작 후 RAM 이 **무한정 선형 증가**:

| 시점 (mixed loaded 후) | RSS / proc |
|---|---|
| ~1분 | ~33 GB |
| ~5분 | ~62 GB |
| ~10분 | ~85 GB |
| ~14분 | **~100 GB** (계속 자람) |

shuffle buffer 1000 × ~50 MB(numpy decoded) = ~50 GB / proc 에서 멈춰야 하는데, 100 GB 넘어서도 계속 자람. 14분간 첫 step 도달 못함. cgroup 한계 임박.

**원인 (가설)**: HF `IterableDataset.shuffle(buffer_size=N)` 의 reservoir buffer 내부에서 yield 된 item에 대한 reference 가 GC 되지 않는 leak. `with_format("numpy")` + interleave_datasets + shuffle 조합에서 발생. 정확한 root cause 는 HF datasets 라이브러리 내부지만 재현 가능.

**수정**: mixed 분기에서 `interleave_datasets` 와 `.shuffle()` 모두 스킵하고 `_HFDataset(_table).with_format("numpy").to_iterable_dataset()` 결과 그대로 반환.

```python
# train_pipeline_override.py mixed 분기
dataset_list.append(ds)
return ds   # interleave 불필요 (단일 dataset), shuffle 불필요 (pack_arrow.py가 이미 셔플)
```

**정당성**:
- **단일 dataset**: mixed 모드는 `dataset_list` 가 항상 길이 1 → `interleave_datasets` 호출 자체가 의미 없음.
- **이미 셔플됨**: [pack_arrow.py:275-282](../precompute/pack_arrow.py#L275-L282) 의 Phase 2 가 모든 sample 인덱스에 Fisher-Yates 셔플 후 packing → 각 shard 내부 1500 bins 가 이미 random 분포.
- **epoch 다양성**: `num_data_splits=5` 로 매 split 마다 다른 shard 로드 → split 단위 다양성 충분.
- **에폭 단위 동일 순서 한계**: 같은 shard 가 여러 epoch 에 걸쳐 같은 순서로 반복됨. 단점이지만 학습에 치명적이진 않음 (이미 random 한 순서, 각 shard 내부 1500 bins 의 단일 ordering).

**결과 (예상)**: shuffle buffer fill phase 0초 → 첫 step 즉시 도달. RAM 누수 사라짐.

## §18. max_steps 계산 버그 (2026-04-12, 학습 시간 N배 부풀림) {#section-18}

*(archived from dataloader_trials.md §18, 2026-04-14)*

**증상**: 학습 estimate 가 wall time 100h 이상으로 계산됨. 사용자 기억으론 비슷한 설정에서 ~5h 가능했음. 심지어 max_steps=2879 자체가 의심스러움.

**3개의 곱셈 버그가 합쳐져 있었음**:

#### A. `per_device_batch_size` hardcoded `2`

[train_pipeline_override.py:1801, 1917](../train_pipeline_override.py#L1801) 의 `calculate_max_steps` 호출이 항상 `per_device_batch_size=2` 로 고정. 그러나 실제 `TrainingArguments` 는 `per_device_train_batch_size=5` (Stage 1 line 1832) / `5` (Stage 2 line 2011).

→ max_steps 가 실제 필요량의 **2.5x 부풀려짐** (5/2 비율).

#### B. `num_data_splits` 미반영 (loop count 곱셈)

main() 의 학습 루프가 `for data_split in range(num_data_splits): run_stage1(...)` 로 N 번 호출. 각 호출이 `cfg["max_steps"]` (= 1 epoch 분량) 풀로 학습 → 총 `N × 1_epoch_steps` 를 학습. doc §11 노트는 "split당 1/N steps" 라고 적혀있는데 **코드는 그렇게 동작 안 함**.

→ 실제 학습 step 수가 **`num_data_splits` 배 (=5x)** 부풀려짐.

#### C. cross-stage `cfg["max_steps"]` 누수

`run_stage1` 이 `cfg["max_steps"]` 에 캐시 → `run_stage2` 가 `if cfg.get("max_steps") is None:` 가드에서 False → **stage1 의 값을 그대로 사용**. stage 2 자체 계산이 무시됨.

→ stage 2 epoch 수가 stage 1 기준으로 잘못됨.

#### 합산 효과

A × B = **2.5 × 5 = 12.5x** 부풀려짐. 거기에 C 로 stage 간 누수.

**수정** ([train_pipeline_override.py](../train_pipeline_override.py#L1765)):

1. `calculate_max_steps` 에 `num_data_splits` 파라미터 추가, 결과를 `math.ceil(total_steps / num_data_splits)` 로 나눔
2. `cfg["stage1_per_device_batch_size"]` / `cfg["stage2_per_device_batch_size"]` 키로 동적 batch size 전달 (기본 5)
3. main() 에서 `cfg["_max_steps_user_set"] = (args.max_steps is not None)` 마커 설정
4. `run_stage2` 에서 `if not cfg.get("_max_steps_user_set", False): cfg["max_steps"] = None` 로 stage1 캐시 초기화 후 재계산

**결과 (예상, 21460 audio-h, cutoff 16384, bs=5, grad_accum=4, gpu=8, num_data_splits=5)**:

```
seconds_per_pack       = 13107.2 / 31.25 = ~419 sec
total_seconds          = 21460 * 3600 = 77,256,000
estimated_total_packs  = 184,205
global_batch_size      = 5 * 8 * 4 = 160
steps_per_epoch_total  = ceil(184205 / 160) = 1152
per_split_max_steps    = ceil(1152 / 5) = 231
```

→ 한 split 당 231 step, 전체 5 split 합 = 1155 step (≈ 1 epoch).

22 s/step 가정 → 1155 × 22 = 25,410 sec ≈ **7시간** wall time. (사용자 기억 ~5h 와 근사한 범위)

## §19. max_steps 폐기 → num_train_epochs + map-style Dataset (2026-04-12, 최종) {#section-19}

*(archived from dataloader_trials.md §19, 2026-04-14)*

**배경**: §18 (calculate_max_steps 의 4개 sub-fix) 와 §19 중간 반복 (`cfg["max_steps"] = math.ceil(_table.num_rows / ...)`) 모두 결국 max_steps 를 어딘가에 박는 형태였음. user 가 "max step 자체를 박지 말라" 고 명확히 지적. 진짜 답은 옵션 4: **TrainingArguments 에서 `max_steps` 를 빼고 `num_train_epochs` 만 쓰기**.

**핵심 깨달음**: HF Trainer 가 IterableDataset 에서는 `max_steps` 를 요구하지만 (`__len__` 없음), **map-style Dataset (`__len__` 있음)** 이면 `num_train_epochs` 만으로 step 수 자동 계산. `_HFDataset(_table)` 는 원래 map-style 이고, `.to_iterable_dataset()` 호출만 빼면 map-style 그대로 유지.

**최종 수정**:

1. **`build_precomputed_pipeline` mixed 분기**: `.to_iterable_dataset()` 제거.
   ```python
   ds = _HFDataset(_table).with_format("numpy")   # map-style 유지, no .to_iterable_dataset()
   ```

2. **TrainingArguments (run_stage1, run_stage2)**: `max_steps` 인자 제거하고 `num_train_epochs` 추가.
   ```python
   training_args = TrainingArguments(
       ...
       num_train_epochs=cfg["stage1_epochs"],   # max_steps 라인 제거
       ...
   )
   ```

3. **§18 흔적 전부 롤백**:
   - `calculate_max_steps` 시그니처 원복 (`num_data_splits` 파라미터 제거), raw audio fallback 용으로만 유지
   - main() 의 `_max_steps_user_set` 마커 제거
   - `stage1_per_device_batch_size` / `stage2_per_device_batch_size` cfg key 제거
   - 단일 `per_device_train_batch_size` cfg key 만 남김

4. **TrainingArguments hardcoded bs 제거**: `cfg.get("per_device_train_batch_size", 10)` 로 통일 (config.py 의 단일 source of truth).

**HF Trainer 의 자동 step 계산 (검증됨)**

각 rank 가 자기 shard 1개 (`_table.num_rows = 1500`) 들고 있고, `StreamingShardedTrainer.get_train_dataloader` 가 DistributedSampler 를 안 끼우므로 HF Trainer 는 dataset 을 per-rank 단위로 처리:

```
len(train_dataloader) = ceil(1500 / per_device_batch_size) = ceil(1500/10) = 150
num_update_steps_per_epoch = 150 // grad_accum_steps = 150 // 4 = 37
```

→ 한 split call (run_stage1 1회) 당 **37 optimizer step** 자동 계산. 5 split call 합 = **185 step** = 1 full epoch over 60,000 bins (8 ranks × 5 shards × 1500).

**검증** (실제 진행 중인 run, bs=10, num_data_splits=5)

| step | loss | grad | epoch (= step / 37.5) |
|---|---|---|---|
| 1 | 7.06 | 8.44 | 0.0267 |
| 2 | 7.09 | 8.88 | 0.0533 |
| 3 | 5.97 | 1.45 | 0.0800 |
| 4 | 5.12 | 1.58 | 0.1067 |
| 5 | 4.81 | 0.70 | 0.1333 |
| 6 | 4.52 | 0.84 | 0.1600 |

epoch fraction 이 1/37.5 ≈ 0.0267 step 단위로 증가 → step 계산 정확. step time **~80 s/step** (bs=10 기준).

**예상 wall time**: 185 step × 80 s = **~4.1h** (1 epoch 전체 데이터).

**복잡도 비교**

| 항목 | §18 (롤백) | §19 중간 (롤백) | §19 최종 (지금) |
|---|---|---|---|
| 변경 cfg key | 4개 | 1개 (`per_device_train_batch_size`) | 1개 |
| `cfg["max_steps"]` 설정 | 함 (calculate_max_steps) | 함 (`_table.num_rows`) | **안 함** (HF Trainer 가 자동) |
| TrainingArguments | `max_steps=...` | `max_steps=...` | `num_train_epochs=...` |
| Dataset 형식 | iterable | iterable | **map-style** (`.to_iterable_dataset()` 호출 안 함) |
| ground truth | audio_hours 추정 | `_table.num_rows` 직접 | HF Trainer 가 `len(dataset)` 자동 |
| dead code | 다수 | 다수 | 최소 (raw audio fallback 만 유지) |

**사이드 노트**:
- run_stage1/run_stage2 안의 `if cfg.get("max_steps") is None: ... calculate_max_steps()` 분기는 dead code 로 남아 있음 (cfg["max_steps"] 에 값 박히지만 TrainingArguments 가 무시). raw audio 모드 호환 위해 유지. 청소 가능하면 별도 §에서.
- per-dataset packed 분기 (line 1402, 1464) 는 여전히 `.to_iterable_dataset()` 사용 + IterableDataset → 만약 mixed 모드 안 쓰면 max_steps 박아야 함. 현재 mixed 만 쓰니 무관.

