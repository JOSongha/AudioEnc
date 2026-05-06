# Whisper-small v4 Stage-1 — step 12850 deterministic hang (postmortem)

> 2026-05-05~06. 같은 step / 같은 elapsed / 같은 NCCL collective seq 에서 3회 연속 죽음.
> Rank 2 가 NCCL reduce-scatter 중 멈췄고, 다른 rank 들이 다음 collective 에서 600 s 대기 후 timeout.

## 1. 증상

| 시도 | 시점 (UTC) | 마지막 step | elapsed | 종료 |
|---|---|---|---|---|
| 1차 | 2026-05-05 15:28 | 12850 | 6 h 44 m 26 s | NCCL timeout (rank 2 grad AllReduce) |
| 2차 | 2026-05-05 23:08 | 12850 | 6 h 44 m 26 s | 동일 |
| 3차 | 2026-05-06 14:39 | 12850 | 6 h 44 m 24 s | 동일, `[FaultTolerantDataLoader]` 0회 발화 |

3회 모두 `WorkNCCL(SeqNum=52323)` — collective sequence 까지 동일. timing pattern 도 비트 수준 일치 (12000→12050 spike 131 s, 12750→12800 spike 112~115 s, 모든 시도). 즉 **deterministic seed → 같은 sample 순서 → 같은 step에서 같은 hang**.

## 2. 비교

| run | 12850 통과? |
|---|---|
| DAC-VAE v3 (통합 manifest) | ✅ step 94 035 까지 도달 |
| Whisper-small Stage-2 | ✅ step 35 675 까지 도달 |
| **Whisper-small v4 Stage-1** | ❌ 12850에서 결정적 종료 |
| DAC-VAE v4 (per-modality, Whisper와 동일 데이터 스킴) | 추적 불가 (디스크/wandb 캐시 정리됨) |

DAC v3 이 step 12700 통과한 점 + Stage-2 whisper-small 이 35k 통과한 점 → **v4 + Whisper-encoder + Stage-1 의 교집합 케이스**.

## 3. 진단 (라이브 캡처)

라이브 시도 시 [`/mnt/tmp/whisper_v4_diag/`](/mnt/tmp/whisper_v4_diag) 에 다음 활성화:

- `TORCH_NCCL_TRACE_BUFFER_SIZE=10485760` + `TORCH_NCCL_DUMP_ON_TIMEOUT=1` → 8 ranks × 9 MB FlightRecorder dump (`nccl/trace_{0..7}`)
- `pyspy_watcher.sh` — step 12700 부터 60 초 간격으로 `py-spy dump` 24 ranks (main 8 + 워커 16) × 19 라운드

### 3.1 NCCL 측 시그니처

```
[Rank 2] WorkNCCL(SeqNum=52323, OpType=ALLREDUCE, NumelIn=18485760, ...)  ran 600 s
[Rank 0,1,3,4,5,6,7] WorkNCCL(SeqNum=52323, OpType=ALLREDUCE, NumelIn=1, ...)  ran 600 s
last enqueued work: 52324, last completed work: 52322
```

같은 SeqNum 인데 NumelIn 이 18 M vs 1 → **ranks 가 서로 다른 collective sequence 에 있음** = 분기. 18 M = ZeRO-2 grad bucket reduce-scatter; 1 = grad-norm scalar AllReduce.

### 3.2 Python stack — rank 2 (29188), 14 : 30 : 34 (hang 5분차)

```
mask_nan_or_inf_with_val_inplace  (deepspeed/runtime/utils.py:830)
get_grad_norm_direct              (deepspeed/runtime/zero/stage_1_and_2.py:1725)
scaled_global_norm                (deepspeed/runtime/zero/stage_1_and_2.py:1807)
step                              (deepspeed/runtime/zero/stage_1_and_2.py:1866)
_take_model_step                  (deepspeed/runtime/engine.py:2281)
backward                          (accelerate/utils/deepspeed.py:281)
training_step                     (transformers/trainer.py:4071)
```

`mask_nan_or_inf_with_val_inplace` 자체는 단순 함수 (`isinf` + `isnan` + `masked_fill_`) — Python 측 hang 아님. 직전에 `dist.all_reduce(total_norm, ...)` + `total_norm.pow(...)` 가 CUDA stream 위에서 실행 중이었고, **rank 2 의 grad bucket reduce-scatter 가 NCCL 측에서 hang** → Python 다음 op 가 implicit stream sync 대기. 그래서 Python frame 은 mask_nan_or_inf 에 잡혔지만 진짜 hang은 reduce-scatter.

## 4. Root cause

1. **NaN/Inf gradient on rank 2 at step 12850**. deterministic seed 라 매번 같은 sample 이 rank 2 로 가서 같은 결과를 만듦.
2. NaN 이 들어간 grad bucket → NCCL reduce-scatter kernel 비정상 진입 → kernel 미완료.
3. 다른 ranks 는 reduce-scatter 정상 완료 → 다음 step 의 grad-norm scalar AllReduce 진입 → rank 2 가 안 와서 600 s 대기 → NCCL collective timeout → SIGABRT.

부수 문제:
- **`patch_default_pg_timeout`(`fault_tolerant.py`) 가 사실상 무동작**. `c10d._DEFAULT_PG_NCCL_TIMEOUT = new` 는 Python alias rebind 만 — C++ 측 상수는 그대로. 그리고 DeepSpeed `dp_process_group` 은 `dist.new_group()` 호출 시 `timeout` 인자를 안 넘김 → 새 PG 가 NCCL 의 hard-default 600 s 사용. yaml 의 `ddp_timeout: 180000000`, launcher 의 `patch_default_pg_timeout(seconds=3600)`, sh 의 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` 모두 NCCL collective timeout 에 영향 못 줌. 그래서 600 s 발화.
- **`FaultTolerantDataLoader` 가 못 막음** — 워커가 안 죽었기 때문 (예외 한 번도 발화 0회). 이번 hang 은 dataloader 외부.

## 5. Fix

[`src/llamafactory/data/fault_tolerant.py`](../../../src/llamafactory/data/fault_tolerant.py)

### 5.1 `patch_default_pg_timeout` 확장

`dist.init_process_group` + `dist.new_group` 둘 다 monkey-patch. 호출자가 `timeout` 안 넘기면 (= DeepSpeed dp_process_group 케이스) 자동으로 3600 s 주입. C++ 상수는 안 건드리고 Python 호출 가로채기.

### 5.2 `install_grad_nan_guard(model)`

모든 trainable param 에 `register_post_accumulate_grad_hook` 등록. backward 가 grad accumulation 마칠 때마다 `torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)`. NCCL reduce-scatter 가 NaN 안 만남.

[`src/llamafactory/train/omni/workflow.py`](../../../src/llamafactory/train/omni/workflow.py) 에서 `trainer.train()` 직전 호출. 설치 시 `[grad-nan-guard] installed N hooks` 로그.

### 5.3 환경변수 (보완)

`/mnt/tmp/whisper_v4_diag/launch_with_diag.sh` (in-tree 아님):

- `DEEPSPEED_TIMEOUT=60` (분) — DeepSpeed world-PG timeout
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` — NCCL 에러를 빠르게 surfacing

## 6. 검증

다음 12850 zone 도달 시 확인 시그널:

| 신호 | 의미 |
|---|---|
| `[grad-nan-guard] installed N hooks` (N > 0) | guard 설치 성공 |
| 12850 통과해서 13 000+ 진행 | 본질 fix 성공 |
| `nan_to_num` 동작 했어도 loss spike 미발생 | NaN frequency 가 일회성 |
| 만약 다시 hang → NCCL timeout 이 ≥ 3600 s 로 길게 | new_group monkey-patch 적용 확인 (그래도 hang 이면 다른 layer 원인) |

## 7. 진단 자료 (보존)

```
/mnt/tmp/whisper_v4_diag/
├── nccl/
│   └── trace_{0..7}                       # NCCL FlightRecorder, 9 MB × 8 ranks
├── pyspy/
│   └── dump_<TS>_pid<P>_rank<N>.txt       # 24 ranks × 19 rounds (14:21~14:39)
├── pyspy_watcher.log
└── launch_with_diag.sh                    # FlightRecorder + DEEPSPEED_TIMEOUT env
```

3차 시도의 launch log: `/mnt/tmp/Qwen3.5_whisper_small_v4_Stage1_jos/launch_20260506_074101.log`
