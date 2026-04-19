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
| **Liger kernel** (partial RoPE fix) | OFF → ON (RMSNorm+SwiGLU+fused CE+partial RoPE) | pre-packed + fla 환경에서 **4패치 모두 정상 동작** (loss=6.3, grad=11.9). RoPE nan 버그 → partial rotary fix로 해결 | ✅ 검증됨 |
| **Stage 1 FSDP** (`--fsdp-stage1`) | DDP → FSDP full_shard | DDP보다 **2.2배 느림** (~119s vs ~55s/step), 메모리 절감 없음 | ✅ 재검증됨 (무효) |
| **streaming=False** (RAM 로딩) | HF streaming → pyarrow.ipc.read_all() + map(num_proc=16) | ls100/ls360/ls500 성공, **MLS에서 OOM kill** (cgroup 제한 + cp 동시 실행) | ✅ 검증됨 (대용량 불가) |
| **Pre-packed Arrow** | on-the-fly pack → 사전 packed Arrow 직접 로드 | map 2단계 완전 스킵 (processor+packer 0s). DataLoader 병목 해소 | ✅ 검증됨 |
| **gradient_checkpointing=False** (Stage 1) | True → False | cutoff_len=16384에서 **OOM** (79.3GB/GPU 소진) | ✅ 검증됨 (불가, checkpointing 필수) |
| **flash-linear-attention 0.4.2** | 미설치 → fla 0.4.2 + causal_conv1d 1.6.1 | **~55s → ~40s/step (27% 가속), ~64GB → ~11GB GPU mem (82% 절감)**. fla fast path 활성화 | ✅ 검증됨 |
| **num_data_splits** (bins 분할) | 1 → 2 | 데이터를 N등분하여 split마다 1/N 로드. mls+gs 포함 시 iteration 메모리 1/2 (peak은 그대로 — §11.3) | ✅ 검증됨 (single packed file 기준) |
| **num_data_splits** (shard 단위, mixed) | bins-slice → shard-slice (§13) | mixed packing(rank당 5 shard)에서 peak 메모리 1/N로 실질 감소. `num_data_splits=5` 시 rank당 ~22 GB / 노드 ~176 GB | ✅ 검증됨 (mixed run 정상 동작) |
| **HF Dataset numpy format** (§15) | list mode → `with_format("numpy")` | nested `list<float32>` decode 메모리 6x 감소 (~300 MB/bin → ~50 MB/bin). C-speed 디코딩 | ✅ 검증됨 (RSS proc당 ~62 GB → §17 적용 후 ~33 GB) |
| **TORCH_WARM_POOL=0** (§16) | unset (=1) → 0 | torch._inductor compile worker pool eager spawn 차단. 256 subprocess × ~750 MB ≈ 192 GB CPU RAM 절약 | ✅ 검증됨 (`pgrep compile_worker` = 0) |
| **mixed shuffle 제거** (§17) | `.shuffle(buffer_size=1000)` → 제거 | HF IterableDataset.shuffle unbounded RAM 누수 회피. 첫 step 즉시 도달. pack_arrow.py 가 이미 셔플 | ✅ 검증됨 (mixed loaded 후 ~30s 내 첫 step) |
| **max_steps 12.5x 부풀림 fix** (§18) | hardcoded bs=2, num_data_splits 미반영, cross-stage cfg 누수 | 문제는 진단 정확, 그러나 fix가 max_steps 를 또 박는 형태라 user 가 거부 | ❌ 전체 롤백 |
| **§19 max_steps 폐기** | TrainingArguments `max_steps` 제거 → `num_train_epochs`, mixed 분기 `.to_iterable_dataset()` 제거 → map-style | HF Trainer 가 `len(dataset)`/`num_train_epochs` 로 step 수 자동. max_steps 어디에도 안 박힘. wall time **~4.1h** (1 epoch, 185 step × 80 s) | ✅ step 1~6 검증 (epoch fraction = 1/37.5 일치) |
| **bs 증가** (1 → 4 → 5 → 10) | per_device_train_batch_size 단계적 상향 | bs=5 → 30 s/step, bs=10 → 80 s/step (선형 스케일). vram bs=5 22GB → bs=10 peak 78GB (spike 관찰). wall time 거의 동일 (linear scaling). **bs=10 에서 OOM risk → §20 에서 bs=8 로 다운** | ✅ 검증됨, bs=8 로 고정 |
| **§20 word-aug offline pre-pack** | 문장 레벨 만 → 문장 + 단어 레벨 혼합 | `_table.num_rows 슬라이싱` 방식으로 재인코딩 없이 word sub-clip 추가. bins/rank 7.5k → 26k (3.5x). ls/gs/vp 적용 (mls 는 id 불일치로 whole-only). wall time 예상 ~13.7h | 🔄 첫 38 step 돌았으나 §21 OOM 크래시 → 패치 후 재시작 중 |
| **§21 `_project_precomputed` clip-by-clip loop** | `(N, T_enc_max, C)` pad-stack → clip 단위 for-loop | word-aug bin 의 peak 메모리 = 가장 긴 clip 1개 기준으로 축소. `(N, T_proj_max, llm_dim)` LN 17.47 GiB 한방 할당 제거 | ✅ 검증됨 (peak 26 GB, 이전 52 GB) |
| **§22 Stage 1 save_strategy="no"** | HF Trainer 내장 save 끔 | Qwen3.5 tied embedding (lm_head.weight ↔ embed_tokens.weight) 이 safetensors 공유 메모리 거부 → crash. Stage 1 은 projector 만 학습하므로 full-model save 자체가 불필요. WerCallback 만 projector 저장 | ✅ 검증됨 |
| **§22 Stage 2 save_safetensors=False** | safetensors → torch pickle (.bin) | 동일 tied embedding 이슈. Stage 2 는 LoRA adapter + projector 저장해야 함. `.bin` 은 shared storage 그대로 보존 가능 | 🔄 적용됨 (Stage 2 검증 미수행) |
| **§23 wandb report_to="none" + CumulativeWandbCallback** | HF Trainer 내장 WandbCallback → 커스텀 on_log 후크 | num_data_splits>1 에서 split 경계마다 HF Trainer 가 `wandb.finish()` 호출 → 다음 split 이 새 run 생성 (project="huggingface" 기본값). CumulativeWandbCallback 이 `step` 인자 생략으로 auto-increment → 하나의 run 에 연속 기록 | ✅ 검증됨 (split 1/8 run 이어붙음) |
| **§24 `evaluate_val_loss_precomputed`** | val_loss=nan fallback → encoder forward 로 직접 계산 | raw waveform → encoder → projector → LLM 경로를 eval 시점에 batch=1 로 돌려 val_loss 계산. training forward path 동일, 수치 직접 비교 가능 | ✅ 검증됨 (val_loss=4.09~4.19 찍힘) |
| **§25 per_device_train_batch_size 12** | 8 → 12 | §21 clip-by-clip 이후 vram peak 대폭 축소 → bs 상향 여유. bs=12 측정: vram 21.6~26.8 GB, step time ~43 s, split 당 step 수 110 → 74 (32% 감소) | ✅ 검증됨 |
| **§25 eval/save_steps 35** | 50 → 35 | bs=12 에서 split 당 73~74 step 이라 eval_steps=50 이면 split 당 eval 1 회만 trigger. 35 로 조정하면 2 회 (step 35, 70) | ✅ 검증됨 |
| **§26 Split 8 NCCL timeout (rank 간 bin 불균형)** | runtime rank-balance + offline rebalance | pack_arrow.py 가 byte 기준 20 GB shard rotation → rank 별 마지막 shard 크기 편차 큼 (split 8: 1374~1851 bin). HF Trainer 가 rank 별 다른 step 수로 진행 → SeqNum=3952 BROADCAST timeout → NCCL deadlock | ✅ 런타임 fix 적용, offline rebalance 돌리는 중 |
| **§27 word-aug projector collapse (별도 문서)** | `docs/word_aug_collapse.md` | HYP 가 audio 무관한 legal-speak 로 수렴. loss plateau 3.39~3.55, WER val 100~151% 진동. 가설: word-aug 믹스 비율 (sample 기준 90%+) 이 projector 를 word-level 분류기로 편향 | 🔄 진단 단계, 검증 실험 대기 |
| **§28 Stage 2 `_greedy_batch` bf16 명시 cast** | `audio_embeds.to(torch.bfloat16)` + `corr_embeds` + `inputs_embeds` 3 개소 | FSDP FlatParameter 하에서 `projector[0].weight.dtype` 이 fp32 로 보이는 케이스 → audio_embeds 가 fp32 로 빠져 Qwen3.5 `linear_attn.in_proj_qkv`(bf16) 호출 시 F.linear mismatch. §6.19 참조 | ✅ 적용됨 (해당 코드 경로는 §29 로 비활성화되어 재현 불가) |
| **§29 Stage 2 FSDP WerCallback 완전 우회** | `if cfg.get("use_fsdp"): return` guard | `summon_full_params` rank0_only=True/False 양쪽 다 NCCL deadlock 600s (SeqNum=7300, rank0 `_ALLGATHER_BASE` vs rank1~7 `ALLREDUCE`). HF Trainer + FSDP + PEFT 조합에서 callback rank 경로 분기 원인 미확인. in-loop eval 포기, WER/val_loss offline 계산으로 전환 | ✅ 적용됨 (v7+ 재시작으로 검증 예정) |
| **§30 `activation_checkpointing` FSDP 이관** | TrainingArguments `gradient_checkpointing=True` → `fsdp_config.activation_checkpointing=True` | HF Trainer grad checkpoint 는 backward 에 redundant AllGather 삽입 (HF #30404). FSDP 가 wrapped layer 에 직접 checkpoint 걸도록 위임. `gradient_checkpointing=not use_fsdp` 로 조건부 | 🔄 적용됨 (v5/v6 는 eval crash 로 step time 수치 못 얻음) |
| **§31 FSDP 실효성 분석 (2B LoRA 한정)** | 현황 유지, 문서화만 | Qwen3.5-2B + LoRA: FSDP 샤딩 이득은 LLM param 4GB → 0.5GB/rank (~3.5GB 절약). LoRA optimizer/grad 는 원래도 tiny. activation 이 peak 지배 (gradient_checkpointing 에도 불구 27 GB). 80GB 카드에선 실질 이득 ~2~3GB. **4B 이상 모델에선 필수**, 2B 는 DDP+--no-fsdp 가 throughput 유리할 수 있음 (벤치 필요) | 📝 문서화 (벤치마크 대기) |
| **packing_cutoff_len=65536** | 16384 → 65536 | DataLoader worker Bus error (shared memory). `num_workers=0`으로 우회 | 🔄 검증 중 |
| **dataloader_num_workers** | 1 → 0 (precomputed) | pre-packed 모드에서 worker 불필요 (데이터 이미 메모리). worker=1은 오버헤드만 추가. Bus error 해결 | ✅ 변경됨 |
| **cross-dataset mixed packing** | 데이터셋별 패킹 → 합쳐서 셔플 후 패킹 | bin 내 여러 데이터셋 샘플 혼합. step당 gradient 편향 방지 | 🔄 구현 완료, 검증 예정 |

**결론**: on-the-fly packing 모드에서 GPU utilization의 핵심 레버는 `packing_cutoff_len` 단 하나.
n_shards/num_workers/max_batch_tokens/packing_bucket_size/process_batch_size 모두 영향 없거나 역효과.
CPU 처리 단위(bucket_size, process_batch_size)를 늘리면 GPU 굶김(starving)이 발생해 오히려 악화됨.
- `process_batch_size`: 32(최적) < 64(역효과, avg ~90s, GPU util ~70%) < 128(역효과, avg ~90s, 불규칙)
- **Pre-packed Arrow** 모드로 전환 시 DataLoader 병목 자체가 해소됨 ([arXiv §10](../arXiv/docs/early_packing_tuning_apr06-07.md#section-10) 참조)

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
| 04-10 23:08 | — | **pre-packed** | 0 | — | 16384 | — | **~30s** (loss=1059, nan) | 100% | ~11GB | fla 0.4.2 + torch 2.6.0 + triton 3.2.0. 속도 향상이지만 **수치 발산**. GPU mem 비정상적으로 낮음 |
| 04-11 02:45 | — | **pre-packed** | 0 | — | 16384 | — | **~22s** (loss=780→0, nan) | 100% | ~13GB | fla 0.3.2 + torch 2.6.0 + split 1/2. 역시 수치 발산 |
| 04-11 13:28 | — | **pre-packed** | 0 | — | 16384 | — | **~37s** (loss=4→0, nan) | — | — | fla 0.3.2 + torch 2.6.0, ls100만. 역시 수치 발산 |
| 04-11 14:11 | — | **pre-packed** | 0 | — | 65536 | — | Bus error | — | — | cutoff_len=65536, worker=1 → shared memory Bus error. worker=0으로 해결 |
| 04-11 14:24 | — | **pre-packed** | 0 | — | 65536 | — | **~41s** (loss=1→0, nan) | 100% | ~13GB | fla 0.3.2 + no-liger + cutoff 65536. 역시 수치 발산 |
| 04-11 ~15:00 | — | **pre-packed** | 0 | — | 65536 | — | 에러 | — | — | fla 0.3.2 + no-liger + transformers 패치. exitcode 1 (원인 불명, cutoff 65536 관련 가능) |
| 04-11 ~15:10 | — | **pre-packed** | 0 | — | 16384 | — | **~167s** (loss=5.787, grad=9.375) | — | — | fla 0.3.2 + **no-liger** + torch 2.6.0. 정상 동작 확인. liger가 nan 원인임을 확정 |
| 04-11 ~15:20 | — | **pre-packed** | 0 | — | 16384 | — | ~170s (loss=10.77, grad=nan) | — | — | fla 0.3.2 + **liger ON** (전체). nan 재현 → liger 원인 확정 |
| 04-11 ~15:30 | — | **pre-packed** | 0 | — | 16384 | — | ~170s (loss=8.28, grad=nan) | — | — | liger RoPE+RMSNorm+SwiGLU (CE 끔). 여전히 nan |
| 04-11 ~15:40 | — | **pre-packed** | 0 | — | 16384 | — | ~170s (loss=4.35, grad=nan) | — | — | liger **RoPE만** 켬. nan → RoPE가 범인 확정 |
| 04-11 ~15:50 | — | **pre-packed** | 0 | — | 16384 | — | **~166s** (loss=6.82, grad=13.6) | — | — | liger RMSNorm+SwiGLU+CE (RoPE 끔). **정상** → RoPE 단독 범인 |
| 04-11 ~16:00 | — | **pre-packed** | 0 | — | 16384 | — | **~167s** (loss=6.97, grad=11.2) | 100% | ~8GB | **liger 4개 모두 ON** (partial RoPE fix). ✅ 정상 동작 확정 |
| 04-11 17:36 | — | **pre-packed** | 0 | — | 16384 | — | **~40s** (step5 steady) | 100% | **~11GB** | **풀 스택: fla 0.4.2 + causal_conv1d + liger 4패치 + ls+vp**. fast_path_warn=0. 이전 55s→40s (27%↑), 64GB→11GB (82%↓) |
| 04-12 09:08 | 5tgzian3 (early kill) | **mixed** | 0 | — | 16384 | — | ~30s (bs=5) | 100% | ~22GB | **mixed packed + bs=5** + §13~§17 fix. shuffle 제거 후 첫 step 즉시 도달. loss 6.26→4.71 (5 step) |
| 04-12 09:36 | — (early kill) | **mixed** | 0 | — | 16384 | — | ~80s (bs=10) | 100% | 47~67GB | **bs=5→10**. step time 정비례. vram peak 67/80 GB. 8 step 후 user 가 §18/§19 단순화 위해 중단 |
| 04-12 10:07~ (진행중) | — | **mixed** | 0 | — | 16384 | — | **~80s** (bs=10, steady) | 100% | 49~67GB | §19 최종: max_steps 폐기, num_train_epochs=1 + map-style Dataset. step 6 기준 loss 4.22 / grad 0.61. 1 epoch ≈ 185 step → **~4.1h wall time** 예상 |
| 04-12 ~ 04-13 01:17 | — (pack) | **mixed + word-aug** | — | — | 16384 | — | — | — | — | §20 word-aug offline pre-pack. 8 rank 병렬 ~11h wall. bins/rank 7.5k → 26.3k (3.5x), shards 5 → 8, disk 936G → 1.3T. ls/gs/vp 적용 (mls 는 uid 불일치로 whole-only) |
| 04-13 01:44~02:17 | — | **mixed+word-aug** | 0 | — | 16384 | — | **~34s/it** (38/110 step, epoch 0.35) | 99~100% | 29~52GB steady, **peak 75GB rank2** | §20 word-aug 첫 학습 run. loss 7.32 → 3.58 (38 step), grad 0.04~0.12 안정. Step 38 에서 **CUDA OOM (rank2, 17.47 GiB 할당 실패)** → §21 |
| 04-13 02:20~06:xx | — (kill) | **mixed+word-aug** | 0 | — | 16384 | — | **~34 s/it** (bs=8, steady) | 99~100% | 17~20 GB | §21 clip-by-clip 패치 후 재시작. peak 52 GB → **20 GB 로 안정**. loss 7.4 → 3.52 (split 1 끝), WER val step 70 =105.6%. 수 차례 wandb project/save_strategy/val_loss 패치 위해 재시작 반복 |
| 04-13 09:00~ (bs=12) | — | **mixed+word-aug** | 0 | — | 16384 | — | **~43 s/it** (bs=12, steady) | 87~100% | 21.6~26.8 GB | §25 bs 8→12, split 당 74 step. loss 7.18→3.55 (split 1), WER val step 35/70 = 104/106% |
| 04-13 10:45~16:51 | — (crash) | **mixed+word-aug** | 0 | — | 16384 | — | **~43 s/it** (split 1~7 완주, split 8 29/38) | ~100% | ~27 GB | **§23 wandb cumulative** + **§24 val_loss precomputed** + **§25 bs=12 + eval_steps=35** 전부 적용. split 7 까지 정상, split 8 step 29 → 30 에서 **NCCL timeout** (rank 6/7 shard 1374/1395 vs rank 4 1851 bin 불균형). loss 3.371 (split 7 끝), WER val 151% / train 98.6%. HYP collapse 관찰 (§27) |
| 04-13 21:30~ (rebalance) | — | — | — | — | — | — | — | — | — | 기존 mixed shard 에 `pack_arrow.py --rebalance` standalone 실행. 64 shard 메타데이터 스캔 → 최소 bin 수로 trim. §26 범주 |
| 04-14 02:07~02:43 (v2, crash) | fb_dacvae_2b_S2_0414_0207 | **Stage 2 FSDP+LoRA** | 0 | — | 16384 | — | **~60s/it** (bs=12, step 1~34) | 0~100% 불규칙 | 26.6~32.9 GB | Stage 2 첫 시도. §22 regression (`_save` override 복구), §6.5 `save_safetensors` kwarg 제거됨 해결 후 정상 start. step 1~34 loss 3.39 → 3.37 steady, LR warmup 1.7e-5 → 2e-5 peak (warmup_ratio=0.1, cosine). **step 35 eval trigger 에서 dtype mismatch crash** (§28/§6.19). FSDP (Stage 2 ✓) 활성 |
| 04-14 02:50~03:07 (v5, crash) | (동일 run 재시작 안됨) | **Stage 2 FSDP+LoRA** | 0 | — | 16384 | — | — (step 1 전 crash) | — | — | §28/§6.19 dtype fix 적용 후 v4 kill + activation_checkpointing FSDP 이관 (§30) 적용하여 v5 재시작. 초기 로딩 후 step 35 eval 에서 **NCCL timeout 600s** (rank 0 `_ALLGATHER_BASE`, rank 1~7 ALLREDUCE(1,1), `summon_full_params rank0_only=True` 데드락 — §29 첫 번째 경로) |
| 04-14 04:57~05:07 (v6, crash) | (동일) | **Stage 2 FSDP+LoRA** | 0 | — | 16384 | — | — | — | — | `summon_full_params rank0_only=False` 로 변경 후 재시작. **동일 NCCL timeout 600s 재발** — 이번에도 rank 0 만 summon 진입, rank 1~7 은 barrier 에 남음. HF Trainer callback path 분기 원인 미확인. §29 두 번째 경로 → `WerCallback.on_step_end` 에서 FSDP 활성 시 즉시 return 으로 완전 우회 (§29 최종) |

## Archive

실험 및 해결 완료된 섹션은 [arXiv/docs/](../arXiv/docs/) 에 주제별로 분리:

- [early_packing_tuning_apr06-07.md](../arXiv/docs/early_packing_tuning_apr06-07.md) — **§1 ~ §10** (2026-04-06 ~ 04-11): 초기 packing/bucket/cutoff_len tuning, n_shards, num_workers, Stage 1 FSDP 검증 (무효), Liger Kernel 효과 측정, flash-linear-attention 환경 구성, offline pre-packing 구현
- [mixed_packing_memory_evolution_apr10-12.md](../arXiv/docs/mixed_packing_memory_evolution_apr10-12.md) — **§11 ~ §19** (2026-04-10 ~ 04-12): streaming RAM 로딩 시도 OOM, Data splits 도입, shard 단위 split, shuffle 축소/제거, HF Dataset numpy format, TORCH_WARM_POOL, max_steps 계산 변천 (rollback), `num_train_epochs` + map-style Dataset 최종
- [stage2_fsdp_nccl_resolution_apr14.md](../arXiv/docs/stage2_fsdp_nccl_resolution_apr14.md) — **§28 ~ §31** (2026-04-14 오전): `_greedy_batch` dtype mismatch fix, FSDP in-loop eval NCCL deadlock → WerCallback 완전 비활성화, `activation_checkpointing` FSDP 이관, FSDP 실효성 분석 (2B LoRA 한정)

현재 문서는 **§20 ~ §27 word-aug 파이프라인** + **§32 ~ §35 최신 fix** 에 집중합니다. `(§N)` 참조가 archive 대상 섹션인 경우 위 파일들에서 `#section-N` 앵커로 찾을 수 있습니다.

## 핵심 관찰

### 20. Word-aug offline pre-pack — 문장 + 단어 레벨 혼합 (2026-04-12~13)

**목적**: 문장 레벨 학습만으로는 encoder 가 word-level 단위를 효과적으로 식별하지 못할 가능성. 단어 boundary 에서 audio clip 을 잘라 "이 audio 조각은 어떤 단어?" 학습 task 를 추가. 전체 utterance + word sub-clip 을 같은 bin 에 섞어 pack.

**구현 방침**: **pre-encoded features 의 frame 단위 슬라이싱** (no re-encoding). alignment 의 word start/end(초) × fps → encoder output frame 범위 → features 2D tensor 슬라이스. 기존 `/mnt/ddn/users/jos/precomputed/fb_dacvae/{ds}/rank{N}.arrow` 그대로 재활용, 재인코딩 없음.

**Alignment 데이터**: `/mnt/ddn/users/sehyun/CACHE/word_aligned_data/` (사용자 제공). `_ALIGNMENT_BASE` 경로 업데이트.

**코드 변경**:

1. **`_ALIGNMENT_BASE`** ([train_pipeline_override.py:638](../train_pipeline_override.py#L638)): `/mnt/tmp/cache/word_alignments_merged` → `/mnt/ddn/users/sehyun/CACHE/word_aligned_data`
2. **`make_precomputed_processor_fn`** ([train_pipeline_override.py:1229](../train_pipeline_override.py#L1229)): `alignment_lookup=None` 인자 추가. 각 row 마다 원본 utterance emit + alignment 존재 시 word sub-clip 추가 emit.
   - fps = `encoder.tgt_sr / encoder.hop` (fb_dacvae = 44100/512 ≈ 86.13)
   - `min_word_frames = max(2, int(0.1 * fps))` → ≥ 100ms word 만 사용
   - word slice: `np.asarray(flat_feats).reshape(T_enc, out_dim)[start_frame:end_frame]`
   - emit format: `[audio_pad × T_proj] + tokenize(word.lower()) + [EOS]`
3. **`pack_arrow.py`**: `--word-aug` flag 추가, `MergedAlignmentLookup` 빌드 후 `make_precomputed_processor_fn` 에 전달.
4. **`run_pack.sh`**: `--word-aug` 전달.
5. **AlignmentLookup 의 logger 호출**: `logger.info` (accelerate logger, accelerate state 필요) → `print()` 로 교체. pack_arrow.py (plain python) 에서 crash 방지.

**Alignment utterance_id 매칭**

| dataset | precompute id (precompute_features.py) | alignment arrow id (sehyun) | 매칭 |
|---|---|---|---|
| ls100/ls360/ls500 | LibriSpeech `id` (`103-1240-0000`) | 동일 | ✅ |
| gs | `segment_id` (`AUD0000000003_S0000001`) | 동일 | ✅ |
| vp | `audio_id` (`20090113-0900-PLENARY...`) | 동일 | ✅ |
| **mls** | `original_path.removesuffix(".opus")` → **URL (`http://.../file.mp3`)** | `10001_8844_000003` | ❌ 형식 불일치 |

→ mls 는 alignment 매칭 실패로 word-aug 적용 안 됨, 원본 utterance 만 통과 (fallthrough 안전). ls/gs/vp 만 word sub-clip 추가.

**Pack 실행 결과 (2026-04-12 13:52 ~ 04-13 00:57, ~11h wall)**

| 지표 | Before (whole only) | After (whole + word) | 비율 |
|---|---|---|---|
| **Bins / rank** | 7,500 | 26,322 | **3.5x** |
| **Shards / rank** | 5 | 8 | 1.6x (20 GB cap 분할) |
| **Disk total** | 936 GB | 1.3 TB | 1.4x |
| **Avg bin size** | ~156 MB | ~51 MB | 0.33x (word sub-clip 이 작아 dense packing) |

ls100 per-rank sample count: 4,077 → 38,352 (**9.4x**) → word-aug 1 utterance 당 평균 8-9 sub-clip 확인.

**Config 업데이트**:
- `num_data_splits`: 5 → **8** (shard 수와 일치)
- `per_device_train_batch_size`: 10 → **8** (bs=10 에서 vram spike 78/80 GB 확인, 안전 마진 확보)

**학습 예상 wall time (bs=8, num_train_epochs=1, 8 splits)**

```
Split 0~6: 3,500 bins × 7 = 24,500 bins → ~109 step / split × 7 = 763 step
Split 7  : 1,822 bins → ~57 step
Total    : ~820 optimizer step / epoch
```

step time ~60 s (bs=8 측정) → 820 × 60 ≈ **~13.7h wall time** (1 epoch). 측정 첫 5 step 기준 loss 7.32 → 5.12 감소, grad norm 안정 (10.7 → 1.30).

**Trade-off**:
- 장점: encoder 의 word-level discrimination 학습 가능. dense bin packing 으로 disk 증가율 낮음 (3.5x bins → 1.4x disk).
- 단점: mls (21k h 중 10k h 차지) 는 utterance_id 형식 불일치로 word-aug 미적용 → librispeech + gs + vp 만 혜택.
- 주의: 1 epoch 학습 시간 4.1h → 13.7h (3.3x) 증가. stage1_epochs 유지하려면 시간 trade-off 감수.

**검증 상태**: 첫 step loss 7.32, grad_norm 10.7 (첫 step 전형) → step 38 까지 loss 3.58 로 안정 감소, grad 0.04~0.12. NaN/Inf 없음. 그러나 step 38 에서 §21 CUDA OOM 발생 → 패치 후 재시작.

### 21. `_project_precomputed` clip-by-clip loop (2026-04-13, word-aug OOM 근본 수정)

**증상**: §20 word-aug 학습 중 step 38 (epoch 0.35) 에서 `rank2` CUDA OOM.

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 17.47 GiB.
GPU 2 has a total capacity of 79.33 GiB of which 3.74 GiB is free.
  File "train_pipeline_override.py", line 484, in forward
    audio_embeds = self._project_precomputed(...)
  File "train_pipeline_override.py", line 390, in _project_precomputed
    embeds = self.proj_norm(proj.transpose(1, 2))   # (N, T_proj_max, llm_dim)
```

step 8~37 까지 steady-state ~52 GB / GPU 로 돌다가 특정 bin 에서 갑자기 LN 한방이 17.47 GiB 할당 실패.

**원인**: `_project_precomputed` 가 bin 안의 모든 clip 을 `(N, T_enc_max, out_dim)` 로 zero-pad stack → projector+LN 에 배치로 먹임. word-aug 이전엔 bin 안 clip 이 whole utterance 로 길이가 대체로 균질해서 괜찮았음. word-aug 이후엔:

- bin 안 clip 수 `N` 이 100+ 로 급증 (word clip 이 짧아 packing 밀도 상승)
- whole utterance clip 1개라도 섞이면 `T_enc_max` 가 그 clip 기준으로 맞춰짐
- 모든 짧은 word clip 도 `T_enc_max` 길이로 pad 되어 `(N, T_enc_max, C)` 텐서 팽창
- `(N, T_proj_max, llm_dim)` = LayerNorm 입력 텐서가 수십 GB

**수정** ([train_pipeline_override.py:379-403](../train_pipeline_override.py#L379-L403)):

```python
def _project_precomputed(self, enc_feats, enc_feat_lengths):
    proj_dtype = self.projector[0].weight.dtype
    llm_dtype  = self.llm.get_input_embeddings().weight.dtype
    clips = []
    for i in range(enc_feats.shape[0]):
        T_enc_valid = int(enc_feat_lengths[i].item())
        if T_enc_valid == 0:
            continue
        feats_i = enc_feats[i:i+1, :T_enc_valid, :].to(proj_dtype)     # (1, T_enc_valid, C)
        proj_i  = self.projector(feats_i.transpose(1, 2))              # (1, llm_dim, T_proj_valid)
        emb_i   = self.proj_norm(proj_i.transpose(1, 2)).squeeze(0)    # (T_proj_valid, llm_dim)
        clips.append(emb_i.to(llm_dtype))
    return torch.cat(clips, dim=0).unsqueeze(0)
```

핵심:
- pad-stack 자체를 제거. clip 하나씩 유효 길이로만 projector+LN 통과
- peak 메모리 = 가장 긴 clip 1개 기준 (bin 전체 padding 낭비 0)
- 출력 shape `(1, total_T_proj, llm_dim)` 는 기존과 동일 → LLM forward 쪽 변경 0
- packed bin 구조/schema 불변. pre-packed Arrow 재생성 필요 없음

**속도 영향 (예상)**:
- Python loop + kernel launch overhead 추가 (bin 당 clip ~100 → 100 × ~50 μs ≈ 5 ms / bin)
- step 당 8 bin 기준 추가 ~40 ms → 34 s/step 대비 0.1% 수준
- projector 는 Conv1d(k=5)×3 + LN 으로 연산량이 LLM forward 대비 <1% → 실질 slowdown 미미

**기존 §6.11/6.12 와의 관계** (`docs/train_pipeline_errors.md`):
- §6.11: raw audio 경로에서 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 로 단편화 완화
- §6.12: raw audio 경로 `_get_audio_embeds_batched` 의 mixed-length padding 폭발 — sequential 전환 제안
- §21: **precomputed 경로** `_project_precomputed` 의 동일 구조적 문제. §6.12 와 같은 clip-by-clip 해법을 pre-encoded 경로에도 적용

**검증 상태**: 패치 적용 후 재시작. 첫 step 도달 및 steady-state 메모리 관찰 중.
→ **검증 완료**: vram peak 52 GB → 20 GB 로 축소 (bs=8). 이후 bs=12 로 올려도 26.8 GB peak 유지 (§25).

### 22. HF Trainer save_strategy → "no" (Stage 1), save_safetensors=False (Stage 2)

**증상**: Stage 1 bs=12 run step 100 (첫 `save_steps=100` trigger) 에서 즉시 RuntimeError.

```
RuntimeError:
    Some tensors share memory, this will lead to duplicate memory on disk and potential
    differences when loading them again: [{'llm.lm_head.weight', 'llm.model.embed_tokens.weight'}].
```

**원인**:
- Qwen3.5-2B 는 **tied embedding** 채택 — `llm.model.embed_tokens.weight` 와 `llm.lm_head.weight` 가
  동일한 torch storage 를 가리킴 (메모리 1× 공유)
- `PreTrainedModel.save_pretrained` 은 이를 인지하고 하나만 저장 후 load 시점에 다시 tie 하지만,
  우리 `AudioQwen` 은 `nn.Module` 직접 상속이라 `save_pretrained` 없음
- HF Trainer 가 `save_model → _save → safetensors.torch.save_file(state_dict)` fallback 경로로 감
- safetensors 는 shared storage 를 가진 두 이름을 거부함 → RuntimeError

**수정** ([train_pipeline_override.py](../train_pipeline_override.py)):

- **Stage 1 (run_stage1)**: `save_strategy="no"`. Stage 1 은 LLM frozen / projector 만 학습이라
  full-model snapshot 이 애초에 불필요. WerCallback 이 eval 시점에 projector state 만 직접 torch.save.
- **Stage 2 (run_stage2)**: `save_safetensors=False` 추가. Stage 2 는 LoRA adapter + projector 를
  저장해야 하므로 save 자체는 활성. safetensors 대신 torch pickle (.bin) 로 저장하면 shared storage
  문제 없음. 파일 사이즈 살짝 크고 로드 약간 느려지지만 내부 workflow 에선 무관.

**대안 검토**: AudioQwen 을 `PreTrainedModel` 상속으로 리팩터. Config class 정의 + `__init__(config)` 시그니처 + `_tied_weights_keys` + `from_pretrained` round-trip 등 대규모 수술 필요 → 현재 단계에선 과함.

### 23. Cumulative wandb logging (num_data_splits>1 에서 split 경계 wandb run 연속성)

**증상**: num_data_splits=8 학습에서 split 1 은 우리 project `Qwen3.5-2b-ASR-fb_dacvae` 에 정상 기록, split 2 부터 **자동으로 `huggingface` 기본 project 에 "pleasant-dew-1" 같은 랜덤 이름으로 떨어짐**.

**원인**: 두 단계 문제가 겹침.

1. HF Trainer 내장 `WandbCallback.on_train_end` 가 split 1 종료 시 `wandb.finish()` 호출 → wandb.run = None
2. Split 2 의 새 Trainer 가 `report_to="wandb"` 로 시작 → `WandbCallback.setup` 이 `wandb.init()` 호출. 이때 환경변수 `WANDB_PROJECT` 가 없어서 HF 기본값 `"huggingface"` 로 떨어짐 + 랜덤 run name
3. 설령 `WANDB_PROJECT` 를 env 로 고정해도, wandb 내부 step counter 는 monotonic 이라
   split 2 의 step 0 로그가 wandb 에 거부됨 (split 1 이 이미 step 100+ 까지 갔으므로)
   → 차트에 split 2+ 가 안 그려짐

**수정**:

1. **`CumulativeWandbCallback`** 신규 클래스 ([train_pipeline_override.py:1808-1825](../train_pipeline_override.py#L1808)):
   ```python
   class CumulativeWandbCallback(TrainerCallback):
       def on_log(self, args, state, control, logs=None, **kwargs):
           if logs is None or wandb.run is None:
               return
           payload = {("train/" + k if not k.startswith(("train/","eval/","val/")) else k): v
                      for k, v in logs.items() if isinstance(v, (int, float))}
           if payload:
               wandb.log(payload)   # step 인자 생략 → auto-increment (monotonic 유지)
   ```
2. **TrainingArguments**: Stage 1 / Stage 2 모두 `report_to="none"` (HF Trainer 내장 WandbCallback 완전 disable)
3. **`wandb.finish()` 제거**: `run_stage1` / `run_stage2` 끝에 있던 `wandb.finish()` 삭제. main() 전체 종료 시점에만 호출
4. **Env 안전장치**: main() 에서 `WANDB_PROJECT` / `WANDB_RUN_ID` / `WANDB_RESUME=allow` 셋팅 — 어떤 코드 경로에서도 우리 project/run 에 연결되게

**핵심 원칙**: wandb run 은 main() 전체 동안 살아있음. split 경계에서 절대 finish/reinit 없음. wandb 내부 step 은 auto-increment 로 계속 단조 증가.

### 24. `evaluate_val_loss_precomputed` (precomputed 경로에서 val_loss 정상 계산)

**배경**: §6.15 에서 precomputed 모드일 때 `val_loss = float("nan")` 로 스킵했었음 (collator 호환 안 맞아서). 그러나 val_loss 는 WER 이 noisy 한 학습 초기 구간에서 안정적인 learning signal 을 제공 → 살려두는 게 유용.

**구현** ([train_pipeline_override.py:1759-1808](../train_pipeline_override.py#L1759)):

```python
@torch.no_grad()
def evaluate_val_loss_precomputed(raw_model, val_dataset, cfg, device, max_samples=100):
    """precomputed 모드 전용 — encoder 를 eval 시점에 직접 호출해 features 생성.
    training forward path (encoder → projector → proj_norm → LLM) 과 동일, batch=1 loop.
    """
    raw_model.eval()
    tokenizer = raw_model.tokenizer
    enc_dtype = next(raw_model.encoder.parameters()).dtype
    audio_pad_id = cfg.get("audio_pad_token_id", 151655)
    total_loss, total_tokens = 0.0, 0

    for i in range(min(len(val_dataset), max_samples)):
        waveform, text = val_dataset[i]
        text_ids = tokenizer.encode(text.lower().strip(), add_special_tokens=False)
        if not text_ids: continue
        wav = waveform.unsqueeze(0).to(device)
        lengths = torch.tensor([wav.shape[-1]], device=device)
        feats, _ = raw_model.encoder(wav.to(enc_dtype), lengths)   # (1, T_enc, C)
        T_enc = feats.shape[1]
        if T_enc == 0: continue
        T_proj = math.ceil(T_enc / raw_model._proj_stride)
        input_ids = [audio_pad_id] * T_proj + text_ids + [tokenizer.eos_token_id]
        labels    = [IGNORE_INDEX]  * T_proj + text_ids + [tokenizer.eos_token_id]
        out = raw_model(
            input_ids=torch.tensor([input_ids], device=device),
            labels=torch.tensor([labels], device=device),
            precomputed_enc_feats=feats.float(),
            audio_lengths=torch.tensor([T_enc], device=device),
        )
        if out.loss is not None and torch.isfinite(out.loss):
            n_valid = (labels_t != IGNORE_INDEX).sum().item()
            total_loss  += out.loss.item() * n_valid
            total_tokens += n_valid
    raw_model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")
```

WerCallback.on_step_end 에서 `cfg.get("precomputed_dir")` 분기해서 이 함수 사용.

**검증**: split 1 step 70 eval 에서 `val_loss=4.1928` 찍힘 (이전에는 nan). training loss (~3.5) 와 비교 가능한 수치.

### 25. per_device_train_batch_size 8 → 12, eval/save_steps 50 → 35

**배경**: §21 clip-by-clip 패치 이후 vram peak 이 대폭 축소돼서 bs 상향 여유 발생.

**관찰 (bs=8)**:
- vram peak steady: 17~20 GB / 80 GB (§21 이전 52 GB → 축소)
- GPU util: 97~100%
- step time: ~34 s

**bs=12 측정**:
- vram peak: 21.6 ~ **26.8 GB** (step 26 에서 first spike, 이후 steady)
- step time: **~43 s** (bs=8 대비 +26%, 선형 스케일 근접)
- split 당 step 수: 73~74 (bs=8 의 110 대비 **32% 감소**)
- → **split 경계 warmup 재시작 횟수 (8회) × 초반 불안정 3~5 step 의 비용 축소**

**eval/save_steps 조정**: bs=12 에서 split 당 74 step 이라 `eval_steps=50` 이면 split 당 1 회만 trigger (split 당 eval 듬성). 35 로 내리면 step 35 + step 70 = **split 당 2 회** eval.

- split 마다 `run_stage1` 이 새 Trainer 생성 → `state.global_step` 이 0 으로 리셋 → eval_steps 는 **split 내부 기준**임에 주의
- 수정 이력: 100 → 50 → **35** (2026-04-13)

**추가 관찰 (bs=12 훈련 trajectory)**:
- split 1: loss 7.18 → 3.55 (step 73)
- split 2: Adam reset 튐 (step 2: loss 5.61) → 빠르게 복구 → 3.56~3.46 plateau
- split 7 끝: loss 3.371, val_loss 4.09
- **loss 가 3.37 근처에서 plateau** — §27 word-aug projector collapse 증상

### 26. NCCL timeout 과 rank 간 bin 개수 불균형

**증상**: split 8 (마지막 split) step 30 에서 NCCL timeout 연쇄.

```
[rank 7] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=3952, OpType=BROADCAST, ...) ran for 600066 ms before timing out.
[rank 4/2/5] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=3953, OpType=ALLREDUCE, ...)
```

**로그 증거**:
```
[rank 6] 1/8 shard(s) (1374 bins) [split 8/8]
[rank 7] 1/8 shard(s) (1395 bins)
[rank 5] 1/8 shard(s) (1697 bins)
[rank 3] 1/8 shard(s) (1661 bins)
[rank 1] 1/8 shard(s) (1796 bins)
[rank 2] 1/8 shard(s) (1802 bins)
[rank 0] 1/8 shard(s) (1822 bins)
[rank 4] 1/8 shard(s) (1851 bins)   ← 최대-최소 차 477 bin
```

**원인**:
- `pack_arrow.py pack_rank_mixed` 가 **byte 기준 shard rotation** 사용 (SHARD_MAX_BYTES=20 GB)
- rank 별로 packed bin 의 평균 size 가 조금씩 다름 → 20 GB 를 채우는 데 필요한 bin 수 편차
- 처음 7 개 shard 는 거의 균등 (20 GB 채운 뒤 rotate) 이지만 **마지막 shard 는 residual** → rank 별 크기 심각하게 편차
- training 시 num_data_splits=8 + shard-level split → split 8 이 각 rank 의 마지막 shard 1 개씩 로드 → rank 간 bin 수 불균형 발생

- `StreamingShardedTrainer` 는 DistributedSampler 우회해서 각 rank 가 자기 `len(dataset)` 기준 step 수 계산 → rank 6 (1374 bin) 은 115 step 에서 dataloader 종료, rank 4 (1851 bin) 은 155 step 까지 진행 시도 → rank 6 이 먼저 loop 빠져나가면서 collective 참여자 사라짐 → rank 4 의 allreduce 가 timeout

**수정** (2단계 방어):

1. **Runtime safety** ([train_pipeline_override.py:1408-1429](../train_pipeline_override.py#L1408)): `build_precomputed_pipeline` mixed 분기에서 `torch.distributed.all_reduce(MIN)` 으로 모든 rank 의 `_table.num_rows` 최소값을 구한 뒤 자기 shard 를 그 크기로 truncate.
   ```python
   local_n = torch.tensor([_table.num_rows], device=accelerator.device)
   dist.all_reduce(local_n, op=dist.ReduceOp.MIN)
   min_n = int(local_n.item())
   if _table.num_rows > min_n:
       _table = _table.slice(0, min_n)
   ```
   → 모든 rank 가 동일한 step 수로 돌게 보장. rank-imbalance 근본 fix.

2. **Offline rebalance** ([precompute/pack_arrow.py](../precompute/pack_arrow.py)): 기존 packed shard 에도 소급 적용 가능한 `rebalance_mixed_shards(base_dir, cutoff_len, num_ranks)` 함수 추가.
   - 각 rank 의 총 bin 수를 스캔
   - 최소값 산출, 다른 rank 들을 그 값으로 trim (뒤에서부터 shard 삭제 / 마지막 shard 일부 잘라냄)
   - CLI: `python precompute/pack_arrow.py --encoder X --rebalance` (--rank 불필요)
   - UX: `run_pack.sh --mixed` 가 packing 종료 직후 **자동 호출** → 새 pack 에선 사용자 개입 불필요

**검증 상태**: runtime fix 코드 완료, offline rebalance 실행 중 (64 shard 스캔에 ~27 분).

### 27. Word-aug projector collapse — 별도 문서 참고

HYP 가 audio 무관한 legal-speak 로 수렴, loss 3.39~3.55 plateau, WER val 100~151% 진동.
word-aug 의 과도한 믹스 비율 (sample 의 90%+ 가 short word sub-clip) 로 인한 projector 편향 가설.

자세한 증상 / 가설 / 검증 계획은 [`docs/word_aug_collapse.md`](word_aug_collapse.md) 참조.

현재 우선순위: §26 NCCL timeout 확정 수정 후 → word-aug 없는 baseline 대조 실험 예정.

---

### 32. Stage 2 LR scheduler split 경계 리셋 → warmup 영구 루프 (2026-04-14)

**증상**: Stage 2 run (`run-20260414_101342`) 로그 관찰 중 "LR 이상" 지적. WandB / output.log 에 cosine decay 가 전혀 안 찍히고 **톱니파** 로 lr 이 split 마다 0 부터 재시작.

```
# split 6/8 (full-epoch 1, 첫 split) 종료 시
{'loss': '3.373', 'learning_rate': '1.333e-05', 'epoch': '1'}
# split 4/8 (두 번째 split) 시작 시
{'loss': '3.365', 'learning_rate': '0',         'epoch': '0.0404'}
```

**원인 연쇄**:

1. `total_max_steps=925`, `warmup_ratio=0.1` → warmup = 92 steps
2. 한 split 의 step 수 ≈ `925 / (8 splits × 2 full-epochs) ≈ 58 steps` < 92
3. **한 split 이 warmup 구간보다 짧음** → 어떤 split 에서도 warmup 종료 불가
4. split 마다 scheduler 가 리셋되면서 `last_epoch=-1` 로 돌아감 → lr 0 에서 다시 linear warmup 시작
5. 결과: lr 은 0 ↔ ~1.33e-5 톱니파, peak 2e-5 미도달, cosine decay 구간 진입 불가

**리셋 원인 — HF Trainer `_inner_training_loop` 내부 동작**:

```python
# transformers/trainer.py _inner_training_loop 초반
if self._created_lr_scheduler:
    self.lr_scheduler = None
    self._created_lr_scheduler = False
self.create_optimizer_and_scheduler(num_training_steps=max_steps)
```

즉, 직전 `trainer.train()` 이 scheduler 를 생성했으면 (`_created_lr_scheduler=True`), 다음 `train()` 호출 시 **HF 가 scheduler 를 강제로 None 으로 만들고 `create_scheduler` 를 재호출** → override 의 `if self.lr_scheduler is not None: return` 가드가 무용지물.

`StreamingShardedTrainer.create_scheduler` 의 기존 구현:
```python
self._created_lr_scheduler = True   # ← 이 한 줄이 HF 의 리셋 트리거
return self.lr_scheduler
```

**수정** ([train_pipeline_override.py:1693](../train_pipeline_override.py#L1693)):

```python
# HF 가 외부에서 주입된 scheduler 로 오인하도록 False 유지 → 다음 train() 호출에서 리셋 안 함.
# 첫 split 에서 이미 existing-scheduler 가드가 있으므로 중복 생성 위험 없음.
self._created_lr_scheduler = False
return self.lr_scheduler
```

**검증 가능 시점**: 수정 적용 후 재시작한 run 에서 split 경계 ([output.log 에서 "split X/8" 로그 직후])에도 lr 이 이전 값에서 연속 증가/감소 해야 함. split 1+2 에 걸쳐 warmup 종료 (step ~92) → 이후 cosine decay 로 monotonic 감소 예상.

**관련**:
- `_total_max_steps_override` 및 기존 설계는 [train_pipeline_override.py:1662-1694](../train_pipeline_override.py#L1662-L1694) docstring 참조
- HF 의 `_created_lr_scheduler` 리셋 로직은 upstream intent 가 "다른 scheduler 타입으로 재호출 허용" 이지만, 단일 run 내 split 루프에서는 역효과

---

### 33. Split shuffle seed rank-불일치 → shard 중복/미방문 (2026-04-14)

**증상**: Stage 2 run 의 tmux 전체 scrollback 을 파싱해 iter 당 각 rank 의 실제 로드 shard 를 집계 → **같은 `trainer.train()` 호출 안에서 rank 마다 다른 shard index 를 로드** 하는 현상 확인. main() 이 announce 하는 `split order` 는 **rank 0 의 순서일 뿐** 다른 rank 는 독립적으로 shuffle.

예 (run-1, 10:15):
```
main() announce: split order: [6, 4, 5, 7, 1, 2, 3, 8]
iter 1: rank 0→6, rank 1→3, rank 2→7, rank 3→3, rank 4→7, rank 5→2, rank 6→3, rank 7→4
iter 2: rank 0→4, rank 1→5, rank 2→6, rank 3→7, rank 4→6, rank 5→5, rank 6→1, rank 7→8
iter 3: rank 0→5, rank 1→6, rank 2→1, rank 3→5, rank 4→5, rank 5→3, rank 6→6, rank 7→2
iter 4: rank 0→7, rank 1→4, rank 2→5, rank 3→6, rank 4→2, rank 5→8, rank 6→8, rank 7→7
```

rank 0 는 [6,4,5,7,…] 로 정상 순회하지만, rank 1 은 [3,5,6,4,…], rank 3 은 [3,7,5,6,…] 등 중복(rank 3 의 3→3 같은 iter) + 미방문 가능. 1 full-epoch 안에 모든 rank 가 1~8 을 정확히 1 회씩 방문하는 것을 의도했으나 깨짐.

**원인**:

```python
# 기존 코드 (train_pipeline_override.py:2698)
split_rng = _random.Random(abs(hash(run_id)) & 0xFFFFFFFF)
```

Python 3 의 `hash(str)` 는 **PYTHONHASHSEED 로 프로세스마다 salt 되어 달라짐** (security feature). `run_id = datetime.now().strftime("%m%d_%H%M")` 는 모든 rank 에서 동일한 문자열이지만 `hash("0414_1013")` 는 rank 마다 다른 int → `split_rng` seed 가 rank 마다 다름 → shuffle 결과도 rank 마다 다름.

**영향**:
- **DDP/NCCL 정합성**: 이상 없음. 모든 rank 가 `rank{rank}_sX.arrow` 에서 `rank-balance truncate` 로 동일 bin 수 (1374) 로 맞춰 돌기 때문에 step 수 동일.
- **Data coverage**: rank 마다 full-epoch 안에 특정 shard 를 중복 방문하거나 미방문할 수 있음. "epoch" 의 의미 (= 전체 데이터 1 회 통과) 가 rank 별로 다름.
- **Word-aug collapse 분석** ([word_aug_collapse.md](word_aug_collapse.md)) 의 교란 변수: 특정 rank 가 데이터 일부를 못 봤거나 몰아서 봤을 가능성 배제 못 함.

**수정** ([train_pipeline_override.py:2698](../train_pipeline_override.py#L2698)):

```python
# run_id = "MMDD_HHMM" → int 변환이 결정적. 모든 rank 동일 seed.
split_rng = _random.Random(int(run_id.replace("_", "")))
```

대안 (고려했으나 채택 안 함):
- `dist.broadcast(rank=0)` 로 shuffle 결과 전파: 네트워크 호출 추가 오버헤드, build 시점에서 broadcast 가드 필요
- `hashlib.md5(run_id.encode()).digest()` : 결정적이지만 import 추가
- 현 fix 는 1 줄 변경, 외부 의존 0

**검증**: 새 run 에서 tmux scrollback 파싱 → iter 마다 모든 rank 가 **동일한** shard index 를 로드하는지 확인. (main() 이 announce 한 split order 와 rank 0 이외 rank 의 실제 로드 index 가 일치해야 함)

**관련**:
- 기존 의도: [train_pipeline_override.py:2731-2733](../train_pipeline_override.py#L2731) 주석 "Outer loop: full data epoch × stage2_epochs, Inner: num_data_splits 를 shuffled 순서로 1 회씩"
- 이 불변식은 rank 0 에만 성립했음을 §33 에서 확인

---

### 34. Stage 2 매 split 종료 수동 `save_model` (2026-04-14)

**문제**: Stage 2 에서 저장이 한 번도 안 일어남. config:
```python
"save_steps": 60,   # HF TrainingArguments
```

그런데 log 관찰 결과 Stage 2 진행 중 `{s2_output_dir}/checkpoint-*/` 디렉터리가 단 하나도 생기지 않음.

**원인**:

1. **split 당 step 수 < save_steps**: `total_max_steps=925`, `num_data_splits=8`, `stage2_epochs=2` → 16 iter × 58 step = 925. 한 split 의 optimizer step 이 **약 58** 로 `save_steps=60` 미만 → HF 의 `state.global_step % 60 == 0` 체크 불가능.
2. **`state.global_step` 리셋**: 각 `trainer.train()` 호출마다 HF 가 `state = TrainerState()` 로 재초기화 → global_step 은 항상 0 에서 시작해 58 에서 종료. 누적되지 않아 save_steps 조건 영원히 미충족.
3. **WerCallback 우회 불가**: [arXiv §29](../arXiv/docs/stage2_fsdp_nccl_resolution_apr14.md#section-29) 에서 FSDP 조합 NCCL deadlock 때문에 `WerCallback.on_step_end` 에 `if cfg.use_fsdp: return` guard 가 걸려있음 → WerCallback 의 projector save 경로도 죽어있음.
4. **main() 끝 save 만**: `s2_trainer.save_model(s2_output_dir)` 가 2 full-epoch 종료 후 **딱 1 회** 만 호출. 중간 이력 없음.

**영향**:
- 중간 checkpoint 가 없어 projector collapse (§27) 분석 시 "언제부터 무너졌는지" 추적 불가
- 장시간 training 중 crash 시 처음부터 재시작해야 함
- word-aug 효과를 timestep 별로 비교하는 실험 불가

**수정** ([train_pipeline_override.py:2746-2752](../train_pipeline_override.py#L2746-L2752)):

main() 의 split 루프 안에 helper 정의 후 각 `trainer.train()` 호출 직후 수동 save:

```python
def _save_split(fe_idx: int, split_idx_1based: int):
    ckpt_dir = os.path.join(
        s2_output_dir, f"checkpoint_fe{fe_idx}_split{split_idx_1based}"
    )
    logger.info(f"  → saving split checkpoint: {os.path.basename(ckpt_dir)}",
                main_process_only=True)
    s2_trainer.save_model(ckpt_dir)   # 모든 rank 호출 필수 (FSDP state_dict gather)

s2_trainer.train()
_save_split(1, first_split_idx + 1)
for split_idx in first_order[1:]:
    ...
    s2_trainer.train()
    _save_split(1, split_idx + 1)
    ...
```

**명명 규칙**: `checkpoint_fe{N}_split{M}` — N 은 full-epoch 1-based index, M 은 split 1-based index. HF 기본의 `checkpoint-{global_step}` 은 매 split 마다 동일 값 (58) 이라 **충돌/덮어쓰기 발생**. 명시적 rename 으로 우회.

**저장 경로 구조**:
```
{model_cache_dir}/fb_dacvae/s2_outputs_{run_id}/
    checkpoint_fe1_split8/    ← iter 1 (full-epoch 1, main() 순서의 첫 split)
    checkpoint_fe1_split3/    ← iter 2
    ...
    checkpoint_fe1_split5/    ← iter 8
    checkpoint_fe2_split?/    ← iter 9 (full-epoch 2 시작)
    ...
    checkpoint_fe2_split?/    ← iter 16
    (s2_output_dir 바로 밑)   ← main() 끝 최종 save
```

**검증 계획**: 첫 split 종료 (~30 min 후) 시 `→ saving split checkpoint: checkpoint_fe1_split8` 로그 + 해당 디렉터리 생성 확인. tied embedding shared storage (§22) 는 `StreamingShardedTrainer._save` override 에서 처리되므로 save 경로 문제 없음.

**관련**:
- [train_pipeline_errors.md §6.16](train_pipeline_errors.md#616-qwen35-tied-embedding-shared-storage-safetensors-거부) — tied embedding save
- [dataloader_trials.md §22](./dataloader_trials.md#22-stage-2-qwen35-tied-embedding-safetensors-공유-메모리-거부) — `_save` override 경로

---

### 35. Stage 2 split 시작 시 rank/GPU/shard 배정 표 로그 (2026-04-14)

**동기**: §33 이후 모든 rank 가 동일 shard index 를 로드하는지 **로그 차원에서 명시적으로 보이게** 만들기. 이전에는 각 rank 의 "[rank N] Mixed pre-packed (…) [split M/8]" 로그를 grep 해서 조합해야 확인 가능했음.

**구현** ([train_pipeline_override.py:2756-2773](../train_pipeline_override.py#L2756-L2773)):

```python
def _print_assignment(fe_idx: int, split_idx_0based: int):
    if not accelerator.is_main_process:
        return
    ws = accelerator.num_processes
    lines = [
        f"┌──────┬──────┬──────────────────────┐",
        f"│ GPU  │ rank │ shard file           │",
        f"├──────┼──────┼──────────────────────┤",
    ]
    for r in range(ws):
        shard_file = f"rank{r}_s{split_idx_0based}.arrow"
        lines.append(f"│ {r:>4} │ {r:>4} │ {shard_file:<20} │")
    lines.append(f"└──────┴──────┴──────────────────────┘")
    header = f" fe{fe_idx}/split{split_idx_0based+1} — rank/GPU/shard 배정"
    logger.info("\n" + header + "\n" + "\n".join(lines), main_process_only=True)
```

각 split 의 `_build_stage2_dataset(split_idx)` 호출 직전에 `_print_assignment(fe, split_idx)` 호출. 예상 출력:

```
 fe1/split8 — rank/GPU/shard 배정
┌──────┬──────┬──────────────────────┐
│ GPU  │ rank │ shard file           │
├──────┼──────┼──────────────────────┤
│    0 │    0 │ rank0_s7.arrow       │
│    1 │    1 │ rank1_s7.arrow       │
│    2 │    2 │ rank2_s7.arrow       │
│    3 │    3 │ rank3_s7.arrow       │
│    4 │    4 │ rank4_s7.arrow       │
│    5 │    5 │ rank5_s7.arrow       │
│    6 │    6 │ rank6_s7.arrow       │
│    7 │    7 │ rank7_s7.arrow       │
└──────┴──────┴──────────────────────┘
```

**GPU ↔ rank 매핑 가정**: accelerate DDP 기본 환경에서 `local_rank == CUDA device index` (즉 rank i = CUDA:i). `CUDA_VISIBLE_DEVICES` 가 설정되지 않은 표준 환경 기준. 향후 rank ↔ device 매핑이 달라지면 `torch.cuda.current_device()` 를 읽어 표시하도록 수정 가능.

**관련**:
- §33 — deterministic seed 로 모든 rank 가 동일 split_idx 를 갖게 된 이후 의미 있는 기능
- 이전 (§33 전): 모든 rank 가 다른 shard 를 로드해서 표 형태로 요약 불가능했음

---

## 핵심 변수 설명

| 변수 | 위치 | 역할 | 현재값 |
|------|------|------|--------|
| `n_shards` | `shard_arrow.py --n-shards` | `load_dataset` num_shards → worker 상한 결정 | 8 |
| `dataloader_num_workers` | `train_pipeline_override.py` | DataLoader 병렬 worker 수 | **0** (precomputed) / 8 (raw audio). pre-packed은 메모리 데이터라 worker 불필요, worker=1은 Bus error+오버헤드 |
| `packing_cutoff_len` | `config.py` | packed bin 최대 토큰 수 (GPU util의 핵심 레버) | 16384 (현재 최적, wall ~44h) |
| `packing_bucket_size` | `config.py` | knapsack packer greedy 탐색 버킷 크기 (fill ratio vs CPU latency trade-off) | **200** (최적 확인) |
| `process_batch_size` | `config.py` | 토크나이징 배치 크기 | **32** (64·128 역효과 확인, 32가 최적) |
| `gradient_accumulation_steps` | `train_pipeline_override.py` | micro-step 수 (업데이트 1회당) | 4 (검증 예정) |
| `per_device_train_batch_size` | `config.py` (단일 source of truth) | GPU당 bin 수. 양 stage 공통 cfg key. step time 정비례, throughput 무관 (§19) | **12** (§21 clip-by-clip 이후 vram peak 대폭 축소 → §25 에서 8→12 상향. peak 26.8 GB) |
| `num_data_splits` | `config.py` | mixed packing 시 shard 단위 split (§13). 1=전체, N=shard를 N그룹으로 | **8** (word-aug mixed 8-shard 기준, §20) |
| `num_train_epochs` (TrainingArguments) | `train_pipeline_override.py` | HF Trainer 가 `len(dataset) × num_train_epochs / (bs × grad_accum)` 으로 step 자동 계산. `cfg["stage1_epochs"]` / `cfg["stage2_epochs"]` 를 그대로 전달 | **stage1=1, stage2=2** (config.py 기본). 1 split call = num_train_epochs × 1 shard |
| `eval_steps` / `save_steps` | `config.py` | HF Trainer eval / WerCallback 주기 (optimizer step 단위). split 마다 리셋되므로 split 내부 기준 | **35 / 35** (§25. bs=12/split 당 74 step → 35 로 2 회 trigger) |
| `save_strategy` (Stage 1 TrainingArguments) | `train_pipeline_override.py` | HF Trainer 내장 save 경로 | **`"no"`** (§22. Qwen tied embedding safetensors crash 회피 + Stage 1 은 projector 만 학습이라 full-model save 불필요) |
| `save_safetensors` (Stage 2 TrainingArguments) | `train_pipeline_override.py` | safetensors 포맷 사용 여부 | **`False`** (§22. torch pickle .bin 사용 → tied embedding shared storage 허용) |
| `report_to` (양 stage TrainingArguments) | `train_pipeline_override.py` | HF Trainer 내장 integration callbacks | **`"none"`** (§23. 내장 WandbCallback 이 split 경계마다 wandb.finish() 호출 + step 리셋 → `CumulativeWandbCallback` 으로 대체) |
| `_ALIGNMENT_BASE` | `train_pipeline_override.py:638` | word-aug alignment arrow 루트. dataset 별 `{base}/{librispeech,mls,gigaspeech,voxpopuli}/*.arrow` 형식 | **`/mnt/ddn/users/sehyun/CACHE/word_aligned_data`** (§20) |

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
| `config.py` | `num_data_splits` 추가 (기본 1). 대용량 데이터셋 OOM 방지용 데이터 N등분 로드 |
| `train_pipeline_override.py` | `build_precomputed_pipeline`: `data_split`/`num_data_splits` 파라미터. main: split 루프 + gc |
| `train_pipeline_override.py` | `dataloader_num_workers`: precomputed 모드 1 → **0** (Bus error 해결, worker 오버헤드 제거) |
| `precompute/pack_arrow.py` | `--mixed` 모드 추가: cross-dataset packing + shard (~20GB/shard) |
| `train_pipeline_override.py` | `build_precomputed_pipeline`: mixed packed shard 자동 탐색 (`mixed/packed_{cutoff}/rank{N}_s*.arrow`) |
| `train_pipeline_override.py` | `build_precomputed_pipeline` mixed 분기: bins-level → **shard-level** split (§13). peak memory 100% → 1/N |
| `config.py` | `num_data_splits`: 2 → **5** (mixed shard 수와 일치, 1 shard / split) |
| `train_pipeline_override.py` | `shuffle(buffer_size)`: 10000 → **1000** (§14). cutoff_len=16384에서 bin 14 MB라 OOM |
| `train_pipeline_override.py` | `_HFDataset(_table).with_format("numpy")` 적용 (§15). decode 메모리 6x 감소, list-of-Python-float → ndarray(float32) |
| `run.sh` | `export TORCH_WARM_POOL=0` (§16). torch._inductor compile pool eager spawn 차단, RAM ~192 GB 절약 |
| `train_pipeline_override.py` | mixed 분기에서 `interleave_datasets` + `.shuffle()` 둘 다 제거 (§17). HF IterableDataset.shuffle 의 unbounded RAM 누수 회피, 첫 step 즉시 도달 |
| `train_pipeline_override.py` | `calculate_max_steps`: `num_data_splits` 파라미터 추가, 결과 1/N (§18). per_device_batch_size hardcoded 2 → cfg key (5) |
| `train_pipeline_override.py` | run_stage2: `cfg["max_steps"]` 재계산 (stage1 캐시 누수 방지, §18-C) |
| `train_pipeline_override.py` | `per_device_train_batch_size`: stage1 line 1832 = **5**, stage2 line 2011 = **5** (이전 1/2 → 4/4 → 5/5) |
| `train_pipeline_override.py` | §19: §18 롤백, `build_precomputed_pipeline` 가 `_table.num_rows` 로 직접 `cfg["max_steps"]` 채움 |
| `config.py` | `per_device_train_batch_size: 10` 단일 cfg key 추가 (TrainingArguments / build_precomputed_pipeline 공용) |
| `train_pipeline_override.py` | TrainingArguments line 1832/2011 `per_device_train_batch_size` → `cfg.get("per_device_train_batch_size", 10)` |
| `train_pipeline_override.py` | §19 최종: TrainingArguments `max_steps=...` 제거 → `num_train_epochs=cfg["stageN_epochs"]`. mixed 분기 `.to_iterable_dataset()` 제거 → map-style Dataset (HF Trainer 자동 step 계산) |
| `config.py` | `per_device_train_batch_size`: 1 → 4 → 5 → **10** (bs=10 vram peak 67/80 GB, 더 키우면 OOM) |
| `train_pipeline_override.py:638` | `_ALIGNMENT_BASE`: `/mnt/tmp/cache/word_alignments_merged` → `/mnt/ddn/users/sehyun/CACHE/word_aligned_data` (§20) |
| `train_pipeline_override.py` | `make_precomputed_processor_fn(alignment_lookup=None)`: word-aug 로직 (§20). features 2D 슬라이싱 + word sub-clip emit |
| `train_pipeline_override.py` | AlignmentLookup / build_alignment_lookups: `logger` → `print` (pack_arrow.py 에서 accelerate state crash 방지) |
| `precompute/pack_arrow.py` | `--word-aug` flag + MergedAlignmentLookup 빌드, processor_fn 에 전달 (§20) |
| `precompute/run_pack.sh` | `--word-aug` 플래그 forwarding |
| `config.py` | `num_data_splits`: 5 → **8** (word-aug mixed 8-shard 기준, §20) |
| `config.py` | `per_device_train_batch_size`: 10 → **8** (§20 bs=10 vram peak 78/80 GB spike 확인, 안전 마진 확보) |
| `config.py` | `eval_steps` / `save_steps`: 500 / 5000 → **100 / 100** (word-aug 로 step 수 증가, 세밀 관찰 위해) |
| `train_pipeline_override.py:379-403` | `_project_precomputed`: `(N, T_enc_max, C)` pad-stack → clip-by-clip loop (§21). word-aug bin 의 LN 17 GB 한방 할당 OOM 해결 |
| `run.sh` | accelerate launch 에 `--precomputed-dir /mnt/ddn/users/jos/precomputed` 하드코딩. 누락 시 raw audio 경로로 빠져 IterableDataset → `num_train_epochs` 자동 step 계산 실패 (§21 재시작 시 발견) |
| `train_pipeline_override.py:1790` | `WerCallback.on_step_end` 에서 precomputed 모드 감지 시 `evaluate_val_loss` 스킵 (`val_loss = nan`). eval_steps 500→100 하향으로 노출된 raw waveform ↔ precomputed collator 호환 문제 회피 (§6.15) |
| `train_pipeline_override.py` | §22 Stage 1 TrainingArguments `save_strategy="no"`: Qwen3.5 tied embedding (`lm_head.weight` ↔ `embed_tokens.weight`) 이 safetensors 공유 메모리 거부 → save 시 RuntimeError. Stage 1 은 projector 만 학습이라 full-model save 불필요 |
| `train_pipeline_override.py` | §22 Stage 2 TrainingArguments `save_safetensors=False`: torch pickle (.bin) 로 저장 → shared storage 문제 없음 |
| `train_pipeline_override.py` | §22 WerCallback `best_wer_val = cfg.get("_best_wer_val", inf)`: split 경계에서 `self.cfg["_best_wer_val"]` 에 캐리오버 → 전체 training 범위의 best 저장 |
| `train_pipeline_override.py:1808-1825` | §23 `CumulativeWandbCallback` 신규 클래스: `on_log` 에서 `wandb.log(payload)` (step 인자 생략 → auto-increment). split 경계에서 step counter 충돌 없이 단일 run 유지 |
| `train_pipeline_override.py` | §23 Stage 1/2 TrainingArguments `report_to="none"`: HF Trainer 내장 `WandbCallback` 완전 disable (split 경계에서 `wandb.finish()` 호출 방지) |
| `train_pipeline_override.py` | §23 run_stage1 / run_stage2 끝 `wandb.finish()` 제거. main() 끝에서만 호출 |
| `train_pipeline_override.py` | §23 main() 시작 env 세팅: `WANDB_PROJECT` / `WANDB_RUN_ID` / `WANDB_RESUME="allow"`. 안전망 |
| `train_pipeline_override.py:1759-1808` | §24 `evaluate_val_loss_precomputed` 신규 함수: encoder 를 eval 시점에 batch=1 직접 호출 → training forward path 와 동일한 수치 |
| `train_pipeline_override.py` | WerCallback.on_step_end 에서 precomputed 분기 시 `evaluate_val_loss_precomputed` 호출 (이전 `float("nan")` 대체) |
| `config.py` | §25 `per_device_train_batch_size`: 8 → **12** (§21 이후 vram 여유). vram peak 20 → 26.8 GB |
| `config.py` | §25 `eval_steps` / `save_steps`: 100 → 50 → **35** (bs=12/split 당 74 step 에서 split 당 2 회 trigger) |
| `train_pipeline_override.py:1408-1429` | §26 `build_precomputed_pipeline` mixed 분기 끝에 `dist.all_reduce(MIN)` 추가 → 모든 rank 의 `_table.num_rows` 최소값으로 truncate. rank-imbalance NCCL timeout 차단 |
| `precompute/pack_arrow.py` | §26 `rebalance_mixed_shards(base_dir, cutoff_len, num_ranks)` 함수: 기존 packed shard 에 소급 rebalance. 뒤쪽 shard 부터 bin 삭제 |
| `precompute/pack_arrow.py` | §26 `--rebalance` CLI flag (`--rank` optional) |
| `precompute/run_pack.sh` | §26 `--mixed` 모드에서 pack 완료 후 자동 `--rebalance` 호출 |
| `docs/word_aug_collapse.md` | §27 신규 문서: projector collapse 진단 + word-aug 가설 + 검증 계획 |
| `train_pipeline_override.py` | liger RoPE: partial rotary 대응 래퍼 (`_partial_liger_rotary_pos_emb`). Qwen3.5 `partial_rotary_factor=0.25` nan 해결 |
| `train_pipeline_override.py:1662-1678` | §28 `_greedy_batch` LLM 입력 3 개소 명시 bf16 cast (`audio_embeds` / `corr_embeds` / `inputs_embeds`). FSDP FlatParameter 의 fp32 dtype 보고 우회 (§6.19) |
| `train_pipeline_override.py:1905-1915` | §29 `WerCallback.on_step_end` 진입 직후 `if cfg.get("use_fsdp"): return` guard. FSDP + PEFT + in-loop eval NCCL deadlock 완전 우회 (§6.20) |
| `train_pipeline_override.py:2200` | §30 `gradient_checkpointing=not cfg.get("use_fsdp", True)` — FSDP 시 HF grad checkpoint off |
| `train_pipeline_override.py:2227` | §30 `fsdp_config.activation_checkpointing=True` 추가. HF #30404 redundant AllGather 회피 |
| `train_pipeline_override.py:2204` | §22 regression 복구: Stage 2 TrainingArguments 에서 `save_safetensors=False` 제거 (transformers 5.5+ 에서 kwarg 삭제). tied embedding 은 `StreamingShardedTrainer._save` override 가 처리 (§6.16) |
| `transformers/utils/import_utils.py` | `is_flash_linear_attention_available()` 패치: fla `__version__` 미정의/'N/A' 시 `return False` |
| `train_pipeline_override.py:1693` | §32 `StreamingShardedTrainer.create_scheduler`: `self._created_lr_scheduler = True` → **`False`**. HF Trainer 가 `train()` 재호출 시 scheduler 를 None 리셋하던 경로 차단 → split 경계에서 lr 이 0 으로 리셋되고 warmup 영구 반복되던 버그 fix |
| `train_pipeline_override.py:2698` | §33 `split_rng` seed: `abs(hash(run_id)) & 0xFFFFFFFF` → `int(run_id.replace("_", ""))`. Python 3 `hash(str)` 의 per-process salt 로 rank 마다 다른 shuffle 순서가 나오던 버그 fix. 모든 rank 가 main() announce 순서와 동일한 shard index 를 로드하도록 보정 |
| `train_pipeline_override.py:2746-2752` | §34 `_save_split` helper 추가 + main() split 루프 안 각 `trainer.train()` 후 수동 호출. HF `save_steps=60` 이 split 당 step (~58) 미만이라 발화 불가 + `state.global_step` 리셋으로 checkpoint 이름 충돌 → `checkpoint_fe{N}_split{M}` 명시적 rename 으로 우회 |
| `train_pipeline_override.py:2756-2773` | §35 `_print_assignment` helper 추가 + 각 split 시작 시 rank/GPU/shard 배정 ASCII 표를 rank-0 로그에 출력. §33 이후 모든 rank 가 동일 split_idx 를 갖는 것을 로그 차원에서 가시화 |
| `eval_ckpts/eval_all_ckpts.py` | §36 신규 스크립트: 한 stage2 run 디렉토리 (`.../s2_outputs_<ts>/`) 의 `checkpoint_fe{N}_split{M}` 들을 시간순 정렬해서 LibriSpeech dev set WER 측정. ckpt 별 ref/hyp pair 를 `eval_ckpts/results/<run>/<split>/<ckpt>.json` 으로 저장 + `summary.json` 로 WER 표 누적. 학습 동시 실행 가능 (단일 GPU + `--max-samples 200` 권장) |
| `eval_ckpts/build_viewer.py` | §36 신규 스크립트: `summary.json` + per-ckpt JSON 들을 읽어 self-contained `index.html` 생성. word-level edit-distance diff (substitution/deletion/insertion 색상), errors-only 필터, 오답수 기준 정렬, ckpt 별 WER best/worst 강조. 서버 없이 브라우저에서 바로 열기 가능 |
| `eval_ckpts/README.md` | §36 사용법 + GPU 동시 실행 안전성 (학습 중 ~50 GB 여유 → eval 6~10 GB 점유) 정리 |
| `eval_ckpts/results/s2_outputs_0414_1442/dev-clean/summary.json` | §36 결과: 17 ckpt × dev-clean 200 samples. fe1_split1 최저 **134.39%** → 이후 전 기간 160-163% 로 퇴화, 학습 진행으로 개선 없음. word-crop projector collapse 가설 ([word_aug_collapse.md](word_aug_collapse.md)) 확정. |
| `train_pipeline_override.py` | §37 vocab mismatch 방어 — Stage 2 inference-time AudioQwen 로드 시 `<\|audio_correspond\|>` special token 추가 + `resize_token_embeddings(len(tokenizer))` 를 train 과 동일하게 재현해야 state_dict shape 가 맞음. eval_ckpts 로더 (`load_eval_model`) 에 구현 |
| `eval_ckpts/eval_all_ckpts.py` | §37 `transcribe_train_format` — inference.py 의 legacy `[Audio:\n, audio, \nTranscript:\n]` prompt 대신 학습 포맷 (`[audio_embeds] + [<\|audio_correspond\|>]`) 로 greedy 디코딩. 기존 `inference.transcribe` 는 outdated 포맷이라 audio 조건부 출력 안 나옴 (garbage hyp 의 원인) |
| `eval_ckpts/eval_all_ckpts.py` | §37 `attn_implementation="sdpa"` 강제 (학습과 동시 실행 시 `flash_attn_2_cuda.so: undefined symbol: __libc_single_threaded` glibc 충돌 산발적 발생 회피). eval 은 단일 GPU greedy 라 flash_attn 불필요 |
| `eval_ckpts/eval_all_ckpts.py` | §37 ckpt 정렬 키를 `model.safetensors` mtime 기준으로 변경 — `split{N}` 의 N 은 §33 `split_rng` 로 셔플된 shard index 라 학습 순서와 일치하지 않음. saved_at / saved_ts 메타 추가 + viewer 에도 표시 |
| `precompute/pack_arrow.py` → `arXiv/scripts/pack_arrow.py` | §38 word-crop 구현 (utterance 당 1 + N_words bin emit) 을 쓰레기로 판단, 아카이브. [word_aug_collapse.md](word_aug_collapse.md) "2026-04-15 후속 조치" 참조 |
| `precompute/run_pack.sh` → `arXiv/scripts/run_pack.sh` | §38 pack_arrow 래퍼라 함께 아카이브 |
| `arXiv/README.md` | §38 아카이브 인덱스 신설 — `scripts/` `data/` `docs/` 별 파일 원래 역할 / 이동 사유 / 대체재 표 |
| `docs/word_alignment.md` | §38 "Interleaving vs word-crop ASR — alignment 의 올바른 사용" 섹션 + "Word-interleaving packing 계획" 섹션 추가. 안 1 (utt 당 둘 다 emit, 2×) vs **안 2 (utt-level hash split, 1:1 데이터 양 유지)** 비교 + 안 2 채택 근거 |
| `precompute/half_inlv_pack_arrow.py` | §39 신규 스크립트: hash-split 50/50 half-interleave packer. `hash_split(utt_id) = md5(utt_id)[0] & 1` → 0=sentence, 1=interleave. interleave bin 은 `[audio_pad*T_w0 + corr + text_w0 + audio_pad*T_w1 + corr + text_w1 + ... + EOS]` 포맷. alignment 없거나 word<2 면 sentence fallback. |
| `precompute/half_inlv_pack_arrow.py` | §39 단어 feature 구간을 `total_stride` 배수로 end-zero-pad → `ceil(sum T_wi / stride) == sum(T_wi_padded / stride)` 정확 매칭, `±1 frame` 방어 로직 불필요. conv kernel 경계 leak 은 수 frame 수준 무시 가능 |
| `precompute/half_inlv_pack_arrow.py` | §39 Phase 3 single-shard + Phase 4 equal-bin resplit (2-pass) — 기존 byte-size 기반 rotation (~20 GB shard) 대신 bin 수 기준 N 등분. `--shards-per-rank 8` CLI arg (기본 8, num_data_splits 와 매칭) |
| `precompute/half_inlv_pack_arrow.py` | §39 `--num-workers N` 옵션 + `_phase1_worker_init/_phase1_worker_process_batch` — pool worker 가 mmap 에서 RecordBatch 를 직접 읽음 (main 은 batch_idx 만 dispatch). pyarrow→Python 변환까지 병렬화 → **~53 sps → ~150 sps (3× 스피드업)** 측정 (ls360/ls500 기준, 4 worker/rank). 첫 구현에서 main 이 chunk 생성하고 worker 로 넘기던 방식은 main 단일 CPU 병목으로 가속 0 이었음 — batch-index dispatch 가 핵심 |
| `precompute/run_half_inlv_pack.sh` | §39 wrapper — 8 rank 병렬 실행 + `--num-workers`/`--shards-per-rank` passthrough + 완료 후 자동 rebalance. 기존 `run_pack.sh` 와 구조 동일 |
| `train_pipeline_override.py:make_precomputed_processor_fn._emit` | §40 corr token 누락 버그 fix — precomputed training path 가 `<\|audio_correspond\|>` 를 input_ids 에 안 넣어서 **training ↔ inference 포맷 불일치** (inference 의 `_greedy_batch` 는 corr 를 audio 뒤에 넣음 → training 중 gradient 받지 못한 random-init 임베딩이 inference 에 주입되어 generic LM prior 로 fallback). raw audio 경로 (`create_processor`) 에는 이미 corr 가 있었음. 한 줄 추가로 포맷 일치시킴 — `[audio_pad]*T_proj + [corr_id] + text_ids + [eos]`, labels 도 `[-100]*(T_proj+1)` |
| `precompute/half_inlv_pack_arrow.py:_emit_sentence` | §40 동일 fix — half_inlv 의 sentence fallback 분기도 corr 누락이었음. interleave 분기는 이미 corr 포함돼 있었음 |
| `train_pipeline_arrow_torch.py` (NEW from remote) | §41 `git pull` 로 추가됨 — pure torch + FSDP + sharded checkpoint. HF Trainer 우회, 477 lines. **corr 버그 처음부터 없음** (line 217-218 에서 `[pad_id]*t_audio + [corr_id] + text_ids + [eos]`). Liger 안 씀, word-aug 안 씀, WerCallback 없음, num_data_splits 없음. Stage 1/2 모두 FSDP. **그럼에도 수렴 실패 → corr 버그가 전부가 아님을 증명** |
| `docs/prompt_format_regression.md` (신규) | §41 **근본 원인 진단 문서**: 두 현행 trainer 가 공유하는 수렴 실패의 근본 원인이 **legacy 의 multi-token text prompt (`"Audio:\n"` / `"\nTranscript:\n"`) 가 단일 특수 토큰 (`<\|audio_correspond\|>`) 으로 교체된 것** 이라고 제안. word-aug / corr 버그 / Liger / FSDP 등은 모두 보조 요인. 안 A (legacy prompt 복구, **채택**) / 안 B (task instruction 추가, 기각) / 안 C (ChatML, 보류) 세 가지 fix 방향 정리. 현재 half_inlv run 이 이 가설의 control |
| `docs/word_aug_collapse.md` | §41 "후속 진단" 섹션 추가 — word-aug 만으로는 수렴 실패 전부 설명 불가 (arrow_torch 가 word-aug 없이도 실패), prompt_format_regression.md 로 연결 |
| `precompute/half_inlv_pack_arrow.py` | §42 안 A 적용 — `_emit_sentence` / `_emit_interleave` 둘 다 `<\|audio_correspond\|>` 제거, legacy `p1="Audio:\n"` / `p2="\nTranscript:\n"` 포맷으로 교체. interleave 분기는 단어별로 `p1 + audio_wi + p2 + text_wi` 반복 |
| `train_pipeline_override.py` | §42 안 A 전파 — **학습/추론 경로 5 곳** `<\|audio_correspond\|>` 제거하고 p1/p2 로 통일: (1) `make_processor_fn._build_one` (raw audio 학습), (2) `make_precomputed_processor_fn._emit` (precomputed 학습, 이미 오전에 수정 완료), (3) `_greedy_batch` (WER eval — `inputs_embeds=cat([p1_embeds, audio_embeds, p2_embeds])`), (4) `evaluate_val_loss` (raw audio val), (5) `evaluate_val_loss_precomputed` (이전엔 prompt 자체 부재 상태였음 → 이중 버그 fix) |
| `train_pipeline_override.py:308-312` | §42 `AudioQwen.__init__` 에서 `tokenizer.add_special_tokens({"<\|audio_correspond\|>"})` + `resize_token_embeddings` 삭제. vocab = Qwen 원본 그대로. **기존 ckpt 재사용 포기** (사용자 승인) |
| `docs/prompt_format_regression.md` | §42 "2026-04-15 코드 수정 요약 (안 A 적용)" 섹션 추가 — 수정 대상 표 + **packed bin pad/EOS 순서 불변식** (샘플 사이에는 pad 없음 / pad 는 bin 끝에만 / 모든 EOS 는 pad 보다 먼저) + 검증 TODO |
| `train_pipeline_override.py:Stage1SplitEndSaveCallback` | §42 신규 클래스 — Stage 1 매 split 종료 시 projector + proj_norm 만 rank0 저장 (`s1_proj_split{N}.pt`). WerCallback 이 `use_fsdp=True` guard 로 Stage 1 에서도 skip 되는 걸 우회. DDP 경로라 gather 불필요. split 카운터는 `cfg["_s1_split_counter"]` 로 split 경계 넘어서 누적 |
| `train_pipeline_override.py` | §42 Stage 1 run 1 완료 결과 — 8 split × 138 optimizer step total (wandb 기준). Loss 궤적: split 1 (3.790) → 2 (3.696) → 3 (3.711, noise) → 4 (3.651) → 5 (3.650) → 6 (3.630) → 7 (3.632) → 8 (3.612). Plateau 는 아니지만 split 당 Δ ≈ -0.02 로 매우 얕음. grad norm 0.02 대까지 하락 |
| `train_pipeline_override.py` | §42 Stage 2 run 1 완료 — 16 ckpt (fe1×8 + fe2×8). Total 267 step, split order shuffled [2,6,4,7,8,1,3,5] (epoch 1). WER 평가 대기 (`eval_ckpts` 수정 후) |
| `train_pipeline_override.py:CumulativeWandbCallback` | §42 확장 — `initial_step_offset` / `initial_epoch_offset` / `cfg` 파라미터 추가. split 경계에서 Trainer/Callback 이 재생성돼도 `cfg["_wandb_step_offset"]` / `cfg["_wandb_epoch_offset"]` 로 offset 이관해서 wandb step 연속 보장. Stage 1 resume 시 외부에서 `cfg["_wandb_step_offset"] = N` 설정 가능 |
| `train_pipeline_override.py` | §42 신규 CLI arg `--stage1-resume-step N` — N>0 일 때 최신 `s1_outputs_*/s1_proj.pt` 자동 preload + `cfg["_wandb_step_offset"] = N`. 새 `run_id` 로 새 wandb run 이 시작되지만 x축이 N+1 부터라 시각적 연속성. 추가 epoch 은 `--stage1-epochs` 와 조합 |
| `train_pipeline_override.py` | §42 Stage 1 outer loop 재구성 — 기존 `for data_split in range(0, num_data_splits)` (sequential) → `for stage1_epoch in range(stage1_epochs)` + 매 epoch `_s1_shuffled_split_order()` 호출 (Stage 2 와 동일 RNG, run_id 기반 deterministic, rank 공통). legacy `--stage1-start-split N` 은 epoch 0 skip 으로만 작동. Stage 1 다회 epoch 돌릴 때 shard 단위 셔플 확보 |
| `eval_ckpts/eval_all_ckpts.py` | §42 `transcribe_train_format` 을 p1/p2 포맷으로 업데이트 — `[audio_embeds] + [corr]` → `[p1_embeds] + [audio_embeds] + [p2_embeds]`. `load_eval_model` 의 `add_special_tokens(<\|audio_correspond\|>)` + `resize_token_embeddings` 삭제 (§42 에서 vocab 안 바꾸므로 eval 측도 불필요) |
