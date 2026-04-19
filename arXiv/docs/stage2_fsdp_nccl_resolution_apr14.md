# Stage 2 FSDP + NCCL Deadlock Resolution (2026-04-14 오전) (Archive)

이 파일은 `dataloader_trials.md` 에서 2026-04-14 에 분리된 archive 입니다.
주제: `_greedy_batch` dtype mismatch fix → FSDP in-loop eval NCCL deadlock 반복 → WerCallback FSDP guard 로 완전 비활성화 → activation_checkpointing FSDP 이관 → FSDP 실효성 분석 (2B LoRA 한정 ~2.5 GB 절약).
원문 섹션: §28 ~ §31 (Stage 2 FSDP 안정화 + 분석)

---

## §28. Stage 2 `_greedy_batch` dtype mismatch (2026-04-14) {#section-28}

*(archived from dataloader_trials.md §28, 2026-04-14)*

**증상**: Stage 2 첫 eval_steps=35 trigger → `_greedy_batch` → `model.llm.generate` → `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16` at Qwen3.5 `linear_attn.in_proj_qkv`.

**원인**: `_get_audio_embeds_sequential` 내부의 `proj_dtype = self.projector[0].weight.dtype` 가 FSDP `FlatParameter` 하에서 fp32 로 보이는 경로 존재. 결과적으로 audio_embeds 가 fp32 로 반환됨 → LLM bf16 linear layer 와 불일치.

정확한 원인 추적은 실패했지만, LoRA wrapping + `use_orig_params=True` + FSDP 상호작용 하에서 param dtype attribute 조회가 master-weight fp32 를 반환하는 것으로 추정.

**수정** ([train_pipeline_override.py:1662-1678](../train_pipeline_override.py#L1662-L1678)):

`_greedy_batch` 에서 LLM 입력 직전 명시적 bf16 cast:
```python
audio_embeds = audio_embeds.to(torch.bfloat16)
corr_embeds  = embed(corr_tensor).expand(B, -1, -1).to(torch.bfloat16)
inputs_embeds = torch.cat([audio_embeds, corr_embeds], dim=1).to(torch.bfloat16)
```

`embed.weight.dtype` 읽는 것 대신 `torch.bfloat16` 리터럴 사용. 같은 FSDP 경로 이슈로 dtype attribute 자체가 신뢰 못 함.

**결과**: §29 NCCL deadlock 으로 인해 해당 경로 (`_greedy_batch` 호출) 자체가 비활성화되어 실제 재현/검증 불가. 코드는 남겨두되 `WerCallback` FSDP guard 가 먼저 걸림.

**관련**: [train_pipeline_errors.md §6.19](train_pipeline_errors.md#619-stage-2-_greedy_batch-dtype-mismatch-fp32-audio_embeds--bf16-qwen35-linear)

---

## §29. Stage 2 FSDP in-loop eval NCCL deadlock → WerCallback 완전 비활성화 (2026-04-14) {#section-29}

*(archived from dataloader_trials.md §29, 2026-04-14)*

**증상**: §28 dtype fix 적용 후 재시작 → step 35 eval trigger 에서 **NCCL timeout 600s** 두 번 연속 발생.

**Trial 1 (v5, `summon_full_params rank0_only=True`)**:
```
[rank0]: stuck at _ALLGATHER_BASE (SeqNum=7300)
[rank1-7]: stuck at ALLREDUCE(1,1) (SeqNum=7300)
```
`rank0_only=True` 는 gather 완료 후 non-zero rank 가 param discard → rank 0 가 `generate()` 호출 시 FSDP hook 이 재-allgather 시도 → 다른 rank 는 이미 param 버린 상태라 참여 불가 → 데드락.

**Trial 2 (v6, `rank0_only=False, recurse=True`)**:
```
[rank0]: stuck at _ALLGATHER_BASE (summon_full_params gather 자체)
[rank1-7]: stuck at ALLREDUCE(1,1) (barrier 급 tiny op)
```
이론적으로 `rank0_only=False` 면 모든 rank 가 full params 를 들고 있어 rank 0 의 forward 가 추가 gather 필요 없어야 함. 하지만 관찰 결과 **rank 1~7 은 summon_full_params 에 아예 진입하지 않음** (SeqNum 7300 이 summon 의 allgather 가 아닌 작은 ALLREDUCE 임). HF Trainer 가 callback 을 rank 별로 다른 경로로 부르거나 `model` kwarg 가 분기하는 것으로 추정.

근본 원인 추적 실패. FSDP + PEFT + HF TrainerCallback 조합은 HF 내부 수명주기 가정과 충돌.

**최종 수정** ([train_pipeline_override.py:1905-1915](../train_pipeline_override.py#L1905-L1915)):

`WerCallback.on_step_end` 첫 줄에 FSDP guard 추가 → 즉시 return:
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
- HF Trainer `save_strategy="steps" save_steps=35` 로 체크포인트는 저장 (tied embedding → `StreamingShardedTrainer._save` override, §22)
- WER/val_loss 는 저장된 체크포인트에서 **offline 계산** 으로 전환 (별도 스크립트 필요)
- Stage 1 (DDP) 은 `use_fsdp=False` 이므로 guard 에 안 걸림 → 기존 경로 그대로 유지

**교훈**:
- FSDP + PEFT 조합에서 in-loop `summon_full_params` + rank-0 eval 패턴은 HF Trainer 와 호환 불가. 전체 rank forward 또는 completely detached eval worker 필요
- Stage 2 in-loop metric 이 필요하면 offline 평가로 분리하거나 eval 전용 distributed forward 경로 따로 구현

**관련**: [train_pipeline_errors.md §6.20](train_pipeline_errors.md#620-stage-2-wercallback-fsdp-in-loop-eval-nccl-deadlock-완전-비활성화로-우회)

---

## §30. `activation_checkpointing` FSDP 이관 (2026-04-14) {#section-30}

*(archived from dataloader_trials.md §30, 2026-04-14)*

**배경**: FSDP full_shard 하에서 HF TrainingArguments `gradient_checkpointing=True` 를 그대로 쓰면 backward 에 **redundant AllGather** 가 삽입됨 (HF issue #30404).

FSDP 는 forward 시 param allgather → layer 계산 → reshard 순서로 동작. HF 의 grad checkpoint 는 backward 시 recomputation 을 위해 intermediate activation 을 재계산하는데, 이때 FSDP 가 이미 reshard 한 params 를 **다시 allgather** 해야 함. 이 중복 gather 가 step time 을 늘림.

**해결**: FSDP 에게 checkpoint 책임을 위임 — `fsdp_config.activation_checkpointing=True` 설정. FSDP 가 wrapped layer 내부에서 activation checkpointing 을 직접 관리하므로 backward 시 추가 gather 없이 동작.

**수정** ([train_pipeline_override.py:2200](../train_pipeline_override.py#L2200), [:2227](../train_pipeline_override.py#L2227)):

```python
training_args = TrainingArguments(
    ...
    # FSDP 활성 시 HF grad checkpoint off, fsdp_config 에서 처리.
    gradient_checkpointing=not cfg.get("use_fsdp", True),
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ...
    **({
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "fsdp_transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"],
            "fsdp_use_orig_params": True,
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "limit_all_gathers": True,
            "fsdp_ignored_modules": ["encoder"],
            "activation_checkpointing": True,   # ← 추가
        },
    } if cfg.get("use_fsdp", True) else {}),
)
```

**검증**: v5/v6 모두 step 35 eval crash 로 steady-state step time 수치 못 얻음. §29 수정 후 v7+ 재시작에서 측정 예정.

**관련 경고**: HF Trainer 가 v5 로그 중 아래 메시지를 여러 번 출력했음 (수정 전):
```
When using FSDP full shard, instead of using `gradient_checkpointing` in TrainingArguments,
please use `activation_checkpointing` in `fsdp_config`. The former introduces a redundant
AllGather operation in backward pass. Reference: https://github.com/huggingface/transformers/issues/30404
```

---

## §31. FSDP 실효성 분석 — Qwen3.5-2B + LoRA 한정 (2026-04-14) {#section-31}

*(archived from dataloader_trials.md §31, 2026-04-14)*

**문제 제기**: "FSDP 효과 있는가?" 의문. 80 GB 카드에 Qwen3.5-**2B** + LoRA + FSDP full_shard 조합에서 VRAM peak 27~33 GB 관찰. 이 중 FSDP 덕분에 절약된 부분이 얼마인가?

**계산** (2B LoRA, 8 GPU):

| 메모리 항목 | DDP + LoRA | FSDP + LoRA | 차이 |
|-------------|------------|-------------|------|
| LLM 파라미터 | 4 GB (full) | 0.5 GB (1/8 샤드) | **-3.5 GB** |
| LoRA grads | ~10 MB | ~10 MB | 0 (LoRA 만 학습) |
| LoRA optimizer state | ~40 MB | ~40 MB | 0 |
| Activation (packed 16384 seq) | ~25 GB | ~25 GB | 0 |
| All-gather buffer (FSDP) | 0 | ~1 GB | **+1 GB** |
| **순 절약** | | | **~2.5 GB** |

- LoRA 는 원래도 optimizer/grad 가 tiny (trainable param 1.47M / 1.88B = 0.08%) → FSDP 의 optimizer 샤딩 이득은 사실상 없음
- activation 은 FSDP 와 무관 (gradient_checkpointing + packed seq 가 여전히 지배적)
- all-gather buffer 로 오히려 약간 추가 소비

**결론**:
- **2B LoRA 에서는 FSDP 이득 ~2.5 GB**, 80 GB 카드의 여유를 고려하면 실효성 낮음
- 대신 forward 시 all-gather 통신 overhead 가 step time 을 조금 늘림 (정량 측정 TBD)
- **4B 이상 모델**에서는 LLM 파라미터 자체가 ~8 GB → FSDP 샤딩 이득이 ~7 GB 로 커져 필수
- **2B 는 `--no-fsdp` + DDP 로 throughput 비교 벤치** 할 가치 있음 (별도 trial)

**현재 운영 결정**: Stage 2 코드 안정성 우선. FSDP 유지. §29 수정 후 정상 진행 확인되면 2B DDP 벤치 스케줄.

**관련**:
- [packing_fa2_liger_fsdp.md](packing_fa2_liger_fsdp.md) §FSDP 섹션 갱신됨
- [train_pipeline_errors.md §6.20](train_pipeline_errors.md#620-stage-2-wercallback-fsdp-in-loop-eval-nccl-deadlock-완전-비활성화로-우회) — FSDP in-loop eval 호환성 이슈

---

