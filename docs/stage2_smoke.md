# Stage 2 — smoke test findings (2026-04-23)

200-step smoke run to validate the Stage 2 setup documented in
[`stage2_design.md`](stage2_design.md) before kicking off the real run.

Config: [`configs/qwen3_5ae-asr/stage2_smoke.yaml`](../configs/qwen3_5ae-asr/stage2_smoke.yaml).
7 GPUs (indices 1–7), `per_device_train_batch_size=3`, `learning_rate=1e-5`,
`warmup_steps=20`, `max_steps=200`, `save_steps=100`, DeepSpeed ZeRO-2 without fused
Adam (`ds_z2_no_fused.json` — HF AdamW handles the optimizer, sidesteps a JIT-build
failure of DeepSpeed's fused_adam on this host).

Baseline WER eval for `s2_init_42k` ran concurrently on GPU 0.

## 1. What passed

- **Training loop alive**: step 1 took 30 s (graph compile + DeepSpeed init), then
  settled at 2.0–2.7 s/it. Total 532 s wall time for 200 steps (≈ 2.66 s/step).
- **Loss**: starts ~0.30, ranges 0.30–0.55 through warmup, settles 0.31–0.41 at LR
  plateau. Average `train_loss=0.383`. Consistent with a converged-projector LoRA
  fine-tune starting point.
- **grad_norm**: 0.5–1.0 throughout, no spikes.
- **No OOM** at bs=3 with gradient checkpointing + LoRA backprop + bf16 on 80 GB GPUs.
  Production `per_device_train_batch_size: 3` is viable.
- **DeepSpeed + PEFT coexistence**: ZeRO-2 (sans fused Adam) works; HF Trainer's AdamW
  handles fp32 LoRA + fp32 projector trainable params without complaint.
- **Liger kernel + PEFT**: no conflict observed at forward or backprop.

## 2. Save-path verification (the main reason for the smoke)

`checkpoint-100/adapter_model.safetensors` (42 MB) on rank-0 inspection:

| category | key count | sample |
|---|---:|---|
| LoRA adapters | 64 | `base_model.model.model.language_model.layers.11.self_attn.k_proj.lora_A.weight` |
| Projector (full) | 39 | `base_model.model.model.audio_encoder.projector.final_norm.weight`, `input_proj.weight`, `layers.0.*`, … |
| Other | 0 | — |

**Conclusion**: `additional_target: audio_encoder.projector` routes correctly through
`peft.modules_to_save`; the projector's full-trainable weights are saved inside a
single `adapter_model.safetensors` alongside the LoRA deltas. Without that flag,
projector gradients would be silently dropped every save. The shim works as intended.

Also in the ckpt dir: `adapter_config.json` (peft metadata), `global_step100/`
(DeepSpeed ZeRO-2 optimizer states), tokenizer files, `trainer_state.json`. Base
model weights are NOT saved — that's deliberate and matches the "adapter-only save,
resolve via `--base_model /mnt/tmp/s2_init_42k` at eval time" plan (see
`stage2_design.md §7`).

## 3. Hybrid attention — surfaced, needs decision

`adapter_config.json:target_modules` lists **32 entries**, not 128 as a flat
`32 layers × 4 projs` would imply. The entries are from layers **3, 7, 11, 15, 19,
23, 27, 31** only — every fourth layer. This is because Qwen3.5AE has
`config.text_config.layer_types` interleaved:

```
["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]
```

Only full-attention layers expose `q_proj / k_proj / v_proj / o_proj`. The other 24
layers use `Qwen3_5AEGatedDeltaNet` with different naming (`in_proj_qkv`, `in_proj_z`,
`in_proj_b`, `in_proj_a`) and are currently untouched by LoRA.

**Implication**: only ≈ 25 % of the attention capacity gets LoRA-adapted. The rest of
the LLM (24 linear-attn layers + every non-attention module — MLP, norms, embeddings)
stays frozen. This is a genuine design choice, not a bug:

- **Option A — keep as-is**: narrowest LoRA footprint (~10 M adapter params for LoRA
  + ~11 M projector = 21 M trainable). Minimum risk to LLM text ability.
- **Option B — add LoRA to linear-attn `in_proj_*` too**: widen to all 32 layers. Add
  `in_proj_qkv,in_proj_z,in_proj_b,in_proj_a` to `lora_target`. Gives the model more
  capacity to shift audio-conditioned behavior through the linear-attn pathway.
  Adapter ~2× but still tiny relative to base.
- **Option C — also target MLP (`down_proj / up_proj / gate_proj`)**: closer to
  standard LoRA practice for LLM fine-tune. Recovers most of the tuning capacity.
  Higher risk to text retention without text-only mix.

Current yaml is **Option A**. If Stage 2 doesn't close the gap from 7.28 % to target,
Option B/C is the obvious next knob.

## 4. Baseline reproduction (s2_init_42k)

Ran `inference_ckpt_sweep.py --qwen3ae --eval --qwen3ae_model_dir /mnt/tmp/s2_init_42k`
on the full 2620-sample LibriSpeech test-clean:

| source | test-clean WER |
|---|---:|
| Original `checkpoint-42000` (Stage 1 sweep) | 7.28 % |
| `s2_init_42k` overlay (this smoke) | **7.26 %** |

Δ = 0.02 pp, inside the sampling noise for 2620 items. Confirms:
- Hardlinked weights intact under overlay.
- Modified `audio_encoder.py` (branch behind `if self.training and self.noise_aug_enabled`)
  is a no-op at eval (`model.eval()` → `self.training = False`).
- `config.json` additions (`audio_config.noise_aug_enabled=true`, `noise_aug_max_k=0.1`)
  don't affect the eval path.

The overlay is safe to use as the Stage 2 `model_name_or_path`.

## 5. Leftover design-doc concerns, now resolved

- ~~OOM risk at bs=3~~ — no OOM across 200 steps.
- ~~PEFT + custom model compat~~ — works end-to-end including save.
- ~~projector save with peft~~ — verified (§2).
- ~~s2_init_42k overlay integrity~~ — verified (§4).

All blockers cleared; remaining items are design decisions, tracked in
[`stage2_eval_plan.md`](stage2_eval_plan.md).

---

# 4-modality smoke (2026-04-24, post-mix expansion)

After the ASR-only smoke above, Stage 2 scope expanded to a 4-modality mix
(ASR 14.5 % / Emo 33.7 % / Env 34.6 % / Text 17.2 %, see
[`stage2_design.md §6.1`](stage2_design.md#61-training-mix-confirmed-plan-2026-04-24)).
A second smoke validated the multimodal data pipeline + processor option C.

## A. Full 200-step smoke (`Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17-smoke`)

Combined manifest: 136 907 rows across 128 shards. Config:
[`stage2_smoke.yaml`](../configs/qwen3_5ae-asr/stage2_smoke.yaml) — Liger ON, sdpa
(before we rebuilt with fa2 + LD_PRELOAD shim), 8 GPUs × batch 3.

| metric | value |
|---|---|
| `train_runtime` | 25:51 |
| `train_loss` | 1.50 (global avg over 200 steps; high because burn-in + new modalities) |
| Samples / sec | 3.09 |
| `checkpoint-100`, `checkpoint-200` | saved; `adapter_model.safetensors` carries 64 LoRA + 39 projector keys |
| 4-modality routing | verified via [`dryrun_processor.py`](../scripts/emo/dryrun_processor.py) — audio_asr / audio_emotion / audio_env_sound / text all round-trip |

Loss trajectory: step 10 → 3.10, step 100 → save, step 195 → 1.07, step 200 → 1.09.

## B. Mini-smoke 30-step (per-task loss logging via Option C)

Goal: verify `loss/<modality>` keys appear in the trainer log dict when Liger
is ON — needed because Liger's `LigerForCausalLMLoss` fuses CE with lm_head so
`outputs.logits` is always `None`.

Attempts:

| # | config | outcome |
|---|---|---|
| 1 | Liger ON, no hook | no per-task keys (silent skip — logits None) |
| 2 | Liger OFF, batch 3 | OOM on 1 GPU (Triton alloc during linear-attn backward) |
| 3 | Liger OFF, batch 1 | ✅ all four `loss/<modality>` + `tokens/<modality>` appear |
| 4 | Liger ON, `output_hidden_states=True` | no hidden states — inner model ignores kwarg |
| 5 | Liger ON + warning diagnostics | confirmed: `outputs.hidden_states is None` |
| 6 | Liger ON + forward hook on `Qwen3_5AEModel` | ✅ works — hook captures `last_hidden_state`, `compute_loss` projects through `lm_head` in `no_grad`, per-task loss values match #3 to 1e-3 |

Final implementation: Option C (attempt #6). Trainer lazy-registers a forward
hook at first `compute_loss` call; hook fires on every forward regardless of
wrapping (DeepSpeed / PEFT / LoRA) because it matches by class name
(`type(mod).__name__ == "Qwen3_5AEModel"`).

## C. Env / launch fixes found during smoke

- **deepspeed config path**: `stage2.yaml` pointed at a stale `/mnt/fr20tb/…` path.
  Updated to `/mnt/ddn/users/jos/…`.
- **`dacvae` python package**: `audio_lmf` env had a dangling egg-link to a non-existent path.
  Re-`pip install -e /mnt/ddn/users/jos/AudioEnc/dacvae` restored it.
- **protobuf dependency conflict**: dacvae's `descript-audiotools` pin of `protobuf<3.20`
  clashes with wandb's `>=4.21`. Pinned `protobuf 4.25.9` — wandb happy, dacvae still imports.
- **flash_attn glibc mismatch**: wheel needs `__libc_single_threaded` (glibc 2.32+),
  system has 2.31. Two options —
  (a) `flash_attn: sdpa` in yaml (slower), or
  (b) `LD_PRELOAD=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf/lib/glibc_compat.so`
    (used by `run_nsml.sh` for Stage 1; real run adopts the same shim to keep fa2).
- **manifest shard requirement**: single-file JSONL manifest fails HF datasets'
  `.shard(N=world_size)` with `IndexError`. Pre-splitting into 128 shards
  (built automatically by [`build_combined_manifest.py`](../scripts/emo/build_combined_manifest.py))
  fixes it.
- **eval sweep ckpt filter**: [`inference_ckpt_sweep.py`](../../../wbl_residency/jos/AudioEnc/eval_ckpts/inference_ckpt_sweep.py)
  originally required `model.safetensors`; patched to also accept adapter-only
  ckpts (`adapter_model.safetensors` + `adapter_config.json`).
