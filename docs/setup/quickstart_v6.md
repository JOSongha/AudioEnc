# Stage-1 v6 quickstart

End-to-end recipe to bring up Stage-1 v6 (projL projector, 5 encoder variants) on a fresh cluster node. Assumes nubes access and a working conda/miniforge install.

## 1. Prerequisites

- Cluster node with a local scratch at `/mnt/tmp` (xfs / NVMe). v6 yaml + launchers assume `/mnt/tmp/datasets/manifests/v6_nubes/` for manifests and `/mnt/tmp/<run>/` for ckpt output.
- 8× A100 80 GB (single node). Multi-node is not the default path — the launchers run `torchrun --nnodes 1`.
- `nubescli` (cli for the internal nubes object store). Default path expected by `setup_models.sh`: `/mnt/ddn/users/jos/cli/nubescli`. Override with `NUBESCLI=/your/path bash scripts/setup_models.sh` if needed.
- Nubes gateway access: `c.nubes.sto.navercorp.com:8000`. The setup script exports `NUBES_GATEWAY_ADDRESS` / `NUBES_IP_LOOKUP_ADDRESS` for you; if your environment uses a different gateway, set them before invoking.

## 2. Clone + Python env

```bash
git clone <fork-url> audiollm-trainer
cd audiollm-trainer
git checkout v6_forPR    # or main, post-merge

# Create conda env + install deps + editable install of this repo
bash scripts/ASR/install_env.sh
```

`install_env.sh` creates a miniforge env (default name: `audio_lmf` under your miniforge3), pins torch 2.5.1 + cu124, installs deepspeed / flash-attn / liger-kernel / fla-core, then `pip install -e .` on the cwd repo.

If your miniforge env lives elsewhere, point the launchers at it via `AUDIO_LMF_ENV=/your/conda/env bash scripts/ASR/run_stage1_<encoder>_v6.sh`.

## 3. Fetch model bases from nubes

```bash
bash scripts/setup_models.sh
```

What it does (idempotent, resume-safe):
1. Downloads 10 safetensors (5 encoder bases × 2 shards, ~40 GB) from `hyperscaleai-audiollm/users/jos/AudioEnc/models/<base>/` into `external/models/<base>/`. Already-correct files are skipped (size match against nubes `X-Object-Size`).
2. Regenerates the 5 projL overlay dirs (`Qwen3.5AE-4B-projL`, `Qwen3.5AE-4B-encodec-projL`, ...) with repo-relative symlinks to the base dir for every file except `config.json` (which stays — it carries the projL-specific `audio_config`: H=1024 / I=4096 / heads=16).

After this step `external/models/` is self-contained — HF transformers' `from_pretrained` can load each `*-projL` dir as a complete model.

## 4. Restore the manifest archive

The yaml's `omni_per_modality_manifests` paths read from `/mnt/tmp/datasets/manifests/v6_nubes/`. Restore the manifest snapshot from nubes:

```bash
export NUBES_GATEWAY_ADDRESS=c.nubes.sto.navercorp.com:8000
export NUBES_IP_LOOKUP_ADDRESS=c.lookup.nubes.navercorp.com:8080
mkdir -p /mnt/tmp/datasets/manifests
/mnt/ddn/users/jos/cli/nubescli dir-download -j 16 \
    hyperscaleai-audiollm/users/jos/AudioEnc/manifests/v6_nubes/ \
    /mnt/tmp/datasets/manifests/v6_nubes/
```

246 jsonl, 5.2 GB, ~30 s with `-j 16`. Every row's `nubes_path` already points at the audio location on nubes — no local audio copy is required.

## 5. Launch training

Pick one of the 5 encoders:

```bash
bash scripts/ASR/run_stage1_dac_vae_v6.sh        # DAC-VAE 48 kHz
bash scripts/ASR/run_stage1_encodec_v6.sh        # EnCodec 24 kHz
bash scripts/ASR/run_stage1_wavtok_v6.sh         # WavTokenizer 40-unify
bash scripts/ASR/run_stage1_whisper_tiny_v6.sh   # Whisper-tiny.en
bash scripts/ASR/run_stage1_whisper_small_v6.sh  # Whisper-small.en
```

Each launcher:
- Auto-detects `REPO` via `$(cd "$(dirname "$0")/../.." && pwd)` and `cd`s there before invoking `llamafactory-cli train`.
- Sets cache dirs under `/mnt/tmp/cache/` (triton / cuda / HF / wandb / tmp).
- Sets NCCL hardening (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`, `NCCL_ASYNC_ERROR_HANDLING`, `TORCH_NCCL_DESYNC_DEBUG`) so a collective hang surfaces the straggler rank instead of timing out silently after 1 h.
- Tees stdout/stderr to `<output_dir>/launch_logs/run_<TIMESTAMP>.log` — when a tmux pane goes away the traceback survives.
- The Whisper variants additionally export `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` to avoid the `whisper-tiny.en/processor_config.json` 401 (file doesn't exist on Hub).

### Resume from a checkpoint

Set `RESUME_FROM` before invoking:

```bash
RESUME_FROM=/mnt/tmp/Qwen3.5_whisper_tiny_v6_Stage1_jos/Qwen3.5AE-ASR-Stage1-whisper-tiny-v6/checkpoint-46000 \
WANDB_RUN_ID=<existing_id> WANDB_RESUME=allow \
    bash scripts/ASR/run_stage1_whisper_tiny_v6.sh
```

The launcher appends `resume_from_checkpoint=$RESUME_FROM` as an OmegaConf override (LLaMA-Factory's yaml CLI takes `key=value` form, not `--key value`). `WANDB_RUN_ID` / `WANDB_RESUME=allow` make wandb continue the same run instead of forking a new curve. The yaml's `ignore_data_skip: true` bypasses the streaming-dataset rewind so the dataloader doesn't re-iterate 46k batches.

## 6. Monitor

- wandb: project `qwen3_5ae-asr` (set via launcher env).
- Container memory: v6 trains at ~45–50 % of the 1.5 TiB cgroup limit. If `cat /sys/fs/cgroup/memory/memory.usage_in_bytes` divided by `memory.limit_in_bytes` crosses 80 %, expect an OOM-kill — the past failures left no traceback because the cgroup reaper sends SIGKILL.
- GPU: `nvidia-smi`. During a checkpoint save (every 1000 steps, `save_steps`) you'll see rank 0 go to 0 % util for ~30–60 s while the model state writes to disk — the other ranks spin at the NCCL barrier (100 % util shown, 80–90 W draw). Not a hang.
- Checkpoints: `<output_dir>/checkpoint-<step>/` (size: ~26 GB each, `save_total_limit: 8`).

## 7. Stage-2 evaluation (optional)

Once Stage-1 produces a checkpoint, evaluate with the shared loader under [`evaluation/audio/`](../../evaluation/audio/):

```bash
python -m evaluation.audio.eval_librispeech_wer \
    --ckpt-root /mnt/tmp/Qwen3.5_<encoder>_v6_Stage1_jos/Qwen3.5AE-ASR-Stage1-<encoder>-v6 \
    --out-root  ./eval_libri \
    --split test.clean --ckpts <step1>,<step2> --batch-size 4
```

The loader (`_loader.py`) auto-detects the encoder from `cfg.audio_config` (Whisper / DAC / EnCodec / WavTok) and packs audio features accordingly. The same script works on every v6 base. Other eval drivers in the same dir cover FSD50K / AudioSet / Clotho / emotion classification — see [eval_prompts.md](../reference/eval_prompts.md) for the exact prompt + metric per task.

## 8. Common gotchas

| Symptom | Cause / fix |
|---|---|
| `OSError: openai/whisper-tiny.en is not a local folder` mid-training | `processor_config.json` 401 from HF Hub. Whisper launchers already export `HF_HUB_OFFLINE=1`, but stale HF cache can still hit the network — clear `$HF_HOME/hub/models--openai--whisper-tiny.en/` and re-launch. |
| `WorkNCCL(SeqNum=…, OpType=ALLREDUCE) ran for 3600000 ms before timing out` | One rank's dataloader stalled on a slow `nubes_path` fetch and never reached the collective. Resume from the last checkpoint; the `TORCH_NCCL_DESYNC_DEBUG=1` env var dumps which rank was the straggler. |
| Container goes silent mid-training, no traceback | cgroup OOM-killer (1.5 TiB limit). Lower `dataloader_num_workers` and `omni_shuffle_buffer_size` in the yaml. |
| `nubescli ... 404 Not Found` for a sample | The path-rewrite map in [`scripts/manifest_builders/rewrite_audio_paths_nubes.py`](../../scripts/manifest_builders/rewrite_audio_paths_nubes.py) is missing an entry, or the manifest has a stale `nubes_path`. Rebuild the manifest for that source. |
| `pip install -e .` fails on `flash-attn` | The GLIBC_2.32 workaround. `install_env.sh` builds `glibc_compat.so` automatically; if it's missing on your env, rerun `install_env.sh`. |
| Multi-node setup | Not the default path. Replace `NPROC_PER_NODE=$NGPU` in the launcher with the NSML-style `NNODES=$NSML_WORLD_SIZE NODE_RANK=$NSML_RANK MASTER_ADDR=$NSML_HOST_RANK0 MASTER_PORT=21267` block (see the legacy `scripts/ASR/run_stage1.sh` template for the pattern). |

## 9. Cross-references

- [`docs/setup/nubes_upload.md`](nubes_upload.md): every audio + manifest + model upload to nubes, plus the byte-perfect audit results (§ 12.18).
- [`docs/setup/datasets.md`](datasets.md): v6 training pool (asr / env_sound / emotion sources, counts, leak audit).
- [`docs/reference/parameter_count.md`](../reference/parameter_count.md): projector parameter math (projL = 69,870,592 ≈ 70 M trainable).
- [`docs/reference/eval_prompts.md`](../reference/eval_prompts.md): Stage-2 eval task prompts + metric definitions.
