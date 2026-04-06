# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Use python venv from /mnt/tmp/venv_audio.

## What this project is

AudioEnc is an **encoder-swappable ASR training framework** that pairs a frozen audio encoder with Qwen3.5-2B/4B (LLM) via a trainable Conv1d projector. The sole purpose is to compare multiple audio encoders on the same ASR pipeline without changing any other code.

Training runs in two stages via `torchrun` (multi-GPU DDP using HuggingFace Accelerate):
- **Stage 1**: Projector alignment — LLM frozen, only projector trains
- **Stage 2**: LoRA fine-tuning — LoRA on LLM attention + optionally projector

## Running training

```bash
# Basic (8 GPUs, default datasets ls100+ls360+ls500+mls)
bash run.sh --encoder fb_dacvae

# Change GPU count
bash run.sh --encoder encodec --gpus 4

# Select datasets
bash run.sh --encoder fb_dacvae --datasets ls100,ls360,mls,gs

# Separate Stage 1 / Stage 2 datasets
bash run.sh --encoder fb_dacvae \
    --s1-datasets ls100,ls360 --s1-ls-samples 58000 \
    --datasets ls100,ls360,ls500,mls --mls-samples 500000

# Stage 1 only
torchrun --nproc_per_node=8 train_stage1.py --encoder fb_dacvae

# Direct torchrun (same as run.sh)
torchrun --nproc_per_node=8 --master_port=29500 train.py --encoder encodec
```

**Encoder choices**: `encodec` | `dac` | `fb_dacvae` | `mimi_acoustic` | `mimi_semantic`

**Dataset keys**: `ls100` (LibriSpeech 100h) | `ls360` (360h) | `ls500` (500h) | `mls` (MLS English) | `gs` (GigaSpeech)

**Early-stop Stage 1**: `kill -USR1 $(cat /mnt/tmp/cache/train.pid)`

## Key paths (from config.py)

- `data_path`: `/mnt/tmp/cache` — LibriSpeech root (auto-downloaded on first run)
- `mls_data_path`: `/mnt/tmp/cache` — MLS / GigaSpeech HuggingFace cache
- `model_cache_dir`: `/mnt/tmp/cache/hf` — HuggingFace model weights, projector checkpoints, bucket length caches

Checkpoint naming:
- Stage 1 projector: `s1_proj_{encoder}.pt` (presence skips Stage 1 entirely)
- Stage 2 best: `best_{encoder}_ckpt_ep{N}_step{N}/`
- Stage 2 final: `final_{encoder}_ckpt_ep{N}_step{N}/`

## Architecture

```
Audio (16kHz) → [frozen encoder] → (B, T_enc, out_dim)
              → [Conv1d projector × strides] → (B, T_proj, llm_dim)
              → [LayerNorm]
              → concat with prompt tokens → [Qwen3.5 LLM] → CTC loss on transcript
```

**Data flow in `model.py:AudioQwen.forward()`**:
1. `encoder(audio, lengths)` → `(feats, enc_mask)` always fp32
2. `projector` (Conv1d, transpose input/output) + `proj_norm` → audio embeds
3. `enc_mask[:, ::proj_stride]` downsamples the mask to match projector output length
4. Prompt tokens (`p1`, `p2`) are pre-tokenized and buffered as `register_buffer`
5. Sequence: `[p1] + [audio_embeds] + [p2] + [transcript]`; labels mask everything except transcript

**Prompt format** (controlled by `cfg["llm_type"]`):
- `"base"` (default): `"Audio:\n" + audio + "\nTranscript:\n" + text`
- `"instruct"`: ChatML format wrapping

## Adding a new encoder

1. Create `encoders/myenc.py` subclassing `BaseAudioEncoder` — implement `forward(audio, lengths) → (feats, mask)`. Encoder must: stay frozen/eval, output fp32, handle resampling from 16kHz internally.
2. Add entry to `ENCODER_REGISTRY` in `config.py` with `out_dim`, `tgt_sr`, `hop`, `proj_strides`.
3. Register in `encoders/__init__.py` `ENCODER_CLASSES` dict.

## Important implementation details

- **LLM dtype**: `bfloat16` (Qwen3.5 produces NaN in fp16 due to SSM architecture)
- **Projector dtype**: fp32 in Stage 1 (`freeze_llm()` calls `.float()`); matches LLM dtype otherwise
- **EnCodec + V100**: EnCodec has an LSTM that needs fp32 — handled inside `encoders/encodec.py`
- **DDP model loading**: rank-0 loads first (populates HF cache), then barrier, then other ranks load from cache
- **DynamicBatchSampler**: packs samples greedily so total LLM tokens per batch ≤ `max_batch_tokens` (2600). Bucket length caches stored as `bucket_lengths_{N}.pkl` in `model_cache_dir`.
- **EOS masking**: `collate_fn` always appends EOS as the last token. The masking in `model.py` masks `pad_token` except the final column (`tgt_labels[:, :-1]`). See `docs/eos_masking_bug.md` for history.
- **Stage 1 skip**: if `s1_proj_{encoder}.pt` already exists in `model_cache_dir`, Stage 1 is skipped unconditionally.
- **`train_stage1.py`**: standalone Stage 1 script with identical logic to `run_stage1()` in `train.py`.

## Encoder registry quick reference

| Key | Model | out_dim | fps | proj_strides |
|-----|-------|---------|-----|--------------|
| `encodec` | facebook/encodec_24khz | 128 | 75 | [2,2] |
| `dac` | descript/dac-44kHz | 1024 | ~86 | [2,2] |
| `fb_dacvae` | facebook/dacvae-watermarked | 8 | ~86 | [2,2] |
| `mimi_acoustic` | kyutai/mimi (encoder only) | 512 | 25 | [2,2] |
| `mimi_semantic` | kyutai/mimi (+ encoder_transformer) | 512 | 25 | [2] |

`get_config(encoder_name)` merges `TRAIN_CONFIG` + encoder-specific overrides and auto-computes `samples_per_token` and `max_batch_tokens`.
