# Stage 2 design — Qwen3.5AE-ASR (2026-04-23)

Stage 2 builds on the converged Stage 1 projector by (1) unfreezing the LLM via LoRA,
(2) broadening the training distribution with GigaSpeech + CommonVoice, and
(3) injecting low-amplitude Gaussian noise into the DAC latent so the projector becomes
robust to small perturbations before the LLM starts adapting.

Config file: [`configs/qwen3_5ae-asr/stage2.yaml`](../../configs/qwen3_5ae-asr/stage2.yaml)

---

## 1. Init checkpoint

Stage 1 ran for 50 278 steps before we stopped it for eval. Full test-clean WER sweep on
all surviving ckpts (40k–50k, one snapshot per 1 000 steps):


|       ckpt |                WER (test-clean, 2620 samples) |
| ---------: | --------------------------------------------: |
|     40 000 |                                         7.31% |
|     41 000 |                                         7.57% |
| **42 000** |                                     **7.28%** |
|     43 000 |                                         7.32% |
|     44 000 |                                         7.45% |
|     45 000 |                                         7.44% |
|     46 000 |                                         7.34% |
|     47 000 | 8.54% (repetition hallucination on 3 samples) |
|     48 000 |                                         7.80% |
|     49 000 |                                         7.48% |
| **50 000** |                                     **7.29%** |

**Pick**: checkpoint-42000. (50 000 is statistically tied but has a small repetition-loop
spike at the tail; 42 000 is the cleanest local minimum.)

Because HF loads `audio_encoder.py` and `config.json` from the ckpt dir under
`trust_remote_code=True`, the raw 42 000 ckpt would run the old un-branched
`audio_encoder.py` and ignore the new noise-aug config even if we added it later. So we
build an overlay:

```
/mnt/tmp/s2_init_42k/
├── model-*.safetensors                  ← hardlink from ckpt-42000
├── modeling_qwen3_5AE.py, tokenizer.*   ← hardlink
├── audio_encoder.py                      ← FRESH COPY from Qwen3.5AE-4B-s2 (branched)
└── config.json                           ← FRESH COPY + audio_config.noise_aug_enabled=true
```

The overlay lives on `/mnt/tmp` (same filesystem as the archive) so the weights stay
hardlinked and take zero extra disk. The S2 yaml's `model_name_or_path` points here.

Verified on load: `noise_aug_enabled=True`, `noise_aug_max_k=0.1`, projector weights
match the 42 000 checkpoint.

## 2. Noise augmentation

Edited [`audio_encoder.py`](../../../models/Qwen3.5AE-4B-s2/audio_encoder.py)
(lives at `ddn/models/Qwen3.5AE-4B-s2/` — relative to the audiollm-trainer repo root;
the overlay copies it into `/mnt/tmp/s2_init_42k/`):

```python
self.noise_aug_enabled = bool(getattr(config, "noise_aug_enabled", False))
# (AUDIO_NOISE_AUG env var overrides config for quick flip)
self.noise_aug_max_k = float(getattr(config, "noise_aug_max_k", 0.1))

# inside forward:
if self.training and self.noise_aug_enabled:
    k = torch.rand(B, 1, 1, device=..., dtype=...) * self.noise_aug_max_k
    epsilon = torch.randn_like(audio_latents)
    projector_input = k.sqrt() * epsilon + (1 - k).sqrt() * audio_latents
```

- `k ~ U(0, 0.1)` per utterance, `epsilon` per frame.
- Stage 1 default: `noise_aug_enabled=false`, branch is a no-op (existing S1 ckpts load
  unchanged).
- Stage 2: `noise_aug_enabled=true` in `/mnt/tmp/s2_init_42k/config.json`.

**Scope (noted 2026-04-24, mid-run)**: originally intended ASR-only, but the
gate is `self.training and self.noise_aug_enabled` with no modality check, so
in the 4-modality Stage 2 run the perturbation fires for **every audio sample
(ASR + emotion + env_sound)**; text rows have no audio and skip the encoder.
Left as-is — variance-preserving form keeps train/inference distributions
continuous and `k ≤ 0.1` is mild (latent SNR ≥ ~13 dB), matching the
feature-space augmentation regime (mixup / SpecAugment-style) routinely used
in emotion and env-sound SOTA. The cost is that the current run cannot be
used to ablate per-modality noise-aug effect cleanly; if an ablation becomes
relevant later, re-train a comparator with a modality-gated variant
(`audio_encoder.forward` would need to accept a per-sample ASR mask derived
from the packed `modality_ids` tensor, since a single packed batch mixes all
four modalities and a scalar flag cannot express it).

## 3. LoRA scope

**Target**: Qwen LLM attention only (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
**Not target**: the 4-layer Llama projector inside `audio_encoder.projector`, which also
has `self_attn.*_proj` and would otherwise be swallowed by PEFT's substring match.

Three-part fix:

1. **`COMPOSITE_MODELS` registration** — added `qwen3_5_ae` to
   [`src/llamafactory/model/model_utils/visual.py`](../../src/llamafactory/model/model_utils/visual.py)
   with `projector_key="audio_encoder.projector"` and
   `vision_model_keys=["audio_encoder"]`. This routes the `patch_target_modules()` filter
   to exclude anything containing those keys.
2. **yaml flags** — `freeze_vision_tower: true`, `freeze_multi_modal_projector: true`
   feed the two forbidden-module lists. Result: PEFT only wraps
   `model.language_model.layers.*.self_attn.{q,k,v,o}_proj`.
3. **workflow freeze filter** (see §4) — re-enables the projector's base weights so it
   trains full even though it's a "frozen multi-modal projector" from PEFT's POV.

LoRA config (yaml):


| field               | value                         |
| ------------------- | ----------------------------- |
| `lora_rank`         | 16                            |
| `lora_alpha`        | 32                            |
| `lora_dropout`      | 0.05                          |
| `lora_target`       | `q_proj,k_proj,v_proj,o_proj` |
| `additional_target` | `audio_encoder.projector`     |

`additional_target` → PEFT's `modules_to_save` — without it, peft's
`save_pretrained()` writes only the adapter and the projector's full-weight updates
would be silently lost every save.

## 4. Freeze semantics

[`src/llamafactory/train/omni/workflow.py`](../../src/llamafactory/train/omni/workflow.py)
line 67–70:

```python
for name, param in model.named_parameters():
    require_grad = ("audio_encoder.projector" in name) or ("lora_" in name)
    param.requires_grad = require_grad
```

- **Stage 1** (finetuning_type=`full`): no `lora_` substrings → only projector trains.
  Unchanged from the original S1 behavior.
- **Stage 2** (finetuning_type=`lora`): projector weights (full) + LoRA adapters train;
  DAC encoder, LLM base, embeddings, lm_head all frozen.

## 5. Training hyperparameters


|                       | Stage 1                          | Stage 2                |
| --------------------- | -------------------------------- | ---------------------- |
| model_name_or_path    | `audiollm/sanghyuk/Qwen3.5AE-4B` | `/mnt/tmp/s2_init_42k` |
| finetuning_type       | full                             | lora                   |
| LR                    | 2.0e-4                           | **1.0e-5**             |
| lr_scheduler          | warmup_stable_decay              | warmup_stable_decay    |
| warmup_steps          | 1 000                            | 1 000                  |
| max_steps             | 100 000                          | **50 000**             |
| per_device_batch_size | 3                                | 3                      |
| grad_accum            | 1                                | 1                      |
| bf16                  | ✓                               | ✓                     |
| DeepSpeed             | ZeRO-2                           | ZeRO-2                 |
| Liger kernel          | ✓                               | ✓                     |
| flash_attn            | fa2                              | fa2                    |
| save_steps            | 1 000                            | 1 000                  |
| save_total_limit      | *removed* (no rotation)          | *removed*              |

LR chosen 20× lower than Stage 1 because LoRA on a pretrained LLM is much more sensitive
than a fresh projector. 50 k steps is half of Stage 1's budget — the projector started
converged, we only need the LoRA / projector co-adaptation to settle.

## 6. Dataset

Stage 2 is **LoRA SFT on a 4-modality mix** (ASR + emotion + env-sound + text), with
ASR kept at a small anchor share (~14 %) to prevent regression from the converged
Stage 1 projector. 자세한 mix 는 §6.1 (이전 draft 의 "no ASR in training" 표현은 폐기 — ASR halving 결정 후에도 14.5 % 비중 유지). Contamination
audit and the raw-corpus sourcing plan: [`leakage_audit.md`](leakage_audit.md).
Evaluation protocol per corpus: [`eval_plan.md §6–§9`](eval_plan.md).

### 6.1 Training mix (confirmed plan, 2026-04-24)

**Direction change from earlier draft**: LISTEN_full is **not** a training source.
Its train split draws heavily from source-corpus test splits (MELD-test 881, MOSEI-Test 62,
IEMOCAP-S5 467 audios; see audit §3), which would contaminate held-out eval. We instead
pull full raw corpora and exclude only the 2 546 LISTEN-test IDs, keeping
LISTEN-test as the primary Tier-2 eval.

**Confirmed mix (user-set, 2026-04-24 — ASR halved 2026-04-24 evening)**:

| Category   | Share | Row count (actual) | Ratio to Emo baseline |
|------------|------:|-------------------:|----------------------:|
| ASR        | 14.5% |             17 150 | × 0.415 (halved from 0.83) |
| Emotion    | 33.7% |             39 919 | × 1 (baseline)        |
| Env sound  | 34.6% |             41 000 | × 1                   |
| Text       | 17.3% |             20 519 | × 0.5                 |
| **Total**  |  100% |        **118 588** |                       |

Ratio: **ASR : EMO : ENV : TXT = 0.415 : 1 : 1 : 0.5**. Emotion pool size
(39 919 after held-out splits + DailyTalk 5 % cutoff fix) sets the scale.
ASR share was halved per user 2026-04-24 evening to free more gradient budget
for the non-ASR modalities (especially after loss-imbalance measurement
showed ASR already dominated 68 % of target tokens at the 25 % row share;
see [audit §9.3](leakage_audit.md#93-target-token-budget--measured-loss-signal-share-per-modality-2026-04-24)).

**Per-category sources** (more detail: [audit §6.1.3](leakage_audit.md),
[eval-plan §6](eval_plan.md)):

| Category  | Sources                                                                                              | On-disk rows           |
|-----------|------------------------------------------------------------------------------------------------------|-----------------------:|
| Emotion   | DailyTalk 22 573 · MELD train+dev 11 043 · EmoV-DB (Bea+Josh+Sam) 5 103 · RAVDESS actors 01-20 (minus 1 hash-match) 1 199 · MUStARD 65 mp4 | 39 919 |
| ASR       | Superset shuffled-128 at `/mnt/fr20tb/audiollm/sanghyuk/datasets/qwen3_5_dacvae_asr_shuffled_128/` (MLS 53% + Giga 39% + CV 5% + libri 2% + vox 0.8% natural distribution), loaded from nubes via `load_from_nubes: true` | 17 150 |
| Env sound | FSD50K dev 35 884 + Clotho development 3 356 + ESC-50 (all 5 folds) 1 760 — fold rotation at eval | 41 000 |
| Text      | HellaSwag 7 500 + WinoGrande 7 500 + ARC-E 2 000 + ARC-C 1 119 + BoolQ 2 000 + COPA 400, from `/mnt/tmp/datasets/text_benchmarks/` | 20 519 |

**Held-out splits (not in training pool)**: MELD-test (2 747) · RAVDESS actors 21-24
(240) · EmoV-DB Jenie speaker (1 790) · DailyTalk last 5 % dialogues (1 168) · IEMOCAP
Session 5 (when IEMOCAP arrives) · FSD50K eval + Clotho eval/validation · Tier-4
benchmark val/test splits.

**Manifest files** (per-category, emitted by `scripts/emo/` and `scripts/env_sound/`):
- `/mnt/tmp/listen_analysis/train_manifest/train_manifest.jsonl` — emotion rows 39 983 (pre-MCQA)
- `/mnt/tmp/listen_analysis/train_manifest/emotion_mcqa_manifest.jsonl` — emotion MCQA 39 919
- `/mnt/tmp/listen_analysis/train_manifest/asr_manifest.jsonl` — ASR 17 150 (halved)
- `/mnt/tmp/listen_analysis/train_manifest/env_sound_manifest.jsonl` — env sound 41 000
- `/mnt/tmp/listen_analysis/train_manifest/text_sft_manifest.jsonl` — text 20 519
- `/mnt/tmp/listen_analysis/train_manifest/stage2_combined_manifest.jsonl` — **final 118 588 rows**

**EULA-pending additions** (will increase emotion pool 6× once arrived):
**IEMOCAP** (~12 h / 10 k) + **ESD** (~29 h / 35 k) + **MSP-Podcast** (200 h+) +
**MSP-IMPROV** (9 h / 8 k, UTD bundle).

**Rows vs. hours mix**: ASR rows ~5-15 s audio each, emotion ~3-5 s, env sound ~5-30 s,
text = 0 s audio but ~100 tokens. Row-share ≠ step-share. Final sample weights must
be set per effective-step once per-corpus mean audio duration + token length is known
(see [audit §9.3](leakage_audit.md)).

### 6.2 Manifest + preprocessing pipeline

**Built** ([scripts under `scripts/emo/`](../../scripts/emo/)):

- `build_training_manifest.py` → `/mnt/tmp/listen_analysis/train_manifest/train_manifest.jsonl`
  (42 871 rows today). Enumerates raw corpora, removes LISTEN-test-mapped
  files, emits `{path, source, relpath}` rows.
- `match_listen_test_to_sources.py` + `hash_match_ravdess.py` → per-source
  exclusion sets consumed by the builder.
- `extract_meld_audio.sh` → MELD `.mp4 → .wav` (16 kHz mono, 13 847 files).

**Still pending** (blocks S2 launch):

1. **Per-source ChatML handler (Option C)** — `omni_dataset.py:create_omni_processor`
   currently hardcodes `"Transcribe the audio to text."` suffix. Per user direction,
   modify to a data-driven template: each manifest row carries its fully-rendered
   `question / choices / answer / rationale` (text side is already in this format
   via `text_sft_manifest.jsonl`); emotion side needs the per-corpus MCQA builder
   that maps each emotion-label taxonomy (MELD 7-class / DailyTalk 7-class /
   EmoV-DB 5-class / RAVDESS 8-class / MUStARD binary) to lettered choices.
   ASR rows keep a fixed `"Transcribe..."` template; env-sound rows use captioning
   or classification prompt depending on source (Clotho → caption, FSD50K/ESC-50 →
   classify).
2. **Rationale-augmentation pass** — emotion rows today only have the ground-truth
   label. Need an offline GPT-4 / Llama-3-70B pass that generates the 20–50-token
   rationale per label, grounded on audio cues only (prompt template in eval-plan §3.1).
   ASR, env-sound, text rows do not need rationales.
3. **Omni collator / mixer** — composite needs rows with `modality` ∈ {`audio_asr`,
   `audio_emotion`, `audio_env_sound`, `text`} — the `audio_features is None`
   branch must short-circuit for `modality == "text"`. Implement via one of:
   (a) interleaving 4 manifests with `llamafactory-cli`'s per-dataset weight knob
   and normalized sampling frequencies matching the 25/30/30/15 mix,
   (b) emit a single pre-mixed JSONL with a `weight` or `sampling_probability` per
   row so the DataLoader approximately follows the target ratio.

### 6.3 Manifest loader knobs

```yaml
omni_max_audio_samples: 1600000   # ~33.3 s @ 48 kHz (Stage 1 used 2.16 M = 45 s)
```

33 s cap covers DailyTalk / MELD / EmoV-DB / RAVDESS (all well under) and
most of Clotho / FSD50K; a small fraction of FSD50K's long-form clips would
truncate. Emotion-side: no effect.

### 6.4 Per-modality interleave option (added 2026-04-24)

For runs where the emotion pool is much smaller than the other modality pools
and you want the large pools to be effectively re-sampled across epochs
(instead of the combined-manifest behavior where every modality pulls from a
fixed subsampled file), there is now an opt-in interleave path in
[`data/loader.py`](../../src/llamafactory/data/loader.py):

```yaml
# Mutually exclusive with omni_manifest: when set, overrides the single-manifest path.
omni_per_modality_manifests:
  audio_asr:        /mnt/tmp/listen_analysis/train_manifest/asr_full_shards
  audio_emotion:    /mnt/tmp/listen_analysis/train_manifest/emotion_mcqa_shards
  audio_env_sound:  /mnt/tmp/listen_analysis/train_manifest/env_full_shards
  text:             /mnt/tmp/listen_analysis/train_manifest/text_full_shards
omni_per_modality_probs:
  audio_asr:       0.145
  audio_emotion:   0.337
  audio_env_sound: 0.346
  text:            0.172
omni_per_modality_stopping: all_exhausted   # or first_exhausted
```

Each modality manifest is loaded, per-node sharded, and shuffled with its own
seed (`training_args.seed + idx * 1_000_003`) before being combined via
`datasets.interleave_datasets(..., probabilities=..., seed=..., stopping_strategy=...)`.
With `all_exhausted`, pools shorter than the largest are cycled automatically,
so a ~40 k emotion pool keeps repeating inside one epoch while a ~1 M-row ASR
superset pool is drawn mostly-fresh. The caller is responsible for supplying
parent-pool manifests (sharded into multiple jsonl files to preserve worker
parallelism). Schema must be consistent across modalities — the first-row peek
in `_build_dataset` determines `col_names` for `map(remove_columns=...)`, so
drift between modality schemas could leak leftover columns into the packed
output.

Leave `omni_per_modality_manifests: null` (the default) to preserve the
existing single-manifest behavior used by the current Stage 2 run.

**Schema requirement (important).** Each modality source JSONL has a different
key set (ASR: `{source, nubes_path, text, modality}`, emotion MCQA:
`{source, path, question, choices, answer, label, transcript, rationale,
modality}`, env sound: `{source, path, labels, modality}`, text SFT:
`{source, question, choices, answer, modality}`). `datasets.load_dataset(json,
data_files=...)` infers Features per-file, so interleaving raw per-modality
streams trips Arrow casts (e.g. `list<string>` vs `null` for `choices`).
Before passing manifests to this option, **pre-normalize every row to a
union schema** (missing scalar → `null`, missing list → `[]`). The reference
implementation is in [`scripts/emo/smoke_interleave.py`](../../scripts/emo/smoke_interleave.py)
(`UNION_FEATURES` + `_normalize`): a 4 000-row drain with the current
per-modality manifests produced realized probabilities within ≤ 0.7 pp of
the 14.5 / 33.7 / 34.6 / 17.2 target. The same normalization should be
baked into whatever script emits the parent-pool shard dirs.

Normalization is intentionally kept out of `loader.py` so the loader stays
agnostic of modality field names — keep it alongside the manifest builders
under `scripts/emo/` and `scripts/env_sound/`.

## 7. Checkpoint format & eval

- **During training**: PEFT saves `adapter_model.safetensors` + `adapter_config.json` +
  (from `additional_target`) projector module weights. Compact (~200 MB / ckpt, mostly
  projector).
- **Eval sweep**: `AudioEnc/eval_ckpts/inference_ckpt_sweep.py` (외부 도구,
  audiollm-trainer 외부 AudioEnc 레포에 위치 — 본 노드 미존재)
  auto-detects adapter-only ckpts and loads via
  `PeftModel.from_pretrained(base, adapter_dir)`. Pass `--base_model /mnt/tmp/s2_init_42k`
  (or rely on `adapter_config.json`'s `base_model_name_or_path`).
- **Merged export**: `llamafactory-cli export` post-hoc for distribution — no
  merge-on-save callback (merging `merge_and_unload()` mid-training mutates the live
  model and would corrupt the run).

## 8. Infrastructure (carried over from Stage 1 eval work)

- **Archive watcher** (`AudioEnc/eval_ckpts/ckpt_archive_watcher.sh`): tmux `ckpt_watch`
  session, 30 s poll. Built because S1's original yaml had `save_total_limit: 8` and
  rotated checkpoint-40000 out before we could eval it. Both S1 and S2 yamls now omit
  `save_total_limit` so rotation is off — the watcher is redundant for S2, keep it
  disabled unless you add rotation back.
- **Sweep script** handles both full-weight (S1) and adapter-only (S2) ckpts
  transparently; the existing `--ckpt_root` + `--skip_if_done` flow works as-is.

## 9. Checklist before launch

**Done** (audit + downloads, 2026-04-23 → 04-24):
- [x] LISTEN contamination audit + per-source split provenance ([audit §2-§5](leakage_audit.md)).
- [x] Direction decision: raw corpora minus LISTEN-test, not LISTEN_full.
- [x] Raw emotion corpora downloaded: MELD, RAVDESS, DailyTalk, EmoV-DB (re-extracted per-speaker), MUStARD metadata.
- [x] Text-reasoning Tier-4 benchmarks downloaded: ARC-e/c · WinoGrande · HellaSwag · BoolQ · COPA.
- [x] Env sound downloaded: ESC-50 done · Clotho extracted · FSD50K extraction in progress.
- [x] LISTEN-test → raw-source mapping + held-out split enforcement in training manifest.
- [x] Text SFT manifest emitted (20 519 rows).
- [x] ASR manifest emitted (34 300 rows, 5-corpus superset subsample via `load_from_nubes`).
- [x] Held-out splits applied: MELD-test / RAVDESS actors 21-24 / EmoV-DB Jenie / DailyTalk 5 %.

**Data-side blockers** (§6.2):
- [x] Env-sound manifest built — 41 000 rows (FSD50K dev 35 884 + Clotho dev 3 356 + ESC-50 1 760).
- [x] Emotion MCQA manifest built — 41 088 rows with per-source lettered choices + answer.
- [x] Text SFT manifest built — 20 519 rows, 6 MCQA bench train splits, unified MCQA form.
- [x] Combined manifest emitted — [`stage2_combined_manifest.jsonl`](file:///mnt/tmp/listen_analysis/train_manifest/stage2_combined_manifest.jsonl) (**현재 디스크 상태: 118 588 rows, 14.5/33.7/34.6/17.3** = §6.1 의 ASR-halved 최종 mix). 이전 draft 의 136 907 rows / 25.05-30.01-29.95-14.99 split 은 ASR halving 전 기록이라 폐기됨.
- [x] `create_omni_processor` per-modality ChatML (Option C) — `audio_asr` / `audio_emotion` / `audio_env_sound` / `text` branches; text rows skip audio download, keep 1:1 index via length-0 placeholder; packer filters `audio_lengths == 0`; collator's empty-list fallback unchanged.
- [x] Dry-run validation ([`scripts/emo/dryrun_processor.py`](../../scripts/emo/dryrun_processor.py)) — 8 rows (2 per modality) round-trip through the processor: audio_lengths `[488,414,104,93,278,50,0,0]`, text rows omit `<|audio_start|>` prefix, 1:1 alignment preserved.
- [x] Yaml `omni_manifest` switched to `/mnt/tmp/listen_analysis/train_manifest/stage2_combined_manifest.jsonl`.
- [x] RAVDESS resample-normalized hash-match ([`hash_match_ravdess_v2.py`](../../scripts/emo/hash_match_ravdess_v2.py)) — 65/200 matched (vs. v1 0/200). 64 of 65 fall in already-held-out actors 23-24; defence-in-depth catches 1 edge case in actor 14.
- [x] MUStARD key lookup — `*_u.mp4` utterance-level files now resolve; 64 of 65 on disk are `*_c.mp4` context (no sarcasm label) so yield stays 1 until GDrive rate-limit clears and more `_u.mp4` download.
- [x] FSD50K full extraction — dev 40 966 + eval 10 231 WAVs (matches paper); eval reserved for Tier-3 classification held-out.
- [~] Rationale synthesis offline pass — **decided against (2026-04-24)**: accept
  letter-only emotion targets and their 4 % loss-share. Script
  [`synthesize_rationales.py`](../../scripts/emo/synthesize_rationales.py) stays in
  the tree for possible later reversal (e.g. after first full run shows emotion
  capability doesn't transfer from LISTEN-test eval). Processor already
  concatenates `rationale` when non-null, so enabling later is just a manifest
  rebuild — no code changes.
- [ ] EULA ingestion — IEMOCAP / ESD / MSP-Podcast / MSP-IMPROV when received.

**Env / launch requirements** (discovered while launching real run):
- System glibc is 2.31 but `flash_attn_2` wheels reference `__libc_single_threaded` (glibc 2.32+). Either use `flash_attn: sdpa` (drops throughput) or preload
  `/mnt/ddn/users/jos/miniforge3/envs/audio_lmf/lib/glibc_compat.so` via `LD_PRELOAD`
  (see `run_nsml.sh` template). Stage 1 used the latter; Stage 2 `stage2.yaml` is now
  `flash_attn: fa2` and the launch shell exports the shim.
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  inherited from `run_nsml.sh` — expandable segments prevents fragmentation-OOM on
  long-audio batches.
- `enable_liger_kernel: true` in real run — Liger's fused CE preserves memory; per-task
  loss logging uses a forward hook on `Qwen3_5AEModel` to capture `last_hidden_state`
  and project through `lm_head` in `no_grad` (see [`OmniTrainer._ensure_hidden_hook`](../../src/llamafactory/train/omni/trainer.py)).

**Runtime smoke** (done):
- [x] Mini-smoke 30 steps @ 1 GPU, Liger ON, hook-based per-task loss verified:
  `loss/audio_asr`, `loss/audio_emotion`, `loss/audio_env_sound`, `loss/text` +
  `tokens/<name>` all appear in each log step; values match the Liger-OFF baseline
  run to ~1e-3 (float noise).

- [x] Torchrun 200 steps of `stage2.yaml` @ `/mnt/tmp/results/Qwen3.5AE-ASR-Stage2-lora-smoke/`
  - LoRA applied only to LLM attention: 64 `lora_*` keys (8 full-attn layers × {q,k,v,o} × {A,B}).
  - Projector weights in save dir: 39 `audio_encoder.projector.*` keys inside `adapter_model.safetensors`
    via `additional_target`.
  - No OOM at batch_size=3 with LoRA backprop + gradient checkpointing; 200 steps completed.
- [x] Step-0 baseline: `s2_init_42k` test-clean **WER 7.26 %** (2 620 samples, 2026-04-23) — within 0.02 pt of
  ckpt-42000's 7.28 %; overlay noise_aug is a no-op in eval path.
- [x] ~~Confirm GigaSpeech / CommonVoice paths fetch from nubes~~ — N/A, ASR aux dropped from S2 mix.

**Eval-side** (모두 완료, 결과는 [`3model_comparison.md`](3model_comparison.md), [`eval_harness.md`](eval_harness.md)):
- [x] Tier-1 ASR regression check — LibriSpeech test-clean/other WER. v1 25k 결과 + v2 31k + W-tiny 31k + W-small 35k 모두 측정 완료.
- [x] Tier-2 LISTEN-test accuracy per sub-corpus (MELD/IEMOCAP는 contamination 으로 dropped, [`leakage_audit.md §1`](leakage_audit.md)).
- [x] Tier-3 Clotho eval + FSD50K eval + ESC-50 CV — env-sound 35 % in-domain 학습 후 측정 완료.
- [x] Tier-4 6-benchmark sweep (HellaSwag, WinoGrande, BoolQ, ARC-e, ARC-c, COPA) — text retention 컬럼 으로 측정 완료.
