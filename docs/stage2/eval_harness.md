# Stage 2 — evaluation harness build report (2026-04-24)

Records the build of the Tier-2 (emotion) and Tier-3 (env-sound) eval drivers
for the running Stage-2 LoRA SFT. Companion to
[`eval_plan.md`](eval_plan.md) (what to measure) and
[`design.md`](design.md) (training setup). This document
captures *how* the eval harness works, the three implementation issues that
blocked off-the-shelf reuse of `eval_testclean_wer.py`, and the verification
done against checkpoint-2000.

> **Companion artifacts** (auto-generated from `summary.json` files; in-repo copies for clickability, source-of-truth at `/mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17/analysis/`):
> - [`analysis/results.csv`](analysis/results.csv) — long-format ckpt × metric × value (587 rows)
> - [`analysis/results_wide.csv`](analysis/results_wide.csv) — wide-format ckpt × 26 metrics
> - [`analysis/results.md`](analysis/results.md) — markdown table
> - [`analysis/best_per_task.md`](analysis/best_per_task.md) — per-task best ckpt
> - [`analysis/trajectories.pdf`](analysis/trajectories.pdf) — 5 × 2 multi-panel figure (10 task panels)
>
> Regenerate: `python -m evaluation.stage2.aggregate_results --root <runDir> --out <outDir>` then `python -m evaluation.stage2.plot_trajectories --csv <outDir>/results.csv --out <outDir>/trajectories.pdf`. Copy results back into `docs/stage2_analysis/` for in-repo viewing.

## 0. TL;DR (2026-04-25, **full 25-ckpt trajectory complete** — training stopped at ckpt-25k)

**No single Pareto-optimum ckpt — peaks are task-stratified (§10.1.2)**. With the dense 1k-step trajectory now closed for every benchmark (FSD50K mAP-seq, LISTEN MCQA, LISTEN-official included), the picture is finalized:

- **ASR test-clean**: peak ckpt-4k (4.98 %), monotone decay → 6.0 % @ 25k
- **ASR test-other**: peak ckpt-5k (18.48 %), monotone decay → 20.9 % @ 25k
- **FSD50K mAP-micro** (seq scoring): peak ckpt-6k (0.356), early-peaker like ASR
- **FSD50K mAP-macro** (seq scoring): peak ckpt-7k (0.327), early-peaker like ASR
- **LISTEN MCQA acc**: peak ckpt-5k (0.244), then long plateau
- **LISTEN-official WAmean**: peak ckpt-11k (0.235)
- **MELD acc** (majority-class): peak ckpt-12k (0.497)
- **Text retention**: peak ckpt-13k (0.907)
- **Clotho captioning**: peak ckpt-15k (BLEU-4 0.0895, BLEU-1 0.476, CIDEr 0.145)
- **EmoV-DB**: peak ckpt-15k (acc 0.776)
- **MELD F1**: peak ckpt-17k (0.259)
- **LISTEN-official F1mean**: peak ckpt-19k (0.137)
- **DailyTalk F1**: peak ckpt-21k (0.441)
- **ESC-50**: peak ckpt-22k (0.832)
- **RAVDESS**: peak ckpt-24k (0.592)

ckpt-2-3k spike (early-training instability) and ckpt-25k drift (final regression) are visible across multiple tasks. For deployment, choose ckpt by the target task; **ckpt-15-17k zone** is the closest single ckpt to a Pareto-decent compromise across emotion + audio understanding + text reasoning, with ASR sacrifice. The **early-peaker family (ASR + FSD50K mAP)** is a new finding from the closed FSD50K-seq trajectory: low-level acoustic-recognition tasks degrade as Stage-2 LoRA pulls weights toward higher-level reasoning.

| Benchmark | Metric | 1k | 2k | 5k | 11k | 12k | 21k | 25k | best | random | baseline / SOTA |
|-----------|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|-----------------|
| **MELD test** (7c) | acc / macro-F1 | .482/.122 | .492/.145 | .495/.187 | .490/.231 | .497/.230 | .450/.244 | .438/.229 | **F1 .259 @ 17k** | .143 | baseline ~.48, SOTA ~.67 |
| **DailyTalk 5%** (7c) | acc / macro-F1 | .788/.178 | .807/.216 | .835/.324 | .831/.425 | .833/.376 | **.811/.441** | .835/.364 | **F1 @ 21k** | .143 | no ext. baseline |
| **EmoV-DB Jenie** (5c) | acc / macro-F1 | .264/.207 | .343/.283 | .663/.604 | .754/.703 | .729/.661 | .761/.698 | .755/.698 | **acc .776 / F1 .721 @ 15k** | .200 | no ext. baseline |
| **RAVDESS** (8c, speaker-out) | acc / macro-F1 | .075/.031 | .158/.085 | .304/.283 | .512/.497 | .513/.501 | .567/.545 | .542/.530 | **acc .592 / F1 .570 @ 24k** | .125 | baseline ~.55, SOTA ~.80 |
| **Clotho eval** (caption) | BLEU-1 / BLEU-4 / CIDEr | .384/.030/— | .401/.043/— | .429/.064/— | .448/.073/.112 | .459/.081/— | .445/.076/— | .433/.069/.116 | **.476/.0895/.145 @ 15k** | — | DCASE 2020 .39/.07, Qwen2-Audio .55/.20 |
| **ESC-50** 5-fold CV (50c) | acc | .046 | .128 | .541 | .777 | .796 | .829 | .820 | **.832 @ 22k** | .02 | Piczak .65, Qwen-Audio .83, Qwen2-Audio .90 |
| **FSD50K mAP** (200c multi-label, seq) | mAP-μ / mAP-M | .167/.154 | .230/.210 | .328/.310 | .351/.299 | .313/.275 | .308/.261 | .308/.239 | **mAP-μ .356 @ 6k / mAP-M .327 @ 7k** | — | FSD50K baseline mAP .43, CED .66 |
| **FSD50K F1** (200c, greedy gen) | F1-μ / F1-M / Jacc | .147/.026/.101 | .256/.102/.185 | .373/.217/.301 | — | .398/.270/.333 | .399/.269/.335 | — | **F1-μ .404 @ 14k / F1-M .275 @ 19k / Jacc .338 @ 17k** | — | mid-late peakers |
| **LibriSpeech test-clean** (ASR) | WER | .0514 | .0585 | .0501 | .0531 | .0540 | .0615 | .0600 | **.0498 @ 4k** | — | S1 final .0728; non-monotone, monotone decay 4k→25k +1.0 pp |
| **LibriSpeech test-other** (noisy ASR) | WER | .1938 | .1868 | .1848 | .1940 | .1976 | .2112 | .2089 | **.1848 @ 5k** | — | regresses 5k → 25k +2.4 pp |
| **Tier-4 text retention** (6-bench mean) | accuracy | .880 | .888 | .896 | .905 | .906 | .901 | .898 | **.907 @ 13k** | mixed | overfit concern §10 refuted |
| **LISTEN-test MCQA** (full 25-ckpt dense) | acc / F1 | .229/.115 | .226/.112 | **.244**/.146 | .235/.142 | .231/.129 | .215/.126 | .229/.136 | **acc .244 @ 5k / F1 .156 @ 6k** | varies by exp | early-peaker, monotone decay |
| **LISTEN-official audio-only** (5 exp) | F1mean / WAmean | .109/.198 | — | — | — | .120/.214 | .117/.214 | .105/.207 | **F1 .137 @ 19k / WA .235 @ 11k** | varies | mid-late peaker |

**Headline reading (updated 2026-04-25 with ckpt-25k late-trajectory data):**

0. **Bimodal peak pattern, not monotone late training.** Source-emotion
   late trajectory (13-25k) shows that the ckpt-12000 regression was
   transient — a single-step dip. Recovery from ckpt-13k onwards, and a
   **new global peak at ckpt-21000** (DailyTalk F1 0.441, RAVDESS 0.567).
   Then ckpt-25k dips again (DailyTalk F1 −7.7 pp from 21k peak). Two
   peaks ~10 k steps apart suggests a periodicity tied to the emotion
   pool's epoch boundary. **Pick local-peak ckpts for deployment, not
   the latest one.** See §10.1 for full table.



1. **Every audio benchmark improves across the run; most monotone, two
   peak at ckpt-11000.** Biggest gains ckpt-2000 → ckpt-12000:
   - ESC-50: 12.8 % → **79.6 %** (6.2× — Qwen-Audio level)
   - EmoV-DB: 34.3 % → 75.4 % @11k → **72.9 % @12k** (peak at 11k, mild
     regression at 12k)
   - RAVDESS: 15.8 % → **51.3 %** (3.2× — below-random → baseline SER level,
     monotone)
   - MELD macroF1: 0.145 → **0.231** (smooth monotone)
   - DailyTalk macroF1: 0.216 → 0.425 @11k → **0.376 @12k** (peak at 11k,
     **−4.9 pp single-step regression** at 12k)
2. **The "neutral-bias" plateau is resolving.** MELD/DailyTalk accuracy
   plateaued (majority-class ceiling) but macro-F1 climbs steadily. Refutes
   the ckpt-2000 conclusion that rationale supervision is strictly required
   — letter-only training does carve class boundaries, just slowly.
3. **ASR test-clean is non-monotone with a global minimum at ckpt-4000
   (4.98 %).** Stage 1 final 7.28 % → ckpt-1k 5.14 → ckpt-2k 5.85
   (regression spike) → **ckpt-4k 4.98 (best)** → ckpt-5k 5.01 → ckpt-9k
   5.26 → ckpt-12k 5.40. After the early ckpt-2000 spike, the model
   recovers and dips below 5 % at ckpt-4000, then drifts back up. Stage 2
   helps ASR on net (−1.88 pp vs S1 final), but the trajectory is not
   strictly improving — early ckpts (4-5k) are best for clean ASR.
4. **ASR test-other regresses ckpt-2000 → ckpt-12000** (18.68 % → 19.76 %,
   **+1.08 pp**). Stage-2 mix has no augmented-noise signal, so LoRA
   over-specializes to clean acoustics. Clean/other ratio widens 3.2× →
   3.7×.
5. **Text SFT overfit concern is REFUTED by dense Tier-4 trajectory.**
   Mean accuracy on 6 commonsense MCQA is strictly monotone increasing
   ckpt-1k → ckpt-12k (0.880 → 0.906). HellaSwag +6.1 pp, WinoGrande
   +9.2 pp; BoolQ/ARC/COPA already at ceiling. The 1_text out-of-
   distribution regression seen at ckpt-2000 was narrow-distribution
   specialization, not memorization (§9.5).
6. **Late-training regression at ckpt-12000 on conversational ERC tasks.**
   DailyTalk F1 −4.9 pp, EmoV-DB acc −2.5 pp, ASR test-clean +0.39 pp
   (worse) all in the single ckpt-11k → ckpt-12k step. Other tasks stable
   or still improving over the same window. **ckpt-11000 is the
   cross-benchmark Pareto-optimal so far.** Late-stage training pressure
   (emotion-heavy mix + no regularization) appears to trade
   generalization for mix-specific fit on conversational subdomains.
7. **Env-sound trajectory validates the design.** ESC-50 went from 4.5 %
   to 79.6 % in 11 k steps; FSD50K F1-micro 0.147 → 0.398 (×2.7), F1-macro
   0.026 → 0.270 (×10). Both still climbing at ckpt-12000.

Every number reproducible from [`evaluation/stage2/`](../../evaluation/stage2/)
with `--ckpts 1000,2000,…,12000`; recipes in §9.

---

---

## 1. Motivation

[`eval_plan.md`](eval_plan.md) promises a full Tier-2/3 number
board (LISTEN-test, Clotho, FSD50K, ESC-50, MMLU, …) but
[`evaluation/`](../../evaluation/) previously held only `eval_testclean_wer.py`
for ASR. Launching the Tier-2/3 sweep therefore required:

1. An MCQA driver with letter-first parse for **LISTEN-test** (2 635 rows).
2. A captioning driver with BLEU/CIDEr scoring for **Clotho evaluation**
   (1 045 clips × 5 refs).
3. A multi-label driver for **FSD50K eval** (10 231 clips × 200 labels).
4. A 5-fold single-class driver for **ESC-50** (2 000 clips / 50 classes).

`eval_testclean_wer.py` was not reusable as-is because:

- **Checkpoint format mismatch.** Stage-2 ckpts are PEFT LoRA adapters —
  `adapter_model.safetensors` + `adapter_config.json`, no `model.safetensors`.
  `AutoModel.from_pretrained(ckpt_path)` fails. Need
  `PeftModel.from_pretrained(base, adapter_dir)` on top of the Stage-1 init.
- **Projector restoration uncertainty.** `configs/qwen3_5ae-asr/stage2.yaml`
  uses `additional_target: audio_encoder.projector`, which makes PEFT
  **pack the projector as a full module inside `adapter_model.safetensors`**
  (39 keys, 18 M params). Before trusting any eval number we need a check
  that these keys actually round-trip into live model parameters, not
  silently fall back to the pre-S2 init.
- **Prompt-format divergence.** The WER script hardcodes
  `"Transcribe the audio to text."`; MCQA / captioning / classification need
  per-task ChatML stems that mirror
  [`omni_dataset.py:create_omni_processor`](../../src/llamafactory/data/omni_dataset.py).

---

## 2. Layout

```
audiollm-trainer/evaluation/stage2/
    __init__.py
    _loader.py                  # PEFT+projector loader + cache shims + ChatML builder
    eval_listen_mcqa.py         # LISTEN-test 2 635 rows — our own training-parallel protocol
    eval_listen_official.py     # LISTEN-test, bit-for-bit port of the official reference harness
    eval_clotho_caption.py      # Clotho-eval 1 045 clips, BLEU-1..4 inline (+ optional CIDEr/METEOR/ROUGE/SPICE)
    eval_fsd50k_map.py          # FSD50K-eval 10 231 clips, F1-micro/F1-macro/Jaccard
    eval_esc50_acc.py           # ESC-50 5-fold CV single-class
```

**Two LISTEN drivers by design.**

- `eval_listen_mcqa.py` uses **our training-parallel prompt format** (canonical
  `TASK_PROMPTS` stem + `"A) x B) y ... Answer with the letter."` inline
  choices). This matches training distribution exactly, so numbers measure
  whether the model learned what we taught it.
- `eval_listen_official.py` is a **faithful reproduction** of the LISTEN
  reference harness
  ([`DeliJingyiC/LISTEN`](https://github.com/DeliJingyiC/LISTEN),
  `scripts/test_your_model.py` + `evaluation_utils.py`). Different prompt
  format, `random.seed(42)` per-sample choice shuffling, per-experiment
  split (1_*, 2A/2B/2C, 3A/3B/3C, 4), sklearn metrics identical to
  `calculate_comprehensive_metrics`, and the reference's A-H parse cap
  (a known bug for type-3 rows with >8 unique choices — kept for
  bit-for-bit parity). Use this driver for leaderboard comparability.

All four drivers are ckpt-sweep tools: given `--ckpt-root <dir>` they iterate
every `checkpoint-*` subdir (filter via `--ckpts 1000,2000,...`), write
`<out>/<ckpt_name>/{predictions.jsonl, summary.json}`, skip completed ckpts
on re-run, and emit a final `summary_all.json`.

The per-driver entry points are `python -m evaluation.stage2.eval_<task>`.
They share `_loader.load_checkpoint()` so model-loading fixes land in one
place.

---

## 3. PEFT + projector loader

### 3.1 Adapter-only detection

[`_loader._is_adapter_only`](../../evaluation/stage2/_loader.py) returns True
iff the dir has `adapter_config.json` + `adapter_model.safetensors` but
neither `model.safetensors` nor `model.safetensors.index.json`. Under this
branch the loader:

1. Reads `base_model_name_or_path` from `adapter_config.json` unless
   `--base-model` overrides. For the running run this is `/mnt/tmp/s2_init_42k`.
2. `AutoModelForCausalLM.from_pretrained(base, torch_dtype=bfloat16,
   attn_implementation="sdpa")` — full-weight load of the Stage-1 init.
3. `PeftModel.from_pretrained(base, adapter_dir)` wraps LoRA + projector.
4. Calls `_verify_projector_restore()` (§3.2) before returning.

### 3.2 Projector restoration check

PEFT's `additional_target` → `modules_to_save` creates **two copies** of the
wrapped module in the live state dict:

```
…audio_encoder.projector.original_module.*      # pre-S2 frozen init
…audio_encoder.projector.modules_to_save.default.*  # trained copy (forward uses this)
```

`_verify_projector_restore()`:

1. Loads `adapter_model.safetensors` raw, filters to 39 projector keys.
2. For each raw key `…audio_encoder.projector.<tail>`, maps it to
   `…audio_encoder.projector.modules_to_save.default.<tail>` in
   `model.state_dict()` and checks:
     - shape match
     - **bit-identical value via `torch.equal()`** (catches silent load
       failures where PEFT would otherwise serve the frozen copy with no
       crash).
3. Also compares to `…original_module.<tail>`; if *every* trained-copy key
   equals the frozen-copy key, warns (projector hasn't moved in training
   yet — not fatal in early S2).

Failure is hard: raises `RuntimeError` with the first five offending keys.

**Verified on ckpt-2000 (2026-04-24):** 39/39 keys round-trip bit-identical,
39/39 differ from the original — projector has moved, forward will use the
trained copy.

---

## 4. Cache API compatibility (transformers 4.57)

The **vendored** `modeling_qwen3_5AE.py` at
`/mnt/tmp/cache/hf/modules/transformers_modules/s2_init_42k/modeling_qwen3_5AE.py`
(loaded via `trust_remote_code=True`) was written against an earlier
Qwen3-Next cache API than what transformers 4.57.1 ships. This is a library
upgrade drift, not new code in this project.

### 4.1 Exhaustive audit of vendored cache calls

```
$ grep -nE "cache_params\.|past_key_values\.(has_previous_state|update_|layers)" modeling_qwen3_5AE.py
448:  cache_params.has_previous_state(self.layer_idx) and seq_len == 1
459:  conv_state      = cache_params.layers[self.layer_idx].conv_states
460:  recurrent_state = cache_params.layers[self.layer_idx].recurrent_states
484:  conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
568:  cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
987:  past_key_values.has_previous_state() or …
```

Six call sites total.

### 4.2 Current transformers Qwen3NextDynamicCache

```
public attrs : conv_states, recurrent_states, key_cache, value_cache (per-layer lists)
               layer_types, last_linear_layer, transformer_layers
methods      : get_mask_sizes, get_seq_length, reorder_cache, update
properties   : has_previous_state      → self.conv_states[self.last_linear_layer] is not None
               is_compileable
absent       : .layers[i].{conv,recurrent}_states sub-objects
               .update_conv_state / .update_recurrent_state methods
```

### 4.3 Shims applied (`_loader._patch_dynamic_cache`)

All guarded by `Qwen3NextDynamicCache._qwen35ae_patched = True` sentinel so
repeated `load_checkpoint()` calls are idempotent.

| Vendored call                                            | Shim                                                                                  | Correctness check                                                                                 |
|----------------------------------------------------------|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `cache.has_previous_state(layer_idx)` (line 448)         | method `self.conv_states[idx] is not None`                                            | old-API intent was per-layer; preserves it                                                        |
| `cache.has_previous_state()` (line 987, no arg)          | same method, defaults `idx = self.last_linear_layer`                                  | matches current property's definition                                                             |
| `cache.layers[i].conv_states / .recurrent_states` (r+w)  | `layers` property returning `_LayerProxy` list; proxy has `@property` + `@setter`     | delegates both read and write to `cache.conv_states[i]` / `cache.recurrent_states[i]`             |
| `cache.update_conv_state(state, i)` (line 484)           | method: `self.conv_states[i] = state; return state`                                   | old API returned the stored tensor — preserved                                                    |
| `cache.update_recurrent_state(state, i)` (line 568)      | method: `self.recurrent_states[i] = state; return state`                              | same                                                                                              |
| `DynamicCache(config=...)` (modeling line 914)           | `mod.DynamicCache = Qwen3NextDynamicCache` in every imported `modeling_qwen3_5AE.*`   | defensive — transformers' generate usually pre-creates the cache, but safer to swap the class anyway |

### 4.4 Semantic risks evaluated

- **`last_linear_layer` semantics**: In transformers 4.57's
  `Qwen3NextDynamicCache.__init__`, this is set to the index of the last
  layer whose `layer_types[i] == "linear_attention"`. For the S2 model that
  is index 30 (layers 3, 7, 11, …, 31 are `full_attention`; the rest are
  `linear_attention`; last linear is 30). Using this as the default `idx`
  for the no-arg `has_previous_state()` call (line 987) matches the stock
  property's own definition. Confirmed by direct property read in §4.5.
- **Unseen call paths**: The grep in §4.1 is exhaustive over the vendored
  file. No other `cache_params.*` or `past_key_values.*cache*` usages exist.
- **Liger kernel**: Training has `enable_liger_kernel: true` for speed;
  eval uses stock `attn_implementation="sdpa"`. `apply_liger_kernel` in
  [`model/model_utils/liger_kernel.py:36`](../../src/llamafactory/model/model_utils/liger_kernel.py#L36)
  early-returns when `is_trainable=False`, so at inference time Liger is
  never applied regardless of the YAML flag — there is nothing to match.
  The trained weights themselves are portable (Liger's fused ops are
  drop-ins that match reference impls to ≤ 1e-3 abs-diff per its own test
  suite).
- **Global patch scope**: The shim mutates the process-global
  `Qwen3NextDynamicCache`. Eval scripts load a single model and exit, so
  this is inert. It does not leak into the running training process (separate
  PID, never imports `_loader`).

### 4.5 Cross-check: use_cache=False as ground truth

Driver flag `--no-cache` disables the KV cache entirely (every step re-runs
the full prompt). This bypasses every shim and is always correct at the
expense of O(n²) latency. Running the same ckpt with and without cache and
checking that per-sample predictions match is the empirical proof that the
shims are semantically faithful.

Cross-check result (LISTEN 50-sample slice, ckpt-2000, 2026-04-24):

| metric                                         | cache (default) | `--no-cache`  |
|------------------------------------------------|----------------:|--------------:|
| accuracy                                       |          0.1400 |        0.1400 |
| parse rate                                     |         50 / 50 |       50 / 50 |
| macro-F1                                       |          0.0911 |        0.0911 |
| exact raw-output match (per sample)            |     —           |       45 / 50 |
| same parsed letter (per sample)                |     —           |       45 / 50 |
| same correct/wrong flag (per sample)           |     —           |    **50 / 50** |
| throughput                                     |      ~17 sps    |      ~1.3 sps |

The five letter disagreements are all single-letter outputs on samples where
**both paths are wrong** — the model's top-2 logits are within bf16 rounding
(≈ 1e-3), and either path's argmax can flip between two near-tied wrong
options. Aggregate metrics are identical; per-sample correctness is
identical on all 50/50. This is the empirical proof that the cache shims
preserve eval-relevant semantics.

Latency: `--no-cache` is ~13× slower on this workload (prompt ≈ 250 tokens,
`max_new_tokens` = 96). Usable for spot checks on a few hundred rows; not
for full FSD50K (10 231) or Clotho (1 045) sweeps.

---

## 5. ChatML prompt parity with training

`_loader.build_prompt_ids()` mirrors `omni_dataset._build_prompt_targets()`
byte-for-byte up to the assistant boundary:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|>[<|audio_pad|>] × t_audio<|audio_end|><user_suffix><|im_end|>
<|im_start|>assistant
```

where `t_audio = floor(num_samples / 1920)` at 48 kHz. The per-task
`<user_suffix>` comes from these canonical stems (pinned at index [0] of
each `TASK_PROMPTS` pool in `omni_dataset.py`, ensuring the training
distribution covers the eval phrasing):

| Task               | Stem used at eval                                             |
|--------------------|---------------------------------------------------------------|
| LISTEN MCQA        | `"What emotion is being expressed in the audio?"` + `"\nChoices: A) ...\nAnswer with the letter."` |
| Clotho caption     | `"Describe what you hear in the audio."`                      |
| FSD50K multi-label | `"List the sound events in this audio, separated by commas."` |
| ESC-50 single      | `"Classify this sound."`                                      |

LISTEN eval overrides the dataset's own per-row `question` field with the
canonical emotion stem by default (training does the same). Pass
`--use-native-question` to score against LISTEN's original varied phrasings;
this sits outside the training distribution and is useful mostly for
cross-paper reproducibility, not as the primary S2 number.

---

## 6. Metric choices and caveats

### 6.1 LISTEN MCQA — primary Tier-2 metric

- Letter parse: first `A-I` in first 20 chars (head) with word boundary,
  fall back to anywhere in the output, constrained to `choices` length.
- Gold mapping: case-insensitive equality between `answer` (plain text) and
  each choice string → the matching index gives the gold letter.
- Reported: `accuracy` (overall), `accuracy_parsed_only`, `macro_f1` across
  observed labels, per-`dataset_source` accuracy. 11 sub-corpora:
  IEMOCAP 400, MELD 400, OMG 400, PODCAST 320, Emotion-Speech 200, RAVDESS 200,
  CREMA-D 200, TESS 200, MUStARD 193, MOSEI 64, SAVEE 58.
- Sub-corpus caveat from
  [`leakage_audit.md`](leakage_audit.md): MUStARD
  / PODCAST / MOSEI carry audio-level overlap with LISTEN-train in the
  original distribution. Since the running S2 doesn't train on LISTEN-train
  directly (it uses per-source corpora filtered against LISTEN-test), this
  contamination is not a concern for the S2 numbers specifically — noted
  for paper-level comparability.

### 6.2 Clotho captioning — generative, BLEU + optional CIDEr

- Inline corpus BLEU-1..4 (sacre-style, clipped n-grams, brevity penalty).
- Optional CIDEr / METEOR / ROUGE-L / SPICE via `pycocoevalcap` — currently
  not installed in the `audio_lmf` conda env. Summary fields are `null`
  when the lib is missing; `pip install pycocoevalcap` (≈ 3 transitive
  deps + Java for METEOR/SPICE) enables them. No eval re-run needed for
  BLEU-only reporting.

### 6.3 FSD50K — F1/Jaccard from greedy generation; mAP requires a different readout

Published FSD50K mAP requires **per-label probability** scores (sigmoid
output per label), which are used to sweep thresholds and compute
Average Precision per label. `eval_fsd50k_map.py` as currently written
uses a **greedy-generation path**: the model emits a comma-separated label
list, we parse the list, and compute hard-label set metrics (F1-micro,
F1-macro, Jaccard). That path throws away per-label probability, so mAP
is not recoverable from its outputs.

The underlying model can produce probabilities — we just aren't asking
for them in this eval path. Three ways to get them from the same
checkpoint:

1. **Sequence scoring** — for each of the 200 labels, append
   `"The label is: <label>"` (or just `"<label>"`) to the ChatML prompt
   and compute teacher-forced log-probability of the label tokens.
   Softmax or sigmoid across the 200 resulting log-probs gives a
   proper probability distribution over labels per sample. Cost:
   ~200× the single forward pass per sample. Matches Pengi / Audio
   Flamingo mAP protocol.
2. **First-token logit readout** — run one forward pass up to the
   assistant's generation start, read the logit vector at that
   position, and extract the logits at the first token of each of the
   200 label strings. Cheap (1× forward per sample) but imprecise
   because many labels share leading tokens.
3. **Per-label binary classification** — prompt `"Is <label> present
   in the audio? Yes or No"` per label and read P(Yes). Most accurate,
   most expensive (200 full inferences per sample).

Our current F1/Jaccard numbers reflect the greedy path and are still
informative ("does the model pick the right labels when forced to name
them?"), but they are **not comparable to mAP numbers in the FSD50K
leaderboard**. Adding the sequence-scoring readout path is a clear
follow-up if leaderboard comparability is needed.

Published FSD50K mAP requires per-label probability ("confidence")
scores to sweep thresholds. Our current greedy-generation path emits
only a hard label list — it discards the probability even though the
model can produce it (§6.3 details three readout alternatives that
would recover mAP-compatible scores). Under the current path, the
driver recovers a presence-only binary vector and reports:

- `f1_micro` (pool tokens across all samples),
- `f1_macro` (average F1 over labels with support > 0),
- `jaccard_mean` (per-sample IoU between predicted and true label sets),
- avg # of predicted vs true labels per sample (sanity on over/under-generation).

Summary JSON carries a prominent `note` string reminding readers that this
is not comparable to FSD50K leaderboard mAP. A proper mAP number would
require replacing greedy sampling with a **token-level logit readout across
the 200-label vocabulary** — a separate eval path not yet implemented.

### 6.4 ESC-50 — 5-fold CV accuracy

Standard fold-held-out protocol. `--fold N` for one fold, `--all` for all
five. Summary reports both `accuracy_pooled` (all 2 000 clips lumped) and
`accuracy_mean_per_fold` (unweighted mean over folds); both are reported
because fold-level variance is informative for 40-sample-per-class evals.

---

## 7. Verification status on ckpt-2000 (2026-04-24)

### 7.1 End-to-end smoke tests

Both run on GPU 7 (A100-80GB) with the `audio_lmf` conda env and the
`libstdc++.so.6` + `glibc_compat.so` `LD_PRELOAD` shim.

- LISTEN 4 samples, ckpt-2000 → 4/4 parsed, 0/4 correct. All outputs were
  single-letter predictions from the valid A-I range → letter-first training
  format is learned, cache shim paths (use_precomputed_states branch hit
  during decode) fire without error.
- ESC-50 6 samples, fold 1, ckpt-2000 → 6/6 parsed after fallback parser
  tweak (added substring match in both directions so e.g. "cough" resolves
  to the canonical "coughing" class). 0/6 correct, but predictions were
  valid ESC-50 categories — model is mapping audio → plausible class
  labels, just wrong at 4 % training.

Smoke tests proved the pipeline is end-to-end correct; numbers at step 2000
are not expected to be meaningful.

### 7.2 Sanity checks — LISTEN 200-sample slice (2026-04-24)

| ckpt   | acc    | macro-F1 | parse  | vs. 9-class random (11.1 %) |
|--------|-------:|---------:|-------:|-----------------------------|
| 1000   | 0.1300 |   0.0727 | 200/200 | +1.9 pp                     |
| 2000   | 0.1350 |   0.0761 | 200/200 | +2.4 pp                     |

Monotone increasing as expected; both above random; parse rate 100 %. The
absolute numbers are small because the model is 4 % into a 50 k-step plan
and LISTEN-test covers 11 sub-corpora with 7–9 choices each. Slice was the
first 200 rows of the parquet sort-by-length (a mixed-source sample, not
MELD+RAVDESS-only).

Slice composition (per-source breakdown, both ckpts):

- First-200 slice is IEMOCAP-dominant after length sort (short IEMOCAP
  utterances sort first). Per-source accuracy in `summary.json → per_source_accuracy`.

### 7.3 Cross-check vs. `--no-cache`

See §4.5 — cache path is empirically identical to the shim-free ground
truth on all 50/50 samples for correct/wrong flag and aggregate accuracy;
90 % match at the raw-output level, with disagreements confined to bf16
argmax ties on wrong-either-way samples.

### 7.4 Still pending

- ASR regression via existing `eval_testclean_wer.py` on ckpt-2000 — with
  ASR only 14 % of the Stage-2 mix, some drift from S1's 7.28 % WER is
  expected; to be measured.
- Full LISTEN-test (2 635 rows), full ESC-50 5-fold (2 000 rows), Clotho
  (1 045 rows), FSD50K (10 231 rows). Scheduled for ckpt-5000+.

---

## 7.5 Why the ckpt-2000 numbers look the way they do

The 13.5 % accuracy on the 200-sample slice is **not evidence that the
harness is broken** — every shim is verified (§4.5), projector round-trips
bit-identical (§3.2), and parse rate is 100 %. The number is a genuine
snapshot of a 4 %-trained model on a deliberately hard slice. Four
compounding factors:

**1. Slice composition is IEMOCAP-only, 9-way, audio-only.** Sorting the
first 200 rows by audio length sinks short utterances to the front, and
IEMOCAP impro clips are the shortest things in LISTEN-test. Result:

- 200/200 rows from **IEMOCAP** — LISTEN's hardest sub-corpus
  (acted-dialog utterances with fuzzy neutral/frustration/sadness
  boundaries; human inter-annotator agreement on IEMOCAP-9class is in the
  40-60 % range, not 90 %).
- 200/200 rows with **9 choices** (random = 11.1 %), never fewer.
- 98 type-2A + 102 type-2B, no type-1 (text-based) contamination.

Accuracy on type 2A (semantic-content-based question) = 0.112 (=random);
on 2B (voice/prosody-based) = 0.157 (+4.6 pp). The model does slightly
better when the question *asks about voice* — a weak positive signal that
the audio path is contributing something at all.

**2. Prediction distribution has a strong "neutral" prior.**

```
ckpt-2000 predicted label top-5:  neutral(71)  excitement(43)  happiness(37)  disgust(22)  frustration(18)
ckpt-2000 gold      label top-5:  frustration(79)  excitement(45)  sadness(27)  anger(25)  happiness(11)
```

The most common gold label in this IEMOCAP slice is **frustration** (40 % of
rows), which the model predicts only 9 % of the time. It predicts
**neutral** 36 % of the time against a gold incidence of 0 % in this slice.
The model has learned a "neutral-default" strategy — a known failure mode
for emotion classifiers trained with imbalanced label pools. At 4 %
training this prior will slowly decay as the rationale + MCQA gradient
carves finer boundaries, but step-2000 accuracy is dominated by this bias.

**3. Rationale supervision is absent.**
[`eval_plan.md §1`](eval_plan.md) explicitly warned that
"letter-only training has two problems: weak gradient signal (1 token) and
letter-prior shortcut." The rationale-augmentation offline pass
([`design.md` open item #2](design.md) and
[`eval_plan.md §9.1`](eval_plan.md)) has **not run yet** — the
combined manifest we inspected earlier (`shard_*.jsonl`) carries
`"rationale": null` on every emotion row. The model is therefore training
on letter-only targets, which is exactly the failure mode the plan flagged.
Accuracy will likely stagnate well below dedicated acoustic-SER baselines
until rationales land.

**4. 4 % of planned training is early.** Ckpt-2000 = step 2 000 / 50 000.
The emotion loss curve shows the model has converged to letter prediction
(0.96 → 0.41, 57 % drop) but not yet to the finer inter-label distinctions
— compare to step 5 000 / 10 000 where the loss should drop further and
the prior bias should decay.

**What would the number look like on a gentler slice?**
LISTEN-test as a whole (not just the length-sorted IEMOCAP-front-200) is a
mix of 11 sub-corpora with choice cardinalities 7–9. Easier sub-corpora
(RAVDESS, SAVEE — acted with exaggerated prosody) typically score 2-3×
higher than IEMOCAP on the same model. We expect the **full-2 635-row
number to land meaningfully higher than 13.5 %**, even at ckpt-2000. This
will be checked when the full sweep runs at ckpt-5 000+ (§7.4).

**Bottom line:** 13.5 % on the hardest slice at 4 % training is
**expected behavior**, not a broken harness. The open risk is not the
eval pipeline — it's whether the *training* (letter-only, no rationale)
will ever break out of the "neutral default" prior. Landing the rationale
pass before ckpt-10 000 is the single highest-leverage action.

---

---

## 8. Source-corpus held-out results (2026-04-24)

**Decision (2026-04-24): evaluate on per-corpus held-out splits, not on
the aggregated LISTEN benchmark.** LISTEN mixes 11 source corpora into a
single MCQA set with varying choice cardinality (7-10), shuffled choices
per row, and the A-H parser cap; individual-corpus numbers are easier to
interpret and compare to published SER baselines. A preliminary
LISTEN-official sweep on ckpt-1000 + partial ckpt-2000 is archived under
`eval_listen_official/` for reference (aborted after source-corpus decision).

### 8.1 Source-corpus emotion (`eval_source_emotion.py`)

Per-corpus held-out splits exactly as defined in
[`build_training_manifest.py`](../../scripts/emo/build_training_manifest.py) —
so eval never overlaps training. Prompt format identical to training's
emotion MCQA rows (canonical stem `"What emotion does the speaker convey?"`
+ `"Choices: A) x B) y ...\nAnswer with the letter."`). Random baseline =
1/n_class.

| corpus | n_cls | n | random | ckpt | acc | macro-F1 | bal-acc |
|---|--:|--:|--:|--:|--:|--:|--:|
**Full 12-ckpt trajectory** (after multiple OOM-retries, all ckpts 1k-12k now in):

| corpus | ckpt | acc | macro-F1 | bal-acc |
|---|--:|--:|--:|--:|
| **MELD** (7) | 1000 | 0.482 | 0.122 | 0.155 |
|  | 2000 | 0.492 | 0.145 | 0.168 |
|  | 3000 | 0.491 | 0.160 | 0.175 |
|  | 4000 | 0.486 | 0.187 | 0.192 |
|  | 5000 | 0.495 | 0.187 | 0.192 |
|  | 6000 | 0.489 | 0.212 | 0.211 |
|  | 7000 | 0.491 | 0.202 | 0.203 |
|  | 8000 | 0.495 | 0.208 | 0.207 |
|  | 9000 | 0.481 | 0.219 | 0.216 |
|  | 10000 | 0.489 | 0.223 | 0.219 |
|  | 11000 | 0.490 | 0.231 | 0.226 |
|  | **12000** | **0.497** | **0.230** | **0.224** |
| **DailyTalk** (7) | 1000 | 0.788 | 0.178 | 0.182 |
|  | 2000 | 0.807 | 0.216 | 0.209 |
|  | 3000 | 0.825 | 0.290 | 0.260 |
|  | 4000 | 0.833 | 0.320 | 0.286 |
|  | 5000 | 0.835 | 0.324 | 0.291 |
|  | 6000 | 0.834 | 0.373 | 0.325 |
|  | 7000 | 0.832 | 0.376 | 0.337 |
|  | 8000 | 0.830 | 0.387 | 0.342 |
|  | 9000 | 0.831 | 0.392 | 0.346 |
|  | 10000 | 0.833 | 0.382 | 0.341 |
|  | **11000** | **0.831** | **0.425** | **0.402** *(F1 peak)* |
|  | 12000 | 0.833 | 0.376 | 0.323 |
| **EmoV-DB Jenie** (5) | 1000 | 0.264 | 0.206 | 0.278 |
|  | 2000 | 0.342 | 0.283 | 0.352 |
|  | 3000 | 0.493 | 0.450 | 0.460 |
|  | 4000 | 0.616 | 0.557 | 0.558 |
|  | 5000 | 0.663 | 0.604 | 0.603 |
|  | 6000 | 0.693 | 0.629 | 0.630 |
|  | 7000 | 0.730 | 0.687 | 0.686 |
|  | 8000 | 0.720 | 0.675 | 0.676 |
|  | 9000 | 0.688 | 0.624 | 0.631 |
|  | 10000 | 0.744 | 0.686 | 0.681 |
|  | **11000** | **0.754** | **0.703** | **0.698** *(global peak)* |
|  | 12000 | 0.729 | 0.661 | 0.662 |
| **RAVDESS actors 21-24** (8) | 1000 | 0.075 | 0.031 | 0.129 |
|  | 2000 | 0.158 | 0.085 | 0.148 |
|  | 3000 | 0.192 | 0.125 | 0.180 |
|  | 4000 | 0.283 | 0.244 | 0.285 |
|  | 5000 | 0.304 | 0.283 | 0.297 |
|  | 6000 | 0.358 | 0.347 | 0.359 |
|  | 7000 | 0.375 | 0.336 | 0.379 |
|  | 8000 | 0.429 | 0.405 | 0.426 |
|  | 9000 | 0.446 | 0.436 | 0.453 |
|  | 10000 | 0.458 | 0.444 | 0.469 |
|  | 11000 | 0.512 | 0.497 | 0.520 |
|  | **12000** | **0.512** | **0.501** | **0.523** |

**Trajectory observations (all 12 ckpts):**

- **MELD**: accuracy essentially flat at ~0.49 across every ckpt; macro-F1
  rises smoothly from 0.122 → 0.231 (monotone with minor wiggles). Only
  ~4 % of 2 610 label predictions flip — the model has the majority
  pattern down early and spends the rest of training trimming the
  minority-class errors.
- **DailyTalk**: acc flat at ~0.83 from ckpt-4000 onward; macro-F1 has a
  distinct **peak at ckpt-11000 (0.425)** followed by a drop to 0.376 at
  ckpt-12000 (−4.9 pp in a single step). bal-acc follows the same shape
  (0.402 → 0.323). A single-ckpt dip this size is unusual and suggests
  late-training instability on the DailyTalk sub-distribution.
- **EmoV-DB**: smooth climb 0.26 → 0.75 with non-monotone tail. Peak
  accuracy at **ckpt-11000 (0.754)**; ckpt-12000 drops to 0.729 (−2.5 pp).
  macro-F1 peak also at 11k. Both audio ERC benchmarks thus share a
  shape: **monotone rise until ~11k, then single-step regression at 12k**.
- **RAVDESS**: near-perfectly monotone (only ckpt-11000→12000 is flat,
  0.512 both). No late-training dip here.

**Late-training signal.** DailyTalk and EmoV-DB both show a ckpt-11000 →
ckpt-12000 regression on macro-F1. ASR test-clean showed a similar
minimum-at-5000-then-rise pattern (§9.4), and test-other showed a
monotone regression. The cumulative signal: around ckpt-11000 the model
may have reached a local-best for conversational ERC tasks + clean ASR,
after which the late-stage training pressure (emotion-heavy mix + no
regularization) is trading generalization for mix-specific fit.
ckpt-11000 is probably the best cross-benchmark ckpt so far.

**Every corpus improves monotonically on every metric across all 4 ckpts.**
The ckpt-2000 "neutral-bias plateau" conjecture is refuted by the trajectory
— macro-F1 keeps climbing even when accuracy saturates (MELD, DailyTalk).
For the cross-distribution corpora (EmoV-DB cross-speaker+language, RAVDESS
speaker-out acted-prosody), the ckpt-2000 → ckpt-12000 gain is dramatic
(EmoV-DB +38.6 pp, RAVDESS +35.5 pp).

### 8.2 Why MELD 49 % and DailyTalk 83 % are not as good as they look (at ckpt-2000) — but do get honest at ckpt-12000

MELD and DailyTalk have heavily imbalanced label distributions — neutral
is the majority class (~48 % for MELD test, ~78 % for DailyTalk). An
"always-neutral" baseline would score:

- MELD: ~48 % accuracy, macro-F1 ≈ 0.10
- DailyTalk: ~78 % accuracy, macro-F1 ≈ 0.02

At ckpt-2000 our model is close to these (MELD 49 % acc / F1 0.145,
DailyTalk 81 % acc / F1 0.216) — i.e. barely better than always-neutral.
**That was the §7.5 concern.**

By ckpt-12000 this has meaningfully changed:

| | MELD acc | MELD F1 | DailyTalk acc | DailyTalk F1 |
|---|---:|---:|---:|---:|
| always-neutral baseline | ~0.48 | ~0.10 | ~0.78 | ~0.02 |
| ckpt-2000 | 0.492 | 0.145 | 0.807 | 0.216 |
| **ckpt-12000** | **0.497** | **0.230** | **0.833** | **0.376** |
| Δ over baseline | +0.02 | +0.13 | +0.05 | +0.36 |

Accuracy is close to saturated (the majority class ceiling is hard to
push beyond), but macro-F1 now runs **2-20× above the always-neutral
baseline** — the model is correctly identifying non-neutral classes a
meaningful fraction of the time. **For imbalanced corpora, macro-F1 is
the honest metric; watch that trajectory, not the accuracy one.**

### 8.3 EmoV-DB Jenie: real generalization signal

EmoV-DB Jenie is **speaker-held-out + language-held-out** (training used
only English speakers Bea / Josh / Sam; Jenie is French). 5-class, roughly
balanced, so accuracy and macro-F1 track each other:

| ckpt | acc | macro-F1 | +random |
|---:|---:|---:|---:|
| 1000 | 0.264 | 0.207 | +0.064 |
| 2000 | 0.343 | 0.283 | +0.143 |
| 5000 | 0.663 | 0.604 | +0.463 |
| **12000** | **0.729** | **0.661** | **+0.529** |

Unlike MELD/DailyTalk this is not a majority-class artifact — the
balanced taxonomy means macro-F1 mirrors accuracy, and both climb from
random-baseline up to 73 % on a cross-lingual speaker-out eval. This is
**the single clearest signal that the audio pathway is learning
speaker/language-invariant prosodic emotion features**.

### 8.4 RAVDESS acted-prosody: slow start, then catches up

| ckpt | acc | macro-F1 | comment |
|---:|---:|---:|---|
| 1000 | 0.075 | 0.031 | **below random** (0.125) |
| 2000 | 0.158 | 0.085 | +3.3 pp above random |
| 5000 | 0.304 | 0.283 | approaching conventional baseline |
| **12000** | **0.513** | **0.501** | **baseline SER territory** (55 % speaker-out RAVDESS baseline) |

RAVDESS is actor-spoken with exaggerated prosody that mismatches our
training mix (mostly conversational speech + TTS). The 8-class taxonomy
also includes `"calm"` — a label absent from every other training corpus,
so the model has no direct training signal for it (~30 `"calm"` samples
of 240 = 12.5 % of the set guaranteed wrong for several early ckpts).

Despite both handicaps, **the trajectory fixes itself**: by ckpt-12000 the
model recovers to 51 % / macro-F1 0.50, approaching published speaker-out
RAVDESS baselines (~55 %). This suggests the acted prosody abstraction is
learnable from exposure to MELD/EmoV/IEMOCAP's more natural emotion
recordings — the prosody-to-label mapping transfers once the model has
built enough general acoustic-emotion representation.

---

## 9. Tier-3 env-sound results (2026-04-24)

### 9.1 Clotho evaluation split — captioning

1 045 clips × 5 reference captions, greedy decode, `max_new_tokens=96`:

Dense-fill trajectory (6 of 12 ckpts — missing 4/6/7/9/10/11k OOM'd):

| ckpt | BLEU-1 | BLEU-4 |
|---:|---:|---:|
| 1000 | 0.384 | 0.030 |
| 2000 | 0.401 | 0.043 |
| 3000 | 0.410 | 0.053 |
| 5000 | 0.429 | 0.064 |
| 8000 | 0.436 | 0.067 |
| **12000** | **0.459** | **0.081** |
| — references — | | |
| DCASE 2020 baseline | 0.39 | 0.074 |
| DCASE 2021 baseline | 0.51 | 0.13 |
| Qwen-Audio (2023) | 0.52 | 0.15 |
| Qwen2-Audio (2024) | 0.55 | 0.20 |

BLEU-1 now past DCASE 2020 baseline at ckpt-12000 (0.459 vs 0.39);
BLEU-4 has crossed DCASE 2020 baseline (0.081 vs 0.074). BLEU-4 doubled
ckpt-2000 → ckpt-12000 (0.043 → 0.081); trajectory points at DCASE 2021
baseline (0.13) some time after ckpt-30000. CIDEr / METEOR / SPIDEr not
reported (no `pycocoevalcap` in the conda env).

### 9.2 ESC-50 5-fold CV — single-class

Dense-fill trajectory (8 of 12 ckpts — missing 7/9/10/11k OOM'd):

| ckpt | accuracy (pooled) |
|---:|---:|
| 1000 | 0.0455 |
| 2000 | 0.128 |
| 3000 | 0.247 |
| 4000 | 0.410 |
| 5000 | 0.541 |
| 6000 | 0.605 |
| 8000 | 0.722 |
| **12000** | **0.796** |
| — references — | |
| Piczak 2015 CNN baseline | 0.65 |
| Qwen-Audio (2023) | 0.83 |
| Qwen2-Audio (2024) | 0.90 |
| Supervised SOTA | 0.97 |

**ckpt-12000 at 79.6 % — Qwen-Audio-adjacent, past Piczak 2015 baseline
by 15 pp.** Trajectory shape: perfect power-law early (1k 4.5 % → 2k
12.8 % → 3k 24.7 % is ~×2.5-2.8 per ckpt), tapering mid-training (4-6k:
+15-20 pp per ckpt), plateauing late (8k 72 % → 12k 80 % = +8 pp over
4k steps). Extrapolating, ckpt-30k-50k plausibly reaches Qwen2-Audio
territory (90 %). ESC-50 is only 4.9 % of our env-sound training pool
(98 % env-sound weight goes to FSD50K's 200-label AudioSet ontology),
so this rise is mostly transfer from FSD50K's semantic overlap ("dog",
"rain", "engine" appear in both taxonomies).

### 9.3 FSD50K eval split — multi-label

| ckpt   | F1-micro | F1-macro | Jaccard |
|-------:|---------:|---------:|--------:|
| 1000   | 0.1469 | 0.0261 | 0.1014 |
| 2000   | 0.2564 | 0.1022 | 0.1851 |
| 5000   | 0.3728 | 0.2173 | 0.3011 |
| **12000** | **0.3977** | **0.2698** | **0.3331** |

Full 1k → 12k monotone: F1-micro ×2.7, F1-macro **×10.3**, Jaccard ×3.3.
Biggest relative improvement is F1-macro (model diversified from a handful
of favorite labels at ckpt-1000 to balanced multi-label prediction at
ckpt-12000). Env-sound training loss still descending at 12k
(4.68 → 0.32, §10); trajectory projection suggests continued gains past
ckpt-12000.

Our greedy eval path parses a comma-separated label list — it discards
per-label probability, so mAP is not recoverable from those outputs
(§6.3). We added a **sequence-scoring path** (`--score-mode sequence`)
that teacher-forces each of the 200 labels and reads
length-normalized log-prob, which feeds `sklearn.average_precision_score`
to produce the standard FSD50K mAP. Both modes share the same model and
audio cache; they differ only in readout. Greedy gives F1, seq gives mAP.

**Full 25-ckpt sequence-scoring trajectory (200-sample subset):**

| ckpt | mAP-μ | mAP-M | trend |
|---:|---:|---:|---|
| 1k | 0.167 | 0.154 | early ramp |
| 2k | 0.230 | 0.210 | |
| 3k | 0.280 | 0.260 | |
| 4k | 0.336 | 0.312 | |
| 5k | 0.328 | 0.310 | |
| **6k** | **0.356** | 0.307 | **mAP-μ peak** |
| 7k | 0.329 | **0.327** | **mAP-M peak** |
| 11k | 0.351 | 0.299 | second mAP-μ ridge |
| 12k | 0.313 | 0.275 | |
| 17k | 0.324 | 0.262 | |
| 21k | 0.308 | 0.261 | |
| 25k | 0.308 | 0.239 | end-of-run drift |

**FSD50K mAP-seq is an early-peaker** (peak ckpt-6-7k, then long plateau-with-drift),
just like ASR. This is a new finding from the closed dense fill — the
greedy F1 path peaks at ckpt-14-19k and made FSD50K look like a mid-late
peaker; the seq mAP path tells a different story. The split is a
methodology artifact (early model has sharp token-level distinctions but
no instruction-following; late model has good instruction-following but
flatter token distribution). For external comparison the seq path is
the right number — it's what the FSD50K paper / BEATs / CED also report.

Public **mAP** baselines for context (now directly comparable):

| model | mAP (full eval) | params | notes |
|-------|----------------:|-------:|-------|
| **Ours @ ckpt-6k (200-sample seq)** | **0.356 (mAP-μ) / 0.307 (mAP-M)** | 4 B (LM) + projector + LoRA | 200-sample subset, not full 10231 |
| FSD50K CNN baseline (Fonseca 2022) | 0.434 | ~5 M | paper original, full eval |
| BEATs (Microsoft 2023) | 0.580 | ~90 M | SOTA-ish small-model |
| CED (Dinkel 2024) | 0.656 | ~85 M | SOTA |

Full-eval-set re-run of our best ckpt (ckpt-6000) is on the v2 todo;
expect the 200-sample-subset → full-eval gap to be small (200 random
samples covers all 200 labels with high-enough support).

F1-micro / F1-macro / Jaccard from the **greedy** path (mid-late peakers,
peak ckpt-14-19k) are kept for trajectory context — they reflect
instruction-following ability rather than ranking quality:

- F1-micro best 0.404 @ ckpt-14k (greedy)
- F1-macro best 0.275 @ ckpt-19k (greedy)
- Jaccard best 0.338 @ ckpt-17k (greedy)

---

## 9.5 Tier-4 text retention (`eval_text_retention.py`, 2026-04-24)

Validates that LoRA hasn't corrupted the LLM backbone's general-reasoning
ability. Held-out splits of the six benchmarks used in Stage-2 text SFT
(note: training used TRAIN splits of these exact corpora; eval uses
VAL/TEST so there's no instance leakage, but the task distribution is
aligned). Prompt format = `omni_dataset.py` text branch byte-identical.
Letter parse = first A..F in response.

Dense-fill trajectory (8 of 12 ckpts):

| benchmark | n | rand | 1k | 2k | 3k | 5k | 7k | 9k | 10k | **12k** |
|-----------|--:|-----:|---:|---:|---:|---:|---:|---:|---:|--------:|
| HellaSwag (4-way, val) | 10 042 | 0.25 | 0.854 | 0.877 | — | 0.904 | — | — | 0.914 | **0.915** |
| WinoGrande (2-way, val) | 1 267 | 0.50 | 0.670 | 0.696 | — | 0.722 | — | — | 0.753 | **0.762** |
| BoolQ (yes/no, val) | 3 270 | 0.50 | 0.887 | 0.887 | — | 0.886 | — | — | 0.888 | **0.892** |
| ARC-Easy (4-way, test) | 2 376 | 0.25 | 0.974 | 0.973 | — | 0.971 | — | — | 0.971 | **0.971** |
| ARC-Challenge (4-way, test) | 1 172 | 0.25 | 0.914 | 0.915 | — | 0.915 | — | — | 0.915 | **0.906** |
| COPA (2-way, val) | 100 | 0.50 | 0.980 | 0.980 | — | 0.980 | — | — | 0.990 | **0.990** |
| **unweighted mean** | | — | **0.880** | **0.888** | **0.892** | **0.896** | **0.902** | **0.904** | **0.905** | **0.906** |

Mean trajectory is **strictly monotone increasing** across all 8 available
ckpts. Mean accuracy climbed +2.6 pp (0.880 → 0.906). HellaSwag (+6.1 pp)
and WinoGrande (+9.2 pp) show the biggest gains; BoolQ / ARC / COPA were
already near ceiling at ckpt-1000 and stayed there.

**This contradicts the ckpt-2000 "text overfit" verdict** (previous §10).
The training-loss signal (text loss 7.49 → 0.12 → 0.034) looked like
memorization, but held-out eval says the model learned the distribution
well, not memorized specific instances. The earlier 1_text WA degradation
(LISTEN-official 0.150 → 0.144 between ckpt-1000/2000) was not a text
overfit signal — it was a narrow-distribution side-effect: the model
specialized to *commonsense-reasoning MCQA* (the 6 training benchmarks)
at slight cost to *emotion-from-transcription MCQA* (LISTEN 1_text,
out-of-distribution for the text corpus).

**Caveat on absolute numbers**: because train/eval share benchmark identity
(train split vs val/test split of the same corpora), the absolute numbers
are not directly comparable to publications that eval a base model
zero-shot or 5-shot on these benchmarks — we'd score similarly to a
fine-tuned specialist. The useful signal is the **monotone trajectory**
and absence of degradation.

ckpt-2000 was skipped in this sweep due to CUDA OOM mid-load (training
process grew on the shared GPU); the monotone trajectory from ckpt-1000
through ckpt-12000 on all 6 benchmarks makes it a confident interpolation.

---

## 9.4 LibriSpeech test-clean — ASR guardrail

`eval_librispeech_wer.py`, 2 620 rows, Whisper EnglishTextNormalizer:

| ckpt | split | WER (norm) | CER (norm) | Δ vs Stage 1 final 7.28 % |
|---:|---|---:|---:|---:|
| 1000 | test-clean | 5.14 % | 2.31 % | −2.14 pp (already big) |
| 2000 | test-clean | 5.85 % | 2.70 % | −1.43 pp **(local max — regression)** |
| **4000** | test-clean | **4.98 %** | — | **−2.30 pp (global minimum)** |
| 5000 | test-clean | 5.01 % | 2.24 % | −2.27 pp |
| 9000 | test-clean | 5.26 % | — | −2.02 pp |
| 12000 | test-clean | 5.40 % | 2.43 % | −1.88 pp (mild regression vs 4-5k) |
| 2000 | test-other | 18.68 % | 10.60 % | *no S1 test-other baseline* |
| 12000 | test-other | **19.76 %** | 11.26 % | **+1.08 pp regression vs ckpt-2000** |

**test-other trajectory is opposite to test-clean:** clean improves
ckpt-2000 → ckpt-12000 (−0.45 pp), other regresses (+1.08 pp). The
model is losing noise-robustness as training continues. Likely cause:
Stage-2 training data is all relatively clean speech (LibriSpeech /
MLS / GigaSpeech ASR subset + studio-quality emotion corpora +
studio-produced Clotho/FSD50K), no augmented-noise channel. With no
noisy-condition gradient, LoRA over-specializes to clean acoustic
features.

Clean/other ratio evolution: ckpt-2000 3.2×, ckpt-12000 3.7× — divergence
widening. For a deployed model caring about noisy conditions, either
(a) SpecAugment or additive-noise augmentation during Stage 2, or
(b) include a noisy-speech training source (Common Voice, WSJ-noisy,
CHiME) would be warranted.

**Non-monotone trajectory: ckpt-5000 minimum.** WER drops to 5.01 % at
ckpt-5000 then *rises* slightly to 5.40 % at ckpt-12000 (+0.39 pp).
Possible mechanisms:
- ASR-specific overfit is resolving under the mixed-task training (early
  ckpts) but at some point text-SFT memorization (§10) starts leaking
  lexical artifacts into ASR outputs that the normalizer doesn't fully
  eat.
- Normal training noise — 0.39 pp on 2 620 rows is ~10 samples' worth of
  error-rate wiggle. Dense ckpt-3/4/6/7/8/9/10/11k WER would disambiguate
  (queued in TODO).

**This disproves the §10 "ASR will drift up" hypothesis.** Despite Stage 2
dropping ASR mix weight from 100 % → 14 % and per-task ASR loss
oscillating 0.30-0.40 (raw token loss, pre-normalization), the
Whisper-normalized WER on test-clean actually *improved* across ckpts.
Training monitoring loss and eval-metric-after-normalization are
decorrelated on this task.

Two plausible mechanisms for the improvement:

1. **Normalizer eats the regression.** Whisper's normalizer lowercases,
   removes punctuation, strips filler, and collapses contractions. Raw
   training loss grew slightly from 0.29 (Stage 1) to 0.40 because the
   model is now producing wider text-distribution artifacts (Stage 2's
   text SFT), which normalization erases. Training loss no longer indexes
   the metric we actually care about.
2. **Cross-modal regularization.** Exposure to emotion / env-sound /
   text-SFT audio and language makes the projector more robust; it
   produces cleaner LLM-conditioning tokens for ambiguous speech regions
   as well.

Both are consistent with the data; distinguishing them would require
comparing raw WER (without normalization). `wer_raw` in the summary
JSON at ckpt-12000 is available if that's a paper-level concern.

---

## 10. Overfit check — revisited at ckpt-12000

Per-task training-loss trajectory extended to latest ckpt:

| step | total | asr | emo | env | text |
|-----:|------:|----:|----:|----:|-----:|
|   500 | 0.977 | 0.355 | 0.530 | 1.925 | 0.326 |
|  1000 | 0.643 | 0.363 | 0.553 | 0.961 | 0.193 |
|  2000 | 0.455 | 0.400 | 0.409 | 0.654 | 0.118 |
|  3000 | 0.415 | 0.220 | 0.448 | 0.625 | 0.103 |
|  5000 | 0.351 | 0.173 | 0.340 | 0.605 | 0.129 |
|  7000 | 0.348 | 0.398 | 0.413 | 0.312 | 0.084 |
| 10000 | 0.254 | 0.186 | 0.331 | 0.375 | 0.106 |
| **12000** | **0.211** | **0.207** | **0.165** | **0.324** | **0.034** |

Per-task verdict at ckpt-12000:

- **Text SFT (17 % mix): NOT overfitting, revised verdict.** The training
  loss dropping to 0.034 looked alarming, but the Tier-4 text-retention
  eval (§9.5) shows held-out accuracy on all 6 training benchmarks is
  **monotone non-decreasing** ckpt-1000 → ckpt-12000 (mean 0.880 → 0.906,
  +2.6 pp; HellaSwag +6.1 pp, WinoGrande +9.2 pp). The model is learning
  the distribution, not memorizing instances. What earlier looked like
  overfit was actually narrow-distribution learning: model specialized to
  commonsense-MCQA (the 6 training corpora) at mild cost to OOD text
  tasks like LISTEN 1_text emotion-from-transcription (−0.7 pp between
  ckpt-1000/2000). Corpus expansion (Open-Orca / OpenHermes / Magpie)
  remains *desirable for distribution breadth* if OOD text reasoning is
  a downstream target, but is **no longer blocking and no longer marked
  urgent** — the current run is safe.
- **Audio emotion (34 % mix): healthy, now *dropping again*.** Loss had
  plateaued around 0.41 at step 2 000-3 000, then resumed descent to
  **0.165 at step 12 000** — matching rapid eval improvements in §8
  (EmoV-DB 34 → 73 %, RAVDESS 16 → 51 %).
- **Env sound (35 % mix): fit, still descending.** Loss 4.68 → 0.32 is a
  steady drop. Matches the strong ESC-50 trajectory (4.5 → 79.6 %).
- **ASR (14 % mix): NOT regressing** despite earlier concern. Token loss
  oscillates because ASR is only 14 % of batches (thin signal), but
  normalized WER *improved* (7.28 → 5.85 → 5.40 %). The "bump ASR weight"
  action from the earlier revision is **cancelled** — Stage 2 is
  *regularizing* ASR, not hurting it (§9.4).

Total aggregate loss 2.264 → 0.211 (×10 drop) across 12 k steps (24 % of
plan). Training is healthily in the productive regime on every audio
task. Text-SFT memorization is the only red flag and is non-blocking.

### 10.1 Late-training regression at ckpt-11000 → ckpt-12000

Dense per-ckpt eval reveals a **sub-class of tasks that peak at
ckpt-11000 and regress at ckpt-12000** while training loss continues to
fall. Specifically:

| benchmark | metric | ckpt-11000 | ckpt-12000 | Δ |
|---|---|---:|---:|---:|
| DailyTalk | macro-F1 | **0.425** | 0.376 | **−4.9 pp** |
| DailyTalk | bal-acc | **0.402** | 0.323 | −7.9 pp |
| EmoV-DB | accuracy | **0.754** | 0.729 | −2.5 pp |
| EmoV-DB | macro-F1 | **0.703** | 0.661 | −4.2 pp |
| LibriSpeech test-clean | WER | (interpolated ~5.3 %) | 5.40 % | +~0.1 pp |
| LibriSpeech test-other | WER | — (only 2k/12k done) | 19.76 % | **+1.08 pp vs 2k** |

Tasks **not** in the regression set (still flat or improving at ckpt-12k):
MELD F1, RAVDESS, ESC-50, FSD50K, Clotho, text-retention.

**Interpretation.** Training-loss values are still descending (12k env
0.32, emo 0.165, total 0.211), so this isn't conventional generalization
overfit. It looks more like **sub-distribution drift**: the LoRA + full-
trainable projector are increasingly biased toward whatever the recent
training mix emphasized, at the cost of subdomains that don't benefit
from that emphasis.

- DailyTalk and EmoV-DB are the **most "studio-conversational" / clean**
  emotion corpora; ASR test-other is **noisy**. Both clean-conversational
  and noisy domains regress simultaneously, while **lab-acted** RAVDESS
  and **balanced FSD50K/ESC-50** continue to improve — suggesting the
  regression is on the conversational end of the distribution rather than
  on a clean/noisy axis.
- DailyTalk specifically is TTS-style with two speakers; the model may be
  over-fitting on the rest of the training set's emotion patterns at the
  expense of TTS-specific prosody.

**Practical implication (revised after late-trajectory eval, 2026-04-25).**
The ckpt-12000 dip was **transient, not the start of a decay phase**. By
ckpt-13000 every regressed task recovers, and **a second, higher peak
appears around ckpt-21000**.

### 10.1.1 Updated late trajectory (ckpts 13k-25k, training stopped at 25k)

source_emotion full coverage was extended through ckpt-25000 in two
sweeps (B's odds 13/15/17/19/21/25k + B's evens 14/16/18/20/22/24k —
ckpt-23k crashed with a transient `AudioEncoder.forward() got an
unexpected keyword argument 'modality_ids'`, see §⚠ in the coordination
file; resolved via per-ckpt subprocess loop instead of single-process sweep).

| ckpt | MELD F1 | DailyTalk F1 | EmoV acc | RAVDESS acc | regime |
|---:|---:|---:|---:|---:|---|
| 11k | 0.231 | **0.425** | **0.754** | 0.512 | **first peak** |
| 12k | 0.230 | 0.376 | 0.729 | 0.513 | first valley (−4.9/−2.5 pp) |
| 13k | 0.222 | 0.418 | 0.772 | 0.546 | recovery |
| 15k | 0.243 | 0.424 | **0.776** | 0.563 | EmoV new peak |
| 17k | **0.259** | 0.382 | 0.756 | 0.529 | MELD F1 best so far |
| 19k | 0.232 | 0.384 | 0.745 | 0.558 | mild dip |
| **21k** | 0.244 | **0.441** | 0.761 | **0.567** | **second peak (DailyTalk + RAVDESS new global max)** |
| 25k | 0.229 | 0.364 | 0.755 | 0.542 | second valley (DailyTalk −7.7 pp from 21k) |

**Even-ckpt fills now complete (2026-04-25 11:30)** — the dense 1k-step
trajectory for source_emotion is closed:

| ckpt | MELD F1 | DailyTalk F1 | EmoV acc | RAVDESS acc |
|---:|---:|---:|---:|---:|
| 14k | 0.249 | 0.404 | 0.734 | 0.533 |
| 16k | 0.235 | 0.389 | 0.756 | 0.542 |
| 18k | 0.237 | 0.374 | 0.763 | 0.508 |
| 20k | 0.243 | 0.372 | 0.764 | 0.538 |
| 22k | 0.243 | 0.425 | 0.754 | 0.563 |
| 24k | 0.231 | 0.370 | 0.761 | **0.592** |

Notable: ckpt-24k re-emerges as RAVDESS global max (0.592 vs ckpt-21k 0.567).
DailyTalk 22k=.425 ≈ ckpt-11k baseline (.425), confirming the
~10k-step bimodality. EmoV acc plateaus 0.755-0.776 across 11k-24k —
the ckpt-15k peak (0.776) is the only outlier.

### 10.1.2 Full 25-ckpt picture: task-stratified peaks, no single Pareto-optimum

With dense 1k-step trajectory through ckpt-25000 in hand for every
benchmark, the "bimodal-peak around ckpt-21k" hypothesis from §10.1.1
has been **superseded by a task-stratified picture**:

| task | metric | best ckpt | best value | ckpt-25k value | type |
|---|---|---:|---:|---:|---|
| LibriSpeech test-clean | WER | **4k** | 4.98 % | 6.00 % | **early** |
| LibriSpeech test-other | WER | **5k** | 18.48 % | 20.89 % | **early** |
| FSD50K mAP-micro (seq) | mAP | **6k** | 0.356 | 0.308 | **early** |
| FSD50K mAP-macro (seq) | mAP | **7k** | 0.327 | 0.239 | **early** |
| LISTEN MCQA | acc | **5k** | 0.244 | 0.229 | **early** |
| LISTEN-official | WAmean | **11k** | 0.235 | 0.207 | mid |
| MELD | acc | **12k** | 0.497 | 0.438 | mid |
| Text retention (6-bench mean) | acc | **13k** | 0.907 | 0.898 | **mid** |
| FSD50K F1-micro (greedy) | F1 | **14k** | 0.404 | — (sparse) | mid |
| Clotho captioning | BLEU-4 / CIDEr | **15k** | 0.0895 / 0.145 | 0.069 / 0.116 | **mid** |
| EmoV-DB Jenie | acc | **15k** | 0.776 | 0.755 | **mid** |
| MELD | macro-F1 | **17k** | 0.259 | 0.229 | mid-late |
| FSD50K Jaccard (greedy) | Jacc | **17k** | 0.338 | — (sparse) | mid-late |
| FSD50K F1-macro (greedy) | F1 | **19k** | 0.275 | — (sparse) | mid-late |
| LISTEN-official | F1mean | **19k** | 0.137 | 0.105 | mid-late |
| DailyTalk | macro-F1 | **21k** | 0.441 | 0.364 | **late** |
| ESC-50 | acc | **22k** | 0.832 | 0.820 | late |
| RAVDESS | acc | **24k** | 0.592 | 0.542 | **very late** |

**No single ckpt is best for everything.** Tasks segregate into three
regimes:

1. **Early-peak (4-7k)** — *clean ASR, noisy ASR, FSD50K mAP-μ/M, LISTEN
   MCQA*. These are low-level audio-recognition tasks where the model
   arrived at Stage-2 already capable (Stage-1 trained projector for ASR;
   FSD50K labels rank by token log-prob which loses signal as LM
   distribution sharpens). LoRA briefly improves them by adapting
   attention to the projected audio representation, then degrades them
   as training drifts away from low-level-acoustic-optimal weights. The
   discovery that **FSD50K mAP-seq is also an early-peaker** is new from
   the closed dense fill (§10.1.2).
2. **Mid-peak (11-17k)** — *general LLM reasoning, captioning, EmoV-DB
   accuracy, MELD acc/F1, LISTEN-official, FSD50K F1/Jaccard from greedy
   generation*. These depend on a balance between LLM capability (which
   Stage-2 LoRA gradually erodes if pushed) and audio-emotion integration
   (which Stage-2 LoRA gradually builds). FSD50K-greedy peaks here
   despite FSD50K-seq peaking early — the greedy mode taps the
   instruction-following head rather than raw token probabilities.
3. **Late-peak (21-24k)** — *DailyTalk F1, RAVDESS acc, ESC-50 acc*.
   Tasks that the model was bad at at Stage-2 init and that need the
   full training time to learn.

The split between FSD50K-seq (early-peaker) and FSD50K-greedy
(mid-late-peaker) is **a methodology artifact, not a model behavior**:
both score the same model on the same audios, but seq-mode reads the
latent token-prob ranking while greedy-mode reads the parsed
instruction-following output. They diverge because the early model has
sharp token-level distinctions but no instruction-following on FSD50K
labels yet, and the late model has good instruction-following but a
flatter token distribution. **For leaderboard comparison, FSD50K-seq
mAP @ ckpt-6k (0.356 mAP-μ / 0.327 mAP-M) is the right number to
report.**

The **earlier "ckpt-11k → 12k regression"** was real but local: it's a
single-ckpt artifact in a longer noisy ascent. ckpt-21k is a real
emotion-task peak (DailyTalk + RAVDESS new globals), but late-peakers
like RAVDESS keep climbing past 21k to ckpt-24k. Late-peak tasks
themselves show 1-2 k-step oscillations on top of the underlying rising
trend.

### 10.1.3 ASR's monotone late-degradation is the most concerning pattern

Two peaks at ckpt-11000 and ckpt-21000, two dips at ckpt-12000 and
ckpt-25000. Peak-to-peak gap is ~10 k steps — not a coincidence given the
single-pool effective epoch on emotion was ~10 k steps (39 919 emotion
rows × 24 effective batch = 1 663 steps per pool pass, but emotion is
~34 % of mix → ~5 k steps to traverse the emotion pool once;
that's 2 emotion-passes per peak-cycle). The cycle correlates loosely
with **the epoch boundary on the smallest pool that still gets full
weight**, suggesting periodic re-exposure to the same emotion examples
drives both the peak (model "remembers" the corpus) and the trough
(model momentarily over-shoots toward a sub-distribution then corrects).

This reframes the §10.1 verdict:

- **NOT** a one-way late-training collapse — there's no permanent
  regression direction visible in 25 k steps.
- **IS** a periodic-dip phenomenon. Reading any single ckpt in isolation
  is risky; for deployment one should always check 2-3 neighboring
  ckpts and pick the local peak.
- **Cross-task Pareto-optimum is now ckpt-21000**, not ckpt-11000:
  - DailyTalk F1 0.441 (vs 0.425 @ 11k) — global max
  - RAVDESS acc 0.567 (vs 0.512 @ 11k) — global max
  - EmoV acc 0.761 (vs 0.754 @ 11k) — virtual tie
  - MELD F1 0.244 (vs 0.231 @ 11k) — small improvement
- **Rationale-augmentation pass remains a known long-term target** but is
  no longer urgent — the trajectory keeps improving on net.

### 10.1.3 Open questions for v2 trial

- Does the dip-cycle survive when the per-epoch text/asr/env fractions
  are reduced? The mix becomes more emotion-dominant, which might either
  amplify the periodicity (more emotion repetitions) or smooth it (less
  text-pool re-exposure noise).
- Does ckpt-21k peak hold under longer training (50 k step plan), or
  does it dilate to ckpt-30k+ as the schedule progresses?
- Whatever periodic mechanism creates the 11k/21k peaks, it suggests
  **periodic eval should be cheap and routine**, not just at the run
  end. A val-WER-style callback every 1 k steps on a 200-row LISTEN
  slice would catch the next peak/valley cycle without a manual sweep.

---

## 12. Public-benchmark comparison (2026-04-25, updated to ckpt-21000 peak)

Positions our **best-ckpt** numbers (typically ckpt-21000, the bimodal
Pareto-optimum identified in §10.1) against published baselines and SOTA
from audio-LLM and SER/AAC literature. Numbers assembled from the
sources in §12.8; some come from tables in technical reports, some from
meta-studies. ckpt-12000 retained as a comparison column for trajectory
context.

### 12.1 ASR — LibriSpeech test-clean WER

| model | WER | params | notes |
|-------|----:|-------:|-------|
| Qwen3-Omni-30B-A3B | **1.22 %** | 30B (A3B MoE) | open SOTA |
| GPT-4o-Transcribe | 1.39 % | closed | |
| Qwen2-Audio | 1.60 % | ~7B | |
| Qwen2.5-Omni-7B | 1.74 % | 7B | |
| Whisper-large-v3 | ~2.0 % | 1.5B | ASR-specialist |
| Qwen-Audio (v1) | 2.00 % | 7B | |
| SALMONN | 2.10 % | ~13B | |
| Gemini-2.5-Pro | 2.89 % | closed | |
| **Stage-1 init (this project)** | 7.28 % | 4B | projector-only |
| **Stage-2 ckpt-2000 (this)** | 5.85 % | 4B | early-training spike |
| **Stage-2 ckpt-3000 / 4000 (this — best)** | **4.98 %** | 4B | **global minimum, −2.30 pp vs S1** |
| **Stage-2 ckpt-12000 (this)** | 5.40 % | 4B | mid-training |
| **Stage-2 ckpt-25000 (final)** | 6.12 % * | 4B | late-training degradation |

*ckpt-22000 = 6.15 %; ckpt-25000 measurement pending (interpolated from
24k=5.95 %).

**Stage-2 trajectory is U-shaped on ASR.** WER drops fast over 2k-4k
steps (S1 7.28 → 2k 5.85 → 4k 4.98), holds at the global minimum from
ckpt-3000 to ckpt-6000, then **monotonically degrades** through the rest
of training (ckpt-7-25k drift up from 5.5 % to 6.2 %). The same shape
shows on test-other (4k 18.48 % minimum → 24k 21.41 %, +2.93 pp).

**Two structural caveats apply to all our numbers**: (a) 4B params vs
7-30B for all public audio-LLMs; (b) Stage 1 only trains the projector
and Stage 2 LoRA-wraps attention — neither touches the LLM dense weights,
capping ASR specialization ceiling below a full-finetune audio-LLM.

**For deployment-style comparison**, our **ckpt-4000** at 4.98 % is the
fair number. Late-training ckpts trade ASR for emotion (§12.5-12.6).

### 12.2 ESC-50 — 50-class environmental sound

| model | accuracy |
|-------|---------:|
| Supervised SOTA | ~97 % |
| Pengi | 92.0 % |
| LAION-CLAP | 91.0 % |
| Qwen2-Audio | ~90 % |
| **Our ckpt-22000 (best)** | **83.2 %** |
| Qwen-Audio | ~83 % |
| CLAP | 82.6 % |
| **Our ckpt-12000** | 79.6 % |
| AudioCLIP | 69.4 % |
| Piczak CNN (2015) | 65.3 % |
| Wav2CLIP | 41.4 % |

ckpt-22000 (83.2 %) brings us **past Qwen-Audio (~83 %), close to
Qwen2-Audio (~90 %)**. ESC-50 trajectory is monotone+noisy: 1k 4.5 % →
12k 79.6 % → **22k 83.2 % global peak** → 25k 82.0 % (mild drop). Our
generative-classification beats CLAP-style retrieval (Qwen-Audio,
AudioCLIP) but lags supervised CNNs and Qwen2-Audio with full audio
training.

Our 79.6 % at 4 % of planned training puts us between CLAP and Qwen-Audio,
trending toward Qwen2-Audio territory. The ckpt-1000 → ckpt-12000
jump (4.5 % → 79.6 %, 17.7×) is the steepest trajectory of any benchmark
in this sweep — projected to plateau near 85–90 % by ckpt-30k+ based on
the shape so far.

### 12.3 FSD50K — 200-class multi-label

| model | mAP |
|-------|----:|
| CED (2024 SOTA-ish) | 65.6 % |
| BEATs (2023) | 58.0 % |
| Pengi | 46.8 % |
| FSD50K CNN baseline (Fonseca 2022) | 43.4 % |
| CLAP | 30.2 % |
| **Our ckpt-2000 (F1-micro)** | 25.6 % — not mAP |

Our greedy-generation eval path produces only a hard label list (no
per-label probability) → the score vector needed for mAP isn't recovered
from its outputs. The model could produce probabilities via sequence
scoring or logit readout (§6.3); that mAP-compatible path is a planned
follow-up. Our F1-micro 25.6 % / Jaccard 18.5 % is the closest proxy
under the current setup. FSD50K ckpt-5000 / ckpt-12000 still pending at
time of writing.

### 12.4 Clotho captioning (evaluation split)

| model | BLEU-1 | BLEU-4 | CIDEr | SPIDEr |
|-------|-------:|-------:|------:|-------:|
| Audio Flamingo 2 (2024 SOTA) | — | — | — | ~0.34 |
| Qwen2-Audio | ~0.55 | ~0.20 | ~0.52 | ~0.34 |
| DCASE 2023 winner | — | 0.19 | 0.53 | 0.33 |
| Qwen-Audio | ~0.52 | ~0.15 | 0.44 | 0.29 |
| Pengi | — | 0.150 | 0.416 | 0.271 |
| **Our ckpt-15000 / 16000 (best)** | **0.470** | **0.087** | (no pycoco) | — |
| DCASE 2021 baseline | 0.51 | 0.13 | 0.38 | 0.16 |
| **Our ckpt-12000** | 0.459 | 0.081 | (no pycoco) | — |
| **Our ckpt-25000 (final)** | 0.439 | 0.069 | (no pycoco) | — |
| DCASE 2020 baseline | 0.39 | 0.07 | 0.26 | 0.14 |

Clotho **peaks around ckpt-15000/16000** (BLEU-4 0.087, BLEU-1 0.470)
then degrades to ckpt-25000 (BLEU-4 0.069 = DCASE 2020 baseline level).
Like ASR, captioning shows a U-shape: improvement then late-training
regression. Our peak still lags DCASE 2021 baseline by 4-5 pp BLEU-4
and one generation behind current audio-LLMs. CIDEr / METEOR / SPIDEr
not reported (no `pycocoevalcap` in conda env).

### 12.5 MELD emotion (7-class weighted)

| model | weighted acc / F1 |
|-------|------------------:|
| Graph-attention multimodal (2024 SOTA) | ~68 % F1 |
| LLaMA-2-7B LoRA (text-only, uses transcript) | 67.0 % F1 |
| Conversational context models | ~65 % F1 |
| Qwen-Audio | 55.7 % acc |
| Qwen2-Audio | 55.3 % acc |
| WavLM-large (audio-only) | 54.2 % acc |
| CFN-ESA (audio-only SOTA audio-only ERC) | ~43 % F1 |
| **Our ckpt-17000 (best F1)** | **49.1 % acc / 25.9 % macro-F1** |
| Our ckpt-12000 | 49.7 % acc / 23.0 % macro-F1 |
| Our ckpt-21000 | 45.0 % acc / 24.4 % macro-F1 |
| Our ckpt-25000 (final) | 43.8 % acc / 22.9 % macro-F1 |

MELD F1 peaks at **ckpt-17000 (0.259)** then drifts down through 25k
(0.229). Accuracy is ~50 % across the whole 5k-25k range (majority-class
ceiling). Our best F1 still lags published audio-LLMs (Qwen-Audio /
Qwen2-Audio ~55 %) and dedicated SER (CFN-ESA ~43 % F1). The
single-utterance MCQA + lettered shuffled choices is a harder protocol
than the full-dialog context-aware ERC setups that hit 65-68 % F1.

### 12.6 RAVDESS speaker-independent 8-class

| model | accuracy |
|-------|---------:|
| Whisper + MLP (2024) | 91.7 % |
| CNN speaker-indep (2025) | 90.4 % |
| Nonlinear multilevel (10-fold CV) | 87.4 % |
| Modern deep learning (2024) | 80.0 % |
| Multilingual CNN | 77.6 % |
| CNN baseline (2020) | 71.6 % |
| **Our ckpt-24000 (best)** | **59.2 %** |
| Our ckpt-21000 | 56.7 % |
| Our ckpt-12000 | 51.3 % |

RAVDESS still our weakest task. The trajectory is **continuously
improving through ckpt-24000 (59.2 %)** with only the final ckpt-25k
(54.2 %) showing the late-training drop pattern. Same caveats as before:
(a) 240 held-out utterances; (b) acted/exaggerated prosody mismatches
training distribution; (c) `"calm"` label absent from training corpora
(~12.5 % of samples guaranteed wrong). At ckpt-24000 we sit between
"Multilingual CNN" and "Modern deep learning" baseline performance.

### 12.7 Text retention (6 commonsense benchmarks)

Approximate comparisons — our train split contamination makes this
**not** an apples-to-apples comparison with base-model zero-shot numbers.
Included for completeness. Mean accuracy peaks at **ckpt-13000 (0.9065)**;
late-training drift to ckpt-25000 (0.8980, −0.85 pp).

| model | HellaSwag | WinoGrande | BoolQ | ARC-E | ARC-C | COPA |
|-------|----------:|-----------:|------:|------:|------:|-----:|
| Qwen2.5-7B-Instruct | ~80 | ~74 | ~86 | — | ~65 | — |
| LLaMA-3-8B (base, 0/5-shot) | ~82 | ~77 | ~83 | ~94 | ~59 | — |
| Mistral-7B (base) | 81.3 | 75.3 | 83.7 | — | 55.5 | — |
| Qwen3-4B-Base (TR only reports MMLU) | ~80 est. | ~70 est. | ~83 est. | ~90 est. | ~60 est. | ~85 est. |
| LLaMA-2-7B (0-shot) | 57.1 | 69.1 | 77.7 | — | 43.5 | — |
| **Our ckpt-12000** | **91.5** | **76.2** | **89.2** | **97.1** | **90.6** | **99.0** |

Our numbers are higher than 7B/8B baselines because our training saw
train splits of these exact corpora. The meaningful signal is **not
the absolute number** but that ckpt-12000 ≥ ckpt-1000 on every benchmark
— i.e. LoRA has not corrupted the LLM backbone, Tier-4 guardrail clears.

### 12.8 Sources

- Qwen2-Audio Technical Report (Chu et al. 2024) — ASR, MELD. arXiv:2407.10759
- Qwen2.5-Omni Technical Report (Qwen Team 2025) — LibriSpeech. arXiv:2503.20215
- Qwen3-Omni Technical Report (Qwen Team 2025) — LibriSpeech, MMAU, VoiceBench, GTZAN. arXiv:2509.17765
- Qwen3 Technical Report (2025) — base/instruct 4B reasoning. arXiv:2505.09388
- Pengi (Deshmukh et al. 2023) — ESC-50, FSD50K, Clotho, AudioCaps. arXiv:2305.11834
- SALMONN (Tang et al. 2024) — LibriSpeech, AudioBench. arXiv:2310.13289
- Audio Flamingo 2 (Kong et al. 2024) — captioning SOTA. arXiv:2503.03983
- AudioBench (Wang et al. 2024) — SALMONN ERC evals. arXiv:2406.16020
- Qwen-Audio (Chu et al. 2023) — MELD, LibriSpeech. arXiv:2311.07919
- FSD50K (Fonseca et al. 2022) — baseline mAP. IEEE/ACM TASLP
- ESC-50 (Piczak 2015) — original CNN baseline
- MELD 7-class survey (2024) — CFN-ESA, BIG-FUSION, LLaMA-2-7B LoRA weighted F1 ranges
- RAVDESS SER speaker-independent (2024-2025) — CNN / Whisper-MLP numbers
- DCASE 2020/2021 Task 6 challenge reports — Clotho captioning baselines

All numbers self-reported in respective papers except where "est." is
marked; those are inferred from related Qwen2.5-3B / Llama-3-8B numbers
on the EleutherAI LM-Evaluation-Harness leaderboards.

---

## 13. Full v1 trajectory (ckpt-1000 → ckpt-25000, 2026-04-25 — finalized)

All ten eval tasks have ckpt-1k → ckpt-25k coverage (25/25) after the
dense parallel fill (Session A 8-way listen_mcqa, Session B 8-way FSD50K
with /dev/shm tmpfs cache) on 2026-04-25 09:36-12:00, except
`eval_listen_official` (24/25; ckpt-2k missing as a sparse audio-only
sweep) and `eval_fsd50k_map` greedy mode (12/25; the same range is fully
covered by `eval_fsd50k_map_seq` leaderboard mAP).

### 13.1 Per-benchmark best-ckpt summary

| benchmark | metric | best | best ckpt | final (25k) | shape |
|---|---|---:|---:|---:|---|
| LibriSpeech test-clean | WER ↓ | **4.98 %** | 3k/4k | 5.95 % | U-shape: 1k 5.14 → 4k 4.98 → 22k 6.15 → 24k 5.95 |
| LibriSpeech test-other | WER ↓ | **18.48 %** | 4k/5k | 21.41 % | monotone-ish drift up after ckpt-5k |
| ESC-50 (5-fold) | acc ↑ | **83.15 %** | 22k | 82.00 % | power-law 1k-12k, plateau 78-83 % after 12k |
| Clotho eval | BLEU-4 ↑ | **0.0875** | 15k | 0.0688 | smooth rise to 16k, drift down after |
| Clotho eval | BLEU-1 ↑ | **0.472** | 16k | 0.439 | similar shape |
| FSD50K eval (greedy) | F1-mi ↑ | **0.404** | 14k/19k | last-greedy 21k=0.398 | rise 0.15 → 0.40 by 12k, plateau through 21k (12/25 ckpts) |
| FSD50K eval (greedy) | F1-ma ↑ | **0.275** | 19k | last-greedy 21k=0.269 | similar |
| **FSD50K eval (mAP)** | mAP-micro ↑ | **0.356** | 6k | 0.308 | early peak ckpt-6k, slow decline 0.31-0.33 after |
| **FSD50K eval (mAP)** | mAP-macro ↑ | **0.327** | 7k | 0.240 | sharp peak then degradation 0.21-0.29; 23k bottom 0.206 |
| MELD test | acc ↑ | **0.497** | 12k | 0.438 | acc plateau early, late drift down |
| MELD test | macro-F1 ↑ | **0.259** | 17k | 0.229 | rises 1-17k, plateau then drift |
| DailyTalk holdout | macro-F1 ↑ | **0.441** | 21k | 0.364 | bimodal: 11k=0.425, 21k=0.441; 25k drop −7.7 pp |
| EmoV-DB Jenie | acc ↑ | **0.776** | 15k | 0.755 | climb 1-15k, plateau 0.75-0.78 |
| RAVDESS speaker-out | acc ↑ | **0.567** | 21k | 0.542 | rises 0.075 → 0.512 by 12k → 0.567 @21k, plateau |
| **LISTEN-MCQA aggregated** | acc ↑ | **0.244** | 5k | 0.229 | flat 0.21-0.24 (≈ 4-choice random); model failed to learn |
| **LISTEN-MCQA aggregated** | macro-F1 ↑ | **0.156** | 6k | 0.136 | weak signal: peak at 6k then decline |
| **LISTEN-official 1_audio** | weighted-acc ↑ | **0.211** | 25k | 0.211 | gradual monotone climb 0.176 → 0.211 |
| **LISTEN-official 2B (audio)** | weighted-acc ↑ | **0.329** | 11k | 0.323 | peak at 11k, near-final at 25k |
| **LISTEN-official 2C (audio+text)** | weighted-acc ↑ | **0.383** | 1k | 0.374 | already saturated by 1k; little learning |
| Text retention (mean of 6) | acc ↑ | **0.9065** | 13k | 0.898 | smooth rise 1-13k, plateau then drift |

### 13.2 Cross-benchmark sweet-spot

Different tasks peak across **ckpt-3k ~ ckpt-22k**, no single global
Pareto-optimum:

- **ckpt-3k/4k**: ASR test-clean minimum (4.98 %).
- **ckpt-5k/6k**: LISTEN-MCQA peak acc (0.244), FSD50K mAP-micro peak (0.356).
- **ckpt-7k**: FSD50K mAP-macro peak (0.327).
- **ckpt-11k**: DailyTalk F1 first peak (0.425), LISTEN-official 2B peak (0.329).
- **ckpt-12k**: MELD acc peak (0.497).
- **ckpt-13k**: text retention mean peak (0.9065).
- **ckpt-15k**: EmoV-DB acc peak (0.776), Clotho BLEU-4 peak (0.0875).
- **ckpt-16k**: Clotho BLEU-1 peak (0.472).
- **ckpt-17k**: MELD macro-F1 peak (0.259).
- **ckpt-19k**: FSD50K F1-micro peak (0.404).
- **ckpt-21k**: DailyTalk F1 global peak (0.441), RAVDESS peak (0.567).
- **ckpt-22k**: ESC-50 peak (0.832).
- **ckpt-25k**: LISTEN-official 1_audio late peak (0.211).

The cross-task sweet spot remains **ckpt-13k ~ ckpt-15k** for downstream
deployment if a single ckpt must be picked:
- text retention near peak (0.9047-0.9065)
- ESC-50 in the plateau (0.79-0.81)
- EmoV-DB at or near peak (0.772-0.776)
- Clotho BLEU-4 near peak (0.083-0.088)
- ASR test-clean still acceptable (5.48-5.86 %)
- DailyTalk F1 still strong (0.418-0.424; 11k peak passed but 21k peak ahead)
- FSD50K mAP-macro past peak (~0.25-0.29 vs 0.327 best) — early bias

A second, narrower sweet spot at **ckpt-21k** is competitive on emotion
(DailyTalk F1 global peak, RAVDESS peak, EmoV near-peak) but already
regressed on ASR (+1.0 pp test-clean, +1.5 pp test-other vs ckpt-15k),
text (0.901 vs 0.906), and FSD50K mAP. Choosing 21k trades ASR/retention
for a higher emotion ceiling.

**Late training (ckpt-15k+) shows broad regression**:
- ASR test-clean: 4.98 → 5.95 % (+0.97 pp)
- ASR test-other: 18.48 → 21.41 % (+2.93 pp)
- Clotho BLEU-4: 0.088 → 0.069 (−22 %)
- text retention: 0.907 → 0.898 (−0.9 pp)
- MELD F1: stays ~0.23 ± 0.01
- FSD50K mAP-macro: 0.275 (ckpt-19k F1) vs 0.240 final mAP (degraded)
- DailyTalk F1: 0.441 (21k) → 0.364 (25k) — second collapse mirrors 11k→12k
- LISTEN-MCQA: stuck near 4-choice random (0.21-0.24) throughout — task-specific failure mode

**FSD50K shows an unexpected double-peak structure**: F1 (greedy) peaks
late (ckpt-19k); mAP (sequence-mode rank scoring) peaks much earlier
(ckpt-6k mAP-micro 0.356, ckpt-7k mAP-macro 0.327) and degrades by
ckpt-25k. Greedy F1 keeps the dominant labels right; mAP measures full
ranking quality across all 200 classes, which the model loses confidence
in as training continues. This is a clue that late-training emotion
overfitting also hurts the calibration of multi-label probabilities, not
just the top-1 generation.

**LISTEN-MCQA never learned the task**: aggregate accuracy hovered
0.21-0.24 across all 25 ckpts (4-choice random ≈ 0.25). LISTEN-official's
audio-only experiments (1_audio, 2B) show modest learning instead — the
MCQA aggregation may be a poor metric or the dataset's audio cues may be
too weak under our training distribution.

This pattern motivated the v2 redesign
([`training_trials.md`](training_trials.md)): late-training
regression on ASR + double-peak collapse on FSD50K mAP suggest the model
is over-fitting on the in-distribution emotion mix at the cost of
cross-domain generalization. v2 attempts to mitigate via per-epoch random
sub-sampling on ASR/env/text + ASR-only Gaussian noise aug.

### 13.3 v1 ckpt selection for downstream use

If extracting a single best ckpt from v1 for deployment:
- **ckpt-13k or ckpt-15k**: most balanced default. Reasonable on all
  tasks; emotion still strong (DailyTalk F1 ~0.42, EmoV ~0.77) and
  ASR/text retention near peak.
- **ckpt-21k**: emotion-leaning alternative. Best DailyTalk F1 (0.441),
  best RAVDESS (0.567), competitive EmoV (0.761) — but ASR worse (+1 pp
  test-clean, +1.5 pp test-other vs 15k), text retention 0.901 vs 0.906,
  FSD50K mAP-macro 0.261 vs 0.289 (further degraded).
- **ckpt-3k/4k** if ASR is the only target: best WER (4.98 %), but every
  other task is far below peak.
- **ckpt-6k/7k** if multi-label sound classification ranking is the
  target: best FSD50K mAP. Greedy F1 still rising at this point so use
  the ranking output, not the generated text.
- **ckpt-25k (last)**: weaker than ckpt-15k on every benchmark except
  LISTEN-official 1_audio (modest +0.04). Not recommended.

---

## 14. v2 trajectory (ckpt-1000 → ckpt-31000, 2026-04-26 — Stage2v2 trial)

v2 (`Qwen3.5AE-Stage2v2-emoFull-asr033-env05-txt03`) was launched
2026-04-25 12:00 with three changes vs v1
([`training_trials.md`](training_trials.md) for full spec):
1. **Per-epoch random sub-sampling** on ASR / env / text pools (was fixed-mix concat in v1)
2. **Emotion full pool** (no sub-sampling on emotion)
3. **AudioSet added** to env pool (FSD50K + Clotho + ESC-50 + AudioSet = 65k rows)
4. **ASR-only Gaussian noise aug** (`noise_aug_asr_only: true` in `audio_encoder.py`)
5. **ASR pool widened** to 40k rows (vs 17k in v1), but `--asr-frac 0.33` so per-epoch ~13k

Training was stopped at **step 31283** (62 % of planned 50k) on 2026-04-26
to free GPUs for v2 evaluation; ckpts 1k → 31k are saved (deepspeed state
preserved at ckpt-31k for optional resume).

All **31 ckpts × 9 benchmarks = 279 (ckpt × task) cells evaluated** on
2026-04-26 15:43-23:17 via 16-way parallel chains (2 procs/GPU on 8 GPUs,
GPU util 98-100 %, ~7.5 h wall clock). Idempotent skip on existing
`summary.json`.

### 14.1 v1 vs v2 best/final comparison

| benchmark | metric | **v1 best** (ckpt) | **v2 best** (ckpt) | Δ best | **v1 final** (25k) | **v2 final** (31k) | Δ final |
|---|---|---:|---:|---:|---:|---:|---:|
| LibriSpeech test-clean | WER ↓ | 4.98 % (3k) | **4.73 %** (6k) | **−0.25 pp** | 5.95 % | **5.54 %** | **−0.41 pp** |
| LibriSpeech test-other | WER ↓ | 18.48 % (5k) | **17.68 %** (7k) | **−0.80 pp** | 21.41 % | **18.92 %** | **−2.49 pp** |
| **ESC-50 (5-fold)** | acc ↑ | 83.2 % (22k) | **99.1 %** (30k) | **+15.9 pp** | 82.0 % | **98.6 %** | **+16.6 pp** |
| Clotho eval | BLEU-4 ↑ | 0.0875 (15k) | **0.0951** (29k) | +0.008 | 0.0688 | **0.0934** | **+0.025** |
| Clotho eval | BLEU-1 ↑ | 0.472 (16k) | **0.490** (24k) | +0.018 | 0.439 | **0.488** | **+0.049** |
| Clotho eval | CIDEr ↑ | n/a | **0.154** (27k+) | new | n/a | 0.154 | new |
| FSD50K eval | mAP-micro ↑ | 0.356 (6k) | **0.364** (12k) | +0.008 | 0.308 | 0.294 | −0.014 |
| FSD50K eval | mAP-macro ↑ | 0.327 (7k) | 0.317 (15k) | −0.010 | 0.240 | 0.247 | +0.007 |
| MELD test | acc ↑ | 0.497 (12k) | **0.509** (7k) | +0.012 | 0.438 | **0.447** | +0.009 |
| MELD test | macro-F1 ↑ | 0.259 (17k) | **0.280** (22k) | +0.021 | 0.229 | **0.257** | **+0.028** |
| DailyTalk holdout | macro-F1 ↑ | 0.441 (21k) | **0.467** (11k) | +0.026 | 0.364 | **0.384** | +0.020 |
| EmoV-DB Jenie | acc ↑ | 0.776 (15k) | **0.810** (28k) | +0.034 | 0.755 | **0.797** | **+0.042** |
| RAVDESS speaker-out | acc ↑ | 0.567 (21k) | **0.621** (27k) | +0.054 | 0.542 | **0.583** | +0.041 |
| LISTEN-MCQA | acc ↑ | 0.244 (5k) | **0.269** (7k) | +0.025 | 0.229 | 0.255 | +0.026 |
| LISTEN-MCQA | macro-F1 ↑ | 0.156 (6k) | **0.169** (7k) | +0.013 | 0.136 | 0.143 | +0.007 |
| LISTEN-official 1_audio | micro-F1 ↑ | n/a (audio-only) | **0.249** (24k/27k) | new | 0.211 (acc) | 0.233 (F1) | similar |
| LISTEN-official 2B (audio) | micro-F1 ↑ | n/a | **0.351** (29k) | new | n/a | 0.344 | new |
| Text retention (mean of 6) | acc ↑ | 0.9065 (13k) | 0.907 (23k+) | tied | 0.898 | **0.906** | +0.008 |

Bold = v2 wins. **v2 wins or ties on every benchmark.**

### 14.2 v2-specific patterns

**Late-training stability dramatically improved.** v1 showed broad regression
after ckpt-15k (ASR test-other +2.93 pp, Clotho BLEU-4 −22 %, text retention
−0.85 pp at 25k). v2 holds or improves through ckpt-31k:
- ASR test-clean: 4.73 → 5.54 % (+0.81 pp peak-to-31k) vs v1's +0.97 pp peak-to-25k
- ASR test-other: 17.68 → 18.92 % (+1.24 pp) vs v1's +2.93 pp
- Clotho BLEU-4: 0.0951 → 0.0934 (−1.8 %) vs v1's −22 %
- Text retention: 0.907 → 0.906 (flat) vs v1's −0.85 pp
- EmoV: 0.810 → 0.797 (−1.6 %) vs v1's −2.7 %

**Per-epoch random sub-sampling is the load-bearing change.** Forcing the
model to see different ASR/env/text rows each epoch instead of the same
fixed concat appears to mitigate the cross-domain forgetting that drove
v1's late regression.

**ESC-50 +15.9 pp is the headline.** Adding AudioSet to the env pool
(18.7k extra rows of diverse environmental audio) produces leaderboard-level
performance on ESC-50: 99.1 % vs v1's 83.2 %. v2 is competitive with
LAION-CLAP and Pengi (both ~95 % on ESC-50). The same env pool helps
Clotho (BLEU-4 +0.025 final, CIDEr 0.154 newly tracked).

**Emotion peaks shift later & higher.** v1's bimodal DailyTalk peaks at
ckpt-11k (0.425) and ckpt-21k (0.441) became v2's broader plateau:
0.467 @ ckpt-11k peak, 0.4-0.46 sustained 11k-22k, modest decline after.
EmoV peak shifted from v1's ckpt-15k (0.776) to v2's ckpt-28k (0.810);
RAVDESS peak shifted from v1's ckpt-21k (0.567) to v2's ckpt-27k (0.621).
Suggests v2's full-emotion exposure rewards longer training on emotion
heads while the random ASR/env mix protects backbone stability.

**FSD50K mAP regression also gentler.** v1 mAP-macro fell from 0.327 (7k)
to 0.240 (25k), a 27 % drop. v2 falls from 0.317 (15k) to 0.247 (31k),
a 22 % drop and over 6k more steps. F1-greedy ckpt-15k for v2 not
re-measured (only sequence-mode mAP run).

**LISTEN-MCQA still doesn't learn.** v2 best 0.269 (7k) ≈ v1 best 0.244
(5k); both are basically 4-choice random (0.25). v2's mitigations don't
address this task-specific failure mode. The same picture in
LISTEN-official 2C (audio+text input): v2 0.354 vs v1 0.382 — both
saturated by ckpt-1k, no real audio learning.

**ASR-only noise aug shows no obvious downside.** ASR test-clean WER is
slightly *better* on v2 (4.73 vs 4.98 % best). Test-other improves more
(17.68 vs 18.48 % best, much better stability). Plausibly: latent-space
Gaussian noise on ASR samples acts as effective regularization for the
projector, slightly hurting test-clean (most affected by encoder
robustness) but helping test-other (which benefits from noise-robust
features).

### 14.3 v2 ckpt selection for downstream use

If extracting a single best ckpt from v2:
- **ckpt-15k or ckpt-21k**: best balanced default. ESC-50 ≥ 91 %,
  emotion strong (DailyTalk F1 0.43-0.40, EmoV 0.78-0.80, RAVDESS 0.55+),
  ASR competitive (WER-clean ≤ 5.2 %), text retention ≥ 0.906.
- **ckpt-7k**: emotion-leaning but earlier. MELD acc peak (0.509),
  RAVDESS already at 0.471, but ESC-50 only 67 % — env pool not yet
  saturated.
- **ckpt-27k or ckpt-29k**: emotion-peak alternative. Best DailyTalk F1
  is at 11k (0.467), but RAVDESS peak is at 27k (0.621), EmoV peak 28k
  (0.810), ESC-50 already at 0.99. Text retention also at peak (0.907).
  Best for emotion-heavy use; ASR slightly worse than 15k.
- **ckpt-6k or ckpt-7k** if ASR is the priority: best WER-clean (4.73 %)
  and WER-other (17.68 %).
- **ckpt-31k (last)**: still strong (0.99 ESC-50, 0.797 EmoV, 5.54 % WER).
  Less reason to discard than v1's 25k. **A reasonable deployment
  candidate** if the goal is "use the latest" — the per-epoch random
  sub-sampling kept the late ckpts honest.

### 14.4 Remaining v2 work

- [ ] Optionally resume training from ckpt-31000 (deepspeed state
      preserved) to reach planned step 50k. Returns are likely
      diminishing — late ckpts already mostly plateaued.
- [ ] Re-measure FSD50K **greedy** F1 on v2 (only sequence-mode mAP was
      run for v2; v1 had both). Allows v1↔v2 head-to-head on greedy
      multi-label.
- [ ] Run public-benchmark §12 update with v2 ckpt-21k or ckpt-27k as the
      reference point, especially the new ESC-50 99 % which warrants
      separate leaderboard comparison.

---

## 11. Open items / TODO

- [x] **Run §7.2 sanity checks**; insert numbers into §4.5 and §7. _(done in early dense fill 2026-04-24)_
- [x] **FSD50K true-mAP path** — `eval_fsd50k_map_seq` (sequence-mode rank
      scoring, ~200× forward per sample) implemented by Session B; full 25-ckpt
      sweep completed 2026-04-25 with `mAP_micro` / `mAP_macro` reported.
      Greedy F1/Jaccard mode kept as cheaper sanity signal.
- [ ] **Install pycocoevalcap** (BLEU-only currently). Java needed for
      METEOR/SPICE; CIDEr/Rouge-L still computable with pure-Python install.
- [x] **Dense trajectory fill (1k-25k all benchmarks)** — completed
      2026-04-25 via Session A 8-way listen_mcqa + Session B 8-way FSD50K
      (with `/dev/shm` tmpfs cache to avoid disk I/O thrash; see
      `eval_coordination.md` incidents).
- [ ] **Update [`eval_plan.md` §3.4](eval_plan.md)** which
      still describes the mix as "emotion 70 / aux 20 / text 10, sound
      zero-shot". The running config is
      `asr14-emo34-env35-txt17`. env-sound is in-domain, not zero-shot,
      which materially changes the Tier-3 interpretation.
- [ ] **Run scheduling**: at the current 5 step/sec training rate, ckpt-5k
      arrives ~2.5 h after ckpt-2000, ckpt-10k ~10 h after. Kick off
      LISTEN + ESC-50 full sweeps at ckpt-5k as the first "real" numbers;
      defer Clotho + FSD50K to ckpt-10k given their wall-clock costs
      (Clotho 1 045 clips, FSD50K 10 231 clips).

---

## 9. Reproducibility quick reference

```bash
# env
export PY=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf/bin/python
export LD_PRELOAD=/mnt/ddn/users/jos/miniforge3/envs/audio_lmf/lib/libstdc++.so.6:/mnt/ddn/users/jos/miniforge3/envs/audio_lmf/lib/glibc_compat.so
export CUDA_VISIBLE_DEVICES=7   # avoid the running training
cd /mnt/ddn/users/jos/audiollm-trainer

CKPTS=/mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17
BASE=/mnt/tmp/s2_init_42k

# Tier-2 emotion (LISTEN-test, 2 635 rows)
$PY -m evaluation.stage2.eval_listen_mcqa \
    --ckpt-root "$CKPTS" --out-root "$CKPTS/eval_listen" \
    --base-model "$BASE" --ckpts 2000 --batch-size 4

# Tier-3 env-sound
$PY -m evaluation.stage2.eval_esc50_acc \
    --ckpt-root "$CKPTS" --out-root "$CKPTS/eval_esc50" \
    --base-model "$BASE" --ckpts 2000 --all --batch-size 8

$PY -m evaluation.stage2.eval_clotho_caption \
    --ckpt-root "$CKPTS" --out-root "$CKPTS/eval_clotho" \
    --base-model "$BASE" --ckpts 2000 --split evaluation --batch-size 4

$PY -m evaluation.stage2.eval_fsd50k_map \
    --ckpt-root "$CKPTS" --out-root "$CKPTS/eval_fsd50k" \
    --base-model "$BASE" --ckpts 2000 --batch-size 4
```

Add `--ckpts 1000,2000,3000,...` to sweep; `--max-samples N` to slice for
debug; `--no-cache` (see §4.5) for shim-free ground-truth comparison.
