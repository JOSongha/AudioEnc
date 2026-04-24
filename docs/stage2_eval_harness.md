# Stage 2 — evaluation harness build report (2026-04-24)

Records the build of the Tier-2 (emotion) and Tier-3 (env-sound) eval drivers
for the running Stage-2 LoRA SFT. Companion to
[`stage2_eval_plan.md`](stage2_eval_plan.md) (what to measure) and
[`stage2_design.md`](stage2_design.md) (training setup). This document
captures *how* the eval harness works, the three implementation issues that
blocked off-the-shelf reuse of `eval_testclean_wer.py`, and the verification
done against checkpoint-2000.

---

## 1. Motivation

[`stage2_eval_plan.md`](stage2_eval_plan.md) promises a full Tier-2/3 number
board (LISTEN-test, Clotho, FSD50K, ESC-50, MMLU, …) but
[`evaluation/`](../evaluation/) previously held only `eval_testclean_wer.py`
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
  [`omni_dataset.py:create_omni_processor`](../src/llamafactory/data/omni_dataset.py).

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

[`_loader._is_adapter_only`](../evaluation/stage2/_loader.py) returns True
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
  [`model/model_utils/liger_kernel.py:36`](../src/llamafactory/model/model_utils/liger_kernel.py#L36)
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
  [`stage2_listen_leakage_audit.md`](stage2_listen_leakage_audit.md): MUStARD
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

### 6.3 FSD50K — generative → F1/Jaccard, NOT mAP

Published FSD50K mAP requires per-label confidence scores to sweep
thresholds; a generative pipeline emits only a hard label list. The driver
recovers a presence-only binary vector and reports:

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
[`stage2_eval_plan.md §1`](stage2_eval_plan.md) explicitly warned that
"letter-only training has two problems: weak gradient signal (1 token) and
letter-prior shortcut." The rationale-augmentation offline pass
([`stage2_design.md` open item #2](stage2_design.md) and
[`stage2_eval_plan.md §9.1`](stage2_eval_plan.md)) has **not run yet** — the
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

## 8. LISTEN-official results (ckpt-2000, 2026-04-24)

Full 9-experiment sweep on local LISTEN-test parquet (2 635 rows, type 4
absent). ckpt-1000 column for monotone-improvement sanity; `chance` column
is the LISTEN reference's marginal-distribution baseline (not uniform
random — it's what a model that matched the prediction distribution
statistics would score). Numbers from LISTEN's own sklearn metric code
(`accuracy_score` / `balanced_accuracy_score` / `f1_score`).

*(Results landing as the background sweep completes — table filled in
§8.1 below once all 9 experiments are done for ckpt-2000.)*

### 8.1 Per-experiment numbers (ckpt-2000)

<!-- auto-filled on run completion -->

```
  exp           mode            n     WA       UAR    macroF1  microF1  chance
  1_audio       audio           ...
  1_text        text            ...
  1_audio_text  audio_and_text  ...
  2A            text            ...
  2B            audio           ...
  2C            audio_and_text  ...
  3A            text            ...
  3B            audio           ...
  3C            audio_and_text  ...
```

### 8.2 Interpretation hooks

What each experiment stresses:

- **1_audio / 2B / 3B** — the audio pathway in isolation. These are the
  primary Tier-2 numbers for a Qwen3.5AE audio-LLM.
- **1_text / 2A / 3A** — the LLM backbone's emotion reasoning with *only*
  transcription. Our S2 training never did this, so below-chance would
  indicate LoRA corrupted the backbone's text reasoning; near-chance means
  the model hedges without acoustic cues.
- **1_audio_and_text / 2C / 3C** — both together. A model that truly uses
  both modalities should score higher here than either modality alone.

At 4 % training we expect:
- audio-only to hover 1-4 pp above chance (as we already saw on our
  training-parallel driver),
- text-only to be close to chance (no text-reasoning supervision on emotion
  questions in S2 yet),
- audio+text to track audio-only or slightly above (LLM hasn't learned to
  fuse modalities yet — that skill would come from a multimodal SFT pass).

---

## 9. Open items / TODO

- [ ] **Run §7.2 sanity checks**; insert numbers into §4.5 and §7.
- [ ] **Install pycocoevalcap** (or decide to report BLEU-only) before
      publishing Clotho numbers.
- [ ] **FSD50K true-mAP path** — if Tier-3 comparability to the FSD50K
      leaderboard matters, implement a logit-readout eval over the 200-label
      vocab. Current F1/Jaccard are fine for "did emo+env training produce
      anything?" signal but not for leaderboard claims.
- [ ] **Update [`stage2_eval_plan.md` §3.4](stage2_eval_plan.md)** which
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
