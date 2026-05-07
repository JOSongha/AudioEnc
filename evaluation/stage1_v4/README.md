# Stage-1 caption-prompt eval (legacy v4 dir name, fits v5 / v6 better)

Stage-2 evaluators (`evaluation.stage2.eval_*`) ask the model with a
classification-style stem (`Classify this sound.`, `List the sound events
in this audio, separated by commas.`) that mirrors the Stage-2 training
mixture. **All Stage-1 chains since v3 train the audio_env_sound modality
in caption-form prompts** (TASK_PROMPTS["sound_caption"] / sound_describe_*),
so the Stage-2 classification prompts are OOD across the entire Stage-1
lineage and underestimate the encoder.

**Caveat (target-text drift)**: caption-form *prompt* is unchanged across
v3 / v4 / v5 / v6, but caption *target text* changed in v5:
- v3 / v4 AudioSet & FSD50K: synthetic "sound of X, Y, Z" comma-list captions
  built from labels.
- v5 + v6: ontology-description sentence captions (see
  [`datasets.md` v5 changelog](../../docs/setup/datasets.md)).

So when these wrappers force `--score-mode sentence` (asking the model for a
sentence response) they fit v5 / v6 ckpts cleanly, but penalize v4 ckpts —
v4 was trained to emit "sound of X, Y" comma-list, not full sentences. F1
on v4 ckpts via this wrapper is artificially low. Treat the dir name
`stage1_v4/` as legacy; for v6 production ckpts these wrappers are the
correct caption-form eval path.

This folder mirrors the three sound-classification evaluators with
caption-form prompts and free-form output parsing.

| Stage-2 evaluator | v4 wrapper | Prompt | Label recovery |
|---|---|---|---|
| `eval_audioset_map` (greedy) | `eval_audioset_caption` | `Describe what you hear in this audio. Mention every distinct sound event.` | substring match against AudioSet 632-class vocab (existing `parse_labels_sentence`) |
| `eval_fsd50k_map` (greedy)   | `eval_fsd50k_caption`   | same | substring match against FSD50K 200-class vocab |
| `eval_esc50_acc`             | `eval_esc50_caption`    | `Describe the sound you hear in this audio in one sentence.` | longest-match-first substring against ESC-50 50-class vocab, first hit wins |

The audioset / fsd50k wrappers are thin: they invoke the existing Stage-2
module with `--score-mode sentence` forced on, so all backend logic
(loader, batching, F1/Jaccard reporting, summary schema) is reused.
The esc50 module is a fresh implementation since the Stage-2 evaluator has
no caption mode.

## Tasks left on Stage-2 (already match v4 training)

These already use prompts that match v4 training and need no v4-specific
variant:

* `eval_librispeech_wer` (`Transcribe the audio to text.`)
* `eval_clotho_caption`  (`Describe what you hear in the audio.`)
* `eval_source_emotion`  (emotion-classify MCQA, identical training format)
* `eval_listen_mcqa`     (training-parallel emotion MCQA)
* `eval_listen_official` (upstream LISTEN harness, OOD by design)
* `eval_text_retention`  (text-only LM benchmarks; v4 keeps LM frozen,
  so caption form vs classification form is moot)

## Usage

```bash
# AudioSet (forces sentence score-mode under the hood):
python -m evaluation.stage1_v4.eval_audioset_caption \
  --ckpt-root <run>/Qwen3.5AE-ASR-Stage1-dac-vae-v4 \
  --base-model <repo>/external/models/Qwen3.5AE-4B \
  --out-root  <run>/eval_v4_caption/eval_audioset_caption \
  --ckpts 68000

# FSD50K:
python -m evaluation.stage1_v4.eval_fsd50k_caption \
  --ckpt-root <run>/Qwen3.5AE-ASR-Stage1-dac-vae-v4 \
  --base-model <repo>/external/models/Qwen3.5AE-4B \
  --out-root  <run>/eval_v4_caption/eval_fsd50k_caption \
  --ckpts 68000

# ESC-50 (5-fold):
python -m evaluation.stage1_v4.eval_esc50_caption \
  --ckpt-root <run>/Qwen3.5AE-ASR-Stage1-dac-vae-v4 \
  --base-model <repo>/external/models/Qwen3.5AE-4B \
  --out-root  <run>/eval_v4_caption/eval_esc50_caption \
  --ckpts 68000 --all --batch-size 8
```

Output schema matches the Stage-2 evaluators so `evaluation.stage2.aggregate_results`
already handles it (treats `eval_audioset_caption` / `eval_fsd50k_caption` /
`eval_esc50_caption` as new eval-dir names — would need a one-line addition
to `EVAL_METRICS` if you want it picked up automatically).

## Caveats

* **Caption-substring label recovery is conservative**: the model may
  describe the sound correctly without saying the canonical class word
  ("a baby crying" instead of "crying_baby"). False-negatives possible.
* **Multi-label sentence scoring** (audioset/fsd50k): the regex anchors
  on word boundaries, so longer compound names match before single words.
  See `evaluation.stage2.eval_audioset_map.parse_labels_sentence` for the
  exact regex.
* **ESC-50 single-class disambiguation**: when multiple classes match the
  caption (e.g., "rain on the wind" matches both `rain` and `wind`), we
  pick the longest pattern first then by source order. This is a heuristic
  and trades recall for precision.
