# Stage 2 — evaluation plan & data decisions (2026-04-23)

Scope: how we'll measure Stage 2, and therefore which training-data additions
(beyond the ASR superset) make sense. Written during smoke, to be iterated through
discussion. Pairs with [`stage2_design.md`](stage2_design.md) (setup) and
[`stage2_smoke.md`](stage2_smoke.md) (pre-launch validation results).

---

## 1. Evaluation philosophy

1. **Train and eval both in "letter + rationale" (CoT-style MCQA) format.** Emotion /
   paralinguistic benchmarks are MCQA by nature (IEMOCAP / MELD / MMAU-speech are all
   classification); the training format must produce outputs from which a letter can
   be parsed. Plain letter-only training has two problems — weak gradient signal
   (1 token) and letter-prior shortcut. Plain free-form training mismatches the eval
   protocol. Letter + rationale threads the needle: 20–50 token gradient, no
   shortcut (must attend to audio to justify), letter-first parseable.
2. **Contamination-first dataset selection.** Prefer eval sets that are guaranteed
   held-out against the training data pool. The biggest risk this project: LISTEN
   aggregates many emotion sources — if it pulls test portions of standard
   benchmarks, those benchmarks can no longer be held-out eval.
3. **Fixed benchmarks, fixed metrics, fixed numeric targets before launch.** So we
   know when to stop and what counts as success. See §4.

---

## 2. Evaluation matrix

| Tier | Capability | Benchmark | Metric | Held-out? |
|---|---|---|---|---|
| 1 | ASR in-domain | LibriSpeech test-clean | WER (Whisper normalizer) | ✅ |
| 1 | ASR in-domain noisy | LibriSpeech test-other | WER | ✅ |
| 1 | ASR OOD read | TED-LIUM 3 test | WER | ✅ |
| 1 | ASR OOD multi-speaker | FLEURS en test | WER | ✅ |
| 2 | Paralinguistic — emotion (primary) | **LISTEN test split** | accuracy / macro F1 | ✅ LISTEN ships with train/test split |
| 2 | Paralinguistic — multi | MMAU-speech | accuracy | ✅ (assumed independent, verify) |
| 2 | Paralinguistic — aux | VibeCheck1 held-out split | accuracy / F1 | ⚠ format TBD |
| 2 | Paralinguistic — cross-corpus (optional) | IEMOCAP / MELD test | macro F1 | ⚠ usable only if LISTEN excludes their test items from LISTEN-train |
| 3 | Sound captioning | AudioCaps test (975) | CIDEr, BLEU-4, SPIDEr | ✅ if AudioCaps not in train |
| 3 | Sound captioning | Clotho eval | CIDEr, BLEU | ✅ **always** (Clotho is eval-only) |
| 3 | Sound understanding | MMAU-sound + MMAU-music | accuracy | ✅ |
| 4 | Text retention | MMLU (or lighter: HellaSwag / ARC-e) | accuracy | ✅ text-only |
| 5 | Unified audio-LLM | AIR-Bench chat + foundation | bench-reported | ✅ (zero-shot) |

Tier intent:

- **Tier 1** — primary success metric, Stage 2 must not regress ASR.
- **Tier 2–3** — new capabilities Stage 2 is meant to unlock.
- **Tier 4** — guardrail against LoRA frying the LLM's general language ability.
- **Tier 5** — single-number aggregate for paper/report comparison against published
  audio-LLM baselines.

---

## 3. Dataset choices (training side)

### 3.1 Emotion — LISTEN (user confirmed) + rationale synthesis

Confirmed: **LISTEN** / **LISTEN_full** is a composite dataset — aggregates multiple
emotion sources, reclassified under a unified label schema, MCQA format.

**Advantages**:
- Already MCQA, matches our "letter + rationale" training format.
- Label balance / unified schema already handled by LISTEN authors.
- Scale: probably the largest single unified audio-emotion source we can use.

**The contamination problem**:
- LISTEN likely draws from IEMOCAP, MELD, MSP-Podcast, CREMA-D, RAVDESS, etc.
- If LISTEN respects each source's official train/val/test split — IEMOCAP/MELD/etc.
  test portions remain valid held-out eval.
- If LISTEN pools everything and re-splits — those benchmark test splits are
  contaminated, unusable as held-out.

**Partitioning**: LISTEN ships with its own **train / test split** (user-confirmed).
LISTEN-test is therefore a clean held-out eval. Still open: whether LISTEN-train
includes items that were test-set in the source corpora (IEMOCAP / MELD / etc.) —
this only matters if we also want to report cross-corpus numbers on those. For the
primary emotion metric we use LISTEN-test and don't need that question answered.

**Training augmentation**: LISTEN items are letter-only today. Synthesize rationales
offline using an LLM with transcript + ground-truth label:
```
prompt: "Speaker says: '{transcript}'. Ground-truth emotion: {label}.
In 1–2 sentences, explain WHY this sounds like {label} using audio cues only —
prosody, pitch, pacing, energy. Do not mention face / visual."
```
Result: each training sample becomes `Answer: {letter}. Justification: {rationale}`.
20–50 tokens of supervision, letter-first.

**Eval protocol** (both for LISTEN-internal held-out and for external benches like
MMAU): prompt model with CoT prefix ("Answer with the letter, then briefly explain."),
parse first letter from output. Standard CoT MCQA evaluation.

### 3.2 Sound (non-speech audio) — reference only, NOT in current S2 mix

Not in the confirmed Stage 2 plan (see §3.4), kept here as reference for a later
stage if/when sound capability is needed.

**Caption-style sources (if included in the future)**:
- **WavCaps** (~400 k, auto-generated + ChatGPT-filtered captions) — main mass.
- **AudioCaps train** (~45 k, human captions) — quality signal. Both produce natural
  captions, loss shape ≈ ASR, autoregressive generation.

**Sound reasoning** (rarer than emotion reasoning, similar shortage of native data):
- **MMAU-sound** — some sub-tasks are reasoning-style ("Why does this sound suggest
  X?"). Designed as benchmark, not training set.
- **Clotho-AQA** — audio + question + answer. Reasoning element present (not pure
  captioning).
- **AudioSetCaps / WavCaps** — caption only, not reasoning.
- Audio-only reasoning data at scale is effectively absent; realistic path = AudioCaps
  (or WavCaps) + GPT-4 rationale augmentation, same pattern as emotion (§3.1).

**Held-out eval** (in the eval matrix regardless of whether we train on sound):
- **AudioCaps test** (975 × 5 captions) — CIDEr / BLEU / SPIDEr. Zero-shot if we
  don't train on sound.
- **Clotho eval** (1 045 × 5 captions) — fully independent corpus, always held out.
- **MMAU-sound + MMAU-music** — zero-shot.

Zero-shot scores without sound training data give a baseline; treating sound as a
future-stage addition rather than bundling into S2 keeps the S2 mix focused.

### 3.3 Text reasoning (in S2 mix — §3.4)

Not "text retention" (preventing MMLU drop as a side-effect) but **text reasoning as a
first-class training signal**: SFT on text-only chain-of-thought / instruction-following
data so the LLM's reasoning pathway stays healthy and actually gets exercised through
LoRA. Complements the "letter + rationale" emotion format by giving the rationale
skill text-only supervision too.

**Candidates** (pick one or a small mix):
- **Open-Orca** / **OpenHermes 2.5** — general instruction + reasoning SFT, ~1 M+ items.
- **MetaMathQA / GSM8K train** — math reasoning specifically.
- **OpenThoughts** / **Reasoning-SFT** family (2024+) — explicit CoT traces.
- **Magpie** — model-distilled SFT, high-diversity.

For this project a general-purpose instruction-tuning set (Open-Orca style) is the
safest bet; math-only would narrow too much given the audio-LLM downstream use.

### 3.4 Stage 2 mix — confirmed plan

User-confirmed scope: **LoRA SFT on emotion (VibeCheck1 / LISTEN_full) + text
reasoning 10%**. ASR superset is NOT in the S2 mix; Stage 1 already spent 50 k steps
on ASR and the projector is converged.

| Category | Source | Share | Rationale |
|---|---|---:|---|
| Paralinguistic — emotion | **LISTEN / LISTEN_full** (MCQA) + rationale augmentation | ~70 % | MCQA + rationale; matches eval protocol |
| Paralinguistic — emotion (aux) | **VibeCheck1** (format TBD — see §5.4) | ~20 % | second emotion source |
| Text reasoning (SFT, audio-free) | text-only CoT / instruction data (§3.3) | **10 %** | keep reasoning skill alive under LoRA; user-set |

Emotion vs aux split (70 / 20) is a placeholder — actual split after LISTEN +
VibeCheck1 absolute sizes are known.

**Implications / risks to flag**:
1. **No ASR in S2 training.** Projector is full-trainable in S2; with zero
   audio-transcript signal, projector representation can drift away from ASR
   optimum. Two mitigations to consider:
   - Freeze projector in S2 (turn `audio_encoder.projector` out of the workflow
     filter). Cleanest way to preserve Stage 1 work.
   - Include a small (≤ 10 %) ASR sample as regularization anchor — a few k
     LibriSpeech items mixed in.
   - Accept drift, measure ASR post-S2 via Tier 1, decide then.
2. **Sound capability will be zero-shot in eval.** Tier 3 numbers reflect transfer
   from emotion + text reasoning to sound understanding — informative about
   generalization, but not comparable to audio-LLM baselines that trained on sound.
3. **Share**: text-reasoning = 10 % (user-set). Emotion sub-split (LISTEN vs
   VibeCheck1) finalizes once both absolute sizes are known.

Manifest build: single `omni_manifest` JSONL mixing emotion (audio + MCQA prompt +
letter + rationale) with text-reasoning (no audio, pure instruction). Collator handles
both cases if `audio_features` is empty for text-only rows — need to verify omni
collator supports this (currently assumes audio_lengths > 0; may need a small patch).

---

## 4. Success criteria (draft — to lock before launch)

Given S2 mix drops ASR (§3.4), ASR criterion is now "don't regress too much" rather
than "improve".

1. **Ship gate (must hit)**:
   - test-clean WER drift ≤ **+1.0 %** from S1's 7.28 % (i.e. ≤ **8.3 %**). Accept
     some drift because no ASR supervision; large drift means projector moved too much.
   - test-other WER ≤ **16 %** (similar tolerance).
   - MMLU or HellaSwag drop ≤ **5 points** vs. base Qwen3.5-4B.
   - IEMOCAP 5-class macro F1 ≥ **0.50** (if held-out after LISTEN check).
2. **Nice to have**:
   - test-clean WER ≤ **7.28 %** (no regression) — tightening the ship gate.
   - AudioCaps test CIDEr zero-shot — report as baseline for future sound-stage.
3. **In-loop signal**:
   - Every 5 k steps, eval on a 200-sample test-clean slice (single GPU, quick).
   - Also every 5 k steps, eval on a small LISTEN-internal held-out slice.
   - Early-stop trigger if both deteriorate for 2 consecutive checks.

Adjust numbers per actual priorities.

---

## 5. Decisions needed before launch

Ranked by urgency (post-scope-narrow to LISTEN + VibeCheck1 + text-reasoning 10 %):

1. **Projector handling in S2 without ASR supervision** — freeze, keep trainable, or
   add small ASR anchor? (§3.4 flag #1) — biggest open architectural question.
2. **Hybrid-attention LoRA coverage** — A (current: 8 full-attn layers) / B
   (+linear-attn `in_proj_*` for all 32) / C (+MLP). See [`stage2_smoke.md §3`](stage2_smoke.md#3-hybrid-attention--surfaced-needs-decision).
3. **Text-reasoning SFT corpus choice** — Open-Orca / OpenHermes / Magpie /
   OpenThoughts? Quality vs. diversity tradeoff. (§3.3)
4. **VibeCheck1 format + source** — spec needed for rationale-augment vs. as-is
   ingestion decision, and to confirm its sources don't overlap LISTEN train.
5. **Optional cross-corpus eval** — do we want to chase IEMOCAP/MELD cross-corpus
   numbers for paper comparability? Only if LISTEN excludes their test items.
4. **VibeCheck1 format + source pool** — spec / sample row determines whether we use
   as-is, augment with rationale, or re-format.
5. **Success criteria numeric values** — §4 thresholds, adjust per project goals.
6. **In-loop eval frequency** — 5 k steps? post-hoc sweep only?

Once 1–4 are decided, remaining work:
- Build the composite manifest (ASR + sound + emotion + optional text) with weights.
- Add a lightweight val-WER trainer callback if we commit to in-loop eval.
- Finalize archive / output dir naming for the real Stage 2 run.

---

## 6. Data inventory (2026-04-24)

Corpora actually downloaded to `/mnt/tmp/datasets/` so far, with their intended
train/eval usage. Contamination audit + filter pipeline documented in
[`stage2_listen_leakage_audit.md`](stage2_listen_leakage_audit.md).

### 6.1 Emotion (speech with affect labels)

Numbers below reflect the training-side kept rows **after** held-out split
enforcement (MELD-test excluded / DailyTalk last 5 % out / EmoV-DB Jenie out /
RAVDESS actors 21-24 out / RAVDESS hash-match exclusions). Subsampling into
the final Stage 2 mix happens on top of these.

| Corpus       | Kept rows (train) | Hours (train) | Eval split                | License / status                       |
|--------------|------------------:|--------------:|---------------------------|----------------------------------------|
| DailyTalk    |            22 573 |       ~20.6 h | internal 5 % held-out (1 168 dialogues) | ✅ CC-BY-NC-SA 4.0 (GDrive) |
| MELD         |            11 043 |        ~9.7 h | MELD-test (2 747 audios, held-out; 57 LISTEN-test audios also excluded from train) | ✅ research |
| EmoV-DB      |             5 103 |        ~7.3 h | Jenie speaker (1 790 wavs, held-out for speaker-out SER eval) | ✅ CC BY 4.0 (OpenSLR 115) |
| RAVDESS      |             1 199 |        ~1.2 h | Actors 21-24 (240 wavs) + 65 hash-matched LISTEN-test audios | ✅ CC-BY-NC (Zenodo) |
| MUStARD++    |                65 |             — | videos partial (65 / 1 202, GDrive rate-limit) | ✅ research |
| CREMA-D      |                 0 |             — | ❌ LFS budget exceeded; HF mirrors all 404 | ✅ ODbL, blocked |
| ESD          |                 0 |             — | ⏸ EULA (email author)                | ⚠ license signing |
| IEMOCAP      |                 0 |             — | ⏸ USC EULA                             | ⚠ pending |
| MSP-Podcast  |                 0 |             — | ⏸ UTD EULA                             | ⚠ pending |
| MSP-IMPROV   |                 0 |             — | ⏸ UTD EULA (bundle)                    | ⚠ pending |
| TESS / SAVEE |                 0 |             — | skipped                               | — |
| OMG-Emotion  |                 0 |             — | deferred (yt-dlp pipeline)            | — |
| CMU-MOSEI    |                 0 |             — | deferred (CMU-MultimodalSDK)          | — |
| SEND / JL-Corpus / EMNS | — |           — | ⚠ attempted — all blocked (Stanford SSNL access, Kaggle auth, unpublished HF) | — |
| **Total (kept)** |       **39 983** |              |                                        |                                        |

After ~2026-04-24 mix subsampling and the DailyTalk dialog-split fix, the emotion
pool entering the combined manifest is **39 919** rows (minor delta from per-source
MCQA label resolution — 64 MUStARD context-only clips dropped since they have no
sarcasm label).

LISTEN_full itself is **not** used as a training source in this revised plan
(§3.1 superseded by [`stage2_listen_leakage_audit.md` §6](stage2_listen_leakage_audit.md#6-current-plan--full-source-corpora-minus-listen-test));
its test split remains the **primary Tier-2 eval**.

### 6.2 Environmental sound

| Corpus | Train rows | Train hours | Eval split | Status |
|---|---:|---:|---|---|
| FSD50K dev | 40 966 | 81.5 h | FSD50K eval (10 231, 27.9 h) | ✅ extracted (dev + eval) |
| Clotho development | 3 839 | 23.99 h | Clotho evaluation 1 045 (6.5 h) + validation 1 045 (6.6 h) | ✅ extracted |
| ESC-50 | 2 000 | 2.78 h | 5-fold CV (hold 1 fold / iter) | ✅ done |
| MACS | 0 | — | caption-only yaml, audio requires TAU Urban | ⚠ audio blocked |
| **Available** | **46 805** | **~108.3 h** | | |
| **Subsampled for S2 mix** | **41 000** | — | | |

### 6.3 Text benchmarks (Tier 4 guardrail + text-reasoning SFT)

| Benchmark    | Train | Validation | Test  | Usage in S2                          |
|--------------|------:|-----------:|------:|--------------------------------------|
| HellaSwag    | 39 905 |     10 042 | 10 003 | Tier 4 eval: validation (labeled)    |
| WinoGrande   | 40 398 |      1 267 |  1 767 | Tier 4 eval: validation (labeled)    |
| BoolQ        |  9 427 |      3 270 |     — | Tier 4 eval: validation              |
| ARC-Easy     |  2 251 |        570 |  2 376 | Tier 4 eval: test (labels public)    |
| ARC-Challenge|  1 119 |        299 |  1 172 | Tier 4 eval: test (labels public)    |
| COPA         |    400 |        100 |    500 | Tier 4 eval: validation              |
| **Total**    | **93 500** | **15 548** | **16 818** |                                      |

All splits at `/mnt/tmp/datasets/text_benchmarks/<name>/<split>.parquet`.
PiQA dropped (legacy loader deprecated, mirrors 404).

---

## 7. Per-corpus eval protocol

### 7.1 Tier 2 — Emotion / paralinguistic

**Primary**: `LISTEN-test` (2 635 rows across 11 source sub-corpora). MCQA
protocol with letter-first parsing. Metric: overall accuracy + per-sub-corpus
accuracy, with contamination caveat (MUStARD / PODCAST / MOSEI carry audio-level
overlap with LISTEN-train in the original LISTEN distribution — see audit §2).
Our training pool avoids this by not training on LISTEN-train directly.

**Secondary (external, audio available)**:
- **MELD-test** — 2 747 WAVs at `/mnt/tmp/datasets/emotion_raw/MELD/audio/test/`.
  Held-out iff we train on MELD train+dev and exclude the 57 LISTEN-test-mapped
  items. Metric: 7-class macro F1 (neutral / surprise / joy / anger / sadness /
  fear / disgust). Use the official `test_sent_emo.csv` labels.
- **DailyTalk internal** — last 5 % of dialogues as held-out split (choose by
  dialog ID modulo to keep speaker stats consistent). Metric: 7-class accuracy.
- **EmoV-DB speaker-out** — hold Jenie (1 speaker) for eval; train on Bea +
  Josh + Sam. Tests speaker-independent emotion recognition. Metric: 5-class
  accuracy on Jenie utterances.
- **RAVDESS** — hold actors 21-24 (standard speaker-held-out protocol for
  RAVDESS SER eval). Metric: 8-class accuracy on held-out actors.

**Tertiary (EULA-pending, add when available)**:
- **IEMOCAP** Session 5 held-out — standard SER protocol, 4-class (happy / sad /
  angry / neutral) macro F1. Already excluded from training pool via audit rule.
- **MSP-Podcast Test1** (if we receive it) — 7-class macro F1.

**Prompt format** (shared across all Tier 2):
```
<audio>
Q: What emotion does the speaker convey?
Choices: A) <emo1>  B) <emo2>  ...
A: Answer with the letter, then briefly explain.
```
Parse: first letter in response. Rationale string is logged but not scored.

### 7.2 Tier 3 — Environmental sound

**Sound captioning**:
- **Clotho evaluation** (1 045 clips, 5 captions each) — CIDEr, BLEU-4, METEOR,
  SPICE, SPIDEr. Standard AudioCaps / Clotho metric suite.
- **Clotho validation** (1 045) — same metrics; use as in-loop dev checkpoint.

**Sound classification**:
- **FSD50K eval** (10 231, 200 labels) — mAP (macro), d'-mAP, precision@1.
  Multi-label, so use `sklearn.metrics.average_precision_score(..., average='macro')`.
- **ESC-50 5-fold CV** — held-one-fold-out accuracy, averaged across 5 folds.
  Standard ESC-50 protocol (fold column in meta CSV).

**Zero-shot note**: if Stage 2 does NOT train on env-sound (per `stage2_design.md`
§6.1 plan-literal), Tier 3 numbers are all zero-shot transfer from emotion +
text training. Report as baseline for future sound stage.

### 7.3 Tier 4 — Text retention

**Protocol**: CoT-MCQA exactly like Tier 2, but audio-free.
```
Q: <question>
Choices: A) <opt1>  B) <opt2>  ...
A: Answer with the letter.
```

**Benchmarks + metrics**:
- HellaSwag validation — accuracy on `label` column (0–3 → A–D).
- WinoGrande validation — accuracy on `answer` column (1 / 2 → A / B).
- ARC-Easy + ARC-Challenge test — accuracy (4-way MCQA).
- BoolQ validation — yes/no accuracy (binary letter A=yes, B=no).
- COPA validation — 2-way accuracy on premise + effect/cause.

**Aggregation**: report each individually + unweighted mean across the 6.

**Leakage guard** (§3.3 caveat): the text-reasoning SFT corpus chosen (Open-Orca
/ OpenHermes / etc.) must be deduped against these six before committing.
Dedup by question-hash on normalized text.

### 7.4 Tier 5 — Unified audio-LLM

AIR-Bench chat + foundation as published. Zero-shot (no AIR-Bench training).
Use the official harness; report bench-native aggregates.

---

## 8. Train/eval split decisions for new corpora

Rationales for split choices added 2026-04-24:

| Corpus        | Split choice                                         | Why                                                                          |
|---------------|------------------------------------------------------|------------------------------------------------------------------------------|
| MELD          | train+dev → training; test → eval (−57 LISTEN-mapped)| Official MELD test is the standard comparison point; LISTEN exclusion keeps held-out clean |
| DailyTalk     | 95 % dialogues → train; last 5 % dialogues → eval    | Dialogue-level split avoids intra-dialog leakage; TTS-style 1-speaker-per-dialog makes speaker-out infeasible |
| EmoV-DB       | 3 speakers → train; Jenie → eval                     | Classic speaker-independent SER protocol; Jenie is the FR speaker so also doubles as cross-lingual robustness indicator |
| RAVDESS       | actors 01-20 → train; actors 21-24 → eval            | Standard RAVDESS speaker-held-out convention (see Livingstone 2018 §3.2)     |
| IEMOCAP       | Session 1-4 → train; Session 5 → eval                | Standard SER protocol; already excluded from LISTEN-derived training pool    |
| FSD50K        | dev split → train; eval split → eval                 | Official FSD50K split, disjoint by design                                    |
| Clotho        | development → train; evaluation → held-out eval; validation → in-loop dev | Matches DCASE Task 6 protocol; validation usable for early-stop trigger      |
| ESC-50        | 5-fold CV (folds rotate each eval run)               | Corpus is too small for fixed test set; the community uses 5-fold CV         |

Splits are enforced by the training-manifest builder
([`scripts/emo/build_training_manifest.py`](../scripts/emo/build_training_manifest.py)):
eval-side files never enter the training manifest.

---

## 9. Outstanding eval-side actions

1. **Rationale synthesis offline pass** (§3.1) — write a GPT-4 / Llama-3-70B prompt
   pipeline that converts each emotion training row into `{letter, rationale}`.
   Input: transcript + ground-truth label. Output: 20–50-token justification
   grounded in audio cues only.
2. **Env-sound inclusion decision** — plan-literal (Tier 3 zero-shot) vs. plan+env
   10 % (include Clotho dev + ESC-50 as training captioning signal, cover ~6 h of
   additional audio, same format as emotion: Q + audio + A+rationale).
3. **Tier 4 contamination audit** — hash-dedup between chosen text-reasoning SFT
   corpus and the six Tier 4 benchmarks (train + test splits both).
4. **RAVDESS hash match** — resample-normalized fingerprint to locate the 200
   LISTEN-test RAVDESS audios within the 1 440-file raw corpus and exclude them.
5. **EULA follow-up** — IEMOCAP, ESD, MSP-Podcast (+IMPROV +Conversation bundle).
