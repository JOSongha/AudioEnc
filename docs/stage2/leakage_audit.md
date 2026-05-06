# LISTEN contamination audit — source-split leakage (2026-04-23)

Scope: does `VibeCheck1/LISTEN_full` pull from the **training splits** of its source
corpora (IEMOCAP, MELD, MOSEI, MSP-Podcast, OMG, MUStARD, CREMA-D, RAVDESS, TESS,
SAVEE, Emotion-Speech)? And — the flip side, which matters more for our eval plan —
does LISTEN-train pull from the **test splits** of those corpora? Pairs with
[`eval_plan.md §3.1`](eval_plan.md) (the open question flagged in §5.5).

Data analyzed: `/mnt/tmp/listen_analysis/data/{train,test}-*.parquet` (LISTEN_full
parquet shards) and `/mnt/tmp/listen_analysis/listen_test_ids_by_source.json`.
LISTEN IDs embed the source-corpus identifier verbatim (e.g.
`MELD_test_189_4_2B` ← MELD's *test* split; `MOSEI_Val_modified_...` ← MOSEI's *val*
split), which makes provenance recoverable from IDs alone without loading the source
corpora themselves.

---

## 1. TL;DR

Two independent contamination problems, ranked by impact on Stage 2 eval:

1. **LISTEN-train pulls source-test items.** Direct leakage into any future eval on
   the source corpora's native test split.
   - **MELD:** 881 unique `MELD_test_*` audios + 361 `MELD_dev_*` audios are in
     **LISTEN-train**. Only 218 LISTEN-train audios come from MELD-train proper.
   - **MOSEI:** 62 unique `MOSEI_Test_*` audios + 42 `MOSEI_Val_*` audios are in
     LISTEN-train. Only 73 come from MOSEI-Train.
   - **PODCAST / OMG / IEMOCAP / MUStARD:** source ID doesn't expose the original
     split string, but LISTEN-train spans all original sessions / splits of these
     corpora (see §3). Treat them as **not-held-out** for any report against their
     native test split.
2. **LISTEN's own train↔test audio overlap** (same base audio, different MCQA
   question variant, appearing in both LISTEN-train and LISTEN-test):
   - **MUStARD 68%**, **PODCAST 69%**, **MOSEI 66%**, **OMG 33%**, **MELD 18%**,
     **IEMOCAP 9%**, others 0%.
   - Effect: LISTEN-test accuracy is an **upper bound**, not a clean held-out number
     — the evaluator will have seen those audios during LISTEN-train, only with a
     different question prompt.

Practical consequence for [`eval_plan.md`](eval_plan.md):
- **Drop the "optional cross-corpus IEMOCAP/MELD eval" (Tier 2 last row, §5.5).**
  Both corpora are contaminated — reported numbers would not be honest held-out.
- **LISTEN-test remains the primary emotion metric, but report its accuracy with a
  caveat** that 33–69% of test base audios for five sub-corpora appeared (as other
  question variants) in LISTEN-train.
- **MMAU-speech becomes the one clean external emotion benchmark** assuming its
  speech items don't overlap LISTEN's source corpora (verify separately).

---

## 2. LISTEN-train sample counts, by source corpus

| Source         | Rows (train) | Unique audios (train) | Rows (test) | Unique audios (test) | Train↔test base-audio overlap |
|----------------|-------------:|----------------------:|------------:|---------------------:|------------------------------:|
| IEMOCAP        |        2 575 |                 2 122 |         392 |                  392 |            35 / 392   ( 9 %) |
| MELD           |        1 600 |                 1 460 |         388 |                  388 |            70 / 388   (18 %) |
| OMG            |        1 600 |                 1 339 |         382 |                  382 |           127 / 382   (33 %) |
| PODCAST        |        1 280 |                   868 |         294 |                  294 |           203 / 294   (69 %) |
| Emotion-Speech |          800 |                   800 |         200 |                  200 |             0 / 200   ( 0 %) |
| CREMA-D        |          800 |                   800 |         200 |                  200 |             0 / 200   ( 0 %) |
| TESS           |          800 |                   800 |         200 |                  200 |             0 / 200   ( 0 %) |
| RAVDESS        |          798 |                   798 |         200 |                  200 |             0 / 200   ( 0 %) |
| MUStARD        |          768 |                   524 |         173 |                  173 |           118 / 173   (68 %) |
| MOSEI          |          256 |                   177 |          59 |                   59 |            39 /  59   (66 %) |
| SAVEE          |          230 |                   230 |          58 |                   58 |             0 /  58   ( 0 %) |
| **Total**      |   **11 507** |            **10 018** |   **2 635** |            **2 635** |                         —     |

Notes:
- *Row vs. unique audio*: LISTEN wraps each audio in multiple MCQA question variants
  (`_2A`, `_2B`, `_3A`, …). "Unique audios" = rows after stripping that suffix.
- *Train↔test base-audio overlap*: the count of LISTEN-test audios whose base ID
  also appears in LISTEN-train under at least one question variant.
- No ID overlap exists at the row level (same ID in both splits): 0 across all
  sources. The problem is *audio-level* repetition via question variants.

---

## 3. Source-split provenance, corpus-by-corpus

Ordered by severity. "Provenance" is recovered from LISTEN IDs directly — wherever
the source corpus's split string is embedded in the ID, we can tell which split each
LISTEN sample came from without touching the source dataset itself.

### 3.1 MELD — severe contamination

MELD IDs preserve the original split: `MELD_{train|dev|test}_{dialog}_{utt}`.

LISTEN-train unique-audio provenance:
| Original MELD split | Unique audios in LISTEN-train | Share  |
|---------------------|------------------------------:|-------:|
| `test`              |                           881 |  60 %  |
| `dev`               |                           361 |  25 %  |
| `train`             |                           218 |  15 %  |

**Implication**: any report against MELD's official test split is invalid —
LISTEN-train trains on 881 of those audios directly. MELD-dev is also consumed.

LISTEN-test itself is also mostly pulled from MELD's non-train splits:
`{test: 235 rows, dev: 96 rows, train: 57 rows}`.

### 3.2 MOSEI — severe contamination

MOSEI IDs: `MOSEI_{Test|Val|Train}_modified_...`.

LISTEN-train unique-audio provenance:
| Original MOSEI split | Unique audios in LISTEN-train |
|----------------------|------------------------------:|
| `Test`               |                            62 |
| `Train`              |                            73 |
| `Val`                |                            42 |

Same pattern as MELD: MOSEI-Test items are trained on. Plus LISTEN's own train↔test
audio overlap is 66 % here — MOSEI is the worst-offender on both contamination axes
despite being the smallest source.

### 3.3 PODCAST (MSP-Podcast) — structure unknown, but 69 % internal overlap

MSP-Podcast IDs in LISTEN are opaque (`PODCAST_005515_...`), no split string embedded,
so we cannot recover original-corpus provenance from ID alone. Ambiguity aside, 203
of LISTEN-test's 294 unique audios (69 %) are also in LISTEN-train under different
question variants — that's enough on its own to rule out PODCAST for any clean
held-out report.

To confirm whether LISTEN draws from MSP-Podcast's official Test1/Test2 splits, we
would need to cross-reference with the official MSP-Podcast metadata — not done
here, flagged as optional follow-up (§5).

### 3.4 IEMOCAP — all 5 sessions present in both splits

IEMOCAP IDs: `IEMOCAP_Session{1-5}_...`. Unique-audio session distribution:

| Split        | S1  | S2  | S3  | S4  | S5  |
|--------------|----:|----:|----:|----:|----:|
| LISTEN-train | 398 | 382 | 441 | 434 | 467 |
| LISTEN-test  |  72 |  64 |  85 |  86 |  85 |

LISTEN ignores the conventional IEMOCAP Session-5-held-out convention — Session 5
audios appear in **both** LISTEN-train (467 unique) and LISTEN-test (85 unique). Any
downstream IEMOCAP eval using the "S5 = test" convention is contaminated because 467
Session-5 audios were seen in LISTEN-train.

### 3.5 OMG — 33 % internal overlap, no split provenance in ID

IDs `OMG_{hash}_{n}_{m}` don't expose original split. 127 / 382 LISTEN-test unique
audios (33 %) also appear in LISTEN-train.

### 3.6 MUStARD — 68 % internal overlap

IDs `MUStARD_{showcode}_{n}_{id}_u` (e.g. `MUStARD_ILL_2_168_u` = a utterance from
*I'm Lost* season 2). 118 / 173 LISTEN-test audios (68 %) are also in LISTEN-train.
MUStARD traditionally uses 5-fold CV rather than a fixed test split, so "leakage
into source test" isn't a meaningful notion — but LISTEN-internal contamination
stands.

### 3.7 CREMA-D / RAVDESS / TESS / SAVEE / Emotion-Speech — clean at audio level

No official source-corpus train/test split exists for CREMA-D, RAVDESS, TESS, SAVEE,
or Emotion-Speech (ESD). ESD ships with a train/eval/test split at the speaker-session
level but LISTEN IDs don't preserve it. Checked:

- LISTEN-train ∩ LISTEN-test audio-level overlap = **0** for all five.
- No speaker-level overlap check performed (a stricter bar). The train/test ID sets
  look speaker-disjoint for RAVDESS / CREMA-D by inspection (RAVDESS actor numbers
  differ between splits, CREMA-D speaker IDs differ), but not verified exhaustively
  here.

These five are safe to treat as uncontaminated **within** LISTEN for the LISTEN-test
metric. They are not viable as *external* benchmarks anyway since there's no standard
test split for them.

---

## 4. What this means for Stage 2 eval plan

Concrete edits to apply to [`eval_plan.md`](eval_plan.md):

1. **§2 Tier 2 "IEMOCAP / MELD test macro F1"** — remove. Both corpora are in the
   contaminated list (MELD severely, IEMOCAP by Session-5 overlap). Reporting these
   numbers after training on LISTEN_full would be dishonest.
2. **§2 Tier 2 "LISTEN test split"** — keep as primary, but footnote: for MUStARD /
   PODCAST / MOSEI sub-corpora the LISTEN-test audios were seen in LISTEN-train
   under different question variants; report per-sub-corpus accuracy separately and
   flag the three as not strictly held-out audios (different questions, same audio).
3. **§2 Tier 2 "MMAU-speech"** — promote from "assumed independent, verify" to the
   de-facto external emotion/paralinguistic check, and do the overlap verification
   before ship. If MMAU-speech sources IEMOCAP / MELD / PODCAST items, it too
   degrades to "not held-out".
4. **§5.5 optional cross-corpus eval** — delete; question is now answered.
5. **§5.4 VibeCheck1 sourcing check** — apply the same procedure (inspect IDs,
   compare against source-corpus splits) to `VibeCheck1/LISTEN_full`'s VibeCheck1
   portion if VibeCheck1 is a separate composite. (If `VibeCheck1` is just the HF
   org name and `LISTEN_full` is the only dataset under it, this item collapses.)

---

## 5. Filtering applied — filtered LISTEN-train at `/mnt/tmp/listen_analysis/filtered/`

> **Superseded — see §6.** This section describes the first-pass approach of
> shrinking LISTEN-train by dropping contaminated rows. Yield: only ~7.26 h of
> unique audio (unrealistically small). The current plan (§6) instead reuses the
> **full real source corpora** and removes only LISTEN-test IDs from them, giving
> a much larger training pool. This §5 is kept as reference / fallback.

Filter script: [`scripts/emo/filter_listen_leakage.py`](../../scripts/emo/filter_listen_leakage.py).
Drop rules applied to LISTEN-train (union; LISTEN-test left untouched):

| Rule                                                        | Rows dropped |
|-------------------------------------------------------------|-------------:|
| (a) base-audio overlap with LISTEN-test                     |          592 |
| (b) `MELD_(test\|dev)_*`                                    |        1 366 |
| (b) `MOSEI_(Test\|Val)_*`                                   |          151 |
| (b) `IEMOCAP_Session5_*`                                    |          592 |
| **Union** (one row may match multiple rules)                |    **2 616** |

Per-source row count after filtering (rows, not unique audios):

| Source         | Before | After | Kept |
|----------------|------:|------:|-----:|
| CREMA-D        |   800 |   800 | 100.0 % |
| Emotion-Speech |   800 |   800 | 100.0 % |
| RAVDESS        |   798 |   798 | 100.0 % |
| SAVEE          |   230 |   230 | 100.0 % |
| TESS           |   800 |   800 | 100.0 % |
| OMG            | 1 600 | 1 473 |  92.1 % |
| MUStARD        |   768 |   650 |  84.6 % |
| PODCAST        | 1 280 | 1 077 |  84.1 % |
| IEMOCAP        | 2 575 | 1 956 |  76.0 % |
| MOSEI          |   256 |    87 |  34.0 % |
| MELD           | 1 600 |   220 |  13.8 % |
| **Total rows** | **11 507** | **8 891** | **77.3 %** |

Outputs (`/mnt/tmp/listen_analysis/filtered/`):
- `train-{00000,00001,00002}-of-00003.parquet` — 8 891 rows (3 balanced shards).
- `test-00000-of-00001.parquet` — unchanged copy of LISTEN-test (2 635 rows).
- `filter_report.json` — structured version of the tables above.

### 5.1 Audio duration (unique audios; question-variant dedup)

Measured via soundfile WAV-header read (script:
[`scripts/emo/measure_listen_duration.py`](../../scripts/emo/measure_listen_duration.py)):

| Split                   | Unique audios | Total duration |
|-------------------------|--------------:|---------------:|
| **filtered train**      |         7 598 |      **7.26 h** |
| filtered test           |         2 546 |       2.68 h   |
| original train (ref.)   |         9 918 |       9.60 h   |
| original test (ref.)    |         2 546 |       2.68 h   |

Filtering removes **2.34 h** (24 %) of training audio — the drop concentrates on
MELD (1.30 h → 0.18 h; most of MELD's hours were `test`/`dev` items) and MOSEI
(0.28 h → 0.09 h). IEMOCAP drops from 2.19 h → 1.74 h (Session 5 excised).

Filtered-train hours by source (descending):

| Source         | Filtered train audios | Filtered train hours |
|----------------|----------------------:|---------------------:|
| IEMOCAP        |                 1 628 |              1.74 h  |
| PODCAST        |                   665 |              1.08 h  |
| OMG            |                 1 212 |              0.88 h  |
| RAVDESS        |                   798 |              0.83 h  |
| Emotion-Speech |                   800 |              0.61 h  |
| CREMA-D        |                   800 |              0.57 h  |
| MUStARD        |                   406 |              0.57 h  |
| TESS           |                   800 |              0.46 h  |
| SAVEE          |                   230 |              0.25 h  |
| MELD           |                   204 |              0.18 h  |
| MOSEI          |                    55 |              0.09 h  |

(Row-level total is 8 891 vs. unique-audio total 7 598 — each audio still has
~1.2 question-variant rows on average post-filter, down from ~1.16 pre-filter.
Rule (a) removed one of the reasons for inflation: any audio that also existed
in LISTEN-test was dropped entirely from train.)

### 5.2 Implication for S2 mix sizing

Stage 2 plan (`design.md §6.1`) had emotion at ~70 % of the mix. With
filtered LISTEN_full = **7.26 h** (down from 9.60 h), the absolute emotion-hour
budget shrinks proportionally; VibeCheck1's share and the 10 % text-reasoning
weighting should be recomputed once VibeCheck1's own post-filter hours are known.

---

## 6. Current plan — full source corpora minus LISTEN-test

The §5 path (filter LISTEN-train down) yields only ~7 h of audio, which is
insufficient for S2. Replacement strategy: **use the full real source corpora
as the training pool; remove only the LISTEN-test IDs from each corpus to
preserve LISTEN-test as the primary held-out eval.** This gives a far larger
training budget (rough estimate 125 h from open-access corpora alone, 300+ h
once the EULA three arrive).

### 6.1 Source-corpus access table

All scripts live under [`scripts/emo/`](../../scripts/emo/).

| # | Dataset       | License               | Approx size     | Source                                                        | Status (2026-04-24)                                         |
|---|---------------|-----------------------|-----------------|---------------------------------------------------------------|-------------------------------------------------------------|
| 1 | IEMOCAP       | ⚠ USC EULA           | ~12 h / 10 k    | sail.usc.edu/iemocap/                                         | ⏸ pending — user applying                                   |
| 2 | MELD          | ✅ research           | ~13 h / 13 k    | `web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz`      | ✅ 13 847 WAVs extracted (train 9 988 + dev 1 112 + test 2 747, 16 kHz mono) |
| 3 | CREMA-D       | ✅ ODbL               | ~5 h  /  7.4 k  | github.com/CheyneyComputerScience/CREMA-D  (git-lfs)          | ❌ **LFS budget exceeded** on source repo; 7 HF mirror candidates all 404 |
| 4 | RAVDESS       | ✅ CC-BY-NC           | ~1.5 h / 2.5 k  | Zenodo 1188976 (Audio_Speech_Actors_01-24.zip)                | ✅ done — 1 440 WAVs / 765 MB                              |
| 5 | TESS          | ✅ CC-BY              | ~1.6 h / 2.8 k  | UoT TSpace (handle 1807/24487)                                | ❌ no zip endpoint — skipped                               |
| 6 | SAVEE         | ⚠ reg page dead      | ~0.5 h / 480    | (kahlan.eps.surrey.ac.uk invalid per user)                    | ❌ skipped — low yield                                     |
| 7 | **ESD**       | ⚠ license email required | ~29 h / 35 k | zhoukun@u.nus.edu + GDrive 1scuFwqh8…                         | ⏸ pending — user applying                                  |
| 8 | MSP-Podcast   | ⚠ UTD EULA           | 200 h +         | msp.utdallas.edu                                              | ⏸ pending — user applying                                  |
| 9 | OMG-Emotion   | ✅ YouTube            | ~10 h / 7 k     | github.com/knowledgetechnologyuhh/OMGEmotionChallenge         | ⏳ deferred — yt-dlp pipeline                              |
|10 | CMU-MOSEI     | ✅ research           | ~65 h / 23 k    | multicomp.cs.cmu.edu + CMU-MultimodalSDK                      | ⏳ deferred — SDK pipeline                                 |
|11 | MUStARD++     | ✅ research           | ~1 h / 1.2 k    | github.com/cfiltnlp/MUStARD_Plus_Plus (+ GDrive videos)       | ⚠ metadata ✅; videos 65 / 1 202 (GDrive rate-limit; retry 24 h) |
|12 | **DailyTalk** | ✅ CC-BY-NC-SA 4.0    | ~22 h / 23.8 k  | GDrive `1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL` (upstream `keonlee9420/DailyTalk`) | ✅ done — 23 773 WAVs / 12 GB     |
|13 | **EmoV-DB**   | ✅ CC BY 4.0          | ~5 h / 3.8 k    | OpenSLR 115 (per-speaker-emotion tarballs)                    | ✅ done — 3 803 WAVs / 4.2 GB                              |

Rows 12–13 were added 2026-04-24 after §6.1 expansion to English audio-text emo
corpora outside the original LISTEN source list (user request).

Base download path: `/mnt/tmp/datasets/emotion_raw/`. Script pipeline
(all under [`scripts/emo/`](../../scripts/emo/)):

1. [`download_source_corpora.sh`](../../scripts/emo/download_source_corpora.sh) — phase 1 (plain wget/git; LISTEN source set).
2. [`download_source_corpora_phase2.sh`](../../scripts/emo/download_source_corpora_phase2.sh) — phase 2 (after `mamba install -n base -c conda-forge git-lfs gdown ffmpeg`).
3. [`download_source_corpora_phase3.sh`](../../scripts/emo/download_source_corpora_phase3.sh) — phase 3 (git-lfs pull + gdown resume).
4. [`download_crema_d_hf.py`](../../scripts/emo/download_crema_d_hf.py) — CREMA-D HF mirror probe (all candidates 404).
5. [`extract_meld_audio.sh`](../../scripts/emo/extract_meld_audio.sh) — ffmpeg mp4 → 16 kHz mono WAV (parallel, JOBS=16).
6. [`download_english_emo_open.sh`](../../scripts/emo/download_english_emo_open.sh) — phase 1 English audio-text emo (DailyTalk + EmoV-DB repo inspect).
7. [`download_english_emo_phase2.sh`](../../scripts/emo/download_english_emo_phase2.sh) — phase 2 English emo (OpenSLR 115 + gdown for DailyTalk).
8. [`match_listen_test_to_sources.py`](../../scripts/emo/match_listen_test_to_sources.py) — LISTEN-test ID → raw-corpus file mapping; emits
    `/mnt/tmp/listen_analysis/exclude_manifests/listen_test_source_mapping.json`.
9. [`hash_match_ravdess.py`](../../scripts/emo/hash_match_ravdess.py) — content-hash match for RAVDESS sequential-index IDs.
10. [`build_training_manifest.py`](../../scripts/emo/build_training_manifest.py) — enumerate raw files, drop LISTEN-test matches, emit `train_manifest.jsonl`.

**Quirks encountered** (documented so future runs don't re-hit them):
- `MELD.Raw.tar.gz` unpacks to non-uniform split dir names:
  `train_splits`, `dev_splits_complete`, `output_repeated_splits_test`.
- MELD test has 4 807 raw MP4s but only 2 747 extract cleanly to WAV —
  remainder hit ffmpeg errors (likely no-audio-track or corruption in the
  "output_repeated_splits_test" variant shipped with MELD.Raw).
- CREMA-D GitHub LFS quota is exhausted repo-wide — any `git lfs pull` returns
  "This repository exceeded its LFS budget". HF mirror candidates probed
  (AbstractTNT/CREMA-D, silpakanneganti/cremad, confit/crema-d, mteb/crema_d,
  ajyy/CREMA_D, Ar4ikov/…) all 404. Recovery options: (a) Kaggle
  `ejlok1/cremad` via `kaggle` CLI + API token, (b) email maintainers for LFS
  reset, (c) fall back to LISTEN-train's CREMA-D slice (800 audios / 0.57 h —
  ~10 % of full corpus).
- MUStARD++ GDrive folder rate-limits after ~65 files; resume script is ready
  but needs to wait ~24 h or switch gdown account.
- ESD reclassified to EULA-pending — upstream README explicitly requires signed
  license emailed to author before using the public GDrive.

### 6.1.1 LISTEN-test mapping results (as of 2026-04-23 13:10)

Output at `/mnt/tmp/listen_analysis/exclude_manifests/listen_test_source_mapping.json`:

| Source         | LISTEN-test IDs | Mapped to raw file | Unmapped | Reason for unmapped                          |
|----------------|----------------:|-------------------:|---------:|----------------------------------------------|
| MELD           |             388 |                 57 |      331 | raw MELD-test extracted variant is partial   |
| RAVDESS        |             200 |                  0 |      200 | LISTEN uses sequential idx — needs hash pass |
| CREMA-D        |             200 |                  0 |      200 | raw corpus not yet on disk                   |
| TESS           |             200 |                  0 |      200 | raw corpus skipped                           |
| Emotion-Speech |             200 |                  0 |      200 | ESD EULA pending                             |
| IEMOCAP        |             392 |                  0 |      392 | IEMOCAP EULA pending                         |
| PODCAST        |             294 |                  0 |      294 | MSP-Podcast EULA pending                     |
| OMG            |             382 |                  0 |      382 | OMG pipeline deferred                        |
| MOSEI          |              59 |                  0 |       59 | MOSEI pipeline deferred                      |
| MUStARD        |             173 |                  0 |      173 | videos partial (65 / 1 202)                  |
| SAVEE          |              58 |                  0 |       58 | skipped                                      |
| **Total**      |        **2 546** |            **57** | **2 489** |                                              |

The mapping pipeline is complete and idempotent; reruns as corpora arrive will
progressively fill in the `mapped` column. Mapping rules per source are in
[`match_listen_test_to_sources.py`](../../scripts/emo/match_listen_test_to_sources.py) docstring.

### 6.1.2 Exclusion pipeline

After mapping, [`build_training_manifest.py`](../../scripts/emo/build_training_manifest.py)
enumerates raw files per corpus, removes paths that appear in the exclusion
mapping, and emits `/mnt/tmp/listen_analysis/train_manifest/train_manifest.jsonl`
plus a `filter_report.json`. Path-format translation is handled inside (MELD
`.mp4` → extracted `.wav` under `MELD/audio/{split}/`).

For corpora whose LISTEN id is a sequential index
([`hash_match_ravdess.py`](../../scripts/emo/hash_match_ravdess.py) for RAVDESS),
the exclusion pass hashes audio content and matches against the LISTEN-test
audio bytes. Current RAVDESS result: **0 / 200 matched** — LISTEN appears to
have resampled or re-encoded the audio so a naive byte hash doesn't collide.
A resample-normalized variant (decode to 16 kHz mono, hash first-N-samples)
is the next step if full exclusion of the 200 RAVDESS LISTEN-test audios from
the 1 440-file raw corpus is required.

### 6.1.3 Current training manifest

Run 2026-04-24 — corpora on disk so far:

| Source     | Raw files | LISTEN-test excluded | Kept    | Hours    |
|------------|----------:|---------------------:|--------:|---------:|
| DailyTalk  |    23 773 |                    0 |  23 773 |  21.67 h |
| MELD       |    13 847 |                   57 |  13 790 |  12.15 h |
| EmoV-DB    |     3 803 |                    0 |   3 803 |   5.43 h |
| RAVDESS    |     1 440 |           0 (⚠ hash) |   1 440 |   1.48 h |
| MUStARD    |        65 |                    0 |      65 |      —¹ |
| **Total**  |**42 928** |                **57** |**42 871**| **40.74 h** |

¹ MUStARD entries are `.mp4` (audio not yet extracted); omitted from the hour
total until an ffmpeg pass runs over the 65 partial-download videos.

Manifest row format:
```json
{"path": "/mnt/tmp/datasets/emotion_raw/MELD/audio/train/dia0_utt0.wav",
 "source": "MELD",
 "relpath": "MELD/audio/train/dia0_utt0.wav"}
```

Current pool is **~5.6 × the §5 filtered LISTEN-train yield (7.26 h)** with
only five sources loaded. Projected terminal state (after IEMOCAP + ESD +
MSP-Podcast EULAs arrive) is **250 + h** of training audio. DailyTalk / EmoV-DB
additions (2026-04-24) are outside the original LISTEN source list — no
LISTEN-test exclusion applies — so all 27.6 k WAVs carry over 1 : 1.

### 6.1.4 Quirks & gotchas (running log)

- RAVDESS exclusion: LISTEN re-encoded the audio so byte-hash against raw
  Zenodo WAVs matches 0 / 200. A resample-normalized fingerprint (decode to
  16 kHz mono, hash first 4 k samples) is the intended follow-up.
- DailyTalk GDrive folder ships one monolithic `dailytalk.zip` (~5 GB)
  which gdown drops at top level — [`download_english_emo_phase2.sh`](../../scripts/emo/download_english_emo_phase2.sh)
  now unzips it post-download. Produced 23 773 WAVs in nested `data/{dialog}/…`.
- EmoV-DB repo README points at a dead Mega.nz link; actual source is
  OpenSLR 115 (per-speaker-emotion tarballs, 4 speakers × 4–5 emotions each).
  Josh speaker has only 3 emotions (no Angry / Disgusted).

---

## 7. Text-only MCQA benchmarks (Tier 4 guardrail + text-reasoning SFT)

Tier 4 in [`eval_plan.md`](eval_plan.md) calls for a text-retention
guardrail to catch LoRA damaging the LLM's language ability. Because the §3.3
text-reasoning SFT mix ("Open-Orca" etc.) overlaps some of these benchmarks at
the item level, both the **train** and **test/validation** splits of each were
pulled so we can (a) use the train splits as SFT signal if needed, and (b) hold
out the canonical val/test splits for Tier-4 eval with known leakage bounds.

Downloader: [`scripts/emo/download_text_benchmarks.py`](../../scripts/emo/download_text_benchmarks.py).
Output path: `/mnt/tmp/datasets/text_benchmarks/<name>/<split>.parquet`.

| Benchmark       | HF source                         | Splits pulled (rows)                              | Disk   |
|-----------------|-----------------------------------|--------------------------------------------------:|-------:|
| ARC-Easy        | `allenai/ai2_arc` / `ARC-Easy`    | train 2 251 · validation 570 · test 2 376         | 1.1 MB |
| ARC-Challenge   | `allenai/ai2_arc` / `ARC-Challenge` | train 1 119 · validation 299 · test 1 172       | 620 KB |
| WinoGrande      | `allenai/winogrande` / `winogrande_xl` | train 40 398 · validation 1 267 · test 1 767  | 2.0 MB |
| HellaSwag       | `Rowan/hellaswag`                 | train 39 905 · validation 10 042 · test 10 003    | 48 MB  |
| BoolQ           | `google/boolq`                    | train 9 427 · validation 3 270 (no labeled test)  | 4.7 MB |
| COPA            | `aps/super_glue` / `copa`         | train 400 · validation 100 · test 500             | 88 KB  |
| ~~PiQA~~        | ~~`ybisk/piqa`~~ (dropped — legacy loader deprecated, all HF mirror candidates 404) | — | — |

**Eval usage (Tier 4)**: use each benchmark's `validation` split (or `test`
where labels are public) with letter-prefix prompting — same CoT-MCQA protocol
as Tier 2 LISTEN-test. Report accuracy pre- and post-S2 to quantify LoRA
damage. Acceptance bar per §4: drop ≤ 5 points on any single benchmark.

**Leakage note for SFT**: if the chosen text-reasoning SFT corpus (Open-Orca /
OpenHermes / Magpie / …) contains items from any of these six benchmarks'
train **or** test splits, the corresponding Tier-4 number is contaminated and
must be flagged. Add to the §5 / VibeCheck1 audit to-do list: run a hash /
prompt-dedup pass between chosen SFT corpus and the benchmarks above before
committing to a Tier 4 panel.

---

## 8. Environmental sound datasets (Stage 2 Tier 3 sound eval)

Tier 3 in [`eval_plan.md`](eval_plan.md) §3.2 calls for sound
captioning + classification corpora. Fully open-access, non-YouTube-dependent
subset fetched (EULA / YouTube-gated sources — AudioSet, VGGSound,
UrbanSound8K, TAU Urban — deferred).

Downloader: [`scripts/env_sound/download.sh`](../../scripts/env_sound/download.sh).
Base path: `/mnt/tmp/datasets/env_sound/`.

| Dataset | License | Size | Downloaded | Status (2026-04-24) |
|---|---|---|---|---|
| **ESC-50** | ✅ CC BY-NC | 2 000 clips / 50 classes / 5 s | ✅ 2 000 WAVs / 1.4 GB | complete (GitHub clone) |
| **Clotho v2.1** | ✅ CC BY (Tampere U. 2021) | dev 3 839 + eval 1 045 + val 1 045 clips + captions | ✅ archives 6.6 GB; ⏳ 7z extraction in progress (~3 300 / 6 000 WAVs) | Zenodo 4783391 |
| **FSD50K** | ✅ CC BY 4.0 | 51 197 clips / 200 labels | ✅ archives 23 GB; ⏳ split-zip join + extract | Zenodo 4060432 (dev `.z01-.z05 + .zip`, eval `.z01 + .zip`, ground_truth/metadata/doc) |
| **MACS** | ✅ CC BY 4.0 (captions) | 3 931 clips | ⚠ captions-only yaml (2.7 MB); **audio not distributed** on MACS Zenodo — comes from TAU Urban Acoustic Scenes 2019 separately | Zenodo 5114771 |

Script: [`scripts/env_sound/extract.sh`](../../scripts/env_sound/extract.sh) runs
`7z x` on Clotho archives and `zip -s 0 --out …` + `unzip` on FSD50K split zips.

**Skipped**: AudioSet (YouTube), VGGSound (YouTube), UrbanSound8K (registration),
TAU Urban Acoustic Scenes (registration).

---

## 9. Sample-count ratio — emotion / text / env-sound / ASR pools

**Available pool sizes** (what's on disk; sets the ceiling for any subsample):

| Pool                    | Source                                                                                     | Rows            |
|-------------------------|--------------------------------------------------------------------------------------------|----------------:|
| Emotion (kept)          | DailyTalk 22 573 + MELD 13 790 + EmoV-DB 5 103 + RAVDESS 1 199 + MUStARD 65                 | **42 730**      |
| Text (train split)      | HellaSwag 39 905 + WinoGrande 40 398 + BoolQ 9 427 + ARC-E 2 251 + ARC-C 1 119 + COPA 400   | **93 500**      |
| Env sound (train split) | FSD50K dev 40 966 + Clotho dev 3 839 + ESC-50 2 000                                         | **46 805**      |
| ASR (superset)          | 5-corpus shuffled-128 (libri-tts + mls_en + voxpopuli + gigaspeech + common_voice_en)       | ~20.7 M         |

**Actual Stage 2 mix (user-set, 2026-04-24 evening; after halving ASR)**:

| Pool      | Rows  | Share |
|-----------|------:|------:|
| ASR       | 17 150 | 14.5 % |
| Emotion   | 39 919 | 33.7 % |
| Env sound | 41 000 | 34.6 % |
| Text      | 20 519 | 17.2 % |
| **Total** | **118 588** | 100 % |

Ratio **ASR : EMO : ENV : TXT = 0.415 : 1 : 1 : 0.5** (Emo = 1 baseline).
Implementation in [`design.md §6.1`](design.md#61-training-mix-confirmed-plan-2026-04-24).
§9.1 strategies below predate this mix — kept for reference.

### 9.1 What sampling knobs hit the plan

`design.md §6.1` plan doesn't allocate a share to env-sound in S2 — it
is a Tier-3 eval-only capability today. Three sampling postures with env-sound
added:

| Strategy                                       | Emotion | Text   | Env sound | Mix total |
|------------------------------------------------|--------:|-------:|----------:|----------:|
| **A. S2 plan-literal** (emo 70 / text 10 / env 0) — env sound held out for eval only |  42 871 |  6 124 |         0 |    48 995 |
| **B. Plan + env 10 %** (emo 70 / text 10 / env 10) — parity with text       |  42 871 |  6 124 |    6 124  |    55 119 |
| **C. 1 : 1 : 1 balance** (loose prior, each pool capped at emo size)       |  42 871 | 42 871 |   42 871  |   128 613 |
| **D. Use everything**                                                      |  42 871 | 93 500 |   46 805  |   183 176 |

Interim (pre-EULA) recommendation:
- **Emotion**: keep full pool (42 871) — still the smallest once text/env are
  subsampled.
- **Text**: subsample to ~6 100 rows per §9.1 (HellaSwag / WinoGrande /
  ARC-easy / ARC-challenge / BoolQ / COPA blend, see original §9.1 budget).
- **Env sound**: default to **strategy A** (held-out, 0 rows) unless you
  explicitly want Stage-2 to pick up sound-captioning capability. If yes →
  **strategy B** with 6 100 env-sound rows; favor Clotho dev + ESC-50 (both
  captionable) and treat FSD50K as class-labels source.

Once IEMOCAP + ESD + MSP-Podcast arrive (≈ +240 k emotion rows), the 7 : 1
plan becomes emotion-abundant:

| Post-EULA Strategy (emotion ≈ 240 k)        | Emotion | Text   | Env sound | Total    |
|---------------------------------------------|--------:|-------:|----------:|---------:|
| Plan-literal (emo 70 / text 10 / env 0)     | 240 000 | 34 286 |         0 | 274 286  |
| Plan + env 10 %                             | 240 000 | 34 286 |    34 286 | 308 572  |
| Use everything                              | 240 000 | 93 500 |    46 805 | 380 305  |

### 9.2 Hours comparison — emotion vs. env sound (audio pools)

Row count is misleading because env-sound clips are on average **2.5 – 3× longer**
than emotion utterances. Measured + published per-split hours:

| Pool      | Corpus                    | Rows   | Hours    |
|-----------|---------------------------|-------:|---------:|
| Emotion   | DailyTalk (full)          | 23 773 |  21.67 h |
|           | MELD (train+dev+test)     | 13 790 |  12.15 h |
|           | EmoV-DB (full)            |  3 803 |   5.43 h |
|           | RAVDESS (full)            |  1 440 |   1.48 h |
|           | MUStARD (65 mp4)          |     65 |      —   |
|           | **Emotion total**         | **42 871** | **40.74 h** |
| Env sound | FSD50K dev¹               | 40 966 |  81.50 h |
|           | Clotho development        |  3 839 |  23.99 h |
|           | ESC-50 (5-fold CV, full)  |  2 000 |   2.78 h |
|           | **Env sound total**       | **46 805** | **~108.3 h** |
| Env sound | *(eval-side, held-out)*   |        |          |
|           | Clotho evaluation         |  1 045 |   6.50 h |
|           | Clotho validation         |  1 045 |   6.57 h |
|           | FSD50K eval¹              | 10 231 |  27.90 h |

¹ FSD50K hour figures are from the FSD50K paper (dev 81.5 h / eval 27.9 h);
on-disk extraction is in progress as of this update.

**Hours ratio: emotion : env-sound ≈ 40.74 h : 108.3 h ≈ 1 : 2.66.**
Despite env-sound row count (46 805) being only ~1.1× emotion's (42 871), its
hour budget is **2.66×** — average clip ~8.3 s vs. emotion's ~3.4 s, with
FSD50K variable-length clips pulling the mean up.

### 9.3 Target-token budget — measured loss-signal share per modality (2026-04-24)

Row shares are 25 : 30 : 30 : 15 by design, but the gradient each row delivers
to the LoRA adapter scales with **assistant-target token count**, not row count.
Emotion and text rows whose target is just a single letter carry ~1 token of
supervision; ASR rows carry a transcript (~30 tokens on our subsample). Measured
by running the tokenizer over 500 rows per modality from
`stage2_combined_manifest.jsonl` and multiplying by the full row count:

(Updated 2026-04-24 evening after ASR share halved 25 % → 14.5 %.)

| Modality       | Mean target tokens | Median | p95 | Rows    | Total target tokens | Share |
|----------------|-------------------:|-------:|----:|--------:|--------------------:|------:|
| ASR            |            **29.2** |     31 |  59 |  17 150 |           500 780   | **52.0 %** |
| Env sound      |             **9.8** |     10 |  18 |  41 000 |           401 800   | **41.7 %** |
| Emotion¹       |             **1.0** |      1 |   1 |  39 919 |            39 919   | **4.1 %**  |
| Text           |             **1.0** |      1 |   1 |  20 519 |            20 519   | **2.1 %**  |

For reference, pre-halving (ASR 34 300 rows): shares were 68.4 / 27.4 / 2.7 / 1.4 %.
Halving ASR shifted 16 pp of the budget mostly to env-sound (+14) rather than
emotion (+1.4), because env-sound-classify mean-tokens (9.8) is already larger
than the letter-only emotion/text targets. The main lever for emotion signal
remains rationale synthesis — with rationale at ~40 tokens/row, emotion totals
~1.6 M tokens and climbs to ~60 % of the budget regardless of ASR share.

¹ Emotion rows currently carry `rationale: null` — target is just the letter.
With rationale synthesised (~40 tokens/row), emotion total would jump to ~1.6 M
tokens (~45 % share), putting it on par with ASR.

Reproduce via the snippet in [`scripts/emo/dryrun_processor.py`](../../scripts/emo/dryrun_processor.py)
(or the inline measurement that produced the table above — it's a standalone
tokenizer-and-count pass over the combined manifest; not committed as a named
script yet).

**Interpretation**:
- ASR already dominates (~68 %). Its transcripts are short in this corpus (LibriSpeech
  / MLS tail), so the imbalance is less severe than a worst-case estimate — but
  emotion + text combined still sit at 4 % of gradient signal despite 45 % of row share.
- Env-sound captioning (~10 tokens) is a meaningful contributor even without rationale.
- **The rationale synthesis pass is effectively the loss-balance lever**: running
  [`synthesize_rationales.py`](../../scripts/emo/synthesize_rationales.py) shifts emotion
  from 2.7 % → ~45 % of target tokens, a ~16× rebalancing with no training-loop changes.

Three mitigation options (pick one before real-scale launch):

1. **Rationale synthesis** — fills `rationale` field; no code changes beyond
   the already-written script.
2. **Lengthen letter-only targets to `"Answer: A. {label_word}"`** — ~5 tokens
   each. Raises emotion/text signal ~5×; cheaper than LLM rationale, less
   pedagogically grounded, but a good interim.
3. **Per-modality loss weighting** in the training loop — weight emotion ×20,
   text ×30. Requires a modality tag on the collated batch and a custom loss;
   doesn't need any data-prep changes.

**User decision (2026-04-24)**: accept the imbalance — proceed with letter-only
emotion/text targets and the ~4 %/~2 % loss shares. Emotion capability will
be driven mostly by the audio encoder + projector adaptation to MCQA-style
inputs rather than by rich language targets. If post-run LISTEN-test / MELD
emotion accuracy disappoints, revisit option 1 (rationale) as the first
mitigation.

**Separate implementation decision — per-task loss visibility**: the total loss
alone can't tell us *which* modality is driving the gradient or converging.
Trainer was modified to emit per-modality loss to wandb even though the
imbalance itself is accepted. Since `enable_liger_kernel: true` in the real
run (fused CE leaves `outputs.logits=None`), a forward hook on the inner
`Qwen3_5AEModel` captures `last_hidden_state`, and `OmniTrainer.compute_loss`
projects it through `lm_head` in `no_grad` to recover logits just long enough
to compute per-modality CE on the shifted labels. Output keys:
`loss/audio_asr`, `loss/audio_emotion`, `loss/audio_env_sound`, `loss/text` +
`tokens/<name>` (target-token count per modality for sanity check). Memory
overhead of the transient logits (~5 GB per GPU at batch 3 × seq 3584 × V 248 k
× bf16) is freed before backward. See
[`OmniTrainer._ensure_hidden_hook`](../../src/llamafactory/train/omni/trainer.py)
and smoke verification in
[`smoke.md §B`](smoke.md#b-mini-smoke-30-step-per-task-loss-logging-via-option-c).

For **smoke runs** (200 steps) this imbalance doesn't matter — smoke validates
mechanical plumbing only.

### 9.4 Hours-vs-rows caveat (overall training budget)

Row-count ratio ≠ hours ratio. Text rows take the LLM O(100) forward-pass
tokens; emotion / env-sound rows include seconds of audio each (env-sound
clips average 5 – 30 s, emotion utterances 2 – 5 s). If the optimizer budget is
measured in step × batch × seq-len rather than raw rows, audio pools will
dominate GPU-time well before the row ratio reaches 1 : 1 — the practical
knob is **effective-step share**, not sample share. Recompute once we have
per-row token lengths for the final text-reasoning SFT choice (§3.3 decision)
and per-corpus mean audio duration.

Full three-way budget (audio hours + text rows):
- **Emotion** 40.74 h / 42 871 rows
- **Env sound** ~108.3 h / 46 805 rows
- **Text** 0 h audio / 93 500 rows (tokens only)

If `omni_max_audio_samples: 1 600 000` (33.3 s cap from
`design.md §6.2`) is applied, both audio pools will see some clips
truncated — FSD50K has clips up to 30 s (just under cap), Clotho up to 30 s,
DailyTalk / MELD / etc. all well under. Cap effect on hours is negligible.

---

## 10. Reproducing this audit

Script fragments (ad-hoc, not committed) that produced the numbers above:

```python
import pandas as pd, json, re, collections
from pathlib import Path

root = Path('/mnt/tmp/listen_analysis/data')
train = pd.concat([pd.read_parquet(p, columns=['id','dataset_source'])
                   for p in sorted(root.glob('train-*.parquet'))], ignore_index=True)
test  = pd.read_parquet(root/'test-00000-of-00001.parquet',
                        columns=['id','dataset_source'])

strip = lambda s: re.sub(r'_\d+[A-Z]$', '', s)
for src in sorted(train['dataset_source'].unique()):
    tr_ids = train.loc[train.dataset_source==src, 'id'].tolist()
    te_ids = test .loc[test .dataset_source==src, 'id'].tolist()
    tr_b = {strip(i) for i in tr_ids}
    te_b = {strip(i) for i in te_ids}
    print(f'{src:15} rows_tr={len(tr_ids):5} base_tr={len(tr_b):5}'
          f' rows_te={len(te_ids):4} base_te={len(te_b):4}'
          f' overlap={len(tr_b & te_b):4}')

# MELD provenance from ID split-prefix:
meld_train = train.loc[train.dataset_source=='MELD', 'id']
prov = collections.Counter(re.match(r'MELD_(train|dev|test)_', i).group(1)
                           for i in {strip(i) for i in meld_train})
print('MELD LISTEN-train unique-audio split provenance:', dict(prov))
```

Open follow-ups left for someone with source-corpus split metadata on hand:
- MSP-Podcast Test1/Test2 cross-ref against PODCAST IDs in LISTEN-train (§3.3).
- OMG-Emotion train/val/test cross-ref (§3.5).
- Speaker-level disjointness check for CREMA-D / RAVDESS / TESS / SAVEE / ESD
  (§3.7) — currently only audio-level disjointness verified.
- RAVDESS / SAVEE sequential-index → source-filename mapping table (§6.2):
  dump LISTEN audio bytes and content-hash match against the raw corpus.
