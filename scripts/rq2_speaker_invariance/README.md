# RQ2 / Analysis 3: Speaker invariance across encoder, projector, LLM

How does an ALM (audio-LM) reshape speech as it flows through (1) audio encoder,
(2) projector, (3) LLM layers? We probe this by pairing utterances that share
the *same transcript* but come from *different speakers*, then comparing the
representation per layer with three orthogonal tools:

- **distance**: matched-control cosine ratio (paper-aligned with §4.3)
- **probe**: cross-validated linear-classifier accuracy for speaker-ID and
  content-ID at each layer
- **cka**: linear CKA between any two (model, layer) pairs — used to quantify
  what Stage1/Stage2 training does to the encoder representation

Two paired corpora:
- **IEMOCAP scripted** — `script01..script03` performed by 10 actors across
  5 sessions. Pair by `(script_id, line_idx)`. By default, cross-gender takes
  are dropped (same actress in F-led and M-led recordings would otherwise be
  double-counted).
- **CMU Arctic** — 7 speakers (bdl, clb, slt, rms, ksp, awb, jmk) recording
  the same ~1132 prompts. Cleaner control than IEMOCAP because read TTS data
  has little emotion / prosody variation across speakers.

Models probed along the "encoder" axis (this is the new analysis surface
compared to `/mnt/ddn/users/jos/AudioEnc/log/sec43_embeddings/`):
- raw HF `openai/whisper-large-v2` encoder (32 layers, d=1280)
- raw HF `openai/whisper-tiny.en` encoder (4 layers, d=384)
- SSL acoustic baseline: `facebook/hubert-base-ls960` (12 layers, d=768)
- trained ALM: Qwen3.5AE-whisper-tiny v6 Stage1 checkpoint (encoder + projector
  + LLM, via forward hooks). Pass `--lora <dir>` to merge a Stage2 adapter.

## Layout

```
rq2_speaker_invariance/
├── pairs/
│   ├── build_iemocap_pairs.py     → iemocap_pairs.jsonl
│   │     flags: --no_match_gender, --scripts ..., --min_speakers
│   ├── build_arctic_pairs.py      → arctic_pairs.jsonl
│   ├── download_cmu_arctic.sh
│   └── combine.py                 → all_pairs.jsonl  (--per_corpus 60 default)
├── extract/
│   ├── _io.py
│   ├── extract_whisper.py         raw HF Whisper encoder
│   ├── extract_ssl.py             HuBERT / WavLM
│   └── extract_alm.py             Qwen3.5AE: manual projector + audio_start/end flanking
│                                   hooks on enc / proj / llm internal layers
│                                   --lora <dir> merges Stage2 adapter
├── analyze/
│   ├── distances.py               matched-control cosine + paper ratio
│   ├── probe.py                   sklearn LogReg, 5-fold, speaker + content
│   ├── cka.py                     linear CKA between (model_a, *) × (model_b, *)
│   ├── pca_plot.py
│   └── report.py                  stitches CSVs → report.md
└── run.sh
```

## Output schema (extractor → analyzer)

Each extractor writes one `.npz` per (model, layer-group) with arrays:
- `emb`     `[N, D]` — time-pooled hidden state
- `pair`    `[N]`    — transcript id
- `spk`     `[N]`    — speaker id
- `src`     `[N]`    — `"iemocap"` / `"arctic"`
- `model`   scalar str
- `layer`   scalar str — `"enc.L03"`, `"proj.L02"`, `"llm.L17"`, `"enc.out"`, `"proj.out"`, `"llm.norm"`

Same order across `.npz`s of the *same* model (within a model the row order
matches across layers). The CKA analyzer canonicalizes by `(src, pair, spk)`
so it can align rows across models.

## How to run

```bash
cd /mnt/ddn/users/jos/audiollm-trainer/scripts/rq2_speaker_invariance
bash run.sh           # end-to-end
```

Knobs:
- `PAIRS=<jsonl>` — override pair file (default = combined IEMOCAP + Arctic, 60 groups each)
- `EMB_DIR=<dir>` — where extractor `.npz` files go
- `ALM_CKPT=<dir>` — Stage1 base
- `ALM_LORA=<dir>` — optional Stage2 adapter; merged then unloaded
- `SKIP_EXTRACT=1` — re-run analyzers only

## Distance metric convention

Three population means per (model, layer, corpus):
```
same_t_diff_s : same transcript, different speaker     (cross-speaker, content-controlled)
same_s_diff_t : same speaker,    different transcript  (within-speaker, content-varying)
diff_s_diff_t : different speaker, different transcript (random baseline)
```

Two ratios reported:
- `ratio_paper = same_t_diff_s / same_s_diff_t`. `>1` = speaker dominates.
  This matches `/mnt/ddn/users/jos/AudioEnc/log/sec43_embeddings/` so figures
  from this folder are directly comparable.
- `ratio_inv   = same_t_diff_s / diff_s_diff_t`. `<1` = same-transcript
  utterances cluster (regardless of speaker).

## Notes / standing constraints

- Pooling: mean over valid (non-pad) frames. For ALM LLM layers, the pool
  window is positions `[1 .. 1+T_audio)` of the `[<|audio_start|>, audio…,
  <|audio_end|>]` sequence.
- Audio truncation: 10 s (matches §4.3).
- All extractors run with `torch.no_grad()` + bf16 autocast on a single GPU.
- ALM extractor forces SDPA for the LM and (optionally) for the internal
  Whisper encoder via `--patch_flash`, because the flash-attn 2 wheel on this
  box links a missing libc symbol (the §4.3 script had the same patch).
