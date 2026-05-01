# Whisper-small Stage1 Implementation Status

**Last Updated:** 2026-04-25  
**Current Branch:** shC_audiollm-trainer  
**Plan Reference:** docs/stage1/whisper/plan.md

---

## ✅ COMPLETED (14/14 Checklist Items)

### Model & Checkpoint Build
- ✅ **AudioEncoder (whisper version)** — `external/models/Qwen3.5AE-4B-whisper-small/audio_encoder.py`
  - Frozen Whisper-small.en encoder (88.15M params)
  - Trainable AudioProjector (LlamaDecoderLayer-based, 768→512→2560)
  
- ✅ **Config Updates** — `external/models/Qwen3.5AE-4B-whisper-small/configuration_qwen3_5AE.py`
  - AudioConfig with whisper fields: `whisper_model_id`, `whisper_sample_rate`, `whisper_fps`, `audio_hidden_size=768`
  - DAC fields removed

- ✅ **Conversion Script** — `external/models/Qwen3.5AE-4B-whisper-small/convert_to_whisper.py`
  - Loads base Qwen/Qwen3.5-4B
  - Embeds Whisper encoder via `WhisperModel.from_pretrained()`
  - Random-initializes projector
  - Saves complete checkpoint with tokenizer

- ✅ **Checkpoint Built** — `external/models/Qwen3.5AE-4B-whisper-small/`
  - Total disk size: ~8GB (LLM 8GB + Whisper 0.18GB + projector 37MB)
  - State dict split: `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`
  - Verified 2026-04-24 13:45

- ✅ **Sanity Check Script** — `external/models/Qwen3.5AE-4B-whisper-small/sanity_check.py`
  - Verifies text-only logits match base model
  - Confirms Whisper encoder weights properly baked in
  - Checks AudioProjector random initialization
  - **Status:** Ready to run, but not yet executed

### Data Pipeline
- ✅ **Manifest Split Script** — `scripts/split_long_audio_30s.py`
  - Splits audio >30s into ≤30s chunks
  - Handles transcript alignment (force-aligner support planned)
  - Status: Complete, ready to apply to Stage1/Stage2 manifests

- ✅ **Whisper Dataset Processor** — `src/llamafactory/data/omni_dataset_whisper.py`
  - WhisperFeatureExtractor: wav → log-mel (80, 3000)
  - 16kHz audio support + resample from 24kHz (LibriTTS-R)
  - Audio padding token count calculation (ceil(samples/320) @ 50fps)
  - Handles `audio_start`/`audio_end` fields for slicing

- ✅ **Whisper Features Helper** — `src/llamafactory/data/whisper_features.py`
  - `extract_mel()` wrapper for WhisperFeatureExtractor
  - `audio_pad_token_count()` for sequence length calculation

- ✅ **Collator Plugin** — `src/llamafactory/data/mm_plugin.py` (updated)
  - Routes to WhisperOmniCollator when `audio_encoder_type == "whisper"`
  - Mel stacking: (N, 80, 3000) uniform tensor (no padding needed)

### Training Config & Scripts
- ✅ **Stage1 Full Config** — `configs/ASR/stage1_whisper_small.yaml`
  - Model: `external/models/Qwen3.5AE-4B-whisper-small`
  - Dataset: `external/datasets/libri_mls_vox` (46,074 hrs LibriTTS-R + MLS + VoxPopuli)
  - Training: 100k steps, bs=3, lr=2.0e-4, warmup=1000, bf16, DeepSpeed ZeRO-2
  - Max audio: 480000 samples (30s @ 16kHz)

- ✅ **Smoke Test Config** — `configs/ASR/stage1_whisper_small_smoke.yaml`
  - Reduced: 2 steps, max_steps=2 (no DeepSpeed, single GPU)
  - Verifies forward/backward/optimizer without OOM

- ✅ **Launch Script** — `scripts/ASR/run_stage1_whisper_smoke.sh`
  - Environment setup (CUDA, LD_LIBRARY_PATH, miniconda activation)
  - Single GPU smoke run command

---

## 🔴 CRITICAL GAPS — NEXT ACTIONS

### 1. **Sanity Check NOT RUN** (Highest Priority)
**File:** `external/models/Qwen3.5AE-4B-whisper-small/sanity_check.py`

**What to do:**
```bash
cd external/models/Qwen3.5AE-4B-whisper-small

# Without base model comparison (offline):
python sanity_check.py --skip-logits

# Or with base model for full validation:
python sanity_check.py --base-id Qwen/Qwen3.5-4B --skip-logits  # (--skip-logits still safe)
```

**Verification checks:**
- ✓ Whisper encoder weights match HF `openai/whisper-small.en`
- ✓ AudioProjector is random-initialized (std >0.001)
- ✓ Text-only logits match base LLM (if --base-id provided)

**Failure handling:**
- If encoder weights mismatch → convert script has bugs
- If projector not random-init → state_dict loading issue
- If logits diverge → key remapping problem in convert

---

### 2. **Smoke Run NOT TESTED** (Second Priority)
**File:** `scripts/ASR/run_stage1_whisper_smoke.sh`

**What to do:**
```bash
# Single GPU smoke test
CUDA_VISIBLE_DEVICES=0 bash scripts/ASR/run_stage1_whisper_smoke.sh
```

**Expected behavior (2 steps):**
- Step 0: Forward pass, loss computed
- Step 1: Backward pass, projector gradients updated
- No NaN/Inf in loss or gradients
- No OOM

**Metrics to watch in logs:**
```
Step 0/1: loss=?.?? | projector.input_proj.weight grad_norm ≈ 0.01-0.1
Step 1/1: loss=?.?? | converging or stable
```

**Likely issues & mitigations:**
| Issue | Mitigation |
|-------|------------|
| CUDA OOM on smoke | Reduce batch_size 3→2, or enable gradient checkpointing |
| Loss = NaN | Whisper encoder bf16 precision issue; wrap forward in `torch.autocast(..., dtype=float32)` for LN |
| Manifest path wrong | Check `external/datasets/libri_mls_vox/shard_*.jsonl` exists |
| Audio loading error | Verify 16kHz resample in `omni_dataset_whisper.py` works |

---

### 3. **Manifest Split Status** (Third Priority)
**File:** `scripts/split_long_audio_30s.py`  
**Status:** Written, but **NOT APPLIED to Stage1 manifest**

**What to do:**
1. Run on Stage1 manifest to check how many entries >30s exist:
   ```bash
   python scripts/split_long_audio_30s.py \
       --input external/datasets/libri_mls_vox/shard_00000.jsonl \
       --output external/datasets/libri_mls_vox/shard_00000_split.jsonl \
       --max-duration 30.0 --num-workers 16 --verbose
   ```

2. Check statistics:
   - **Expected:** LibriTTS-R + MLS utterances mostly <20s → minimal splitting
   - **If >10% entries split:** May indicate data quality issue

3. **Decision:** Apply split to all shards only if justified by statistics

---

### 4. **BF16 Precision Strategy** (Pending Validation)
**Plan Section:** §11  
**Status:** Documented, implementation location TBD

**Issue:** Whisper originally trained in fp32; bf16 can cause NaN in LayerNorm with large activation variance

**Current code location:** `external/models/Qwen3.5AE-4B-whisper-small/audio_encoder.py:forward()`

**Verify in sanity_check or smoke run:**
```python
# Check if already using autocast for LN fp32:
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = self.encoder(audio_features).last_hidden_state
```

**If smoke run produces NaN → add autocast** in `audio_encoder.py`

---

## 📋 WORK QUEUE (In Priority Order)

1. **[BLOCKER]** Run sanity check → validate checkpoint is correct
2. **[BLOCKER]** Run smoke test (2 steps) → validate training loop
3. [OPTIONAL] Apply manifest split, check statistics
4. [IF NEEDED] Fix BF16 precision if smoke produces NaN
5. [NEXT PHASE] Full Stage1 training (100k steps) on multi-GPU with DeepSpeed

---

## 🔗 File Inventory

### Model & Config
- `external/models/Qwen3.5AE-4B-whisper-small/` ← **Main deliverable** (8GB)
  - `audio_encoder.py` — AudioEncoder + AudioProjector
  - `configuration_qwen3_5AE.py` — Config with whisper fields
  - `convert_to_whisper.py` — Reproducible conversion
  - `sanity_check.py` — ← **RUN THIS FIRST**
  - `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`
  - `config.json`, tokenizer files

### Training Configs
- `configs/ASR/stage1_whisper_small.yaml` — Full run (100k steps)
- `configs/ASR/stage1_whisper_small_smoke.yaml` — Smoke test (2 steps)
- `configs/qwen3_5ae-asr/stage1_projector_whisper_small.yaml` — [MISSING] alternate location, not currently used

### Data & Scripts
- `src/llamafactory/data/omni_dataset_whisper.py` — Mel extraction, 16k resample
- `src/llamafactory/data/whisper_features.py` — Helper functions
- `scripts/split_long_audio_30s.py` — Manifest chunking (not yet applied)
- `scripts/ASR/run_stage1_whisper_smoke.sh` — Launch script
- `external/datasets/libri_mls_vox/` — Stage1 manifest (46,074 hrs)

---

## 🎯 Success Criteria (Per Phase) — **모두 완료 (2026-04-30 retrospective)**

### Phase 1: Validation
- [x] `sanity_check.py` runs → all 3 checks PASS
- [x] `smoke run` 2 steps → no NaN, loss curves stable

### Phase 2: Training
- [x] Stage1 runs 100k steps on 4-8 GPUs (실제로 13k step 진행 후 Stage2 init 으로 사용)
- [x] Loss converges smoothly (no spikes)
- [x] Checkpoint saved every 1000 steps (`save_steps: 1000`)
- [x] Final checkpoint → Stage2 input (`checkpoint-13000` 이 [`stage2_whisper_small.yaml`](../../../configs/qwen3_5ae-asr/stage2_whisper_small.yaml) `model_name_or_path` 로 활용 중)

### Phase 3: Evaluation
- [x] ASR eval on LibriSpeech test-clean (WER benchmark) — Stage2 ckpt-8k WER 2.51 % (best), [`3model_comparison.md §3`](../../stage2/3model_comparison.md)
- [x] Compare vs. DACVAE baseline — DAC-VAE v2 best LS-c 4.73 % vs Whisper-small 2.51 %, W-small 우세
- [x] Proceed to Stage2 (Whisper-small Stage2 학습 + 35-ckpt eval 모두 완료, 2026-04-29)

---

## 📝 Notes for Next Session

1. **Sanity check is the gate:** If it fails, debug the convert script before attempting smoke run
2. **BF16 handling:** Watch for NaN in smoke logs; if seen, apply autocast wrapper in `audio_encoder.py:forward()`
3. **Manifest split:** Low priority for Stage1 (most utterances <20s), but needed for Stage2 (podcasts/lectures have long audio)
4. **Plan still valid:** The `plan.md` checklist in §14 is comprehensive; most items are done

---

## Git History Summary
- **19 commits ahead** of origin/shC_audiollm-trainer
- Recent commits (d60a40d9 ~ 12feb34a): config/tokenizer/smoke yaml
- Model build (5a5851b9): convert script + audio_encoder
- Data pipeline (efae1850, c9f10089, e7d903d5): whisper processor, audio I/O, mel features
- Manifest split (3d376a13): long audio chunking script

**Untracked changes:**
- `scripts/ASR/run_stage1_whisper_smoke.sh` (ready to commit)
- `scripts/ASR/install_env.sh` (modified, review before commit)
