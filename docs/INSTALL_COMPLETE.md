# Installation Complete ✅

**Date:** 2026-04-25  
**Environment:** `audiollm` conda environment  
**Status:** Ready for training

---

## Summary

✅ **All packages installed successfully** with fixed version compatibility.

### Fixed Issues

1. **fla-core version conflict** (torch>=2.7.0 requirement)
   - **Solution:** Downgraded from fla-core 0.5.0 → 0.4.0
   - **Why:** fla-core 0.4.0 is compatible with torch 2.5.1 without requiring torch>=2.7.0
   
2. **Triton version mismatch** (fla-core 0.5.0 pulled triton 3.3.0)
   - **Solution:** Downgraded triton from 3.3.0 → 3.1.0
   - **Why:** torch 2.5.1 requires triton==3.1.0
   - **Rationale:** Keep torch 2.5.1 stable; flash-linear-attention 0.4.0 is compatible

### Modified Files

- `scripts/ASR/install_env.sh` — Updated fla-core and triton pinning (lines 30-34)

---

## Verified Packages

```
✓ torch              2.5.1+cu124
✓ torchaudio        2.5.1+cu124
✓ transformers      4.57.1
✓ deepspeed         0.16.9
✓ flash-attn        2.8.3 (with glibc patch)
✓ fla-core          0.4.0 (flash-linear-attention)
✓ triton            3.1.0
✓ FusedRMSNormGated (from fla.modules)
✓ CUDA              12.4 (nccl, cudnn, cublas, nvrtc, cufft, curand, cusolver, cusparse)
```

---

## Next Steps

### 1. Run Sanity Check (Highest Priority)
Verify the Whisper-small checkpoint is correctly built:

```bash
cd external/models/Qwen3.5AE-4B-whisper-small
python sanity_check.py --skip-logits
```

**Expected output:**
```
[PASS] Whisper encoder weights match HF original.
[PASS] Projector is random-initialized (to be trained in Stage1).
RESULT: ALL CHECKS PASSED
```

### 2. Run Smoke Test (Single GPU, 2 steps)
Test forward/backward/optimizer pass without OOM:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/ASR/run_stage1_whisper_smoke.sh
```

**Expected behavior:** Completes 2 steps with stable loss, no NaN/Inf/OOM

### 3. Run Full Stage1 Training (Multi-GPU with DeepSpeed)
Once smoke passes:

```bash
bash scripts/ASR/run_stage1_whisper_small.sh  # or modify yaml and run llamafactory-cli
```

---

## Environment Variables (Already Set)

For any future runs, these are configured in the run scripts:

```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD=$CONDA_PREFIX/lib/glibc_stub.so  # Flash-attn GLIBC workaround
export LD_LIBRARY_PATH=... # CUDA libraries
```

---

## Troubleshooting

### Issue: `ImportError: undefined symbol: __libc_single_threaded` (flash_attn)
**Solution:** Already handled! The `glibc_stub.so` patch is applied and `LD_PRELOAD` is set in run scripts.

### Issue: `fla-core 0.5.0 requires torch>=2.7.0`
**Solution:** Already downgraded to fla-core 0.4.0 in `install_env.sh`.

### Issue: `torch 2.5.1 requires triton==3.1.0`
**Solution:** Already pinned to triton 3.1.0 in `install_env.sh`.

### Issue: CUDA out of memory
If smoke test OOMs:
1. Reduce `per_device_train_batch_size: 3 → 2` in yaml
2. Increase `gradient_accumulation_steps: 1 → 2` to maintain effective batch size
3. Reduce `omni_packing_bucket_size: 128 → 64`

---

## Session Info

```bash
# To activate the environment in future sessions:
source /mnt/tmp/miniconda3/etc/profile.d/conda.sh
conda activate audiollm
```

---

## Commit Status

**Untracked files to consider committing:**
- `scripts/ASR/run_stage1_whisper_smoke.sh` (ready)

**Modified files:**
- `scripts/ASR/install_env.sh` (fixed fla-core/triton versions)

---

## Ready to Proceed

✅ Environment fully functional
✅ All dependencies compatible
✅ Ready for sanity check → smoke test → full training
