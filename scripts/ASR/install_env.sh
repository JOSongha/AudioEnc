# ── Conda env setup ──────────────────────────────────────────── ~1 min ───────
source /mnt/tmp/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q '^audiollm '; then
    conda create -n audiollm python=3.11 -y
fi
conda activate audiollm

# # ── Package installation ───────────────────────────────────────────────────────
# torch wheel ~2.5 GB; torchaudio/torchvision bundled ──────── ~5-10 min ──────
pip install torch==2.5.1+cu124 torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu124 -q

# deepspeed builds CUDA extensions on first import, not here ── ~2-3 min ──────
pip install deepspeed==0.16.9 liger-kernel wandb requests -q
pip install --upgrade pydantic pydantic-core -q

# ── CUDA 12.4 libraries (torch 2.5.1 requirements) ───────────── ~1-2 min ────
pip install --force-reinstall \
    nvidia-nccl-cu12==2.21.5 \
    nvidia-cublas-cu12==12.4.5.8 \
    nvidia-cuda-nvrtc-cu12==12.4.127 \
    nvidia-cudnn-cu12==9.1.0.70 -q

# causal-conv1d compiles CUDA kernel ──────────────────────────── ~2-5 min ────
pip install causal-conv1d --no-build-isolation -q

# flash-attn wheel ~400 MB; skips build if pre-built wheel exists ~5-15 min ──
pip install flash-attn==2.8.3 --no-build-isolation -q

# editable install of this repo (cwd = audiollm-trainer/) ─────── ~1 min ──────
# Run from repo root; v6 projL audio_encoder.py is self-contained, so the
# separate dacvae package install is no longer needed.
pip install -e . -q

# ── flash-linear-attention (modeling_qwen3_5AE.py imports fla.modules.FusedRMSNormGated)
# Use fla-core 0.4.x which is compatible with torch 2.5.1 (avoids torch>=2.7.0 requirement)
# Downgrade triton back to 3.1.0 to match torch 2.5.1 requirement ─ ~1-2 min ─
pip install --no-deps "flash-linear-attention==0.4.0" "fla-core==0.4.0" -q
pip install "triton==3.1.0" -q  # torch 2.5.1 requires triton==3.1.0

# ── glibc_stub.so (flash_attn GLIBC_2.32 workaround) ─────────── ~5 sec ──────
if [ ! -f "$CONDA_PREFIX/lib/glibc_stub.so" ]; then
    echo 'char __libc_single_threaded = 0;' > /tmp/glibc_stub.c
    gcc -shared -fPIC -o "$CONDA_PREFIX/lib/glibc_stub.so" /tmp/glibc_stub.c
fi

# ── flash_attn binary patch (remove GLIBC_2.32 version requirement) ── ~5 sec ─
# Use `find` instead of `import flash_attn` to locate the .so — the import itself
# fails on GLIBC_2.32-absent systems before the patch is applied.
FLASH_SO=$(find "$CONDA_PREFIX" -name "flash_attn_2_cuda*.so" 2>/dev/null | head -1)
if [ -n "$FLASH_SO" ] && python -c "
import subprocess, sys
out = subprocess.check_output(['objdump', '-p', sys.argv[1]], text=True)
sys.exit(0 if 'GLIBC_2.32' in out else 1)
" "$FLASH_SO" 2>/dev/null; then
    python - "$FLASH_SO" <<'EOF'
import sys, struct

path = sys.argv[1]
data = bytearray(open(path, 'rb').read())

# VERSYM: set __libc_single_threaded's version index to 1 (global/unversioned)
struct.pack_into('<H', data, 900152, 0x0001)
# VERNEED: libc.so.6 vn_cnt 5->4
struct.pack_into('<H', data, 907618, 4)
# VERNEED: GLIBC_2.14 vna_next 16->32 (skip GLIBC_2.32 entry)
struct.pack_into('<I', data, 907644, 32)

open(path, 'wb').write(data)
print(f"[patch] Applied GLIBC_2.32 patch to {path}")
EOF
fi

# ── wandb credential (writes ~/.netrc directly; CLI rejects 86-char keys) ─────
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! WARNING: HARDCODED PERSONAL WANDB API KEY BELOW.                        !!
# !!   - DO NOT `git add` / `git push` this file with the key intact.        !!
# !!   - DO NOT share this script (Slack, email, paste) without scrubbing.   !!
# !!   - Strip the password line back to `<REDACTED>` before any commit.     !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cat > "$HOME/.netrc" <<'EOF'
machine api.wandb.ai
  login user
  password <REDACTED>
EOF
chmod 600 "$HOME/.netrc"

# ── 총 소요 시간: 약 20 분 (실측), 캐시 없으면 최대 40 분 ───────────────────
