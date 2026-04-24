# ── Conda env setup ───────────────────────────────────────────────────────────
source /mnt/tmp/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q '^audiollm '; then
    conda create -n audiollm python=3.11 -y
fi
conda activate audiollm

# # ── Package installation ───────────────────────────────────────────────────────
pip install torch==2.5.1+cu124 torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu124 -q

pip install deepspeed==0.16.9 liger-kernel "wandb==0.19.11" requests -q
pip install --upgrade pydantic pydantic-core -q
pip install --force-reinstall nvidia-nccl-cu12==2.21.5 nvidia-cudnn-cu12==9.10.2.21 -q

# conda install cudatoolkit-dev=12.4 -c conda-forge

pip install causal-conv1d --no-build-isolation

pip install flash-attn==2.8.3 --no-build-isolation -q

pip install -e /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer -q
pip install -e /mnt/ddn/users/sehyun/AudioEncoder/AudioEnc/dacvae -q

# ── flash-linear-attention (modeling_qwen3_5AE.py imports fla.modules.FusedRMSNormGated)
# `--no-deps` prevents fla from upgrading torch (it pulls torch>=2.10).
pip install --no-deps "flash-linear-attention==0.5.0" "fla-core==0.5.0" -q

# ── glibc_stub.so (flash_attn GLIBC_2.32 workaround) ─────────────────────────
if [ ! -f "$CONDA_PREFIX/lib/glibc_stub.so" ]; then
    echo 'char __libc_single_threaded = 0;' > /tmp/glibc_stub.c
    gcc -shared -fPIC -o "$CONDA_PREFIX/lib/glibc_stub.so" /tmp/glibc_stub.c
fi

# ── flash_attn binary patch (remove GLIBC_2.32 version requirement) ───────────
FLASH_SO=$(python -c "
import flash_attn, os, glob
d = os.path.dirname(flash_attn.__file__)
hits = glob.glob(os.path.join(d, 'flash_attn_2_cuda*.so'))
print(hits[0] if hits else '')
")
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
