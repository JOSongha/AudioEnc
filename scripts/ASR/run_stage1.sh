# ── Conda env setup ───────────────────────────────────────────────────────────
source /mnt/tmp/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q '^audiollm '; then
    conda create -n audiollm python=3.11 -y
fi
conda activate audiollm

# # ── Package installation ───────────────────────────────────────────────────────
# pip install torch==2.5.1+cu124 torchaudio torchvision \
#     --index-url https://download.pytorch.org/whl/cu124 -q

# pip install deepspeed==0.16.9 liger-kernel "wandb==0.19.11" requests -q
# pip install --upgrade pydantic pydantic-core -q
# pip install --force-reinstall nvidia-nccl-cu12==2.21.5 nvidia-cudnn-cu12==9.10.2.21 -q

# CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ \
#     pip install causal-conv1d==1.6.1 -q

# pip install flash-attn==2.8.3 --no-build-isolation -q

# pip install -e /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer -q
# pip install -e /mnt/ddn/users/sehyun/AudioEncoder/AudioEnc/dacvae -q

# # ── glibc_stub.so (flash_attn GLIBC_2.32 workaround) ─────────────────────────
# if [ ! -f "$CONDA_PREFIX/lib/glibc_stub.so" ]; then
#     echo 'char __libc_single_threaded = 0;' > /tmp/glibc_stub.c
#     gcc -shared -fPIC -o "$CONDA_PREFIX/lib/glibc_stub.so" /tmp/glibc_stub.c
# fi

# # ── flash_attn binary patch (remove GLIBC_2.32 version requirement) ───────────
# FLASH_SO=$(python -c "
# import flash_attn, os, glob
# d = os.path.dirname(flash_attn.__file__)
# hits = glob.glob(os.path.join(d, 'flash_attn_2_cuda*.so'))
# print(hits[0] if hits else '')
# ")
# if [ -n "$FLASH_SO" ] && python -c "
# import subprocess, sys
# out = subprocess.check_output(['objdump', '-p', sys.argv[1]], text=True)
# sys.exit(0 if 'GLIBC_2.32' in out else 1)
# " "$FLASH_SO" 2>/dev/null; then
#     python - "$FLASH_SO" <<'EOF'
# import sys, struct

# path = sys.argv[1]
# data = bytearray(open(path, 'rb').read())

# # VERSYM: set __libc_single_threaded's version index to 1 (global/unversioned)
# struct.pack_into('<H', data, 900152, 0x0001)
# # VERNEED: libc.so.6 vn_cnt 5->4
# struct.pack_into('<H', data, 907618, 4)
# # VERNEED: GLIBC_2.14 vna_next 16->32 (skip GLIBC_2.32 entry)
# struct.pack_into('<I', data, 907644, 32)

# open(path, 'wb').write(data)
# print(f"[patch] Applied GLIBC_2.32 patch to {path}")
# EOF
# fi

# ── Environment variables ──────────────────────────────────────────────────────
export WANDB_MODE=online
export WANDB_PROJECT=audiollm-trainer
export WANDB_API_KEY=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
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

AUDIOLLM_PREFIX=/mnt/tmp/miniconda3/envs/audiollm
NVIDIA_LIBS=$AUDIOLLM_PREFIX/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cuda_cupti/lib:$NVIDIA_LIBS/cuda_nvrtc/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/nvjitlink/lib:$NVIDIA_LIBS/nvtx/lib:$AUDIOLLM_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$AUDIOLLM_PREFIX/lib/glibc_stub.so
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

# ── Training ──────────────────────────────────────────────────────────────────
# MODEL_PATH=/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/models/Qwen3AE-4B
# OMNI_MANIFEST=/mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/datasets/libri_mls_vox

FORCE_TORCHRUN=1 NNODES=$NSML_WORLD_SIZE NODE_RANK=$NSML_RANK MASTER_ADDR=$NSML_HOST_RANK0 MASTER_PORT=21267 \
    llamafactory-cli train /mnt/ddn/users/sehyun/AudioEncoder/audiollm-trainer/external/configs/test.yaml