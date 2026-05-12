#!/usr/bin/env bash
# Setup script for audiollm conda environment
# Tested on: Python 3.11 / CUDA 12.4 / PyTorch 2.5.1+cu124
#
# Usage:
#   bash setup_audiollm.sh                        # full install
#   bash setup_audiollm.sh --skip-flash           # skip flash-attn build (slow)
#   bash setup_audiollm.sh --cu121                # use CUDA 12.1 wheels instead
#   bash setup_audiollm.sh --name audiollm_test   # use a custom env name

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
SKIP_FLASH=0
CUDA_TAG="cu124"
ENV_NAME="audiollm"
_PREV_ARG=""
for arg in "$@"; do
    case $arg in
        --skip-flash) SKIP_FLASH=1 ;;
        --cu121)      CUDA_TAG="cu121" ;;
        --cu118)      CUDA_TAG="cu118" ;;
        --name)       ;;  # value read on next iteration
        --help|-h)
            echo "Usage: $0 [--skip-flash] [--cu121] [--cu118] [--name ENV_NAME]"
            exit 0
            ;;
        *)
            if [ "${_PREV_ARG}" = "--name" ]; then ENV_NAME="${arg}"; fi
            ;;
    esac
    _PREV_ARG="${arg}"
done
PYTHON_VER="3.11"
TORCH_VER="2.5.1"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

# ── Helpers ───────────────────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

# ── 0. Conda 확인 ─────────────────────────────────────────────────────────────
command -v conda >/dev/null 2>&1 || die "conda not found. Install miniconda first."
info "Using conda: $(conda --version)"

# ── 1. 환경 생성 ──────────────────────────────────────────────────────────────
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    warn "Conda env '${ENV_NAME}' already exists. Skipping creation."
    warn "To recreate: conda env remove -n ${ENV_NAME} && bash $0"
else
    info "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VER}, conda-forge)..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -c conda-forge -y
fi

PIP="conda run -n ${ENV_NAME} python -m pip"
PYTHON="conda run -n ${ENV_NAME} python"

# ── 1.5. pip 업그레이드 (구버전 pip은 최신 패키지 메타데이터 못 읽음) ──────────
info "Upgrading pip..."
${PIP} install --upgrade pip

# ── 2. conda-forge: ffmpeg ────────────────────────────────────────────────────
info "Installing ffmpeg via conda-forge..."
conda install -n "${ENV_NAME}" -c conda-forge ffmpeg -y

# ── 3. PyTorch ───────────────────────────────────────────────────────────────
info "Installing PyTorch ${TORCH_VER}+${CUDA_TAG}..."
${PIP} install \
    "torch==${TORCH_VER}+${CUDA_TAG}" \
    "torchaudio==${TORCH_VER}+${CUDA_TAG}" \
    "torchvision==0.20.1+${CUDA_TAG}" \
    --index-url "${TORCH_INDEX}"
# Alternative (conda channel, no bundled CUDA wheels):
# conda install -n "${ENV_NAME}" pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# ── 4. 핵심 LLM/학습 패키지 ──────────────────────────────────────────────────
info "Installing core training packages..."
${PIP} install \
    transformers==4.57.1 \
    accelerate==1.11.0 \
    peft==0.17.1 \
    trl==0.24.0 \
    datasets==4.0.0 \
    deepspeed==0.16.9 \
    wandb==0.26.1 \
    huggingface_hub==0.36.2 \
    sentencepiece==0.2.1 \
    tiktoken==0.12.0 \
    tokenizers==0.22.2 \
    safetensors==0.7.0 \
    liger_kernel==0.7.0 \
    torchdata==0.11.0 \
    modelscope==1.36.2

# ── 5. 오디오 처리 패키지 ────────────────────────────────────────────────────
info "Installing audio packages..."
${PIP} install \
    librosa==0.11.0 \
    soundfile==0.13.1 \
    av==17.0.1 \
    pydub==0.25.1 \
    soxr==1.0.0 \
    julius==0.2.7 \
    pyloudnorm==0.2.0 \
    pystoi==0.4.1 \
    torch-stoi==0.2.3 \
    descript-audiotools==0.7.2 \
    argbind==0.3.9

# ── 6. 수치/과학 패키지 ───────────────────────────────────────────────────────
info "Installing scientific computing packages..."
${PIP} install \
    numpy==1.26.4 \
    scipy==1.17.1 \
    scikit-learn==1.8.0 \
    pandas==2.3.3 \
    matplotlib==3.10.9 \
    seaborn==0.13.2 \
    h5py==3.16.0 \
    numba==0.65.1 \
    pyarrow==24.0.0 \
    einops==0.8.2 \
    omegaconf==2.3.0

# ── 7. EEG/신호처리 특수 패키지 (probe 실험용) ───────────────────────────────
info "Installing signal processing / EEG packages..."
${PIP} install \
    mlend==1.0.0.4 \
    phyaat==0.0.3 \
    spkit==0.0.9.7 \
    pylfsr==1.0.7 \
    PyWavelets==1.9.0

# ── 8. 유틸리티 ──────────────────────────────────────────────────────────────
info "Installing utilities..."
${PIP} install \
    tensorboard==2.20.0 \
    gradio==5.50.0 \
    fire==0.7.1 \
    gdown==6.0.0 \
    ninja \
    psutil==7.2.2 \
    tqdm==4.67.3 \
    rich==15.0.0 \
    hf_transfer==0.1.9 \
    ipython==9.13.0

# ── 9. Flash Attention ───────────────────────────────────────────────────────
if [ "${SKIP_FLASH}" -eq 1 ]; then
    warn "Skipping flash-attn (--skip-flash). Install manually later:"
    warn "  conda run -n ${ENV_NAME} pip install flash-attn==2.8.3 --no-build-isolation"
else
    info "Installing flash-attn 2.8.3 (CUDA 빌드 — 10~30분 소요)..."
    info "미리 빌드된 whl이 있으면 Ctrl-C 후 직접 pip install 권장"
    info "  → https://github.com/Dao-AILab/flash-attention/releases"
    ${PIP} install flash-attn==2.8.3 --no-build-isolation
fi

# ── 10. causal_conv1d & Flash Linear Attention ───────────────────────────────
info "Installing causal_conv1d and flash-linear-attention..."
# --no-build-isolation: 빌드 격리 환경이 최신 torch를 당겨오는 버그 방지
${PIP} install causal_conv1d==1.6.1 --no-build-isolation
${PIP} install flash-linear-attention==0.4.0   # fla-core 포함

# ── 11. dacvae (editable install) ─────────────────────────────────────────────
info "Installing dacvae..."
DACVAE_SRC="${DACVAE_PATH:-}"

if [ -z "${DACVAE_SRC}" ]; then
    # 환경변수 미지정 시 기본 경로 시도
    DACVAE_CANDIDATES=(
        "$(dirname "$(realpath "$0")")/../AudioEnc/dacvae"
        "${HOME}/AudioEnc/dacvae"
        "/mnt/ddn/users/sehyun/AudioEncoder/AudioEnc/dacvae"
    )
    for candidate in "${DACVAE_CANDIDATES[@]}"; do
        if [ -d "${candidate}" ]; then
            DACVAE_SRC="${candidate}"
            break
        fi
    done
fi

if [ -n "${DACVAE_SRC}" ] && [ -d "${DACVAE_SRC}" ]; then
    info "dacvae found at: ${DACVAE_SRC}"
    ${PIP} install -e "${DACVAE_SRC}"
else
    warn "dacvae 소스를 찾을 수 없습니다."
    warn "다음 중 하나로 직접 설치하세요:"
    warn "  1) DACVAE_PATH=/path/to/dacvae bash $0"
    warn "  2) git clone https://github.com/facebookresearch/dacvae && pip install -e dacvae"
fi

# ── 12. audiollm-trainer (llamafactory, editable install) ────────────────────
TRAINER_DIR="$(dirname "$(realpath "$0")")"
if [ -f "${TRAINER_DIR}/pyproject.toml" ] || [ -f "${TRAINER_DIR}/setup.py" ]; then
    info "Installing audiollm-trainer (llamafactory) in editable mode..."
    ${PIP} install -e "${TRAINER_DIR}"
else
    warn "audiollm-trainer pyproject.toml 미발견 — editable install 건너뜀"
    warn "스크립트를 audiollm-trainer 루트에서 실행하거나 수동으로 pip install -e . 실행"
fi

# ── 13. wandb netrc 설정 안내 ─────────────────────────────────────────────────
echo ""
info "=== wandb 설정 안내 ==="
info "wandb 설정: wandb login 명령으로 로그인하세요."

# ── 14. 검증 ─────────────────────────────────────────────────────────────────
echo ""
info "=== 설치 검증 ==="
${PYTHON} - <<'PYEOF'
import importlib, sys

checks = {
    "torch":            lambda m: f"{m.__version__} (CUDA: {m.version.cuda})",
    "torchaudio":       lambda m: m.__version__,
    "transformers":     lambda m: m.__version__,
    "accelerate":       lambda m: m.__version__,
    "deepspeed":        lambda m: m.__version__,
    "flash_attn":       lambda m: m.__version__,
    "causal_conv1d":    lambda m: m.__version__,
    "librosa":          lambda m: m.__version__,
    "dacvae":           lambda m: m.__version__,
    "llamafactory":     lambda m: m.__version__,
}

ok, fail = [], []
for name, ver_fn in checks.items():
    try:
        mod = importlib.import_module(name)
        ok.append(f"  [OK] {name}: {ver_fn(mod)}")
    except Exception as e:
        fail.append(f"  [FAIL] {name}: {e}")

print("\n".join(ok))
if fail:
    print("\n--- 실패 ---")
    print("\n".join(fail))
    sys.exit(1)
PYEOF

echo ""
info "Done. 활성화: conda activate ${ENV_NAME}"
