#!/usr/bin/env bash
# End-to-end driver for RQ2 / Analysis 3.
#
# Env knobs:
#   ALM_CKPT     : Stage1 base ckpt   (default: whisper-tiny v6, checkpoint-64000)
#   ALM_LORA     : Stage2 LoRA dir    (optional; merged + unloaded after load)
#   EMB_DIR      : where .npz files go
#   ANALYZE_OUT  : where csv/png/md go
#   PAIRS        : pre-built combined pair file (jsonl)
#   SKIP_EXTRACT : =1 skips extract phase (re-run analyzers only)
#   SKIP_PROBE   : =1 skips linear probe (slow if many layers / classes)
#   SKIP_CKA     : =1 skips CKA pass
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

EMB_DIR="${EMB_DIR:-${HERE}/emb}"
ANALYZE_OUT="${ANALYZE_OUT:-${HERE}/analyze/out}"
PAIRS="${PAIRS:-${HERE}/pairs/all_pairs.jsonl}"
ALM_CKPT="${ALM_CKPT:-/mnt/tmp/Qwen3.5_whisper_tiny_v6_Stage1/Qwen3.5AE-ASR-Stage1-whisper-tiny-v6/checkpoint-64000}"
ALM_LORA="${ALM_LORA:-}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ----- 1. Pairs -----
if [ ! -f "${HERE}/pairs/iemocap_pairs.jsonl" ]; then
  echo "[run] building IEMOCAP pairs"
  python pairs/build_iemocap_pairs.py
fi
if [ ! -f "${HERE}/pairs/arctic_pairs.jsonl" ]; then
  if [ ! -d /mnt/tmp/datasets/cmu_arctic ]; then
    echo "[run] downloading CMU Arctic (this takes a while)"
    bash pairs/download_cmu_arctic.sh
  fi
  python pairs/build_arctic_pairs.py
fi
if [ ! -f "${PAIRS}" ]; then
  python pairs/combine.py --per_corpus 60 --out "${PAIRS}"
fi

# ----- 2. Extract -----
mkdir -p "${EMB_DIR}"

if [ "${SKIP_EXTRACT:-0}" != "1" ]; then
  # raw HF Whisper encoders (no projector / no LLM)
  python extract/extract_whisper.py --model openai/whisper-large-v2 \
         --pairs "${PAIRS}" --out "${EMB_DIR}/whisper_large_v2"
  python extract/extract_whisper.py --model openai/whisper-tiny.en \
         --pairs "${PAIRS}" --out "${EMB_DIR}/whisper_tiny"

  # SSL acoustic baseline
  python extract/extract_ssl.py --model facebook/hubert-base-ls960 \
         --pairs "${PAIRS}" --out "${EMB_DIR}/hubert_base"

  # trained ALM (encoder + projector + LLM, hooked)
  alm_args=( --ckpt "${ALM_CKPT}" --pairs "${PAIRS}"
             --out "${EMB_DIR}/alm_whisper_tiny_v6"
             --llm_stride 2 --patch_flash )
  if [ -n "${ALM_LORA}" ]; then
    alm_args+=( --lora "${ALM_LORA}" )
  fi
  python extract/extract_alm.py "${alm_args[@]}"
fi

# ----- 3. Analyze -----
mkdir -p "${ANALYZE_OUT}"
python analyze/distances.py --emb_dir "${EMB_DIR}" --out "${ANALYZE_OUT}"
python analyze/pca_plot.py  --emb_dir "${EMB_DIR}" --out "${ANALYZE_OUT}" --per_src

if [ "${SKIP_PROBE:-0}" != "1" ]; then
  python analyze/probe.py --emb_dir "${EMB_DIR}" --out "${ANALYZE_OUT}"
fi
if [ "${SKIP_CKA:-0}" != "1" ]; then
  python analyze/cka.py --emb_dir "${EMB_DIR}" --out "${ANALYZE_OUT}" --heatmap
fi

python analyze/report.py --out "${ANALYZE_OUT}"

echo "[run] DONE  artifacts in ${ANALYZE_OUT}"
