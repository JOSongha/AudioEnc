#!/bin/bash
# Run every benchmark in evaluation/audio/ against a single checkpoint.
#
# Usage:
#   bash evaluation/audio/run_all_evals.sh <ckpt-path> [out-root] [base-model]
#
#   <ckpt-path>   /path/to/<run>/checkpoint-<step>     (required)
#   [out-root]    where per-task subdirs are written   (default: <ckpt-path>/eval)
#   [base-model]  Stage-2 LoRA base model dir          (optional)
#
# Each benchmark gets its own subdir under <out-root>/, its own stdout/stderr log,
# and runs sequentially. A failure in one benchmark does NOT abort the rest.
# Final status of all 11 evals is summarized at the end.
#
# Conventions (cf. CLAUDE memory): all work in tmux, all outputs under /mnt/tmp,
# target 100% GPU util.

set -u
set -o pipefail

# ---- args -------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "usage: $(basename "$0") <ckpt-path> [out-root] [base-model]" >&2
    exit 2
fi
CKPT_PATH="${1%/}"
OUT_ROOT="${2:-$CKPT_PATH/eval}"
BASE_MODEL="${3:-}"

if [ ! -d "$CKPT_PATH" ]; then
    echo "[run_all_evals] ckpt-path is not a directory: $CKPT_PATH" >&2
    exit 1
fi

CKPT_DIR=$(dirname "$CKPT_PATH")
CKPT_NAME=$(basename "$CKPT_PATH")     # e.g. checkpoint-46000
STEP="${CKPT_NAME#checkpoint-}"
if [ "$STEP" = "$CKPT_NAME" ]; then
    echo "[run_all_evals] expected basename 'checkpoint-<step>', got: $CKPT_NAME" >&2
    exit 1
fi

# ---- env --------------------------------------------------------------------
REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -n "${AUDIO_LMF_ENV:-}" ]; then
    ENV_PREFIX="$AUDIO_LMF_ENV"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    ENV_PREFIX="$CONDA_PREFIX"
else
    echo "[run_all_evals] Set AUDIO_LMF_ENV or activate your conda env first." >&2
    exit 1
fi
PY="$ENV_PREFIX/bin/python"
export PATH="$ENV_PREFIX/bin:$PATH"

export TRITON_CACHE_DIR=/mnt/tmp/cache/triton
export CUDA_CACHE_PATH=/mnt/tmp/cache/nv_compute
export HF_HOME=/mnt/tmp/cache/huggingface
export TMPDIR=/mnt/tmp/cache/tmp
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$TMPDIR" "$OUT_ROOT/logs"

cd "$REPO"

BASE_FLAG=()
if [ -n "$BASE_MODEL" ]; then
    BASE_FLAG=(--base-model "$BASE_MODEL")
fi

# ---- task table -------------------------------------------------------------
# Each row: <task-id>|<module>|<extra args separated by | as a single field>
# Extra args are split on $'\x1f' (unit separator) to keep spaces intact.
US=$'\x1f'
TASKS=(
    "librispeech_clean|evaluation.audio.eval_librispeech_wer|--split${US}test.clean"
    "librispeech_other|evaluation.audio.eval_librispeech_wer|--split${US}test.other"
    "asr_external|evaluation.audio.eval_asr_external|--datasets${US}mls${US}voxpopuli${US}gigaspeech"
    "clotho_caption|evaluation.audio.eval_clotho_caption|--split${US}evaluation"
    "fsd50k_map_greedy|evaluation.audio.eval_fsd50k_map|--score-mode${US}greedy"
    "fsd50k_map_seq|evaluation.audio.eval_fsd50k_map|--score-mode${US}sequence"
    "audioset_map_greedy|evaluation.audio.eval_audioset_map|--score-mode${US}greedy"
    "audioset_map_seq|evaluation.audio.eval_audioset_map|--score-mode${US}sequence"
    "source_emotion|evaluation.audio.eval_source_emotion|"
    "iemocap_s5|evaluation.audio.eval_iemocap_session5|"
    "listen_mcqa|evaluation.audio.eval_listen_mcqa|"
    "listen_official|evaluation.audio.eval_listen_official|"
)

# ---- run loop ---------------------------------------------------------------
declare -A STATUS
START_ALL=$(date +%s)

for entry in "${TASKS[@]}"; do
    TASK_ID="${entry%%|*}"
    rest="${entry#*|}"
    MODULE="${rest%%|*}"
    EXTRA_STR="${rest#*|}"

    EXTRA=()
    if [ -n "$EXTRA_STR" ]; then
        IFS="$US" read -ra EXTRA <<<"$EXTRA_STR"
    fi

    TASK_OUT="$OUT_ROOT/$TASK_ID"
    TASK_LOG="$OUT_ROOT/logs/${TASK_ID}.log"
    mkdir -p "$TASK_OUT"

    echo ""
    echo "===================================================================="
    echo "[$(date '+%H:%M:%S')] $TASK_ID  (ckpt $STEP)"
    echo "  module : $MODULE"
    echo "  out    : $TASK_OUT"
    echo "  log    : $TASK_LOG"
    echo "===================================================================="

    T0=$(date +%s)
    set +e
    "$PY" -m "$MODULE" \
        --ckpt-root "$CKPT_DIR" \
        --out-root "$TASK_OUT" \
        --ckpts "$STEP" \
        "${BASE_FLAG[@]}" \
        "${EXTRA[@]}" \
        > "$TASK_LOG" 2>&1
    RC=$?
    set -e
    DT=$(( $(date +%s) - T0 ))

    if [ $RC -eq 0 ]; then
        STATUS["$TASK_ID"]="OK   ${DT}s"
        echo "[$(date '+%H:%M:%S')] $TASK_ID  ✓  ${DT}s"
    else
        STATUS["$TASK_ID"]="FAIL rc=$RC ${DT}s  see $TASK_LOG"
        echo "[$(date '+%H:%M:%S')] $TASK_ID  ✗  rc=$RC  ${DT}s  (see $TASK_LOG)"
    fi
done

# ---- summary ----------------------------------------------------------------
DT_ALL=$(( $(date +%s) - START_ALL ))
echo ""
echo "===================================================================="
echo "Summary  (ckpt $STEP, total ${DT_ALL}s)"
echo "===================================================================="
for entry in "${TASKS[@]}"; do
    TASK_ID="${entry%%|*}"
    printf "  %-22s %s\n" "$TASK_ID" "${STATUS[$TASK_ID]}"
done | tee "$OUT_ROOT/summary.txt"

# Exit non-zero if any task failed (so wrappers / CI can tell).
for s in "${STATUS[@]}"; do
    [[ "$s" == FAIL* ]] && exit 1
done
exit 0
