#!/bin/bash
# Run every benchmark in evaluation/audio/ against a single checkpoint,
# distributed across all visible GPUs via a FIFO-based slot pool.
#
# Usage:
#   bash evaluation/audio/run_all_evals.sh <ckpt-path> [out-root] [base-model]
#
#   <ckpt-path>   /path/to/<run>/checkpoint-<step>     (required)
#   [out-root]    where per-task subdirs are written   (default: <ckpt-path>/eval)
#   [base-model]  Stage-2 LoRA base model dir          (optional)
#
# GPU pool sizing:
#   - If CUDA_VISIBLE_DEVICES is set, those physical GPUs are used.
#   - Otherwise auto-detect via nvidia-smi (every GPU on the node).
#   - Override count with EVAL_MAX_GPUS=4 to cap parallel workers.
#
# Each benchmark gets its own subdir under <out-root>/, its own stdout/stderr log,
# and runs concurrently against the slot pool. A failure in one benchmark does
# NOT abort the rest. Per-task status (rc / wall / gpu) is written to
# <out-root>/logs/<task>.status and aggregated into summary.txt.
#
# Task launch order is longest-first (mAP-sequence + asr_external + librispeech)
# so the slowest jobs claim slots first and the trailing wave stays balanced.
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

# ---- GPU pool ---------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "[run_all_evals] no GPUs detected (CUDA_VISIBLE_DEVICES empty and nvidia-smi found none)" >&2
    exit 1
fi
if [ -n "${EVAL_MAX_GPUS:-}" ] && [ "$EVAL_MAX_GPUS" -lt "${#GPU_LIST[@]}" ]; then
    GPU_LIST=("${GPU_LIST[@]:0:$EVAL_MAX_GPUS}")
fi
N_GPU=${#GPU_LIST[@]}
echo "[run_all_evals] GPU pool: ${GPU_LIST[*]}  (N=$N_GPU)"

# Unset parent CUDA_VISIBLE_DEVICES so each child subshell controls device
# visibility via its own export. Child's CUDA_VISIBLE_DEVICES=<physical-id> is
# read fresh by the CUDA driver at process init.
unset CUDA_VISIBLE_DEVICES

# ---- task table -------------------------------------------------------------
# Each row: <task-id>|<module>|<extra args separated by US (\x1f)>
# Order = longest-first so slow jobs claim pool slots before short ones.
US=$'\x1f'
TASKS=(
    # fsd50k_map_seq: 200 label × 1000 sample teacher-forced — ~40 min/ckpt. OK.
    # audioset_map_seq DROPPED: 632 label × FLAC-parquet decode = 0.054 sps =
    # ~5 hr/ckpt. Trajectory 용으로 비현실. greedy F1/Jaccard 로 대체. 최종
    # ckpt 에서만 별도 sequence-mode mAP rerun 권장.
    "fsd50k_map_seq|evaluation.audio.eval_fsd50k_map|--score-mode${US}sequence${US}--max-samples${US}1000${US}--batch-size${US}32"
    "asr_external|evaluation.audio.eval_asr_external|--datasets${US}gigaspeech${US}--batch-size${US}32"
    "librispeech_clean|evaluation.audio.eval_librispeech_wer|--split${US}test.clean${US}--batch-size${US}32"
    "librispeech_other|evaluation.audio.eval_librispeech_wer|--split${US}test.other${US}--batch-size${US}32"
    "source_emotion|evaluation.audio.eval_source_emotion|--batch-size${US}32"
    "audiocaps_caption|evaluation.audio.eval_audiocaps_caption|--batch-size${US}32"
    "clotho_caption|evaluation.audio.eval_clotho_caption|--split${US}evaluation${US}--batch-size${US}32"
    "audioset_map_greedy|evaluation.audio.eval_audioset_map|--score-mode${US}greedy${US}--batch-size${US}32"
    "fsd50k_map_greedy|evaluation.audio.eval_fsd50k_map|--score-mode${US}greedy${US}--batch-size${US}32"
    "iemocap_s5|evaluation.audio.eval_iemocap_session5|--batch-size${US}32"
    # LISTEN evals (listen_mcqa / listen_official) 는 Stage-2 LISTEN-mix ckpt
    # 전용. Stage-1 ckpt + 본 노드 (LISTEN parquet 부재 + listen_mcqa 의
    # causal_conv1d GLIBCXX 이슈) 에서는 무의미. 필요 시 별도 호출.
)

# ---- FIFO semaphore: holds free GPU ids -------------------------------------
GPU_FIFO=$(mktemp -u --tmpdir="$TMPDIR" gpu_pool.XXXXXX.fifo)
mkfifo "$GPU_FIFO"
exec 9<>"$GPU_FIFO"
rm -f "$GPU_FIFO"
for g in "${GPU_LIST[@]}"; do
    echo "$g" >&9
done
trap 'exec 9>&-; exec 9<&-' EXIT

# ---- run loop (parallel) ----------------------------------------------------
START_ALL=$(date +%s)
PIDS=()

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
    TASK_STATUS="$OUT_ROOT/logs/${TASK_ID}.status"
    mkdir -p "$TASK_OUT"

    read -r GPU_ID <&9   # block until a GPU is free

    echo "[$(date '+%H:%M:%S')] dispatch $TASK_ID -> gpu $GPU_ID"

    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        T0=$(date +%s)
        "$PY" -m "$MODULE" \
            --ckpt-root "$CKPT_DIR" \
            --out-root "$TASK_OUT" \
            --ckpts "$STEP" \
            "${BASE_FLAG[@]}" \
            "${EXTRA[@]}" \
            > "$TASK_LOG" 2>&1
        RC=$?
        DT=$(( $(date +%s) - T0 ))
        echo "rc=$RC dt=$DT gpu=$GPU_ID" > "$TASK_STATUS"
        if [ $RC -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] $TASK_ID  ✓  ${DT}s  (gpu $GPU_ID)"
        else
            echo "[$(date '+%H:%M:%S')] $TASK_ID  ✗  rc=$RC  ${DT}s  (gpu $GPU_ID, see $TASK_LOG)"
        fi
        echo "$GPU_ID" >&9
    ) &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

# ---- summary ----------------------------------------------------------------
DT_ALL=$(( $(date +%s) - START_ALL ))
echo ""
echo "===================================================================="
echo "Summary  (ckpt $STEP, total ${DT_ALL}s, N_GPU=$N_GPU)"
echo "===================================================================="
{
    for entry in "${TASKS[@]}"; do
        TASK_ID="${entry%%|*}"
        TASK_STATUS="$OUT_ROOT/logs/${TASK_ID}.status"
        if [ -f "$TASK_STATUS" ]; then
            rc=$(grep -oE "rc=-?[0-9]+" "$TASK_STATUS" | cut -d= -f2)
            dt=$(grep -oE "dt=[0-9]+"     "$TASK_STATUS" | cut -d= -f2)
            gpu=$(grep -oE "gpu=[0-9]+"   "$TASK_STATUS" | cut -d= -f2)
            if [ "$rc" = "0" ]; then
                printf "  %-22s OK    %5ss   gpu=%s\n" "$TASK_ID" "$dt" "$gpu"
            else
                printf "  %-22s FAIL  rc=%s  %5ss   gpu=%s  see %s\n" \
                    "$TASK_ID" "$rc" "$dt" "$gpu" "$OUT_ROOT/logs/${TASK_ID}.log"
            fi
        else
            printf "  %-22s MISSING-STATUS  (subshell never wrote .status)\n" "$TASK_ID"
        fi
    done
} | tee "$OUT_ROOT/summary.txt"

exit $FAIL
