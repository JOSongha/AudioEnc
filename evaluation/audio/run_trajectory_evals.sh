#!/bin/bash
# Sequential wrapper around `run_all_evals.sh` — iterate every checkpoint-*
# subdir under a run-root and run the full 13-task GPU pool against each.
#
# Usage:
#   bash evaluation/audio/run_trajectory_evals.sh <run-root> [out-root] [base-model]
#
#   <run-root>    e.g. .../Qwen3.5_dac_vae_v6_Stage1_jos/Qwen3.5AE-ASR-Stage1-dac-vae-v6/
#                 (contains checkpoint-10000, checkpoint-20000, …)
#   [out-root]    where per-ckpt eval dirs are written
#                 (default: <run-root>/eval_trajectory/<ckpt-name>/)
#   [base-model]  Stage-2 LoRA base model dir          (optional)
#
# Each ckpt's run delegates to `run_all_evals.sh`, which distributes its 13
# benchmarks across the full visible GPU pool (FIFO semaphore). A failure on
# one ckpt does NOT abort the rest — per-ckpt summary printed at the end.

set -u
set -o pipefail

if [ $# -lt 1 ]; then
    echo "usage: $(basename "$0") <run-root> [out-root] [base-model]" >&2
    exit 2
fi
RUN_ROOT="${1%/}"
OUT_ROOT="${2:-$RUN_ROOT/eval_trajectory}"
BASE_MODEL="${3:-}"

if [ ! -d "$RUN_ROOT" ]; then
    echo "[trajectory] run-root is not a directory: $RUN_ROOT" >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUN_ALL="$SCRIPT_DIR/run_all_evals.sh"
if [ ! -x "$RUN_ALL" ] && [ ! -r "$RUN_ALL" ]; then
    echo "[trajectory] cannot locate run_all_evals.sh at $RUN_ALL" >&2
    exit 1
fi

# Enumerate checkpoint dirs, sort by step number (numeric, not lex).
mapfile -t CKPTS < <(find "$RUN_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' \
    | awk -F'checkpoint-' '{print $2 "\t" $0}' | sort -n | cut -f2)
if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "[trajectory] no checkpoint-* subdirs under $RUN_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
TRAJ_LOG="$OUT_ROOT/trajectory.log"
TRAJ_SUMMARY="$OUT_ROOT/trajectory_summary.txt"
: > "$TRAJ_SUMMARY"

echo "[trajectory] $(date '+%F %T') start"     | tee -a "$TRAJ_LOG"
echo "[trajectory] run-root: $RUN_ROOT"        | tee -a "$TRAJ_LOG"
echo "[trajectory] out-root: $OUT_ROOT"        | tee -a "$TRAJ_LOG"
echo "[trajectory] ckpts (${#CKPTS[@]}):"      | tee -a "$TRAJ_LOG"
for c in "${CKPTS[@]}"; do
    echo "  - $(basename "$c")"                | tee -a "$TRAJ_LOG"
done

T_ALL=$(date +%s)
declare -a FAILED=()

for CKPT in "${CKPTS[@]}"; do
    CKPT_NAME=$(basename "$CKPT")
    CKPT_OUT="$OUT_ROOT/$CKPT_NAME"
    mkdir -p "$CKPT_OUT"

    echo ""                                                | tee -a "$TRAJ_LOG"
    echo "===================================================================="   | tee -a "$TRAJ_LOG"
    echo "[trajectory] $(date '+%F %T')  $CKPT_NAME  →  $CKPT_OUT"                 | tee -a "$TRAJ_LOG"
    echo "===================================================================="   | tee -a "$TRAJ_LOG"

    T0=$(date +%s)
    set +e
    bash "$RUN_ALL" "$CKPT" "$CKPT_OUT" "$BASE_MODEL" >> "$TRAJ_LOG" 2>&1
    RC=$?
    set -e
    DT=$(( $(date +%s) - T0 ))

    if [ $RC -eq 0 ]; then
        line="$CKPT_NAME  OK    ${DT}s  ($CKPT_OUT/summary.txt)"
    else
        line="$CKPT_NAME  FAIL  rc=$RC  ${DT}s  see $TRAJ_LOG"
        FAILED+=("$CKPT_NAME")
    fi
    echo "$line" | tee -a "$TRAJ_LOG" "$TRAJ_SUMMARY" > /dev/null
    echo "[trajectory] $(date '+%F %T')  $line"
done

DT_ALL=$(( $(date +%s) - T_ALL ))
echo ""                                                                  | tee -a "$TRAJ_LOG"
echo "===================================================================="  | tee -a "$TRAJ_LOG"
echo "[trajectory] DONE  ${#CKPTS[@]} ckpts  total ${DT_ALL}s"            | tee -a "$TRAJ_LOG"
echo "[trajectory] failed: ${#FAILED[@]}  (${FAILED[*]:-none})"           | tee -a "$TRAJ_LOG"
echo "===================================================================="  | tee -a "$TRAJ_LOG"

if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
exit 0
