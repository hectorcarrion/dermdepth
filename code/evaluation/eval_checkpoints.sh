#!/bin/bash
# Evaluate checkpoints as they appear during training.
# Usage: bash eval_checkpoints.sh <exp_dir> <model_name> [gpu_id]
# Example: bash eval_checkpoints.sh output/training/exp_a exp_a 0

set -e

EXP_DIR="${1:?Usage: eval_checkpoints.sh <exp_dir> <model_name> [gpu_id]}"
MODEL_NAME="${2:?Usage: eval_checkpoints.sh <exp_dir> <model_name> [gpu_id]}"
GPU_ID="${3:-0}"
EVAL_SCRIPT="code/evaluation/eval_depth.py"
CKPT_DIR="${EXP_DIR}/checkpoint"
EVAL_OUTPUT="output/evaluation"

cd .

# Track which checkpoints we've already evaluated
EVALUATED_FILE="${EXP_DIR}/evaluated_checkpoints.txt"
touch "$EVALUATED_FILE"

echo "=== Checkpoint Evaluator ==="
echo "  Experiment: ${EXP_DIR}"
echo "  Model name: ${MODEL_NAME}"
echo "  GPU: ${GPU_ID}"
echo "  Watching: ${CKPT_DIR}"
echo ""

eval_checkpoint() {
    local ckpt_path="$1"
    local ckpt_name="$2"
    local tag="$3"  # "raw" or "ema"

    local run_name="${MODEL_NAME}_${ckpt_name}_${tag}"
    echo "[$(date '+%H:%M:%S')] Evaluating ${run_name}..."

    CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n MoGe python ${EVAL_SCRIPT} \
        --model "${ckpt_path}" \
        --dataset all \
        --output_dir "${EVAL_OUTPUT}" \
        --model_name "${run_name}" \
        --device cuda 2>&1 | tail -30

    echo "${ckpt_name}_${tag}" >> "$EVALUATED_FILE"
    echo "[$(date '+%H:%M:%S')] Done: ${run_name}"
    echo ""
}

# Evaluate all existing checkpoints
for ckpt_file in $(ls ${CKPT_DIR}/[0-9]*.pt 2>/dev/null | grep -v '_optimizer\|_ema' | sort); do
    step_name=$(basename "$ckpt_file" .pt)

    # Evaluate raw checkpoint
    if ! grep -q "^${step_name}_raw$" "$EVALUATED_FILE" 2>/dev/null; then
        eval_checkpoint "$ckpt_file" "$step_name" "raw"
    fi

    # Evaluate EMA checkpoint if it exists
    ema_file="${CKPT_DIR}/${step_name}_ema.pt"
    if [ -f "$ema_file" ] && ! grep -q "^${step_name}_ema$" "$EVALUATED_FILE" 2>/dev/null; then
        eval_checkpoint "$ema_file" "$step_name" "ema"
    fi
done

echo "[$(date '+%H:%M:%S')] All current checkpoints evaluated."
echo "Re-run this script to evaluate new checkpoints as they appear."
