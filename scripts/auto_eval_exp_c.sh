#!/bin/bash
# Auto-evaluate Exp C checkpoints as they appear
# Evaluates on both SKINL2 and WoundsDB (CPU, no GPU needed)

cd .

CKPT_DIR="output/training/exp_c/checkpoint"
EVAL_BASE="output/evaluation"
STEPS="3000 6000 9000 12000 15000"

echo "[$(date -u)] Starting auto-eval for Exp C..."

for step in $STEPS; do
    ema_ckpt="${CKPT_DIR}/$(printf '%08d' $step)_ema.pt"
    result_dir="${EVAL_BASE}/exp_c_step${step}_ema"

    # Wait for checkpoint to appear
    echo "[$(date -u)] Waiting for step ${step} checkpoint: ${ema_ckpt}"
    while [ ! -f "$ema_ckpt" ]; do
        sleep 60
    done

    # Wait a bit more for file to finish writing
    sleep 10
    echo "[$(date -u)] Found step ${step} checkpoint! Starting evaluation..."

    # Check if already evaluated
    if [ -f "${result_dir}/skinl2/results.json" ] && [ -f "${result_dir}/woundsdb/results.json" ]; then
        echo "[$(date -u)] Step ${step} already evaluated, skipping."
        continue
    fi

    # Evaluate SKINL2
    echo "[$(date -u)] Evaluating step ${step} on SKINL2..."
    CUDA_VISIBLE_DEVICES="" conda run -n MoGe python code/evaluation/eval_depth.py \
        --model "$ema_ckpt" \
        --dataset skinl2 \
        --device cpu \
        --output_dir "${result_dir}" \
        --model_name skinl2 2>&1

    # Evaluate WoundsDB
    echo "[$(date -u)] Evaluating step ${step} on WoundsDB..."
    CUDA_VISIBLE_DEVICES="" conda run -n MoGe python code/evaluation/eval_depth.py \
        --model "$ema_ckpt" \
        --dataset woundsdb \
        --device cpu \
        --output_dir "${result_dir}" \
        --model_name woundsdb 2>&1

    # Print summary
    echo "[$(date -u)] Step ${step} evaluation complete."
    echo "SKINL2 results:"
    python3 -c "
import json
with open('${result_dir}/skinl2/results.json') as f:
    r = json.load(f)
print(f'  Scale: {r[\"overall\"][\"scale_ratio\"]:.3f}, AbsRel: {r[\"overall\"][\"abs_rel\"]:.3f}, SI-Delta1: {r[\"overall\"][\"si_delta1\"]:.3f}')
" 2>/dev/null || echo "  (failed to read)"
    echo "WoundsDB results:"
    python3 -c "
import json
with open('${result_dir}/woundsdb/results.json') as f:
    r = json.load(f)
print(f'  Scale: {r[\"overall\"][\"scale_ratio\"]:.3f}, AbsRel: {r[\"overall\"][\"abs_rel\"]:.3f}, SI-Delta1: {r[\"overall\"][\"si_delta1\"]:.3f}')
" 2>/dev/null || echo "  (failed to read)"
    echo "---"
done

echo "[$(date -u)] All Exp C evaluations complete!"
