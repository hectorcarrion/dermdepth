#!/bin/bash
# Evaluate Exp H checkpoints on WoundsDB test, SKINL2 test, and DDI rulers
set -e

GPU=3
CKPT_DIR="output/training/exp_h/checkpoint"
STEPS="200 400 600 800 1000 1200 1400 1600 1800"

echo "============================================"
echo "Exp H Evaluation: WoundsDB + SKINL2 + DDI"
echo "============================================"

for STEP in $STEPS; do
    STEP_PADDED=$(printf "%08d" $STEP)
    CKPT="${CKPT_DIR}/${STEP_PADDED}_ema.pt"

    if [ ! -f "$CKPT" ]; then
        echo "Skipping step $STEP (checkpoint not found)"
        continue
    fi

    NAME="exp_h_s${STEP}"
    echo ""
    echo "========================================"
    echo "Step $STEP EMA"
    echo "========================================"

    # WoundsDB test
    CUDA_VISIBLE_DEVICES=$GPU python code/evaluation/eval_depth.py \
        --model "$CKPT" --dataset woundsdb --model_name "$NAME" --split test

    # SKINL2 test
    CUDA_VISIBLE_DEVICES=$GPU python code/evaluation/eval_depth.py \
        --model "$CKPT" --dataset skinl2 --model_name "$NAME" --split test

    # DDI rulers: save predictions
    CUDA_VISIBLE_DEVICES=$GPU python code/evaluation/eval_ddi_rulers.py \
        --save --method "$NAME" --checkpoint "$CKPT"
done

# Run DDI evaluation on all cached methods
echo ""
echo "========================================"
echo "DDI Ruler Evaluation (all methods)"
echo "========================================"
python code/evaluation/eval_ddi_rulers.py --evaluate

echo ""
echo "========================================"
echo "SUMMARY TABLE"
echo "========================================"
python3 -c "
import json, os, glob

models = [
    ('Base MoGe-2', 'output/evaluation/base_moge2_test', 'moge2'),
    ('Exp A (synth)', 'output/evaluation/exp_a_step1000_ema_test', 'dermdepth'),
]
# Add Exp G best (will be determined after G eval)
for s in [200,400,600,800,1000,1200,1400,1600,1800]:
    path = f'output/evaluation/exp_g_s{s}'
    if os.path.exists(path):
        models.append((f'Exp G s{s}', path, f'exp_g_s{s}'))
for s in [200,400,600,800,1000,1200,1400,1600,1800]:
    path = f'output/evaluation/exp_h_s{s}'
    if os.path.exists(path):
        models.append((f'Exp H s{s}', path, f'exp_h_s{s}'))

# Load DDI results
ddi = {}
ddi_path = 'output/evaluation/ddi_rulers/ddi_ruler_results.json'
if os.path.exists(ddi_path):
    ddi_data = json.load(open(ddi_path))
    for m, v in ddi_data.get('summary', {}).items():
        if 'all' in v:
            ddi[m] = v['all']['median_ratio']

print(f\"{'Model':<18} {'WDB AbsRel':>10} {'WDB Scale':>10} {'SK AbsRel':>10} {'SK Scale':>10} {'DDI Ratio':>10}\")
print('-' * 68)
for name, path, ddi_key in models:
    wdb_ar = wdb_sc = sk_ar = sk_sc = ddi_r = 'N/A'
    try:
        w = json.load(open(f'{path}/woundsdb/results.json'))
        wdb_ar = f\"{w['summary']['absrel']['mean']:.4f}\"
        wdb_sc = f\"{w['summary']['scale_ratio']['mean']:.3f}\"
    except: pass
    try:
        s = json.load(open(f'{path}/skinl2/results.json'))
        sk_ar = f\"{s['summary']['absrel']['mean']:.4f}\"
        sk_sc = f\"{s['summary']['scale_ratio']['mean']:.3f}\"
    except: pass
    if ddi_key in ddi:
        ddi_r = f'{ddi[ddi_key]:.2f}x'
    print(f'{name:<18} {wdb_ar:>10} {wdb_sc:>10} {sk_ar:>10} {sk_sc:>10} {ddi_r:>10}')
"

echo ""
echo "Done!"
