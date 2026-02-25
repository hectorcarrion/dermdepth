#!/usr/bin/env python3
"""Post-hoc calibration baseline.

Step 1: Run MoGe-2 on synthetic training data to compute calibration constant k.
Step 2: Analytically apply k to existing per-sample eval results and compare with DermDepth.
"""

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
SYNTH_DIR = PROJECT_ROOT / "data" / "dermdepth_train" / "colab_gen" / "DermDepthSynth"
sys.path.insert(0, str(MOGE_ROOT))


def compute_calibration_constant(n_samples=200, device='cuda'):
    """Run MoGe-2 on synthetic training samples to get median scale ratio."""
    from moge.model import import_model_class_by_version
    from moge.utils.io import read_depth
    import torchvision.transforms.functional as TF
    from PIL import Image

    MoGeModel = import_model_class_by_version("v2")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

    all_samples = sorted([d.name for d in SYNTH_DIR.iterdir()
                          if d.is_dir() and (d / "image.png").exists()])
    random.seed(0)
    samples = random.sample(all_samples, min(n_samples, len(all_samples)))
    print(f"Running MoGe-2 on {len(samples)} synthetic samples...")

    per_sample_scales = []
    for i, name in enumerate(samples):
        sample_dir = SYNTH_DIR / name
        gt_depth_mm = read_depth(str(sample_dir / "depth.png"))
        gt_depth_m = gt_depth_mm * 0.001

        img = Image.open(sample_dir / "image.png").convert('RGB')
        img_t = TF.to_tensor(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model.infer(img_t)
        pred = out['depth'].cpu().numpy()
        if pred.ndim > 2 and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        mask = np.isfinite(gt_depth_m) & (gt_depth_m > 0) & np.isfinite(pred) & (pred > 0)
        if mask.sum() > 100:
            scale = float(np.median(pred[mask] / gt_depth_m[mask]))
            per_sample_scales.append(scale)
            if i < 5 or i % 50 == 0:
                print(f"  [{i+1}/{len(samples)}] {name}: scale={scale:.1f}x")

    k = float(np.median(per_sample_scales))
    print(f"\nCalibration constant k = {k:.2f}x (from {len(per_sample_scales)} samples)")
    print(f"  Mean: {np.mean(per_sample_scales):.2f}, Std: {np.std(per_sample_scales):.2f}")

    out_path = PROJECT_ROOT / "output" / "evaluation" / "posthoc_calibration_k.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"k": k, "n_samples": len(per_sample_scales),
                   "mean": float(np.mean(per_sample_scales)),
                   "std": float(np.std(per_sample_scales)),
                   "per_sample_scales": per_sample_scales}, f, indent=2)
    print(f"Saved: {out_path}")
    return k


def evaluate_posthoc(k):
    """Apply post-hoc calibration to existing per-sample results and compare."""
    print(f"\n{'='*80}")
    print(f"POST-HOC CALIBRATION: divide MoGe-2 predictions by k={k:.2f}")
    print(f"{'='*80}")

    # Load baseline MoGe-2 per-sample results
    for ds in ['skinl2', 'woundsdb']:
        res_path = PROJECT_ROOT / "output" / "evaluation" / "exp_a_step0_baseline" / ds / "results.json"
        with open(res_path) as f:
            data = json.load(f)

        per_sample = data['per_sample']

        # Post-hoc correction: new_scale = old_scale / k
        corrected_scales = []
        corrected_absrels = []
        version_data = {'v1': [], 'v2': [], 'v3': []}

        for s in per_sample:
            if 'scale_ratio' not in s:
                continue
            old_scale = s['scale_ratio']
            new_scale = old_scale / k

            # For SKINL2 (SI-d1=100%), pred ≈ scale * gt, so:
            # absrel_new = |new_scale - 1| (exact when SI-d1=100%)
            # For WoundsDB (SI-d1=91%), this is approximate
            new_absrel = abs(new_scale - 1.0)

            corrected_scales.append(new_scale)
            corrected_absrels.append(new_absrel)

            # Per-version for SKINL2
            name = s.get('sample', '')
            for v in ['v1', 'v2', 'v3']:
                if name.startswith(v + '_'):
                    version_data[v].append(new_scale)

        mean_scale = np.mean(corrected_scales)
        mean_absrel = np.mean(corrected_absrels)

        print(f"\n  {ds.upper()} (n={len(corrected_scales)}):")
        print(f"    Scale: {mean_scale:.2f}x (target=1.0)")
        print(f"    AbsRel: {mean_absrel:.2f}")

        if ds == 'skinl2':
            for v in ['v1', 'v2', 'v3']:
                if version_data[v]:
                    print(f"    {v}: scale={np.mean(version_data[v]):.2f}x (n={len(version_data[v])})")

    # Now load DermDepth results for comparison
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'SK Scale':>9} {'SK AbsRel':>10} {'WDB Scale':>10} {'WDB AbsRel':>11}")
    print("-" * 72)

    # Baselines
    for name, m in [("DA3-Nested", 'da3nested'), ("MapAnything", 'mapanything'), ("PPD", 'ppd')]:
        with open(PROJECT_ROOT / f'output/evaluation/baselines/{m}/results_summary.json') as f:
            b = json.load(f)
        print(f"{name:<30} {b['skinl2']['scale_ratio']:>9.2f} {b['skinl2']['abs_rel']:>10.2f} {b['woundsdb']['scale_ratio']:>10.2f} {b['woundsdb']['abs_rel']:>11.2f}")

    # MoGe-2 baseline
    with open(PROJECT_ROOT / "output/evaluation/exp_a_step0_baseline/skinl2/results.json") as f:
        base_sk = json.load(f)['summary']
    with open(PROJECT_ROOT / "output/evaluation/exp_a_step0_baseline/woundsdb/results.json") as f:
        base_wdb = json.load(f)['summary']
    print(f"{'MoGe-2 (raw)':<30} {base_sk['scale_ratio']['mean']:>9.2f} {base_sk['absrel']['mean']:>10.2f} {base_wdb['scale_ratio']['mean']:>10.2f} {base_wdb['absrel']['mean']:>11.2f}")

    # Post-hoc calibrated
    # Recompute for the table
    for ds_name, ds_key in [('skinl2', 'skinl2'), ('woundsdb', 'woundsdb')]:
        res_path = PROJECT_ROOT / "output" / "evaluation" / "exp_a_step0_baseline" / ds_key / "results.json"
        with open(res_path) as f:
            data = json.load(f)
        scales = [s['scale_ratio'] / k for s in data['per_sample'] if 'scale_ratio' in s]
        absrels = [abs(s['scale_ratio'] / k - 1.0) for s in data['per_sample'] if 'scale_ratio' in s]
        if ds_key == 'skinl2':
            ph_sk_scale, ph_sk_absrel = np.mean(scales), np.mean(absrels)
        else:
            ph_wdb_scale, ph_wdb_absrel = np.mean(scales), np.mean(absrels)

    print(f"{'MoGe-2 + post-hoc (k=' + f'{k:.1f}' + ')':<30} {ph_sk_scale:>9.2f} {ph_sk_absrel:>10.2f} {ph_wdb_scale:>10.2f} {ph_wdb_absrel:>11.2f}")

    # DermDepth
    with open(PROJECT_ROOT / "output/evaluation/exp_a_step1000_ema/skinl2/results.json") as f:
        dd_sk = json.load(f)['summary']
    with open(PROJECT_ROOT / "output/evaluation/exp_a_step1000_ema/woundsdb/results.json") as f:
        dd_wdb = json.load(f)['summary']
    print(f"{'DermDepth (learned)':<30} {dd_sk['scale_ratio']['mean']:>9.2f} {dd_sk['absrel']['mean']:>10.2f} {dd_wdb['scale_ratio']['mean']:>10.2f} {dd_wdb['absrel']['mean']:>11.2f}")

    # Per-version comparison
    print(f"\n{'='*80}")
    print("PER-VERSION COMPARISON (SKINL2)")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'v1 (20cm)':>10} {'v2 (30cm)':>10} {'v3 (50cm)':>10}")
    print("-" * 62)

    # MoGe-2 raw
    with open(PROJECT_ROOT / "output/evaluation/exp_a_step0_baseline/skinl2/results.json") as f:
        base_data = json.load(f)
    versions_raw = {'v1': [], 'v2': [], 'v3': []}
    versions_ph = {'v1': [], 'v2': [], 'v3': []}
    for s in base_data['per_sample']:
        if 'scale_ratio' not in s:
            continue
        name = s.get('sample', '')
        for v in ['v1', 'v2', 'v3']:
            if name.startswith(v + '_'):
                versions_raw[v].append(s['scale_ratio'])
                versions_ph[v].append(s['scale_ratio'] / k)

    print(f"{'MoGe-2 (raw)':<30}", end="")
    for v in ['v1', 'v2', 'v3']:
        print(f" {np.mean(versions_raw[v]):>9.2f}x", end="")
    print()

    print(f"{'MoGe-2 + post-hoc':<30}", end="")
    for v in ['v1', 'v2', 'v3']:
        print(f" {np.mean(versions_ph[v]):>9.2f}x", end="")
    print()

    # DermDepth per-version
    with open(PROJECT_ROOT / "output/evaluation/exp_a_step1000_ema/skinl2/results.json") as f:
        dd_data = json.load(f)
    versions_dd = {'v1': [], 'v2': [], 'v3': []}
    for s in dd_data['per_sample']:
        if 'scale_ratio' not in s:
            continue
        name = s.get('sample', '')
        for v in ['v1', 'v2', 'v3']:
            if name.startswith(v + '_'):
                versions_dd[v].append(s['scale_ratio'])

    print(f"{'DermDepth (learned)':<30}", end="")
    for v in ['v1', 'v2', 'v3']:
        print(f" {np.mean(versions_dd[v]):>9.2f}x", end="")
    print()
    print(f"\n  Target = 1.0x for all versions")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_k', action='store_true', help='Run MoGe-2 on synthetics to compute k')
    parser.add_argument('--k', type=float, default=None, help='Use this k value directly')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_samples', type=int, default=200)
    args = parser.parse_args()

    if args.compute_k:
        k = compute_calibration_constant(args.n_samples, args.device)
    elif args.k:
        k = args.k
    else:
        # Try to load saved k
        k_path = PROJECT_ROOT / "output" / "evaluation" / "posthoc_calibration_k.json"
        if k_path.exists():
            with open(k_path) as f:
                k = json.load(f)['k']
            print(f"Loaded k={k:.2f} from {k_path}")
        else:
            print("No k available. Run with --compute_k first.")
            sys.exit(1)

    evaluate_posthoc(k)
