#!/usr/bin/env python3
"""
Test MoGe-2 pretrained baseline on SKINL2 cases.
Runs inference on CPU, computes metric-scale and scale-invariant metrics.
Focus: metric scale error (main paper contribution).

Usage:
    python test_moge_skinl2.py [--device cpu]
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter, zoom
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))

DATA_ROOT = PROJECT_ROOT / "data" / "SKINL2"
OUTPUT_DIR = PROJECT_ROOT / "output" / "verification"

random.seed(42)


def compute_metrics(pred, gt, mask):
    """Compute depth metrics. pred and gt in meters."""
    p = pred[mask]
    g = gt[mask]

    valid = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
    p, g = p[valid], g[valid]

    if len(p) < 10:
        return {'valid_pixels': len(p)}

    # Metric-scale (no alignment)
    absrel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    scale_ratio = float(np.median(p / g))

    # Scale-invariant (least-squares alignment)
    scale = np.sum(g * p) / np.sum(p * p)
    p_aligned = p * scale
    si_absrel = float(np.mean(np.abs(p_aligned - g) / g))
    si_rmse = float(np.sqrt(np.mean((p_aligned - g) ** 2)))
    ratio_si = np.maximum(p_aligned / g, g / p_aligned)
    si_delta1 = float(np.mean(ratio_si < 1.25))

    return {
        'valid_pixels': int(len(p)),
        'absrel': absrel,
        'rmse_m': rmse,
        'rmse_mm': rmse * 1000,
        'delta1': delta1,
        'scale_ratio': scale_ratio,
        'scale_error_pct': abs(scale_ratio - 1.0) * 100,
        'pred_mean_m': float(np.mean(p)),
        'gt_mean_m': float(np.mean(g)),
        'pred_mean_mm': float(np.mean(p)) * 1000,
        'gt_mean_mm': float(np.mean(g)) * 1000,
        'optimal_scale': float(scale),
        'si_absrel': si_absrel,
        'si_rmse_mm': si_rmse * 1000,
        'si_delta1': si_delta1,
    }


def discover_v1():
    cases = []
    cv_root = DATA_ROOT / 'SKINL2_v1' / 'Central View'
    dm_root = DATA_ROOT / 'SKINL2_v1' / 'DepthMap'
    for cat in sorted(os.listdir(cv_root)):
        cat_cv = cv_root / cat
        cat_dm = dm_root / cat
        if not cat_cv.is_dir() or not cat_dm.is_dir():
            continue
        for sid in sorted(os.listdir(cat_cv)):
            cv_files = list((cat_cv / sid).glob('*.png'))
            dm_files = list((cat_dm / sid).glob('*.tiff'))
            if cv_files and dm_files:
                cases.append(('v1', cat, sid, cv_files[0], dm_files[0]))
    return cases


def discover_v2v3(version):
    cases = []
    root = DATA_ROOT / f'SKINL2_{version}'
    for case_id in sorted(os.listdir(root)):
        case_dir = root / case_id
        if not case_dir.is_dir():
            continue
        for cat in os.listdir(case_dir):
            cv_dir = case_dir / cat / 'Light Field' / 'Central View'
            dm_dir = case_dir / cat / 'Light Field' / 'Depth Map'
            if cv_dir.is_dir() and dm_dir.is_dir():
                cv_files = list(cv_dir.glob('*TotalFocus*.png'))
                dm_files = list(dm_dir.glob('*DepthMap.tiff'))
                if cv_files and dm_files:
                    cases.append((version, cat, case_id, cv_files[0], dm_files[0]))
    return cases


def select_diverse(cases_list, n=5):
    by_cat = {}
    for c in cases_list:
        by_cat.setdefault(c[1], []).append(c)
    selected = []
    for cat in sorted(by_cat.keys()):
        if len(selected) >= n:
            break
        selected.append(random.choice(by_cat[cat]))
    remaining = [c for c in cases_list if c not in selected]
    while len(selected) < n and remaining:
        choice = random.choice(remaining)
        selected.append(choice)
        remaining.remove(choice)
    return selected[:n]


def main():
    device = 'cpu'
    if len(sys.argv) > 1 and sys.argv[1] == '--device':
        device = sys.argv[2]

    # Discover and select same cases as visualization
    print("Discovering cases...")
    v1_cases = discover_v1()
    v2_cases = discover_v2v3('v2')
    v3_cases = discover_v2v3('v3')
    print(f"  v1: {len(v1_cases)}, v2: {len(v2_cases)}, v3: {len(v3_cases)}")

    sel_v1 = select_diverse(v1_cases, 5)
    sel_v2 = select_diverse(v2_cases, 5)
    sel_v3 = select_diverse(v3_cases, 5)
    all_selected = sel_v1 + sel_v2 + sel_v3

    print(f"\nSelected {len(all_selected)} cases:")
    for ver, cat, sid, _, _ in all_selected:
        print(f"  {ver}/{cat}/{sid}")

    # Load model
    print(f"\nLoading MoGe-2 on {device}...")
    from moge.model.v2 import MoGeModel
    import torchvision.transforms.functional as TF
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    model = model.to(device).eval()
    print("Model loaded.")

    # Process each case
    all_metrics = []

    for i, (ver, cat, sid, cv_path, dm_path) in enumerate(all_selected):
        label = f"{ver}/{cat}/{sid}"
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(all_selected)}] {label}")
        print(f"{'='*60}")

        # Load image and depth
        cv_img = np.array(Image.open(cv_path).convert('RGB'))
        depth_raw = np.array(Image.open(dm_path), dtype=np.float32)

        # GT depth: absolute value in meters (raw is negative mm)
        gt_depth_mm = np.abs(depth_raw)
        gt_depth_m = gt_depth_mm / 1000.0  # Convert mm to meters

        # Gaussian σ=15 smoothing on raw depth
        gt_depth_smooth_mm = np.abs(gaussian_filter(depth_raw, sigma=15))
        gt_depth_smooth_m = gt_depth_smooth_mm / 1000.0

        # Valid mask: where depth is reasonable (100-300mm = 0.1-0.3m)
        gt_mask = (gt_depth_m > 0.05) & (gt_depth_m < 0.5)

        print(f"  Image: {cv_img.shape}")
        print(f"  Depth: {depth_raw.shape}, raw range: [{depth_raw.min():.1f}, {depth_raw.max():.1f}]mm")
        print(f"  GT depth (abs): mean={gt_depth_mm[gt_mask].mean():.1f}mm, "
              f"range=[{gt_depth_mm[gt_mask].min():.1f}, {gt_depth_mm[gt_mask].max():.1f}]mm")
        print(f"  Valid pixels: {gt_mask.sum()} / {gt_mask.size} ({gt_mask.sum()/gt_mask.size*100:.1f}%)")

        # Run MoGe inference on central view
        img_pil = Image.open(cv_path).convert('RGB')
        img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model.infer(img_tensor)

        pred_depth = output['depth'].squeeze(0).cpu().numpy() \
            if output['depth'].dim() > 2 else output['depth'].cpu().numpy()

        pred_valid = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
        print(f"  MoGe prediction: {pred_depth.shape}, "
              f"range=[{pred_valid.min():.3f}, {pred_valid.max():.3f}]m, "
              f"mean={pred_valid.mean():.3f}m")

        # Resize prediction to match GT depth resolution
        if pred_depth.shape != gt_depth_m.shape:
            scale_h = gt_depth_m.shape[0] / pred_depth.shape[0]
            scale_w = gt_depth_m.shape[1] / pred_depth.shape[1]
            pred_depth = zoom(pred_depth, (scale_h, scale_w), order=1)
            print(f"  Resized prediction to {pred_depth.shape}")

        # Compute metrics
        metrics = compute_metrics(pred_depth, gt_depth_smooth_m, gt_mask)
        metrics['version'] = ver
        metrics['category'] = cat
        metrics['sample_id'] = sid
        all_metrics.append(metrics)

        if 'absrel' in metrics:
            print(f"  --- Metric-Scale ---")
            print(f"  AbsRel:      {metrics['absrel']:.4f}")
            print(f"  Scale Ratio: {metrics['scale_ratio']:.3f}  "
                  f"(pred {metrics['pred_mean_mm']:.0f}mm vs GT {metrics['gt_mean_mm']:.0f}mm)")
            print(f"  Scale Error: {metrics['scale_error_pct']:.1f}%")
            print(f"  RMSE:        {metrics['rmse_mm']:.1f}mm")
            print(f"  Delta1:      {metrics['delta1']:.4f}")
            print(f"  --- Scale-Invariant ---")
            print(f"  SI-AbsRel:   {metrics['si_absrel']:.4f}")
            print(f"  SI-Delta1:   {metrics['si_delta1']:.4f}")
            print(f"  SI-RMSE:     {metrics['si_rmse_mm']:.1f}mm")

    # ============ Summary table ============
    print(f"\n{'='*80}")
    print("SUMMARY: MoGe-2 Baseline on SKINL2")
    print(f"{'='*80}")

    valid_metrics = [m for m in all_metrics if 'absrel' in m]

    print(f"\n{'Ver':<4} {'Category':<22} {'ID':<6} {'GT(mm)':<8} {'Pred(mm)':<9} "
          f"{'Scale':<7} {'ScaleErr':<9} {'AbsRel':<8} {'SI-d1':<7}")
    print("-" * 80)

    for m in valid_metrics:
        print(f"{m['version']:<4} {m['category']:<22} {m['sample_id']:<6} "
              f"{m['gt_mean_mm']:<8.0f} {m['pred_mean_mm']:<9.0f} "
              f"{m['scale_ratio']:<7.3f} {m['scale_error_pct']:<9.1f}% "
              f"{m['absrel']:<8.4f} {m['si_delta1']:<7.4f}")

    # Aggregates
    if valid_metrics:
        print("-" * 80)
        mean_scale = np.mean([m['scale_ratio'] for m in valid_metrics])
        mean_scale_err = np.mean([m['scale_error_pct'] for m in valid_metrics])
        mean_absrel = np.mean([m['absrel'] for m in valid_metrics])
        mean_si_d1 = np.mean([m['si_delta1'] for m in valid_metrics])
        mean_gt = np.mean([m['gt_mean_mm'] for m in valid_metrics])
        mean_pred = np.mean([m['pred_mean_mm'] for m in valid_metrics])
        print(f"{'MEAN':<4} {'':<22} {'':<6} "
              f"{mean_gt:<8.0f} {mean_pred:<9.0f} "
              f"{mean_scale:<7.3f} {mean_scale_err:<9.1f}% "
              f"{mean_absrel:<8.4f} {mean_si_d1:<7.4f}")

    # ============ Visualization ============
    print("\nCreating visualization...")

    n_cases = len(valid_metrics)
    fig, axes = plt.subplots(n_cases, 4, figsize=(24, 4 * n_cases))
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    for row, m in enumerate(valid_metrics):
        ver, cat, sid = m['version'], m['category'], m['sample_id']
        case_data = [(v, c, s, cp, dp) for v, c, s, cp, dp in all_selected
                     if v == ver and c == cat and s == sid][0]
        _, _, _, cv_path, dm_path = case_data

        cv_img = np.array(Image.open(cv_path).convert('RGB'))
        depth_raw = np.array(Image.open(dm_path), dtype=np.float32)
        gt_m = np.abs(gaussian_filter(depth_raw, sigma=15)) / 1000.0
        gt_mask = (gt_m > 0.05) & (gt_m < 0.5)

        # Re-run inference for visualization
        img_tensor = TF.to_tensor(Image.open(cv_path).convert('RGB')).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model.infer(img_tensor)
        pred = output['depth'].squeeze(0).cpu().numpy() \
            if output['depth'].dim() > 2 else output['depth'].cpu().numpy()
        if pred.shape != gt_m.shape:
            pred = zoom(pred, (gt_m.shape[0]/pred.shape[0], gt_m.shape[1]/pred.shape[1]), order=1)

        # Resize image for display
        cv_small = np.array(Image.fromarray(cv_img).resize(
            (gt_m.shape[1], gt_m.shape[0]), Image.LANCZOS))

        # Col 0: RGB with label
        axes[row, 0].imshow(cv_small)
        axes[row, 0].set_title(f'{ver}/{cat}/{sid}', fontsize=9, fontweight='bold')
        axes[row, 0].axis('off')

        # Col 1: GT depth (smoothed)
        gt_vis = gt_m.copy()
        gt_vis[~gt_mask] = np.nan
        vmin_gt, vmax_gt = np.nanpercentile(gt_vis, [2, 98])
        im = axes[row, 1].imshow(gt_vis, cmap='turbo', vmin=vmin_gt, vmax=vmax_gt)
        axes[row, 1].set_title(f'GT: {m["gt_mean_mm"]:.0f}mm', fontsize=9)
        axes[row, 1].axis('off')

        # Col 2: MoGe predicted depth
        pred_vis = pred.copy()
        pred_vis[pred <= 0] = np.nan
        vmin_p, vmax_p = np.nanpercentile(pred_vis, [2, 98])
        im = axes[row, 2].imshow(pred_vis, cmap='turbo', vmin=vmin_p, vmax=vmax_p)
        axes[row, 2].set_title(f'MoGe: {m["pred_mean_mm"]:.0f}mm', fontsize=9)
        axes[row, 2].axis('off')

        # Col 3: Metrics text
        axes[row, 3].axis('off')
        color = 'red' if m['scale_error_pct'] > 100 else 'orange' if m['scale_error_pct'] > 50 else 'green'
        txt = (f"Scale Ratio: {m['scale_ratio']:.2f}\n"
               f"Scale Error: {m['scale_error_pct']:.0f}%\n"
               f"GT: {m['gt_mean_mm']:.0f}mm → Pred: {m['pred_mean_mm']:.0f}mm\n"
               f"AbsRel: {m['absrel']:.3f}\n"
               f"SI-AbsRel: {m['si_absrel']:.3f}\n"
               f"SI-Delta1: {m['si_delta1']:.3f}")
        axes[row, 3].text(0.05, 0.5, txt, transform=axes[row, 3].transAxes,
                          fontsize=10, verticalalignment='center', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=color, linewidth=2))

    plt.suptitle('MoGe-2 Pretrained Baseline on SKINL2\n'
                 'Key metric: Scale Ratio (1.0=perfect, >1=overestimates distance)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = OUTPUT_DIR / 'fig38_moge_skinl2_baseline.png'
    plt.savefig(fig_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")

    # Save metrics JSON
    metrics_path = OUTPUT_DIR / 'moge_skinl2_baseline_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
