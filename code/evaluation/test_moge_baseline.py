#!/usr/bin/env python3
"""
Test MoGe-2 pretrained baseline on a single WoundsDB scene.

Runs inference on CPU, computes metrics against sparse ToF GT,
and produces visualization comparing prediction vs ground truth.

Usage:
    python test_moge_baseline.py [--scene SCENE_NAME] [--device cpu]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))

DEFAULT_SCENE = "case_1_day_1_scene_1"
EVAL_DATA_DIR = PROJECT_ROOT / "output" / "eval_data" / "woundsdb"
OUTPUT_DIR = PROJECT_ROOT / "output" / "verification"


def compute_metrics(pred, gt, mask):
    """Compute depth metrics at sparse GT locations."""
    p = pred[mask]
    g = gt[mask]

    valid = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
    p, g = p[valid], g[valid]

    if len(p) < 10:
        return {'valid_pixels': len(p)}

    # Metric-scale
    absrel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))
    scale_ratio = float(np.median(p / g))

    # Scale-invariant
    scale = np.sum(g * p) / np.sum(p * p)
    p_aligned = p * scale
    si_absrel = float(np.mean(np.abs(p_aligned - g) / g))
    si_rmse = float(np.sqrt(np.mean((p_aligned - g) ** 2)))
    ratio_si = np.maximum(p_aligned / g, g / p_aligned)
    si_delta1 = float(np.mean(ratio_si < 1.25))

    return {
        'valid_pixels': int(len(p)),
        'absrel': absrel,
        'rmse': rmse,
        'rmse_mm': rmse * 1000,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3,
        'scale_ratio': scale_ratio,
        'scale_error_pct': abs(scale_ratio - 1.0) * 100,
        'pred_mean': float(np.mean(p)),
        'gt_mean': float(np.mean(g)),
        'optimal_scale': float(scale),
        'si_absrel': si_absrel,
        'si_rmse': si_rmse,
        'si_rmse_mm': si_rmse * 1000,
        'si_delta1': si_delta1,
    }


def visualize_results(image, pred_depth, gt_depth, gt_mask, metrics, scene_name, output_path):
    """Create comprehensive visualization of prediction vs GT."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Input image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Input: {scene_name}\n(320x240 photo)")
    axes[0, 0].axis('off')

    # 2. Predicted depth (full dense)
    pred_valid = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
    vmin = pred_valid.min() if len(pred_valid) > 0 else 0
    vmax = pred_valid.max() if len(pred_valid) > 0 else 1
    im_pred = axes[0, 1].imshow(pred_depth, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"MoGe-2 Predicted Depth\nRange: {vmin:.2f} - {vmax:.2f}m")
    axes[0, 1].axis('off')
    plt.colorbar(im_pred, ax=axes[0, 1], shrink=0.8, label='Depth (m)')

    # 3. GT sparse depth overlay on image
    gt_vis = np.array(image, dtype=np.float32) / 255.0
    gt_valid_depths = gt_depth[gt_mask]
    gt_vmin, gt_vmax = gt_valid_depths.min(), gt_valid_depths.max()
    cmap = plt.cm.turbo
    norm = Normalize(vmin=gt_vmin, vmax=gt_vmax)
    gt_colored = np.zeros((*gt_depth.shape, 4), dtype=np.float32)
    gt_colored[gt_mask] = cmap(norm(gt_valid_depths))

    axes[0, 2].imshow(gt_vis * 0.4 + 0.6 * np.ones_like(gt_vis))  # Dimmed background
    axes[0, 2].imshow(gt_colored)
    axes[0, 2].set_title(f"ToF GT (sparse, {gt_mask.sum()} pts)\nRange: {gt_vmin:.2f} - {gt_vmax:.2f}m")
    axes[0, 2].axis('off')

    # 4. Predicted depth at GT locations (same colorscale as GT)
    pred_at_gt = np.full_like(gt_depth, np.nan)
    pred_at_gt[gt_mask] = pred_depth[gt_mask]
    pred_colored = np.zeros((*gt_depth.shape, 4), dtype=np.float32)
    pred_at_gt_valid = pred_depth[gt_mask]
    pred_colored[gt_mask] = cmap(norm(np.clip(pred_at_gt_valid, gt_vmin, gt_vmax)))

    axes[1, 0].imshow(gt_vis * 0.4 + 0.6 * np.ones_like(gt_vis))
    axes[1, 0].imshow(pred_colored)
    axes[1, 0].set_title(f"Prediction at GT locations\n(same colorscale as GT)")
    axes[1, 0].axis('off')

    # 5. Error map at GT locations
    error = np.full_like(gt_depth, np.nan)
    error[gt_mask] = pred_depth[gt_mask] - gt_depth[gt_mask]
    error_abs = np.abs(error)
    error_at_gt = error_abs[gt_mask]
    error_colored = np.zeros((*gt_depth.shape, 4), dtype=np.float32)
    e_norm = Normalize(vmin=0, vmax=np.percentile(error_at_gt, 95))
    error_colored[gt_mask] = plt.cm.hot(e_norm(error_at_gt))

    axes[1, 1].imshow(gt_vis * 0.4 + 0.6 * np.ones_like(gt_vis))
    axes[1, 1].imshow(error_colored)
    axes[1, 1].set_title(f"Absolute Error\nMedian: {np.median(error_at_gt)*1000:.0f}mm, "
                         f"P95: {np.percentile(error_at_gt, 95)*1000:.0f}mm")
    axes[1, 1].axis('off')

    # 6. Metrics text
    axes[1, 2].axis('off')
    text_lines = [
        f"Scene: {scene_name}",
        f"",
        f"--- Metric-Scale (no alignment) ---",
        f"AbsRel:       {metrics['absrel']:.4f}",
        f"RMSE:         {metrics['rmse_mm']:.1f} mm",
        f"Delta1:       {metrics['delta1']:.4f}",
        f"Delta2:       {metrics['delta2']:.4f}",
        f"Delta3:       {metrics['delta3']:.4f}",
        f"Scale Ratio:  {metrics['scale_ratio']:.3f}",
        f"Scale Error:  {metrics['scale_error_pct']:.1f}%",
        f"",
        f"--- Scale-Invariant ---",
        f"SI-AbsRel:    {metrics['si_absrel']:.4f}",
        f"SI-RMSE:      {metrics['si_rmse_mm']:.1f} mm",
        f"SI-Delta1:    {metrics['si_delta1']:.4f}",
        f"",
        f"--- Statistics ---",
        f"Valid pixels: {metrics['valid_pixels']}",
        f"GT mean:      {metrics['gt_mean']:.3f} m",
        f"Pred mean:    {metrics['pred_mean']:.3f} m",
        f"Optimal scale:{metrics['optimal_scale']:.3f}",
    ]
    axes[1, 2].text(0.05, 0.95, '\n'.join(text_lines),
                    transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f"MoGe-2 Pretrained Baseline vs ToF Ground Truth", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test MoGe-2 baseline on single WoundsDB scene")
    parser.add_argument('--scene', type=str, default=DEFAULT_SCENE,
                        help='Scene name (default: case_1_day_1_scene_1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (default: cpu)')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    scene_dir = EVAL_DATA_DIR / args.scene
    if not scene_dir.exists():
        print(f"Scene not found: {scene_dir}")
        return

    print("=" * 60)
    print("MoGe-2 Pretrained Baseline Test")
    print(f"  Scene: {args.scene}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # Load prepared data
    print("\n1. Loading prepared data...")
    image = np.array(Image.open(scene_dir / "image.png").convert('RGB'))
    gt_depth = np.load(scene_dir / "gt_depth.npy")
    gt_mask = np.load(scene_dir / "gt_mask.npy")
    meta = json.load(open(scene_dir / "meta.json"))

    print(f"  Image: {image.shape}")
    print(f"  GT depth: {gt_depth.shape}, {gt_mask.sum()} valid pixels ({meta['coverage_pct']:.1f}%)")
    gt_valid = gt_depth[gt_mask]
    print(f"  GT range: {gt_valid.min():.3f} - {gt_valid.max():.3f}m (mean {gt_valid.mean():.3f}m)")

    # Load model
    print(f"\n2. Loading MoGe-2 pretrained model on {args.device}...")
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    model = model.to(args.device).eval()
    print("  Model loaded.")

    # Run inference
    print("\n3. Running inference...")
    import torchvision.transforms.functional as TF
    img_pil = Image.open(scene_dir / "image.png").convert('RGB')
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(args.device)
    print(f"  Input tensor: {img_tensor.shape}, range [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

    with torch.inference_mode():
        output = model.infer(img_tensor)

    pred_depth = output['depth'].squeeze(0).cpu().numpy() if output['depth'].dim() > 2 else output['depth'].cpu().numpy()
    pred_mask = output.get('mask', None)
    if pred_mask is not None:
        pred_mask = pred_mask.squeeze(0).cpu().numpy() if pred_mask.dim() > 2 else pred_mask.cpu().numpy()

    print(f"  Prediction shape: {pred_depth.shape}")
    pred_valid = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
    print(f"  Predicted range: {pred_valid.min():.3f} - {pred_valid.max():.3f}m (mean {pred_valid.mean():.3f}m)")

    # Check if we need to resize prediction
    if pred_depth.shape != gt_depth.shape:
        print(f"  Resizing prediction from {pred_depth.shape} to {gt_depth.shape}...")
        from scipy.ndimage import zoom
        scale_h = gt_depth.shape[0] / pred_depth.shape[0]
        scale_w = gt_depth.shape[1] / pred_depth.shape[1]
        pred_depth = zoom(pred_depth, (scale_h, scale_w), order=1)

    # Compute metrics
    print("\n4. Computing metrics at sparse GT locations...")
    metrics = compute_metrics(pred_depth, gt_depth, gt_mask)

    print(f"\n  === Results ===")
    print(f"  Valid pixels:   {metrics['valid_pixels']}")
    print(f"  AbsRel:         {metrics['absrel']:.4f}")
    print(f"  RMSE:           {metrics['rmse_mm']:.1f} mm")
    print(f"  Delta1:         {metrics['delta1']:.4f}")
    print(f"  Scale Ratio:    {metrics['scale_ratio']:.3f}  (1.0 = perfect)")
    print(f"  Scale Error:    {metrics['scale_error_pct']:.1f}%")
    print(f"  SI-AbsRel:      {metrics['si_absrel']:.4f}")
    print(f"  SI-Delta1:      {metrics['si_delta1']:.4f}")
    print(f"  Pred mean:      {metrics['pred_mean']:.3f}m")
    print(f"  GT mean:        {metrics['gt_mean']:.3f}m")
    print(f"  Optimal scale:  {metrics['optimal_scale']:.3f}")

    # Visualize
    print("\n5. Creating visualization...")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"fig16_moge_baseline_{args.scene}.png")
    visualize_results(image, pred_depth, gt_depth, gt_mask, metrics, args.scene, output_path)

    # Save metrics as JSON
    metrics_path = os.path.join(args.output_dir, f"moge_baseline_metrics_{args.scene}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
