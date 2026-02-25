#!/usr/bin/env python3
"""Generate point cloud & depth comparison figures: Baseline MoGe-2 vs DermDepth.

Uses prepared eval data from output/eval_data/{skinl2,woundsdb}/.
Each sample has: image.png, gt_depth.npy, gt_mask.npy, meta.json.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../MoGe'))

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json


OUT_DIR = Path("output/figures/pointclouds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SKINL2_DIR = Path("output/eval_data/skinl2")
WOUNDSDB_DIR = Path("output/eval_data/woundsdb")


def load_model(checkpoint_path, device='cuda'):
    """Load MoGe model from checkpoint or HuggingFace."""
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model_config = checkpoint.get('model_config', None)
        if model_config:
            model = MoGeModel(**model_config)
        else:
            model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model = MoGeModel.from_pretrained(checkpoint_path)
    return model.to(device).eval()


def infer_image(model, image_path, device='cuda'):
    """Run inference on a single image, return points + depth + mask + colors."""
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w = img.shape[:2]
    image_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

    with torch.no_grad():
        output = model.infer(image_tensor, resolution_level=9, force_projection=True)

    points = output['points'].cpu().numpy()  # (H, W, 3) in meters
    depth = output['depth'].cpu().numpy()     # (H, W) in meters
    mask = output['mask'].cpu().numpy() > 0.5  # (H, W) bool
    colors = img / 255.0

    return points, depth, mask, colors


def pick_skinl2_samples(n=3):
    """Pick diverse SKINL2 samples (one per version, different diseases)."""
    # Preferred samples for visual diversity
    preferred = [
        'v1_Nevus_0004',
        'v2_Melanoma_0003',
        'v3_Dermatofibroma_0026',
    ]
    selected = []
    for name in preferred:
        d = SKINL2_DIR / name
        if d.is_dir() and (d / 'image.png').exists():
            selected.append(d)
        if len(selected) >= n:
            return selected

    # Fallback: one per version, prefer non-BCC
    samples_by_version = {}
    for d in sorted(SKINL2_DIR.iterdir()):
        if not d.is_dir() or d in selected:
            continue
        meta_path = d / 'meta.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        ver = meta.get('version', 'v1')
        disease = meta.get('disease', '')
        # Prefer non-BCC for variety
        if ver not in samples_by_version or 'Basal' in samples_by_version[ver][1]:
            samples_by_version[ver] = (d, disease)

    for v in ['v1', 'v2', 'v3']:
        if v in samples_by_version and len(selected) < n:
            selected.append(samples_by_version[v][0])
    return selected


def pick_woundsdb_samples(n=2):
    """Pick WoundsDB samples with high GT coverage."""
    candidates = []
    for d in sorted(WOUNDSDB_DIR.iterdir()):
        if not d.is_dir():
            continue
        mask_path = d / 'gt_mask.npy'
        img_path = d / 'image.png'
        if not mask_path.exists() or not img_path.exists():
            continue
        mask = np.load(mask_path)
        coverage = mask.sum() / mask.size
        if coverage > 0.8:
            candidates.append((d, coverage))

    # Pick highest coverage, diverse cases
    candidates.sort(key=lambda x: -x[1])
    seen_cases = set()
    selected = []
    for d, cov in candidates:
        case_id = '_'.join(d.name.split('_')[:2])  # e.g. case_12
        if case_id not in seen_cases:
            seen_cases.add(case_id)
            selected.append(d)
        if len(selected) >= n:
            break
    return selected


def make_depth_comparison(baseline_model, dermdepth_model, device='cuda'):
    """Main figure: side-by-side depth maps for SKINL2 + WoundsDB.

    Layout per row: [Input Image | GT Depth | Baseline Depth | DermDepth Depth]
    Each depth map uses its OWN range (showing geometry preservation),
    with median depth prominently annotated to show scale difference.
    """
    skinl2_samples = pick_skinl2_samples(2)
    woundsdb_samples = pick_woundsdb_samples(2)

    all_samples = []
    for d in skinl2_samples:
        meta = json.loads((d / 'meta.json').read_text())
        label = f"SKINL2 {meta.get('version', '?')} — {meta.get('disease', '?')}"
        all_samples.append((d, label, 'skinl2'))
    for d in woundsdb_samples:
        label = f"WoundsDB — {d.name}"
        all_samples.append((d, label, 'woundsdb'))

    n = len(all_samples)
    fig, axes = plt.subplots(n, 4, figsize=(7.16, 2.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (sample_dir, label, dataset) in enumerate(all_samples):
        img_path = sample_dir / 'image.png'
        gt_depth = np.load(sample_dir / 'gt_depth.npy')
        gt_mask = np.load(sample_dir / 'gt_mask.npy').astype(bool)

        print(f"  [{i+1}/{n}] {label}")

        # Inference
        _, dep_base, mask_base, colors = infer_image(baseline_model, str(img_path), device)
        _, dep_ours, mask_ours, _ = infer_image(dermdepth_model, str(img_path), device)

        # Resize predictions to GT shape if needed
        h_gt, w_gt = gt_depth.shape
        h_pred, w_pred = dep_base.shape
        if (h_gt, w_gt) != (h_pred, w_pred):
            from scipy.ndimage import zoom
            scale_h, scale_w = h_gt / h_pred, w_gt / w_pred
            dep_base = zoom(dep_base, (scale_h, scale_w), order=1)
            dep_ours = zoom(dep_ours, (scale_h, scale_w), order=1)
            mask_base = zoom(mask_base.astype(float), (scale_h, scale_w), order=0) > 0.5
            mask_ours = zoom(mask_ours.astype(float), (scale_h, scale_w), order=0) > 0.5

        # Combined mask for scale computation
        valid = gt_mask & mask_base & mask_ours & (gt_depth > 0) & np.isfinite(gt_depth)

        # Compute median depths (meters -> mm)
        med_gt = np.median(gt_depth[valid]) * 1000 if valid.any() else 0
        med_base = np.median(dep_base[valid]) * 1000 if valid.any() else 0
        med_ours = np.median(dep_ours[valid]) * 1000 if valid.any() else 0

        scale_base = med_base / med_gt if med_gt > 0 else float('inf')
        scale_ours = med_ours / med_gt if med_gt > 0 else float('inf')

        # Col 0: Input
        img = np.array(Image.open(img_path).convert('RGB'))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(label, fontsize=7, fontweight='bold')
        axes[i, 0].axis('off')

        # Helper: show depth map with own range + annotation
        def show_depth(ax, dep, mask, title_str, annotation_str, ann_color='black'):
            dep_mm = dep.copy() * 1000
            dep_mm[~mask] = np.nan
            valid_vals = dep_mm[mask & np.isfinite(dep_mm)]
            if len(valid_vals) > 0:
                vmin, vmax = np.percentile(valid_vals, [2, 98])
            else:
                vmin, vmax = 0, 1
            ax.imshow(dep_mm, cmap='turbo', vmin=vmin, vmax=vmax)
            ax.set_title(title_str, fontsize=7, fontweight='bold')
            ax.axis('off')
            ax.text(0.5, 0.03, annotation_str, transform=ax.transAxes,
                    fontsize=8, fontweight='bold', ha='center', va='bottom',
                    color='white',
                    bbox=dict(facecolor=ann_color, alpha=0.85, edgecolor='none',
                              pad=2, boxstyle='round,pad=0.3'))

        # Col 1: GT depth
        show_depth(axes[i, 1], gt_depth, gt_mask,
                   'Ground Truth', f'{med_gt:.0f} mm', '#2ca02c')

        # Col 2: Baseline (red annotation = wrong scale)
        show_depth(axes[i, 2], dep_base, mask_base,
                   'Baseline MoGe-2',
                   f'{med_base:.0f} mm ({scale_base:.1f}x)', '#d62728')

        # Col 3: DermDepth (blue annotation = corrected)
        show_depth(axes[i, 3], dep_ours, mask_ours,
                   'DermDepth (Ours)',
                   f'{med_ours:.0f} mm ({scale_ours:.1f}x)', '#1f77b4')

    plt.tight_layout(h_pad=0.8, w_pad=0.3)
    fig.savefig(OUT_DIR / 'depth_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'depth_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved depth_comparison.{{png,pdf}}")


def make_pointcloud_figure(baseline_model, dermdepth_model, device='cuda'):
    """Point cloud top-down view: shows spatial extent in mm.

    Layout per row: [Input | Baseline PC (top-down) | DermDepth PC (top-down) | Scale Bar]
    """
    skinl2_samples = pick_skinl2_samples(1)
    woundsdb_samples = pick_woundsdb_samples(1)

    all_samples = []
    for d in skinl2_samples:
        meta = json.loads((d / 'meta.json').read_text())
        label = f"SKINL2 {meta.get('version', '?')}"
        all_samples.append((d, label, 'skinl2'))
    for d in woundsdb_samples:
        label = f"WoundsDB"
        all_samples.append((d, label, 'woundsdb'))

    n = len(all_samples)
    fig, axes = plt.subplots(n, 4, figsize=(7.16, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (sample_dir, label, dataset) in enumerate(all_samples):
        img_path = sample_dir / 'image.png'
        gt_depth = np.load(sample_dir / 'gt_depth.npy')
        gt_mask = np.load(sample_dir / 'gt_mask.npy').astype(bool)

        print(f"  [{i+1}/{n}] {label}")

        pts_base, dep_base, mask_base, colors = infer_image(baseline_model, str(img_path), device)
        pts_ours, dep_ours, mask_ours, _ = infer_image(dermdepth_model, str(img_path), device)

        # Subsample
        stride = 4
        def sub(pts, cols, m):
            p = pts[::stride, ::stride]
            c = cols[::stride, ::stride]
            mm = m[::stride, ::stride]
            return p[mm].reshape(-1, 3), c[mm].reshape(-1, 3)

        pc_b, cc_b = sub(pts_base, colors, mask_base)
        pc_o, cc_o = sub(pts_ours, colors, mask_ours)

        # Median depths
        med_base = np.median(dep_base[mask_base]) * 1000
        med_ours = np.median(dep_ours[mask_ours]) * 1000
        med_gt = np.median(gt_depth[gt_mask]) * 1000 if gt_mask.any() else 0

        # Col 0: Input
        img = np.array(Image.open(img_path).convert('RGB'))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(label, fontsize=8, fontweight='bold')
        axes[i, 0].axis('off')

        # Col 1: Baseline top-down point cloud
        ax = axes[i, 1]
        x, y = pc_b[:, 0] * 1000, -pc_b[:, 1] * 1000  # mm, flip Y
        if len(x) > 25000:
            idx = np.random.choice(len(x), 25000, replace=False)
            x, y, cc_plot = x[idx], y[idx], cc_b[idx]
        else:
            cc_plot = cc_b
        ax.scatter(x, y, c=cc_plot, s=0.2, alpha=0.8, rasterized=True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)', fontsize=6)
        ax.set_ylabel('Y (mm)', fontsize=6)
        ax.set_title(f'Baseline ({med_base:.0f} mm depth)', fontsize=7, fontweight='bold')
        ax.tick_params(labelsize=5)

        # Col 2: DermDepth top-down point cloud
        ax = axes[i, 2]
        x, y = pc_o[:, 0] * 1000, -pc_o[:, 1] * 1000
        if len(x) > 25000:
            idx = np.random.choice(len(x), 25000, replace=False)
            x, y, cc_plot = x[idx], y[idx], cc_o[idx]
        else:
            cc_plot = cc_o
        ax.scatter(x, y, c=cc_plot, s=0.2, alpha=0.8, rasterized=True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)', fontsize=6)
        ax.set_ylabel('Y (mm)', fontsize=6)
        ax.set_title(f'DermDepth ({med_ours:.0f} mm depth)', fontsize=7, fontweight='bold')
        ax.tick_params(labelsize=5)

        # Col 3: Scale bar chart
        ax = axes[i, 3]
        labels_bar = ['GT', 'Baseline', 'DermDepth']
        values = [med_gt, med_base, med_ours]
        bar_colors = ['#2ca02c', '#d62728', '#1f77b4']
        bars = ax.barh(labels_bar, values, color=bar_colors, edgecolor='black', linewidth=0.5, height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.03, bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f} mm', va='center', fontsize=7, fontweight='bold')
        ax.set_xlabel('Median Depth (mm)', fontsize=7)
        ax.set_title('Scale Comparison', fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, max(values) * 1.3)

    plt.tight_layout(h_pad=0.8, w_pad=0.5)
    fig.savefig(OUT_DIR / 'pointcloud_topdown.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'pointcloud_topdown.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved pointcloud_topdown.{{png,pdf}}")


def make_scale_overview(baseline_model, dermdepth_model, device='cuda'):
    """Summary figure: one SKINL2 + one WoundsDB showing the scale problem.

    Layout: 2 rows x 3 cols = [Input | Baseline depth (own colorbar) | DermDepth depth (GT colorbar)]
    Clean, paper-ready with clear scale annotations.
    """
    skinl2_samples = pick_skinl2_samples(1)
    woundsdb_samples = pick_woundsdb_samples(1)

    samples = []
    for d in skinl2_samples:
        meta = json.loads((d / 'meta.json').read_text())
        samples.append((d, f"Dermatoscopic ({meta.get('disease', '?')})"))
    for d in woundsdb_samples:
        samples.append((d, "Clinical wound"))

    n = len(samples)
    fig = plt.figure(figsize=(7.16, 2.5 * n))
    gs = GridSpec(n, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.15, hspace=0.35)

    for i, (sample_dir, label) in enumerate(samples):
        img_path = sample_dir / 'image.png'
        gt_depth = np.load(sample_dir / 'gt_depth.npy')
        gt_mask = np.load(sample_dir / 'gt_mask.npy').astype(bool)

        print(f"  [{i+1}/{n}] {label}")

        _, dep_base, mask_base, colors = infer_image(baseline_model, str(img_path), device)
        _, dep_ours, mask_ours, _ = infer_image(dermdepth_model, str(img_path), device)

        # Resize to GT
        h_gt, w_gt = gt_depth.shape
        h_pred, w_pred = dep_base.shape
        if (h_gt, w_gt) != (h_pred, w_pred):
            from scipy.ndimage import zoom
            scale_h, scale_w = h_gt / h_pred, w_gt / w_pred
            dep_base = zoom(dep_base, (scale_h, scale_w), order=1)
            dep_ours = zoom(dep_ours, (scale_h, scale_w), order=1)
            mask_base = zoom(mask_base.astype(float), (scale_h, scale_w), order=0) > 0.5
            mask_ours = zoom(mask_ours.astype(float), (scale_h, scale_w), order=0) > 0.5

        valid = gt_mask & (gt_depth > 0) & np.isfinite(gt_depth)
        med_gt = np.median(gt_depth[valid]) * 1000
        med_base = np.median(dep_base[valid & mask_base]) * 1000
        med_ours = np.median(dep_ours[valid & mask_ours]) * 1000

        # Col 0: Input with GT annotation
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(Image.open(img_path))
        ax.set_title(label, fontsize=8, fontweight='bold')
        ax.axis('off')
        ax.text(0.5, 0.03, f'GT: {med_gt:.0f} mm',
                transform=ax.transAxes, fontsize=8, fontweight='bold', ha='center', va='bottom',
                color='white',
                bbox=dict(facecolor='#2ca02c', alpha=0.85, edgecolor='none',
                          pad=2, boxstyle='round,pad=0.3'))

        # Col 1: Baseline depth (own range)
        ax = fig.add_subplot(gs[i, 1])
        base_mm = dep_base * 1000
        base_mm[~mask_base] = np.nan
        bvals = base_mm[mask_base & np.isfinite(base_mm)]
        bmin, bmax = np.percentile(bvals, [2, 98])
        ax.imshow(base_mm, cmap='turbo', vmin=bmin, vmax=bmax)
        ratio_b = med_base / med_gt
        ax.set_title(f'Baseline MoGe-2', fontsize=7, fontweight='bold')
        ax.axis('off')
        ax.text(0.5, 0.03, f'{med_base:.0f} mm ({ratio_b:.1f}x)',
                transform=ax.transAxes, fontsize=8, fontweight='bold', ha='center', va='bottom',
                color='white',
                bbox=dict(facecolor='#d62728', alpha=0.85, edgecolor='none',
                          pad=2, boxstyle='round,pad=0.3'))

        # Col 2: DermDepth depth (own range)
        ax = fig.add_subplot(gs[i, 2])
        ours_mm = dep_ours * 1000
        ours_mm[~mask_ours] = np.nan
        ovals = ours_mm[mask_ours & np.isfinite(ours_mm)]
        omin, omax = np.percentile(ovals, [2, 98])
        ax.imshow(ours_mm, cmap='turbo', vmin=omin, vmax=omax)
        ratio_o = med_ours / med_gt
        ax.set_title(f'DermDepth (Ours)', fontsize=7, fontweight='bold')
        ax.axis('off')
        ax.text(0.5, 0.03, f'{med_ours:.0f} mm ({ratio_o:.1f}x)',
                transform=ax.transAxes, fontsize=8, fontweight='bold', ha='center', va='bottom',
                color='white',
                bbox=dict(facecolor='#1f77b4', alpha=0.85, edgecolor='none',
                          pad=2, boxstyle='round,pad=0.3'))

    fig.savefig(OUT_DIR / 'scale_overview.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'scale_overview.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved scale_overview.{{png,pdf}}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Point cloud and depth comparison figures')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--baseline', default='Ruicheng/moge-2-vitl-normal', help='Baseline model')
    parser.add_argument('--dermdepth', default='output/training/exp_a/checkpoint/00001000_ema.pt',
                        help='DermDepth checkpoint')
    parser.add_argument('--figure', choices=['all', 'depth', 'pointcloud', 'overview'],
                        default='all', help='Which figure(s) to generate')
    args = parser.parse_args()

    np.random.seed(42)

    print(f"Loading baseline: {args.baseline}")
    baseline = load_model(args.baseline, args.device)

    print(f"Loading DermDepth: {args.dermdepth}")
    dermdepth = load_model(args.dermdepth, args.device)

    if args.figure in ['all', 'overview']:
        print("\n=== Scale Overview ===")
        make_scale_overview(baseline, dermdepth, args.device)

    if args.figure in ['all', 'depth']:
        print("\n=== Depth Comparison ===")
        make_depth_comparison(baseline, dermdepth, args.device)

    if args.figure in ['all', 'pointcloud']:
        print("\n=== Point Cloud Top-Down ===")
        make_pointcloud_figure(baseline, dermdepth, args.device)

    print("\nDone! Figures saved to", OUT_DIR)
