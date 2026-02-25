#!/usr/bin/env python3
"""Generate 5-method depth/normal comparison figure for DermDepth paper.

Reads saved predictions from save_method_predictions.py, computes normals
from depth where model normals are unavailable, and creates a paper-quality
comparison figure.

Layout:
  - 5 samples (3 SKINL2 + 2 WoundsDB)
  - 2 sub-rows per sample: depth + normal
  - 7 columns: Input | GT | DA3 | MapAnything | PPD | MoGe-2 | DermDepth
  - Scale ratio badge on each depth cell
  - GT column shows depth range in mm

Usage:
    python method_comparison_figure.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / "output" / "figures" / "method_comparison" / "predictions"
OUT_DIR = PROJECT_ROOT / "output" / "figures" / "method_comparison"
SKINL2_DIR = PROJECT_ROOT / "output" / "eval_data" / "skinl2"
WOUNDSDB_DIR = PROJECT_ROOT / "output" / "eval_data" / "woundsdb"

# Sample order (matching save_method_predictions.py)
SAMPLES = [
    ('v1_Melanoma_0203', 'skinl2', 'SKINL2 v1\nMelanoma'),
    ('v2_Seborrheic Keratosis_0051', 'skinl2', 'SKINL2 v2\nSeb. Keratosis'),
    ('v3_Hemangioma_0010', 'skinl2', 'SKINL2 v3\nHemangioma'),
    ('case_22_day_1_scene_1', 'woundsdb', 'WoundsDB\nCase 22'),
    ('case_39_day_1_scene_1', 'woundsdb', 'WoundsDB\nCase 39'),
]

METHODS = ['da3nested', 'mapanything', 'ppd', 'moge2', 'dermdepth']
METHOD_LABELS = ['DA3-Nested', 'MapAnything', 'PPD', 'MoGe-2', 'DermDepth\n(Ours)']
METHODS_WITH_MODEL_NORMAL = {'moge2', 'dermdepth'}

DEPTH_CMAP = 'Spectral_r'

# MICCAI style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


def depth_to_normal(depth, normalize_depth=True):
    """Compute surface normals from depth map using finite differences.

    If normalize_depth=True, normalizes depth to [0,1] before computing
    gradients so that normals show relative surface structure regardless
    of absolute scale.
    """
    d = depth.copy()
    if normalize_depth:
        valid = np.isfinite(d) & (d > 0)
        if valid.any():
            dmin, dmax = d[valid].min(), d[valid].max()
            if dmax > dmin:
                d = (d - dmin) / (dmax - dmin)
    dz_dy, dz_dx = np.gradient(d)
    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(d)], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    normal = normal / norm
    return normal


def normal_to_rgb(normal):
    """Convert normal map to RGB image: (n+1)/2 mapping."""
    rgb = (normal + 1.0) / 2.0
    rgb = np.clip(rgb, 0, 1)
    return rgb


def colorize_depth(depth, mask=None, vmin=None, vmax=None):
    """Colorize depth map with Spectral_r colormap."""
    if mask is None:
        mask = np.isfinite(depth) & (depth > 0)

    depth_norm = np.zeros_like(depth)
    if vmin is not None and vmax is not None and vmax > vmin:
        depth_norm = (depth - vmin) / (vmax - vmin)
    depth_norm = np.clip(depth_norm, 0, 1)

    cmap = plt.colormaps.get_cmap(DEPTH_CMAP)
    colored = cmap(depth_norm)[..., :3]
    colored[~mask] = 0.3  # gray for invalid
    return colored


def compute_scale_ratio(pred, gt, mask):
    """Compute median scale ratio pred/gt on valid pixels."""
    valid = mask & np.isfinite(pred) & (pred > 0) & np.isfinite(gt) & (gt > 0)
    if valid.sum() < 10:
        return None
    return float(np.median(pred[valid]) / np.median(gt[valid]))


def scale_color(ratio):
    """Return color for scale ratio badge: green near 1.0, red far from 1.0."""
    if ratio is None:
        return 'gray'
    err = abs(ratio - 1.0)
    if err < 0.15:
        return '#2ecc71'  # green
    elif err < 0.5:
        return '#f39c12'  # orange
    else:
        return '#e74c3c'  # red


def load_sample_data(sample_name, dataset):
    """Load image, GT depth, GT mask for a sample."""
    if dataset == 'skinl2':
        base = SKINL2_DIR / sample_name
    else:
        base = WOUNDSDB_DIR / sample_name

    img = np.array(Image.open(base / 'image.png').convert('RGB'))
    gt_depth = np.load(base / 'gt_depth.npy')
    gt_mask = np.load(base / 'gt_mask.npy').astype(bool)

    return img, gt_depth, gt_mask


def load_method_prediction(sample_name, method):
    """Load saved depth (and optionally normal) prediction."""
    pred_dir = PRED_DIR / sample_name
    depth_path = pred_dir / f'{method}_depth.npy'
    normal_path = pred_dir / f'{method}_normal.npy'

    if not depth_path.exists():
        return None, None

    depth = np.load(depth_path)
    normal = np.load(normal_path) if normal_path.exists() else None
    return depth, normal


def make_figure():
    """Create the full comparison figure."""
    n_samples = len(SAMPLES)
    n_cols = 2 + len(METHODS)  # Input + GT + methods
    n_rows = n_samples * 2  # depth + normal per sample

    # Figure dimensions — wide enough for 7 columns to be readable
    col_w = 1.3
    row_h = 1.0
    fig_w = col_w * n_cols
    fig_h = row_h * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    # Column headers
    col_headers = ['Input', 'GT'] + METHOD_LABELS

    for sample_idx, (sample_name, dataset, row_label) in enumerate(SAMPLES):
        depth_row = sample_idx * 2
        normal_row = sample_idx * 2 + 1

        # Load GT data
        img, gt_depth, gt_mask = load_sample_data(sample_name, dataset)

        # GT depth range for normalization
        valid_gt = gt_depth[gt_mask & np.isfinite(gt_depth) & (gt_depth > 0)]
        if len(valid_gt) > 0:
            vmin = np.percentile(valid_gt, 2)
            vmax = np.percentile(valid_gt, 98)
            gt_range_mm = (valid_gt.min() * 1000, valid_gt.max() * 1000)
        else:
            vmin, vmax = 0, 1
            gt_range_mm = (0, 0)

        # GT normal from depth
        gt_depth_clean = gt_depth.copy()
        gt_depth_clean[~gt_mask] = np.nanmedian(gt_depth[gt_mask]) if gt_mask.any() else 0
        gt_normal = depth_to_normal(gt_depth_clean)
        gt_normal_rgb = normal_to_rgb(gt_normal)
        gt_normal_rgb[~gt_mask] = 0.3

        # Resize image to depth resolution
        h, w = gt_depth.shape
        img_resized = np.array(Image.fromarray(img).resize((w, h), Image.LANCZOS))

        # === Col 0: Input image ===
        axes[depth_row, 0].imshow(img_resized)
        axes[normal_row, 0].imshow(img_resized)

        # === Col 1: GT depth + normal ===
        gt_colored = colorize_depth(gt_depth, gt_mask, vmin, vmax)
        axes[depth_row, 1].imshow(gt_colored)
        # Depth range annotation
        axes[depth_row, 1].text(
            0.5, 0.02,
            f'{gt_range_mm[0]:.0f}-{gt_range_mm[1]:.0f}mm',
            transform=axes[depth_row, 1].transAxes,
            fontsize=5.5, color='white', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7, linewidth=0))

        axes[normal_row, 1].imshow(gt_normal_rgb)

        # === Cols 2+: Methods ===
        for m_idx, method in enumerate(METHODS):
            col = 2 + m_idx
            pred_depth, pred_normal = load_method_prediction(sample_name, method)

            if pred_depth is None:
                # Missing prediction — show placeholder
                axes[depth_row, col].imshow(np.full((h, w, 3), 0.15))
                axes[depth_row, col].text(0.5, 0.5, 'N/A',
                    transform=axes[depth_row, col].transAxes,
                    fontsize=6, color='white', ha='center', va='center')
                axes[normal_row, col].imshow(np.full((h, w, 3), 0.15))
                continue

            # Resize to GT resolution if needed
            if pred_depth.shape != gt_depth.shape:
                from scipy.ndimage import zoom
                pred_depth = zoom(pred_depth, (h / pred_depth.shape[0], w / pred_depth.shape[1]), order=1)

            # Scale ratio
            scale = compute_scale_ratio(pred_depth, gt_depth, gt_mask)

            # Colorize depth with GT range (no alignment — shows raw metric scale)
            pred_colored = colorize_depth(pred_depth, gt_mask, vmin, vmax)
            axes[depth_row, col].imshow(pred_colored)

            # Scale badge
            if scale is not None:
                badge_color = scale_color(scale)
                axes[depth_row, col].text(
                    0.03, 0.97,
                    f'{scale:.2f}x',
                    transform=axes[depth_row, col].transAxes,
                    fontsize=5.5, color='white', ha='left', va='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=badge_color, alpha=0.85, linewidth=0))

            # Normal map
            if pred_normal is not None and method in METHODS_WITH_MODEL_NORMAL:
                # Use model normal
                if pred_normal.shape[:2] != (h, w):
                    from scipy.ndimage import zoom
                    pred_normal = zoom(pred_normal, (h / pred_normal.shape[0], w / pred_normal.shape[1], 1), order=1)
                    # Re-normalize
                    norm = np.linalg.norm(pred_normal, axis=-1, keepdims=True) + 1e-8
                    pred_normal = pred_normal / norm
                normal_rgb = normal_to_rgb(pred_normal)
            else:
                # Compute normal from depth
                pred_clean = pred_depth.copy()
                invalid = ~(np.isfinite(pred_depth) & (pred_depth > 0))
                pred_clean[invalid] = np.nanmedian(pred_depth[~invalid]) if (~invalid).sum() < pred_depth.size else 0
                computed_normal = depth_to_normal(pred_clean)
                normal_rgb = normal_to_rgb(computed_normal)

            normal_rgb[~gt_mask] = 0.3
            axes[normal_row, col].imshow(normal_rgb)

        # Row labels
        axes[depth_row, 0].set_ylabel(row_label, fontsize=6.5, rotation=90,
                                       labelpad=10, va='center', linespacing=1.3)

    # Column headers
    for col_idx, header in enumerate(col_headers):
        axes[0, col_idx].set_title(header, fontsize=7, fontweight='bold', pad=4,
                                    linespacing=1.2)

    # Sub-row type labels on right side
    for sample_idx in range(n_samples):
        depth_row = sample_idx * 2
        normal_row = sample_idx * 2 + 1
        axes[depth_row, -1].yaxis.set_label_position('right')
        axes[depth_row, -1].set_ylabel('Depth', fontsize=6, rotation=270, labelpad=10)
        axes[normal_row, -1].yaxis.set_label_position('right')
        axes[normal_row, -1].set_ylabel('Normal', fontsize=6, rotation=270, labelpad=10)

    # Clean up all axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
            spine.set_color('#cccccc')

    # Add thin separator lines between samples
    for sample_idx in range(1, n_samples):
        y_pos = 1.0 - (sample_idx * 2) / n_rows
        fig.add_artist(plt.Line2D(
            [0.02, 0.98], [y_pos, y_pos],
            transform=fig.transFigure,
            color='#999999', linewidth=0.5, linestyle='-'))

    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / 'comparison_all.png')
    fig.savefig(OUT_DIR / 'comparison_all.pdf')
    plt.close(fig)
    print(f"Saved to {OUT_DIR / 'comparison_all.{{png,pdf}}'}")


def check_predictions():
    """Check which predictions are available."""
    print("Checking predictions...")
    all_found = True
    for sample_name, dataset, label in SAMPLES:
        for method in METHODS:
            depth_path = PRED_DIR / sample_name / f'{method}_depth.npy'
            status = 'OK' if depth_path.exists() else 'MISSING'
            if not depth_path.exists():
                all_found = False
                print(f"  {status}: {sample_name}/{method}_depth.npy")
    if all_found:
        print("  All predictions found.")
    return all_found


if __name__ == '__main__':
    if not check_predictions():
        print("\nWARNING: Some predictions are missing. Run save_method_predictions.py first.")
        print("Generating figure with available predictions...\n")
    make_figure()
