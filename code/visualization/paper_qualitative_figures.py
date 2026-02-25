#!/usr/bin/env python3
"""Generate paper-quality qualitative comparison figures.

Produces:
  Fig 1: S-SYNTH training data showcase (diverse synthetics with GT)
  Fig 2: SKINL2 qualitative comparison (RGB, orig GT, clean GT, Base, Ours, D3 normals)
  Fig 3: WoundsDB ALL test cases (multi-page PDF)
  Fig 4: DDI ALL test cases with ruler annotation (DermDepth-centered colormap)
  Fig 4B: DDI ALL test cases with baseline depth predictions (cached)

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n MoGe python -u code/visualization/paper_qualitative_figures.py
    CUDA_VISIBLE_DEVICES=3 conda run -n MoGe python -u code/visualization/paper_qualitative_figures.py --figs 3 4 4b
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import json
import sys
import os
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'MoGe'))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "figures" / "paper_qualitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEPTH_CMAP = 'Spectral_r'

# MICCAI style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.titlesize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})


# ============================================================
# Utility functions
# ============================================================

def crop_to_aspect(arr, target_aspect):
    """Center-crop a 2D (or 3D) array to match target_aspect (H/W ratio).
    Returns a view/slice — no interpolation."""
    h, w = arr.shape[:2]
    cur_aspect = h / w
    if abs(cur_aspect - target_aspect) < 0.01:
        return arr
    if cur_aspect > target_aspect:
        # Too tall → crop height
        new_h = int(round(w * target_aspect))
        top = (h - new_h) // 2
        return arr[top:top + new_h]
    else:
        # Too wide → crop width
        new_w = int(round(h / target_aspect))
        left = (w - new_w) // 2
        return arr[:, left:left + new_w]


def depth_to_color(depth, vmin=None, vmax=None, cmap=DEPTH_CMAP, mask=None):
    """Convert depth to color image."""
    if mask is None:
        mask = np.isfinite(depth) & (depth > 0)
    d = depth.copy()
    d[~mask] = np.nan
    if vmin is None:
        vmin = np.nanpercentile(d, 2)
    if vmax is None:
        vmax = np.nanpercentile(d, 98)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    colored = cm(norm(d))[:, :, :3]
    colored[~mask] = 1.0  # white background
    return colored


def normal_to_color(normal):
    """Convert normal map (H,W,3) to RGB in [0,1].
    MoGe convention: R=X, G=-Y, B=-Z → blue for camera-facing surfaces.
    Matches MoGe/moge/utils/vis.py:colorize_normal()."""
    n = normal.copy()
    invalid = ~np.all(np.isfinite(n), axis=-1)
    n[invalid] = 0
    rgb = n * np.array([0.5, -0.5, -0.5]) + 0.5
    return rgb.clip(0, 1)


def depth_to_normal(depth, fx=None, fy=None):
    """Compute surface normals from depth via finite differences."""
    h, w = depth.shape
    if fx is None:
        fx = fy = w / (2 * np.tan(np.radians(30)))  # 60 deg FoV

    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    X = (jj - w/2) * depth / fx
    Y = (ii - h/2) * depth / fy
    Z = depth.copy()

    dXdx = np.zeros_like(X); dXdx[:, :-1] = X[:, 1:] - X[:, :-1]
    dYdx = np.zeros_like(Y); dYdx[:, :-1] = Y[:, 1:] - Y[:, :-1]
    dZdx = np.zeros_like(Z); dZdx[:, :-1] = Z[:, 1:] - Z[:, :-1]
    dXdy = np.zeros_like(X); dXdy[:-1] = X[1:] - X[:-1]
    dYdy = np.zeros_like(Y); dYdy[:-1] = Y[1:] - Y[:-1]
    dZdy = np.zeros_like(Z); dZdy[:-1] = Z[1:] - Z[:-1]

    nx = dYdx * dZdy - dZdx * dYdy
    ny = dZdx * dXdy - dXdx * dZdy
    nz = dXdx * dYdy - dYdx * dXdy
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-10
    normal = np.stack([nx/norm, ny/norm, nz/norm], axis=-1)
    return normal


def add_scale_badge(ax, scale, color='white'):
    """Add scale ratio badge to axis."""
    text = f'Scale: {scale:.2f}x'
    ax.text(0.05, 0.92, text, transform=ax.transAxes,
            fontsize=7, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
            va='top', ha='left')


def add_depth_range_badge(ax, depth, unit='mm'):
    """Add depth range badge to axis."""
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        return
    vmin, vmax = np.percentile(depth[valid], [5, 95])
    text = f'{vmin:.0f}-{vmax:.0f} {unit}'
    ax.text(0.05, 0.08, text, transform=ax.transAxes,
            fontsize=6, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6),
            va='bottom', ha='left')


def add_area_badge(ax, ratio, good=False):
    """Add area ratio badge to axis."""
    color = '#27ae60' if good else '#c0392b'
    text = f'Area: {ratio:.1f}x' if abs(ratio) < 100 else f'Area: {ratio:.0f}x'
    ax.text(0.05, 0.92, text, transform=ax.transAxes,
            fontsize=7, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8),
            va='top', ha='left')


def run_inference(model, img_rgb, device='cuda'):
    """Run MoGe inference and return depth + normal."""
    import torch
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        out = model.infer(img_t)
    depth = out['depth'].squeeze().cpu().numpy()
    normal = out['normal'].squeeze().cpu().numpy()
    if normal.shape[0] == 3:
        normal = normal.transpose(1, 2, 0)
    return depth, normal


def load_models(device='cuda'):
    """Load Base MoGe-2, Exp H s1800, and Exp D3 models."""
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version('v2')

    print("Loading models...")
    base = MoGeModel.from_pretrained(
        str(PROJECT_ROOT / 'MoGe' / 'pretrained_moge2.pt')).to(device).eval()
    exp_h = MoGeModel.from_pretrained(
        str(PROJECT_ROOT / 'output' / 'training' / 'exp_h' / 'checkpoint' / '00001800_ema.pt')).to(device).eval()
    exp_d3 = MoGeModel.from_pretrained(
        str(PROJECT_ROOT / 'output' / 'training' / 'exp_d3' / 'checkpoint' / '00002500_ema.pt')).to(device).eval()
    exp_a = MoGeModel.from_pretrained(
        str(PROJECT_ROOT / 'output' / 'training' / 'exp_a' / 'checkpoint' / '00001000_ema.pt')).to(device).eval()
    print("Models loaded.")
    return {'base': base, 'exp_h': exp_h, 'exp_d3': exp_d3, 'exp_a': exp_a}


# ============================================================
# Fig 1: S-SYNTH Training Data Showcase
# ============================================================

def generate_fig1(models, device='cuda'):
    """S-SYNTH samples with diverse skin tones and lesion sizes."""
    from moge.utils.io import read_depth

    print("\n=== Fig 1: S-SYNTH Training Data ===")

    synth_dir = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'colab_gen' / 'DermDepthSynth'
    index_file = synth_dir / '.index.txt'
    with open(index_file) as f:
        all_samples = [l.strip() for l in f if l.strip()]

    # Classify samples by Fitzpatrick group
    candidates = {'I-II': [], 'III-IV': [], 'V-VI': []}
    for sname in all_samples:
        sdir = synth_dir / sname
        params_file = sdir / 'generation_params.json'
        if not params_file.exists():
            continue
        with open(params_file) as f:
            params = json.load(f)
        fp_group = params.get('fitzpatrick_group', 'III-IV')
        if fp_group not in candidates:
            fp_group = 'III-IV'
        candidates[fp_group].append(sname)

    print(f"  Diversity: {', '.join(f'{k}={len(v)}' for k,v in candidates.items())}")

    # Pick 2 per FP group with different seeds for variety
    np.random.seed(42)
    picks = []
    for group in ['I-II', 'III-IV', 'V-VI']:
        pool = candidates[group]
        if len(pool) >= 2:
            idxs = np.random.choice(len(pool), size=2, replace=False)
            picks.extend([pool[i] for i in idxs])
        elif pool:
            picks.append(pool[0])

    print(f"  Selected {len(picks)} samples")

    n_rows = len(picks)
    n_cols = 3
    col_labels = ['Input Image', 'GT Depth Map', 'GT Surface Normal']

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, sname in enumerate(picks):
        sdir = synth_dir / sname

        img_path = sdir / 'image.png'
        if not img_path.exists():
            img_path = sdir / 'image.jpg'
        img = np.array(Image.open(img_path))
        gt_depth_mm = read_depth(str(sdir / 'depth.png'))
        with open(sdir / 'meta.json') as f:
            meta = json.load(f)

        K = np.array(meta['intrinsics'])
        fx_px = K[0][0] * img.shape[1]
        fy_px = K[1][1] * img.shape[0]
        gt_normal = -depth_to_normal(gt_depth_mm, fx=fx_px, fy=fy_px)  # negate to match MoGe convention

        params_file = sdir / 'generation_params.json'
        fp_label = ''
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            fp_group = params.get('fitzpatrick_group', '')
            fp_label = f'FP {fp_group}' if fp_group else ''

        axes[i, 0].imshow(img)
        if fp_label:
            axes[i, 0].text(0.05, 0.92, fp_label, transform=axes[i, 0].transAxes,
                           fontsize=7, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6),
                           va='top')

        axes[i, 1].imshow(depth_to_color(gt_depth_mm))
        add_depth_range_badge(axes[i, 1], gt_depth_mm, 'mm')

        axes[i, 2].imshow(normal_to_color(gt_normal))

    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=10, fontweight='bold', pad=4)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig1_ssynth.{ext}')
        print(f"  Saved: {OUT_DIR / f'fig1_ssynth.{ext}'}")
    plt.close()


# ============================================================
# Fig 2: SKINL2 Qualitative Comparison
# ============================================================

def generate_fig2(models, device='cuda'):
    """SKINL2: ALL test cases. Multi-page PDF.
    Columns: RGB | Orig GT | Clean GT | Base MoGe-2 | DermDepth | Normal (D3)"""
    print("\n=== Fig 2: SKINL2 ALL Test Cases ===")

    skinl2_dir = PROJECT_ROOT / 'output' / 'eval_data' / 'skinl2'
    skinl2_raw_base = PROJECT_ROOT / 'data' / 'SKINL2'

    # Load stratified test split
    split_file = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'skinl2_moge' / 'test.txt'
    if split_file.exists():
        with open(split_file) as f:
            test_samples = sorted([l.strip().split('/')[-1] for l in f if l.strip()])
    else:
        test_samples = sorted([d.name for d in skinl2_dir.iterdir()
                       if d.is_dir() and (d.name.startswith('v2_') or d.name.startswith('v3_'))])

    print(f"  Test samples: {len(test_samples)}")

    col_labels = ['Input', 'Raw Plenoptic GT', 'Cleaned GT (Ours)', 'Base MoGe-2', 'DermDepth', 'Normal (D3)']
    n_cols = len(col_labels)
    rows_per_page = 6

    # Build all row data
    rows = []
    for sample_name in test_samples:
        sdir = skinl2_dir / sample_name
        parts = sample_name.split('_', 1)
        version_tag = parts[0]
        remainder = parts[1]
        case_id = remainder.rsplit('_', 1)[-1]
        disease = remainder.rsplit('_', 1)[0]
        version_dir = {'v1': 'SKINL2_v1', 'v2': 'SKINL2_v2', 'v3': 'SKINL2_v3'}.get(version_tag)

        img_path = sdir / 'image.png'
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path))

        gt_depth = np.load(sdir / 'gt_depth.npy')
        gt_mask = np.load(sdir / 'gt_mask.npy') if (sdir / 'gt_mask.npy').exists() else np.ones_like(gt_depth, bool)

        # Load raw SKINL2 depth
        orig_gt = gt_depth
        if version_dir:
            raw_depth_path = skinl2_raw_base / version_dir / 'DepthMap' / disease / case_id / f'{case_id}_DepthMap.tiff'
            if raw_depth_path.exists():
                raw_depth = -np.array(Image.open(raw_depth_path)).astype(np.float32)
                raw_depth[raw_depth <= 0] = np.nan
                raw_depth_m = raw_depth * 0.001
                raw_resized = np.array(Image.fromarray(raw_depth_m).resize(
                    (img.shape[1], img.shape[0]), Image.BILINEAR))
                orig_gt = raw_resized

        # Run inference
        base_depth, base_normal = run_inference(models['base'], img, device)
        ours_depth, _ = run_inference(models['exp_h'], img, device)
        _, d3_normal = run_inference(models['exp_d3'], img, device)

        valid = gt_mask & np.isfinite(gt_depth) & (gt_depth > 0)
        base_scale = np.median(base_depth[valid]) / np.median(gt_depth[valid]) if valid.sum() > 0 else 0
        ours_scale = np.median(ours_depth[valid]) / np.median(gt_depth[valid]) if valid.sum() > 0 else 0

        rows.append({
            'name': sample_name,
            'label': f'{version_tag} {disease[:18]}',
            'img': img,
            'orig_gt': orig_gt,
            'gt_depth': gt_depth,
            'gt_mask': gt_mask,
            'valid': valid,
            'base_depth': base_depth,
            'ours_depth': ours_depth,
            'd3_normal': d3_normal,
            'base_scale': base_scale,
            'ours_scale': ours_scale,
        })

    print(f"  Rendering {len(rows)} rows across {(len(rows) + rows_per_page - 1) // rows_per_page} pages")

    def render_skinl2_page(page_rows, axes):
        for i, row in enumerate(page_rows):
            axes[i, 0].imshow(row['img'])
            axes[i, 0].set_ylabel(row['label'], fontsize=7, fontweight='bold', rotation=90, labelpad=8)
            axes[i, 1].imshow(depth_to_color(row['orig_gt'], mask=row['gt_mask']))
            if row['valid'].sum() > 0:
                add_depth_range_badge(axes[i, 1], row['gt_depth'][row['valid']] * 1000, 'mm')
            axes[i, 2].imshow(depth_to_color(row['gt_depth'], mask=row['gt_mask']))
            axes[i, 3].imshow(depth_to_color(row['base_depth']))
            add_scale_badge(axes[i, 3], row['base_scale'])
            axes[i, 4].imshow(depth_to_color(row['ours_depth']))
            add_scale_badge(axes[i, 4], row['ours_scale'])
            axes[i, 5].imshow(normal_to_color(row['d3_normal']))

    # Multi-page PDF
    pdf_path = OUT_DIR / 'fig2_skinl2.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            render_skinl2_page(page_rows, axes)

            for j, lbl in enumerate(col_labels):
                axes[0, j].set_title(lbl, fontsize=10, fontweight='bold', pad=4)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'SKINL2 Test Set (page {page_num})', fontsize=11, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    # Also save first page as PNG
    page_rows = rows[:rows_per_page]
    n_rows = len(page_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    render_skinl2_page(page_rows, axes)
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=10, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    fig.savefig(OUT_DIR / 'fig2_skinl2.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path} ({(len(rows) + rows_per_page - 1) // rows_per_page} pages)")
    print(f"  Saved: {OUT_DIR / 'fig2_skinl2.png'} (page 1)")


# ============================================================
# Fig 2 Supp: SKINL2 Base MoGe-2 Normals (supplemental)
# ============================================================

def generate_fig2_supp(models, device='cuda'):
    """SKINL2: Input + Base MoGe-2 normals for ALL test cases. Multi-page PDF."""
    print("\n=== Fig 2 Supp: SKINL2 Base MoGe-2 Normals ===")

    skinl2_dir = PROJECT_ROOT / 'output' / 'eval_data' / 'skinl2'

    # Load stratified test split
    split_file = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'skinl2_moge' / 'test.txt'
    if split_file.exists():
        with open(split_file) as f:
            test_samples = sorted([l.strip().split('/')[-1] for l in f if l.strip()])
    else:
        test_samples = sorted([d.name for d in skinl2_dir.iterdir()
                       if d.is_dir() and d.name.startswith('v')])

    print(f"  Test samples: {len(test_samples)}")

    col_labels = ['Input', 'Base MoGe-2 Normal']
    n_cols = len(col_labels)
    rows_per_page = 8

    rows = []
    for idx, sample_name in enumerate(test_samples):
        sdir = skinl2_dir / sample_name
        img_path = sdir / 'image.png'
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path))

        # Run base model inference
        _, base_normal = run_inference(models['base'], img, device)

        parts = sample_name.split('_', 1)
        version_tag = parts[0]
        disease = parts[1].rsplit('_', 1)[0] if len(parts) > 1 else ''

        rows.append({
            'name': sample_name,
            'label': f'{version_tag} {disease[:18]}',
            'img': img,
            'base_normal': base_normal,
        })
        if (idx + 1) % 20 == 0:
            print(f"    Processed {idx + 1}/{len(test_samples)}")

    print(f"  Rendering {len(rows)} rows across {(len(rows) + rows_per_page - 1) // rows_per_page} pages")

    def render_page(page_rows, axes):
        for i, row in enumerate(page_rows):
            axes[i, 0].imshow(row['img'])
            axes[i, 0].set_ylabel(row['label'], fontsize=7, fontweight='bold', rotation=90, labelpad=8)
            axes[i, 1].imshow(normal_to_color(row['base_normal']))

    # Multi-page PDF
    pdf_path = OUT_DIR / 'fig2_supp_skinl2_base_normals.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.2))
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            render_page(page_rows, axes)

            for j, lbl in enumerate(col_labels):
                axes[0, j].set_title(lbl, fontsize=10, fontweight='bold', pad=4)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'SKINL2 Test Set — Base MoGe-2 Normals (page {page_num})',
                        fontsize=11, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    # Also save first page as PNG
    page_rows = rows[:rows_per_page]
    n_rows = len(page_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    render_page(page_rows, axes)
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=10, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    fig.savefig(OUT_DIR / 'fig2_supp_skinl2_base_normals.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path} ({(len(rows) + rows_per_page - 1) // rows_per_page} pages)")
    print(f"  Saved: {OUT_DIR / 'fig2_supp_skinl2_base_normals.png'} (page 1)")


# ============================================================
# Fig 2C: SKINL2 v1 Nevus+Hemangioma full comparison
# ============================================================

def generate_fig2c(models, device='cuda'):
    """SKINL2 v1 Nevus & Hemangioma: 8-column comparison.
    Columns: Input | GT Depth | DA³ | MapAnything | PPD | DermDepth | Base Normal | D3 Normal
    Depths use cached predictions; normals require inference."""
    from scipy.ndimage import zoom as scipy_zoom
    print("\n=== Fig 2C: SKINL2 v1 Nevus + Hemangioma ===")

    skinl2_dir = PROJECT_ROOT / 'output' / 'eval_data' / 'skinl2'

    # Filter test split for v1 Nevus and Hemangioma
    split_file = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'skinl2_moge' / 'test.txt'
    with open(split_file) as f:
        all_test = sorted([l.strip().split('/')[-1] for l in f if l.strip()])
    test_samples = [s for s in all_test if s.startswith('v1_Nevus_') or s.startswith('v1_Hemangioma_')]
    print(f"  Samples: {len(test_samples)} ({sum(1 for s in test_samples if 'Hemangioma' in s)} Hemangioma, "
          f"{sum(1 for s in test_samples if 'Nevus' in s)} Nevus)")

    depth_methods = [
        ('da3nested', 'DA$^3$-Nested'),
        ('mapanything', 'MapAnything'),
        ('ppd', 'PPD'),
        ('exp_h', 'DermDepth'),
    ]
    col_labels = ['Input', 'GT Depth'] + [m[1] for m in depth_methods] + ['Base MoGe-2\nNormal', 'DermDepth (D3)\nNormal']
    n_cols = len(col_labels)
    rows_per_page = 6

    rows = []
    for idx, sample_name in enumerate(test_samples):
        sdir = skinl2_dir / sample_name
        img_path = sdir / 'image.png'
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path))

        gt_depth = np.load(sdir / 'gt_depth.npy')
        gt_mask = np.load(sdir / 'gt_mask.npy') if (sdir / 'gt_mask.npy').exists() else np.ones_like(gt_depth, bool)
        valid = gt_mask & np.isfinite(gt_depth) & (gt_depth > 0)

        # Load cached depth predictions + compute scales
        depths = {}
        scales = {}
        for method_key, _ in depth_methods:
            pred = load_cached_depth('skinl2', sample_name, method_key)
            depths[method_key] = pred
            scales[method_key] = compute_scale_badge(pred, gt_depth, gt_mask) if pred is not None else 0.0

        # Inference for normals
        _, base_normal = run_inference(models['base'], img, device)
        _, d3_normal = run_inference(models['exp_d3'], img, device)

        disease = sample_name.split('_', 1)[1].rsplit('_', 1)[0]
        rows.append({
            'name': sample_name,
            'label': disease,
            'img': img,
            'gt_depth': gt_depth,
            'gt_mask': gt_mask,
            'valid': valid,
            'depths': depths,
            'scales': scales,
            'base_normal': base_normal,
            'd3_normal': d3_normal,
        })
        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx + 1}/{len(test_samples)}")

    print(f"  Rendering {len(rows)} rows across {(len(rows) + rows_per_page - 1) // rows_per_page} pages")

    def render_page(page_rows, axes):
        for i, row in enumerate(page_rows):
            target_aspect = row['img'].shape[0] / row['img'].shape[1]
            # Col 0: Input
            axes[i, 0].imshow(row['img'])
            axes[i, 0].set_ylabel(row['label'], fontsize=7, fontweight='bold', rotation=90, labelpad=8)
            # Col 1: GT Depth
            axes[i, 1].imshow(depth_to_color(row['gt_depth'], mask=row['gt_mask']))
            if row['valid'].sum() > 0:
                add_depth_range_badge(axes[i, 1], row['gt_depth'][row['valid']] * 1000, 'mm')
            # Cols 2-5: Method depths (cached, crop to match image aspect)
            for j, (method_key, _) in enumerate(depth_methods):
                d = row['depths'].get(method_key)
                if d is not None:
                    d_cropped = crop_to_aspect(d, target_aspect)
                    axes[i, j + 2].imshow(depth_to_color(d_cropped))
                    scale = row['scales'].get(method_key, 0)
                    if scale > 0:
                        add_scale_badge(axes[i, j + 2], scale)
                else:
                    axes[i, j + 2].text(0.5, 0.5, 'N/A', transform=axes[i, j + 2].transAxes,
                                       ha='center', va='center', fontsize=12, color='gray')
            # Col 6: Base MoGe-2 Normal
            axes[i, 6].imshow(normal_to_color(row['base_normal']))
            # Col 7: D3 Normal
            axes[i, 7].imshow(normal_to_color(row['d3_normal']))

    # Multi-page PDF
    pdf_path = OUT_DIR / 'fig2c_skinl2_v1_nevus_hemangioma.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            render_page(page_rows, axes)

            for j, lbl in enumerate(col_labels):
                axes[0, j].set_title(lbl, fontsize=8, fontweight='bold', pad=4)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'SKINL2 v1 Nevus + Hemangioma (page {page_num})',
                        fontsize=11, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    # PNG first page
    page_rows = rows[:rows_per_page]
    n_rows = len(page_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    render_page(page_rows, axes)
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=8, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    fig.savefig(OUT_DIR / 'fig2c_skinl2_v1_nevus_hemangioma.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path} ({(len(rows) + rows_per_page - 1) // rows_per_page} pages)")
    print(f"  Saved: {OUT_DIR / 'fig2c_skinl2_v1_nevus_hemangioma.png'} (page 1)")


# ============================================================
# Fig DDI-C: DDI test set full comparison (same format as 2c)
# ============================================================

def generate_fig_ddi_c(models, device='cuda'):
    """DDI test set: 8-column comparison matching fig2c format.
    Columns: Input | GT Depth (N/A) | DA³ | MapAnything | PPD | DermDepth | Base Normal | D3 Normal
    Uses ruler area ratio as metric badge on depth columns."""
    from scipy.ndimage import zoom as scipy_zoom
    print("\n=== Fig DDI-C: DDI Test Set Full Comparison ===")

    ddi_dir = PROJECT_ROOT / 'data' / 'DDI'
    cache_dir = PROJECT_ROOT / 'output' / 'evaluation' / 'ddi_rulers' / '_cache'
    labels_dir = ddi_dir / 'FEDD' / 'ddi_labels'

    # Load DDI metadata
    import csv
    ddi_meta = {}
    with open(ddi_dir / 'map.csv') as f:
        for row in csv.DictReader(f):
            ddi_meta[row['DDI_file']] = row

    # Test split
    split_file = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'ddi_moge' / 'test.txt'
    with open(split_file) as f:
        test_stems = sorted([l.strip().split('/')[-1] for l in f if l.strip()])
    print(f"  Test samples: {len(test_stems)}")

    # Ruler area computation helpers (from eval_ddi_rulers.py)
    GT_AREA_CM2 = 6.6
    FOV_DEG = 60.0

    def estimate_intrinsics(h, w):
        fx = fy = w / (2.0 * np.tan(np.radians(FOV_DEG / 2.0)))
        return np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]])

    def compute_surface_area(depth, mask, intrinsics):
        h, w = depth.shape[:2]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        jj, ii = np.meshgrid(np.arange(w), np.arange(h))
        X = (jj - cx) * depth / fx
        Y = (ii - cy) * depth / fy
        Z = depth
        dXdx = np.zeros_like(X); dYdx = np.zeros_like(Y); dZdx = np.zeros_like(Z)
        dXdx[:, :-1] = X[:, 1:] - X[:, :-1]
        dYdx[:, :-1] = Y[:, 1:] - Y[:, :-1]
        dZdx[:, :-1] = Z[:, 1:] - Z[:, :-1]
        dXdy = np.zeros_like(X); dYdy = np.zeros_like(Y); dZdy = np.zeros_like(Z)
        dXdy[:-1] = X[1:] - X[:-1]; dYdy[:-1] = Y[1:] - Y[:-1]; dZdy[:-1] = Z[1:] - Z[:-1]
        nx = dYdx * dZdy - dZdx * dYdy
        ny = dZdx * dXdy - dXdx * dZdy
        nz = dXdx * dYdy - dYdx * dXdy
        area_elem = np.sqrt(nx**2 + ny**2 + nz**2)
        valid = mask & np.isfinite(depth) & (depth > 0)
        valid[:-1] &= np.isfinite(depth[1:]); valid[:, :-1] &= np.isfinite(depth[:, 1:])
        return float(np.sum(area_elem[valid]))

    def load_ruler_mask(stem, img_h, img_w):
        for f in labels_dir.rglob(f'{stem}.npy'):
            mask_256 = np.load(f)
            ruler_256 = (mask_256 == 3).astype(np.uint8)
            ruler_full = scipy_zoom(ruler_256, (img_h / ruler_256.shape[0], img_w / ruler_256.shape[1]), order=0)
            return ruler_full.astype(bool)
        return None

    # Use SKINL2 aspect ratio (h/w ≈ 0.695) for uniform grid
    STANDARD_ASPECT = 1423 / 2048  # h/w from SKINL2

    depth_methods = [
        ('da3nested', 'DA$^3$-Nested'),
        ('mapanything', 'MapAnything'),
        ('ppd', 'PPD'),
        ('exp_h_s1800', 'DermDepth'),
    ]
    col_labels = ['Input', 'GT Depth'] + [m[1] for m in depth_methods] + ['Base MoGe-2\nNormal', 'DermDepth (D3)\nNormal']
    n_cols = len(col_labels)
    rows_per_page = 6

    rows = []
    for idx, stem in enumerate(test_stems):
        img_path = ddi_dir / 'images' / f'{stem}.png'
        if not img_path.exists():
            img_path = ddi_dir / 'images' / f'{stem}.jpg'
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path).convert('RGB'))
        img_h, img_w = img.shape[:2]

        # Ruler mask for area ratio — compute BEFORE cropping
        ruler_mask = load_ruler_mask(stem, img_h, img_w)
        intrinsics = estimate_intrinsics(img_h, img_w)

        # Load cached depths + compute ruler area ratios on original resolution
        depths = {}
        ratios = {}
        for method_key, _ in depth_methods:
            depth_path = cache_dir / method_key / f'{stem}_depth.npy'
            if depth_path.exists():
                d = np.load(depth_path)
                depths[method_key] = d
                if ruler_mask is not None:
                    if d.shape != (img_h, img_w):
                        d_eval = scipy_zoom(d, (img_h / d.shape[0], img_w / d.shape[1]), order=1)
                    else:
                        d_eval = d
                    area_m2 = compute_surface_area(d_eval, ruler_mask, intrinsics)
                    area_cm2 = area_m2 * 1e4
                    ratios[method_key] = area_cm2 / GT_AREA_CM2
                else:
                    ratios[method_key] = 0
            else:
                depths[method_key] = None
                ratios[method_key] = 0

        # Inference for normals on original image
        _, base_normal = run_inference(models['base'], img, device)
        _, d3_normal = run_inference(models['exp_d3'], img, device)

        # Crop everything to standard aspect ratio
        img = crop_to_aspect(img, STANDARD_ASPECT)
        base_normal = crop_to_aspect(base_normal, STANDARD_ASPECT)
        d3_normal = crop_to_aspect(d3_normal, STANDARD_ASPECT)
        for mk in depths:
            if depths[mk] is not None:
                depths[mk] = crop_to_aspect(depths[mk], STANDARD_ASPECT)

        m = ddi_meta.get(f'{stem}.png', {})
        tone = m.get('skin_tone', '?')
        tone_label = {'12': 'FP I-II', '34': 'FP III-IV', '56': 'FP V-VI'}.get(tone, tone)
        disease = m.get('disease', 'unknown')

        rows.append({
            'stem': stem,
            'label': f'{tone_label}\n{disease[:22]}',
            'img': img,
            'depths': depths,
            'ratios': ratios,
            'base_normal': base_normal,
            'd3_normal': d3_normal,
        })
        print(f"    [{idx+1}/{len(test_stems)}] {stem} ({tone_label}, {disease[:20]})")

    print(f"  Rendering {len(rows)} rows across {(len(rows) + rows_per_page - 1) // rows_per_page} pages")

    def render_page(page_rows, axes):
        for i, row in enumerate(page_rows):
            img_h, img_w = row['img'].shape[:2]
            # Col 0: Input
            axes[i, 0].imshow(row['img'])
            axes[i, 0].set_ylabel(row['label'], fontsize=6, fontweight='bold', rotation=90, labelpad=8)
            # Col 1: GT Depth — N/A placeholder matching image dimensions
            na_img = np.full((img_h, img_w, 3), 0.94)  # light gray
            axes[i, 1].imshow(na_img)
            axes[i, 1].text(0.5, 0.5, 'N/A', transform=axes[i, 1].transAxes,
                           ha='center', va='center', fontsize=14, color='#888888', fontweight='bold')
            # Cols 2-5: Method depths (already cropped to standard aspect)
            # Scale derived from ruler area: area ~ scale², so scale = sqrt(area_ratio)
            for j, (method_key, _) in enumerate(depth_methods):
                d = row['depths'].get(method_key)
                if d is not None:
                    axes[i, j + 2].imshow(depth_to_color(d))
                    ratio = row['ratios'].get(method_key, 0)
                    if ratio > 0:
                        add_scale_badge(axes[i, j + 2], np.sqrt(ratio))
                else:
                    axes[i, j + 2].text(0.5, 0.5, 'N/A', transform=axes[i, j + 2].transAxes,
                                       ha='center', va='center', fontsize=12, color='gray')
            # Col 6: Base MoGe-2 Normal
            axes[i, 6].imshow(normal_to_color(row['base_normal']))
            # Col 7: D3 Normal
            axes[i, 7].imshow(normal_to_color(row['d3_normal']))

    # Multi-page PDF
    pdf_path = OUT_DIR / 'fig_ddi_c_full_comparison.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            render_page(page_rows, axes)

            for j, lbl in enumerate(col_labels):
                axes[0, j].set_title(lbl, fontsize=8, fontweight='bold', pad=4)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'DDI Test Set — Full Comparison (page {page_num})',
                        fontsize=11, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    # PNG first page
    page_rows = rows[:rows_per_page]
    n_rows = len(page_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    render_page(page_rows, axes)
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=8, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    fig.savefig(OUT_DIR / 'fig_ddi_c_full_comparison.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path} ({(len(rows) + rows_per_page - 1) // rows_per_page} pages)")
    print(f"  Saved: {OUT_DIR / 'fig_ddi_c_full_comparison.png'} (page 1)")


# ============================================================
# Fig 3: WoundsDB ALL Test Cases (multi-page)
# ============================================================

def generate_fig3(models, device='cuda'):
    """WoundsDB: ALL test cases. Multi-page PDF."""
    print("\n=== Fig 3: WoundsDB ALL Test Cases ===")

    woundsdb_dir = PROJECT_ROOT / 'output' / 'eval_data' / 'woundsdb'

    # Get all test cases (case_id > 30)
    all_dirs = sorted([d for d in woundsdb_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])
    test_dirs = []
    for d in all_dirs:
        try:
            case_num = int(d.name.split('_')[1])
            if case_num > 30:
                test_dirs.append(d)
        except (ValueError, IndexError):
            continue

    print(f"  Found {len(test_dirs)} test cases")

    col_labels = ['Input', 'GT Depth', 'Base MoGe-2', 'DermDepth', 'Base Normal', 'D3 Normal']
    n_cols = len(col_labels)
    rows_per_page = 6

    # Build all row data first
    rows = []
    for sdir in test_dirs:
        img_path = sdir / 'image.png'
        if not img_path.exists():
            continue
        gt_depth_path = sdir / 'gt_depth.npy'
        if not gt_depth_path.exists():
            continue

        img = np.array(Image.open(img_path))
        gt_depth = np.load(gt_depth_path)
        gt_mask_path = sdir / 'gt_mask.npy'
        gt_mask = np.load(gt_mask_path) if gt_mask_path.exists() else (np.isfinite(gt_depth) & (gt_depth > 0))

        base_depth, base_normal = run_inference(models['base'], img, device)
        ours_depth, _ = run_inference(models['exp_h'], img, device)
        _, d3_normal = run_inference(models['exp_d3'], img, device)

        valid = gt_mask & np.isfinite(gt_depth) & (gt_depth > 0)
        if valid.sum() > 0:
            base_scale = np.median(base_depth[valid]) / np.median(gt_depth[valid])
            ours_scale = np.median(ours_depth[valid]) / np.median(gt_depth[valid])
        else:
            base_scale = ours_scale = 0.0

        rows.append({
            'name': sdir.name,
            'img': img,
            'gt_depth': gt_depth,
            'gt_mask': gt_mask,
            'base_depth': base_depth,
            'base_normal': base_normal,
            'ours_depth': ours_depth,
            'd3_normal': d3_normal,
            'base_scale': base_scale,
            'ours_scale': ours_scale,
            'valid': valid,
        })

    print(f"  Rendering {len(rows)} rows across {(len(rows) + rows_per_page - 1) // rows_per_page} pages")

    # Multi-page PDF
    pdf_path = OUT_DIR / 'fig3_woundsdb.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            for i, row in enumerate(page_rows):
                # Simplify case name for label
                label = row['name'].replace('_scene_1', '').replace('_scene_2', ' s2')
                label = label.replace('_day_', ' d').replace('case_', 'C')

                axes[i, 0].imshow(row['img'])
                axes[i, 0].set_ylabel(label, fontsize=7, fontweight='bold', rotation=90, labelpad=6)

                axes[i, 1].imshow(depth_to_color(row['gt_depth'], mask=row['gt_mask']))
                if row['valid'].sum() > 0:
                    add_depth_range_badge(axes[i, 1], row['gt_depth'][row['valid']] * 1000, 'mm')

                axes[i, 2].imshow(depth_to_color(row['base_depth']))
                add_scale_badge(axes[i, 2], row['base_scale'])

                axes[i, 3].imshow(depth_to_color(row['ours_depth']))
                add_scale_badge(axes[i, 3], row['ours_scale'])

                axes[i, 4].imshow(normal_to_color(row['base_normal']))
                axes[i, 5].imshow(normal_to_color(row['d3_normal']))

            for j, label in enumerate(col_labels):
                axes[0, j].set_title(label, fontsize=10, fontweight='bold', pad=4)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'WoundsDB Test Set (page {page_num})', fontsize=11, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    # Also save first page as PNG
    page_rows = rows[:rows_per_page]
    n_rows = len(page_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for i, row in enumerate(page_rows):
        label = row['name'].replace('_scene_1', '').replace('_scene_2', ' s2')
        label = label.replace('_day_', ' d').replace('case_', 'C')
        axes[i, 0].imshow(row['img'])
        axes[i, 0].set_ylabel(label, fontsize=7, fontweight='bold', rotation=90, labelpad=6)
        axes[i, 1].imshow(depth_to_color(row['gt_depth'], mask=row['gt_mask']))
        if row['valid'].sum() > 0:
            add_depth_range_badge(axes[i, 1], row['gt_depth'][row['valid']] * 1000, 'mm')
        axes[i, 2].imshow(depth_to_color(row['base_depth']))
        add_scale_badge(axes[i, 2], row['base_scale'])
        axes[i, 3].imshow(depth_to_color(row['ours_depth']))
        add_scale_badge(axes[i, 3], row['ours_scale'])
        axes[i, 4].imshow(normal_to_color(row['base_normal']))
        axes[i, 5].imshow(normal_to_color(row['d3_normal']))
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=10, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    fig.savefig(OUT_DIR / 'fig3_woundsdb.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path} ({(len(rows) + rows_per_page - 1) // rows_per_page} pages)")
    print(f"  Saved: {OUT_DIR / 'fig3_woundsdb.png'} (page 1)")


# ============================================================
# Fig 4: DDI ALL Test Cases (DermDepth-centered colormap)
# ============================================================

def generate_fig4(models, device='cuda'):
    """DDI: ALL test cases. DermDepth-centered depth colormap. Base + D3 normals."""
    print("\n=== Fig 4: DDI ALL Test Cases ===")

    ddi_dir = PROJECT_ROOT / 'data' / 'DDI'
    fedd_dir = ddi_dir / 'FEDD' / 'ddi_labels'

    # Build label index
    label_index = {}
    for f in fedd_dir.rglob('*.npy'):
        label_index.setdefault(f.stem, f)

    # Load ruler results
    with open(PROJECT_ROOT / 'output' / 'evaluation' / 'ddi_rulers' / 'ddi_ruler_results.json') as f:
        ruler_data = json.load(f)
    ruler_meta = {s['stem']: s for s in ruler_data['per_sample']}

    # Load test split
    with open(PROJECT_ROOT / 'data' / 'dermdepth_train' / 'ddi_moge' / 'test.txt') as f:
        test_stems = [l.strip().split('/')[-1] for l in f if l.strip()]

    print(f"  DDI test split: {len(test_stems)} samples")

    col_labels = ['Input + Ruler', 'Base MoGe-2\nDepth', 'DermDepth\nDepth', 'Base MoGe-2\nNormal', 'DermDepth\nNormal (D3)']
    n_cols = len(col_labels)
    rows_per_page = 5

    # Build row data
    rows = []
    for stem in sorted(test_stems):
        # Find image
        img_path = ddi_dir / 'images' / f'{stem}.png'
        if not img_path.exists():
            img_path = ddi_dir / 'images' / f'{stem}.jpg'
        if not img_path.exists():
            matches = list(ddi_dir.rglob(f'{stem}.png')) + list(ddi_dir.rglob(f'{stem}.jpg'))
            img_path = matches[0] if matches else None
        if img_path is None:
            print(f"  Skipping {stem}: image not found")
            continue

        img = np.array(Image.open(img_path).convert('RGB'))

        # Load FEDD labels for overlay
        label_path = label_index.get(stem)
        if label_path is not None:
            labels = np.load(label_path)
            labels_full = np.array(Image.fromarray(labels.astype(np.uint8)).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST))
            ruler_mask = (labels_full == 3)
            lesion_mask = (labels_full == 1)
        else:
            ruler_mask = np.zeros(img.shape[:2], dtype=bool)
            lesion_mask = np.zeros(img.shape[:2], dtype=bool)

        # Create overlay
        overlay = img.copy().astype(float) / 255.0
        if ruler_mask.sum() > 0:
            overlay[ruler_mask] = overlay[ruler_mask] * 0.5 + np.array([0.2, 0.4, 0.9]) * 0.5
        if lesion_mask.sum() > 0:
            from scipy.ndimage import binary_dilation, binary_erosion
            border = binary_dilation(lesion_mask, iterations=2) & ~binary_erosion(lesion_mask, iterations=1)
            overlay[border] = np.array([0.2, 0.9, 0.3])

        # Run inference
        base_depth, base_normal = run_inference(models['base'], img, device)
        ours_depth, _ = run_inference(models['exp_h'], img, device)
        _, d3_normal = run_inference(models['exp_d3'], img, device)

        # Ruler area ratios
        rm = ruler_meta.get(stem, {})
        base_ratio = rm.get('methods', {}).get('moge2', {}).get('ratio', 0)
        ours_ratio = rm.get('methods', {}).get('exp_h_s1800', {}).get('ratio', 0)
        tone = rm.get('skin_tone', '??')
        disease = rm.get('disease', 'unknown')
        tone_label = {'12': 'FP I-II', '34': 'FP III-IV', '56': 'FP V-VI'}.get(tone, tone)

        rows.append({
            'stem': stem,
            'overlay': overlay,
            'base_depth': base_depth,
            'base_normal': base_normal,
            'ours_depth': ours_depth,
            'd3_normal': d3_normal,
            'base_ratio': base_ratio,
            'ours_ratio': ours_ratio,
            'tone_label': tone_label,
            'disease': disease,
            'ruler_mask': ruler_mask,
        })

    print(f"  Rendering {len(rows)} rows")

    # Multi-page PDF
    pdf_path = OUT_DIR / 'fig4_ddi.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows_page = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows_page, n_cols, figsize=(n_cols * 2.5, n_rows_page * 2.5))
            if n_rows_page == 1:
                axes = axes[np.newaxis, :]

            for i, row in enumerate(page_rows):
                # DermDepth-centered colormap: vmin/vmax from DermDepth prediction
                ours_valid = np.isfinite(row['ours_depth']) & (row['ours_depth'] > 0)
                if ours_valid.sum() > 0:
                    vmin = np.percentile(row['ours_depth'][ours_valid], 2)
                    vmax = np.percentile(row['ours_depth'][ours_valid], 98)
                else:
                    vmin, vmax = None, None

                axes[i, 0].imshow(row['overlay'])
                axes[i, 0].set_ylabel(f"{row['tone_label']}\n{row['disease'][:20]}", fontsize=7,
                                     fontweight='bold', rotation=90, labelpad=6)
                if row['ruler_mask'].sum() > 0:
                    axes[i, 0].text(0.05, 0.08, 'Ruler: 6.6 cm$^2$', transform=axes[i, 0].transAxes,
                                   fontsize=5.5, color='white',
                                   bbox=dict(boxstyle='round,pad=0.12', facecolor='#2255aa', alpha=0.8),
                                   va='bottom')

                axes[i, 1].imshow(depth_to_color(row['base_depth'], vmin=vmin, vmax=vmax))
                add_area_badge(axes[i, 1], row['base_ratio'], good=False)

                axes[i, 2].imshow(depth_to_color(row['ours_depth'], vmin=vmin, vmax=vmax))
                add_area_badge(axes[i, 2], row['ours_ratio'], good=(0.5 < row['ours_ratio'] < 3.0))

                axes[i, 3].imshow(normal_to_color(row['base_normal']))
                axes[i, 4].imshow(normal_to_color(row['d3_normal']))

            for j, lbl in enumerate(col_labels):
                axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'DDI Test Set — Depth & Normal Comparison (page {page_num})',
                        fontsize=10, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    # PNG of first page
    page_rows = rows[:rows_per_page]
    n_rows_page = len(page_rows)
    fig, axes = plt.subplots(n_rows_page, n_cols, figsize=(n_cols * 2.5, n_rows_page * 2.5))
    if n_rows_page == 1:
        axes = axes[np.newaxis, :]
    for i, row in enumerate(page_rows):
        ours_valid = np.isfinite(row['ours_depth']) & (row['ours_depth'] > 0)
        vmin = np.percentile(row['ours_depth'][ours_valid], 2) if ours_valid.sum() > 0 else None
        vmax = np.percentile(row['ours_depth'][ours_valid], 98) if ours_valid.sum() > 0 else None
        axes[i, 0].imshow(row['overlay'])
        axes[i, 0].set_ylabel(f"{row['tone_label']}\n{row['disease'][:20]}", fontsize=7,
                             fontweight='bold', rotation=90, labelpad=6)
        axes[i, 1].imshow(depth_to_color(row['base_depth'], vmin=vmin, vmax=vmax))
        add_area_badge(axes[i, 1], row['base_ratio'], good=False)
        axes[i, 2].imshow(depth_to_color(row['ours_depth'], vmin=vmin, vmax=vmax))
        add_area_badge(axes[i, 2], row['ours_ratio'], good=(0.5 < row['ours_ratio'] < 3.0))
        axes[i, 3].imshow(normal_to_color(row['base_normal']))
        axes[i, 4].imshow(normal_to_color(row['d3_normal']))
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    fig.savefig(OUT_DIR / 'fig4_ddi.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {OUT_DIR / 'fig4_ddi.png'} (page 1)")


# ============================================================
# Fig 4B: DDI Baseline Depth Comparison (from cached predictions)
# ============================================================

def generate_fig4b(models=None, device='cuda'):
    """DDI: baseline depth predictions from cache. No inference needed."""
    print("\n=== Fig 4B: DDI Baseline Depths ===")

    ddi_dir = PROJECT_ROOT / 'data' / 'DDI'
    cache_dir = PROJECT_ROOT / 'output' / 'evaluation' / 'ddi_rulers' / '_cache'

    # Load ruler results for area ratios
    with open(PROJECT_ROOT / 'output' / 'evaluation' / 'ddi_rulers' / 'ddi_ruler_results.json') as f:
        ruler_data = json.load(f)
    ruler_meta = {s['stem']: s for s in ruler_data['per_sample']}

    # Test split
    with open(PROJECT_ROOT / 'data' / 'dermdepth_train' / 'ddi_moge' / 'test.txt') as f:
        test_stems = [l.strip().split('/')[-1] for l in f if l.strip()]

    methods = [
        ('moge2', 'Base MoGe-2'),
        ('da3nested', 'DA$^3$-Nested'),
        ('mapanything', 'MapAnything'),
        ('ppd', 'PPD'),
        ('exp_h_s1800', 'DermDepth'),
    ]
    col_labels = ['Input'] + [m[1] for m in methods]
    n_cols = len(col_labels)
    rows_per_page = 5

    rows = []
    for stem in sorted(test_stems):
        img_path = ddi_dir / 'images' / f'{stem}.png'
        if not img_path.exists():
            img_path = ddi_dir / 'images' / f'{stem}.jpg'
        if not img_path.exists():
            continue

        img = np.array(Image.open(img_path).convert('RGB'))

        # Load cached depths
        depths = {}
        for method_key, method_label in methods:
            depth_path = cache_dir / method_key / f'{stem}_depth.npy'
            if depth_path.exists():
                depths[method_key] = np.load(depth_path)
            else:
                depths[method_key] = None

        # Get area ratios
        rm = ruler_meta.get(stem, {})
        ratios = {}
        for method_key, _ in methods:
            ratios[method_key] = rm.get('methods', {}).get(method_key, {}).get('ratio', 0)

        tone = rm.get('skin_tone', '??')
        disease = rm.get('disease', 'unknown')
        tone_label = {'12': 'FP I-II', '34': 'FP III-IV', '56': 'FP V-VI'}.get(tone, tone)

        rows.append({
            'stem': stem,
            'img': img,
            'depths': depths,
            'ratios': ratios,
            'tone_label': tone_label,
            'disease': disease,
        })

    print(f"  Rendering {len(rows)} rows with {len(methods)} methods")

    # Multi-page PDF
    pdf_path = OUT_DIR / 'fig4b_ddi_baselines.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows_page = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows_page, n_cols, figsize=(n_cols * 2.2, n_rows_page * 2.2))
            if n_rows_page == 1:
                axes = axes[np.newaxis, :]

            for i, row in enumerate(page_rows):
                axes[i, 0].imshow(row['img'])
                axes[i, 0].set_ylabel(f"{row['tone_label']}\n{row['disease'][:20]}", fontsize=7,
                                     fontweight='bold', rotation=90, labelpad=6)

                for j, (method_key, method_label) in enumerate(methods):
                    d = row['depths'].get(method_key)
                    if d is not None:
                        # Self-normalized: each method uses its own depth range
                        axes[i, j+1].imshow(depth_to_color(d))
                    else:
                        axes[i, j+1].text(0.5, 0.5, 'N/A', transform=axes[i, j+1].transAxes,
                                         ha='center', va='center', fontsize=12, color='gray')

                    ratio = row['ratios'].get(method_key, 0)
                    if ratio > 0:
                        good = (0.5 < ratio < 3.0)
                        add_area_badge(axes[i, j+1], ratio, good=good)

            for j, lbl in enumerate(col_labels):
                axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'DDI Test Set — Method Comparison (page {page_num})',
                        fontsize=10, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    # PNG first page
    page_rows = rows[:rows_per_page]
    n_rows_page = len(page_rows)
    fig, axes = plt.subplots(n_rows_page, n_cols, figsize=(n_cols * 2.2, n_rows_page * 2.2))
    if n_rows_page == 1:
        axes = axes[np.newaxis, :]
    for i, row in enumerate(page_rows):
        axes[i, 0].imshow(row['img'])
        axes[i, 0].set_ylabel(f"{row['tone_label']}\n{row['disease'][:20]}", fontsize=7,
                             fontweight='bold', rotation=90, labelpad=6)
        for j, (method_key, method_label) in enumerate(methods):
            d = row['depths'].get(method_key)
            if d is not None:
                axes[i, j+1].imshow(depth_to_color(d))  # self-normalized
            ratio = row['ratios'].get(method_key, 0)
            if ratio > 0:
                add_area_badge(axes[i, j+1], ratio, good=(0.5 < ratio < 3.0))
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    fig.savefig(OUT_DIR / 'fig4b_ddi_baselines.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {OUT_DIR / 'fig4b_ddi_baselines.png'} (page 1)")


# ============================================================
# Shared: Baseline method definitions for B-versions
# ============================================================

PRED_DIR = PROJECT_ROOT / "output" / "figures" / "all_predictions"

BASELINE_METHODS = [
    ('moge2', 'Base MoGe-2'),
    ('da3nested', 'DA$^3$-Nested'),
    ('mapanything', 'MapAnything'),
    ('ppd', 'PPD'),
    ('exp_h', 'DermDepth'),
]


def load_cached_depth(dataset, sample_name, method_key):
    """Load a cached depth prediction from all_predictions/."""
    p = PRED_DIR / dataset / sample_name / f'{method_key}_depth.npy'
    if p.exists():
        return np.load(p)
    return None


def compute_scale_badge(pred, gt_depth, gt_mask):
    """Compute scale ratio between prediction and GT."""
    from scipy.ndimage import zoom as scipy_zoom
    if pred is None:
        return 0.0
    # Resize pred to match GT if needed
    if pred.shape != gt_depth.shape:
        pred = scipy_zoom(pred, (gt_depth.shape[0] / pred.shape[0],
                                  gt_depth.shape[1] / pred.shape[1]), order=1)
    valid = gt_mask & np.isfinite(gt_depth) & (gt_depth > 0) & np.isfinite(pred) & (pred > 0)
    if valid.sum() < 100:
        return 0.0
    return float(np.median(pred[valid]) / np.median(gt_depth[valid]))


# ============================================================
# Fig 1B: S-SYNTH Baseline Depth Comparison
# ============================================================

def generate_fig1b(models=None, device='cuda'):
    """S-SYNTH: baseline depth predictions from cache. Shows GT + all methods."""
    from moge.utils.io import read_depth

    print("\n=== Fig 1B: S-SYNTH Baseline Depths ===")

    synth_dir = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'colab_gen' / 'DermDepthSynth'
    index_file = synth_dir / '.index.txt'
    with open(index_file) as f:
        all_samples = [l.strip() for l in f if l.strip()]

    # Same selection as Fig 1 (seed=42, 2 per FP group)
    candidates = {'I-II': [], 'III-IV': [], 'V-VI': []}
    for sname in all_samples:
        params_file = synth_dir / sname / 'generation_params.json'
        if not params_file.exists():
            continue
        with open(params_file) as f:
            params = json.load(f)
        fp_group = params.get('fitzpatrick_group', 'III-IV')
        if fp_group not in candidates:
            fp_group = 'III-IV'
        candidates[fp_group].append(sname)

    np.random.seed(42)
    picks = []
    for group in ['I-II', 'III-IV', 'V-VI']:
        pool = candidates[group]
        if len(pool) >= 2:
            idxs = np.random.choice(len(pool), size=2, replace=False)
            picks.extend([pool[i] for i in idxs])
        elif pool:
            picks.append(pool[0])

    print(f"  Selected {len(picks)} samples")

    col_labels = ['Input', 'GT Depth'] + [m[1] for m in BASELINE_METHODS]
    n_cols = len(col_labels)
    n_rows = len(picks)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, sname in enumerate(picks):
        sdir = synth_dir / sname
        img_path = sdir / 'image.png'
        if not img_path.exists():
            img_path = sdir / 'image.jpg'
        img = np.array(Image.open(img_path))
        gt_depth_mm = read_depth(str(sdir / 'depth.png'))

        # GT depth in meters for scale comparison
        with open(sdir / 'meta.json') as f:
            meta = json.load(f)
        depth_unit = meta.get('depth_unit', 0.001)
        gt_depth_m = gt_depth_mm * depth_unit
        gt_mask = np.isfinite(gt_depth_mm) & (gt_depth_mm > 0)

        # FP label
        params_file = sdir / 'generation_params.json'
        fp_label = ''
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            fp_group = params.get('fitzpatrick_group', '')
            fp_label = f'FP {fp_group}' if fp_group else ''

        axes[i, 0].imshow(img)
        if fp_label:
            axes[i, 0].text(0.05, 0.92, fp_label, transform=axes[i, 0].transAxes,
                           fontsize=7, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6),
                           va='top')

        axes[i, 1].imshow(depth_to_color(gt_depth_mm, mask=gt_mask))
        add_depth_range_badge(axes[i, 1], gt_depth_mm, 'mm')

        for j, (method_key, _) in enumerate(BASELINE_METHODS):
            pred = load_cached_depth('synth', sname, method_key)
            if pred is not None:
                axes[i, j+2].imshow(depth_to_color(pred))
                scale = compute_scale_badge(pred, gt_depth_m, gt_mask)
                if scale > 0:
                    add_scale_badge(axes[i, j+2], scale)
            else:
                axes[i, j+2].text(0.5, 0.5, 'N/A', transform=axes[i, j+2].transAxes,
                                 ha='center', va='center', fontsize=12, color='gray')

    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig1b_ssynth_baselines.{ext}')
        print(f"  Saved: {OUT_DIR / f'fig1b_ssynth_baselines.{ext}'}")
    plt.close()


# ============================================================
# Fig 2B: SKINL2 Baseline Depth Comparison
# ============================================================

def generate_fig2b(models=None, device='cuda'):
    """SKINL2: ALL test cases, baseline depth predictions from cache. Multi-page PDF."""
    print("\n=== Fig 2B: SKINL2 ALL Baseline Depths ===")

    skinl2_dir = PROJECT_ROOT / 'output' / 'eval_data' / 'skinl2'

    split_file = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'skinl2_moge' / 'test.txt'
    if split_file.exists():
        with open(split_file) as f:
            test_samples = sorted([l.strip().split('/')[-1] for l in f if l.strip()])
    else:
        test_samples = sorted([d.name for d in skinl2_dir.iterdir()
                       if d.is_dir() and (d.name.startswith('v2_') or d.name.startswith('v3_'))])

    print(f"  Test samples: {len(test_samples)}")

    col_labels = ['Input', 'GT Depth'] + [m[1] for m in BASELINE_METHODS]
    n_cols = len(col_labels)
    rows_per_page = 6

    # Build row data
    rows = []
    for sample_name in test_samples:
        sdir = skinl2_dir / sample_name
        img_path = sdir / 'image.png'
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path))

        gt_depth = np.load(sdir / 'gt_depth.npy')
        gt_mask = np.load(sdir / 'gt_mask.npy') if (sdir / 'gt_mask.npy').exists() else np.ones_like(gt_depth, bool)
        valid = gt_mask & np.isfinite(gt_depth) & (gt_depth > 0)

        parts = sample_name.split('_', 1)
        version_tag = parts[0]
        disease = parts[1].rsplit('_', 1)[0] if len(parts) > 1 else ''

        depths = {}
        scales = {}
        for method_key, _ in BASELINE_METHODS:
            pred = load_cached_depth('skinl2', sample_name, method_key)
            depths[method_key] = pred
            scales[method_key] = compute_scale_badge(pred, gt_depth, gt_mask) if pred is not None else 0.0

        rows.append({
            'name': sample_name,
            'label': f'{version_tag} {disease[:18]}',
            'img': img,
            'gt_depth': gt_depth,
            'gt_mask': gt_mask,
            'valid': valid,
            'depths': depths,
            'scales': scales,
        })

    print(f"  Rendering {len(rows)} rows across {(len(rows) + rows_per_page - 1) // rows_per_page} pages")

    def render_skinl2b_page(page_rows, axes):
        for i, row in enumerate(page_rows):
            axes[i, 0].imshow(row['img'])
            axes[i, 0].set_ylabel(row['label'], fontsize=7, fontweight='bold', rotation=90, labelpad=8)
            axes[i, 1].imshow(depth_to_color(row['gt_depth'], mask=row['gt_mask']))
            if row['valid'].sum() > 0:
                add_depth_range_badge(axes[i, 1], row['gt_depth'][row['valid']] * 1000, 'mm')
            for j, (method_key, _) in enumerate(BASELINE_METHODS):
                d = row['depths'].get(method_key)
                if d is not None:
                    axes[i, j+2].imshow(depth_to_color(d))
                    scale = row['scales'].get(method_key, 0)
                    if scale > 0:
                        add_scale_badge(axes[i, j+2], scale)
                else:
                    axes[i, j+2].text(0.5, 0.5, 'N/A', transform=axes[i, j+2].transAxes,
                                     ha='center', va='center', fontsize=12, color='gray')

    # Multi-page PDF
    pdf_path = OUT_DIR / 'fig2b_skinl2_baselines.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows_page = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows_page, n_cols, figsize=(n_cols * 2.0, n_rows_page * 2.0))
            if n_rows_page == 1:
                axes = axes[np.newaxis, :]

            render_skinl2b_page(page_rows, axes)

            for j, lbl in enumerate(col_labels):
                axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'SKINL2 Test Set — Method Comparison (page {page_num})',
                        fontsize=10, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    # PNG first page
    page_rows = rows[:rows_per_page]
    n_rows_page = len(page_rows)
    fig, axes = plt.subplots(n_rows_page, n_cols, figsize=(n_cols * 2.0, n_rows_page * 2.0))
    if n_rows_page == 1:
        axes = axes[np.newaxis, :]
    render_skinl2b_page(page_rows, axes)
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    fig.savefig(OUT_DIR / 'fig2b_skinl2_baselines.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path} ({(len(rows) + rows_per_page - 1) // rows_per_page} pages)")
    print(f"  Saved: {OUT_DIR / 'fig2b_skinl2_baselines.png'} (page 1)")


# ============================================================
# Fig 3B: WoundsDB Baseline Depth Comparison (multi-page)
# ============================================================

def generate_fig3b(models=None, device='cuda'):
    """WoundsDB: ALL test cases, baseline depth predictions from cache."""
    print("\n=== Fig 3B: WoundsDB Baseline Depths ===")

    woundsdb_dir = PROJECT_ROOT / 'output' / 'eval_data' / 'woundsdb'

    # All test cases (case_id > 30)
    all_dirs = sorted([d for d in woundsdb_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])
    test_dirs = []
    for d in all_dirs:
        try:
            case_num = int(d.name.split('_')[1])
            if case_num > 30:
                test_dirs.append(d)
        except (ValueError, IndexError):
            continue

    print(f"  Found {len(test_dirs)} test cases")

    col_labels = ['Input', 'GT Depth'] + [m[1] for m in BASELINE_METHODS]
    n_cols = len(col_labels)
    rows_per_page = 6

    # Build row data
    rows = []
    for sdir in test_dirs:
        if not (sdir / 'image.png').exists() or not (sdir / 'gt_depth.npy').exists():
            continue
        img = np.array(Image.open(sdir / 'image.png'))
        gt_depth = np.load(sdir / 'gt_depth.npy')
        gt_mask_path = sdir / 'gt_mask.npy'
        gt_mask = np.load(gt_mask_path) if gt_mask_path.exists() else (np.isfinite(gt_depth) & (gt_depth > 0))
        valid = gt_mask & np.isfinite(gt_depth) & (gt_depth > 0)

        # Load cached predictions
        depths = {}
        scales = {}
        for method_key, _ in BASELINE_METHODS:
            pred = load_cached_depth('woundsdb', sdir.name, method_key)
            depths[method_key] = pred
            scales[method_key] = compute_scale_badge(pred, gt_depth, gt_mask) if pred is not None else 0.0

        label = sdir.name.replace('_scene_1', '').replace('_scene_2', ' s2')
        label = label.replace('_day_', ' d').replace('case_', 'C')

        rows.append({
            'name': sdir.name,
            'label': label,
            'img': img,
            'gt_depth': gt_depth,
            'gt_mask': gt_mask,
            'valid': valid,
            'depths': depths,
            'scales': scales,
        })

    print(f"  Rendering {len(rows)} rows across {(len(rows) + rows_per_page - 1) // rows_per_page} pages")

    def render_page(page_rows, axes):
        for i, row in enumerate(page_rows):
            axes[i, 0].imshow(row['img'])
            axes[i, 0].set_ylabel(row['label'], fontsize=7, fontweight='bold', rotation=90, labelpad=6)

            axes[i, 1].imshow(depth_to_color(row['gt_depth'], mask=row['gt_mask']))
            if row['valid'].sum() > 0:
                add_depth_range_badge(axes[i, 1], row['gt_depth'][row['valid']] * 1000, 'mm')

            for j, (method_key, _) in enumerate(BASELINE_METHODS):
                d = row['depths'].get(method_key)
                if d is not None:
                    axes[i, j+2].imshow(depth_to_color(d))
                    scale = row['scales'].get(method_key, 0)
                    if scale > 0:
                        add_scale_badge(axes[i, j+2], scale)
                else:
                    axes[i, j+2].text(0.5, 0.5, 'N/A', transform=axes[i, j+2].transAxes,
                                     ha='center', va='center', fontsize=12, color='gray')

    # Multi-page PDF
    pdf_path = OUT_DIR / 'fig3b_woundsdb_baselines.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows_page = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows_page, n_cols, figsize=(n_cols * 2.0, n_rows_page * 2.0))
            if n_rows_page == 1:
                axes = axes[np.newaxis, :]

            render_page(page_rows, axes)

            for j, lbl in enumerate(col_labels):
                axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'WoundsDB Test Set — Method Comparison (page {page_num})',
                        fontsize=10, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    # PNG first page
    page_rows = rows[:rows_per_page]
    n_rows_page = len(page_rows)
    fig, axes = plt.subplots(n_rows_page, n_cols, figsize=(n_cols * 2.0, n_rows_page * 2.0))
    if n_rows_page == 1:
        axes = axes[np.newaxis, :]
    render_page(page_rows, axes)
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    fig.savefig(OUT_DIR / 'fig3b_woundsdb_baselines.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path} ({(len(rows) + rows_per_page - 1) // rows_per_page} pages)")
    print(f"  Saved: {OUT_DIR / 'fig3b_woundsdb_baselines.png'} (page 1)")


# ============================================================
# Helper: S-SYNTH sample selection by FP group
# ============================================================

def select_synth_samples(seed=42):
    """Select 6 S-SYNTH samples (2 per FP group) with given seed."""
    synth_dir = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'colab_gen' / 'DermDepthSynth'
    index_file = synth_dir / '.index.txt'
    with open(index_file) as f:
        all_samples = [l.strip() for l in f if l.strip()]

    candidates = {'I-II': [], 'III-IV': [], 'V-VI': []}
    for sname in all_samples:
        params_file = synth_dir / sname / 'generation_params.json'
        if not params_file.exists():
            continue
        with open(params_file) as f:
            params = json.load(f)
        fp_group = params.get('fitzpatrick_group', 'III-IV')
        if fp_group not in candidates:
            fp_group = 'III-IV'
        candidates[fp_group].append(sname)

    np.random.seed(seed)
    picks = []
    for group in ['I-II', 'III-IV', 'V-VI']:
        pool = candidates[group]
        if len(pool) >= 2:
            idxs = np.random.choice(len(pool), size=2, replace=False)
            picks.extend([pool[i] for i in idxs])
        elif pool:
            picks.append(pool[0])
    return picks


# ============================================================
# Fig 1 v2 / 1B v2: alternate S-SYNTH batch (seed=123)
# ============================================================

def generate_fig1_v2(models, device='cuda'):
    """S-SYNTH v2: different random batch (seed=123), same FP group strategy."""
    from moge.utils.io import read_depth
    print("\n=== Fig 1 v2: S-SYNTH (alternate batch) ===")

    synth_dir = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'colab_gen' / 'DermDepthSynth'
    picks = select_synth_samples(seed=123)
    print(f"  Selected {len(picks)} samples")

    n_rows = len(picks)
    n_cols = 3
    col_labels = ['Input Image', 'GT Depth Map', 'GT Surface Normal']

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, sname in enumerate(picks):
        sdir = synth_dir / sname
        img_path = sdir / 'image.png'
        if not img_path.exists():
            img_path = sdir / 'image.jpg'
        img = np.array(Image.open(img_path))
        gt_depth_mm = read_depth(str(sdir / 'depth.png'))
        with open(sdir / 'meta.json') as f:
            meta = json.load(f)
        K = np.array(meta['intrinsics'])
        fx_px = K[0][0] * img.shape[1]
        fy_px = K[1][1] * img.shape[0]
        gt_normal = -depth_to_normal(gt_depth_mm, fx=fx_px, fy=fy_px)

        params_file = sdir / 'generation_params.json'
        fp_label = ''
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            fp_group = params.get('fitzpatrick_group', '')
            fp_label = f'FP {fp_group}' if fp_group else ''

        axes[i, 0].imshow(img)
        if fp_label:
            axes[i, 0].text(0.05, 0.92, fp_label, transform=axes[i, 0].transAxes,
                           fontsize=7, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6), va='top')
        gt_mask = np.isfinite(gt_depth_mm) & (gt_depth_mm > 0)
        axes[i, 1].imshow(depth_to_color(gt_depth_mm, mask=gt_mask))
        add_depth_range_badge(axes[i, 1], gt_depth_mm, 'mm')
        axes[i, 2].imshow(normal_to_color(gt_normal))

    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=10, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig1_ssynth_v2.{ext}')
        print(f"  Saved: {OUT_DIR / f'fig1_ssynth_v2.{ext}'}")
    plt.close()


def generate_fig1b_v2(models=None, device='cuda'):
    """S-SYNTH v2: baseline depth comparison with alternate batch (seed=123)."""
    from moge.utils.io import read_depth
    print("\n=== Fig 1B v2: S-SYNTH Baseline Depths (alternate batch) ===")

    synth_dir = PROJECT_ROOT / 'data' / 'dermdepth_train' / 'colab_gen' / 'DermDepthSynth'
    picks = select_synth_samples(seed=123)
    print(f"  Selected {len(picks)} samples")

    col_labels = ['Input', 'GT Depth'] + [m[1] for m in BASELINE_METHODS]
    n_cols = len(col_labels)
    n_rows = len(picks)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, sname in enumerate(picks):
        sdir = synth_dir / sname
        img_path = sdir / 'image.png'
        if not img_path.exists():
            img_path = sdir / 'image.jpg'
        img = np.array(Image.open(img_path))
        gt_depth_mm = read_depth(str(sdir / 'depth.png'))
        with open(sdir / 'meta.json') as f:
            meta = json.load(f)
        depth_unit = meta.get('depth_unit', 0.001)
        gt_depth_m = gt_depth_mm * depth_unit
        gt_mask = np.isfinite(gt_depth_mm) & (gt_depth_mm > 0)

        params_file = sdir / 'generation_params.json'
        fp_label = ''
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            fp_group = params.get('fitzpatrick_group', '')
            fp_label = f'FP {fp_group}' if fp_group else ''

        axes[i, 0].imshow(img)
        if fp_label:
            axes[i, 0].text(0.05, 0.92, fp_label, transform=axes[i, 0].transAxes,
                           fontsize=7, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6), va='top')
        axes[i, 1].imshow(depth_to_color(gt_depth_mm, mask=gt_mask))
        add_depth_range_badge(axes[i, 1], gt_depth_mm, 'mm')

        for j, (method_key, _) in enumerate(BASELINE_METHODS):
            pred = load_cached_depth('synth', sname, method_key)
            if pred is not None:
                axes[i, j+2].imshow(depth_to_color(pred))
                scale = compute_scale_badge(pred, gt_depth_m, gt_mask)
                if scale > 0:
                    add_scale_badge(axes[i, j+2], scale)
            else:
                axes[i, j+2].text(0.5, 0.5, 'N/A', transform=axes[i, j+2].transAxes,
                                 ha='center', va='center', fontsize=12, color='gray')

    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=9, fontweight='bold', pad=4)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig1b_ssynth_baselines_v2.{ext}')
        print(f"  Saved: {OUT_DIR / f'fig1b_ssynth_baselines_v2.{ext}'}")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--figs', nargs='*', default=['1', '2', '3', '4', '4b'],
                       help='Which figures to generate')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Only load models if needed (B-versions use cached depths)
    need_models = any(f in args.figs for f in ['1', '1v2', '2', '2supp', '2c', '3', '4', 'ddic'])
    models = load_models(device) if need_models else None

    fig_funcs = {
        '1': generate_fig1,
        '1v2': generate_fig1_v2,
        '2': generate_fig2,
        '2supp': generate_fig2_supp,
        '2c': generate_fig2c,
        'ddic': generate_fig_ddi_c,
        '3': generate_fig3,
        '4': generate_fig4,
        '4b': generate_fig4b,
        '1b': generate_fig1b,
        '1bv2': generate_fig1b_v2,
        '2b': generate_fig2b,
        '3b': generate_fig3b,
    }
    for f in args.figs:
        if f in fig_funcs:
            fig_funcs[f](models, device)

    print("\nDone!")
