#!/usr/bin/env python3
"""Fig 4C: DDI Lesion Measurements — calibrated area, width, and volume per method.

For each DDI test image:
  - Overlay FEDD lesion mask on image
  - Compute ruler-calibrated lesion area and width for all methods
  - Show color-coded measurement labels

No GPU needed — uses cached depth predictions.

Usage:
    conda run -n MoGe python -u code/visualization/fig4c_ddi_lesion_measurements.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import json
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label as ndlabel
from scipy.ndimage import zoom as scipy_zoom

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "figures" / "paper_qualitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DDI_DIR = PROJECT_ROOT / 'data' / 'DDI'
FEDD_DIR = DDI_DIR / 'FEDD' / 'ddi_labels'
CACHE_DIR = PROJECT_ROOT / 'output' / 'evaluation' / 'ddi_rulers' / '_cache'
DDI_RESULTS = PROJECT_ROOT / 'output' / 'evaluation' / 'ddi_rulers' / 'ddi_ruler_results.json'
VOL_RESULTS = PROJECT_ROOT / 'output' / 'evaluation' / 'ddi_rulers' / 'lesion_volume_results.json'
GT_RULER_AREA_CM2 = 6.6  # 6cm x 1.1cm ruler

# Methods to display (key in ruler_results, display label, color)
METHODS = [
    ('moge2',       'MoGe-2',        '#3498db'),
    ('da3nested',   'DA$^3$',        '#e74c3c'),
    ('mapanything', 'MapAnything',    '#9b59b6'),
    ('ppd',         'PPD',           '#795548'),
    ('exp_h_s1800', 'DermDepth',     '#27ae60'),
]


def build_label_index():
    index = {}
    for f in FEDD_DIR.rglob('*.npy'):
        index.setdefault(f.stem, f)
    return index

LABEL_INDEX = build_label_index()


def load_fedd_mask(stem, img_h, img_w, cls=1):
    label_path = LABEL_INDEX.get(stem)
    if label_path is None:
        return None
    labels = np.load(label_path)
    mask = (labels == cls).astype(np.uint8)
    mask_full = np.array(Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST)).astype(bool)
    return mask_full


def estimate_intrinsics(h, w, fov_deg=60.0):
    fx = fy = w / (2 * np.tan(np.radians(fov_deg / 2)))
    return fx, fy, w / 2.0, h / 2.0


def compute_lesion_measurements(depth, lesion_mask, fov_deg=60.0):
    """Compute raw (uncalibrated) 3D area, volume, and width of lesion.

    Returns dict with area_m2, volume_m3, width_m (major axis length).
    """
    h, w = depth.shape
    fx, fy, cx, cy = estimate_intrinsics(h, w, fov_deg)

    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    X = (jj - cx) * depth / fx
    Y = (ii - cy) * depth / fy
    Z = depth.copy()

    valid_depth = np.isfinite(depth) & (depth > 0)
    lesion_valid = lesion_mask & valid_depth

    if lesion_valid.sum() < 10:
        return {'area_m2': 0, 'volume_m3': 0, 'width_m': 0}

    # --- Surface area via cross product of tangent vectors ---
    dXdx = np.zeros_like(X); dXdx[:, :-1] = X[:, 1:] - X[:, :-1]
    dYdx = np.zeros_like(Y); dYdx[:, :-1] = Y[:, 1:] - Y[:, :-1]
    dZdx = np.zeros_like(Z); dZdx[:, :-1] = Z[:, 1:] - Z[:, :-1]
    dXdy = np.zeros_like(X); dXdy[:-1] = X[1:] - X[:-1]
    dYdy = np.zeros_like(Y); dYdy[:-1] = Y[1:] - Y[:-1]
    dZdy = np.zeros_like(Z); dZdy[:-1] = Z[1:] - Z[:-1]

    nx = dYdx * dZdy - dZdx * dYdy
    ny = dZdx * dXdy - dXdx * dZdy
    nz = dXdx * dYdy - dYdx * dXdy
    area_elem = np.sqrt(nx**2 + ny**2 + nz**2)

    area_m2 = float(np.sum(area_elem[lesion_valid]))

    # --- Volume (bump above fitted plane) ---
    dilated = binary_dilation(lesion_mask, iterations=3)
    eroded = binary_erosion(lesion_mask, iterations=3)
    boundary = dilated & ~eroded & valid_depth
    if boundary.sum() < 3:
        boundary = lesion_valid

    bx, by, bz = X[boundary], Y[boundary], Z[boundary]
    A_mat = np.column_stack([bx, by, np.ones_like(bx)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, bz, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.array([0, 0, np.median(bz)])
    a, b, c = coeffs
    plane_z = a * X[lesion_valid] + b * Y[lesion_valid] + c
    heights = Z[lesion_valid] - plane_z
    volume_m3 = float(np.sum(np.maximum(heights, 0) * area_elem[lesion_valid]))

    # --- Width: major axis via PCA on 3D lesion points ---
    pts = np.column_stack([X[lesion_valid], Y[lesion_valid], Z[lesion_valid]])
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    try:
        _, S, Vt = np.linalg.svd(pts_c, full_matrices=False)
        # Project onto first principal component
        proj = pts_c @ Vt[0]
        width_m = float(proj.max() - proj.min())
    except np.linalg.LinAlgError:
        width_m = 0.0

    return {'area_m2': area_m2, 'volume_m3': volume_m3, 'width_m': width_m}


def calibrate(raw, ruler_area_cm2):
    """Apply ruler calibration to raw measurements.
    Scale factor for areas: k = GT_RULER / predicted_ruler.
    For lengths: sqrt(k). For volumes: k^(3/2).
    """
    if ruler_area_cm2 <= 0:
        return {'area_cm2': 0, 'volume_mm3': 0, 'width_mm': 0}
    k_area = GT_RULER_AREA_CM2 / ruler_area_cm2
    k_length = np.sqrt(k_area)
    k_volume = k_area * k_length

    area_cm2 = raw['area_m2'] * 1e4 * k_area
    volume_mm3 = raw['volume_m3'] * 1e9 * k_volume
    width_mm = raw['width_m'] * 1e3 * k_length

    return {'area_cm2': area_cm2, 'volume_mm3': volume_mm3, 'width_mm': width_mm}


def create_lesion_overlay(img, lesion_mask, ruler_mask=None):
    """Create image with lesion mask overlay (green border + semi-transparent fill)."""
    overlay = img.copy().astype(float) / 255.0

    # Semi-transparent green fill over lesion
    if lesion_mask.sum() > 0:
        overlay[lesion_mask] = overlay[lesion_mask] * 0.6 + np.array([0.2, 0.85, 0.3]) * 0.4
        # Bright border
        border = binary_dilation(lesion_mask, iterations=2) & ~binary_erosion(lesion_mask, iterations=1)
        overlay[border] = np.array([0.1, 1.0, 0.2])

    # Blue highlight for ruler
    if ruler_mask is not None and ruler_mask.sum() > 0:
        overlay[ruler_mask] = overlay[ruler_mask] * 0.5 + np.array([0.2, 0.4, 0.9]) * 0.5

    return overlay.clip(0, 1)


def main():
    print("Loading data...")
    with open(DDI_RESULTS) as f:
        ruler_data = json.load(f)
    ruler_meta = {s['stem']: s for s in ruler_data['per_sample']}

    with open(VOL_RESULTS) as f:
        vol_data = json.load(f)
    vol_meta = {s['stem']: s for s in vol_data['per_sample']}

    with open(PROJECT_ROOT / 'data' / 'dermdepth_train' / 'ddi_moge' / 'test.txt') as f:
        test_stems = sorted([l.strip().split('/')[-1] for l in f if l.strip()])

    print(f"  DDI test: {len(test_stems)} samples")

    # Build row data
    rows = []
    for stem in test_stems:
        # Find image
        img_path = DDI_DIR / 'images' / f'{stem}.png'
        if not img_path.exists():
            img_path = DDI_DIR / 'images' / f'{stem}.jpg'
        if not img_path.exists():
            print(f"  Skip {stem}: no image")
            continue

        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]

        # Load masks
        lesion_mask = load_fedd_mask(stem, h, w, cls=1)
        ruler_mask = load_fedd_mask(stem, h, w, cls=3)
        if lesion_mask is None or lesion_mask.sum() < 10:
            print(f"  Skip {stem}: no lesion mask")
            continue

        rm = ruler_meta.get(stem, {})
        vm = vol_meta.get(stem, {})

        tone = rm.get('skin_tone', '??')
        disease = rm.get('disease', 'unknown')
        tone_label = {'12': 'FP I-II', '34': 'FP III-IV', '56': 'FP V-VI'}.get(tone, tone)

        # Create overlay image
        overlay = create_lesion_overlay(img, lesion_mask, ruler_mask)

        # Compute measurements per method
        method_results = {}
        for method_key, method_label, color in METHODS:
            depth_path = CACHE_DIR / method_key / f'{stem}_depth.npy'
            if not depth_path.exists():
                # Try the all_predictions cache
                depth_path = PROJECT_ROOT / 'output' / 'figures' / 'all_predictions' / 'ddi' / stem / f'{method_key}_depth.npy'
            if not depth_path.exists():
                method_results[method_key] = None
                continue

            depth = np.load(depth_path)
            # Resize depth to match image
            if depth.shape != (h, w):
                depth = scipy_zoom(depth, (h / depth.shape[0], w / depth.shape[1]), order=1)

            raw = compute_lesion_measurements(depth, lesion_mask)

            # Get ruler calibration
            ruler_area = rm.get('methods', {}).get(method_key, {}).get('area_cm2', 0)
            cal = calibrate(raw, ruler_area)

            method_results[method_key] = {
                'raw': raw,
                'calibrated': cal,
                'ruler_area': ruler_area,
                'ruler_ratio': rm.get('methods', {}).get(method_key, {}).get('ratio', 0),
            }

        rows.append({
            'stem': stem,
            'overlay': overlay,
            'tone_label': tone_label,
            'disease': disease,
            'lesion_pixels': int(lesion_mask.sum()),
            'method_results': method_results,
        })

    print(f"  Rendering {len(rows)} samples")

    # Layout: 1 row per sample
    # Col 1: Overlay image with lesion mask
    # Col 2: Measurement table (area, width, volume per method, color-coded)
    n_cols = 2

    pdf_path = OUT_DIR / 'fig4c_ddi_measurements.pdf'
    rows_per_page = 4

    def fmt_area(v):
        if v <= 0: return 'N/A'
        return f'{v:.2f}' if v < 100 else f'{v:.1f}'

    def fmt_width(v):
        if v <= 0: return 'N/A'
        return f'{v:.1f}' if v < 100 else f'{v:.0f}'

    def fmt_vol(v):
        if v <= 0: return 'N/A'
        if v < 0.01: return f'{v:.4f}'
        if v < 1: return f'{v:.3f}'
        if v < 100: return f'{v:.2f}'
        return f'{v:.1f}'

    def render_page(page_rows, fig, axes):
        for i, row in enumerate(page_rows):
            ax_img = axes[i, 0]
            ax_table = axes[i, 1]

            # Image with lesion overlay
            ax_img.imshow(row['overlay'])
            ax_img.set_ylabel(f"{row['tone_label']}\n{row['disease'][:22]}", fontsize=6.5,
                             fontweight='bold', rotation=90, labelpad=6)

            # DermDepth measurement badge directly on image
            dd = row['method_results'].get('exp_h_s1800')
            if dd is not None:
                cal = dd['calibrated']
                badge = f"A={fmt_area(cal['area_cm2'])} cm\u00b2  W={fmt_width(cal['width_mm'])} mm  V={fmt_vol(cal['volume_mm3'])} mm\u00b3"
                ax_img.text(0.50, 0.04, badge, transform=ax_img.transAxes, fontsize=5.5,
                           fontweight='bold', color='white', ha='center',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='#27ae60', alpha=0.85),
                           va='bottom')

            # ---- Measurement table with horizontal bars ----
            ax_table.set_xlim(0, 1)
            ax_table.set_ylim(0, 1)
            ax_table.set_frame_on(False)

            # Collect values for bar normalization
            areas = []
            widths = []
            vols = []
            for mk, _, _ in METHODS:
                mr = row['method_results'].get(mk)
                if mr is not None:
                    c = mr['calibrated']
                    areas.append(c['area_cm2'])
                    widths.append(c['width_mm'])
                    vols.append(c['volume_mm3'])

            # Use DermDepth as reference for bar length (so its bar = full width)
            dd_cal = dd['calibrated'] if dd else None
            ref_area = dd_cal['area_cm2'] if dd_cal and dd_cal['area_cm2'] > 0 else (np.median(areas) if areas else 1)
            ref_width = dd_cal['width_mm'] if dd_cal and dd_cal['width_mm'] > 0 else (np.median(widths) if widths else 1)
            ref_vol = dd_cal['volume_mm3'] if dd_cal and dd_cal['volume_mm3'] > 0 else (np.median(vols) if vols else 1)

            # Column headers
            header_y = 0.95
            sections = [
                ('Area (cm$^2$)', 0.0, 0.30),
                ('Width (mm)', 0.34, 0.64),
                ('Volume (mm$^3$)', 0.68, 0.98),
            ]
            for title, x_start, x_end in sections:
                ax_table.text((x_start + x_end) / 2, header_y, title,
                             fontsize=6.5, fontweight='bold', ha='center', va='top', color='#333')

            ax_table.axhline(y=0.895, xmin=0.0, xmax=0.98, color='#ccc', linewidth=0.5)

            # Method rows with bars
            y_start = 0.85
            dy = 0.155
            bar_h = 0.065

            for j, (method_key, method_label, color) in enumerate(METHODS):
                y = y_start - j * dy
                mr = row['method_results'].get(method_key)

                if mr is None:
                    for title, x_start, x_end in sections:
                        ax_table.text(x_start + 0.01, y - bar_h / 2, f'{method_label}: N/A',
                                     fontsize=5.5, color='#999', va='center')
                    continue

                cal = mr['calibrated']
                vals = [cal['area_cm2'], cal['width_mm'], cal['volume_mm3']]
                refs = [ref_area, ref_width, ref_vol]
                fmts = [fmt_area, fmt_width, fmt_vol]

                for (title, x_start, x_end), val, ref, fmt in zip(sections, vals, refs, fmts):
                    bar_width = x_end - x_start - 0.02
                    # Bar length relative to reference (DermDepth), capped at 2x
                    frac = min(val / ref, 2.0) / 2.0 if ref > 0 else 0
                    bw = bar_width * frac

                    # Draw bar
                    rect = plt.Rectangle((x_start + 0.01, y - bar_h), bw, bar_h,
                                        facecolor=color, alpha=0.3, edgecolor='none')
                    ax_table.add_patch(rect)

                    # Label text
                    label = f'{method_label}: {fmt(val)}'
                    ax_table.text(x_start + 0.015, y - bar_h / 2, label,
                                fontsize=5.5, fontweight='bold', color=color, va='center')

                # Ruler ratio in parentheses
                ratio = mr.get('ruler_ratio', 0)
                if ratio > 0:
                    ratio_color = '#27ae60' if 0.5 < ratio < 2.0 else '#c0392b'
                    ax_table.text(0.99, y - bar_h / 2, f'ruler: {ratio:.1f}x',
                                 fontsize=4.5, color=ratio_color, va='center', ha='right', alpha=0.6)

    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(rows), rows_per_page):
            page_rows = rows[page_start:page_start + rows_per_page]
            n_rows = len(page_rows)
            page_num = page_start // rows_per_page + 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 2.5),
                                     gridspec_kw={'width_ratios': [1, 1.3]})
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            render_page(page_rows, fig, axes)

            for ax in axes[:, 0]:
                ax.set_xticks([]); ax.set_yticks([])
            for ax in axes[:, 1]:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f'DDI Test Set — Calibrated Lesion Measurements (page {page_num})',
                        fontsize=10, fontweight='bold', y=1.01)
            plt.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
            pdf.savefig(fig)
            plt.close(fig)

    # PNG of first page
    page_rows = rows[:rows_per_page]
    n_rows = len(page_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 2.5),
                             gridspec_kw={'width_ratios': [1, 1.3]})
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    render_page(page_rows, fig, axes)
    for ax in axes[:, 0]:
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[:, 1]:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    fig.savefig(OUT_DIR / 'fig4c_ddi_measurements.png')
    plt.close(fig)

    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {OUT_DIR / 'fig4c_ddi_measurements.png'}")
    print("Done!")


if __name__ == '__main__':
    main()
