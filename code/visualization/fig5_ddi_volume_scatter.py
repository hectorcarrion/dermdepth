#!/usr/bin/env python3
"""Fig 5: DDI Volume Scatter — Disease vs estimated lesion volume.

Shows per-sample lesion volume estimates from raw (uncalibrated) model predictions,
colored by skin tone. Compares Base MoGe-2 (wildly scattered) vs DermDepth (tight).
Below each panel: RGB thumbnails of largest/smallest volume samples per disease.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# MICCAI style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUT_DIR = Path("output/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths
DDI_DIR = Path("data/DDI")
FEDD_DIR = DDI_DIR / "FEDD" / "ddi_labels"
CACHE_DIR = Path("output/evaluation/ddi_rulers/_cache")
DDI_RESULTS = Path("output/evaluation/ddi_rulers/ddi_ruler_results.json")
VOLUME_RESULTS = Path("output/evaluation/ddi_rulers/lesion_volume_results.json")
DDI_TRAIN_SPLIT = Path("data/dermdepth_train/ddi_moge/train.txt")
DDI_TEST_SPLIT = Path("data/dermdepth_train/ddi_moge/test.txt")


def build_label_index():
    """Build index of FEDD label files (recursive search)."""
    index = {}
    for f in FEDD_DIR.rglob('*.npy'):
        index.setdefault(f.stem, f)
    return index

LABEL_INDEX = build_label_index()


def load_fedd_mask(stem, img_h, img_w, cls=1):
    """Load FEDD label and extract binary mask for given class."""
    label_path = LABEL_INDEX.get(stem)
    if label_path is None:
        return None
    labels = np.load(label_path)  # 256x256
    mask = (labels == cls).astype(np.uint8)
    mask_img = Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST)
    return np.array(mask_img).astype(bool)


def compute_lesion_volume(depth, lesion_mask, fov_deg=60.0):
    """Compute lesion bump volume from raw depth + mask. Returns volume in m^3."""
    h, w = depth.shape
    fx = fy = w / (2 * np.tan(np.radians(fov_deg / 2)))
    cx, cy = w / 2, h / 2

    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    X = (jj - cx) * depth / fx
    Y = (ii - cy) * depth / fy
    Z = depth

    valid_depth = np.isfinite(depth) & (depth > 0)
    lesion_valid = lesion_mask & valid_depth

    if lesion_valid.sum() < 10:
        return 0.0, 0.0

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
        return 0.0, 0.0

    a, b, c = coeffs
    plane_z = a * X[lesion_valid] + b * Y[lesion_valid] + c
    heights = Z[lesion_valid] - plane_z

    dXdx = np.zeros_like(X); dYdx = np.zeros_like(Y); dZdx = np.zeros_like(Z)
    dXdx[:, :-1] = X[:, 1:] - X[:, :-1]
    dYdx[:, :-1] = Y[:, 1:] - Y[:, :-1]
    dZdx[:, :-1] = Z[:, 1:] - Z[:, :-1]
    dXdy = np.zeros_like(X); dYdy = np.zeros_like(Y); dZdy = np.zeros_like(Z)
    dXdy[:-1, :] = X[1:, :] - X[:-1, :]
    dYdy[:-1, :] = Y[1:, :] - Y[:-1, :]
    dZdy[:-1, :] = Z[1:, :] - Z[:-1, :]
    nx = dYdx * dZdy - dZdx * dYdy
    ny = dZdx * dXdy - dXdx * dZdy
    nz = dXdx * dYdy - dYdx * dXdy
    area_elem = np.sqrt(nx**2 + ny**2 + nz**2)

    volume = float(np.sum(np.maximum(heights, 0) * area_elem[lesion_valid]))
    surface_area = float(np.sum(area_elem[lesion_valid]))
    return volume, surface_area


def load_ddi_thumbnail(stem, size=48):
    """Load DDI image as small thumbnail."""
    for ext in ['png', 'jpg']:
        p = DDI_DIR / 'images' / f'{stem}.{ext}'
        if p.exists():
            img = Image.open(p).convert('RGB')
            img.thumbnail((size, size))
            return np.array(img)
    return None


def main():
    with open(DDI_RESULTS) as f:
        ddi_data = json.load(f)
    with open(VOLUME_RESULTS) as f:
        vol_data = json.load(f)
    with open(DDI_TRAIN_SPLIT) as f:
        train_stems = set(l.strip().split('/')[-1] for l in f if l.strip())
    with open(DDI_TEST_SPLIT) as f:
        test_stems = set(l.strip().split('/')[-1] for l in f if l.strip())

    methods = ['moge2', 'exp_h_s1800', 'dermdepth']

    samples_data = []
    for ps in vol_data['per_sample']:
        stem = ps['stem']
        skin_tone = ps['skin_tone']
        disease = ps['disease']
        malignant = ps['malignant']
        img_h, img_w = ps['image_size']

        mask = load_fedd_mask(stem, img_h, img_w, cls=1)
        if mask is None or mask.sum() < 10:
            continue

        split = 'train' if stem in train_stems else ('test' if stem in test_stems else 'unknown')

        sample = {
            'stem': stem, 'skin_tone': skin_tone, 'disease': disease,
            'malignant': malignant, 'split': split, 'lesion_pixels': int(mask.sum()),
            'volumes': {},
        }

        for method in methods:
            depth_path = CACHE_DIR / method / f"{stem}_depth.npy"
            if not depth_path.exists():
                continue
            depth = np.load(depth_path)
            if depth.shape != (img_h, img_w):
                from scipy.ndimage import zoom
                depth = zoom(depth, (img_h / depth.shape[0], img_w / depth.shape[1]), order=1)
            vol, area = compute_lesion_volume(depth, mask)
            sample['volumes'][method] = {
                'volume_mm3': vol * 1e9,
                'surface_area_mm2': area * 1e6,
            }

        samples_data.append(sample)

    print(f"Computed volumes for {len(samples_data)} samples")

    # Disease display names
    disease_map = {
        'melanocytic-nevi': 'Melanocytic\nNevi',
        'seborrheic-keratosis': 'Seb.\nKeratosis',
        'verruca-vulgaris': 'Verruca\nVulgaris',
        'squamous-cell-carcinoma-in-situ': 'SCC\nin situ',
        'dermatofibroma': 'Dermato-\nfibroma',
        'inverted-follicular-keratosis': 'Inv. Foll.\nKeratosis',
        'epidermal-cyst': 'Epidermal\nCyst',
        'nevus-lipomatosus-superficialis': 'Nevus\nLipomatosus',
        'dysplastic-nevus': 'Dysplastic\nNevus',
        'basal-cell-carcinoma-superficial': 'BCC\nSuperficial',
        'foreign-body-granuloma': 'Foreign Body\nGranuloma',
        'acrochordon': 'Acrochordon',
        'squamous-cell-carcinoma-keratoacanthoma': 'SCC-KA',
        'eccrine-poroma': 'Eccrine\nPoroma',
        'prurigo-nodularis': 'Prurigo\nNodularis',
        'melanoma': 'Melanoma',
        'basal-cell-carcinoma': 'BCC',
    }

    tone_colors = {'12': '#F2D2A9', '34': '#C8956C', '56': '#6B3A2A'}
    tone_labels = {'12': 'FP I-II', '34': 'FP III-IV', '56': 'FP V-VI'}

    from collections import Counter
    disease_counts = Counter(s['disease'] for s in samples_data)
    diseases_sorted = [d for d, _ in disease_counts.most_common()]
    diseases_show = [d for d in diseases_sorted if disease_counts[d] >= 2]
    disease_to_x = {d: i for i, d in enumerate(diseases_show)}

    # Use DermDepth method for the extreme-volume RGB panel
    target_method = 'exp_h_s1800'

    # ============================================================
    # Main scatter plot (2 panels) + thumbnail strip below
    # ============================================================
    fig = plt.figure(figsize=(12, 6.5))

    # Top: scatter panels
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.2], hspace=0.35, wspace=0.08)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)
    ax_thumb_left = fig.add_subplot(gs[1, 0])
    ax_thumb_right = fig.add_subplot(gs[1, 1])

    np.random.seed(42)

    # Track per-disease extremes for DermDepth panel
    disease_extremes = {d: {'max_vol': -1, 'min_vol': 1e18,
                            'max_stem': None, 'min_stem': None} for d in diseases_show}

    for ax_idx, (method, method_label, ax) in enumerate([
        ('moge2', 'Base MoGe-2', ax_left),
        ('exp_h_s1800', 'DermDepth (Ours)', ax_right)
    ]):
        disease_vols = {d: [] for d in diseases_show}

        for s in samples_data:
            if s['disease'] not in disease_to_x:
                continue
            if method not in s['volumes']:
                continue

            x = disease_to_x[s['disease']]
            vol = s['volumes'][method]['volume_mm3']
            if vol <= 0:
                vol = 0.001

            disease_vols[s['disease']].append(vol)

            # Track extremes (on DermDepth only)
            if method == target_method:
                d = s['disease']
                if vol > disease_extremes[d]['max_vol']:
                    disease_extremes[d]['max_vol'] = vol
                    disease_extremes[d]['max_stem'] = s['stem']
                if vol < disease_extremes[d]['min_vol']:
                    disease_extremes[d]['min_vol'] = vol
                    disease_extremes[d]['min_stem'] = s['stem']

            tone = s['skin_tone']
            color = tone_colors[tone]
            marker = 'o' if s['split'] == 'test' else 's'
            # Add edge to ALL markers for consistency
            edge = '#333'
            lw = 0.6

            jitter = np.random.uniform(-0.2, 0.2)
            ax.scatter(x + jitter, vol, c=color, marker=marker, s=50,
                      edgecolors=edge, linewidths=lw, alpha=0.85, zorder=5)

        for d in diseases_show:
            if disease_vols[d]:
                med = np.median(disease_vols[d])
                x = disease_to_x[d]
                ax.plot([x - 0.3, x + 0.3], [med, med], color='#333',
                       linewidth=1.5, zorder=6, alpha=0.7)

        ax.set_yscale('log')
        ax.set_xticks(range(len(diseases_show)))
        ax.set_xticklabels([disease_map.get(d, d) for d in diseases_show],
                          rotation=45, ha='right', fontsize=7)
        ax.set_title(method_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.15, axis='y')
        ax.set_xlim(-0.5, len(diseases_show) - 0.5)

        all_vols = [v for vlist in disease_vols.values() for v in vlist]
        if all_vols:
            med_all = np.median(all_vols)
            ax.axhline(y=med_all, color='#7f8c8d', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.text(len(diseases_show) - 0.6, med_all * 1.3,
                   f'Median: {med_all:.1f} mm$^3$', fontsize=7, color='#7f8c8d',
                   ha='right', va='bottom')

    ax_left.set_ylabel('Estimated Lesion Volume (mm$^3$)', fontsize=10)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Legend
    tone_handles = [Line2D([0], [0], marker='o', color=tone_colors[t], markersize=7,
                           markeredgecolor='#333', markeredgewidth=0.6,
                           linestyle='None', label=tone_labels[t]) for t in ['12', '34', '56']]
    split_handles = [
        Line2D([0], [0], marker='s', color='gray', markersize=6, markeredgecolor='#333',
               markeredgewidth=0.6, linestyle='None', label='Train'),
        Line2D([0], [0], marker='o', color='gray', markersize=6, markeredgecolor='#333',
               markeredgewidth=0.6, linestyle='None', label='Test'),
    ]
    med_handle = [Line2D([0], [0], color='#333', linewidth=1.5, label='Median')]
    ax_right.legend(handles=tone_handles + split_handles + med_handle, loc='upper right',
                   fontsize=7.5, framealpha=0.95, ncol=1, edgecolor='#bdc3c7')

    # ============================================================
    # Thumbnail strips: largest (top row) and smallest (bottom row) per disease
    # ============================================================
    for ax_thumb, label_text in [(ax_thumb_left, 'Largest vol.'), (ax_thumb_right, 'Smallest vol.')]:
        ax_thumb.set_xlim(-0.5, len(diseases_show) - 0.5)
        ax_thumb.set_ylim(-0.8, 1.8)
        ax_thumb.set_xticks([])
        ax_thumb.set_yticks([])
        ax_thumb.set_frame_on(False)

    # Place thumbnails with disease labels
    for d_idx, d in enumerate(diseases_show):
        ex = disease_extremes[d]
        disease_short = disease_map.get(d, d).replace('\n', ' ')

        # Largest volume thumbnail (left panel)
        if ex['max_stem']:
            thumb = load_ddi_thumbnail(ex['max_stem'], size=40)
            if thumb is not None:
                im = OffsetImage(thumb, zoom=0.7)
                ab = AnnotationBbox(im, (d_idx, 1.0), frameon=True,
                                   bboxprops=dict(edgecolor='#c0392b', linewidth=1.2),
                                   xycoords='data', box_alignment=(0.5, 0.5))
                ax_thumb_left.add_artist(ab)
            # Disease label below thumbnail
            ax_thumb_left.text(d_idx, -0.1, disease_short, fontsize=5.5,
                              ha='center', va='top', fontweight='bold', color='#555',
                              rotation=30)

        # Smallest volume thumbnail (right panel)
        if ex['min_stem']:
            thumb = load_ddi_thumbnail(ex['min_stem'], size=40)
            if thumb is not None:
                im = OffsetImage(thumb, zoom=0.7)
                ab = AnnotationBbox(im, (d_idx, 1.0), frameon=True,
                                   bboxprops=dict(edgecolor='#27ae60', linewidth=1.2),
                                   xycoords='data', box_alignment=(0.5, 0.5))
                ax_thumb_right.add_artist(ab)
            # Disease label below thumbnail
            ax_thumb_right.text(d_idx, -0.1, disease_short, fontsize=5.5,
                               ha='center', va='top', fontweight='bold', color='#555',
                               rotation=30)

    ax_thumb_left.set_ylabel('Largest\nvol.', fontsize=7, fontweight='bold', rotation=0,
                             labelpad=25, va='center')
    ax_thumb_right.set_ylabel('Smallest\nvol.', fontsize=7, fontweight='bold', rotation=0,
                              labelpad=25, va='center')

    fig.suptitle('DDI Lesion Volume Estimates by Disease and Skin Tone',
                 fontsize=13, fontweight='bold', y=0.98)

    for ext in ['pdf', 'png']:
        fig.savefig(OUT_DIR / f'fig5_ddi_volume_scatter.{ext}')
        print(f'Saved: {OUT_DIR / f"fig5_ddi_volume_scatter.{ext}"}')

    plt.close()
    print('Done!')


if __name__ == '__main__':
    main()
