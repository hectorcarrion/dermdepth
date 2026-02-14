#!/usr/bin/env python3
"""
Compare Raw vs Gaussian σ=15 vs NL-means+Guided on random cases from SKINL2 v1, v2, and v3.
Pick 5 cases per version (15 total), sampling different categories.
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
from scipy.ndimage import gaussian_filter
from pathlib import Path
import random
import os

data_root = Path('/workspace/hector/dermdepth/data/SKINL2')
out_dir = Path('/workspace/hector/dermdepth/output/verification')

random.seed(42)

def guided_filter(guide, src, radius=30, eps=0.005):
    ksize = (radius, radius)
    mg = cv2.boxFilter(guide, -1, ksize)
    ms = cv2.boxFilter(src, -1, ksize)
    mgs = cv2.boxFilter(guide * src, -1, ksize)
    mgg = cv2.boxFilter(guide * guide, -1, ksize)
    a = (mgs - mg * ms) / (mgg - mg * mg + eps)
    b = ms - a * mg
    return cv2.boxFilter(a, -1, ksize) * guide + cv2.boxFilter(b, -1, ksize)

def extract_relief(depth):
    elev = depth
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
    return elev - (coeffs[0] * xx + coeffs[1] * yy + coeffs[2])

def find_lesion(rgb):
    r, g, b = rgb[:,:,0].astype(float), rgb[:,:,1].astype(float), rgb[:,:,2].astype(float)
    darkness = 255 - (r + g + b) / 3.0
    redness = r / (g + 1)
    score = gaussian_filter(darkness * redness, sigma=30)
    py, px = np.unravel_index(np.argmax(score), score.shape)
    return py, px

def process_case(cv_path, dm_path):
    cv_img = np.array(Image.open(cv_path).convert('RGB'))
    depth_raw = np.array(Image.open(dm_path), dtype=np.float32)

    cv_small = np.array(Image.fromarray(cv_img).resize(
        (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

    peak_y, peak_x = find_lesion(cv_small)
    crop = 250
    h, w = depth_raw.shape
    y1, y2 = max(0, peak_y - crop), min(h, peak_y + crop)
    x1, x2 = max(0, peak_x - crop), min(w, peak_x + crop)

    depth_crop = depth_raw[y1:y2, x1:x2]
    rgb_crop = cv_small[y1:y2, x1:x2]

    relief_raw = extract_relief(depth_crop)
    relief_gauss = extract_relief(gaussian_filter(depth_crop, sigma=15))

    dmin, dmax = depth_crop.min(), depth_crop.max()
    if dmax - dmin < 1e-6:
        relief_nlm = relief_raw.copy()
    else:
        depth_u8 = ((depth_crop - dmin) / (dmax - dmin) * 255).astype(np.uint8)
        nlm_u8 = cv2.fastNlMeansDenoising(depth_u8, h=10, templateWindowSize=7, searchWindowSize=21)
        nlm_f32 = nlm_u8.astype(np.float32) / 255 * (dmax - dmin) + dmin
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        nlm_guided = guided_filter(gray, nlm_f32)
        relief_nlm = extract_relief(nlm_guided)

    return rgb_crop, relief_raw, relief_gauss, relief_nlm

# ============ Discover all cases ============
def discover_v1():
    """v1: organized by category folders"""
    cases = []
    cv_root = data_root / 'SKINL2_v1' / 'Central View'
    dm_root = data_root / 'SKINL2_v1' / 'DepthMap'
    for cat in sorted(os.listdir(cv_root)):
        cat_cv = cv_root / cat
        cat_dm = dm_root / cat
        if not cat_cv.is_dir() or not cat_dm.is_dir():
            continue
        for sample_id in sorted(os.listdir(cat_cv)):
            cv_files = list((cat_cv / sample_id).glob('*.png'))
            dm_files = list((cat_dm / sample_id).glob('*.tiff'))
            if cv_files and dm_files:
                cases.append(('v1', cat, sample_id, cv_files[0], dm_files[0]))
    return cases

def discover_v2v3(version):
    """v2/v3: organized by case ID / category"""
    cases = []
    root = data_root / f'SKINL2_{version}'
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

print("Discovering cases...")
v1_cases = discover_v1()
v2_cases = discover_v2v3('v2')
v3_cases = discover_v2v3('v3')

print(f"  v1: {len(v1_cases)} cases")
print(f"  v2: {len(v2_cases)} cases")
print(f"  v3: {len(v3_cases)} cases")

# Show categories per version
for name, cases_list in [('v1', v1_cases), ('v2', v2_cases), ('v3', v3_cases)]:
    cats = {}
    for _, cat, _, _, _ in cases_list:
        cats[cat] = cats.get(cat, 0) + 1
    print(f"  {name} categories: {dict(sorted(cats.items()))}")

# Select 5 random cases from each, trying to spread categories
def select_diverse(cases_list, n=5):
    by_cat = {}
    for c in cases_list:
        by_cat.setdefault(c[1], []).append(c)
    selected = []
    cats = sorted(by_cat.keys())
    # One from each category first
    for cat in cats:
        if len(selected) >= n:
            break
        selected.append(random.choice(by_cat[cat]))
    # Fill remaining randomly
    remaining = [c for c in cases_list if c not in selected]
    while len(selected) < n and remaining:
        choice = random.choice(remaining)
        selected.append(choice)
        remaining.remove(choice)
    return selected[:n]

sel_v1 = select_diverse(v1_cases, 5)
sel_v2 = select_diverse(v2_cases, 5)
sel_v3 = select_diverse(v3_cases, 5)

all_selected = sel_v1 + sel_v2 + sel_v3
print(f"\nSelected {len(all_selected)} cases:")
for ver, cat, sid, _, _ in all_selected:
    print(f"  {ver} / {cat} / {sid}")

# ============ Process and plot ============
n_cases = len(all_selected)
fig, axes = plt.subplots(n_cases, 4, figsize=(24, 4.5 * n_cases))

for row, (ver, cat, sid, cv_path, dm_path) in enumerate(all_selected):
    print(f"Processing {ver}/{cat}/{sid}...")
    try:
        rgb_crop, relief_raw, relief_gauss, relief_nlm = process_case(cv_path, dm_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        for col in range(4):
            axes[row, col].text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center', fontsize=8)
            axes[row, col].axis('off')
        continue

    all_relief = np.concatenate([relief_raw.ravel(), relief_gauss.ravel(), relief_nlm.ravel()])
    vmin = np.percentile(all_relief, 1)
    vmax = np.percentile(all_relief, 99)

    # RGB
    axes[row, 0].imshow(rgb_crop)
    axes[row, 0].set_title(f'{ver} | {cat}\n(ID: {sid})', fontsize=9, fontweight='bold')
    axes[row, 0].axis('off')

    # Raw
    axes[row, 1].imshow(relief_raw, cmap='hot', vmin=vmin, vmax=vmax)
    axes[row, 1].set_title(f'Raw\nstd={relief_raw.std():.3f}mm', fontsize=9)
    axes[row, 1].axis('off')

    # Gaussian σ=15
    axes[row, 2].imshow(relief_gauss, cmap='hot', vmin=vmin, vmax=vmax)
    axes[row, 2].set_title(f'Gaussian σ=15\nstd={relief_gauss.std():.3f}mm', fontsize=9)
    axes[row, 2].axis('off')

    # NL-means + Guided
    im = axes[row, 3].imshow(relief_nlm, cmap='hot', vmin=vmin, vmax=vmax)
    axes[row, 3].set_title(f'NL-means + Guided\nstd={relief_nlm.std():.3f}mm', fontsize=9)
    axes[row, 3].axis('off')
    plt.colorbar(im, ax=axes[row, 3], shrink=0.7, label='mm')

plt.suptitle('SKINL2 Denoising: Gaussian σ=15 vs NL-means+Guided Across v1, v2, v3\n'
             '(5 cases per version, diverse categories)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig35_skinl2_all_versions_denoise.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved fig35_skinl2_all_versions_denoise.png")
