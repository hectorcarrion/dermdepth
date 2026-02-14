#!/usr/bin/env python3
"""
Compare Raw vs Gaussian σ=15 vs NL-means+Guided on random cases from SKINL2 v1, v2, v3.
Full images (no crop), with 3D reconstructions.
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

def process_case(cv_path, dm_path):
    cv_img = np.array(Image.open(cv_path).convert('RGB'))
    depth_raw = np.array(Image.open(dm_path), dtype=np.float32)

    # Resize central view to depth resolution
    cv_small = np.array(Image.fromarray(cv_img).resize(
        (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

    # Full image, no crop
    relief_raw = extract_relief(depth_raw)
    relief_gauss = extract_relief(gaussian_filter(depth_raw, sigma=15))

    dmin, dmax = depth_raw.min(), depth_raw.max()
    if dmax - dmin < 1e-6:
        relief_nlm = relief_raw.copy()
    else:
        depth_u8 = ((depth_raw - dmin) / (dmax - dmin) * 255).astype(np.uint8)
        nlm_u8 = cv2.fastNlMeansDenoising(depth_u8, h=10, templateWindowSize=7, searchWindowSize=21)
        nlm_f32 = nlm_u8.astype(np.float32) / 255 * (dmax - dmin) + dmin
        gray = cv2.cvtColor(cv_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        nlm_guided = guided_filter(gray, nlm_f32)
        relief_nlm = extract_relief(nlm_guided)

    return cv_small, relief_raw, relief_gauss, relief_nlm

# ============ Discover cases ============
def discover_v1():
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
print(f"  v1: {len(v1_cases)}, v2: {len(v2_cases)}, v3: {len(v3_cases)}")

def select_diverse(cases_list, n=5):
    by_cat = {}
    for c in cases_list:
        by_cat.setdefault(c[1], []).append(c)
    selected = []
    cats = sorted(by_cat.keys())
    for cat in cats:
        if len(selected) >= n:
            break
        selected.append(random.choice(by_cat[cat]))
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

print(f"Selected {len(all_selected)} cases")

# ============ Plot per version: 5 rows × 5 cols (RGB, Raw, Gauss, NLM, 3D) ============
for ver_name, ver_cases in [('v1', sel_v1), ('v2', sel_v2), ('v3', sel_v3)]:
    n = len(ver_cases)
    fig = plt.figure(figsize=(30, 6 * n))

    for row, (ver, cat, sid, cv_path, dm_path) in enumerate(ver_cases):
        print(f"Processing {ver}/{cat}/{sid}...")
        try:
            rgb, relief_raw, relief_gauss, relief_nlm = process_case(cv_path, dm_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        all_vals = np.concatenate([relief_gauss.ravel(), relief_nlm.ravel()])
        vmin = np.percentile(all_vals, 1)
        vmax = np.percentile(all_vals, 99)

        # Col 0: RGB
        ax = fig.add_subplot(n, 5, row * 5 + 1)
        ax.imshow(rgb)
        ax.set_title(f'{cat}\n({ver}, ID:{sid})', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Col 1: Raw relief
        ax = fig.add_subplot(n, 5, row * 5 + 2)
        ax.imshow(relief_raw, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(f'Raw\nstd={relief_raw.std():.3f}mm', fontsize=10)
        ax.axis('off')

        # Col 2: Gaussian σ=15
        ax = fig.add_subplot(n, 5, row * 5 + 3)
        ax.imshow(relief_gauss, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(f'Gaussian σ=15\nstd={relief_gauss.std():.3f}mm', fontsize=10)
        ax.axis('off')

        # Col 3: NL-means + Guided
        ax = fig.add_subplot(n, 5, row * 5 + 4)
        im = ax.imshow(relief_nlm, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(f'NL-means + Guided\nstd={relief_nlm.std():.3f}mm', fontsize=10)
        ax.axis('off')

        # Col 4: 3D reconstruction (Gaussian σ=15)
        ax3d = fig.add_subplot(n, 5, row * 5 + 5, projection='3d')
        step = max(relief_gauss.shape[0] // 150, 1)
        Z = relief_gauss[::step, ::step]
        rgb_sub = rgb[::step, ::step].astype(np.float64) / 255.0
        ys = np.arange(Z.shape[0]) * step
        xs = np.arange(Z.shape[1]) * step
        X, Y = np.meshgrid(xs, ys)
        ax3d.plot_surface(X, Y, Z,
                          facecolors=rgb_sub, rstride=1, cstride=1, shade=True,
                          lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
                          antialiased=True)
        ax3d.view_init(elev=55, azim=-55)
        ax3d.set_box_aspect([1, Z.shape[0] / max(Z.shape[1], 1), 0.25])
        ax3d.set_xticks([]); ax3d.set_yticks([])
        ax3d.set_zlabel('mm')
        ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
        ax3d.set_title('3D (Gaussian σ=15)', fontsize=10)

    plt.suptitle(f'SKINL2 {ver_name.upper()} — Denoising Comparison (Full Image)\n'
                 'RGB | Raw | Gaussian σ=15 | NL-means+Guided | 3D Reconstruction',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_num = {'v1': 35, 'v2': 36, 'v3': 37}[ver_name]
    path = out_dir / f'fig{fig_num}_skinl2_{ver_name}_denoise_full.png'
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved {path.name}")
