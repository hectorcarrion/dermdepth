#!/usr/bin/env python3
"""
Compare Raw vs Gaussian σ=15 vs NL-means+Guided denoising on 5 SKINL2 cases
from different clinical categories. 3 columns (methods) × 5 rows (cases),
each showing depth map + 3D surface.
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

data_root = Path('/workspace/hector/dermdepth/data/SKINL2/SKINL2_v1')
out_dir = Path('/workspace/hector/dermdepth/output/verification')

cases = [
    ('Basal-cell Carcinoma', '0001'),
    ('Melanoma', '0101'),
    ('Nevus', '0004'),
    ('Psoriasis', '0235'),
    ('Seborrheic Keratosis', '0016'),
]

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
    elev = depth  # closer = less negative = bump UP
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

def process_case(category, sample_id):
    cv_path = list((data_root / 'Central View' / category / sample_id).glob('*.png'))[0]
    dm_path = list((data_root / 'DepthMap' / category / sample_id).glob('*.tiff'))[0]

    cv_img = np.array(Image.open(cv_path).convert('RGB'))
    depth_raw = np.array(Image.open(dm_path), dtype=np.float32)

    # Resize central view to depth resolution
    cv_small = np.array(Image.fromarray(cv_img).resize(
        (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

    # Find lesion and crop
    peak_y, peak_x = find_lesion(cv_small)
    crop = 250
    h, w = depth_raw.shape
    y1, y2 = max(0, peak_y - crop), min(h, peak_y + crop)
    x1, x2 = max(0, peak_x - crop), min(w, peak_x + crop)

    depth_crop = depth_raw[y1:y2, x1:x2]
    rgb_crop = cv_small[y1:y2, x1:x2]

    # Raw relief
    relief_raw = extract_relief(depth_crop)

    # Gaussian σ=15
    relief_gauss = extract_relief(gaussian_filter(depth_crop, sigma=15))

    # NL-means h=10 + Guided filter
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

# ============ Create figure: 5 rows × 6 columns ============
# For each case: [RGB | Raw depth | Raw 3D | Gauss depth | Gauss 3D | NLM+G depth | NLM+G 3D]
# Simplified: 5 rows × 4 columns: [RGB | Raw relief | Gaussian σ=15 | NL-means+Guided]

fig, axes = plt.subplots(5, 4, figsize=(24, 30))

for row, (category, sample_id) in enumerate(cases):
    print(f"Processing {category}/{sample_id}...")
    rgb_crop, relief_raw, relief_gauss, relief_nlm = process_case(category, sample_id)

    # Consistent color scale across methods for this case
    all_relief = np.concatenate([relief_raw.ravel(), relief_gauss.ravel(), relief_nlm.ravel()])
    vmin = np.percentile(all_relief, 1)
    vmax = np.percentile(all_relief, 99)

    # Column 0: RGB
    axes[row, 0].imshow(rgb_crop)
    axes[row, 0].set_title(f'{category}\n(ID: {sample_id})', fontsize=11, fontweight='bold')
    axes[row, 0].axis('off')

    # Column 1: Raw
    im = axes[row, 1].imshow(relief_raw, cmap='hot', vmin=vmin, vmax=vmax)
    bump_raw = relief_raw.max() - relief_raw.min()
    axes[row, 1].set_title(f'Raw\nrange={bump_raw:.3f}mm  std={relief_raw.std():.3f}', fontsize=10)
    axes[row, 1].axis('off')

    # Column 2: Gaussian σ=15
    im = axes[row, 2].imshow(relief_gauss, cmap='hot', vmin=vmin, vmax=vmax)
    bump_g = relief_gauss.max() - relief_gauss.min()
    axes[row, 2].set_title(f'Gaussian σ=15\nrange={bump_g:.3f}mm  std={relief_gauss.std():.3f}', fontsize=10)
    axes[row, 2].axis('off')

    # Column 3: NL-means + Guided
    im = axes[row, 3].imshow(relief_nlm, cmap='hot', vmin=vmin, vmax=vmax)
    bump_n = relief_nlm.max() - relief_nlm.min()
    axes[row, 3].set_title(f'NL-means + Guided\nrange={bump_n:.3f}mm  std={relief_nlm.std():.3f}', fontsize=10)
    axes[row, 3].axis('off')

    # Add colorbar to last column
    plt.colorbar(im, ax=axes[row, 3], shrink=0.7, label='mm')

plt.suptitle('SKINL2 Denoising Comparison Across 5 Clinical Categories\n'
             'Columns: Central View | Raw Relief | Gaussian σ=15 | NL-means + Guided Filter',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig33_skinl2_5cases_denoise.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig33_skinl2_5cases_denoise.png")

# ============ Also make 3D comparison for each case ============
fig = plt.figure(figsize=(24, 30))

for row, (category, sample_id) in enumerate(cases):
    print(f"3D for {category}/{sample_id}...")
    rgb_crop, relief_raw, relief_gauss, relief_nlm = process_case(category, sample_id)

    for col, (name, data) in enumerate([
        ('Raw', relief_raw),
        ('Gaussian σ=15', relief_gauss),
        ('NL-means + Guided', relief_nlm),
    ]):
        ax = fig.add_subplot(5, 3, row * 3 + col + 1, projection='3d')
        step = 4
        Z = data[::step, ::step]
        rgb_sub = rgb_crop[::step, ::step].astype(np.float64) / 255.0
        ys = np.arange(Z.shape[0]) * step
        xs = np.arange(Z.shape[1]) * step
        X, Y = np.meshgrid(xs, ys)
        ax.plot_surface(X, Y, Z,
                        facecolors=rgb_sub, rstride=1, cstride=1, shade=True,
                        lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
                        antialiased=True)
        ax.view_init(elev=55, azim=-55)
        ax.set_box_aspect([1, Z.shape[0] / Z.shape[1], 0.35])
        ax.set_xticks([]); ax.set_yticks([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        if row == 0:
            ax.set_title(name, fontsize=13, fontweight='bold')
        if col == 0:
            ax.set_ylabel(f'{category}\n({sample_id})', fontsize=10, labelpad=20)

plt.suptitle('SKINL2 3D Reconstructions — Denoising Comparison\n'
             '5 Clinical Categories × 3 Methods',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig34_skinl2_5cases_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig34_skinl2_5cases_3d.png")
