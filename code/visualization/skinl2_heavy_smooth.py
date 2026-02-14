#!/usr/bin/env python3
"""
SKINL2 heavy smoothing: the paper Figure 1 depth map is very smooth.
The lesion is ~200px across at depth resolution — we can use sigma=10-20.
Also try much heavier TV, and raw depth (not relief) smoothing.
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

base = Path('/workspace/hector/dermdepth/data/SKINL2/SKINL2_v2/0001/Hemangioma/all_data/Hemangioma')
out_dir = Path('/workspace/hector/dermdepth/output/verification')

# Load
cv_img = np.array(Image.open(base / 'Light Field/Central View/0001_TotalFocus.png').convert('RGB'))
depth_raw = np.array(Image.open(base / 'Light Field/Depth Map/0001_DepthMap.tiff'), dtype=np.float32)
cv_small = np.array(Image.fromarray(cv_img).resize(
    (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

# Lesion detection
r, g, b = cv_small[:,:,0].astype(float), cv_small[:,:,1].astype(float), cv_small[:,:,2].astype(float)
darkness = 255 - (r + g + b) / 3.0
redness = r / (g + 1)
lesion_score = gaussian_filter(darkness * redness, sigma=30)
peak_y, peak_x = np.unravel_index(np.argmax(lesion_score), lesion_score.shape)

# Crop around lesion
crop = 300  # slightly larger crop
h, w = depth_raw.shape
y1, y2 = max(0, peak_y-crop), min(h, peak_y+crop)
x1, x2 = max(0, peak_x-crop), min(w, peak_x+crop)

depth_crop = depth_raw[y1:y2, x1:x2]
rgb_crop = cv_small[y1:y2, x1:x2]

# KEY INSIGHT: smooth the RAW depth first, THEN remove plane
# This preserves the large-scale lesion shape better than smoothing relief
print("=== Strategy 1: Smooth raw depth, then remove plane ===")

results = {}
for sigma in [5, 10, 15, 20, 30]:
    depth_smooth = gaussian_filter(depth_crop, sigma=sigma)
    elev = -depth_smooth
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    relief = elev - plane
    results[f'gauss_σ={sigma}'] = relief
    print(f"  σ={sigma}: relief range=[{relief.min():.4f}, {relief.max():.4f}], "
          f"bump={relief.max():.4f}mm, std={relief.std():.4f}")

print("\n=== Strategy 2: Smooth raw depth with bilateral, then remove plane ===")
for d, sc in [(31, 3.0), (51, 5.0), (71, 7.0)]:
    # Bilateral on raw depth (not normalized — use actual mm values)
    depth_bil = cv2.bilateralFilter(depth_crop, d=d, sigmaColor=sc, sigmaSpace=d//2)
    elev = -depth_bil
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    relief = elev - plane
    results[f'bilateral_d={d}_sc={sc}'] = relief
    print(f"  d={d}, σc={sc}: relief range=[{relief.min():.4f}, {relief.max():.4f}], "
          f"bump={relief.max():.4f}mm, std={relief.std():.4f}")

print("\n=== Strategy 3: Guided filter on raw depth (RGB guide), then plane ===")
gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255

def guided_filter(guide, src, radius=15, eps=0.01):
    ksize = (radius, radius)
    mean_g = cv2.boxFilter(guide, -1, ksize)
    mean_s = cv2.boxFilter(src, -1, ksize)
    mean_gs = cv2.boxFilter(guide * src, -1, ksize)
    mean_gg = cv2.boxFilter(guide * guide, -1, ksize)
    cov_gs = mean_gs - mean_g * mean_s
    var_g = mean_gg - mean_g * mean_g
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)
    return mean_a * guide + mean_b

for radius, eps in [(30, 0.01), (50, 0.005), (80, 0.001)]:
    depth_guided = guided_filter(gray, depth_crop, radius=radius, eps=eps)
    elev = -depth_guided
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    relief = elev - plane
    results[f'guided_r={radius}_ε={eps}'] = relief
    print(f"  r={radius}, ε={eps}: relief range=[{relief.min():.4f}, {relief.max():.4f}], "
          f"bump={relief.max():.4f}mm")

print("\n=== Strategy 4: Gaussian + Guided combo ===")
for g_sigma, g_radius, g_eps in [(10, 40, 0.005), (15, 50, 0.003), (10, 30, 0.01)]:
    depth_g = gaussian_filter(depth_crop, sigma=g_sigma)
    depth_gf = guided_filter(gray, depth_g.astype(np.float32), radius=g_radius, eps=g_eps)
    elev = -depth_gf
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    relief = elev - plane
    key = f'gauss{g_sigma}+guided_r{g_radius}'
    results[key] = relief
    print(f"  σ={g_sigma}, r={g_radius}, ε={g_eps}: bump={relief.max():.4f}mm")

# ============ PICK TOP 6 + RAW for final comparison ============
print("\n=== Creating final comparison ===")

# Raw for reference
elev_raw = -depth_crop
yy, xx = np.mgrid[0:elev_raw.shape[0], 0:elev_raw.shape[1]]
A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
coeffs, _, _, _ = np.linalg.lstsq(A, elev_raw.ravel(), rcond=None)
plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
relief_raw = elev_raw - plane

# Top candidates by visual quality (smooth + preserves lesion)
top = [
    ('Raw (no filtering)', relief_raw),
    ('Gaussian σ=10', results['gauss_σ=10']),
    ('Gaussian σ=15', results['gauss_σ=15']),
    ('Gaussian σ=20', results['gauss_σ=20']),
    ('Bilateral d=51', results['bilateral_d=51_sc=5.0']),
    ('Guided r=50', results['guided_r=50_ε=0.005']),
    ('Gauss10+Guided40', results['gauss10+guided_r40']),
    ('Gauss15+Guided50', results['gauss15+guided_r50']),
]

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
for idx, (name, data) in enumerate(top):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    im = ax.imshow(data, cmap='hot_r')
    ax.set_title(f'{name}\nbump={data.max():.3f}mm, range={data.max()-data.min():.3f}mm',
                 fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.suptitle('SKINL2 Hemangioma — Smoothing on RAW Depth Before Plane Removal\n'
             '(Key insight: smooth depth first, then extract relief)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig26_skinl2_smooth_raw_first.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig26")

# ============ FIGURE 1 RECREATION WITH BEST ============
print("\n=== Figure 1 recreation ===")

# Best: Gaussian σ=15 gives smooth + visible bump
best = results['gauss_σ=15']
best_name = 'Gaussian σ=15 on raw depth'

# Also try the authors' colored depth map approach
dc = np.array(Image.open(base / 'Light Field/Depth Map/0001_DepthMapColored.png').convert('RGB'))

fig = plt.figure(figsize=(24, 10))

# Panel 1: RGB
ax1 = fig.add_subplot(141)
ax1.imshow(rgb_crop)
ax1.set_title('Central View', fontsize=13, fontweight='bold')
ax1.axis('off')

# Panel 2: Authors' colored depth
ax2 = fig.add_subplot(142)
# Crop authors' depth to same region
dc_small = np.array(Image.fromarray(dc).resize(
    (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))
dc_crop = dc_small[y1:y2, x1:x2]
ax2.imshow(dc_crop)
ax2.set_title("Authors' Colored Depth\n(Raytrix Software)", fontsize=13, fontweight='bold')
ax2.axis('off')

# Panel 3: Our best depth map
ax3 = fig.add_subplot(143)
im = ax3.imshow(best, cmap='hot_r')
ax3.set_title(f'Our Depth Map\n({best_name})', fontsize=13, fontweight='bold')
ax3.axis('off')
plt.colorbar(im, ax=ax3, label='Relief (mm)', shrink=0.7)

# Panel 4: 3D surface
ax4 = fig.add_subplot(144, projection='3d')
step = 3
Z = best[::step, ::step]
rgb_sub = rgb_crop[::step, ::step].astype(np.float64) / 255.0
ys = np.arange(Z.shape[0]) * step
xs = np.arange(Z.shape[1]) * step
X, Y = np.meshgrid(xs, ys)

ax4.plot_surface(X, Y, Z,
                 facecolors=rgb_sub,
                 rstride=1, cstride=1,
                 shade=True,
                 lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
                 antialiased=True)
ax4.view_init(elev=55, azim=-45)
ax4.set_box_aspect([1, Z.shape[0]/Z.shape[1], 0.4])
ax4.set_zlabel('Relief (mm)')
ax4.set_title('3D Reconstruction', fontsize=13, fontweight='bold')
ax4.set_xticks([]); ax4.set_yticks([])

plt.suptitle(f'SKINL2 Hemangioma 0001 — Paper Figure 1 Recreation\n'
             f'Lesion elevation: {best.max():.3f}mm | Total relief: {best.max()-best.min():.3f}mm',
             fontsize=14)
plt.tight_layout()
plt.savefig(out_dir / 'fig27_skinl2_fig1_final.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig27")

# ============ SIDE-BY-SIDE WITH PAPER FIG 1 STYLE ============
# Recreate the exact same layout as the paper: 3D left, depth right
fig = plt.figure(figsize=(18, 9))

ax1 = fig.add_subplot(121, projection='3d')
# Exaggerate Z for visual impact (paper likely does this)
Z_exag = Z * 3  # 3x vertical exaggeration
ax1.plot_surface(X, Y, Z_exag,
                 facecolors=rgb_sub,
                 rstride=1, cstride=1,
                 shade=True,
                 lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
                 antialiased=True)
ax1.view_init(elev=50, azim=-50)
ax1.set_box_aspect([1, Z.shape[0]/Z.shape[1], 0.6])
ax1.set_title('3D Skin Lesion Reconstruction\n(3x vertical exaggeration)', fontsize=13, fontweight='bold')
ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

ax2 = fig.add_subplot(122)
im = ax2.imshow(best, cmap='hot_r', vmin=best.min(), vmax=best.max())
ax2.set_title('Corresponding Depth Map', fontsize=13, fontweight='bold')
ax2.axis('off')
cbar = plt.colorbar(im, ax=ax2, label='Relief (mm)', shrink=0.8)

plt.suptitle(f'Recreating SKINL2 Paper Figure 1\n'
             f'Hemangioma 0001 | Gaussian σ=15 denoised',
             fontsize=14)
plt.tight_layout()
plt.savefig(out_dir / 'fig28_skinl2_paper_style.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig28")
