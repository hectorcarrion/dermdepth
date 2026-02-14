#!/usr/bin/env python3
"""
Investigate SKINL2 depth quality: filtering approaches + multi-view stereo.
Goal: recreate paper Figure 1 quality from Hemangioma case.
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from pathlib import Path

base = Path('/workspace/hector/dermdepth/data/SKINL2/SKINL2_v2/0001/Hemangioma/all_data/Hemangioma')
out_dir = Path('/workspace/hector/dermdepth/output/verification')

# ============ LOAD DATA ============
print("Loading data...")
cv_img = np.array(Image.open(base / 'Light Field/Central View/0001_TotalFocus.png').convert('RGB'))
depth_raw = np.array(Image.open(base / 'Light Field/Depth Map/0001_DepthMap.tiff'), dtype=np.float32)

# Downsample central view to match depth
cv_small = np.array(Image.fromarray(cv_img).resize(
    (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

# Find lesion center
r, g, b = cv_small[:,:,0].astype(float), cv_small[:,:,1].astype(float), cv_small[:,:,2].astype(float)
darkness = 255 - (r + g + b) / 3.0
redness = r / (g + 1)
lesion_score = gaussian_filter(darkness * redness, sigma=30)
peak_y, peak_x = np.unravel_index(np.argmax(lesion_score), lesion_score.shape)
print(f"Lesion at: ({peak_x}, {peak_y})")

# Crop around lesion
crop = 250
h, w = depth_raw.shape
y1, y2 = max(0, peak_y-crop), min(h, peak_y+crop)
x1, x2 = max(0, peak_x-crop), min(w, peak_x+crop)

depth_crop = depth_raw[y1:y2, x1:x2]
rgb_crop = cv_small[y1:y2, x1:x2]

# Elevation (negate, remove plane)
elev = -depth_crop
yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
relief_raw = elev - plane

print(f"Raw relief: range=[{relief_raw.min():.4f}, {relief_raw.max():.4f}]mm, std={relief_raw.std():.4f}")

# ============ APPROACH 1: FILTERING ============
print("\n--- Approach 1: Filtering ---")

# 1a. Gaussian smoothing
relief_gauss = gaussian_filter(relief_raw, sigma=5)

# 1b. Median filter (removes salt-and-pepper noise)
relief_median = median_filter(relief_raw, size=7)

# 1c. Bilateral filter (edge-preserving)
relief_f32 = relief_raw.astype(np.float32)
# Normalize to 0-255 range for bilateral filter
rmin, rmax = relief_f32.min(), relief_f32.max()
relief_norm = ((relief_f32 - rmin) / (rmax - rmin) * 255).astype(np.uint8)
relief_bilateral_u8 = cv2.bilateralFilter(relief_norm, d=15, sigmaColor=30, sigmaSpace=15)
relief_bilateral = relief_bilateral_u8.astype(np.float32) / 255 * (rmax - rmin) + rmin

# 1d. Guided filter (using RGB as guide)
gray_guide = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
# Implement simple guided filter
def guided_filter(guide, src, radius=15, eps=0.01):
    """Edge-preserving filter guided by the RGB image."""
    mean_g = cv2.boxFilter(guide, -1, (radius, radius))
    mean_s = cv2.boxFilter(src, -1, (radius, radius))
    mean_gs = cv2.boxFilter(guide * src, -1, (radius, radius))
    mean_gg = cv2.boxFilter(guide * guide, -1, (radius, radius))

    cov_gs = mean_gs - mean_g * mean_s
    var_g = mean_gg - mean_g * mean_g

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    return mean_a * guide + mean_b

relief_guided = guided_filter(gray_guide, relief_raw.astype(np.float32), radius=20, eps=0.001)

# 1e. Aggressive: bilateral + guided combo
relief_combo = guided_filter(gray_guide, relief_bilateral.astype(np.float32), radius=15, eps=0.0005)

print("Filtering done")

# ============ APPROACH 2: MULTI-VIEW STEREO ============
print("\n--- Approach 2: Multi-view stereo from light field views ---")

views_dir = base / 'Light Field/Views'

# Load center + neighboring views
def load_view(r, c):
    path = views_dir / f'0001_View_{r:02d}_{c:02d}.png'
    return np.array(Image.open(path).convert('L'))

# Center view
center = load_view(5, 5)
print(f"Center view gray: {center.shape}")

# Compute disparity from horizontal view pairs using block matching
# Use views along the horizontal axis: (5,1) to (5,9)
left_view = load_view(5, 3)   # slightly left
right_view = load_view(5, 7)  # slightly right

# OpenCV StereoSGBM for semi-global matching
min_disp = -16
num_disp = 32  # must be divisible by 16
block_size = 5

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * block_size * block_size,
    P2=32 * block_size * block_size,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Downscale for speed (full res is 3858x2682)
scale = 0.5
left_small = cv2.resize(left_view, None, fx=scale, fy=scale)
right_small = cv2.resize(right_view, None, fx=scale, fy=scale)

disparity = stereo.compute(left_small, right_small).astype(np.float32) / 16.0
print(f"Disparity: shape={disparity.shape}, range=[{disparity.min():.1f}, {disparity.max():.1f}]")

# Crop disparity to lesion region (scale coordinates)
dy1, dy2 = int(y1*scale*2), int(y2*scale*2)  # *2 because depth is half-res of views
dx1, dx2 = int(x1*scale*2), int(x2*scale*2)
disp_crop = disparity[dy1:dy2, dx1:dx2]

# Also try: accumulate disparity from multiple view pairs for more robust estimate
print("\nAccumulating disparities from multiple baselines...")
disparities = []
pairs = [(5,3,5,7), (5,2,5,8), (5,4,5,6),  # horizontal
         (3,5,7,5), (2,5,8,5), (4,5,6,5)]  # vertical

for r1,c1,r2,c2 in pairs:
    v1 = cv2.resize(load_view(r1,c1), None, fx=scale, fy=scale)
    v2 = cv2.resize(load_view(r2,c2), None, fx=scale, fy=scale)
    d = stereo.compute(v1, v2).astype(np.float32) / 16.0
    valid = d > min_disp
    disparities.append((d, valid))
    print(f"  Pair ({r1},{c1})-({r2},{c2}): valid={valid.sum()}/{valid.size} ({valid.sum()/valid.size*100:.0f}%)")

# Average valid disparities
disp_sum = np.zeros_like(disparities[0][0])
disp_count = np.zeros_like(disparities[0][0])
for d, valid in disparities:
    disp_sum[valid] += d[valid]
    disp_count[valid] += 1

disp_avg = np.where(disp_count > 0, disp_sum / disp_count, 0)
disp_avg_crop = disp_avg[dy1:dy2, dx1:dx2]

# Remove plane from disparity too
if disp_avg_crop.size > 0 and np.any(disp_avg_crop != 0):
    valid_disp = disp_avg_crop != 0
    yyd, xxd = np.mgrid[0:disp_avg_crop.shape[0], 0:disp_avg_crop.shape[1]]
    if valid_disp.sum() > 100:
        Ad = np.column_stack([xxd[valid_disp], yyd[valid_disp], np.ones(valid_disp.sum())])
        cd, _, _, _ = np.linalg.lstsq(Ad, disp_avg_crop[valid_disp], rcond=None)
        plane_d = cd[0]*xxd + cd[1]*yyd + cd[2]
        disp_relief = disp_avg_crop - plane_d
        disp_relief[~valid_disp] = np.nan
    else:
        disp_relief = disp_avg_crop
else:
    disp_relief = disp_avg_crop

# ============ APPROACH 3: EPI SLOPE ANALYSIS ============
print("\n--- Approach 3: EPI (Epipolar Plane Image) analysis ---")

# Extract horizontal EPI at the lesion's y-coordinate
# An EPI is formed by stacking one row from each horizontal view
epi_y = int(peak_y * 2)  # view coordinates (2x depth resolution)
epi_rows = []
for c in range(1, 10):
    v = load_view(5, c)
    if epi_y < v.shape[0]:
        epi_rows.append(v[epi_y, :])
epi_h = np.array(epi_rows)  # (9, width)
print(f"Horizontal EPI at y={epi_y}: {epi_h.shape}")

# Vertical EPI
epi_cols = []
epi_x = int(peak_x * 2)
for r in range(1, 10):
    v = load_view(r, 5)
    if epi_x < v.shape[1]:
        epi_cols.append(v[:, epi_x])
epi_v = np.array(epi_cols)  # (9, height)
print(f"Vertical EPI at x={epi_x}: {epi_v.shape}")

# ============ VISUALIZE ALL APPROACHES ============
print("\nCreating visualization...")

fig, axes = plt.subplots(3, 4, figsize=(24, 18))

# Row 1: Filtering approaches
vmin = np.percentile(relief_raw, 2)
vmax = np.percentile(relief_raw, 98)

axes[0,0].imshow(relief_raw, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[0,0].set_title(f'Raw Relief\nstd={relief_raw.std():.4f}mm')
axes[0,0].axis('off')

axes[0,1].imshow(relief_bilateral, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[0,1].set_title(f'Bilateral Filter\nstd={relief_bilateral.std():.4f}mm')
axes[0,1].axis('off')

axes[0,2].imshow(relief_guided, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[0,2].set_title(f'Guided Filter (RGB guide)\nstd={relief_guided.std():.4f}mm')
axes[0,2].axis('off')

axes[0,3].imshow(relief_combo, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[0,3].set_title(f'Bilateral + Guided\nstd={relief_combo.std():.4f}mm')
axes[0,3].axis('off')

# Row 2: Multi-view stereo + comparison
axes[1,0].imshow(rgb_crop)
axes[1,0].set_title('Central View (cropped)')
axes[1,0].axis('off')

if disp_crop.size > 0:
    dmin, dmax = np.percentile(disp_crop[disp_crop > min_disp], [2, 98]) if (disp_crop > min_disp).sum() > 0 else (0, 1)
    im = axes[1,1].imshow(disp_crop, cmap='turbo', vmin=dmin, vmax=dmax)
    axes[1,1].set_title('Stereo Disparity\n(single pair)')
    plt.colorbar(im, ax=axes[1,1], shrink=0.7)
else:
    axes[1,1].text(0.5, 0.5, 'No disparity', ha='center', va='center')
axes[1,1].axis('off')

if disp_relief is not None and disp_relief.size > 0:
    valid_dr = disp_relief[np.isfinite(disp_relief)]
    if len(valid_dr) > 0:
        im = axes[1,2].imshow(disp_relief, cmap='RdBu_r')
        axes[1,2].set_title('Multi-pair Disparity Relief\n(plane removed)')
        plt.colorbar(im, ax=axes[1,2], shrink=0.7)
    else:
        axes[1,2].text(0.5, 0.5, 'No valid disparity', ha='center', va='center')
else:
    axes[1,2].text(0.5, 0.5, 'No disparity', ha='center', va='center')
axes[1,2].axis('off')

# Best filtered result as paper Fig 1 style
best = relief_combo
axes[1,3].imshow(best, cmap='hot_r')
axes[1,3].set_title(f'Best Filtered (paper style)\nRelief: {best.max()-best.min():.3f}mm')
axes[1,3].axis('off')

# Row 3: EPIs + 3D reconstruction with best depth
axes[2,0].imshow(epi_h, cmap='gray', aspect='auto')
axes[2,0].set_title(f'Horizontal EPI at lesion y\n(9 views x {epi_h.shape[1]}px)')
axes[2,0].set_ylabel('View index')
axes[2,0].set_xlabel('x pixel')

axes[2,1].imshow(epi_v, cmap='gray', aspect='auto')
axes[2,1].set_title(f'Vertical EPI at lesion x\n(9 views x {epi_v.shape[1]}px)')
axes[2,1].set_ylabel('View index')

# 3D surface with best filtered depth
ax3d = fig.add_subplot(3, 4, 11, projection='3d')
step = 3
Z_best = best[::step, ::step]
rgb_sub = rgb_crop[::step, ::step].astype(float) / 255
yy3, xx3 = np.mgrid[0:Z_best.shape[0], 0:Z_best.shape[1]]
ax3d.plot_surface(xx3*step, yy3*step, Z_best,
                  facecolors=rgb_sub, rstride=1, cstride=1,
                  shade=True,
                  lightsource=matplotlib.colors.LightSource(315, 45))
ax3d.view_init(elev=50, azim=-50)
ax3d.set_box_aspect([1, Z_best.shape[0]/Z_best.shape[1], 0.5])
ax3d.set_title('3D Reconstruction\n(bilateral+guided filtered)')
ax3d.set_xticks([]); ax3d.set_yticks([])

# Depth histogram
axes[2,3].hist(relief_raw.ravel(), bins=100, alpha=0.5, label='Raw', density=True)
axes[2,3].hist(best.ravel(), bins=100, alpha=0.5, label='Filtered', density=True)
axes[2,3].set_xlabel('Relief (mm)')
axes[2,3].set_title('Relief Distribution')
axes[2,3].legend()

plt.suptitle('SKINL2 Hemangioma 0001 — Depth Quality Investigation\n'
             'Filtering + Multi-view Stereo + EPI Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig22_skinl2_depth_investigation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig22_skinl2_depth_investigation.png")
