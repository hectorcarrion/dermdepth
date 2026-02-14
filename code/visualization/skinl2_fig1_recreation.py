#!/usr/bin/env python3
"""Recreate SKINL2 paper Figure 1: 3D reconstruction + depth map for Hemangioma."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
from scipy.ndimage import gaussian_filter

base = '/workspace/hector/dermdepth/data/SKINL2/SKINL2_v2/0001/Hemangioma/all_data/Hemangioma'
out_dir = '/workspace/hector/dermdepth/output/verification'

# Load data
cv_img = np.array(Image.open(f'{base}/Light Field/Central View/0001_TotalFocus.png').convert('RGB'))
depth_raw = np.array(Image.open(f'{base}/Light Field/Depth Map/0001_DepthMap.tiff'), dtype=np.float32)

print(f'Central view: {cv_img.shape}')
print(f'Depth TIFF: {depth_raw.shape}, range=[{depth_raw.min():.3f}, {depth_raw.max():.3f}]')

# Downsample central view to match depth (depth is half-resolution)
cv_small = np.array(Image.fromarray(cv_img).resize(
    (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

# Find the lesion by color: it's reddish/dark compared to skin
# Convert to normalized red channel excess
r, g, b = cv_small[:,:,0].astype(float), cv_small[:,:,1].astype(float), cv_small[:,:,2].astype(float)
brightness = (r + g + b) / 3.0
# Lesion is darker and redder
darkness = 255 - brightness
redness = r / (g + 1)  # red/green ratio
lesion_score = darkness * redness
lesion_smooth = gaussian_filter(lesion_score, sigma=30)
peak_y, peak_x = np.unravel_index(np.argmax(lesion_smooth), lesion_smooth.shape)
print(f'Lesion detected at: ({peak_x}, {peak_y})')

# Depth values are negative. Negate for elevation (closer = higher).
depth = -depth_raw
elevation = depth - depth.min()
print(f'Elevation span: {elevation.max():.3f}mm')

h, w = depth_raw.shape

# Crop around lesion — generous crop to show surrounding skin
crop_size = 250
y1 = max(0, peak_y - crop_size)
y2 = min(h, peak_y + crop_size)
x1 = max(0, peak_x - crop_size)
x2 = min(w, peak_x + crop_size)

elev_crop = elevation[y1:y2, x1:x2]
rgb_crop = cv_small[y1:y2, x1:x2]

# Remove global plane to isolate lesion relief
yy, xx = np.mgrid[0:elev_crop.shape[0], 0:elev_crop.shape[1]]
A_mat = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
plane_coeffs, _, _, _ = np.linalg.lstsq(A_mat, elev_crop.ravel(), rcond=None)
plane = (plane_coeffs[0] * xx + plane_coeffs[1] * yy + plane_coeffs[2])
relief = elev_crop - plane
relief_smooth = gaussian_filter(relief, sigma=2)

print(f'Relief span: {relief_smooth.max() - relief_smooth.min():.3f}mm')
print(f'Lesion bump height: {relief_smooth.max():.3f}mm')

# ============= FIGURE 1 RECREATION =============
fig = plt.figure(figsize=(18, 9))

# Left: 3D surface with texture
ax1 = fig.add_subplot(121, projection='3d')
step = 3
elev_sub = relief_smooth[::step, ::step]  # Use relief (plane-removed) for Z
rgb_sub = rgb_crop[::step, ::step]
ys_arr = np.arange(elev_sub.shape[0]) * step
xs_arr = np.arange(elev_sub.shape[1]) * step
X, Y = np.meshgrid(xs_arr, ys_arr)
Z = elev_sub

rgb_norm = rgb_sub.astype(np.float64) / 255.0

ax1.plot_surface(X, Y, Z,
                 facecolors=rgb_norm,
                 rstride=1, cstride=1,
                 shade=True,
                 lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
                 antialiased=True)
ax1.view_init(elev=50, azim=-50)
ax1.set_box_aspect([1, elev_sub.shape[0]/elev_sub.shape[1], 0.5])
ax1.set_zlabel('Relief (mm)')
ax1.set_title('3D Skin Lesion Reconstruction', fontsize=13, fontweight='bold')
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.set_xticks([])
ax1.set_yticks([])

# Right: Colored depth map (relief)
ax2 = fig.add_subplot(122)
im = ax2.imshow(relief_smooth, cmap='hot_r', origin='upper',
                vmin=relief_smooth.min(), vmax=relief_smooth.max())
ax2.set_title('Corresponding Depth Map', fontsize=13, fontweight='bold')
ax2.axis('off')
plt.colorbar(im, ax=ax2, label='Relief (mm)', shrink=0.8)

plt.suptitle('SKINL2 Hemangioma 0001 — Recreating Paper Fig. 1\n'
             f'Light field depth, {elev_crop.max()-elev_crop.min():.1f}mm total span, '
             f'lesion relief ~{relief_smooth.max():.2f}mm',
             fontsize=14)
plt.tight_layout()
plt.savefig(f'{out_dir}/fig20_skinl2_fig1_recreation.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig20')

# ============= DETAILED PIPELINE =============
fig2, axes = plt.subplots(2, 3, figsize=(20, 13))

# Row 1: Full image context
axes[0, 0].imshow(cv_small)
axes[0, 0].set_title(f'Central View ({cv_small.shape[1]}x{cv_small.shape[0]})')
axes[0, 0].axis('off')
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
axes[0, 0].add_patch(rect)
axes[0, 0].plot(peak_x, peak_y, 'r+', markersize=15, markeredgewidth=2)

im1 = axes[0, 1].imshow(depth_raw, cmap='turbo')
axes[0, 1].set_title(f'Raw Depth TIFF\n[{depth_raw.min():.1f}, {depth_raw.max():.1f}]mm')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], shrink=0.7)
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
axes[0, 1].add_patch(rect)

# Authors' colored depth map
dc = np.array(Image.open(f'{base}/Light Field/Depth Map/0001_DepthMapColored.png').convert('RGB'))
axes[0, 2].imshow(dc)
axes[0, 2].set_title("Authors' Colored Depth Map")
axes[0, 2].axis('off')

# Row 2: Cropped and processed
axes[1, 0].imshow(rgb_crop)
axes[1, 0].set_title('Cropped Central View')
axes[1, 0].axis('off')

im2 = axes[1, 1].imshow(elev_crop, cmap='turbo')
axes[1, 1].set_title(f'Cropped Elevation\nSpan: {elev_crop.max()-elev_crop.min():.2f}mm')
axes[1, 1].axis('off')
plt.colorbar(im2, ax=axes[1, 1], shrink=0.7, label='mm')

im3 = axes[1, 2].imshow(relief_smooth, cmap='hot_r')
axes[1, 2].set_title(f'Relief (plane removed)\nLesion: +{relief_smooth.max():.2f}mm')
axes[1, 2].axis('off')
plt.colorbar(im3, ax=axes[1, 2], shrink=0.7, label='mm')

plt.suptitle('SKINL2 Hemangioma 0001 — Depth Processing Pipeline', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{out_dir}/fig21_skinl2_depth_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig21')
