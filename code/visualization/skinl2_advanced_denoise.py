#!/usr/bin/env python3
"""
Advanced denoising approaches for SKINL2 depth:
1. NL-means on depth map (self-similarity based)
2. Cost-volume from all 81 light field views (proper light field depth)
3. Bilateral + NL-means combo
4. Compare all against best Gaussian result
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

# Load data
print("Loading data...")
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
print(f"Lesion at: ({peak_x}, {peak_y})")

# Crop
crop = 300
h, w = depth_raw.shape
y1, y2 = max(0, peak_y-crop), min(h, peak_y+crop)
x1, x2 = max(0, peak_x-crop), min(w, peak_x+crop)
depth_crop = depth_raw[y1:y2, x1:x2]
rgb_crop = cv_small[y1:y2, x1:x2]

def extract_relief(depth):
    """Convert depth to relief by removing planar trend."""
    elev = depth  # closer = less negative = higher bump
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    return elev - plane

# ============ BASELINE: Gaussian σ=15 ============
print("\n=== Baseline: Gaussian σ=15 ===")
relief_gauss15 = extract_relief(gaussian_filter(depth_crop, sigma=15))
print(f"  Bump: {relief_gauss15.max():.4f}mm, std: {relief_gauss15.std():.4f}mm")

# ============ APPROACH 1: NL-means on depth ============
print("\n=== Approach 1: NL-means denoising ===")
# NL-means works on uint8/uint16, so normalize depth to uint16 range
dmin, dmax = depth_crop.min(), depth_crop.max()
depth_u16 = ((depth_crop - dmin) / (dmax - dmin) * 65535).astype(np.uint16)

# Try different NL-means parameters
nlm_results = {}
for h_val in [5, 10, 15, 20]:
    # OpenCV NL-means expects uint8
    depth_u8 = (depth_u16 / 256).astype(np.uint8)
    denoised_u8 = cv2.fastNlMeansDenoising(depth_u8, h=h_val, templateWindowSize=7, searchWindowSize=21)
    # Convert back to original scale
    denoised = denoised_u8.astype(np.float32) / 255 * (dmax - dmin) + dmin
    relief = extract_relief(denoised)
    nlm_results[f'h={h_val}'] = relief
    print(f"  h={h_val}: bump={relief.max():.4f}mm, std={relief.std():.4f}mm")

# ============ APPROACH 2: Light field cost volume ============
print("\n=== Approach 2: Light field depth from cost volume (all 81 views) ===")
views_dir = base / 'Light Field/Views'

# Strategy: shift-and-compare all views to center view at different disparities
# The center view is (5,5). Each view (r,c) has a shift of (c-5, r-5) * disparity.
# For the correct depth, all views align → minimum SSD cost.

# Work at depth-map resolution (half of view resolution)
scale = depth_crop.shape[0] / cv_img.shape[0]  # ~0.5
print(f"Scale factor: {scale:.3f}")

# Load center view at depth resolution
center_gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
ch, cw = center_gray.shape

# Load all views at depth resolution (just the cropped region)
print("Loading and cropping all 81 views...")
views = {}
for vr in range(1, 10):
    for vc in range(1, 10):
        path = views_dir / f'0001_View_{vr:02d}_{vc:02d}.png'
        if path.exists():
            img = np.array(Image.open(path).convert('L'))
            # Resize to depth resolution
            img_small = cv2.resize(img, (depth_raw.shape[1], depth_raw.shape[0]))
            # Crop
            views[(vr, vc)] = img_small[y1:y2, x1:x2].astype(np.float32)

print(f"Loaded {len(views)} views")

# Disparity search range: from raw depth variation, the disparity is tiny
# Depth varies by ~3mm over ~200mm -> ~1.5% -> ~20 pixels at view res -> ~10 at depth res
disp_range = np.arange(-8, 9, 0.5)  # sub-pixel disparities
print(f"Testing {len(disp_range)} disparity levels")

center_view = views.get((5, 5), center_gray)
cost_volume = np.zeros((ch, cw, len(disp_range)), dtype=np.float32)

for di, d in enumerate(disp_range):
    ssd_sum = np.zeros((ch, cw), dtype=np.float32)
    count = 0
    for (vr, vc), view in views.items():
        if (vr, vc) == (5, 5):
            continue
        # Expected shift for this view at disparity d
        dx = (vc - 5) * d
        dy = (vr - 5) * d

        # Shift the view
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        shifted = cv2.warpAffine(view, M, (cw, ch),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

        # SSD with center view (use small window via box filter)
        diff2 = (shifted - center_view) ** 2
        ssd = cv2.boxFilter(diff2, -1, (9, 9))
        ssd_sum += ssd
        count += 1

    cost_volume[:, :, di] = ssd_sum / count

# Find best disparity at each pixel
best_disp_idx = np.argmin(cost_volume, axis=2)
best_disp = disp_range[best_disp_idx]
min_cost = np.min(cost_volume, axis=2)

# Sub-pixel refinement via parabola fitting
print("Sub-pixel refinement...")
disp_refined = best_disp.copy()
for i in range(ch):
    for j in range(cw):
        idx = best_disp_idx[i, j]
        if 0 < idx < len(disp_range) - 1:
            c0 = cost_volume[i, j, idx - 1]
            c1 = cost_volume[i, j, idx]
            c2 = cost_volume[i, j, idx + 1]
            denom = 2 * (c0 - 2*c1 + c2)
            if abs(denom) > 1e-6:
                offset = (c0 - c2) / denom * (disp_range[1] - disp_range[0])
                disp_refined[i, j] = disp_range[idx] + offset

# Convert disparity to relief
relief_lf = extract_relief(gaussian_filter(disp_refined, sigma=5))
print(f"  LF depth: bump={relief_lf.max():.4f}mm, std={relief_lf.std():.4f}mm")

# Also smooth the LF result more heavily
relief_lf_smooth = extract_relief(gaussian_filter(disp_refined, sigma=15))
print(f"  LF depth (σ=15): bump={relief_lf_smooth.max():.4f}mm, std={relief_lf_smooth.std():.4f}mm")

# ============ APPROACH 3: Fuse Raytrix depth + LF cost volume ============
print("\n=== Approach 3: Fuse Raytrix depth with LF confidence ===")
# Use LF cost as confidence weight for smoothing the Raytrix depth
# Low cost = high confidence = preserve Raytrix value, high cost = smooth more

# Normalize confidence from cost
max_cost = np.percentile(min_cost, 95)
confidence = 1.0 - np.clip(min_cost / max_cost, 0, 1)
confidence = gaussian_filter(confidence, sigma=3)

# Weighted blend: at high confidence, use lightly smoothed Raytrix; at low, use heavily smoothed
depth_light = gaussian_filter(depth_crop, sigma=5)
depth_heavy = gaussian_filter(depth_crop, sigma=20)
depth_fused = confidence * depth_light + (1 - confidence) * depth_heavy
relief_fused = extract_relief(depth_fused)
print(f"  Fused: bump={relief_fused.max():.4f}mm, std={relief_fused.std():.4f}mm")

# ============ APPROACH 4: NL-means + Guided filter combo ============
print("\n=== Approach 4: NL-means h=10 + Guided filter ===")
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

# NL-means first
depth_u8 = ((depth_crop - dmin) / (dmax - dmin) * 255).astype(np.uint8)
nlm_u8 = cv2.fastNlMeansDenoising(depth_u8, h=10, templateWindowSize=7, searchWindowSize=21)
nlm_f32 = nlm_u8.astype(np.float32) / 255 * (dmax - dmin) + dmin

# Then guided filter
nlm_guided = guided_filter(gray, nlm_f32, radius=30, eps=0.005)
relief_nlm_guided = extract_relief(nlm_guided)
print(f"  NLM+Guided: bump={relief_nlm_guided.max():.4f}mm, std={relief_nlm_guided.std():.4f}mm")

# ============ VISUALIZATION ============
print("\nCreating comparison...")

approaches = [
    ('Raw (no filter)', extract_relief(depth_crop)),
    ('Gaussian σ=15\n(current best)', relief_gauss15),
    ('NL-means h=10', nlm_results['h=10']),
    ('NL-means h=15', nlm_results['h=15']),
    ('LF Cost Volume\n(81 views, σ=5)', relief_lf),
    ('LF Cost Volume\n(σ=15 smooth)', relief_lf_smooth),
    ('Raytrix + LF\nConfidence Fusion', relief_fused),
    ('NL-means + Guided\n(edge-preserving)', relief_nlm_guided),
]

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
for idx, (name, data) in enumerate(approaches):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    im = ax.imshow(data, cmap='hot', vmin=-0.2, vmax=0.4)
    bump = data.max()
    noise_std = data.std()
    # SNR: bump height / noise std
    snr = bump / noise_std if noise_std > 0 else 0
    ax.set_title(f'{name}\nbump={bump:.3f}mm  SNR={snr:.1f}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.suptitle('SKINL2 Hemangioma — Advanced Denoising Comparison\n'
             'Goal: maximize bump visibility (SNR) while preserving lesion shape',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig30_skinl2_advanced_denoise.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig30")

# ============ BEST RESULT: 3D + depth side-by-side ============
# Pick the best by SNR
best_name = None
best_snr = 0
best_data = None
for name, data in approaches[1:]:  # skip raw
    snr = data.max() / data.std()
    if snr > best_snr:
        best_snr = snr
        best_name = name
        best_data = data

print(f"\nBest approach: {best_name} (SNR={best_snr:.1f})")

# Also export best as GLB
import trimesh
step = 2
Z = best_data[::step, ::step]
rgb_sub = rgb_crop[::step, ::step]
rows, cols = Z.shape
xs = np.arange(cols) * step
ys = np.arange(rows) * step
X, Y = np.meshgrid(xs, ys)
pixel_to_mm = 0.02
X_mm = X * pixel_to_mm
Y_mm = Y * pixel_to_mm

vertices = np.column_stack([X_mm.ravel(), -Y_mm.ravel(), Z.ravel()])
faces = []
for i in range(rows - 1):
    for j in range(cols - 1):
        idx = i * cols + j
        faces.append([idx, idx + cols, idx + 1])
        faces.append([idx + 1, idx + cols, idx + cols + 1])
faces = np.array(faces)
colors = np.column_stack([rgb_sub.reshape(-1, 3), np.full(rows*cols, 255, dtype=np.uint8)])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
glb_path = out_dir / 'skinl2_hemangioma_0001_best.glb'
mesh.export(str(glb_path), file_type='glb')
print(f"Saved best GLB: {glb_path}")

# Final paper-style figure with best result
fig = plt.figure(figsize=(18, 9))

ax1 = fig.add_subplot(121, projection='3d')
step_plot = 3
Z_plot = best_data[::step_plot, ::step_plot]
rgb_plot = rgb_crop[::step_plot, ::step_plot].astype(np.float64) / 255.0
ys_plot = np.arange(Z_plot.shape[0]) * step_plot
xs_plot = np.arange(Z_plot.shape[1]) * step_plot
Xp, Yp = np.meshgrid(xs_plot, ys_plot)

ax1.plot_surface(Xp, Yp, Z_plot,
                 facecolors=rgb_plot, rstride=1, cstride=1, shade=True,
                 lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
                 antialiased=True)
ax1.view_init(elev=60, azim=-60)
ax1.set_box_aspect([1, Z_plot.shape[0]/Z_plot.shape[1], 0.3])
ax1.set_title('3D Reconstruction (best denoising)\nTrue scale', fontsize=13, fontweight='bold')
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_zlabel('Relief (mm)')
ax1.xaxis.pane.fill = False; ax1.yaxis.pane.fill = False; ax1.zaxis.pane.fill = False

ax2 = fig.add_subplot(122)
im = ax2.imshow(best_data, cmap='hot', vmin=-0.2, vmax=0.4)
ax2.set_title('Depth Map (best denoising)', fontsize=13, fontweight='bold')
ax2.axis('off')
plt.colorbar(im, ax=ax2, label='Relief (mm)', shrink=0.8)

plt.suptitle(f'SKINL2 Hemangioma — Best Result: {best_name}\n'
             f'Bump={best_data.max():.3f}mm | SNR={best_snr:.1f}',
             fontsize=14)
plt.tight_layout()
plt.savefig(out_dir / 'fig31_skinl2_best_result.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig31")
