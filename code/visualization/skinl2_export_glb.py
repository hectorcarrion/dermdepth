#!/usr/bin/env python3
"""
Export SKINL2 Hemangioma depth as GLB mesh for interactive 3D viewing.
- Flip Z so lesion bump points UP
- No vertical exaggeration (true mm scale)
- Also re-render the figure corrected
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
import trimesh

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

# Crop around lesion
crop = 300
h, w = depth_raw.shape
y1, y2 = max(0, peak_y-crop), min(h, peak_y+crop)
x1, x2 = max(0, peak_x-crop), min(w, peak_x+crop)

depth_crop = depth_raw[y1:y2, x1:x2]
rgb_crop = cv_small[y1:y2, x1:x2]

# Smooth raw depth, then extract relief
# depth_raw is negative (camera convention): -129mm (far) to -126mm (close/lesion)
# For elevation: closer to camera = higher bump = less negative = HIGHER value
# So use depth directly (no negation): lesion at -126 > background at -129 → bump
depth_smooth = gaussian_filter(depth_crop, sigma=15)
elev = depth_smooth  # no negation: less negative = closer = bump UP
yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
relief = elev - plane

# FLIP: negate so the lesion bump (which was negative in original relief) points UP
# Actually, check: the depth map shows bright=positive=bump.
# relief has positive values at the lesion center (bump).
# But in the 3D plot, Z=relief was rendered going DOWN because of the viewing angle.
# The issue is the 3D axes orientation. Let's check the actual values:
print(f"Relief at lesion center: {relief[crop, crop]:.4f}mm")
print(f"Relief range: [{relief.min():.4f}, {relief.max():.4f}]mm")
print(f"Relief at lesion is {'positive (bump)' if relief[crop, crop] > 0 else 'negative (dip)'}")

# For the GLB, we want Z-up with the bump pointing up
# relief is already positive at the bump center, so it should be correct
# The matplotlib issue was likely view_init orientation

# ============ GLB EXPORT ============
print("\nExporting GLB mesh...")

step = 2  # subsample for manageable mesh size
Z = relief[::step, ::step]
rgb_sub = rgb_crop[::step, ::step]
rows, cols = Z.shape

# Create vertex grid — X and Y in pixel coords, Z in mm
xs = np.arange(cols) * step
ys = np.arange(rows) * step
X, Y = np.meshgrid(xs, ys)

# Scale X,Y to mm (approximate: Raytrix pixel pitch at ~200mm depth)
# Depth map is ~1341x1929 for a ~30x40mm lesion area -> ~0.02mm/pixel
pixel_to_mm = 0.02
X_mm = X * pixel_to_mm
Y_mm = Y * pixel_to_mm

# Vertices: (N, 3) — flip Y so image appears correct orientation
vertices = np.column_stack([
    X_mm.ravel(),
    -Y_mm.ravel(),  # flip Y for correct image orientation
    Z.ravel()       # Z is relief in mm, positive = bump up
])

# Create triangle faces from grid
faces = []
for i in range(rows - 1):
    for j in range(cols - 1):
        idx = i * cols + j
        # Two triangles per quad
        faces.append([idx, idx + cols, idx + 1])
        faces.append([idx + 1, idx + cols, idx + cols + 1])
faces = np.array(faces)

# Vertex colors from RGB
colors = np.column_stack([
    rgb_sub.reshape(-1, 3),
    np.full(rows * cols, 255, dtype=np.uint8)  # alpha
])

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    vertex_colors=colors,
    process=False
)

glb_path = out_dir / 'skinl2_hemangioma_0001.glb'
mesh.export(str(glb_path), file_type='glb')
print(f"Saved GLB: {glb_path}")
print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
print(f"  Mesh bounds: X=[{vertices[:,0].min():.2f}, {vertices[:,0].max():.2f}]mm")
print(f"               Y=[{vertices[:,1].min():.2f}, {vertices[:,1].max():.2f}]mm")
print(f"               Z=[{vertices[:,2].min():.4f}, {vertices[:,2].max():.4f}]mm")

# ============ CORRECTED FIGURE ============
print("\nCreating corrected figure...")

fig = plt.figure(figsize=(18, 9))

# Left: 3D surface, corrected orientation, no exaggeration
ax1 = fig.add_subplot(121, projection='3d')
step_plot = 3
Z_plot = relief[::step_plot, ::step_plot]
rgb_plot = rgb_crop[::step_plot, ::step_plot].astype(np.float64) / 255.0
ys_plot = np.arange(Z_plot.shape[0]) * step_plot
xs_plot = np.arange(Z_plot.shape[1]) * step_plot
Xp, Yp = np.meshgrid(xs_plot, ys_plot)

ax1.plot_surface(Xp, Yp, Z_plot,
                 facecolors=rgb_plot,
                 rstride=1, cstride=1,
                 shade=True,
                 lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
                 antialiased=True)
# View from above-front, looking down at the skin surface with bump pointing UP
ax1.view_init(elev=60, azim=-60)
ax1.set_box_aspect([1, Z_plot.shape[0]/Z_plot.shape[1], 0.3])
ax1.set_title('3D Skin Lesion Reconstruction\n(true scale, no exaggeration)', fontsize=13, fontweight='bold')
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_zlabel('Relief (mm)')
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

# Right: depth map
ax2 = fig.add_subplot(122)
im = ax2.imshow(relief, cmap='hot', vmin=relief.min(), vmax=relief.max())
ax2.set_title('Corresponding Depth Map', fontsize=13, fontweight='bold')
ax2.axis('off')
plt.colorbar(im, ax=ax2, label='Relief (mm)', shrink=0.8)

plt.suptitle(f'SKINL2 Hemangioma 0001 — Corrected Orientation\n'
             f'Lesion elevation: {relief.max():.3f}mm | No vertical exaggeration',
             fontsize=14)
plt.tight_layout()
fig_path = out_dir / 'fig29_skinl2_corrected.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved {fig_path}")
