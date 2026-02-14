#!/usr/bin/env python3
"""
Final SKINL2 denoising comparison: top 3 approaches side-by-side in paper Figure 1 style.
Export best as GLB.
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

# Load
cv_img = np.array(Image.open(base / 'Light Field/Central View/0001_TotalFocus.png').convert('RGB'))
depth_raw = np.array(Image.open(base / 'Light Field/Depth Map/0001_DepthMap.tiff'), dtype=np.float32)
cv_small = np.array(Image.fromarray(cv_img).resize(
    (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

# Lesion detection
r, g, b = cv_small[:,:,0].astype(float), cv_small[:,:,1].astype(float), cv_small[:,:,2].astype(float)
lesion_score = gaussian_filter((255 - (r+g+b)/3) * r/(g+1), sigma=30)
peak_y, peak_x = np.unravel_index(np.argmax(lesion_score), lesion_score.shape)

crop = 300
h, w = depth_raw.shape
y1, y2 = max(0, peak_y-crop), min(h, peak_y+crop)
x1, x2 = max(0, peak_x-crop), min(w, peak_x+crop)
depth_crop = depth_raw[y1:y2, x1:x2]
rgb_crop = cv_small[y1:y2, x1:x2]

def extract_relief(depth):
    elev = depth
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
    return elev - (coeffs[0]*xx + coeffs[1]*yy + coeffs[2])

def guided_filter(guide, src, radius=15, eps=0.01):
    ksize = (radius, radius)
    mg = cv2.boxFilter(guide, -1, ksize)
    ms = cv2.boxFilter(src, -1, ksize)
    mgs = cv2.boxFilter(guide*src, -1, ksize)
    mgg = cv2.boxFilter(guide*guide, -1, ksize)
    a = (mgs - mg*ms) / (mgg - mg*mg + eps)
    b = ms - a*mg
    return cv2.boxFilter(a, -1, ksize)*guide + cv2.boxFilter(b, -1, ksize)

# Generate all three approaches
relief_raw = extract_relief(depth_crop)
relief_gauss = extract_relief(gaussian_filter(depth_crop, sigma=15))

# NL-means + Guided
dmin, dmax = depth_crop.min(), depth_crop.max()
depth_u8 = ((depth_crop - dmin) / (dmax - dmin) * 255).astype(np.uint8)
nlm_u8 = cv2.fastNlMeansDenoising(depth_u8, h=10, templateWindowSize=7, searchWindowSize=21)
nlm_f32 = nlm_u8.astype(np.float32) / 255 * (dmax - dmin) + dmin
gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
relief_nlm_guided = extract_relief(guided_filter(gray, nlm_f32, radius=30, eps=0.005))

# Confidence fusion (Raytrix + LF cost)
# Approximate: use bilateral on Raytrix as a simpler fusion proxy
depth_bil = cv2.bilateralFilter(depth_crop, d=31, sigmaColor=3.0, sigmaSpace=15)
depth_fused = guided_filter(gray, gaussian_filter(depth_bil, sigma=5).astype(np.float32), radius=20, eps=0.003)
relief_fused = extract_relief(depth_fused)

approaches = [
    ('Raw Raytrix Depth', relief_raw),
    ('Gaussian σ=15', relief_gauss),
    ('NL-means + Guided', relief_nlm_guided),
    ('Bilateral + Guided', relief_fused),
]

# ============ Figure: 4 columns, 2 rows (depth map + 3D) ============
fig = plt.figure(figsize=(28, 14))

vmin, vmax = -0.15, 0.40

for col, (name, data) in enumerate(approaches):
    # Top row: depth map
    ax_top = fig.add_subplot(2, 4, col + 1)
    im = ax_top.imshow(data, cmap='hot', vmin=vmin, vmax=vmax)
    snr = data.max() / data.std()
    ax_top.set_title(f'{name}\nbump={data.max():.3f}mm | std={data.std():.3f}mm | SNR={snr:.1f}',
                     fontsize=11)
    ax_top.axis('off')
    plt.colorbar(im, ax=ax_top, shrink=0.7, label='mm')

    # Bottom row: 3D surface
    ax_3d = fig.add_subplot(2, 4, col + 5, projection='3d')
    step = 3
    Z = data[::step, ::step]
    rgb_sub = rgb_crop[::step, ::step].astype(np.float64) / 255.0
    ys = np.arange(Z.shape[0]) * step
    xs = np.arange(Z.shape[1]) * step
    X, Y = np.meshgrid(xs, ys)
    ax_3d.plot_surface(X, Y, Z,
                       facecolors=rgb_sub, rstride=1, cstride=1, shade=True,
                       lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
                       antialiased=True)
    ax_3d.view_init(elev=55, azim=-55)
    ax_3d.set_box_aspect([1, Z.shape[0]/Z.shape[1], 0.35])
    ax_3d.set_xticks([]); ax_3d.set_yticks([])
    ax_3d.set_zlabel('mm')
    ax_3d.xaxis.pane.fill = False; ax_3d.yaxis.pane.fill = False; ax_3d.zaxis.pane.fill = False

plt.suptitle('SKINL2 Hemangioma 0001 — Denoising Comparison (Top: Depth Map, Bottom: 3D Surface)\n'
             'Lesion is a ~0.3mm raised bump. Paper Figure 1 likely used heavy smoothing.',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig32_skinl2_final_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig32")

# ============ Export best (Gaussian σ=15) as GLB ============
best = relief_gauss
step = 2
Z = best[::step, ::step]
rgb_sub = rgb_crop[::step, ::step]
rows, cols = Z.shape
xs = np.arange(cols) * step * 0.02  # mm
ys = np.arange(rows) * step * 0.02
X, Y = np.meshgrid(xs, ys)

vertices = np.column_stack([X.ravel(), -Y.ravel(), Z.ravel()])
faces = []
for i in range(rows-1):
    for j in range(cols-1):
        idx = i*cols + j
        faces.append([idx, idx+cols, idx+1])
        faces.append([idx+1, idx+cols, idx+cols+1])

mesh = trimesh.Trimesh(
    vertices=np.array(vertices),
    faces=np.array(faces),
    vertex_colors=np.column_stack([rgb_sub.reshape(-1,3), np.full(rows*cols, 255, dtype=np.uint8)]),
    process=False
)
mesh.export(str(out_dir / 'skinl2_hemangioma_0001_gauss15.glb'), file_type='glb')
print("Saved GLB (Gaussian σ=15)")

print("\n=== INVESTIGATION SUMMARY ===")
print("1. Raytrix API depth has ~0.1mm noise std vs ~0.3mm lesion bump")
print("2. Multi-view stereo from 81 views FAILED — baseline too small (~5-7px parallax)")
print("3. Best denoising: Gaussian σ=15 on raw depth, then plane removal")
print("4. Alternatives (NL-means+Guided, Bilateral+Guided) offer marginal improvement")
print("5. Paper Figure 1 used heavy smoothing (likely from Raytrix proprietary software)")
print("6. For evaluation: use Raytrix depth as-is (scale-invariant metrics tolerate noise)")
