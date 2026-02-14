#!/usr/bin/env python3
"""
Aggressive denoising of SKINL2 depth to recreate paper Figure 1.
Key insight: lesion relief (~0.3mm) is only 3x noise std (0.1mm).
Need very strong denoising that preserves the large-scale lesion shape.
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from scipy import signal
from pathlib import Path

base = Path('/workspace/hector/dermdepth/data/SKINL2/SKINL2_v2/0001/Hemangioma/all_data/Hemangioma')
out_dir = Path('/workspace/hector/dermdepth/output/verification')

# ============ LOAD DATA ============
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
crop = 250
h, w = depth_raw.shape
y1, y2 = max(0, peak_y-crop), min(h, peak_y+crop)
x1, x2 = max(0, peak_x-crop), min(w, peak_x+crop)

depth_crop = depth_raw[y1:y2, x1:x2]
rgb_crop = cv_small[y1:y2, x1:x2]

# Elevation + plane removal
elev = -depth_crop
yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
coeffs, _, _, _ = np.linalg.lstsq(A, elev.ravel(), rcond=None)
plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
relief_raw = elev - plane

print(f"Raw relief: range=[{relief_raw.min():.4f}, {relief_raw.max():.4f}]mm, std={relief_raw.std():.4f}")

# ============ APPROACH A: MULTI-PASS BILATERAL ============
print("\n--- A: Multi-pass bilateral ---")
relief_f32 = relief_raw.astype(np.float32)

# Normalize to [0, 1] range for better bilateral behavior
rmin, rmax = relief_f32.min(), relief_f32.max()
rng = rmax - rmin
norm = (relief_f32 - rmin) / rng

# Multiple passes with increasing d
result_a = norm.copy()
for i, (d, sc, ss) in enumerate([(9, 0.1, 9), (15, 0.08, 15), (25, 0.05, 25), (35, 0.03, 35)]):
    # bilateralFilter works on float32 in [0,1] range
    result_a = cv2.bilateralFilter(result_a, d=d, sigmaColor=sc, sigmaSpace=ss)
    print(f"  Pass {i+1}: d={d}, sigmaColor={sc}, sigmaSpace={ss}")

relief_multipass = result_a * rng + rmin
print(f"  Result std: {relief_multipass.std():.4f}mm")

# ============ APPROACH B: TV DENOISING (Total Variation) ============
print("\n--- B: Total Variation denoising ---")
# Implement Rudin-Osher-Fatemi TV denoising via iterative proximal gradient
def tv_denoise(image, weight=0.1, n_iter=100):
    """Chambolle's projection algorithm for TV denoising."""
    u = image.copy()
    px = np.zeros_like(image)
    py = np.zeros_like(image)
    tau = 0.25  # step size

    for _ in range(n_iter):
        # Gradient of u
        gx = np.diff(u, axis=1, append=u[:, -1:])
        gy = np.diff(u, axis=0, append=u[-1:, :])

        # Update dual variables
        norm = np.sqrt(gx**2 + gy**2)
        norm = np.maximum(norm, 1e-8)

        px = (px + tau * gx) / (1 + tau * norm / weight)
        py = (py + tau * gy) / (1 + tau * norm / weight)

        # Divergence
        dx = px - np.roll(px, 1, axis=1)
        dx[:, 0] = px[:, 0]
        dy = py - np.roll(py, 1, axis=0)
        dy[0, :] = py[0, :]

        u = image + weight * (dx + dy)

    return u

# Different TV weights
relief_tv_light = tv_denoise(relief_raw.astype(np.float64), weight=0.02, n_iter=200).astype(np.float32)
relief_tv_heavy = tv_denoise(relief_raw.astype(np.float64), weight=0.05, n_iter=300).astype(np.float32)
print(f"  TV light std: {relief_tv_light.std():.4f}, heavy std: {relief_tv_heavy.std():.4f}")

# ============ APPROACH C: LOW-PASS FREQUENCY DOMAIN ============
print("\n--- C: Low-pass frequency domain ---")
F = np.fft.fft2(relief_raw)
F_shift = np.fft.fftshift(F)

rows, cols = relief_raw.shape
cy, cx = rows//2, cols//2

# Create Gaussian low-pass filters at different cutoff frequencies
results_freq = {}
for cutoff in [10, 20, 40, 60]:
    Y, X = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    dist = np.sqrt(X*X + Y*Y)
    H = np.exp(-(dist**2) / (2 * cutoff**2))
    F_filtered = F_shift * H
    relief_freq = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered))).astype(np.float32)
    results_freq[cutoff] = relief_freq
    print(f"  Cutoff={cutoff}: std={relief_freq.std():.4f}")

# ============ APPROACH D: NORMAL INTEGRATION ============
print("\n--- D: Normal estimation + Poisson integration ---")
# Compute surface normals from noisy depth
# Smooth the depth first, compute normals, then integrate via Poisson

# Step 1: Compute normals from raw depth (using central differences)
def compute_normals(depth, scale=1.0):
    """Compute surface normals from depth map."""
    dzdx = np.gradient(depth, axis=1) * scale
    dzdy = np.gradient(depth, axis=0) * scale
    normals = np.dstack([-dzdx, -dzdy, np.ones_like(depth)])
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    return normals / (norm + 1e-8)

# Step 2: Smooth normals (this is the key insight - smooth normals, not depth)
normals_raw = compute_normals(relief_raw)
print(f"  Normal z mean: {normals_raw[:,:,2].mean():.4f} (should be ~1.0 for flat)")

# Smooth normal x and y components
for sigma in [3, 5, 10]:
    nx_smooth = gaussian_filter(normals_raw[:,:,0], sigma=sigma)
    ny_smooth = gaussian_filter(normals_raw[:,:,1], sigma=sigma)
    nz = np.ones_like(nx_smooth)

    # Renormalize
    norm = np.sqrt(nx_smooth**2 + ny_smooth**2 + nz**2)
    nx_smooth /= norm
    ny_smooth /= norm

    # Step 3: Poisson integration from smoothed normals
    # Solve Laplacian(z) = div(p, q) where p=-nx/nz, q=-ny/nz
    p = -nx_smooth / (nz / norm + 1e-8)  # dz/dx
    q = -ny_smooth / (nz / norm + 1e-8)  # dz/dy

    # Divergence of (p, q)
    dp_dx = np.gradient(p, axis=1)
    dq_dy = np.gradient(q, axis=0)
    div = dp_dx + dq_dy

    # Solve via FFT (Poisson equation in frequency domain)
    F_div = np.fft.fft2(div)
    rows, cols = div.shape
    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols).reshape(1, -1)
    denom = (2 * np.cos(2*np.pi*u/rows) - 2) + (2 * np.cos(2*np.pi*v/cols) - 2)
    denom[0, 0] = 1  # avoid division by zero
    Z = np.real(np.fft.ifft2(F_div / denom)).astype(np.float32)
    Z -= Z.mean()  # zero-mean

    print(f"  Normal sigma={sigma}: integrated relief std={Z.std():.4f}")

# Use sigma=5 as our normal integration result
sigma_best = 5
nx_s = gaussian_filter(normals_raw[:,:,0], sigma=sigma_best)
ny_s = gaussian_filter(normals_raw[:,:,1], sigma=sigma_best)
nz_s = np.ones_like(nx_s)
norm_s = np.sqrt(nx_s**2 + ny_s**2 + nz_s**2)
p = -nx_s / (nz_s / norm_s + 1e-8)
q = -ny_s / (nz_s / norm_s + 1e-8)
dp_dx = np.gradient(p, axis=1)
dq_dy = np.gradient(q, axis=0)
div = dp_dx + dq_dy
F_div = np.fft.fft2(div)
rows, cols = div.shape
u = np.arange(rows).reshape(-1, 1)
v = np.arange(cols).reshape(1, -1)
denom = (2 * np.cos(2*np.pi*u/rows) - 2) + (2 * np.cos(2*np.pi*v/cols) - 2)
denom[0, 0] = 1
relief_normal = np.real(np.fft.ifft2(F_div / denom)).astype(np.float32)
relief_normal -= relief_normal.mean()

# ============ APPROACH E: RGB-GUIDED ANISOTROPIC DIFFUSION ============
print("\n--- E: RGB-guided anisotropic diffusion ---")
def anisotropic_diffusion(img, guide_gray, n_iter=50, kappa=0.02, gamma=0.1):
    """Perona-Malik diffusion guided by RGB edges."""
    out = img.copy().astype(np.float64)
    # Edge indicator from guide image
    gx = np.gradient(guide_gray.astype(np.float64), axis=1)
    gy = np.gradient(guide_gray.astype(np.float64), axis=0)
    edge_mag = np.sqrt(gx**2 + gy**2)
    # Edge-stopping function (lower diffusion at RGB edges)
    edge_weight = np.exp(-(edge_mag / edge_mag.std())**2)

    for _ in range(n_iter):
        # 4-neighbor differences
        dN = np.roll(out, -1, axis=0) - out
        dS = np.roll(out, 1, axis=0) - out
        dE = np.roll(out, -1, axis=1) - out
        dW = np.roll(out, 1, axis=1) - out

        # Diffusion coefficients (Perona-Malik + edge weight)
        cN = edge_weight * np.exp(-(dN/kappa)**2)
        cS = edge_weight * np.exp(-(dS/kappa)**2)
        cE = edge_weight * np.exp(-(dE/kappa)**2)
        cW = edge_weight * np.exp(-(dW/kappa)**2)

        out += gamma * (cN*dN + cS*dS + cE*dE + cW*dW)

    return out.astype(np.float32)

gray_guide = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
relief_aniso = anisotropic_diffusion(relief_raw, gray_guide, n_iter=100, kappa=0.03, gamma=0.15)
print(f"  Anisotropic result std: {relief_aniso.std():.4f}")

# ============ APPROACH F: MEGA COMBO ============
print("\n--- F: Mega combo (TV + bilateral + guided) ---")
# TV first to remove noise while preserving edges
step1 = tv_denoise(relief_raw.astype(np.float64), weight=0.03, n_iter=200).astype(np.float32)
# Then bilateral
rmin2, rmax2 = step1.min(), step1.max()
step2_norm = (step1 - rmin2) / (rmax2 - rmin2)
for d, sc, ss in [(15, 0.08, 15), (25, 0.05, 25)]:
    step2_norm = cv2.bilateralFilter(step2_norm, d=d, sigmaColor=sc, sigmaSpace=ss)
step2 = step2_norm * (rmax2 - rmin2) + rmin2
# Then guided filter
def guided_filter(guide, src, radius=15, eps=0.01):
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

relief_mega = guided_filter(gray_guide, step2, radius=20, eps=0.0005)
print(f"  Mega combo std: {relief_mega.std():.4f}")

# ============ VISUALIZE BEST CANDIDATES ============
print("\nCreating comparison...")

candidates = {
    'Raw Relief': relief_raw,
    'Multi-pass Bilateral': relief_multipass,
    'TV Denoise (w=0.05)': relief_tv_heavy,
    'Freq LP (cutoff=40)': results_freq[40],
    'Normal Integration (σ=5)': relief_normal,
    'Anisotropic Diffusion': relief_aniso,
    'Mega Combo\n(TV+Bilateral+Guided)': relief_mega,
    'Freq LP (cutoff=20)': results_freq[20],
}

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
vmin = np.percentile(relief_raw, 2)
vmax = np.percentile(relief_raw, 98)

for idx, (name, data) in enumerate(candidates.items()):
    row, col = idx // 4, idx % 4
    im = axes[row, col].imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[row, col].set_title(f'{name}\nstd={data.std():.4f}mm', fontsize=10)
    axes[row, col].axis('off')

plt.suptitle('SKINL2 Hemangioma — Aggressive Denoising Comparison\n'
             f'Lesion relief: ~0.3mm, Noise std: ~0.1mm',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig23_skinl2_denoising_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig23")

# ============ PAPER FIG 1 RECREATION WITH BEST RESULT ============
print("\nCreating Figure 1 recreation with best result...")

# Pick the best: mega combo
best = relief_mega

fig = plt.figure(figsize=(20, 10))

# Left: 3D surface with texture
ax1 = fig.add_subplot(131, projection='3d')
step = 3
Z = best[::step, ::step]
rgb_sub = rgb_crop[::step, ::step].astype(np.float64) / 255.0
ys = np.arange(Z.shape[0]) * step
xs = np.arange(Z.shape[1]) * step
X, Y = np.meshgrid(xs, ys)

ax1.plot_surface(X, Y, Z,
                 facecolors=rgb_sub,
                 rstride=1, cstride=1,
                 shade=True,
                 lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
                 antialiased=True)
ax1.view_init(elev=55, azim=-45)
ax1.set_box_aspect([1, Z.shape[0]/Z.shape[1], 0.4])
ax1.set_zlabel('Relief (mm)')
ax1.set_title('3D Skin Lesion Reconstruction', fontsize=13, fontweight='bold')
ax1.set_xticks([]); ax1.set_yticks([])

# Center: Colored depth map (like paper)
ax2 = fig.add_subplot(132)
im = ax2.imshow(best, cmap='hot_r')
ax2.set_title('Depth Map (relief)', fontsize=13, fontweight='bold')
ax2.axis('off')
plt.colorbar(im, ax=ax2, label='Relief (mm)', shrink=0.7)

# Right: RGB reference
ax3 = fig.add_subplot(133)
ax3.imshow(rgb_crop)
ax3.set_title('Central View', fontsize=13, fontweight='bold')
ax3.axis('off')

plt.suptitle(f'SKINL2 Hemangioma 0001 — Figure 1 Recreation\n'
             f'Lesion bump: {best.max():.3f}mm, Total relief: {best.max()-best.min():.3f}mm',
             fontsize=14)
plt.tight_layout()
plt.savefig(out_dir / 'fig24_skinl2_fig1_best.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig24")

# Also compare authors' colored depth map vs our best
print("\nComparing with authors' colored depth map...")
dc = np.array(Image.open(base / 'Light Field/Depth Map/0001_DepthMapColored.png').convert('RGB'))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(dc)
axes[0].set_title("Authors' Colored Depth Map", fontsize=12)
axes[0].axis('off')

# Our best full-frame (not just crop)
elev_full = -depth_raw
yy_f, xx_f = np.mgrid[0:elev_full.shape[0], 0:elev_full.shape[1]]
A_f = np.column_stack([xx_f.ravel(), yy_f.ravel(), np.ones(xx_f.size)])
c_f, _, _, _ = np.linalg.lstsq(A_f, elev_full.ravel(), rcond=None)
plane_f = c_f[0]*xx_f + c_f[1]*yy_f + c_f[2]
relief_full = elev_full - plane_f

# Apply TV + bilateral to full frame
relief_full_tv = tv_denoise(relief_full.astype(np.float64), weight=0.03, n_iter=200).astype(np.float32)
r1, r2 = relief_full_tv.min(), relief_full_tv.max()
norm_full = (relief_full_tv - r1) / (r2 - r1)
for d, sc, ss in [(15, 0.08, 15), (25, 0.05, 25)]:
    norm_full = cv2.bilateralFilter(norm_full, d=d, sigmaColor=sc, sigmaSpace=ss)
relief_full_clean = norm_full * (r2 - r1) + r1

im1 = axes[1].imshow(relief_full_clean, cmap='hot_r')
axes[1].set_title("Our Denoised Relief (TV+Bilateral)", fontsize=12)
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], shrink=0.7, label='mm')

im2 = axes[2].imshow(relief_full, cmap='hot_r',
                      vmin=np.percentile(relief_full, 2),
                      vmax=np.percentile(relief_full, 98))
axes[2].set_title("Raw Relief (no denoising)", fontsize=12)
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], shrink=0.7, label='mm')

plt.suptitle('Authors\' Colored Depth vs Our Processing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig25_skinl2_vs_authors.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig25")
