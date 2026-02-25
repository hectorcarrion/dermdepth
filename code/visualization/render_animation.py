#!/usr/bin/env python3
"""
Professional 3D Lesion Visualization with Wipe Transitions (v3 - Final)
- Clean colormaps generated from raw data (no embedded colorbars)
- Proper triangulated mesh (no gaps)
- Smooth vertical wipe transitions with text labels
- Sinusoidal wobble orbit centered on lesion
- Seamless loop
"""

import numpy as np
import pyvista as pv
import json
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Configuration ──────────────────────────────────────────────────────
DATA_DIR = os.environ.get("RENDER_DATA_DIR", "ddi_000379_d3_package")
OUTPUT_FILE = os.environ.get("RENDER_OUTPUT_FILE", "lesion_reconstruction.mp4")
RESOLUTION = (1088, 1088)
FPS = 30
TOTAL_SECONDS = 9
N_FRAMES = FPS * TOTAL_SECONDS  # 270
WOBBLE_DEG = 18
ELEV_WOBBLE_DEG = 7
BG_COLOR = "#1a1a2e"

# ── 1. Load Data ──────────────────────────────────────────────────────
print("Loading data...")
with open(os.path.join(DATA_DIR, "scale_estimation.json")) as f:
    meta = json.load(f)

depth = np.load(os.path.join(DATA_DIR, "depth_calibrated.npy"))
mask = np.load(os.path.join(DATA_DIR, "segmentation.npy"))

rgb = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, "image.png")), cv2.COLOR_BGR2RGB)
normal_cmap = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, "normal_map_colorized.png")), cv2.COLOR_BGR2RGB)

h, w = depth.shape
print(f"Image: {w}×{h}")

# ── Generate clean depth colormap (the provided one has an embedded colorbar) ──
print("Generating depth colormap...")
depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
depth_cmap_raw = cm.turbo(depth_norm)[:, :, :3]
depth_cmap = (depth_cmap_raw * 255).astype(np.uint8)

# Use provided normal map colorized (already correct size and colors)
normal_vis = normal_cmap
if normal_vis.shape[:2] != (h, w):
    normal_vis = cv2.resize(normal_vis, (w, h), interpolation=cv2.INTER_LANCZOS4)

print(f"  Depth colormap: {depth_cmap.shape}, Normal vis: {normal_vis.shape}")

# ── 2. Back-project to 3D ────────────────────────────────────────────
print("Building 3D mesh...")
K = np.array(meta["intrinsics"])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

yy, xx = np.mgrid[0:h, 0:w]
z = depth.copy()
x3d = (xx - cx) * z / fx
y3d = -(yy - cy) * z / fy

# ── 3. Create Triangulated Mesh ───────────────────────────────────────
points = np.stack((x3d.ravel(), y3d.ravel(), z.ravel()), axis=-1)

print("  Generating triangle faces...")
r_idx, c_idx = np.mgrid[0:h-1, 0:w-1]
r_flat = r_idx.ravel()
c_flat = c_idx.ravel()

v00 = r_flat * w + c_flat
v10 = (r_flat + 1) * w + c_flat
v01 = r_flat * w + (c_flat + 1)
v11 = (r_flat + 1) * w + (c_flat + 1)

# Filter large depth jumps
depth_flat_arr = depth.ravel()
max_jump = 0.012
d00 = depth_flat_arr[v00]
d10 = depth_flat_arr[v10]
d01 = depth_flat_arr[v01]
d11 = depth_flat_arr[v11]

valid_t1 = (np.abs(d00-d10) < max_jump) & (np.abs(d00-d01) < max_jump) & (np.abs(d10-d01) < max_jump)
valid_t2 = (np.abs(d10-d11) < max_jump) & (np.abs(d10-d01) < max_jump) & (np.abs(d11-d01) < max_jump)

tri1 = np.column_stack([np.full(valid_t1.sum(), 3), v00[valid_t1], v10[valid_t1], v01[valid_t1]])
tri2 = np.column_stack([np.full(valid_t2.sum(), 3), v10[valid_t2], v11[valid_t2], v01[valid_t2]])
faces = np.vstack([tri1, tri2]).ravel()

n_tri = valid_t1.sum() + valid_t2.sum()
print(f"  {n_tri:,} triangles, {h*w:,} vertices")

mesh = pv.PolyData(points, faces=faces)

# Flatten textures
rgb_flat = rgb.reshape(-1, 3).astype(np.uint8)
depth_flat = depth_cmap.reshape(-1, 3).astype(np.uint8)
normal_flat = normal_vis.reshape(-1, 3).astype(np.uint8)
mesh.point_data["tex"] = rgb_flat.copy()

# Normalized x-coords for wipe
x_norm = xx.ravel().astype(np.float32)
x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min())

mesh.compute_normals(cell_normals=True, point_normals=True, inplace=True)

# ── 4. Camera Target ─────────────────────────────────────────────────
lesion_ys, lesion_xs = np.where(mask > 0)
cy_les = int(np.mean(lesion_ys)) if len(lesion_ys) > 0 else h//2
cx_les = int(np.mean(lesion_xs)) if len(lesion_xs) > 0 else w//2

focal_point = np.array([x3d[cy_les, cx_les], y3d[cy_les, cx_les], z[cy_les, cx_les]])
print(f"Focal point: {focal_point}")

mesh_bounds = np.array(mesh.bounds)
mesh_extent = max(mesh_bounds[1]-mesh_bounds[0], mesh_bounds[3]-mesh_bounds[2], mesh_bounds[5]-mesh_bounds[4])
cam_distance = mesh_extent * 1.5

# ── 5. Text Labels (overlay via post-processing) ─────────────────────
# We'll burn text labels into the frame using OpenCV after rendering

LABELS = ["RGB Reconstruction", "Depth Prediction", "Normal Map Prediction"]
LABEL_COLORS = [(255, 255, 255)] * 3
LABEL_BG = [(40, 40, 80)] * 3


def add_label_to_frame(frame, text, progress=1.0):
    """Add a semi-transparent label at the bottom of the frame."""
    h_f, w_f = frame.shape[:2]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Label bar position
    bar_h = th + baseline + 40
    y_start = h_f - bar_h
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w_f, h_f), (26, 26, 46), -1)
    alpha = 0.75 * progress
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Text
    tx = (w_f - tw) // 2
    ty = y_start + (bar_h + th) // 2 - baseline // 2
    text_alpha = min(progress * 2, 1.0)  # fade in faster
    if text_alpha > 0.01:
        txt_overlay = frame.copy()
        cv2.putText(txt_overlay, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        frame = cv2.addWeighted(txt_overlay, text_alpha, frame, 1 - text_alpha, 0)
    
    return frame


# ── 6. Wipe Logic ────────────────────────────────────────────────────
def blend_textures(tex_a, tex_b, wipe_progress, x_norm):
    edge = 0.05
    alpha = np.clip((wipe_progress - x_norm) / edge, 0.0, 1.0)[:, np.newaxis]
    blended = (1.0 - alpha) * tex_a.astype(np.float32) + alpha * tex_b.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def get_texture_and_label(frame_idx, n_frames, textures, labels, x_norm):
    """
    3 segments: RGB→Depth→Normal→(back to RGB for loop)
    Each: 55% hold, 45% wipe
    Returns: texture array, label text, label opacity
    """
    seg_len = n_frames / 3.0
    seg = min(int(frame_idx / seg_len), 2)
    local_t = (frame_idx - seg * seg_len) / seg_len

    tex_a = textures[seg % 3]
    tex_b = textures[(seg + 1) % 3]
    label_a = labels[seg % 3]
    label_b = labels[(seg + 1) % 3]

    hold = 0.55
    if local_t < hold:
        # During hold: label fully visible
        label_opacity = 1.0
        # Gentle fade-in at start of segment
        if local_t < 0.1:
            label_opacity = local_t / 0.1
        return tex_a.copy(), label_a, label_opacity
    else:
        wp = min((local_t - hold) / (1.0 - hold) * 1.05, 1.0)
        tex = blend_textures(tex_a, tex_b, wp, x_norm)
        # During wipe: cross-fade labels
        if wp < 0.5:
            return tex, label_a, 1.0 - wp
        else:
            return tex, label_b, wp
        

# ── 7. Render ─────────────────────────────────────────────────────────
print("Setting up renderer...")
pv.start_xvfb()

plotter = pv.Plotter(off_screen=True, window_size=RESOLUTION)
plotter.set_background(BG_COLOR)

actor = plotter.add_mesh(
    mesh,
    scalars="tex",
    rgb=True,
    smooth_shading=True,
    show_edges=False,
    specular=0.12,
    specular_power=25,
    diffuse=0.85,
    ambient=0.12,
)

# Three-point lighting
plotter.remove_all_lights()
for pos, color, intensity in [
    ((focal_point[0]-0.3, focal_point[1]+0.3, focal_point[2]-0.8), [1.0, 0.97, 0.92], 1.0),
    ((focal_point[0]+0.4, focal_point[1]-0.2, focal_point[2]-0.6), [0.85, 0.9, 1.0], 0.4),
    ((focal_point[0], focal_point[1], focal_point[2]+0.4), [1.0, 1.0, 1.0], 0.2),
]:
    plotter.add_light(pv.Light(position=pos, focal_point=focal_point.tolist(), color=color, intensity=intensity))

# Render frames to images, then compose with labels, then write video
import imageio
writer = imageio.get_writer(OUTPUT_FILE, fps=FPS, codec='libx264',
                            quality=9, pixelformat='yuv420p',
                            macro_block_size=1)

textures_list = [rgb_flat, depth_flat, normal_flat]

print(f"Rendering {N_FRAMES} frames...")
for i in range(N_FRAMES):
    t = i / N_FRAMES
    azimuth = WOBBLE_DEG * np.sin(2 * np.pi * t)
    elevation = ELEV_WOBBLE_DEG * np.sin(4 * np.pi * t)

    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    cam_x = focal_point[0] + cam_distance * np.sin(az_rad) * np.cos(el_rad)
    cam_y = focal_point[1] + cam_distance * np.sin(el_rad)
    cam_z = focal_point[2] - cam_distance * np.cos(az_rad) * np.cos(el_rad)

    plotter.camera.position = (cam_x, cam_y, cam_z)
    plotter.camera.focal_point = focal_point.tolist()
    plotter.camera.up = (0.0, -1.0, 0.0)
    plotter.camera.clipping_range = (cam_distance * 0.01, cam_distance * 10)

    # Get blended texture and label
    tex, label_text, label_opacity = get_texture_and_label(
        i, N_FRAMES, textures_list, LABELS, x_norm
    )
    mesh.point_data["tex"] = tex

    # Render frame
    plotter.render()
    img = plotter.screenshot(return_img=True)
    
    # Add text label overlay
    img = add_label_to_frame(img, label_text, label_opacity)
    
    writer.append_data(img)

    if (i + 1) % 90 == 0:
        print(f"  {100*(i+1)/N_FRAMES:.0f}%")

writer.close()
plotter.close()

file_size = os.path.getsize(OUTPUT_FILE) / (1024*1024)
print(f"\n✓ Saved: {OUTPUT_FILE} ({file_size:.1f} MB)")
print(f"  {RESOLUTION[0]}×{RESOLUTION[1]}, {FPS}fps, {TOTAL_SECONDS}s, seamless loop")
