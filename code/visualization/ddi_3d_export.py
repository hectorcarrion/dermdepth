#!/usr/bin/env python3
"""DDI 3D Reconstruction: Normals, GLB Export, Animation, Volume.

Uses 47 DDI images with fully-visible ruler masks (FEDD class 3) and cached
depth predictions from 5 methods. The known ruler area (6.6 cm²) provides
per-image scale correction: scale = sqrt(GT_area / pred_area).

Phases (all use cached depth from output/evaluation/ddi_rulers/_cache/{method}/):
  --normals   Normal map comparison PDF (derives normals from calibrated depth)
  --glb       Export textured 3D meshes as GLB files
  --animate   Turntable animation MP4s (matplotlib 3D, headless)
  --volume    Lesion volume estimation (FEDD class 1 = lesion)
  --all       Run all phases

Usage:
  # Normal comparison (no GPU needed, uses cached depths)
  conda run -n MoGe python -u code/visualization/ddi_3d_export.py --normals

  # GLB export
  conda run -n MoGe python -u code/visualization/ddi_3d_export.py --glb

  # Turntable animation
  conda run -n MoGe python -u code/visualization/ddi_3d_export.py --animate

  # Lesion volume
  conda run -n MoGe python -u code/visualization/ddi_3d_export.py --volume

  # All phases
  conda run -n MoGe python -u code/visualization/ddi_3d_export.py --all
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))
import utils3d

# ── Paths ────────────────────────────────────────────────────────────────────
DDI_IMAGES = PROJECT_ROOT / "data" / "DDI" / "images"
DDI_MAP = PROJECT_ROOT / "data" / "DDI" / "map.csv"
LABELS_DIR = PROJECT_ROOT / "data" / "DDI" / "FEDD" / "ddi_labels"
CACHE_DIR = PROJECT_ROOT / "output" / "evaluation" / "ddi_rulers" / "_cache"

OUT_NORMALS = PROJECT_ROOT / "output" / "figures" / "ddi_normals"
OUT_3D = PROJECT_ROOT / "output" / "3d_exports" / "ddi"
OUT_VOLUME = PROJECT_ROOT / "output" / "evaluation" / "ddi_rulers"

# ── Constants ────────────────────────────────────────────────────────────────
RULER_LENGTH_CM = 6.0
RULER_WIDTH_CM = 1.1
GT_AREA_CM2 = RULER_LENGTH_CM * RULER_WIDTH_CM  # 6.6 cm²
EXCLUDED = {'000186', '000559'}

METHODS = ['dermdepth', 'exp_d3', 'da3nested', 'mapanything', 'ppd', 'moge2']
METHOD_LABELS = {
    'moge2': 'MoGe-2',
    'dermdepth': 'DermDepth (Exp A)',
    'exp_d3': 'DermDepth-D3',
    'da3nested': 'DA3-Nested',
    'mapanything': 'MapAnything',
    'ppd': 'PPD',
}

# MoGe-family checkpoints (these models predict normals directly)
MOGE_CHECKPOINTS = {
    'moge2': 'Ruicheng/moge-2-vitl-normal',
    'dermdepth': str(PROJECT_ROOT / "output" / "training" / "exp_a" / "checkpoint" / "00001000_ema.pt"),
    'exp_d1': str(PROJECT_ROOT / "output" / "training" / "exp_d1" / "checkpoint" / "00002500_ema.pt"),
    'exp_d3': str(PROJECT_ROOT / "output" / "training" / "exp_d3" / "checkpoint" / "00002500_ema.pt"),
}

# Normal comparison: dedicated model list (separate from depth-based METHODS)
NORMAL_MODELS = [
    ('MoGe-2\n(base)', 'moge2'),
    ('Exp A\n(scale only)', 'dermdepth'),
    ('D1\n(norm from A)', 'exp_d1'),
    ('D3\n(joint)', 'exp_d3'),
]


# ════════════════════════════════════════════════════════════════════════════
#  Reused helpers (from eval_ddi_rulers.py)
# ════════════════════════════════════════════════════════════════════════════

def load_ddi_metadata():
    import csv
    meta = {}
    with open(DDI_MAP) as f:
        for row in csv.DictReader(f):
            meta[row['DDI_file']] = row
    return meta


def collect_label_files():
    label_files = {}
    for f in LABELS_DIR.rglob('*.npy'):
        label_files.setdefault(f.name, f)
    return label_files


def get_ruler_samples():
    label_files = collect_label_files()
    meta = load_ddi_metadata()
    samples = []
    for name, path in sorted(label_files.items()):
        stem = name.replace('.npy', '')
        if stem in EXCLUDED:
            continue
        mask = np.load(path)
        if 3 not in mask:
            continue
        ruler = (mask == 3)
        if (ruler[0, :].any() or ruler[-1, :].any() or
                ruler[:, 0].any() or ruler[:, -1].any()):
            continue
        rows = np.where(ruler.any(axis=1))[0]
        cols = np.where(ruler.any(axis=0))[0]
        bb_h = rows[-1] - rows[0] + 1
        bb_w = cols[-1] - cols[0] + 1
        ar = max(bb_h, bb_w) / max(min(bb_h, bb_w), 1)
        if ar < 3.0 or ar > 10.0:
            continue
        filename = stem + '.png'
        if not (DDI_IMAGES / filename).exists():
            continue
        m = meta.get(filename, {})
        samples.append({
            'stem': stem,
            'filename': filename,
            'label_path': str(path),
            'skin_tone': m.get('skin_tone', '?'),
            'disease': m.get('disease', '?'),
            'malignant': m.get('malignant', '?'),
        })
    return samples


def load_mask(label_path, img_h, img_w, cls=3):
    """Load FEDD mask for given class, upscale to image resolution."""
    from scipy.ndimage import zoom
    mask_256 = np.load(label_path)
    binary = (mask_256 == cls).astype(np.uint8)
    if binary.sum() == 0:
        return None
    scale_h = img_h / binary.shape[0]
    scale_w = img_w / binary.shape[1]
    full = zoom(binary, (scale_h, scale_w), order=0)
    return full.astype(bool)


def estimate_intrinsics(height, width, fov_deg=60.0):
    fx = fy = width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def compute_surface_area(depth, mask, intrinsics):
    h, w = depth.shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    X = (jj - cx) * depth / fx
    Y = (ii - cy) * depth / fy
    Z = depth
    dXdx = np.zeros_like(X); dYdx = np.zeros_like(Y); dZdx = np.zeros_like(Z)
    dXdx[:, :-1] = X[:, 1:] - X[:, :-1]
    dYdx[:, :-1] = Y[:, 1:] - Y[:, :-1]
    dZdx[:, :-1] = Z[:, 1:] - Z[:, :-1]
    dXdy = np.zeros_like(X); dYdy = np.zeros_like(Y); dZdy = np.zeros_like(Z)
    dXdy[:-1, :] = X[1:, :] - X[:-1, :]
    dYdy[:-1, :] = Y[1:, :] - Y[:-1, :]
    dZdy[:-1, :] = Z[1:, :] - Z[:-1, :]
    nx = dYdx * dZdy - dZdx * dYdy
    ny = dZdx * dXdy - dXdx * dZdy
    nz = dXdx * dYdy - dYdx * dXdy
    area_element = np.sqrt(nx**2 + ny**2 + nz**2)
    valid = mask & np.isfinite(depth) & (depth > 0)
    valid[:-1, :] &= np.isfinite(depth[1:, :])
    valid[:, :-1] &= np.isfinite(depth[:, 1:])
    return float(np.sum(area_element[valid])), int(valid.sum())


def load_depth(stem, method, img_h, img_w):
    """Load cached depth, resize to image resolution if needed."""
    from scipy.ndimage import zoom
    path = CACHE_DIR / method / f"{stem}_depth.npy"
    if not path.exists():
        return None
    depth = np.load(path)
    if depth.shape[:2] != (img_h, img_w):
        depth = zoom(depth, (img_h / depth.shape[0], img_w / depth.shape[1]), order=1)
    return depth


def calibrate_depth(depth, ruler_mask, intrinsics):
    """Apply ruler-based scale correction. Returns (calibrated_depth, scale_factor)."""
    area_m2, _ = compute_surface_area(depth, ruler_mask, intrinsics)
    area_cm2 = area_m2 * 1e4
    if area_cm2 <= 0:
        return depth, 1.0
    k = np.sqrt(GT_AREA_CM2 / area_cm2)
    return depth * k, float(k)


# ════════════════════════════════════════════════════════════════════════════
#  Normal map helpers
# ════════════════════════════════════════════════════════════════════════════

def colorize_normal(normal, mask=None):
    """MoGe standard colormap: R=X, G=-Y, B=-Z (blue = camera-facing)."""
    n = normal.copy()
    if mask is not None:
        n = np.where(mask[..., None], n, 0)
    else:
        invalid = ~np.all(np.isfinite(n), axis=-1)
        n[invalid] = 0
    n = n * [0.5, -0.5, -0.5] + 0.5
    return (n.clip(0, 1) * 255).astype(np.uint8)


def derive_normals(depth, intrinsics, mask=None):
    """Derive normal map from depth using utils3d (fallback for non-MoGe methods)."""
    if mask is None:
        mask = np.isfinite(depth) & (depth > 0)
    normal, normal_mask = utils3d.np.depth_map_to_normal_map(
        depth, intrinsics=intrinsics, mask=mask)
    normal = np.where(normal_mask[..., None], normal, np.nan)
    return normal


def load_moge_model(checkpoint_path, device='cuda'):
    """Load a MoGe-family model (same as normal_comparison_grid.py)."""
    import os
    import torch
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        cfg = ckpt.get('model_config', None)
        model = MoGeModel(**cfg) if cfg else MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model = MoGeModel.from_pretrained(checkpoint_path)
    return model.to(device).eval()


def infer_normal(model, image_path, device='cuda'):
    """Run MoGe model inference to get predicted normal map."""
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model.infer(img_tensor)
    n = out['normal']
    if isinstance(n, torch.Tensor):
        n = n.cpu().numpy()
        if n.ndim > 3 and n.shape[0] == 1:
            n = n.squeeze(0)
    return n


# ════════════════════════════════════════════════════════════════════════════
#  Mesh helpers
# ════════════════════════════════════════════════════════════════════════════

def depth_to_mesh(depth, image, intrinsics, mask=None):
    """Convert depth map + image to textured triangle mesh."""
    import trimesh

    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    X = (jj - cx) * depth / fx
    Y = (ii - cy) * depth / fy
    Z = depth

    vertices = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

    # Ensure image matches depth resolution
    if image.shape[:2] != (h, w):
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(image).resize((w, h), PILImage.LANCZOS)
        image = np.array(img_pil)

    colors = image  # uint8 (H, W, 3)

    # Valid pixels
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask

    # Vertex indices
    idx = np.arange(h * w).reshape(h, w)
    quad_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]

    qi, qj = np.where(quad_valid)
    # Winding order: CCW when viewed from camera (face normals toward -Z)
    f1 = np.stack([idx[qi, qj], idx[qi + 1, qj], idx[qi, qj + 1]], axis=-1)
    f2 = np.stack([idx[qi + 1, qj], idx[qi + 1, qj + 1], idx[qi, qj + 1]], axis=-1)
    faces = np.concatenate([f1, f2], axis=0)

    # Add alpha channel to colors
    alpha = np.full((h * w, 1), 255, dtype=np.uint8)
    vc = np.concatenate([colors.reshape(-1, 3), alpha], axis=-1)

    mesh = trimesh.Trimesh(
        vertices=vertices.reshape(-1, 3),
        faces=faces,
        vertex_colors=vc,
    )
    return mesh


# ════════════════════════════════════════════════════════════════════════════
#  Phase 0: Normal Map Comparison PDF
# ════════════════════════════════════════════════════════════════════════════

def phase_normals(samples, available_methods, device='cuda'):
    """Generate normal map comparison PDF using model inference.

    Uses NORMAL_MODELS list: loads each MoGe-family checkpoint and runs
    model.infer() for direct normal prediction. Layout per row:
    [Image | MoGe-2 (base) | Exp A (scale only) | D1 (norm from A) | D3 (joint)]
    """
    from PIL import Image
    from scipy.ndimage import zoom as scipy_zoom

    OUT_NORMALS.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_NORMALS / "ddi_normal_comparison.pdf"

    # Load all normal models
    print("  Loading normal models...")
    loaded_models = {}
    for label, key in NORMAL_MODELS:
        ckpt = MOGE_CHECKPOINTS[key]
        print(f"    {label.replace(chr(10), ' ')}...", end="", flush=True)
        loaded_models[label] = load_moge_model(ckpt, device)
        print(" OK")

    n_cols = 1 + len(NORMAL_MODELS)  # Image + models
    col_labels = ['Image'] + [label for label, _ in NORMAL_MODELS]
    rows_per_page = 6

    print(f"\n=== Phase 0: Normal Map Comparison PDF ===")
    print(f"  Samples: {len(samples)}, Models: {len(NORMAL_MODELS)}")
    print(f"  Output: {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(samples), rows_per_page):
            page_samples = samples[page_start:page_start + rows_per_page]
            n_rows = len(page_samples)

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 2.8, n_rows * 2.5),
                                     squeeze=False)

            for si, s in enumerate(page_samples):
                img = Image.open(DDI_IMAGES / s['filename']).convert('RGB')
                img_np = np.array(img)
                img_w, img_h = img.size

                # Input image
                axes[si, 0].imshow(img_np)
                axes[si, 0].axis('off')
                if si == 0:
                    axes[si, 0].set_title(col_labels[0], fontsize=9, fontweight='bold')

                for mi, (label, _) in enumerate(NORMAL_MODELS):
                    col = 1 + mi
                    model = loaded_models[label]

                    normal = infer_normal(model,
                                          str(DDI_IMAGES / s['filename']), device)
                    if normal.shape[:2] != (img_h, img_w):
                        normal = scipy_zoom(normal,
                                            (img_h / normal.shape[0],
                                             img_w / normal.shape[1], 1), order=1)
                    rgb = colorize_normal(normal)
                    axes[si, col].imshow(rgb)

                    axes[si, col].axis('off')
                    if si == 0:
                        axes[si, col].set_title(col_labels[col], fontsize=9,
                                                fontweight='bold')

                # Row label
                label = f"{s['stem']} ({s['skin_tone']})"
                axes[si, 0].set_ylabel(label, fontsize=6, rotation=0,
                                       labelpad=5, ha='right', va='center')

            page_num = page_start // rows_per_page + 1
            total_pages = (len(samples) + rows_per_page - 1) // rows_per_page
            fig.suptitle(f'DDI Normal Comparison (page {page_num}/{total_pages})',
                         fontsize=12, fontweight='bold', y=1.01)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Page {page_num}/{total_pages} ({len(page_samples)} rows)")

    print(f"  Saved: {pdf_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Phase 1: GLB Export
# ════════════════════════════════════════════════════════════════════════════

def phase_glb(samples, available_methods):
    """Export textured 3D meshes as GLB files."""
    from PIL import Image

    print(f"\n=== Phase 1: GLB Export ===")
    print(f"  Samples: {len(samples)}, Methods: {len(available_methods)}")

    exported = 0
    for si, s in enumerate(samples):
        img = Image.open(DDI_IMAGES / s['filename']).convert('RGB')
        img_np = np.array(img)
        img_w, img_h = img.size
        intrinsics = estimate_intrinsics(img_h, img_w)
        ruler_mask = load_mask(s['label_path'], img_h, img_w, cls=3)
        if ruler_mask is None:
            continue

        sample_dir = OUT_3D / s['stem']
        sample_dir.mkdir(parents=True, exist_ok=True)

        for method in available_methods:
            out_path = sample_dir / f"{method}.glb"
            if out_path.exists():
                exported += 1
                continue

            depth = load_depth(s['stem'], method, img_h, img_w)
            if depth is None:
                continue

            depth_cal, k = calibrate_depth(depth, ruler_mask, intrinsics)
            mesh = depth_to_mesh(depth_cal, img_np, intrinsics)
            mesh.export(str(out_path))
            exported += 1

        if si < 3 or si % 10 == 0:
            print(f"  [{si+1}/{len(samples)}] {s['stem']}")

    print(f"  Exported {exported} GLB files to {OUT_3D}")


# ════════════════════════════════════════════════════════════════════════════
#  Phase 2: Turntable Animation
# ════════════════════════════════════════════════════════════════════════════

def render_turntable_frame(vertices, colors, elev, azim, ax, subsample=4):
    """Render a single frame of the mesh using matplotlib scatter."""
    # Subsample for speed
    step = max(1, subsample)
    vs = vertices[::step]
    cs = colors[::step]

    # Filter valid
    valid = np.all(np.isfinite(vs), axis=1)
    vs = vs[valid]
    cs = cs[valid]

    ax.clear()
    ax.scatter(vs[:, 0], vs[:, 1], vs[:, 2],
               c=cs, s=0.3, marker='.', linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # Auto-scale axes equally
    center = vs.mean(axis=0)
    extent = np.abs(vs - center).max()
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)


def phase_animate(samples, available_methods, n_frames=60, fps=30, subsample=8):
    """Generate turntable animation MP4s."""
    import imageio
    from PIL import Image
    from io import BytesIO

    print(f"\n=== Phase 2: Turntable Animation ===")
    print(f"  Samples: {len(samples)}, Methods: {len(available_methods)}")
    print(f"  Frames: {n_frames}, FPS: {fps}, subsample: {subsample}")

    for si, s in enumerate(samples):
        sample_dir = OUT_3D / s['stem']
        out_path = sample_dir / "turntable.mp4"
        if out_path.exists():
            print(f"  [{si+1}/{len(samples)}] {s['stem']} — skipping (exists)")
            continue

        img = Image.open(DDI_IMAGES / s['filename']).convert('RGB')
        img_np = np.array(img)
        img_w, img_h = img.size
        intrinsics = estimate_intrinsics(img_h, img_w)
        ruler_mask = load_mask(s['label_path'], img_h, img_w, cls=3)
        if ruler_mask is None:
            continue

        # Load and calibrate all methods
        method_data = {}
        for method in available_methods:
            depth = load_depth(s['stem'], method, img_h, img_w)
            if depth is None:
                continue
            depth_cal, _ = calibrate_depth(depth, ruler_mask, intrinsics)

            # Backproject to 3D
            h, w = depth_cal.shape
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            jj, ii = np.meshgrid(np.arange(w), np.arange(h))
            X = (jj - cx) * depth_cal / fx
            Y = (ii - cy) * depth_cal / fy
            Z = depth_cal
            verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

            # Match image to depth resolution
            if img_np.shape[:2] != (h, w):
                img_resized = np.array(Image.fromarray(img_np).resize((w, h), Image.LANCZOS))
            else:
                img_resized = img_np
            cols = img_resized.reshape(-1, 3) / 255.0

            method_data[method] = (verts, cols)

        if not method_data:
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)

        n_methods = len(method_data)
        panel_w = 3.0
        panel_h = 3.0

        azimuths = np.linspace(0, 360, n_frames, endpoint=False)
        frames = []

        for fi, azim in enumerate(azimuths):
            fig = plt.figure(figsize=(panel_w * n_methods, panel_h), dpi=100)

            for mi, (method, (verts, cols)) in enumerate(method_data.items()):
                ax = fig.add_subplot(1, n_methods, mi + 1, projection='3d')
                render_turntable_frame(verts, cols, elev=20, azim=azim,
                                       ax=ax, subsample=subsample)
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=10,
                             fontweight='bold', pad=-5)

            plt.tight_layout(pad=0.5)

            # Render to numpy array
            buf = BytesIO()
            fig.savefig(buf, format='png', facecolor='white',
                        bbox_inches='tight', dpi=100)
            plt.close(fig)
            buf.seek(0)
            frame = np.array(Image.open(buf))[:, :, :3]  # drop alpha
            frames.append(frame)

            if fi == 0:
                print(f"  [{si+1}/{len(samples)}] {s['stem']} — frame size: {frame.shape}")

        # Pad frames to consistent size (tight bbox may vary slightly)
        max_h = max(f.shape[0] for f in frames)
        max_w = max(f.shape[1] for f in frames)
        # Round up to even (required by most codecs)
        max_h = max_h + (max_h % 2)
        max_w = max_w + (max_w % 2)
        padded = []
        for f in frames:
            canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            canvas[:f.shape[0], :f.shape[1]] = f
            padded.append(canvas)

        writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264',
                                    pixelformat='yuv420p')
        for frame in padded:
            writer.append_data(frame)
        writer.close()

        print(f"  [{si+1}/{len(samples)}] {s['stem']} — {out_path.name} "
              f"({n_frames} frames, {max_h}x{max_w})")

    print(f"  Animations saved to {OUT_3D}")


# ════════════════════════════════════════════════════════════════════════════
#  Phase 3: Lesion Volume Estimation
# ════════════════════════════════════════════════════════════════════════════

def compute_lesion_volume(depth, lesion_mask, intrinsics):
    """Compute lesion 'bump volume' above surrounding skin plane.

    1. Backproject lesion pixels to 3D
    2. Fit plane to boundary pixels (surrounding skin surface)
    3. Volume = sum of heights above plane × pixel area elements
    Returns (volume, surface_area, n_pixels) in depth-unit cubed.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    X = (jj - cx) * depth / fx
    Y = (ii - cy) * depth / fy
    Z = depth

    valid_depth = np.isfinite(depth) & (depth > 0)
    lesion_valid = lesion_mask & valid_depth

    if lesion_valid.sum() < 10:
        return 0.0, 0.0, 0

    # Boundary: dilated minus eroded lesion mask
    dilated = binary_dilation(lesion_mask, iterations=3)
    eroded = binary_erosion(lesion_mask, iterations=3)
    boundary = dilated & ~eroded & valid_depth
    if boundary.sum() < 3:
        boundary = lesion_valid  # fallback

    # Fit plane to boundary pixels: aX + bY + cZ = d
    bx = X[boundary]
    by = Y[boundary]
    bz = Z[boundary]
    A_mat = np.column_stack([bx, by, np.ones_like(bx)])
    # Least-squares: Z = a*X + b*Y + c
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, bz, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0

    a, b, c = coeffs
    # Plane Z at each lesion pixel
    plane_z = a * X[lesion_valid] + b * Y[lesion_valid] + c
    actual_z = Z[lesion_valid]

    # Height above plane (positive = protrusion)
    heights = actual_z - plane_z

    # Pixel area elements via cross-product (same as compute_surface_area)
    dXdx = np.zeros_like(X); dYdx = np.zeros_like(Y); dZdx = np.zeros_like(Z)
    dXdx[:, :-1] = X[:, 1:] - X[:, :-1]
    dYdx[:, :-1] = Y[:, 1:] - Y[:, :-1]
    dZdx[:, :-1] = Z[:, 1:] - Z[:, :-1]
    dXdy = np.zeros_like(X); dYdy = np.zeros_like(Y); dZdy = np.zeros_like(Z)
    dXdy[:-1, :] = X[1:, :] - X[:-1, :]
    dYdy[:-1, :] = Y[1:, :] - Y[:-1, :]
    dZdy[:-1, :] = Z[1:, :] - Z[:-1, :]
    nx = dYdx * dZdy - dZdx * dYdy
    ny = dZdx * dXdy - dXdx * dZdy
    nz = dXdx * dYdy - dYdx * dXdy
    area_elem = np.sqrt(nx**2 + ny**2 + nz**2)

    pixel_areas = area_elem[lesion_valid]

    # Volume: sum of height × pixel_area for protrusion
    volume = float(np.sum(np.maximum(heights, 0) * pixel_areas))

    # Surface area
    surface_area = float(np.sum(pixel_areas))
    n_pixels = int(lesion_valid.sum())

    return volume, surface_area, n_pixels


def phase_volume(samples, available_methods):
    """Estimate lesion volume for samples with FEDD lesion mask (class 1)."""
    from PIL import Image

    print(f"\n=== Phase 3: Lesion Volume Estimation ===")
    print(f"  Samples: {len(samples)}, Methods: {len(available_methods)}")

    results = []
    by_disease = defaultdict(lambda: defaultdict(list))
    by_tone = defaultdict(lambda: defaultdict(list))

    n_with_lesion = 0

    for si, s in enumerate(samples):
        img = Image.open(DDI_IMAGES / s['filename']).convert('RGB')
        img_w, img_h = img.size
        intrinsics = estimate_intrinsics(img_h, img_w)

        ruler_mask = load_mask(s['label_path'], img_h, img_w, cls=3)
        lesion_mask = load_mask(s['label_path'], img_h, img_w, cls=1)

        if ruler_mask is None or lesion_mask is None:
            continue

        n_with_lesion += 1
        sample_result = {
            'stem': s['stem'],
            'skin_tone': s['skin_tone'],
            'disease': s['disease'],
            'malignant': s['malignant'],
            'image_size': [img_w, img_h],
            'lesion_pixels': int(lesion_mask.sum()),
            'methods': {},
        }

        for method in available_methods:
            depth = load_depth(s['stem'], method, img_h, img_w)
            if depth is None:
                continue

            depth_cal, k = calibrate_depth(depth, ruler_mask, intrinsics)

            volume_m3, area_m2, n_px = compute_lesion_volume(
                depth_cal, lesion_mask, intrinsics)

            # Convert to clinical units
            volume_cm3 = volume_m3 * 1e6   # m³ → cm³
            area_cm2 = area_m2 * 1e4       # m² → cm²

            sample_result['methods'][method] = {
                'volume_cm3': float(volume_cm3),
                'surface_area_cm2': float(area_cm2),
                'scale_factor': float(k),
                'n_valid_pixels': n_px,
            }

            by_disease[s['disease']][method].append(volume_cm3)
            by_tone[s['skin_tone']][method].append(volume_cm3)

        results.append(sample_result)

        if si < 5 or si % 10 == 0:
            vols = ", ".join(
                f"{m}={sample_result['methods'][m]['volume_cm3']:.4f}"
                for m in available_methods if m in sample_result['methods']
            )
            print(f"  [{si+1}/{len(samples)}] {s['stem']} ({s['disease'][:20]}): {vols} cm³")

    # Summary
    print(f"\n  Samples with lesion mask: {n_with_lesion}/{len(samples)}")

    # Per-disease summary
    disease_summary = {}
    for disease, method_vols in sorted(by_disease.items()):
        disease_summary[disease] = {}
        for method in available_methods:
            vols = method_vols.get(method, [])
            if vols:
                disease_summary[disease][method] = {
                    'mean_volume_cm3': float(np.mean(vols)),
                    'median_volume_cm3': float(np.median(vols)),
                    'std_volume_cm3': float(np.std(vols)),
                    'n': len(vols),
                }

    # Per-skin-tone summary
    tone_summary = {}
    for tone, method_vols in sorted(by_tone.items()):
        tone_summary[tone] = {}
        for method in available_methods:
            vols = method_vols.get(method, [])
            if vols:
                tone_summary[tone][method] = {
                    'mean_volume_cm3': float(np.mean(vols)),
                    'median_volume_cm3': float(np.median(vols)),
                    'n': len(vols),
                }

    # Print summary table
    print(f"\n{'Disease':<30} ", end='')
    for m in available_methods:
        print(f" {METHOD_LABELS.get(m, m):>12}", end='')
    print(f" {'n':>5}")
    print("-" * (30 + 13 * len(available_methods) + 6))

    for disease in sorted(disease_summary.keys()):
        row = f"{disease[:28]:<30} "
        n = 0
        for m in available_methods:
            d = disease_summary[disease].get(m, {})
            if d:
                row += f" {d['median_volume_cm3']:>11.4f}"
                n = d['n']
            else:
                row += f" {'N/A':>12}"
        row += f" {n:>5}"
        print(row)

    # Save
    OUT_VOLUME.mkdir(parents=True, exist_ok=True)
    out_path = OUT_VOLUME / "lesion_volume_results.json"
    output = {
        'gt_ruler_area_cm2': GT_AREA_CM2,
        'n_samples_with_lesion': n_with_lesion,
        'disease_summary': disease_summary,
        'skin_tone_summary': tone_summary,
        'per_sample': results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Phase 4: Prepare animation data packages (for render_animation.py)
# ════════════════════════════════════════════════════════════════════════════

OUT_ANIM = PROJECT_ROOT / "output" / "3d_exports" / "ddi_animations"


def phase_prepare(samples, method='exp_d3', device='cuda'):
    """Prepare data packages for render_animation.py.

    Each package contains:
      - image.png (original DDI image)
      - depth_calibrated.npy (ruler-scaled depth, float32)
      - segmentation.npy (lesion mask, uint8, 0/1)
      - normal_map_colorized.png (colorized normal from model inference)
      - scale_estimation.json (intrinsics, scale factor, metadata)
    """
    from PIL import Image
    import cv2

    print(f"\n=== Phase 4: Prepare Animation Packages ===")
    print(f"  Samples: {len(samples)}, Method: {method}")

    # Load model for normal inference
    ckpt = MOGE_CHECKPOINTS[method]
    print(f"  Loading {method} model...", end="", flush=True)
    model = load_moge_model(ckpt, device)
    print(" OK")

    prepared = 0
    for si, s in enumerate(samples):
        pkg_dir = OUT_ANIM / s['stem']

        # Skip if already complete
        if (pkg_dir / "scale_estimation.json").exists():
            prepared += 1
            if si < 3:
                print(f"  [{si+1}/{len(samples)}] {s['stem']} — skipped (exists)")
            continue

        img = Image.open(DDI_IMAGES / s['filename']).convert('RGB')
        img_np = np.array(img)
        img_w, img_h = img.size
        intrinsics = estimate_intrinsics(img_h, img_w)

        # Ruler mask for calibration
        ruler_mask = load_mask(s['label_path'], img_h, img_w, cls=3)
        if ruler_mask is None:
            continue

        # Lesion mask (FEDD class 1)
        lesion_mask = load_mask(s['label_path'], img_h, img_w, cls=1)
        if lesion_mask is None:
            lesion_mask = np.zeros((img_h, img_w), dtype=bool)

        # Load and calibrate depth
        depth = load_depth(s['stem'], method, img_h, img_w)
        if depth is None:
            print(f"  [{si+1}/{len(samples)}] {s['stem']} — no cached depth for {method}")
            continue
        depth_cal, k = calibrate_depth(depth, ruler_mask, intrinsics)

        # Predict normals via model inference
        normal = infer_normal(model, str(DDI_IMAGES / s['filename']), device)
        if normal.shape[:2] != (img_h, img_w):
            from scipy.ndimage import zoom as scipy_zoom
            normal = scipy_zoom(normal,
                                (img_h / normal.shape[0],
                                 img_w / normal.shape[1], 1), order=1)
        normal_rgb = colorize_normal(normal)

        # Compute ruler area for metadata
        area_m2, _ = compute_surface_area(depth, ruler_mask, intrinsics)
        pred_area_cm2 = area_m2 * 1e4

        # Save package
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # Image
        cv2.imwrite(str(pkg_dir / "image.png"),
                     cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # Calibrated depth
        np.save(pkg_dir / "depth_calibrated.npy", depth_cal.astype(np.float32))

        # Segmentation (lesion = 1, background = 0)
        np.save(pkg_dir / "segmentation.npy", lesion_mask.astype(np.uint8))

        # Colorized normal map
        cv2.imwrite(str(pkg_dir / "normal_map_colorized.png"),
                     cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))

        # Metadata JSON
        meta = {
            'intrinsics': intrinsics.tolist(),
            'scale_factor': k,
            'pred_ruler_area_cm2': pred_area_cm2,
            'gt_ruler_area_cm2': GT_AREA_CM2,
            'fov_deg': 60.0,
            'stem': s['stem'],
            'method': method,
            'skin_tone': s['skin_tone'],
            'disease': s['disease'],
        }
        with open(pkg_dir / "scale_estimation.json", 'w') as f:
            json.dump(meta, f, indent=2)

        prepared += 1
        if si < 3 or si % 10 == 0:
            print(f"  [{si+1}/{len(samples)}] {s['stem']} "
                  f"({img_w}x{img_h}, k={k:.3f}, lesion={int(lesion_mask.sum())}px)")

    print(f"  Prepared {prepared} packages in {OUT_ANIM}")


def phase_render(samples):
    """Render pyvista animations using prepared data packages."""
    import subprocess

    print(f"\n=== Phase 5: Render Animations (pyvista) ===")

    render_script = PROJECT_ROOT / "code" / "visualization" / "render_animation.py"
    if not render_script.exists():
        print(f"  ERROR: {render_script} not found")
        return

    rendered = 0
    for si, s in enumerate(samples):
        pkg_dir = OUT_ANIM / s['stem']
        out_file = pkg_dir / "lesion_reconstruction.mp4"

        if not (pkg_dir / "scale_estimation.json").exists():
            continue

        if out_file.exists():
            rendered += 1
            if si < 3:
                print(f"  [{si+1}/{len(samples)}] {s['stem']} — skipped (exists)")
            continue

        print(f"  [{si+1}/{len(samples)}] {s['stem']}...", end="", flush=True)
        result = subprocess.run(
            [sys.executable, '-u', str(render_script)],
            env={**os.environ,
                 'RENDER_DATA_DIR': str(pkg_dir),
                 'RENDER_OUTPUT_FILE': str(out_file)},
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            rendered += 1
            print(" OK")
        else:
            print(f" FAILED")
            if result.stderr:
                # Print last few lines of error
                lines = result.stderr.strip().split('\n')
                for line in lines[-3:]:
                    print(f"    {line}")

    print(f"  Rendered {rendered} animations in {OUT_ANIM}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='DDI 3D Reconstruction Export')
    parser.add_argument('--normals', action='store_true', help='Normal map comparison PDF')
    parser.add_argument('--glb', action='store_true', help='Export GLB meshes')
    parser.add_argument('--animate', action='store_true', help='Turntable animation MP4 (matplotlib)')
    parser.add_argument('--volume', action='store_true', help='Lesion volume estimation')
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare animation data packages (D3, needs GPU)')
    parser.add_argument('--render', action='store_true',
                        help='Render pyvista animations from prepared packages')
    parser.add_argument('--all', action='store_true', help='Run all phases')
    parser.add_argument('--fov', type=float, default=60.0, help='Assumed FoV (degrees)')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model inference')
    parser.add_argument('--subsample', type=int, default=8,
                        help='Vertex subsample factor for animation (default: 8)')
    parser.add_argument('--n-frames', type=int, default=60,
                        help='Animation frames (default: 60)')
    args = parser.parse_args()

    if args.all:
        args.normals = args.glb = args.animate = args.volume = True
        args.prepare = args.render = True

    any_phase = (args.normals or args.glb or args.animate or args.volume
                 or args.prepare or args.render)
    if not any_phase:
        parser.print_help()
        print("\nSpecify at least one phase.")
        return

    # Get samples and available methods
    samples = get_ruler_samples()
    print(f"DDI 3D Export — {len(samples)} ruler samples")

    available = []
    for m in METHODS:
        cache = CACHE_DIR / m
        if cache.exists() and len(list(cache.glob("*.npy"))) > 0:
            available.append(m)
    print(f"Available methods: {', '.join(METHOD_LABELS.get(m, m) for m in available)}")

    if not available:
        print("No cached predictions found. Run eval_ddi_rulers.py --save first.")
        return

    if args.normals:
        phase_normals(samples, available, device=args.device)

    if args.glb:
        phase_glb(samples, available)

    if args.animate:
        phase_animate(samples, available, n_frames=args.n_frames,
                      subsample=args.subsample)

    if args.volume:
        phase_volume(samples, available)

    if args.prepare:
        phase_prepare(samples, method='exp_d3', device=args.device)

    if args.render:
        phase_render(samples)

    print("\nDone!")


if __name__ == '__main__':
    main()
