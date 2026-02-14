#!/usr/bin/env python3
"""
Visualize MoGe normals and 3D reconstructions vs GT for SKINL2 cases.
For each case: RGB | GT normals | MoGe normals | GT 3D | MoGe 3D (scale-aligned)
"""
import os
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "MoGe"))

DATA_ROOT = PROJECT_ROOT / "data" / "SKINL2"
OUTPUT_DIR = PROJECT_ROOT / "output" / "verification"

random.seed(42)


def compute_normals_from_depth(depth, mask=None):
    """Compute surface normals from a depth map using gradients."""
    dz_dy, dz_dx = np.gradient(depth)
    # Normal = (-dz/dx, -dz/dy, 1), then normalize
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    normals = normals / norm
    if mask is not None:
        normals[~mask] = 0
    return normals


def normal_to_rgb(normals):
    """Convert normal map to RGB visualization (standard convention)."""
    # Map from [-1,1] to [0,1]: R=nx, G=ny, B=nz
    rgb = (normals + 1.0) / 2.0
    return np.clip(rgb, 0, 1)


def discover_v1():
    cases = []
    cv_root = DATA_ROOT / 'SKINL2_v1' / 'Central View'
    dm_root = DATA_ROOT / 'SKINL2_v1' / 'DepthMap'
    for cat in sorted(os.listdir(cv_root)):
        cat_cv = cv_root / cat
        cat_dm = dm_root / cat
        if not cat_cv.is_dir() or not cat_dm.is_dir():
            continue
        for sid in sorted(os.listdir(cat_cv)):
            cv_files = list((cat_cv / sid).glob('*.png'))
            dm_files = list((cat_dm / sid).glob('*.tiff'))
            if cv_files and dm_files:
                cases.append(('v1', cat, sid, cv_files[0], dm_files[0]))
    return cases


def discover_v2v3(version):
    cases = []
    root = DATA_ROOT / f'SKINL2_{version}'
    for case_id in sorted(os.listdir(root)):
        case_dir = root / case_id
        if not case_dir.is_dir():
            continue
        for cat in os.listdir(case_dir):
            cv_dir = case_dir / cat / 'Light Field' / 'Central View'
            dm_dir = case_dir / cat / 'Light Field' / 'Depth Map'
            if cv_dir.is_dir() and dm_dir.is_dir():
                cv_files = list(cv_dir.glob('*TotalFocus*.png'))
                dm_files = list(dm_dir.glob('*DepthMap.tiff'))
                if cv_files and dm_files:
                    cases.append((version, cat, case_id, cv_files[0], dm_files[0]))
    return cases


def select_diverse(cases_list, n=5):
    by_cat = {}
    for c in cases_list:
        by_cat.setdefault(c[1], []).append(c)
    selected = []
    for cat in sorted(by_cat.keys()):
        if len(selected) >= n:
            break
        selected.append(random.choice(by_cat[cat]))
    remaining = [c for c in cases_list if c not in selected]
    while len(selected) < n and remaining:
        choice = random.choice(remaining)
        selected.append(choice)
        remaining.remove(choice)
    return selected[:n]


def main():
    device = 'cpu'

    # Discover same cases
    v1_cases = discover_v1()
    v2_cases = discover_v2v3('v2')
    v3_cases = discover_v2v3('v3')
    all_selected = select_diverse(v1_cases, 5) + select_diverse(v2_cases, 5) + select_diverse(v3_cases, 5)

    print(f"Selected {len(all_selected)} cases")

    # Load model
    print(f"Loading MoGe-2 on {device}...")
    from moge.model.v2 import MoGeModel
    import torchvision.transforms.functional as TF
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    model = model.to(device).eval()
    print("Model loaded.\n")

    # Process in pages of 5
    pages = [
        ('v1', all_selected[:5]),
        ('v2', all_selected[5:10]),
        ('v3', all_selected[10:15]),
    ]

    for page_name, page_cases in pages:
        n = len(page_cases)
        # 5 rows × 6 cols: RGB | GT normal | MoGe normal | GT 3D | MoGe 3D (aligned) | MoGe 3D (raw scale)
        fig = plt.figure(figsize=(36, 7 * n))

        for row, (ver, cat, sid, cv_path, dm_path) in enumerate(page_cases):
            label = f"{ver}/{cat}/{sid}"
            print(f"Processing {label}...")

            # Load data
            cv_img = np.array(Image.open(cv_path).convert('RGB'))
            depth_raw = np.array(Image.open(dm_path), dtype=np.float32)

            # GT depth: smooth, absolute, in meters
            gt_depth_mm = np.abs(gaussian_filter(depth_raw, sigma=15))
            gt_depth_m = gt_depth_mm / 1000.0

            # Resize image to depth resolution for display
            cv_small = np.array(Image.fromarray(cv_img).resize(
                (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

            # GT normals from depth
            gt_normals = compute_normals_from_depth(gt_depth_m)

            # MoGe inference
            img_tensor = TF.to_tensor(Image.open(cv_path).convert('RGB')).unsqueeze(0).to(device)
            with torch.inference_mode():
                output = model.infer(img_tensor)

            pred_depth = output['depth'].squeeze(0).cpu().numpy()
            pred_normal = output['normal'].squeeze(0).cpu().numpy()  # (H, W, 3)
            pred_points = output['points'].squeeze(0).cpu().numpy()  # (H, W, 3)
            pred_mask = output['mask'].squeeze(0).cpu().numpy()

            # Resize predictions to GT resolution
            gt_h, gt_w = gt_depth_m.shape
            pred_h, pred_w = pred_depth.shape

            if pred_depth.shape != gt_depth_m.shape:
                sh = gt_h / pred_h
                sw = gt_w / pred_w
                pred_depth_r = zoom(pred_depth, (sh, sw), order=1)
                pred_normal_r = np.stack([
                    zoom(pred_normal[:,:,c], (sh, sw), order=1) for c in range(3)
                ], axis=-1)
                # Re-normalize
                norm = np.linalg.norm(pred_normal_r, axis=-1, keepdims=True)
                pred_normal_r = pred_normal_r / np.maximum(norm, 1e-8)
                pred_mask_r = zoom(pred_mask.astype(float), (sh, sw), order=0) > 0.5
            else:
                pred_depth_r = pred_depth
                pred_normal_r = pred_normal
                pred_mask_r = pred_mask

            # Scale-aligned prediction
            valid = pred_mask_r & (pred_depth_r > 0) & (gt_depth_m > 0)
            if valid.sum() > 100:
                scale = np.sum(gt_depth_m[valid] * pred_depth_r[valid]) / np.sum(pred_depth_r[valid] ** 2)
            else:
                scale = 1.0
            pred_aligned_m = pred_depth_r * scale

            # Relief for 3D vis (plane removal)
            def to_relief(depth):
                yy, xx = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
                A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
                coeffs, _, _, _ = np.linalg.lstsq(A, depth.ravel(), rcond=None)
                return depth - (coeffs[0] * xx + coeffs[1] * yy + coeffs[2])

            gt_relief = to_relief(gt_depth_m)
            pred_relief = to_relief(pred_aligned_m)

            gt_mean = gt_depth_mm[gt_depth_mm > 0].mean()
            pred_mean = pred_depth_r[pred_depth_r > 0].mean() * 1000

            # ---- Plot columns ----
            # Col 0: RGB
            ax = fig.add_subplot(n, 6, row * 6 + 1)
            ax.imshow(cv_small)
            ax.set_title(f'{cat}\n({ver}, {sid})', fontsize=10, fontweight='bold')
            ax.axis('off')

            # Col 1: GT normal map
            ax = fig.add_subplot(n, 6, row * 6 + 2)
            ax.imshow(normal_to_rgb(gt_normals))
            ax.set_title(f'GT Normals\n(from depth gradient)', fontsize=10)
            ax.axis('off')

            # Col 2: MoGe normal map
            ax = fig.add_subplot(n, 6, row * 6 + 3)
            ax.imshow(normal_to_rgb(pred_normal_r))
            ax.set_title(f'MoGe Normals\n(predicted)', fontsize=10)
            ax.axis('off')

            # Col 3: GT 3D
            ax3d = fig.add_subplot(n, 6, row * 6 + 4, projection='3d')
            step = max(gt_relief.shape[0] // 120, 1)
            Z_gt = gt_relief[::step, ::step]
            rgb_sub = cv_small[::step, ::step].astype(np.float64) / 255.0
            ys = np.arange(Z_gt.shape[0]) * step
            xs = np.arange(Z_gt.shape[1]) * step
            X, Y = np.meshgrid(xs, ys)
            ax3d.plot_surface(X, Y, Z_gt, facecolors=rgb_sub, rstride=1, cstride=1,
                              shade=True, lightsource=matplotlib.colors.LightSource(315, 45),
                              antialiased=True)
            ax3d.view_init(elev=55, azim=-55)
            ax3d.set_box_aspect([1, Z_gt.shape[0] / max(Z_gt.shape[1], 1), 0.25])
            ax3d.set_xticks([]); ax3d.set_yticks([])
            ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
            ax3d.set_title(f'GT 3D\n({gt_mean:.0f}mm)', fontsize=10)

            # Col 4: MoGe 3D (scale-aligned)
            ax3d = fig.add_subplot(n, 6, row * 6 + 5, projection='3d')
            Z_pred = pred_relief[::step, ::step]
            ax3d.plot_surface(X, Y, Z_pred, facecolors=rgb_sub, rstride=1, cstride=1,
                              shade=True, lightsource=matplotlib.colors.LightSource(315, 45),
                              antialiased=True)
            ax3d.view_init(elev=55, azim=-55)
            ax3d.set_box_aspect([1, Z_pred.shape[0] / max(Z_pred.shape[1], 1), 0.25])
            ax3d.set_xticks([]); ax3d.set_yticks([])
            ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
            ax3d.set_title(f'MoGe 3D (aligned)\n(raw: {pred_mean:.0f}mm, ×{scale:.1f})', fontsize=10)

            # Col 5: MoGe depth vs GT depth side-by-side color comparison
            ax = fig.add_subplot(n, 6, row * 6 + 6)
            # Show scale comparison as a bar chart
            ax.barh(['GT depth', 'MoGe raw', 'MoGe aligned'],
                    [gt_mean, pred_mean, pred_mean * scale],
                    color=['#2ecc71', '#e74c3c', '#3498db'])
            ax.set_xlabel('Depth (mm)')
            scale_ratio = pred_mean / gt_mean
            ax.set_title(f'Scale: {scale_ratio:.1f}× overestimate\n'
                         f'({gt_mean:.0f}mm → {pred_mean:.0f}mm)', fontsize=10)

        plt.suptitle(f'SKINL2 {page_name.upper()} — MoGe Normals & 3D vs Ground Truth\n'
                     'RGB | GT Normals | MoGe Normals | GT 3D | MoGe 3D (aligned) | Scale Comparison',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_num = {'v1': 39, 'v2': 40, 'v3': 41}[page_name]
        path = OUTPUT_DIR / f'fig{fig_num}_skinl2_{page_name}_normals_3d.png'
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"Saved {path.name}\n")


if __name__ == "__main__":
    main()
