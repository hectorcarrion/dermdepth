#!/usr/bin/env python3
"""
Fix GT normal map computation for SKINL2 by using physical pixel spacing.
Show before/after comparison for the same 15 cases.
"""
import os
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "MoGe"))

DATA_ROOT = PROJECT_ROOT / "data" / "SKINL2"
OUTPUT_DIR = PROJECT_ROOT / "output" / "verification"

random.seed(42)

# Raytrix R42 setup: camera at ~197mm from skin, ~40mm FoV width
# Image res varies but depth map is ~1341x1929
# Approximate: 40mm / 1929px ≈ 0.021mm/px
APPROX_FOV_MM = 40.0  # approximate field of view width in mm


def compute_normals_physical(depth_mm, fov_mm=APPROX_FOV_MM):
    """Compute normals with proper physical pixel spacing."""
    h, w = depth_mm.shape
    pixel_spacing = fov_mm / w  # mm per pixel

    dz_dy, dz_dx = np.gradient(depth_mm)
    # Convert pixel gradients to physical gradients (unitless slope)
    dz_dx_phys = dz_dx / pixel_spacing
    dz_dy_phys = dz_dy / pixel_spacing

    normals = np.stack([-dz_dx_phys, -dz_dy_phys, np.ones_like(depth_mm)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(norm, 1e-8)


def compute_normals_naive(depth_mm):
    """Original naive normals (z=1 per pixel, no spacing correction)."""
    dz_dy, dz_dx = np.gradient(depth_mm)
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth_mm)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(norm, 1e-8)


def normal_to_rgb(normals):
    return np.clip((normals + 1.0) / 2.0, 0, 1)


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

    v1_cases = discover_v1()
    v2_cases = discover_v2v3('v2')
    v3_cases = discover_v2v3('v3')
    all_selected = select_diverse(v1_cases, 5) + select_diverse(v2_cases, 5) + select_diverse(v3_cases, 5)

    print(f"Selected {len(all_selected)} cases")

    # Load MoGe
    print("Loading MoGe-2...")
    from moge.model.v2 import MoGeModel
    import torchvision.transforms.functional as TF
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    print("Model loaded.\n")

    # One page per version
    pages = [('v1', all_selected[:5]), ('v2', all_selected[5:10]), ('v3', all_selected[10:15])]

    for page_name, page_cases in pages:
        n = len(page_cases)
        # 5 cols: RGB | GT normals (fixed) | MoGe normals | GT 3D | MoGe 3D
        fig = plt.figure(figsize=(30, 6 * n))

        for row, (ver, cat, sid, cv_path, dm_path) in enumerate(page_cases):
            print(f"Processing {ver}/{cat}/{sid}...")

            cv_img = np.array(Image.open(cv_path).convert('RGB'))
            depth_raw = np.array(Image.open(dm_path), dtype=np.float32)

            # For v2: crop black borders
            if ver == 'v2':
                pct = 0.03
                h, w = cv_img.shape[:2]
                t, b, l, r = int(h*pct), h-int(h*pct), int(w*pct), w-int(w*pct)
                cv_img = cv_img[t:b, l:r]
                dh, dw = depth_raw.shape
                dt, db, dl, dr = int(t*dh/h), int(b*dh/h), int(l*dw/w), int(r*dw/w)
                depth_raw = depth_raw[dt:db, dl:dr]

            gt_depth_mm = np.abs(gaussian_filter(depth_raw, sigma=15))
            cv_small = np.array(Image.fromarray(cv_img).resize(
                (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

            # GT normals — physical spacing
            gt_normals = compute_normals_physical(gt_depth_mm)

            # MoGe inference
            img_tensor = TF.to_tensor(Image.fromarray(cv_img)).unsqueeze(0).to(device)
            with torch.inference_mode():
                output = model.infer(img_tensor)

            pred_normal = output['normal'].squeeze(0).cpu().numpy()
            pred_depth = output['depth'].squeeze(0).cpu().numpy()
            pred_mask = output['mask'].squeeze(0).cpu().numpy()

            # Resize to GT res
            gt_h, gt_w = gt_depth_mm.shape
            if pred_normal.shape[:2] != (gt_h, gt_w):
                sh = gt_h / pred_normal.shape[0]
                sw = gt_w / pred_normal.shape[1]
                pred_normal_r = np.stack([
                    zoom(pred_normal[:,:,c], (sh, sw), order=1) for c in range(3)
                ], axis=-1)
                norm = np.linalg.norm(pred_normal_r, axis=-1, keepdims=True)
                pred_normal_r = pred_normal_r / np.maximum(norm, 1e-8)
                pred_depth_r = zoom(pred_depth, (sh, sw), order=1)
            else:
                pred_normal_r = pred_normal
                pred_depth_r = pred_depth

            # Scale-align for 3D
            gt_m = gt_depth_mm / 1000.0
            valid = (pred_depth_r > 0) & (gt_m > 0.05)
            scale = np.sum(gt_m[valid] * pred_depth_r[valid]) / np.sum(pred_depth_r[valid]**2) if valid.sum() > 100 else 1.0

            def to_relief(d):
                yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
                A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
                c, _, _, _ = np.linalg.lstsq(A, d.ravel(), rcond=None)
                return d - (c[0]*xx + c[1]*yy + c[2])

            gt_relief = to_relief(gt_m)
            pred_relief = to_relief(pred_depth_r * scale)

            # ---- Plot ----
            ax = fig.add_subplot(n, 5, row*5 + 1)
            ax.imshow(cv_small)
            ax.set_title(f'{cat}\n({ver}, {sid})', fontsize=10, fontweight='bold')
            ax.axis('off')

            ax = fig.add_subplot(n, 5, row*5 + 2)
            ax.imshow(normal_to_rgb(gt_normals))
            ax.set_title('GT Normals\n(physical spacing)', fontsize=10)
            ax.axis('off')

            ax = fig.add_subplot(n, 5, row*5 + 3)
            ax.imshow(normal_to_rgb(pred_normal_r))
            ax.set_title('MoGe Normals', fontsize=10)
            ax.axis('off')

            # GT 3D
            ax3d = fig.add_subplot(n, 5, row*5 + 4, projection='3d')
            step = max(gt_relief.shape[0] // 120, 1)
            Z_gt = gt_relief[::step, ::step]
            rgb_sub = cv_small[::step, ::step].astype(np.float64) / 255.0
            ys = np.arange(Z_gt.shape[0]) * step
            xs = np.arange(Z_gt.shape[1]) * step
            X, Y = np.meshgrid(xs, ys)
            import matplotlib.colors as mcolors
            ax3d.plot_surface(X, Y, Z_gt, facecolors=rgb_sub, rstride=1, cstride=1,
                              shade=True, lightsource=mcolors.LightSource(315, 45))
            ax3d.view_init(elev=55, azim=-55)
            ax3d.set_box_aspect([1, Z_gt.shape[0]/max(Z_gt.shape[1],1), 0.25])
            ax3d.set_xticks([]); ax3d.set_yticks([])
            ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
            ax3d.set_title('GT 3D', fontsize=10)

            # MoGe 3D (aligned)
            ax3d = fig.add_subplot(n, 5, row*5 + 5, projection='3d')
            Z_pred = pred_relief[::step, ::step]
            ax3d.plot_surface(X, Y, Z_pred, facecolors=rgb_sub, rstride=1, cstride=1,
                              shade=True, lightsource=mcolors.LightSource(315, 45))
            ax3d.view_init(elev=55, azim=-55)
            ax3d.set_box_aspect([1, Z_pred.shape[0]/max(Z_pred.shape[1],1), 0.25])
            ax3d.set_xticks([]); ax3d.set_yticks([])
            ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
            ax3d.set_title('MoGe 3D (aligned)', fontsize=10)

        plt.suptitle(f'SKINL2 {page_name.upper()} — GT Normals (Fixed) vs MoGe\n'
                     'RGB | GT Normals (physical spacing) | MoGe Normals | GT 3D | MoGe 3D',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_num = {'v1': 43, 'v2': 44, 'v3': 45}[page_name]
        path = OUTPUT_DIR / f'fig{fig_num}_skinl2_{page_name}_normals_fixed.png'
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"Saved {path.name}\n")


if __name__ == "__main__":
    main()
