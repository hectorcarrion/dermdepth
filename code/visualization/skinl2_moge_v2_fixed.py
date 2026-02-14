#!/usr/bin/env python3
"""
Re-run MoGe on SKINL2 v2 cases with 3% edge crop to remove black borders.
Produces: metrics table, normals + 3D comparison figure.
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

CROP_PCT = 0.03  # 3% crop from each edge for v2


def compute_normals_from_depth(depth):
    dz_dy, dz_dx = np.gradient(depth)
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(norm, 1e-8)


def normal_to_rgb(normals):
    return np.clip((normals + 1.0) / 2.0, 0, 1)


def compute_metrics(pred, gt, mask):
    p = pred[mask]
    g = gt[mask]
    valid = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
    p, g = p[valid], g[valid]
    if len(p) < 10:
        return {'valid_pixels': len(p)}

    absrel = float(np.mean(np.abs(p - g) / g))
    scale_ratio = float(np.median(p / g))
    scale = np.sum(g * p) / np.sum(p * p)
    p_aligned = p * scale
    si_absrel = float(np.mean(np.abs(p_aligned - g) / g))
    ratio_si = np.maximum(p_aligned / g, g / p_aligned)
    si_delta1 = float(np.mean(ratio_si < 1.25))

    return {
        'valid_pixels': int(len(p)),
        'absrel': absrel,
        'scale_ratio': scale_ratio,
        'scale_error_pct': abs(scale_ratio - 1.0) * 100,
        'pred_mean_mm': float(np.mean(p)) * 1000,
        'gt_mean_mm': float(np.mean(g)) * 1000,
        'optimal_scale': float(scale),
        'si_absrel': si_absrel,
        'si_delta1': si_delta1,
    }


def discover_v2():
    cases = []
    root = DATA_ROOT / 'SKINL2_v2'
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
                    cases.append(('v2', cat, case_id, cv_files[0], dm_files[0]))
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


def crop_black_border(img, pct):
    """Crop pct fraction from each edge."""
    h, w = img.shape[:2]
    t = int(h * pct)
    b = h - int(h * pct)
    l = int(w * pct)
    r = w - int(w * pct)
    return img[t:b, l:r], (t, b, l, r)


def main():
    device = 'cpu'

    v2_cases = discover_v2()
    selected = select_diverse(v2_cases, 5)

    print(f"Selected {len(selected)} v2 cases:")
    for ver, cat, sid, _, _ in selected:
        print(f"  {ver}/{cat}/{sid}")

    # Load model
    print(f"\nLoading MoGe-2 on {device}...")
    from moge.model.v2 import MoGeModel
    import torchvision.transforms.functional as TF
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    model = model.to(device).eval()
    print("Model loaded.\n")

    n = len(selected)
    fig = plt.figure(figsize=(36, 7 * n))

    all_metrics = []

    for row, (ver, cat, sid, cv_path, dm_path) in enumerate(selected):
        label = f"{ver}/{cat}/{sid}"
        print(f"Processing {label}...")

        # Load full image and depth
        cv_img_full = np.array(Image.open(cv_path).convert('RGB'))
        depth_raw_full = np.array(Image.open(dm_path), dtype=np.float32)

        # Crop black borders from image
        cv_img_cropped, (t, b, l, r) = crop_black_border(cv_img_full, CROP_PCT)
        h_full, w_full = cv_img_full.shape[:2]
        print(f"  Cropped image: {cv_img_full.shape} -> {cv_img_cropped.shape} "
              f"(removed {t}px top, {h_full-b}px bot, {l}px left, {w_full-r}px right)")

        # Crop depth to matching region (depth is half-res of image)
        dh, dw = depth_raw_full.shape
        scale_h = dh / h_full
        scale_w = dw / w_full
        dt = int(t * scale_h)
        db = int(b * scale_h)
        dl = int(l * scale_w)
        dr = int(r * scale_w)
        depth_raw_cropped = depth_raw_full[dt:db, dl:dr]

        # GT depth: smoothed, absolute, meters
        gt_depth_mm = np.abs(gaussian_filter(depth_raw_cropped, sigma=15))
        gt_depth_m = gt_depth_mm / 1000.0
        gt_mask = (gt_depth_m > 0.05) & (gt_depth_m < 0.5)

        # Resize image to depth resolution
        cv_small = np.array(Image.fromarray(cv_img_cropped).resize(
            (depth_raw_cropped.shape[1], depth_raw_cropped.shape[0]), Image.LANCZOS))

        # GT normals
        gt_normals = compute_normals_from_depth(gt_depth_m)

        # MoGe inference on CROPPED image
        img_pil_cropped = Image.fromarray(cv_img_cropped)
        img_tensor = TF.to_tensor(img_pil_cropped).unsqueeze(0).to(device)
        print(f"  MoGe input: {img_tensor.shape}")

        with torch.inference_mode():
            output = model.infer(img_tensor)

        pred_depth = output['depth'].squeeze(0).cpu().numpy()
        pred_normal = output['normal'].squeeze(0).cpu().numpy()
        pred_mask = output['mask'].squeeze(0).cpu().numpy()

        pred_valid = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
        print(f"  MoGe prediction: {pred_depth.shape}, "
              f"range=[{pred_valid.min():.3f}, {pred_valid.max():.3f}]m, "
              f"mean={pred_valid.mean():.3f}m")

        # Resize predictions to GT resolution
        gt_h, gt_w = gt_depth_m.shape
        if pred_depth.shape != (gt_h, gt_w):
            sh = gt_h / pred_depth.shape[0]
            sw = gt_w / pred_depth.shape[1]
            pred_depth_r = zoom(pred_depth, (sh, sw), order=1)
            pred_normal_r = np.stack([
                zoom(pred_normal[:, :, c], (sh, sw), order=1) for c in range(3)
            ], axis=-1)
            norm = np.linalg.norm(pred_normal_r, axis=-1, keepdims=True)
            pred_normal_r = pred_normal_r / np.maximum(norm, 1e-8)
            pred_mask_r = zoom(pred_mask.astype(float), (sh, sw), order=0) > 0.5
        else:
            pred_depth_r = pred_depth
            pred_normal_r = pred_normal
            pred_mask_r = pred_mask

        # Metrics
        metrics = compute_metrics(pred_depth_r, gt_depth_m, gt_mask)
        metrics['version'] = ver
        metrics['category'] = cat
        metrics['sample_id'] = sid
        all_metrics.append(metrics)

        if 'absrel' in metrics:
            print(f"  Scale Ratio: {metrics['scale_ratio']:.2f} "
                  f"(pred {metrics['pred_mean_mm']:.0f}mm vs GT {metrics['gt_mean_mm']:.0f}mm)")
            print(f"  Scale Error: {metrics['scale_error_pct']:.0f}%")
            print(f"  SI-Delta1: {metrics['si_delta1']:.3f}")

        # Scale-aligned depth
        valid = pred_mask_r & (pred_depth_r > 0) & (gt_depth_m > 0)
        scale = np.sum(gt_depth_m[valid] * pred_depth_r[valid]) / np.sum(pred_depth_r[valid] ** 2) if valid.sum() > 100 else 1.0
        pred_aligned_m = pred_depth_r * scale

        # Relief for 3D
        def to_relief(depth):
            yy, xx = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
            A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
            coeffs, _, _, _ = np.linalg.lstsq(A, depth.ravel(), rcond=None)
            return depth - (coeffs[0] * xx + coeffs[1] * yy + coeffs[2])

        gt_relief = to_relief(gt_depth_m)
        pred_relief = to_relief(pred_aligned_m)

        gt_mean = gt_depth_mm[gt_depth_mm > 0].mean()
        pred_mean = pred_depth_r[pred_depth_r > 0].mean() * 1000

        # ---- Plot ----
        # Col 0: RGB (cropped)
        ax = fig.add_subplot(n, 6, row * 6 + 1)
        ax.imshow(cv_small)
        ax.set_title(f'{cat}\n({ver}, {sid}) [3% cropped]', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Col 1: GT normals
        ax = fig.add_subplot(n, 6, row * 6 + 2)
        ax.imshow(normal_to_rgb(gt_normals))
        ax.set_title('GT Normals', fontsize=10)
        ax.axis('off')

        # Col 2: MoGe normals
        ax = fig.add_subplot(n, 6, row * 6 + 3)
        ax.imshow(normal_to_rgb(pred_normal_r))
        ax.set_title('MoGe Normals', fontsize=10)
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
                          shade=True, lightsource=matplotlib.colors.LightSource(315, 45))
        ax3d.view_init(elev=55, azim=-55)
        ax3d.set_box_aspect([1, Z_gt.shape[0] / max(Z_gt.shape[1], 1), 0.25])
        ax3d.set_xticks([]); ax3d.set_yticks([])
        ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
        ax3d.set_title(f'GT 3D\n({gt_mean:.0f}mm)', fontsize=10)

        # Col 4: MoGe 3D (aligned)
        ax3d = fig.add_subplot(n, 6, row * 6 + 5, projection='3d')
        Z_pred = pred_relief[::step, ::step]
        ax3d.plot_surface(X, Y, Z_pred, facecolors=rgb_sub, rstride=1, cstride=1,
                          shade=True, lightsource=matplotlib.colors.LightSource(315, 45))
        ax3d.view_init(elev=55, azim=-55)
        ax3d.set_box_aspect([1, Z_pred.shape[0] / max(Z_pred.shape[1], 1), 0.25])
        ax3d.set_xticks([]); ax3d.set_yticks([])
        ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
        ax3d.set_title(f'MoGe 3D (aligned)\n(raw: {pred_mean:.0f}mm, ×{scale:.1f})', fontsize=10)

        # Col 5: Scale bar
        ax = fig.add_subplot(n, 6, row * 6 + 6)
        ax.barh(['GT', 'MoGe raw', 'MoGe aligned'],
                [gt_mean, pred_mean, pred_mean * scale],
                color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_xlabel('Depth (mm)')
        scale_ratio = pred_mean / gt_mean
        ax.set_title(f'Scale: {scale_ratio:.1f}× overestimate\n'
                     f'({gt_mean:.0f}mm → {pred_mean:.0f}mm)', fontsize=10)

    plt.suptitle('SKINL2 V2 — MoGe Normals & 3D vs GT (3% border crop applied)\n'
                 'RGB | GT Normals | MoGe Normals | GT 3D | MoGe 3D (aligned) | Scale',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig42_skinl2_v2_normals_3d_fixed.png'
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {path.name}")

    # Print metrics comparison
    print(f"\n{'='*70}")
    print("V2 Metrics with 3% border crop:")
    print(f"{'='*70}")
    print(f"{'Category':<22} {'ID':<6} {'GT(mm)':<8} {'Pred(mm)':<9} {'Scale':<7} {'ScaleErr':<9} {'SI-d1':<7}")
    print("-" * 70)
    for m in all_metrics:
        if 'absrel' in m:
            print(f"{m['category']:<22} {m['sample_id']:<6} "
                  f"{m['gt_mean_mm']:<8.0f} {m['pred_mean_mm']:<9.0f} "
                  f"{m['scale_ratio']:<7.1f} {m['scale_error_pct']:<9.0f}% {m['si_delta1']:<7.3f}")


if __name__ == "__main__":
    main()
