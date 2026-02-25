#!/usr/bin/env python3
"""Compute wound/lesion 3D surface area from annotations + depth predictions.

Uses polygon annotations from the wound_annotator tool and depth predictions
from each method to compute metric 3D surface area.

For each annotated sample:
1. Load polygon annotation → binary mask
2. For each method's depth map + intrinsics → 3D point cloud
3. Compute surface area within mask: sum of cross-product magnitudes of
   adjacent pixel tangent vectors
4. Compare area (cm^2) across methods and against GT

Usage:
    conda run -n MoGe python code/evaluation/compute_wound_area.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATA = PROJECT_ROOT / "output" / "eval_data"
ANNOTATIONS_FILE = PROJECT_ROOT / "output" / "annotations" / "wound_annotations.json"
CACHE_DIR = PROJECT_ROOT / "output" / "figures" / "depth_comparison" / "_cache"
OUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "wound_area"

# Methods and their depth cache locations
METHODS = {
    'gt': None,  # loaded directly from eval data
    'dermdepth': None,  # loaded via model inference or cache
    'd3': 'depth_comparison',
    'da3nested': 'depth_comparison',
    'mapanything': 'depth_comparison',
    'ppd': 'depth_comparison',
}


def polygon_to_mask(points, height, width):
    """Convert polygon vertices to binary mask using matplotlib."""
    from matplotlib.path import Path as MplPath
    poly_path = MplPath(points)
    y, x = np.mgrid[:height, :width]
    coords = np.column_stack([x.ravel(), y.ravel()])
    mask = poly_path.contains_points(coords).reshape(height, width)
    return mask


def estimate_intrinsics(height, width, fov_deg=60.0):
    """Estimate pinhole intrinsics."""
    fx = fy = width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def compute_surface_area(depth, mask, intrinsics):
    """Compute 3D surface area within mask using depth + intrinsics.

    For each pixel (i,j) in the mask, compute the 3D position:
        X = (j - cx) * d / fx
        Y = (i - cy) * d / fy
        Z = d

    Then compute the surface area element as the cross product of
    tangent vectors from adjacent pixels.

    Returns area in the square of the depth's units (e.g., m^2 if depth is in meters).
    """
    h, w = depth.shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Backproject to 3D
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    X = (jj - cx) * depth / fx
    Y = (ii - cy) * depth / fy
    Z = depth

    # Compute tangent vectors (forward differences)
    # dP/dx (horizontal tangent)
    dXdx = np.zeros_like(X)
    dYdx = np.zeros_like(Y)
    dZdx = np.zeros_like(Z)
    dXdx[:, :-1] = X[:, 1:] - X[:, :-1]
    dYdx[:, :-1] = Y[:, 1:] - Y[:, :-1]
    dZdx[:, :-1] = Z[:, 1:] - Z[:, :-1]

    # dP/dy (vertical tangent)
    dXdy = np.zeros_like(X)
    dYdy = np.zeros_like(Y)
    dZdy = np.zeros_like(Z)
    dXdy[:-1, :] = X[1:, :] - X[:-1, :]
    dYdy[:-1, :] = Y[1:, :] - Y[:-1, :]
    dZdy[:-1, :] = Z[1:, :] - Z[:-1, :]

    # Cross product: dP/dx × dP/dy
    nx = dYdx * dZdy - dZdx * dYdy
    ny = dZdx * dXdy - dXdx * dZdy
    nz = dXdx * dYdy - dYdx * dXdy

    # Area element = magnitude of cross product
    area_element = np.sqrt(nx**2 + ny**2 + nz**2)

    # Valid mask: within annotation AND finite depth AND not at boundary
    valid = mask & np.isfinite(depth) & (depth > 0)
    valid[:-1, :] &= np.isfinite(depth[1:, :])
    valid[:, :-1] &= np.isfinite(depth[:, 1:])

    total_area = float(np.sum(area_element[valid]))
    n_pixels = int(valid.sum())

    return total_area, n_pixels


def load_gt_depth(dataset, sample_name):
    """Load GT depth in meters."""
    sample_dir = EVAL_DATA / dataset / sample_name
    gt = np.load(sample_dir / "gt_depth.npy")
    if dataset == 'skinl2':
        gt = gt * 0.001  # mm → meters
    return gt


def load_method_depth(method, dataset, sample_name):
    """Load cached depth prediction for a method (in meters)."""
    cache_path = CACHE_DIR / dataset / sample_name / f"{method}_depth.npy"
    if cache_path.exists():
        return np.load(cache_path)
    return None


def main():
    # Load annotations
    if not ANNOTATIONS_FILE.exists():
        print(f"No annotations found at {ANNOTATIONS_FILE}")
        print("Run the annotation tool first: python code/annotation/wound_annotator.py")
        return

    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    if not annotations:
        print("No annotations found. Annotate some samples first.")
        return

    print(f"Found {len(annotations)} annotations")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    methods_to_eval = ['gt', 'd3', 'da3nested', 'mapanything', 'ppd']
    results = []

    for key, ann in sorted(annotations.items()):
        dataset = ann['dataset']
        name = ann['name']
        points = ann['points']
        img_h, img_w = ann['height'], ann['width']

        print(f"\n{'='*60}")
        print(f"{dataset}/{name} ({len(points)} vertices)")

        # Create binary mask from polygon
        mask = polygon_to_mask(points, img_h, img_w)
        mask_pixels = int(mask.sum())
        print(f"  Mask: {mask_pixels} pixels ({mask_pixels/(img_h*img_w)*100:.1f}% of image)")

        # Estimate intrinsics (we don't have true intrinsics for eval data)
        intrinsics = estimate_intrinsics(img_h, img_w, fov_deg=60.0)

        sample_result = {'key': key, 'dataset': dataset, 'name': name,
                         'mask_pixels': mask_pixels, 'areas': {}}

        for method in methods_to_eval:
            if method == 'gt':
                depth = load_gt_depth(dataset, name)
            else:
                depth = load_method_depth(method, dataset, name)

            if depth is None:
                print(f"  {method}: no depth available, skipping")
                continue

            # Resize depth to annotation resolution if needed
            if depth.shape[:2] != (img_h, img_w):
                from scipy.ndimage import zoom
                depth = zoom(depth, (img_h / depth.shape[0], img_w / depth.shape[1]), order=1)

            area_m2, n_valid = compute_surface_area(depth, mask, intrinsics)
            area_cm2 = area_m2 * 1e4  # m^2 → cm^2
            area_mm2 = area_m2 * 1e6  # m^2 → mm^2

            sample_result['areas'][method] = {
                'area_m2': area_m2, 'area_cm2': area_cm2, 'area_mm2': area_mm2,
                'n_valid_pixels': n_valid,
            }

            # Choose display unit based on GT magnitude
            if method == 'gt':
                gt_area_cm2 = area_cm2
            print(f"  {method:>12}: {area_cm2:>10.2f} cm^2  ({area_mm2:.0f} mm^2, {n_valid} px)")

        # Compute ratios vs GT
        if 'gt' in sample_result['areas']:
            gt_a = sample_result['areas']['gt']['area_cm2']
            print(f"  {'--- ratios ---':>12}")
            for method in methods_to_eval:
                if method == 'gt' or method not in sample_result['areas']:
                    continue
                pred_a = sample_result['areas'][method]['area_cm2']
                ratio = pred_a / gt_a if gt_a > 0 else float('inf')
                sample_result['areas'][method]['ratio_vs_gt'] = ratio
                print(f"  {method:>12}: {ratio:>10.2f}x  ({'over' if ratio > 1 else 'under'}estimate)")

        results.append(sample_result)

    # Summary table
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE — Wound/Lesion Area Estimation")
        print(f"{'='*80}")
        print(f"{'Sample':<35} {'GT (cm2)':>9}", end="")
        for m in methods_to_eval[1:]:
            print(f" {m+' (cm2)':>15} {'ratio':>7}", end="")
        print()
        print("-" * 120)

        for r in results:
            gt_a = r['areas'].get('gt', {}).get('area_cm2', 0)
            print(f"{r['name'][:35]:<35} {gt_a:>9.2f}", end="")
            for m in methods_to_eval[1:]:
                if m in r['areas']:
                    a = r['areas'][m]['area_cm2']
                    ratio = r['areas'][m].get('ratio_vs_gt', 0)
                    print(f" {a:>15.2f} {ratio:>6.1f}x", end="")
                else:
                    print(f" {'N/A':>15} {'':>7}", end="")
            print()

    # Save results
    out_path = OUT_DIR / "wound_area_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
