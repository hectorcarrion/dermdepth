#!/usr/bin/env python3
"""
Prepare WoundsDB (DB_ALL) for depth evaluation using ToF ground truth.

For each scene:
1. Use photo.png as the input RGB image (320x240, in thermal camera frame)
2. Load depth-mesh.ply as ToF ground-truth (SwissRanger SR4000, 176x144)
3. Warp ToF depth into thermal/photo space using cv2.warpAffine (dense)
4. Output: image + dense GT depth at 320x240

Following the official WoundsDB processing pipeline:
  - The matrix named 'ThermalToTof' maps ToF coords -> thermal coords (forward)
  - cv2.warpAffine treats M as forward mapping (src->dst), inverts internally
  - ToF has LARGER FoV than thermal -- the photo fits inside the ToF FoV
  - Result: ~100% dense depth coverage at photo resolution

PLY vertex conventions:
  - Units are mm (converted to meters for output)
  - Column-major storage: reshape(176, 144).T -> (144, 176)
  - Depth axis varies between cases (auto-detected per scene)

Usage:
    python prepare_woundsdb.py [--output_dir OUTPUT]
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

TOF_W = 176
TOF_H = 144
MIN_VALID_DEPTH_M = 0.5  # Filter threshold for corrupted ToF data
THERMAL_SIZE = (240, 320)  # (height, width)


def load_ply(ply_path):
    """Load PLY point cloud and return vertices."""
    import trimesh
    mesh = trimesh.load(str(ply_path), process=False)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    return vertices


def load_registration(reg_path):
    """Load registration transforms from JSON.

    Returns dict mapping type -> full dict (with 'transformation_matrix' key),
    matching the official wound_db_viewer convention.
    """
    with open(reg_path) as f:
        reg_list = json.load(f)
    return {r['type']: np.array(r['transformation_matrix']) for r in reg_list}


def detect_depth_axis(vertices):
    """Auto-detect which vertex axis (0=x, 1=y, 2=z) contains depth.

    The depth axis is the one where valid (>200mm) values have the largest
    mean, since depth values (700-3000mm) are much larger than lateral offsets.
    """
    best_axis = 2  # default fallback
    best_mean = 0
    for axis in range(3):
        vals = vertices[:, axis]
        positive = vals[vals > 200]
        if len(positive) > len(vals) * 0.1:  # at least 10% valid
            mean_val = positive.mean()
            if mean_val > best_mean:
                best_mean = mean_val
                best_axis = axis
    return best_axis


def extract_tof_depth_image(vertices):
    """Extract (144, 176) depth image from PLY vertices.

    Vertices are column-major: first 144 vertices = column 0.
    Depth axis is auto-detected per scene.
    Returns depth in mm as float32.
    """
    depth_axis = detect_depth_axis(vertices)
    depth_mm = vertices[:, depth_axis].reshape(TOF_W, TOF_H).T.astype(np.float32)
    return depth_mm, depth_axis


def warp_tof_to_thermal(tof_depth, M_tof_to_thermal, output_size=THERMAL_SIZE):
    """Warp ToF depth image to thermal/photo space using cv2.warpAffine.

    Following the official WoundsDB pipeline:
    - M_tof_to_thermal maps ToF coords -> thermal coords (forward mapping)
    - cv2.warpAffine interprets the matrix as forward and inverts internally
    - ToF has larger FoV, so entire thermal image fits inside ToF

    Args:
        tof_depth: (144, 176) depth image in mm
        M_tof_to_thermal: 3x3 affine matrix (ToF -> thermal)
        output_size: (height, width) of output

    Returns:
        warped_depth: (height, width) depth in thermal space (mm)
        valid_mask: (height, width) bool mask of valid pixels
    """
    out_h, out_w = output_size

    # Mark invalid ToF pixels (zero or negative depth)
    invalid_tof = tof_depth <= 0

    # Replace invalid with 0 for warping (will be masked out after)
    tof_clean = tof_depth.copy()
    tof_clean[invalid_tof] = 0

    # Warp depth to thermal space
    affine_2x3 = M_tof_to_thermal[:2, :].astype(np.float64)
    warped_depth = cv2.warpAffine(
        tof_clean, affine_2x3, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Also warp a validity mask to detect where interpolation used invalid pixels
    tof_valid_float = (~invalid_tof).astype(np.float32)
    warped_valid = cv2.warpAffine(
        tof_valid_float, affine_2x3, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Only keep pixels where the validity mask is high (not contaminated by zeros)
    valid_mask = warped_valid > 0.9

    return warped_depth, valid_mask


def prepare_scene(scene_dir, output_dir):
    """Prepare a single WoundsDB scene for evaluation."""
    scene_dir = Path(scene_dir)
    photo_path = scene_dir / "photo.png"
    tof_ply_path = scene_dir / "depth-mesh.ply"
    reg_path = scene_dir / "registration.json"

    result = {
        'scene_dir': str(scene_dir),
        'has_photo': photo_path.exists(),
        'has_tof_ply': tof_ply_path.exists(),
        'has_registration': reg_path.exists(),
    }

    # Need all three files
    if not (photo_path.exists() and tof_ply_path.exists() and reg_path.exists()):
        missing = [f for f, exists in [('photo', photo_path.exists()),
                                        ('depth-mesh.ply', tof_ply_path.exists()),
                                        ('registration.json', reg_path.exists())]
                   if not exists]
        result['status'] = f'skip_missing_{"+".join(missing)}'
        return result

    # Load ToF PLY
    vertices = load_ply(tof_ply_path)
    if vertices is None or len(vertices) != TOF_W * TOF_H:
        result['status'] = f'skip_bad_ply_{len(vertices) if vertices is not None else 0}'
        return result

    # Extract ToF depth image (auto-detect depth axis)
    tof_depth_mm, depth_axis = extract_tof_depth_image(vertices)
    tof_depth_m = tof_depth_mm / 1000.0  # mm -> meters

    # Filter corrupted scenes (mean depth unreasonably small)
    valid_depth = tof_depth_m[tof_depth_m > 0.1]
    if len(valid_depth) == 0 or valid_depth.mean() < MIN_VALID_DEPTH_M:
        result['status'] = (f'skip_corrupted_depth_{valid_depth.mean():.3f}m'
                            if len(valid_depth) > 0 else 'skip_no_valid_depth')
        return result

    # Load registration transforms
    transforms = load_registration(reg_path)
    if 'ThermalToTof' not in transforms:
        result['status'] = 'skip_no_tof_transform'
        return result

    M_tof_to_thermal = transforms['ThermalToTof']

    # Load photo to get dimensions
    photo = Image.open(photo_path)
    photo_w, photo_h = photo.size

    # Warp ToF depth to thermal/photo space (dense)
    warped_depth_mm, valid_mask = warp_tof_to_thermal(
        tof_depth_mm, M_tof_to_thermal, (photo_h, photo_w)
    )
    warped_depth_m = warped_depth_mm / 1000.0

    # Apply validity mask
    gt_depth = np.full((photo_h, photo_w), np.nan, dtype=np.float32)
    gt_depth[valid_mask] = warped_depth_m[valid_mask]

    # Additional filter: remove outlier depth values after warping
    gt_depth[gt_depth < 0.1] = np.nan
    valid_mask = np.isfinite(gt_depth)

    n_gt_pixels = int(valid_mask.sum())
    if n_gt_pixels < 100:
        result['status'] = f'skip_too_few_points_{n_gt_pixels}'
        return result

    coverage_pct = n_gt_pixels / (photo_w * photo_h) * 100

    os.makedirs(output_dir, exist_ok=True)

    # Save photo as input image
    photo.save(os.path.join(output_dir, "image.png"))

    # Save GT depth and mask
    np.save(os.path.join(output_dir, "gt_depth.npy"), gt_depth)
    np.save(os.path.join(output_dir, "gt_mask.npy"), valid_mask)

    # Stats
    gt_valid = gt_depth[valid_mask]

    result.update({
        'status': 'success',
        'image_size': (photo_w, photo_h),
        'depth_axis': int(depth_axis),
        'depth_axis_name': 'xyz'[depth_axis],
        'tof_valid_pixels': int((tof_depth_mm > 0).sum()),
        'gt_pixels': n_gt_pixels,
        'coverage_pct': float(coverage_pct),
        'gt_depth_range_m': {
            'min': float(gt_valid.min()),
            'max': float(gt_valid.max()),
            'mean': float(gt_valid.mean()),
        },
    })

    # Save metadata
    meta = {
        'width': photo_w,
        'height': photo_h,
        'depth_units': 'meters',
        'depth_source': 'depth-mesh.ply (SwissRanger SR4000 ToF, warped via cv2.warpAffine)',
        'is_metric': True,
        'gt_type': 'dense',
        'gt_pixels': n_gt_pixels,
        'coverage_pct': float(coverage_pct),
        'depth_axis': int(depth_axis),
    }
    with open(os.path.join(output_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare WoundsDB for depth evaluation (ToF GT)")
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "eval_data" / "woundsdb"),
                        help='Output directory')
    parser.add_argument('--max_scenes', type=int, default=0,
                        help='Max scenes to process (0 for all)')
    args = parser.parse_args()

    print("=" * 60)
    print("Preparing WoundsDB for Evaluation (ToF GT)")
    print("  Input: photo.png (320x240 color photo)")
    print("  GT: depth-mesh.ply (SwissRanger SR4000)")
    print("  Method: cv2.warpAffine (dense, ~100% coverage)")
    print("=" * 60)

    db_dir = DATA_DIR / "DB_ALL"
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all scenes
    all_scenes = []
    for case_dir in sorted(db_dir.glob("case_*")):
        for day_dir in sorted(case_dir.glob("day_*")):
            results_dir = day_dir / "results"
            if not results_dir.exists():
                continue
            for scene_dir in sorted(results_dir.glob("scene_*")):
                all_scenes.append({
                    'path': str(scene_dir),
                    'case': case_dir.name,
                    'day': day_dir.name,
                    'scene': scene_dir.name,
                })

    if args.max_scenes > 0:
        all_scenes = all_scenes[:args.max_scenes]

    print(f"  Found {len(all_scenes)} scenes")

    results = []
    for i, scene_info in enumerate(all_scenes):
        scene_name = f"{scene_info['case']}_{scene_info['day']}_{scene_info['scene']}"
        out_dir = os.path.join(args.output_dir, scene_name)
        print(f"  [{i+1}/{len(all_scenes)}] {scene_name}...", end="")

        result = prepare_scene(scene_info['path'], out_dir)
        result['scene_name'] = scene_name
        results.append(result)

        if result.get('status') == 'success':
            dr = result['gt_depth_range_m']
            axis_name = result.get('depth_axis_name', '?')
            print(f" {result['gt_pixels']} GT pts ({result['coverage_pct']:.1f}%) "
                  f"depth={dr['mean']:.2f}m (axis={axis_name})")
        else:
            print(f" ({result['status']})")

    # Summary
    success = [r for r in results if r.get('status') == 'success']
    skipped = [r for r in results if r.get('status', '').startswith('skip')]

    depth_means = [r['gt_depth_range_m']['mean'] for r in success]
    gt_pixels_list = [r['gt_pixels'] for r in success]

    summary = {
        'total_scenes': len(all_scenes),
        'prepared': len(success),
        'skipped': len(skipped),
        'skip_reasons': {},
        'image_source': 'photo.png (Fujifilm X-T1, registered to thermal frame)',
        'depth_source': 'depth-mesh.ply (SwissRanger SR4000 ToF)',
        'gt_type': 'dense (warped via cv2.warpAffine)',
    }

    for r in skipped:
        reason = r['status']
        summary['skip_reasons'][reason] = summary['skip_reasons'].get(reason, 0) + 1

    if depth_means:
        summary['depth_stats_m'] = {
            'mean_of_means': float(np.mean(depth_means)),
            'std_of_means': float(np.std(depth_means)),
            'range': [float(np.min(depth_means)), float(np.max(depth_means))],
        }
        summary['gt_coverage'] = {
            'mean_pixels': float(np.mean(gt_pixels_list)),
            'mean_pct': float(np.mean([r['coverage_pct'] for r in success])),
        }

    with open(os.path.join(args.output_dir, "preparation_summary.json"), 'w') as f:
        json.dump({**summary, 'scenes': results}, f, indent=2, default=str)

    print(f"\nPrepared: {len(success)}/{len(all_scenes)} scenes")
    if skipped:
        print(f"Skipped: {len(skipped)} ({summary['skip_reasons']})")
    if depth_means:
        print(f"Mean depth: {np.mean(depth_means):.3f}m +/- {np.std(depth_means):.3f}m")
        print(f"Mean GT pixels: {np.mean(gt_pixels_list):.0f} ({np.mean([r['coverage_pct'] for r in success]):.1f}%)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
