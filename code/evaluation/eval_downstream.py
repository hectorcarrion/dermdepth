#!/usr/bin/env python3
"""
Downstream task evaluation for DermDepth.

Evaluates depth predictions on clinically relevant tasks:
1. Wound volume estimation (WoundsDB: predicted depth vs PLY GT)
2. Lesion area estimation (3D surface area from depth + mask)
3. Normal map quality (mean angular error vs GT)

Usage:
    python eval_downstream.py --model MODEL --dataset woundsdb --data_dir PREPARED_DATA
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))


def load_model(model_path, device="cuda"):
    """Load MoGe-2 model."""
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model_config = checkpoint.get('model_config', None)
        if model_config:
            model = MoGeModel(**model_config)
        else:
            model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model = MoGeModel.from_pretrained(model_path)

    return model.to(device).eval()


def run_inference(model, image_path, device="cuda"):
    """Run inference and return all outputs."""
    import torchvision.transforms.functional as TF

    img = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model.infer(img_tensor)

    result = {}
    for key in ['depth', 'points', 'mask', 'intrinsics', 'normal']:
        if key in output:
            val = output[key]
            if isinstance(val, torch.Tensor):
                result[key] = val.cpu().numpy()
                if result[key].ndim > 2 and result[key].shape[0] == 1:
                    result[key] = result[key].squeeze(0)

    return result


def estimate_wound_volume(depth_map, mask=None, reference_plane='border'):
    """
    Estimate wound volume from a depth map.

    Volume = sum of (reference_depth - wound_depth) * pixel_area
    where the reference plane is the surrounding skin surface.

    Args:
        depth_map: HxW depth array
        mask: Optional wound mask (HxW bool)
        reference_plane: 'border' (use wound border depth) or 'median' (use median depth)

    Returns:
        volume: Estimated volume in cubic model units
        area: Surface area
        mean_depth_deviation: Mean depth below reference plane
    """
    valid = np.isfinite(depth_map) & (depth_map > 0)
    if mask is not None:
        wound_mask = mask & valid
        surrounding_mask = ~mask & valid
    else:
        # Use center 50% as wound, border as surrounding
        h, w = depth_map.shape
        wound_mask = np.zeros_like(valid)
        wound_mask[h//4:3*h//4, w//4:3*w//4] = True
        wound_mask &= valid
        surrounding_mask = valid & ~wound_mask

    if wound_mask.sum() < 10 or surrounding_mask.sum() < 10:
        return None, None, None

    # Reference plane from surrounding skin
    if reference_plane == 'border':
        ref_depth = np.median(depth_map[surrounding_mask])
    else:
        ref_depth = np.median(depth_map[valid])

    # Volume: positive means wound is deeper (further from camera)
    wound_depths = depth_map[wound_mask]
    depth_deviations = wound_depths - ref_depth
    # Positive deviation = wound goes deeper (further from camera)

    # Approximate pixel area (assume unit pixels for now)
    pixel_area = 1.0  # Would need intrinsics for true area

    volume = float(np.abs(depth_deviations).sum() * pixel_area)
    area = float(wound_mask.sum() * pixel_area)
    mean_deviation = float(np.mean(depth_deviations))

    return volume, area, mean_deviation


def compute_surface_area_3d(points, mask=None):
    """
    Compute 3D surface area from a point map using triangle mesh approximation.

    Each pixel quad is split into two triangles, and triangle areas are summed.
    """
    if mask is None:
        mask = np.isfinite(points).all(axis=-1)

    h, w = points.shape[:2]

    # For each 2x2 pixel quad, form two triangles
    # Triangle 1: (i,j), (i+1,j), (i,j+1)
    # Triangle 2: (i+1,j), (i+1,j+1), (i,j+1)

    p00 = points[:-1, :-1, :]  # (H-1, W-1, 3)
    p10 = points[1:, :-1, :]
    p01 = points[:-1, 1:, :]
    p11 = points[1:, 1:, :]

    m00 = mask[:-1, :-1]
    m10 = mask[1:, :-1]
    m01 = mask[:-1, 1:]
    m11 = mask[1:, 1:]

    # Triangle 1: p00, p10, p01
    valid_t1 = m00 & m10 & m01
    v1 = p10 - p00
    v2 = p01 - p00
    cross1 = np.cross(v1, v2)
    area_t1 = 0.5 * np.linalg.norm(cross1, axis=-1)
    area_t1 = np.where(valid_t1, area_t1, 0)

    # Triangle 2: p10, p11, p01
    valid_t2 = m10 & m11 & m01
    v3 = p11 - p10
    v4 = p01 - p10
    cross2 = np.cross(v3, v4)
    area_t2 = 0.5 * np.linalg.norm(cross2, axis=-1)
    area_t2 = np.where(valid_t2, area_t2, 0)

    total_area = float(area_t1.sum() + area_t2.sum())
    return total_area


def compute_normal_angular_error(pred_normal, gt_normal, mask=None):
    """
    Compute mean angular error between predicted and GT normal maps.

    Args:
        pred_normal: HxWx3 predicted normals
        gt_normal: HxWx3 ground truth normals
        mask: HxW valid mask

    Returns:
        mean_angle_deg: Mean angular error in degrees
        median_angle_deg: Median angular error in degrees
    """
    if mask is None:
        mask = np.isfinite(pred_normal).all(axis=-1) & np.isfinite(gt_normal).all(axis=-1)

    # Normalize
    pred_norm = pred_normal / (np.linalg.norm(pred_normal, axis=-1, keepdims=True) + 1e-8)
    gt_norm = gt_normal / (np.linalg.norm(gt_normal, axis=-1, keepdims=True) + 1e-8)

    # Dot product
    dots = np.sum(pred_norm * gt_norm, axis=-1)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)
    angles_deg = np.degrees(angles)

    valid_angles = angles_deg[mask]
    if len(valid_angles) == 0:
        return None, None

    return float(np.mean(valid_angles)), float(np.median(valid_angles))


def eval_wound_volume(model, data_dir, output_dir, device="cuda"):
    """Evaluate wound volume estimation on WoundsDB."""
    print("\n" + "=" * 60)
    print("Evaluating Wound Volume Estimation")
    print("=" * 60)

    save_dir = Path(output_dir) / "wound_volume"
    save_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
    results = []

    for i, scene_dir in enumerate(scene_dirs):
        scene_name = scene_dir.name
        image_path = scene_dir / "image.png"
        gt_depth_path = scene_dir / "gt_depth.npy"

        if not image_path.exists():
            continue

        print(f"  [{i+1}/{len(scene_dirs)}] {scene_name}...", end="")

        try:
            output = run_inference(model, str(image_path), device)
            pred_depth = output['depth']
            if isinstance(pred_depth, torch.Tensor):
                pred_depth = pred_depth.numpy()

            pred_volume, pred_area, pred_deviation = estimate_wound_volume(pred_depth)

            result = {
                'scene': scene_name,
                'pred_volume': pred_volume,
                'pred_area': pred_area,
                'pred_mean_depth_deviation': pred_deviation,
            }

            # Compare with GT if available
            if gt_depth_path.exists():
                gt_depth = np.load(gt_depth_path)
                gt_volume, gt_area, gt_deviation = estimate_wound_volume(gt_depth)
                result['gt_volume'] = gt_volume
                result['gt_area'] = gt_area

                if gt_volume and pred_volume:
                    result['volume_ratio'] = float(pred_volume / gt_volume) if gt_volume > 0 else None
                    result['area_ratio'] = float(pred_area / gt_area) if gt_area and gt_area > 0 else None

            results.append(result)

            if pred_volume:
                print(f" vol={pred_volume:.2f}")
            else:
                print(f" (invalid)")

        except Exception as e:
            print(f" Error: {e}")

    # Summary
    if results:
        valid_results = [r for r in results if r.get('volume_ratio') is not None]
        if valid_results:
            volume_ratios = [r['volume_ratio'] for r in valid_results]
            summary = {
                'num_evaluated': len(valid_results),
                'volume_ratio_mean': float(np.mean(volume_ratios)),
                'volume_ratio_std': float(np.std(volume_ratios)),
                'volume_ratio_median': float(np.median(volume_ratios)),
            }
        else:
            summary = {'num_evaluated': len(results), 'note': 'No GT volumes for comparison'}
    else:
        summary = {}

    output_data = {'summary': summary, 'per_scene': results}
    with open(save_dir / "results.json", 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    return output_data


def eval_surface_area(model, data_dir, output_dir, device="cuda"):
    """Evaluate 3D lesion surface area estimation."""
    print("\n" + "=" * 60)
    print("Evaluating 3D Surface Area Estimation")
    print("=" * 60)

    save_dir = Path(output_dir) / "surface_area"
    save_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
    results = []

    for i, scene_dir in enumerate(scene_dirs):
        scene_name = scene_dir.name
        image_path = scene_dir / "image.png"

        if not image_path.exists():
            continue

        print(f"  [{i+1}/{len(scene_dirs)}] {scene_name}...", end="")

        try:
            output = run_inference(model, str(image_path), device)

            if 'points' in output:
                points = output['points']
                if isinstance(points, torch.Tensor):
                    points = points.numpy()

                mask = None
                if 'mask' in output:
                    m = output['mask']
                    if isinstance(m, torch.Tensor):
                        m = m.numpy()
                    mask = m > 0.5 if m.dtype != bool else m

                area = compute_surface_area_3d(points, mask)
                result = {'scene': scene_name, 'surface_area_3d': area}
                results.append(result)
                print(f" area={area:.4f}")
            else:
                print(f" (no points)")

        except Exception as e:
            print(f" Error: {e}")

    output_data = {'per_scene': results}
    with open(save_dir / "results.json", 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    return output_data


def eval_normals(model, data_dir, output_dir, device="cuda"):
    """Evaluate normal map quality."""
    print("\n" + "=" * 60)
    print("Evaluating Normal Map Quality")
    print("=" * 60)

    save_dir = Path(output_dir) / "normals"
    save_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
    results = []

    for i, scene_dir in enumerate(scene_dirs):
        scene_name = scene_dir.name
        image_path = scene_dir / "image.png"
        gt_depth_path = scene_dir / "gt_depth.npy"

        if not (image_path.exists() and gt_depth_path.exists()):
            continue

        print(f"  [{i+1}/{len(scene_dirs)}] {scene_name}...", end="")

        try:
            output = run_inference(model, str(image_path), device)

            if 'normal' not in output:
                print(f" (no normals)")
                continue

            pred_normal = output['normal']
            if isinstance(pred_normal, torch.Tensor):
                pred_normal = pred_normal.numpy()

            # Derive GT normals from GT depth using finite differences
            gt_depth = np.load(gt_depth_path)

            # Simple normal estimation from depth gradient
            dz_dx = np.gradient(gt_depth, axis=1)
            dz_dy = np.gradient(gt_depth, axis=0)
            gt_normal = np.stack([-dz_dx, -dz_dy, np.ones_like(gt_depth)], axis=-1)
            gt_normal_len = np.linalg.norm(gt_normal, axis=-1, keepdims=True)
            gt_normal = gt_normal / (gt_normal_len + 1e-8)

            # Resize if needed
            if pred_normal.shape[:2] != gt_normal.shape[:2]:
                from scipy.ndimage import zoom
                scale = (gt_normal.shape[0] / pred_normal.shape[0],
                         gt_normal.shape[1] / pred_normal.shape[1], 1)
                pred_normal = zoom(pred_normal, scale, order=1)

            mean_err, median_err = compute_normal_angular_error(pred_normal, gt_normal)

            result = {
                'scene': scene_name,
                'mean_angular_error_deg': mean_err,
                'median_angular_error_deg': median_err,
            }
            results.append(result)

            if mean_err:
                print(f" MAE={mean_err:.2f} deg")
            else:
                print(f" (invalid)")

        except Exception as e:
            print(f" Error: {e}")

    # Summary
    if results:
        valid = [r for r in results if r.get('mean_angular_error_deg') is not None]
        if valid:
            maes = [r['mean_angular_error_deg'] for r in valid]
            summary = {
                'num_evaluated': len(valid),
                'mean_angular_error_deg': float(np.mean(maes)),
                'median_angular_error_deg': float(np.median(maes)),
            }
        else:
            summary = {}
    else:
        summary = {}

    output_data = {'summary': summary, 'per_scene': results}
    with open(save_dir / "results.json", 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Downstream task evaluation")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='woundsdb',
                        choices=['woundsdb', 'skinl2'])
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "evaluation"))
    parser.add_argument('--tasks', nargs='+',
                        default=['volume', 'area', 'normals'],
                        choices=['volume', 'area', 'normals'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='model')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.model_name, "downstream")
    os.makedirs(output_dir, exist_ok=True)

    default_data_dirs = {
        'woundsdb': str(PROJECT_ROOT / "output" / "eval_data" / "woundsdb"),
        'skinl2': str(PROJECT_ROOT / "output" / "eval_data" / "skinl2"),
    }
    data_dir = args.data_dir or default_data_dirs[args.dataset]

    model = load_model(args.model, args.device)

    if 'volume' in args.tasks:
        eval_wound_volume(model, data_dir, output_dir, args.device)
    if 'area' in args.tasks:
        eval_surface_area(model, data_dir, output_dir, args.device)
    if 'normals' in args.tasks:
        eval_normals(model, data_dir, output_dir, args.device)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
