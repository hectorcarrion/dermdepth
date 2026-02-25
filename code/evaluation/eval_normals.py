#!/usr/bin/env python3
"""
Normal map evaluation for DermDepth.

Evaluates surface normal quality by comparing model-predicted normals against
GT normals derived from GT depth maps using utils3d.np.depth_map_to_normal_map()
(same function used in MoGe-2 training).

Metrics:
- Mean Angular Error (MAE) in degrees
- Median Angular Error
- % within 11.25 / 22.5 / 30 degrees

Usage:
    python eval_normals.py --model MODEL_PATH --dataset all --model_name exp_d1
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

import utils3d


def load_model(model_path, device="cuda"):
    """Load MoGe-2 model from checkpoint or HuggingFace."""
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

    model = model.to(device).eval()
    return model


def run_inference(model, image_path, device="cuda"):
    """Run model inference, returning depth and normal maps."""
    import torchvision.transforms.functional as TF

    img = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model.infer(img_tensor)

    result = {}
    for key in ['depth', 'normal', 'intrinsics', 'mask']:
        if key in output:
            val = output[key]
            if isinstance(val, torch.Tensor):
                result[key] = val.cpu().numpy()
                if result[key].ndim > 2 and result[key].shape[0] == 1:
                    result[key] = result[key].squeeze(0)
            else:
                result[key] = val

    return result


def estimate_intrinsics(height, width, fov_deg=60.0):
    """Estimate pinhole intrinsics from image dimensions and assumed FoV."""
    fx = fy = width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def derive_gt_normal(gt_depth, intrinsics, mask=None):
    """Derive GT normal map from depth using the same method as MoGe-2 training."""
    if mask is None:
        mask = np.isfinite(gt_depth) & (gt_depth > 0)
    normal, normal_mask = utils3d.np.depth_map_to_normal_map(
        gt_depth, intrinsics=intrinsics, mask=mask, edge_threshold=88
    )
    normal = np.where(normal_mask[..., None], normal, np.nan)
    return normal, normal_mask


def compute_normal_metrics(pred_normal, gt_normal, mask=None):
    """
    Compute angular error metrics between predicted and GT normals.

    Returns dict with MAE, median AE, and % within thresholds.
    """
    if mask is None:
        mask = np.all(np.isfinite(gt_normal), axis=-1) & np.all(np.isfinite(pred_normal), axis=-1)
    else:
        mask = mask & np.all(np.isfinite(gt_normal), axis=-1) & np.all(np.isfinite(pred_normal), axis=-1)

    if mask.sum() < 10:
        return {'valid_pixels': 0}

    pred_n = pred_normal[mask]  # (N, 3)
    gt_n = gt_normal[mask]      # (N, 3)

    # Normalize (in case they aren't unit vectors)
    pred_n = pred_n / (np.linalg.norm(pred_n, axis=-1, keepdims=True) + 1e-8)
    gt_n = gt_n / (np.linalg.norm(gt_n, axis=-1, keepdims=True) + 1e-8)

    # Angular error in degrees
    cos_sim = np.clip(np.sum(pred_n * gt_n, axis=-1), -1.0, 1.0)
    angular_error = np.degrees(np.arccos(cos_sim))

    return {
        'valid_pixels': int(mask.sum()),
        'mae': float(np.mean(angular_error)),
        'median_ae': float(np.median(angular_error)),
        'within_11.25': float(np.mean(angular_error < 11.25) * 100),
        'within_22.5': float(np.mean(angular_error < 22.5) * 100),
        'within_30': float(np.mean(angular_error < 30.0) * 100),
    }


def resize_to_gt(pred, gt_shape):
    """Resize prediction to GT resolution."""
    from scipy.ndimage import zoom
    h, w = pred.shape[:2]
    th, tw = gt_shape[:2]
    if (h, w) == (th, tw):
        return pred
    if pred.ndim == 2:
        return zoom(pred, (th / h, tw / w), order=1)
    else:
        return zoom(pred, (th / h, tw / w, 1), order=1)


def load_split(split_file):
    """Load split file (one sample name per line)."""
    if not Path(split_file).exists():
        return None
    return set(Path(split_file).read_text().strip().split("\n"))


def eval_normals_on_dataset(model, data_dir, dataset_name, output_dir, device="cuda",
                            min_coverage=0.5, split=None):
    """Evaluate normal quality on a prepared dataset."""
    split_label = f" ({split} split)" if split else ""
    print(f"\n{'=' * 60}")
    print(f"Evaluating normals on {dataset_name}{split_label}")
    print(f"{'=' * 60}")

    save_dir = Path(output_dir) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])

    # Apply split filter
    split_names = None
    if split:
        split_file = PROJECT_ROOT / "data" / "dermdepth_train" / f"{dataset_name}_moge" / f"{split}.txt"
        split_names = load_split(split_file)
        if split_names:
            sample_dirs = [d for d in sample_dirs if d.name in split_names]
            print(f"  Split filter: {len(sample_dirs)} samples ({split})")
        else:
            print(f"  WARNING: Split file {split_file} not found, evaluating all")

    print(f"  Found {len(sample_dirs)} samples")

    all_normal_metrics = []
    all_depth_metrics = []
    per_sample_results = []
    disease_metrics = defaultdict(list) if dataset_name == 'skinl2' else None

    for i, sample_dir in enumerate(sample_dirs):
        sample_name = sample_dir.name
        image_path = sample_dir / "image.png"
        gt_depth_path = sample_dir / "gt_depth.npy"
        gt_mask_path = sample_dir / "gt_mask.npy"

        if not image_path.exists() or not gt_depth_path.exists():
            continue

        # Filter WoundsDB by coverage
        if dataset_name == 'woundsdb' and gt_mask_path.exists():
            gt_mask = np.load(gt_mask_path)
            coverage = gt_mask.sum() / gt_mask.size
            if coverage < min_coverage:
                continue

        print(f"  [{i+1}/{len(sample_dirs)}] {sample_name}...", end="", flush=True)

        try:
            # Load GT
            gt_depth = np.load(gt_depth_path)
            gt_mask = np.load(gt_mask_path) if gt_mask_path.exists() else None
            h, w = gt_depth.shape[:2]

            # Estimate intrinsics for GT normal derivation
            intrinsics = estimate_intrinsics(h, w, fov_deg=60.0)

            # Derive GT normals from GT depth
            gt_normal, gt_normal_mask = derive_gt_normal(gt_depth, intrinsics, mask=gt_mask)

            # Combine masks
            eval_mask = gt_normal_mask
            if gt_mask is not None:
                eval_mask = eval_mask & gt_mask

            if eval_mask.sum() < 100:
                print(" skipped (too few valid normal pixels)")
                continue

            # Run inference
            output = run_inference(model, str(image_path), device=device)
            pred_normal = output.get('normal', None)

            if pred_normal is None:
                print(" skipped (model has no normal output)")
                continue

            # Resize pred to GT size
            if pred_normal.shape[:2] != gt_normal.shape[:2]:
                pred_normal = resize_to_gt(pred_normal, gt_normal.shape)

            # Normal metrics
            n_metrics = compute_normal_metrics(pred_normal, gt_normal, mask=eval_mask)

            # Also compute depth scale metrics for reference
            pred_depth = output.get('depth', None)
            d_metrics = {}
            if pred_depth is not None:
                if isinstance(pred_depth, torch.Tensor):
                    pred_depth = pred_depth.numpy()
                if pred_depth.shape[:2] != gt_depth.shape[:2]:
                    pred_depth = resize_to_gt(pred_depth, gt_depth.shape)
                depth_mask = (gt_mask if gt_mask is not None else
                              np.isfinite(gt_depth) & (gt_depth > 0))
                depth_mask = depth_mask & np.isfinite(pred_depth) & (pred_depth > 0)
                if depth_mask.sum() > 10:
                    pred_d = pred_depth[depth_mask]
                    gt_d = gt_depth[depth_mask]
                    d_metrics['scale_ratio'] = float(np.median(pred_d / gt_d))

            result = {'sample': sample_name, **n_metrics, **d_metrics}
            per_sample_results.append(result)
            all_normal_metrics.append(n_metrics)
            if d_metrics:
                all_depth_metrics.append(d_metrics)

            # Per-disease tracking for SKINL2
            if disease_metrics is not None:
                name = sample_name
                if name.startswith(('v1_', 'v2_', 'v3_')):
                    name = name[3:]
                disease = '_'.join(name.split('_')[:-1])
                disease_metrics[disease].append(n_metrics)

            print(f" MAE={n_metrics['mae']:.2f} Med={n_metrics['median_ae']:.2f}"
                  f" <11.25={n_metrics['within_11.25']:.1f}%", end="")
            if 'scale_ratio' in d_metrics:
                print(f" scale={d_metrics['scale_ratio']:.3f}", end="")
            print()

        except Exception as e:
            print(f" Error: {e}")
            per_sample_results.append({'sample': sample_name, 'error': str(e)})

    # Aggregate
    if all_normal_metrics:
        summary = {}
        for k in ['mae', 'median_ae', 'within_11.25', 'within_22.5', 'within_30']:
            values = [m[k] for m in all_normal_metrics if k in m]
            if values:
                summary[k] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                }

        if all_depth_metrics:
            scales = [m['scale_ratio'] for m in all_depth_metrics if 'scale_ratio' in m]
            if scales:
                summary['depth_scale_ratio'] = {
                    'mean': float(np.mean(scales)),
                    'median': float(np.median(scales)),
                }

        print(f"\n  {dataset_name} Normal Summary ({len(all_normal_metrics)} samples):")
        for k in ['mae', 'median_ae', 'within_11.25', 'within_22.5', 'within_30']:
            if k in summary:
                print(f"    {k}: {summary[k]['mean']:.2f} +/- {summary[k]['std']:.2f}")
        if 'depth_scale_ratio' in summary:
            print(f"    depth_scale_ratio: {summary['depth_scale_ratio']['mean']:.3f}")

        # Per-disease if SKINL2
        disease_summary = {}
        if disease_metrics:
            for disease, mets in disease_metrics.items():
                ds = {'count': len(mets)}
                for k in ['mae', 'median_ae', 'within_11.25']:
                    vals = [m[k] for m in mets if k in m]
                    if vals:
                        ds[f'{k}_mean'] = float(np.mean(vals))
                disease_summary[disease] = ds

            print(f"\n  Per-disease:")
            for disease, ds in sorted(disease_summary.items()):
                print(f"    {disease}: n={ds['count']}, MAE={ds.get('mae_mean', 0):.2f}"
                      f", <11.25={ds.get('within_11.25_mean', 0):.1f}%")
    else:
        summary = {'note': 'No samples evaluated'}
        disease_summary = {}

    results = {
        'summary': summary,
        'per_sample': per_sample_results,
        'num_evaluated': len(all_normal_metrics),
    }
    if disease_summary:
        results['per_disease'] = disease_summary

    with open(save_dir / "normal_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description="Normal map evaluation for DermDepth")
    parser.add_argument('--model', type=str, required=True,
                        help='Model path (checkpoint .pt or HuggingFace ID)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['woundsdb', 'skinl2', 'all'],
                        help='Dataset to evaluate on')
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "evaluation"),
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--min_coverage', type=float, default=0.5,
                        help='Min GT coverage for WoundsDB (default 0.5)')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Name for this model run')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'test'],
                        help='Evaluate on train or test split only')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"DermDepth Normal Evaluation: {args.model_name}")
    print(f"Model: {args.model}")
    print("=" * 60)

    model = load_model(args.model, args.device)

    eval_dirs = {
        'woundsdb': str(PROJECT_ROOT / "output" / "eval_data" / "woundsdb"),
        'skinl2': str(PROJECT_ROOT / "output" / "eval_data" / "skinl2"),
    }

    datasets = ['woundsdb', 'skinl2'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        eval_normals_on_dataset(
            model, eval_dirs[ds], ds, output_dir,
            device=args.device, min_coverage=args.min_coverage,
            split=args.split
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
