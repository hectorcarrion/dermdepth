#!/usr/bin/env python3
"""
Unified depth evaluation script for DermDepth.

Runs a MoGe-2 model (pretrained or fine-tuned) on evaluation datasets
and computes comprehensive metrics.

Datasets:
- WoundsDB: metric depth (AbsRel, Delta1, RMSE, Scale Error)
- SKINL2: scale-invariant + structural (since depth units unknown)

Uses MoGe's built-in metrics from moge/test/metrics.py.

Usage:
    python eval_depth.py --model MODEL_PATH --dataset woundsdb --data_dir PREPARED_DATA
    python eval_depth.py --model MODEL_PATH --dataset skinl2 --data_dir PREPARED_DATA
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))


def load_model(model_path, device="cuda"):
    """Load MoGe-2 model from checkpoint or HuggingFace."""
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")

    if os.path.isfile(model_path):
        # Load from local checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model_config = checkpoint.get('model_config', None)
        if model_config:
            model = MoGeModel(**model_config)
        else:
            model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        # Load from HuggingFace
        model = MoGeModel.from_pretrained(model_path)

    model = model.to(device).eval()
    return model


def run_inference(model, image_path, device="cuda"):
    """Run model inference on a single image."""
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
                result[key] = val.cpu()
                if result[key].ndim > 2 and result[key].shape[0] == 1:
                    result[key] = result[key].squeeze(0)
            else:
                result[key] = val

    return result


def compute_depth_metrics(pred_depth, gt_depth, mask=None, is_metric=True):
    """
    Compute standard depth evaluation metrics.

    Args:
        pred_depth: Predicted depth (HxW tensor or array)
        gt_depth: Ground truth depth (HxW tensor or array)
        mask: Valid pixel mask (HxW bool)
        is_metric: If True, compute metric-scale metrics

    Returns:
        dict of metrics
    """
    if isinstance(pred_depth, np.ndarray):
        pred_depth = torch.from_numpy(pred_depth).float()
    if isinstance(gt_depth, np.ndarray):
        gt_depth = torch.from_numpy(gt_depth).float()
    if mask is None:
        mask = torch.isfinite(gt_depth) & (gt_depth > 0) & torch.isfinite(pred_depth) & (pred_depth > 0)
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).bool()

    mask = mask & torch.isfinite(gt_depth) & (gt_depth > 0) & torch.isfinite(pred_depth) & (pred_depth > 0)

    if mask.sum() < 10:
        return {'valid_pixels': 0}

    pred = pred_depth[mask]
    gt = gt_depth[mask]

    metrics = {'valid_pixels': int(mask.sum().item())}

    # Metric-scale metrics (no alignment)
    if is_metric:
        # AbsRel
        metrics['absrel'] = float((torch.abs(pred - gt) / gt).mean().item())
        # RMSE (in same units as depth)
        metrics['rmse'] = float(torch.sqrt(((pred - gt) ** 2).mean()).item())
        # Delta1 (threshold 1.25)
        ratio = torch.max(pred / gt, gt / pred)
        metrics['delta1'] = float((ratio < 1.25).float().mean().item())
        metrics['delta2'] = float((ratio < 1.25 ** 2).float().mean().item())
        metrics['delta3'] = float((ratio < 1.25 ** 3).float().mean().item())
        # Scale error: median ratio
        scale_ratio = (pred / gt).median().item()
        metrics['scale_ratio'] = float(scale_ratio)
        metrics['scale_error_pct'] = float(abs(scale_ratio - 1.0) * 100)

    # Scale-invariant metrics (align scale before computing)
    scale = (gt * pred).sum() / (pred * pred).sum()
    pred_aligned = pred * scale
    metrics['si_absrel'] = float((torch.abs(pred_aligned - gt) / gt).mean().item())
    metrics['si_rmse'] = float(torch.sqrt(((pred_aligned - gt) ** 2).mean()).item())
    ratio_si = torch.max(pred_aligned / gt, gt / pred_aligned)
    metrics['si_delta1'] = float((ratio_si < 1.25).float().mean().item())

    # Affine-invariant metrics (align scale + shift)
    A = torch.stack([pred, torch.ones_like(pred)], dim=-1)
    x, _, _, _ = torch.linalg.lstsq(A, gt.unsqueeze(-1))
    pred_affine = (A @ x).squeeze(-1)
    metrics['ai_absrel'] = float((torch.abs(pred_affine - gt) / gt).mean().item())
    ratio_ai = torch.max(pred_affine / gt.clamp_min(1e-6), gt / pred_affine.clamp_min(1e-6))
    metrics['ai_delta1'] = float((ratio_ai < 1.25).float().mean().item())

    return metrics


def eval_woundsdb(model, data_dir, output_dir, device="cuda", min_coverage=0.5):
    """
    Evaluate on prepared WoundsDB data.
    """
    print("\n" + "=" * 60)
    print("Evaluating on WoundsDB")
    print("=" * 60)

    save_dir = Path(output_dir) / "woundsdb"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find prepared scenes
    scene_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
    print(f"  Found {len(scene_dirs)} prepared scenes")

    all_metrics = []
    per_scene_results = []

    for i, scene_dir in enumerate(scene_dirs):
        scene_name = scene_dir.name
        image_path = scene_dir / "image.png"
        gt_depth_path = scene_dir / "gt_depth.npy"
        gt_mask_path = scene_dir / "gt_mask.npy"
        meta_path = scene_dir / "meta.json"

        if not image_path.exists() or not gt_mask_path.exists():
            continue

        # Filter by coverage
        gt_mask = np.load(gt_mask_path)
        coverage = gt_mask.sum() / gt_mask.size
        if coverage < min_coverage:
            print(f"  [{i+1}/{len(scene_dirs)}] {scene_name}... skipped (coverage {coverage*100:.1f}% < {min_coverage*100:.0f}%)")
            continue

        print(f"  [{i+1}/{len(scene_dirs)}] {scene_name}...", end="")

        try:
            # Run inference
            output = run_inference(model, str(image_path), device=device)
            pred_depth = output['depth'].numpy() if isinstance(output['depth'], torch.Tensor) else output['depth']

            # Load GT depth
            gt_depth = np.load(gt_depth_path)

            # Resize prediction to GT size if needed
            if pred_depth.shape != gt_depth.shape:
                from PIL import Image as PILImage
                pred_pil = PILImage.fromarray(pred_depth)
                pred_pil = pred_pil.resize((gt_depth.shape[1], gt_depth.shape[0]), PILImage.BILINEAR)
                pred_depth = np.array(pred_pil)

            metrics = compute_depth_metrics(pred_depth, gt_depth, mask=gt_mask, is_metric=True)

            result = {'scene': scene_name, 'coverage': float(coverage), **metrics}
            per_scene_results.append(result)

            all_metrics.append(metrics)
            print(f" AbsRel={metrics['absrel']:.4f} Scale={metrics.get('scale_ratio', 0):.3f}")

        except Exception as e:
            print(f" Error: {e}")
            per_scene_results.append({'scene': scene_name, 'error': str(e)})

    # Aggregate metrics
    if all_metrics:
        summary = {}
        metric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        for k in metric_keys:
            values = [m[k] for m in all_metrics if k in m]
            if values:
                summary[k] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                }

        print(f"\n  WoundsDB Summary ({len(all_metrics)} scenes with GT):")
        for k in ['absrel', 'rmse', 'delta1', 'scale_ratio', 'scale_error_pct']:
            if k in summary:
                print(f"    {k}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")
    else:
        summary = {'note': 'No scenes with GT depth available'}

    results = {
        'summary': summary,
        'per_scene': per_scene_results,
        'num_evaluated': len(all_metrics),
    }

    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def eval_skinl2(model, data_dir, output_dir, device="cuda"):
    """
    Evaluate on prepared SKINL2 data.

    Supports both metric evaluation (depth in meters, verified from paper)
    and scale-invariant metrics. Checks meta.json per sample for is_metric flag.
    """
    print("\n" + "=" * 60)
    print("Evaluating on SKINL2")
    print("=" * 60)

    save_dir = Path(output_dir) / "skinl2"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find prepared samples
    sample_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
    print(f"  Found {len(sample_dirs)} prepared samples")

    all_metrics = []
    per_sample_results = []
    disease_metrics = defaultdict(list)

    for i, sample_dir in enumerate(sample_dirs):
        sample_name = sample_dir.name
        image_path = sample_dir / "image.png"
        gt_depth_path = sample_dir / "gt_depth.npy"
        mask_path = sample_dir / "gt_mask.npy"
        meta_path = sample_dir / "meta.json"

        if not image_path.exists() or not gt_depth_path.exists():
            continue

        print(f"  [{i+1}/{len(sample_dirs)}] {sample_name}...", end="")

        try:
            # Check if metric evaluation is available
            is_metric = False
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                is_metric = meta.get('is_metric', False)

            output = run_inference(model, str(image_path), device=device)
            pred_depth = output['depth'].numpy() if isinstance(output['depth'], torch.Tensor) else output['depth']
            gt_depth = np.load(gt_depth_path)

            # Load mask if available
            mask = None
            if mask_path.exists():
                mask = np.load(mask_path)

            # Resize prediction to GT size if needed
            if pred_depth.shape != gt_depth.shape:
                from PIL import Image as PILImage
                pred_pil = PILImage.fromarray(pred_depth.astype(np.float32))
                pred_pil = pred_pil.resize((gt_depth.shape[1], gt_depth.shape[0]), PILImage.BILINEAR)
                pred_depth = np.array(pred_pil)

            # Compute metrics (both metric and scale-invariant)
            metrics = compute_depth_metrics(pred_depth, gt_depth, mask=mask, is_metric=is_metric)

            # Extract disease from sample name (handles v1_Disease Name_0001 format)
            name = sample_name
            if name.startswith(('v1_', 'v2_', 'v3_')):
                name = name[3:]  # strip version prefix
            disease = '_'.join(name.split('_')[:-1])
            metrics['disease'] = disease

            result = {'sample': sample_name, 'is_metric': is_metric, **metrics}
            per_sample_results.append(result)
            all_metrics.append(metrics)
            disease_metrics[disease].append(metrics)

            if is_metric and 'absrel' in metrics:
                print(f" AbsRel={metrics['absrel']:.4f} Scale={metrics.get('scale_ratio', 0):.3f}"
                      f" SI-d1={metrics['si_delta1']:.4f}")
            else:
                print(f" SI-AbsRel={metrics['si_absrel']:.4f} SI-d1={metrics['si_delta1']:.4f}")

        except Exception as e:
            print(f" Error: {e}")

    # Aggregate
    if all_metrics:
        summary = {}
        # Always include scale-invariant metrics
        metric_keys = ['si_absrel', 'si_rmse', 'si_delta1', 'ai_absrel', 'ai_delta1']
        # Include metric-scale metrics if available
        if any('absrel' in m for m in all_metrics):
            metric_keys = ['absrel', 'rmse', 'delta1', 'scale_ratio', 'scale_error_pct'] + metric_keys

        for k in metric_keys:
            values = [m[k] for m in all_metrics if k in m]
            if values:
                summary[k] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }

        # Per-disease breakdown
        disease_summary = {}
        for disease, disease_mets in disease_metrics.items():
            ds = {'count': len(disease_mets)}
            for k in ['absrel', 'scale_ratio', 'si_absrel', 'si_delta1']:
                vals = [m[k] for m in disease_mets if k in m]
                if vals:
                    ds[f'{k}_mean'] = float(np.mean(vals))
            disease_summary[disease] = ds

        print(f"\n  SKINL2 Summary ({len(all_metrics)} samples):")
        for k in metric_keys:
            if k in summary:
                print(f"    {k}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")

        print(f"\n  Per-disease:")
        for disease, ds in sorted(disease_summary.items()):
            parts = [f"n={ds['count']}"]
            if 'absrel_mean' in ds:
                parts.append(f"AbsRel={ds['absrel_mean']:.4f}")
            if 'si_delta1_mean' in ds:
                parts.append(f"SI-d1={ds['si_delta1_mean']:.4f}")
            print(f"    {disease}: {', '.join(parts)}")
    else:
        summary = {'note': 'No samples evaluated'}
        disease_summary = {}

    results = {
        'summary': summary,
        'per_disease': disease_summary,
        'per_sample': per_sample_results,
        'num_evaluated': len(all_metrics),
    }

    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified depth evaluation for DermDepth")
    parser.add_argument('--model', type=str, required=True,
                        help='Model path (checkpoint .pt or HuggingFace ID)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['woundsdb', 'skinl2', 'all'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Prepared data directory (from prepare_*.py)')
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "evaluation"),
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--min_coverage', type=float, default=0.5,
                        help='Min GT coverage to include a WoundsDB scene (default 0.5)')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Name for this model run (used in output path)')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"DermDepth Evaluation: {args.model_name}")
    print(f"Model: {args.model}")
    print("=" * 60)

    model = load_model(args.model, args.device)

    eval_default_dirs = {
        'woundsdb': str(PROJECT_ROOT / "output" / "eval_data" / "woundsdb"),
        'skinl2': str(PROJECT_ROOT / "output" / "eval_data" / "skinl2"),
    }

    if args.dataset in ('woundsdb', 'all'):
        data_dir = args.data_dir or eval_default_dirs['woundsdb']
        eval_woundsdb(model, data_dir, output_dir, args.device, min_coverage=args.min_coverage)

    if args.dataset in ('skinl2', 'all'):
        data_dir = args.data_dir or eval_default_dirs['skinl2']
        eval_skinl2(model, data_dir, output_dir, args.device)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
