#!/usr/bin/env python3
"""Evaluate baseline depth estimation models on SKINL2 and WoundsDB.

Supports: DA3NESTED, MapAnything, PPD (Pixel-Perfect Depth).
Uses same metrics as eval_depth.py for direct comparison with MoGe-2 / DermDepth.

Usage:
    python eval_baselines.py --method da3nested --dataset all --device cuda
    python eval_baselines.py --method ppd --dataset all --device cuda
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKINL2_DIR = PROJECT_ROOT / "output" / "eval_data" / "skinl2"
WOUNDSDB_DIR = PROJECT_ROOT / "output" / "eval_data" / "woundsdb"
OUTPUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "baselines"


# ---- Metrics (same as eval_depth.py) ----

def compute_metrics(pred, gt, mask):
    """Compute depth evaluation metrics on valid pixels."""
    valid = mask & (gt > 0) & np.isfinite(gt) & (pred > 0) & np.isfinite(pred)
    if valid.sum() < 100:
        return None

    p = pred[valid]
    g = gt[valid]

    # Scale ratio: median(pred) / median(gt)
    scale_ratio = float(np.median(p) / np.median(g))

    # AbsRel
    abs_rel = float(np.mean(np.abs(p - g) / g))

    # Scale-invariant metrics (align pred to gt by median ratio)
    s = np.median(g) / np.median(p)
    p_aligned = p * s

    si_abs_rel = float(np.mean(np.abs(p_aligned - g) / g))

    # Delta thresholds
    ratio = np.maximum(p_aligned / g, g / p_aligned)
    si_delta1 = float(np.mean(ratio < 1.25))
    si_delta2 = float(np.mean(ratio < 1.25 ** 2))
    si_delta3 = float(np.mean(ratio < 1.25 ** 3))

    return {
        'scale_ratio': scale_ratio,
        'abs_rel': abs_rel,
        'si_abs_rel': si_abs_rel,
        'si_delta1': si_delta1,
        'si_delta2': si_delta2,
        'si_delta3': si_delta3,
        'n_valid': int(valid.sum()),
        'median_pred_mm': float(np.median(p) * 1000),
        'median_gt_mm': float(np.median(g) * 1000),
    }


# ---- DA3 Nested Model (outputs metric depth in meters directly) ----

def load_da3nested(device='cuda'):
    """Load DA3NESTED-GIANT-LARGE-1.1 model."""
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "Depth-Anything-3" / "src"))
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = model.to(device).eval()
    return model


def infer_da3nested(model, image_path, device='cuda'):
    """Run DA3NESTED inference, return metric depth in meters.

    DA3NESTED-GIANT-LARGE outputs depth directly in meters (is_metric=1).
    No focal length conversion needed — internally uses a metric sub-model
    to align the depth scale.
    """
    pred = model.inference([str(image_path)])
    depth = pred.depth[0]  # (H, W) in meters
    return depth


# ---- MapAnything Model (outputs metric depth in meters directly) ----

def load_mapanything(device='cuda'):
    """Load MapAnything model."""
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "map-anything"))
    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    return model


def infer_mapanything(model, image_path, device='cuda'):
    """Run MapAnything inference, return metric depth in meters.

    MapAnything outputs depth_z in meters directly.
    Uses memory-efficient inference with AMP.
    """
    from mapanything.utils.image import load_images
    views = load_images([str(image_path)])
    preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype='bf16')
    depth = preds[0]['depth_z'][0, :, :, 0].cpu().numpy()  # (H, W) in meters
    return depth


# ---- Pixel-Perfect Depth (diffusion + MoGe2 metric alignment) ----

PPD_DIR = PROJECT_ROOT / "baseline_methods" / "pixel-perfect-depth"


def load_ppd(device='cuda'):
    """Load PPD model + MoGe2 for metric alignment.

    Returns a dict with both models since PPD requires two-stage inference:
    1. PPD DiT produces relative depth
    2. MoGe2 produces metric depth
    3. RANSAC aligns PPD relative → MoGe2 metric scale
    """
    sys.path.insert(0, str(PPD_DIR))
    import torch
    from ppd.models.ppd import PixelPerfectDepth
    from ppd.moge.model.v2 import MoGeModel

    # Load MoGe2 for metric grounding
    moge = MoGeModel.from_pretrained(str(PPD_DIR / 'checkpoints' / 'moge2.pt')).to(device).eval()

    # Load PPD with DA2 semantics
    ppd_model = PixelPerfectDepth(
        semantics_model='DA2',
        semantics_pth=str(PPD_DIR / 'checkpoints' / 'depth_anything_v2_vitl.pth'),
        sampling_steps=20
    )
    ppd_model.load_state_dict(
        torch.load(str(PPD_DIR / 'checkpoints' / 'ppd.pth'), map_location='cpu'),
        strict=False
    )
    ppd_model = ppd_model.to(device).eval()

    return {'ppd': ppd_model, 'moge': moge, 'device': device}


def infer_ppd(model_dict, image_path, device='cuda'):
    """Run PPD inference with MoGe2 metric alignment.

    Pipeline (following run_point_cloud.py):
    1. PPD: image → relative depth at resized resolution
    2. MoGe2: resized image → metric depth + mask
    3. RANSAC: align PPD relative depth to MoGe2 metric scale
    Returns metric depth in meters at resized resolution.
    """
    import torch
    import cv2
    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    ppd_model = model_dict['ppd']
    moge = model_dict['moge']
    dev = model_dict['device']

    image = cv2.imread(str(image_path))

    # PPD relative depth
    depth, resize_image = ppd_model.infer_image(image)
    depth = depth.squeeze().cpu().numpy()

    # MoGe2 metric depth on the same resized image
    moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    moge_image = torch.tensor(moge_image / 255, dtype=torch.float32, device=dev).permute(2, 0, 1)
    moge_depth, mask, intrinsic = moge.infer(moge_image)
    moge_depth[~mask] = moge_depth[mask].max()

    # RANSAC alignment: PPD relative → MoGe2 metric scale
    metric_depth = recover_metric_depth_ransac(depth, moge_depth, mask)

    return metric_depth


# ---- Evaluation Functions ----

def resize_depth(depth, target_h, target_w):
    """Resize depth map to target resolution."""
    from scipy.ndimage import zoom
    h, w = depth.shape
    if (h, w) == (target_h, target_w):
        return depth
    scale_h = target_h / h
    scale_w = target_w / w
    return zoom(depth, (scale_h, scale_w), order=1)


def eval_skinl2(model, method_name, infer_fn, device='cuda'):
    """Evaluate on SKINL2 dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating {method_name} on SKINL2")
    print(f"{'='*60}")

    results_per_sample = []
    results_by_version = {}

    sample_dirs = sorted([d for d in SKINL2_DIR.iterdir() if d.is_dir()])
    total = len(sample_dirs)

    for i, sample_dir in enumerate(sample_dirs):
        meta_path = sample_dir / 'meta.json'
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        img_path = sample_dir / 'image.png'
        gt_depth = np.load(sample_dir / 'gt_depth.npy')  # meters
        gt_mask = np.load(sample_dir / 'gt_mask.npy').astype(bool)

        if not img_path.exists():
            continue

        # Run inference
        pred_depth = infer_fn(model, img_path, device)

        # Resize to GT resolution
        h_gt, w_gt = gt_depth.shape
        pred_resized = resize_depth(pred_depth, h_gt, w_gt)

        metrics = compute_metrics(pred_resized, gt_depth, gt_mask)
        if metrics is None:
            continue

        version = meta.get('version', 'v1')
        disease = meta.get('disease', 'unknown')

        result = {
            'sample': sample_dir.name,
            'version': version,
            'disease': disease,
            **metrics,
        }
        results_per_sample.append(result)

        if version not in results_by_version:
            results_by_version[version] = []
        results_by_version[version].append(result)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {sample_dir.name}: scale={metrics['scale_ratio']:.2f}, absrel={metrics['abs_rel']:.3f}")

    # Aggregate
    if not results_per_sample:
        print("  No valid results!")
        return {}

    agg_keys = ['scale_ratio', 'abs_rel', 'si_abs_rel', 'si_delta1']
    agg = {k: float(np.mean([r[k] for r in results_per_sample])) for k in agg_keys}

    print(f"\n  SKINL2 Overall (n={len(results_per_sample)}):")
    print(f"    Scale={agg['scale_ratio']:.3f}, AbsRel={agg['abs_rel']:.3f}, "
          f"SI-AbsRel={agg['si_abs_rel']:.4f}, SI-δ1={agg['si_delta1']:.3f}")

    # Per-version
    for ver in sorted(results_by_version.keys()):
        ver_results = results_by_version[ver]
        ver_agg = {k: float(np.mean([r[k] for r in ver_results])) for k in agg_keys}
        print(f"    {ver} (n={len(ver_results)}): Scale={ver_agg['scale_ratio']:.3f}, "
              f"AbsRel={ver_agg['abs_rel']:.3f}, SI-δ1={ver_agg['si_delta1']:.3f}")

    return {
        'dataset': 'skinl2',
        'n_samples': len(results_per_sample),
        'aggregate': agg,
        'per_version': {
            ver: {
                'n': len(results_by_version[ver]),
                **{k: float(np.mean([r[k] for r in results_by_version[ver]])) for k in agg_keys},
            }
            for ver in sorted(results_by_version.keys())
        },
        'per_sample': results_per_sample,
    }


def eval_woundsdb(model, method_name, infer_fn, device='cuda', min_coverage=0.5):
    """Evaluate on WoundsDB dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating {method_name} on WoundsDB")
    print(f"{'='*60}")

    results_per_sample = []
    sample_dirs = sorted([d for d in WOUNDSDB_DIR.iterdir() if d.is_dir()])
    total = len(sample_dirs)

    for i, sample_dir in enumerate(sample_dirs):
        img_path = sample_dir / 'image.png'
        gt_path = sample_dir / 'gt_depth.npy'
        mask_path = sample_dir / 'gt_mask.npy'

        if not all(p.exists() for p in [img_path, gt_path, mask_path]):
            continue

        gt_depth = np.load(gt_path)   # meters
        gt_mask = np.load(mask_path).astype(bool)

        # Coverage check
        coverage = gt_mask.sum() / gt_mask.size
        if coverage < min_coverage:
            continue

        # Run inference
        pred_depth = infer_fn(model, img_path, device)

        # Resize to GT resolution
        h_gt, w_gt = gt_depth.shape
        pred_resized = resize_depth(pred_depth, h_gt, w_gt)

        metrics = compute_metrics(pred_resized, gt_depth, gt_mask)
        if metrics is None:
            continue

        result = {
            'sample': sample_dir.name,
            **metrics,
        }
        results_per_sample.append(result)

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {sample_dir.name}: scale={metrics['scale_ratio']:.2f}, absrel={metrics['abs_rel']:.3f}")

    if not results_per_sample:
        print("  No valid results!")
        return {}

    agg_keys = ['scale_ratio', 'abs_rel', 'si_abs_rel', 'si_delta1']
    agg = {k: float(np.mean([r[k] for r in results_per_sample])) for k in agg_keys}

    print(f"\n  WoundsDB Overall (n={len(results_per_sample)}):")
    print(f"    Scale={agg['scale_ratio']:.3f}, AbsRel={agg['abs_rel']:.3f}, "
          f"SI-AbsRel={agg['si_abs_rel']:.4f}, SI-δ1={agg['si_delta1']:.3f}")

    return {
        'dataset': 'woundsdb',
        'n_samples': len(results_per_sample),
        'aggregate': agg,
        'per_sample': results_per_sample,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate baseline depth models')
    parser.add_argument('--method', type=str, required=True,
                        choices=['da3nested', 'mapanything', 'ppd'],
                        help='Baseline method to evaluate')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['skinl2', 'woundsdb', 'all'],
                        help='Dataset to evaluate on')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--min_coverage', type=float, default=0.5,
                        help='Min GT coverage for WoundsDB (default 0.5)')
    args = parser.parse_args()

    # Load model
    if args.method == 'da3nested':
        print("Loading DA3NESTED-GIANT-LARGE-1.1...")
        model = load_da3nested(args.device)
        method_name = 'DA3-Nested-Giant-Large'
        infer_fn = infer_da3nested
    elif args.method == 'mapanything':
        print("Loading MapAnything...")
        model = load_mapanything(args.device)
        method_name = 'MapAnything'
        infer_fn = infer_mapanything
    elif args.method == 'ppd':
        print("Loading Pixel-Perfect Depth + MoGe2...")
        model = load_ppd(args.device)
        method_name = 'Pixel-Perfect Depth'
        infer_fn = infer_ppd
    else:
        raise ValueError(f"Unknown method: {args.method}")

    results = {'method': method_name}
    out_dir = OUTPUT_DIR / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ('skinl2', 'all'):
        skinl2_results = eval_skinl2(model, method_name, infer_fn, args.device)
        results['skinl2'] = skinl2_results

    if args.dataset in ('woundsdb', 'all'):
        woundsdb_results = eval_woundsdb(model, method_name, infer_fn, args.device, args.min_coverage)
        results['woundsdb'] = woundsdb_results

    # Save summary
    summary = {'method': method_name}
    for ds in ['skinl2', 'woundsdb']:
        if ds in results and results[ds]:
            summary[ds] = {
                'n_samples': results[ds]['n_samples'],
                **results[ds]['aggregate'],
            }
            if 'per_version' in results[ds]:
                summary[ds]['per_version'] = results[ds]['per_version']

    summary_path = out_dir / 'results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Save full per-sample results
    full_path = out_dir / 'results_full.json'
    with open(full_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {full_path}")
