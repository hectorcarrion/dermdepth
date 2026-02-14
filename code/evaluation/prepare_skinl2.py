#!/usr/bin/env python3
"""
Prepare SKINL2 dataset for depth evaluation.

Processes all three SKINL2 versions (v1, v2, v3) using the morphological
enhancement pipeline (Lourenço 2022 + Faria 2021) for depth denoising.

Output per sample:
  - image.png: Central view (cropped for v2 to remove black borders)
  - gt_depth.npy: GT depth in meters (float32)
  - gt_mask.npy: Valid depth mask (bool)
  - meta.json: Metadata including disease, version, depth stats

SKINL2 depth is metric: Raytrix R42 outputs mm, verified against paper.
Both metric scale and scale-invariant evaluation are supported.

Usage:
    python prepare_skinl2.py [--output_dir OUTPUT] [--versions v1 v2 v3]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Add SKINL2 tools to path
sys.path.insert(0, str(DATA_DIR / "SKINL2" / "SKINL2_tools"))
from skinl2_depth_enhance import enhance_depth_map


V2_CROP_PCT = 0.03  # 3% crop from each edge for v2 black borders


def discover_v1():
    """Discover v1 samples: Central View/{disease}/{id}/ + DepthMap/{disease}/{id}/"""
    cases = []
    root = DATA_DIR / "SKINL2" / "SKINL2_v1"
    cv_root = root / "Central View"
    dm_root = root / "DepthMap"

    if not cv_root.exists() or not dm_root.exists():
        return cases

    for disease_dir in sorted(cv_root.iterdir()):
        if not disease_dir.is_dir():
            continue
        disease = disease_dir.name
        dm_disease = dm_root / disease

        for sample_dir in sorted(disease_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sid = sample_dir.name

            cv_files = list(sample_dir.glob("*_TotalFocus.png"))
            if not cv_files:
                cv_files = list(sample_dir.glob("*.png"))

            dm_dir = dm_disease / sid
            dm_files = list(dm_dir.glob("*_DepthMap.tiff")) if dm_dir.exists() else []

            if cv_files and dm_files:
                cases.append({
                    'version': 'v1',
                    'disease': disease,
                    'sample_id': sid,
                    'cv_path': cv_files[0],
                    'dm_path': dm_files[0],
                })
    return cases


def discover_v2v3(version):
    """Discover v2/v3 samples: {id}/{disease}/Light Field/{Central View,Depth Map}/"""
    cases = []
    root = DATA_DIR / "SKINL2" / f"SKINL2_{version}"

    if not root.exists():
        return cases

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        sid = case_dir.name

        for disease_dir in case_dir.iterdir():
            if not disease_dir.is_dir():
                continue
            disease = disease_dir.name

            cv_dir = disease_dir / "Light Field" / "Central View"
            dm_dir = disease_dir / "Light Field" / "Depth Map"

            if not cv_dir.is_dir() or not dm_dir.is_dir():
                continue

            cv_files = list(cv_dir.glob("*TotalFocus*.png"))
            dm_files = list(dm_dir.glob("*DepthMap.tiff"))

            if cv_files and dm_files:
                cases.append({
                    'version': version,
                    'disease': disease,
                    'sample_id': sid,
                    'cv_path': cv_files[0],
                    'dm_path': dm_files[0],
                })
    return cases


def crop_v2_borders(img, depth, crop_pct=V2_CROP_PCT):
    """Crop black borders from v2 images. Returns cropped img, depth."""
    h, w = img.shape[:2]
    t, b = int(h * crop_pct), h - int(h * crop_pct)
    l, r = int(w * crop_pct), w - int(w * crop_pct)
    img_cropped = img[t:b, l:r]

    dh, dw = depth.shape
    dt = int(t * dh / h)
    db = int(b * dh / h)
    dl = int(l * dw / w)
    dr = int(r * dw / w)
    depth_cropped = depth[dt:db, dl:dr]

    return img_cropped, depth_cropped


def prepare_sample(case, output_dir):
    """
    Prepare a single SKINL2 sample for evaluation.

    Uses morphological pipeline for depth enhancement.
    Returns result dict with stats.
    """
    os.makedirs(output_dir, exist_ok=True)

    ver = case['version']
    disease = case['disease']
    sid = case['sample_id']

    result = {
        'version': ver,
        'disease': disease,
        'sample_id': sid,
    }

    # Load central view and depth
    cv_img = np.array(Image.open(case['cv_path']).convert('RGB'))
    depth_raw = np.array(Image.open(case['dm_path']), dtype=np.float32)

    result['original_image_size'] = list(cv_img.shape[:2])
    result['original_depth_size'] = list(depth_raw.shape)

    # v2: crop black borders
    if ver == 'v2':
        cv_img, depth_raw = crop_v2_borders(cv_img, depth_raw)
        result['v2_cropped'] = True

    # Resize central view to depth resolution for enhancement pipeline
    dh, dw = depth_raw.shape
    cv_at_depth_res = np.array(
        Image.fromarray(cv_img).resize((dw, dh), Image.LANCZOS))

    # Run morphological enhancement pipeline
    enhance_result = enhance_depth_map(
        depth_raw.copy(), cv_at_depth_res.copy(),
        method='morphological', verbose=False)
    depth_enhanced = enhance_result['enhanced']

    # Convert to positive meters
    depth_m = np.abs(depth_enhanced) / 1000.0

    # Valid mask: finite, positive, reasonable range (50-500mm)
    valid_mask = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 0.5)

    # Resize depth to image resolution if needed
    img_h, img_w = cv_img.shape[:2]
    if (dh, dw) != (img_h, img_w):
        depth_at_img_res = cv2.resize(depth_m, (img_w, img_h),
                                       interpolation=cv2.INTER_LINEAR)
        mask_at_img_res = cv2.resize(valid_mask.astype(np.uint8),
                                      (img_w, img_h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        depth_at_img_res = depth_m
        mask_at_img_res = valid_mask

    # Downscale if very large (MoGe handles up to ~2048 well)
    MAX_DIM = 2048
    if max(img_h, img_w) > MAX_DIM:
        scale = MAX_DIM / max(img_h, img_w)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        cv_img = np.array(Image.fromarray(cv_img).resize((new_w, new_h), Image.LANCZOS))
        depth_at_img_res = cv2.resize(depth_at_img_res, (new_w, new_h),
                                       interpolation=cv2.INTER_LINEAR)
        mask_at_img_res = cv2.resize(mask_at_img_res.astype(np.uint8),
                                      (new_w, new_h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
        result['downscaled'] = True
        result['final_size'] = [new_h, new_w]
    else:
        result['final_size'] = [img_h, img_w]

    # Save outputs
    Image.fromarray(cv_img).save(os.path.join(output_dir, "image.png"))
    np.save(os.path.join(output_dir, "gt_depth.npy"), depth_at_img_res.astype(np.float32))
    np.save(os.path.join(output_dir, "gt_mask.npy"), mask_at_img_res)

    # Depth stats
    valid_vals = depth_at_img_res[mask_at_img_res]
    if len(valid_vals) > 0:
        result['depth_stats'] = {
            'min_mm': float(valid_vals.min() * 1000),
            'max_mm': float(valid_vals.max() * 1000),
            'mean_mm': float(valid_vals.mean() * 1000),
            'std_mm': float(valid_vals.std() * 1000),
            'span_mm': float((valid_vals.max() - valid_vals.min()) * 1000),
            'valid_ratio': float(len(valid_vals) / depth_at_img_res.size),
            'valid_pixels': int(len(valid_vals)),
        }
    else:
        result['depth_stats'] = {'valid_ratio': 0.0, 'valid_pixels': 0}

    # Save metadata
    meta = {
        'width': result['final_size'][1],
        'height': result['final_size'][0],
        'depth_units': 'meters',
        'depth_source': 'Raytrix R42 plenoptic camera (depth from light field)',
        'enhancement': 'morphological (Lourenço 2022 + Faria 2021)',
        'is_metric': True,
        'disease': disease,
        'version': ver,
        'sample_id': sid,
    }
    with open(os.path.join(output_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    result['status'] = 'success'
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare SKINL2 for depth evaluation")
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "eval_data" / "skinl2"),
                        help='Output directory')
    parser.add_argument('--versions', nargs='+', default=['v1', 'v2', 'v3'],
                        choices=['v1', 'v2', 'v3'],
                        help='Which SKINL2 versions to process')
    args = parser.parse_args()

    print("=" * 60)
    print("Preparing SKINL2 for Evaluation")
    print(f"  Enhancement: morphological pipeline (Lourenço 2022 + Faria 2021)")
    print(f"  Versions: {args.versions}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all cases
    all_cases = []
    for ver in args.versions:
        if ver == 'v1':
            cases = discover_v1()
        else:
            cases = discover_v2v3(ver)
        print(f"  {ver}: {len(cases)} paired samples found")
        all_cases.extend(cases)

    print(f"  Total: {len(all_cases)} samples\n")

    if not all_cases:
        print("ERROR: No samples found.")
        return

    # Process
    results = []
    disease_counts = defaultdict(int)
    version_counts = defaultdict(int)

    for i, case in enumerate(all_cases):
        ver = case['version']
        disease = case['disease']
        sid = case['sample_id']
        sample_name = f"{ver}_{disease}_{sid}"
        out_dir = os.path.join(args.output_dir, sample_name)

        print(f"  [{i+1}/{len(all_cases)}] {sample_name}", end="", flush=True)

        # Skip if already processed
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            print("  (exists, skipping)")
            # Count it as success
            try:
                with open(os.path.join(out_dir, "meta.json")) as f:
                    meta = json.load(f)
                disease_counts[disease] += 1
                version_counts[ver] += 1
                results.append({
                    'version': ver, 'disease': disease,
                    'sample_id': sid, 'status': 'success',
                    'depth_stats': {'valid_ratio': 1.0, 'valid_pixels': 1, 'mean_mm': 0, 'span_mm': 0, 'min_mm': 0, 'max_mm': 0},
                })
            except:
                pass
            continue

        try:
            result = prepare_sample(case, out_dir)
            results.append(result)
            disease_counts[disease] += 1
            version_counts[ver] += 1

            stats = result.get('depth_stats', {})
            if stats.get('valid_ratio', 0) > 0:
                print(f"  depth={stats['mean_mm']:.1f}mm "
                      f"span={stats['span_mm']:.1f}mm "
                      f"cover={stats['valid_ratio']:.1%}")
            else:
                print("  (no valid depth)")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'version': ver, 'disease': disease,
                'sample_id': sid, 'status': 'failed', 'error': str(e)
            })

    # Summary
    success = sum(1 for r in results if r.get('status') == 'success')
    valid_results = [r for r in results
                     if r.get('status') == 'success'
                     and r.get('depth_stats', {}).get('valid_ratio', 0) > 0]

    summary = {
        'total_found': len(all_cases),
        'prepared': success,
        'failed': len(all_cases) - success,
        'versions_processed': dict(version_counts),
        'disease_distribution': dict(disease_counts),
        'enhancement': 'morphological (Lourenço 2022 + Faria 2021)',
        'depth_units': 'meters (converted from mm)',
        'is_metric': True,
        'v2_border_crop': f'{V2_CROP_PCT*100:.0f}%',
    }

    if valid_results:
        all_means = [r['depth_stats']['mean_mm'] for r in valid_results]
        all_spans = [r['depth_stats']['span_mm'] for r in valid_results]
        all_coverage = [r['depth_stats']['valid_ratio'] for r in valid_results]
        summary['aggregate_stats'] = {
            'mean_depth_mm': float(np.mean(all_means)),
            'std_depth_mm': float(np.std(all_means)),
            'min_depth_mm': float(min(r['depth_stats']['min_mm'] for r in valid_results)),
            'max_depth_mm': float(max(r['depth_stats']['max_mm'] for r in valid_results)),
            'mean_span_mm': float(np.mean(all_spans)),
            'mean_coverage': float(np.mean(all_coverage)),
            'n_valid': len(valid_results),
        }

    with open(os.path.join(args.output_dir, "preparation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Create index file
    sample_names = [f"{r['version']}_{r['disease']}_{r['sample_id']}"
                    for r in results if r.get('status') == 'success']
    with open(os.path.join(args.output_dir, ".index.txt"), 'w') as f:
        f.write('\n'.join(sorted(sample_names)))

    print(f"\n{'='*60}")
    print(f"Done. {success}/{len(all_cases)} samples prepared.")
    print(f"  Versions: {dict(version_counts)}")
    print(f"  Diseases: {dict(disease_counts)}")
    if valid_results:
        s = summary['aggregate_stats']
        print(f"  Mean depth: {s['mean_depth_mm']:.1f} ± {summary['aggregate_stats']['std_depth_mm']:.1f} mm")
        print(f"  Mean span: {s['mean_span_mm']:.1f} mm")
        print(f"  Mean coverage: {s['mean_coverage']:.1%}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
