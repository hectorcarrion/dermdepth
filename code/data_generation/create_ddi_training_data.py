#!/usr/bin/env python3
"""
Create pseudo-GT metric depth maps for DDI training from ruler-based scale corrections.

Approach:
1. Load cached MoGe-2 base predictions (geometry is good, scale is wrong)
2. For each sample, compute scale correction from ruler area:
   - area ∝ depth², so scale_factor = 1 / sqrt(area_ratio)
   - corrected_depth = base_depth * scale_factor
3. Save as MoGe training format (image, depth.png, meta.json)
4. Stratified train/test split by skin tone (70/30)

The ruler provides an external metric anchor — the corrected depth maps have
correct absolute scale while preserving MoGe-2's excellent relative geometry.

Usage:
    conda run -n MoGe python code/data_generation/create_ddi_training_data.py
    conda run -n MoGe python code/data_generation/create_ddi_training_data.py --test_fraction 0.3 --seed 42
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))

from moge.utils.io import write_depth, read_depth

DDI_IMAGES = PROJECT_ROOT / "data" / "DDI" / "images"
CACHE_DIR = PROJECT_ROOT / "output" / "evaluation" / "ddi_rulers" / "_cache"
RESULTS_JSON = PROJECT_ROOT / "output" / "evaluation" / "ddi_rulers" / "ddi_ruler_results.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dermdepth_train" / "ddi_moge"


def estimate_intrinsics(width, height, fov_deg=60.0):
    """Estimate normalized intrinsics assuming a given FoV."""
    fov_rad = np.radians(fov_deg)
    fx_norm = 1.0 / (2.0 * np.tan(fov_rad / 2.0))
    fy_norm = fx_norm
    return [
        [fx_norm, 0.0, 0.5],
        [0.0, fy_norm, 0.5],
        [0.0, 0.0, 1.0],
    ]


def create_ddi_training_data(fov_deg=60.0, test_fraction=0.3, seed=42):
    """Create pseudo-GT depth maps for DDI and split by skin tone."""

    # Load evaluation results for per-sample area ratios
    if not RESULTS_JSON.exists():
        print(f"ERROR: {RESULTS_JSON} not found. Run DDI evaluation first.")
        sys.exit(1)

    results = json.load(open(RESULTS_JSON))
    per_sample = results['per_sample']

    # Build lookup: stem -> per-sample info
    sample_info = {}
    for s in per_sample:
        if 'moge2' not in s['methods']:
            continue
        sample_info[s['stem']] = {
            'skin_tone': s['skin_tone'],
            'disease': s['disease'],
            'filename': s['filename'],
            'area_cm2': s['methods']['moge2']['area_cm2'],
            'ratio': s['methods']['moge2']['ratio'],
        }

    print(f"DDI Pseudo-GT Creation")
    print(f"  Source: MoGe-2 base predictions (ruler-corrected)")
    print(f"  Available samples: {len(sample_info)}")
    print(f"  GT ruler area: {results['gt_area_cm2']:.1f} cm²")
    print(f"  Assumed FoV: {fov_deg}°")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converted = []

    for stem, info in sorted(sample_info.items()):
        # Load base MoGe-2 depth prediction
        depth_path = CACHE_DIR / "moge2" / f"{stem}_depth.npy"
        if not depth_path.exists():
            print(f"  SKIP {stem}: no cached MoGe-2 prediction")
            continue

        # Load image
        img_path = DDI_IMAGES / info['filename']
        if not img_path.exists():
            print(f"  SKIP {stem}: image not found")
            continue

        depth = np.load(depth_path).astype(np.float32)
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # Resize depth to image resolution if needed
        if depth.shape[:2] != (img_h, img_w):
            from scipy.ndimage import zoom
            depth = zoom(depth, (img_h / depth.shape[0], img_w / depth.shape[1]), order=1)

        # Compute scale correction from ruler area ratio
        # area ∝ depth², so to correct: new_depth = old_depth / sqrt(ratio)
        ratio = info['ratio']
        scale_correction = 1.0 / np.sqrt(ratio)
        corrected_depth = depth * scale_correction

        # Verify: corrected depth should give area_ratio ≈ 1.0
        # (since area ∝ depth², new_ratio = old_ratio * scale_correction² = old_ratio * (1/old_ratio) = 1.0)

        # Create output directory
        out_dir = OUTPUT_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save image as JPEG (consistent with other MoGe datasets)
        img.save(out_dir / "image.jpg", quality=95)

        # Encode depth as MoGe 16-bit PNG
        # Set pixels with non-finite depth to NaN
        corrected_depth[~np.isfinite(corrected_depth)] = np.nan
        corrected_depth[corrected_depth <= 0] = np.nan
        write_depth(out_dir / "depth.png", corrected_depth)

        # Verify round-trip
        depth_rt = read_depth(out_dir / "depth.png")
        valid = np.isfinite(corrected_depth) & np.isfinite(depth_rt)
        if valid.sum() > 0:
            rel_error = np.abs(depth_rt[valid] - corrected_depth[valid]) / corrected_depth[valid]
            max_rel = rel_error.max()
            if max_rel > 0.01:
                print(f"  WARNING {stem}: round-trip error {max_rel:.4f}")

        # Create meta.json
        intrinsics = estimate_intrinsics(img_w, img_h, fov_deg)
        with open(out_dir / "meta.json", "w") as f:
            json.dump({"intrinsics": intrinsics}, f, indent=2)

        median_depth_mm = np.nanmedian(corrected_depth) * 1000
        converted.append({
            'stem': stem,
            'skin_tone': info['skin_tone'],
            'disease': info['disease'],
            'ratio': ratio,
            'scale_correction': scale_correction,
            'median_depth_mm': float(median_depth_mm),
        })

        print(f"  [{len(converted):>2}/{len(sample_info)}] {stem} "
              f"(tone {info['skin_tone']}): ratio={ratio:.1f}x → "
              f"correction={scale_correction:.3f}, med_depth={median_depth_mm:.0f}mm")

    # Stratified train/test split by skin tone
    rng = random.Random(seed)
    by_tone = defaultdict(list)
    for c in converted:
        by_tone[c['skin_tone']].append(c['stem'])

    train_names = []
    test_names = []

    print(f"\nStratified split (test_fraction={test_fraction}, seed={seed}):")
    for tone in sorted(by_tone.keys()):
        names = sorted(by_tone[tone])
        rng.shuffle(names)
        n_test = max(1, round(len(names) * test_fraction))
        test_split = names[:n_test]
        train_split = names[n_test:]
        test_names.extend(test_split)
        train_names.extend(train_split)
        print(f"  Tone {tone}: {len(train_split)} train / {len(test_split)} test "
              f"(of {len(names)} total)")

    train_names.sort()
    test_names.sort()
    all_names = sorted(train_names + test_names)

    # Write index files
    with open(OUTPUT_DIR / ".index.txt", "w") as f:
        f.write("\n".join(all_names) + "\n")
    with open(OUTPUT_DIR / "train.txt", "w") as f:
        f.write("\n".join(train_names) + "\n")
    with open(OUTPUT_DIR / "test.txt", "w") as f:
        f.write("\n".join(test_names) + "\n")

    # Write conversion log
    log = {
        'fov_deg': fov_deg,
        'gt_area_cm2': results['gt_area_cm2'],
        'source': 'MoGe-2 base predictions with ruler-based scale correction',
        'n_converted': len(converted),
        'n_train': len(train_names),
        'n_test': len(test_names),
        'split_seed': seed,
        'test_fraction': test_fraction,
        'by_tone': {tone: len(names) for tone, names in by_tone.items()},
        'samples': converted,
    }
    with open(OUTPUT_DIR / "conversion_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Summary stats
    corrections = [c['scale_correction'] for c in converted]
    depths = [c['median_depth_mm'] for c in converted]
    print(f"\nConversion complete:")
    print(f"  Total: {len(converted)} samples")
    print(f"  Train: {len(train_names)}, Test: {len(test_names)}")
    print(f"  Scale corrections: median={np.median(corrections):.3f}, "
          f"range=[{min(corrections):.3f}, {max(corrections):.3f}]")
    print(f"  Corrected depth: median={np.median(depths):.0f}mm, "
          f"range=[{min(depths):.0f}, {max(depths):.0f}]mm")
    print(f"  Output: {OUTPUT_DIR}")

    return train_names, test_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create DDI pseudo-GT training data')
    parser.add_argument('--fov', type=float, default=60.0, help='Assumed FoV (degrees)')
    parser.add_argument('--test_fraction', type=float, default=0.3, help='Test split fraction per tone')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    args = parser.parse_args()

    create_ddi_training_data(fov_deg=args.fov, test_fraction=args.test_fraction, seed=args.seed)
