#!/usr/bin/env python3
"""
Convert WoundsDB and SKINL2 evaluation data to MoGe training format.

Reads prepared eval data (gt_depth.npy, gt_mask.npy, image.png, meta.json)
and converts to MoGe format (image.png, depth.png, meta.json with normalized intrinsics).

Creates train/test splits:
- WoundsDB: cases 1-30 → train, cases 31+ → test (by case ID to avoid leakage)
- SKINL2: v1 → train, v2+v3 → test

Usage:
    python convert_eval_to_moge.py --dataset woundsdb
    python convert_eval_to_moge.py --dataset skinl2
    python convert_eval_to_moge.py --dataset all
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))

from moge.utils.io import write_depth, read_depth


def estimate_intrinsics(width, height, fov_deg=60.0):
    """Estimate normalized intrinsics assuming a given FoV."""
    fov_rad = np.radians(fov_deg)
    # fx_pixel = width / (2 * tan(fov/2)), normalized = fx_pixel / width = 1 / (2 * tan(fov/2))
    fx_norm = 1.0 / (2.0 * np.tan(fov_rad / 2.0))
    fy_norm = fx_norm  # square pixels
    intrinsics = [
        [fx_norm, 0.0, 0.5],
        [0.0, fy_norm, 0.5],
        [0.0, 0.0, 1.0],
    ]
    return intrinsics


def convert_sample(eval_dir, output_dir, fov_deg=60.0):
    """Convert a single eval sample to MoGe format.

    Returns True if successful, False otherwise.
    """
    gt_depth_path = eval_dir / "gt_depth.npy"
    gt_mask_path = eval_dir / "gt_mask.npy"
    image_path = eval_dir / "image.png"

    if not gt_depth_path.exists() or not image_path.exists():
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load depth and mask
    gt_depth = np.load(gt_depth_path).astype(np.float32)
    if gt_mask_path.exists():
        gt_mask = np.load(gt_mask_path).astype(bool)
    else:
        gt_mask = np.isfinite(gt_depth) & (gt_depth > 0)

    # Apply mask: set invalid pixels to NaN
    depth_for_moge = gt_depth.copy()
    depth_for_moge[~gt_mask] = np.nan

    # Write depth as MoGe 16-bit PNG
    write_depth(output_dir / "depth.png", depth_for_moge)

    # Verify round-trip
    depth_rt = read_depth(output_dir / "depth.png")
    valid = gt_mask & np.isfinite(depth_rt)
    if valid.sum() > 0:
        rel_error = np.abs(depth_rt[valid] - gt_depth[valid]) / gt_depth[valid]
        max_rel_error = rel_error.max()
        if max_rel_error > 0.01:
            print(f"  WARNING: depth round-trip max relative error = {max_rel_error:.4f}")

    # Copy image
    shutil.copy2(image_path, output_dir / "image.png")

    # Create meta.json with estimated intrinsics
    img = Image.open(image_path)
    width, height = img.size
    intrinsics = estimate_intrinsics(width, height, fov_deg)

    meta = {"intrinsics": intrinsics}
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return True


def get_woundsdb_case_id(scene_name):
    """Extract case ID number from scene name like 'case_12_day_1_scene_1'."""
    parts = scene_name.split("_")
    return int(parts[1])


def convert_woundsdb(eval_dir, output_base, fov_deg=60.0, min_coverage=0.5,
                     train_max_case=30):
    """Convert WoundsDB eval data to MoGe format with train/test split."""
    eval_dir = Path(eval_dir)
    output_base = Path(output_base)

    scene_dirs = sorted([d for d in eval_dir.iterdir() if d.is_dir()])
    print(f"Found {len(scene_dirs)} WoundsDB scenes")

    train_names = []
    test_names = []
    skipped = []

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name

        # Check coverage
        gt_mask_path = scene_dir / "gt_mask.npy"
        if gt_mask_path.exists():
            gt_mask = np.load(gt_mask_path)
            coverage = gt_mask.sum() / gt_mask.size
            if coverage < min_coverage:
                skipped.append((scene_name, coverage))
                continue

        # Convert
        out_dir = output_base / scene_name
        success = convert_sample(scene_dir, out_dir, fov_deg)
        if not success:
            continue

        # Split by case ID
        case_id = get_woundsdb_case_id(scene_name)
        if case_id <= train_max_case:
            train_names.append(scene_name)
        else:
            test_names.append(scene_name)

    # Write index files
    all_names = sorted(train_names + test_names)
    with open(output_base / ".index.txt", "w") as f:
        f.write("\n".join(all_names) + "\n")
    with open(output_base / "train.txt", "w") as f:
        f.write("\n".join(sorted(train_names)) + "\n")
    with open(output_base / "test.txt", "w") as f:
        f.write("\n".join(sorted(test_names)) + "\n")

    print(f"\nWoundsDB conversion complete:")
    print(f"  Train: {len(train_names)} scenes (cases <= {train_max_case})")
    print(f"  Test:  {len(test_names)} scenes (cases > {train_max_case})")
    print(f"  Skipped: {len(skipped)} (low coverage)")
    for name, cov in skipped:
        print(f"    {name}: {cov*100:.1f}%")

    return {"train": train_names, "test": test_names, "skipped": skipped}


def convert_skinl2(eval_dir, output_base, fov_deg=60.0, stratified=False,
                   test_fraction=0.3, seed=42):
    """Convert SKINL2 eval data to MoGe format with train/test split.

    Args:
        stratified: If True, split 70/30 within each version (v1/v2/v3).
                    If False, use v1=train, v2+v3=test (legacy behavior).
        test_fraction: Fraction of each version to hold out (only if stratified).
        seed: Random seed for reproducible stratified split.
    """
    eval_dir = Path(eval_dir)
    output_base = Path(output_base)

    sample_dirs = sorted([d for d in eval_dir.iterdir() if d.is_dir()])
    print(f"Found {len(sample_dirs)} SKINL2 samples")

    # Convert all samples
    all_names = []
    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        out_dir = output_base / sample_name
        success = convert_sample(sample_dir, out_dir, fov_deg)
        if success:
            all_names.append(sample_name)

    if stratified:
        # Group by version
        import random
        rng = random.Random(seed)
        by_version = {}
        for name in all_names:
            version = name.split("_")[0]  # v1, v2, v3
            by_version.setdefault(version, []).append(name)

        train_names = []
        test_names = []
        print(f"\n  Stratified split (seed={seed}, test_frac={test_fraction}):")
        for version in sorted(by_version.keys()):
            names = sorted(by_version[version])
            rng.shuffle(names)
            n_test = max(1, round(len(names) * test_fraction))
            test_names.extend(names[:n_test])
            train_names.extend(names[n_test:])
            print(f"    {version}: {len(names)} total → {len(names) - n_test} train / {n_test} test")
    else:
        # Legacy: v1=train, v2+v3=test
        train_names = [n for n in all_names if n.startswith("v1_")]
        test_names = [n for n in all_names if not n.startswith("v1_")]

    # Write index files
    with open(output_base / ".index.txt", "w") as f:
        f.write("\n".join(sorted(all_names)) + "\n")
    with open(output_base / "train.txt", "w") as f:
        f.write("\n".join(sorted(train_names)) + "\n")
    with open(output_base / "test.txt", "w") as f:
        f.write("\n".join(sorted(test_names)) + "\n")

    print(f"\nSKINL2 conversion complete:")
    print(f"  Train: {len(train_names)} samples")
    print(f"  Test:  {len(test_names)} samples")

    return {"train": train_names, "test": test_names}


def main():
    parser = argparse.ArgumentParser(description="Convert eval data to MoGe training format")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["woundsdb", "skinl2", "all"])
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Eval data root (default: output/eval_data/)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output root (default: data/dermdepth_train/)")
    parser.add_argument("--fov_deg", type=float, default=60.0,
                        help="Assumed field of view in degrees")
    parser.add_argument("--min_coverage", type=float, default=0.5,
                        help="Min GT coverage for WoundsDB scenes")
    parser.add_argument("--stratified", action="store_true",
                        help="Use stratified split (70/30 per version) for SKINL2")
    parser.add_argument("--test_fraction", type=float, default=0.3,
                        help="Test fraction for stratified split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stratified split")
    args = parser.parse_args()

    eval_root = Path(args.eval_dir) if args.eval_dir else PROJECT_ROOT / "output" / "eval_data"
    output_root = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "dermdepth_train"

    results = {}

    if args.dataset in ("woundsdb", "all"):
        print("=" * 60)
        print("Converting WoundsDB")
        print("=" * 60)
        results["woundsdb"] = convert_woundsdb(
            eval_root / "woundsdb",
            output_root / "woundsdb_moge",
            fov_deg=args.fov_deg,
            min_coverage=args.min_coverage,
        )

    if args.dataset in ("skinl2", "all"):
        print("\n" + "=" * 60)
        print("Converting SKINL2")
        print("=" * 60)
        results["skinl2"] = convert_skinl2(
            eval_root / "skinl2",
            output_root / "skinl2_moge",
            fov_deg=args.fov_deg,
            stratified=args.stratified,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    # Save summary
    summary_path = output_root / "conversion_summary.json"
    summary = {}
    for ds, res in results.items():
        summary[ds] = {
            "train_count": len(res["train"]),
            "test_count": len(res["test"]),
            "train_samples": res["train"],
            "test_samples": res["test"],
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
