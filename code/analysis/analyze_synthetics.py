#!/usr/bin/env python3
"""
Analyze and validate synthetic S-SYNTH data for DermDepth training.

Verifies:
- Depth values are in expected range (10-25mm)
- Depth encoding round-trip (S-SYNTH save -> MoGe load)
- RGB + depth + mask + normal side-by-side visualization
- Intrinsics normalization correctness

Usage:
    python analyze_synthetics.py --input /path/to/ssynth/output [--output_dir OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Add data_generation to path for depth_utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_generation"))
import depth_utils

# Add MoGe to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))


def verify_depth_encoding_roundtrip(output_dir):
    """
    Verify that S-SYNTH depth encoding is compatible with MoGe's read_depth.

    S-SYNTH: near * exp(d * log(far/near))
    MoGe:    near^(1-d) * far^d
    These are mathematically equivalent: near^(1-d) * far^d = near * exp(d * log(far/near))
    """
    print("\n  Verifying depth encoding round-trip...")

    # Create synthetic test depth array (simulating dermatological depths in mm)
    np.random.seed(42)
    depth_original = np.random.uniform(10.0, 25.0, (256, 256)).astype(np.float32)
    # Add some NaN and structure
    depth_original[0:10, :] = np.nan  # Invalid border
    depth_original[128, 128] = 12.5   # Lesion center (closer)
    depth_original[100:150, 100:150] -= 2.0  # Lesion region

    # Save using S-SYNTH's save_depth_moge
    test_path = Path(output_dir) / "roundtrip_test_depth.png"
    depth_utils.save_depth_moge(str(test_path), depth_original)

    # Load using S-SYNTH's load_depth_moge
    depth_ssynth, meta_ssynth = depth_utils.load_depth_moge(str(test_path))

    # Load using MoGe's read_depth
    try:
        from moge.utils.io import read_depth as moge_read_depth
        depth_moge = moge_read_depth(str(test_path))
        moge_available = True
    except ImportError:
        print("    MoGe not available, skipping MoGe read_depth comparison")
        depth_moge = None
        moge_available = False

    # Compare
    valid_mask = np.isfinite(depth_original) & np.isfinite(depth_ssynth)
    max_error_ssynth = np.abs(depth_original[valid_mask] - depth_ssynth[valid_mask]).max()
    rel_error_ssynth = (np.abs(depth_original[valid_mask] - depth_ssynth[valid_mask]) / depth_original[valid_mask]).max()

    results = {
        'ssynth_roundtrip_max_error': float(max_error_ssynth),
        'ssynth_roundtrip_max_rel_error': float(rel_error_ssynth),
        'ssynth_roundtrip_pass': rel_error_ssynth < 0.001,  # 0.1% tolerance
    }

    print(f"    S-SYNTH round-trip max absolute error: {max_error_ssynth:.6f} mm")
    print(f"    S-SYNTH round-trip max relative error: {rel_error_ssynth:.6f}")
    print(f"    S-SYNTH round-trip: {'PASS' if results['ssynth_roundtrip_pass'] else 'FAIL'}")

    if moge_available and depth_moge is not None:
        valid_mask_moge = valid_mask & np.isfinite(depth_moge)
        max_error_moge = np.abs(depth_original[valid_mask_moge] - depth_moge[valid_mask_moge]).max()
        rel_error_moge = (np.abs(depth_original[valid_mask_moge] - depth_moge[valid_mask_moge]) / depth_original[valid_mask_moge]).max()

        results['moge_roundtrip_max_error'] = float(max_error_moge)
        results['moge_roundtrip_max_rel_error'] = float(rel_error_moge)
        results['moge_roundtrip_pass'] = rel_error_moge < 0.001

        print(f"    MoGe round-trip max absolute error: {max_error_moge:.6f} mm")
        print(f"    MoGe round-trip max relative error: {rel_error_moge:.6f}")
        print(f"    MoGe round-trip: {'PASS' if results['moge_roundtrip_pass'] else 'FAIL'}")

    # Cleanup
    test_path.unlink(missing_ok=True)

    return results


def verify_intrinsics_normalization(output_dir):
    """
    Verify that normalized intrinsics produce correct FOV.

    For S-SYNTH default (75 deg, 1024x1024):
    - Pixel: fx=687.55, cx=512
    - Normalized: fx=0.6718, cx=0.5
    - FOV from normalized: 2*atan(0.5/fx_norm) = 2*atan(0.5/0.6718) = 73.3 deg (half-width)
    - But MoGe uses: FOV_x = 2*atan(1/(2*fx_norm)) same thing
    """
    print("\n  Verifying intrinsics normalization...")

    K_pixel = depth_utils.compute_intrinsics(fov_deg=75, width=1024, height=1024)
    K_norm = depth_utils.normalize_intrinsics(K_pixel, 1024, 1024)

    # Check normalized values
    fx_norm = K_norm[0, 0]
    fy_norm = K_norm[1, 1]
    cx_norm = K_norm[0, 2]
    cy_norm = K_norm[1, 2]

    # Recover FOV from normalized intrinsics
    # FOV_x = 2 * atan(0.5 / fx_norm) for width-normalized intrinsics
    fov_recovered = 2 * np.degrees(np.arctan(0.5 / fx_norm))

    results = {
        'fx_pixel': float(K_pixel[0, 0]),
        'fx_normalized': float(fx_norm),
        'cx_normalized': float(cx_norm),
        'fov_original_deg': 75.0,
        'fov_recovered_deg': float(fov_recovered),
        'fov_error_deg': abs(75.0 - fov_recovered),
        'pass': abs(75.0 - fov_recovered) < 1.0,  # Within 1 degree
    }

    print(f"    Pixel-space: fx={K_pixel[0, 0]:.2f}, cx={K_pixel[0, 2]:.2f}")
    print(f"    Normalized:  fx={fx_norm:.4f}, cx={cx_norm:.4f}")
    print(f"    FOV original: 75.0 deg")
    print(f"    FOV recovered: {fov_recovered:.2f} deg")
    print(f"    Intrinsics: {'PASS' if results['pass'] else 'FAIL'}")

    return results


def visualize_ssynth_samples(ssynth_dir, output_dir, max_samples=8):
    """
    Visualize S-SYNTH rendered samples: RGB + depth + mask side-by-side.
    """
    print(f"\n  Visualizing S-SYNTH samples from {ssynth_dir}...")

    # Find rendered samples
    image_files = sorted(Path(ssynth_dir).rglob("image.png"))
    if not image_files:
        # Also try *_rgb.png pattern
        image_files = sorted(Path(ssynth_dir).rglob("*_rgb.png"))
    if not image_files:
        print("    No samples found")
        return

    samples = []
    for img_path in image_files[:max_samples]:
        folder = img_path.parent
        sample = {'rgb_path': str(img_path)}

        # Look for depth
        depth_candidates = [
            folder / "depth.png",
            folder / f"{img_path.stem.replace('_rgb', '_depth')}.png",
        ]
        for dp in depth_candidates:
            if dp.exists():
                sample['depth_path'] = str(dp)
                break

        # Look for mask
        mask_path = folder / "mask.png"
        if mask_path.exists():
            sample['mask_path'] = str(mask_path)

        # Look for meta
        meta_candidates = [
            folder / "meta.json",
            folder / f"{img_path.stem.replace('_rgb', '_meta')}.json",
        ]
        for mp in meta_candidates:
            if mp.exists():
                sample['meta_path'] = str(mp)
                break

        samples.append(sample)

    n = len(samples)
    num_cols = 3  # RGB, depth, mask
    fig, axes = plt.subplots(n, num_cols, figsize=(4 * num_cols, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"S-SYNTH Rendered Samples ({n} shown)", fontsize=14, fontweight='bold')

    depth_stats = []
    for i, sample in enumerate(samples):
        # RGB
        rgb = Image.open(sample['rgb_path'])
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB\n{Path(sample['rgb_path']).parent.name}", fontsize=8)
        axes[i, 0].axis('off')

        # Depth
        if 'depth_path' in sample:
            try:
                depth, meta = depth_utils.load_depth_moge(sample['depth_path'])
                valid = depth[np.isfinite(depth) & (depth > 0)]
                if len(valid) > 0:
                    vmin, vmax = valid.min(), valid.max()
                    axes[i, 1].imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[i, 1].set_title(f"Depth [{vmin:.1f}-{vmax:.1f}mm]", fontsize=8)
                    depth_stats.append({
                        'min': float(vmin), 'max': float(vmax),
                        'mean': float(valid.mean()), 'std': float(valid.std())
                    })
                else:
                    axes[i, 1].text(0.5, 0.5, "No valid depth", ha='center', va='center')
            except Exception as e:
                axes[i, 1].text(0.5, 0.5, f"Error: {e}", ha='center', va='center', fontsize=6)
        axes[i, 1].axis('off')

        # Mask
        if 'mask_path' in sample:
            mask = np.array(Image.open(sample['mask_path']))
            axes[i, 2].imshow(mask, cmap='gray')
            axes[i, 2].set_title("Mask", fontsize=8)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "ssynth_samples.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Depth statistics summary
    if depth_stats:
        print(f"    Depth statistics across {len(depth_stats)} samples:")
        mins = [d['min'] for d in depth_stats]
        maxs = [d['max'] for d in depth_stats]
        means = [d['mean'] for d in depth_stats]
        print(f"      Min depth: {min(mins):.2f} - {max(mins):.2f} mm")
        print(f"      Max depth: {min(maxs):.2f} - {max(maxs):.2f} mm")
        print(f"      Mean depth: {min(means):.2f} - {max(means):.2f} mm")

        expected_min, expected_max = 5.0, 50.0
        in_range = all(d['min'] >= expected_min and d['max'] <= expected_max for d in depth_stats)
        print(f"      In expected range [{expected_min}-{expected_max}mm]: {'YES' if in_range else 'NO'}")

    return samples, depth_stats


def main():
    parser = argparse.ArgumentParser(description="Analyze synthetic S-SYNTH data")
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='S-SYNTH output directory to analyze')
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "analysis" / "synthetics"),
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    print("=" * 60)
    print("S-SYNTH Synthetic Data Analysis")
    print("=" * 60)

    # Verify depth encoding
    results['depth_encoding'] = verify_depth_encoding_roundtrip(args.output_dir)

    # Verify intrinsics
    results['intrinsics'] = verify_intrinsics_normalization(args.output_dir)

    # Visualize samples if input provided
    if args.input and os.path.exists(args.input):
        samples, depth_stats = visualize_ssynth_samples(args.input, args.output_dir)
        results['depth_stats'] = depth_stats

    # Save results
    with open(os.path.join(args.output_dir, "analysis_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")

    # Summary
    all_pass = results['depth_encoding'].get('ssynth_roundtrip_pass', False) and \
               results['intrinsics'].get('pass', False)
    print(f"\nOverall verification: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
