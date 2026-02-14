#!/usr/bin/env python3
"""
Generate DermDepth training dataset from S-SYNTH.

Stratified parameter sampling across Fitzpatrick skin tone groups:
- Group I-II (light): melanin 0.01-0.05
- Group III-IV (medium): melanin 0.05-0.15
- Group V-VI (dark): melanin 0.15-0.45

Calls render_sample_with_depth() from render_extended_features.py.
Supports parallel rendering with multiple processes.

Usage:
    python generate_dermdepth_dataset.py --output /path/to/output --num_samples 10000
"""

import os
import sys
import json
import argparse
import random
import time
import itertools
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

# Add local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Fitzpatrick skin tone groups mapped to melanin fractions.
# These are the STRING values matching actual S-SYNTH .spd filenames.
# Total 131 melanin levels available (0.005 to 2.0).
# Groupings based on ITA (Individual Typology Angle) mapping from S-SYNTH paper.
MELANIN_LEVELS = [
    '0.005', '0.01', '0.015', '0.02', '0.025', '0.03', '0.031', '0.033',
    '0.035', '0.039', '0.04', '0.041', '0.042', '0.05', '0.06', '0.061',
    '0.07', '0.08', '0.09', '0.1', '0.11', '0.111', '0.116', '0.12',
    '0.123', '0.13', '0.14', '0.141', '0.142', '0.144', '0.15', '0.153',
    '0.155', '0.159', '0.16', '0.17', '0.18', '0.19', '0.2', '0.207',
    '0.21', '0.217', '0.22', '0.23', '0.24', '0.25', '0.254', '0.258',
    '0.26', '0.27', '0.28', '0.29', '0.297', '0.3', '0.31', '0.32',
    '0.33', '0.34', '0.341', '0.35', '0.36', '0.37', '0.38', '0.387',
    '0.39', '0.391', '0.4', '0.41', '0.42', '0.424', '0.43', '0.44',
    '0.45', '0.46', '0.47', '0.48', '0.49', '0.5', '0.51', '0.52',
    '0.53', '0.54', '0.55', '0.56', '0.57', '0.58', '0.59', '0.6',
    '0.61', '0.62', '0.63', '0.64', '0.65', '0.66', '0.67', '0.68',
    '0.69', '0.7', '0.71', '0.72', '0.73', '0.74', '0.75', '0.76',
    '0.77', '0.78', '0.79', '0.8', '0.81', '0.82', '0.83', '0.84',
    '0.85', '0.86', '0.87', '0.88', '0.89', '0.9', '0.91', '0.92',
    '0.93', '0.94', '0.95', '0.96', '0.97', '0.98', '0.99', '1.0',
    '1.1', '1.5', '2.0',
]

FITZPATRICK_GROUPS = {
    'I-II':   [m for m in MELANIN_LEVELS if float(m) <= 0.05],    # 14 levels (light)
    'III-IV': [m for m in MELANIN_LEVELS if 0.05 < float(m) <= 0.25],  # 33 levels (medium)
    'V-VI':   [m for m in MELANIN_LEVELS if float(m) > 0.25],     # 84 levels (dark)
}

# Blood fractions
BLOOD_FRACTIONS = [0.002, 0.005, 0.02, 0.05]

# Available skin model IDs (100 models: 0-99)
SKIN_MODELS = list(range(0, 100))

# Available lesion IDs and time points
LESION_IDS_VER1 = list(range(1, 21))  # 20 lesions
LESION_TIMEPOINTS_VER1 = [2, 5, 10, 15, 20, 25, 30]  # 7 timepoints

LESION_IDS_VER0 = list(range(1, 21))
LESION_TIMEPOINTS_VER0 = [10, 20, 30, 40, 50]

# Lesion materials (18 HbO2-based materials, verified against actual .spd files)
LESION_MATERIALS = [
    'HbO2x0.1Epix0.025', 'HbO2x0.1Epix0.05', 'HbO2x0.1Epix0.1',
    'HbO2x0.1Epix0.15', 'HbO2x0.1Epix0.25', 'HbO2x0.1Epix0.4',
    'HbO2x0.5Epix0.025', 'HbO2x0.5Epix0.05', 'HbO2x0.5Epix0.1',
    'HbO2x0.5Epix0.15', 'HbO2x0.5Epix0.25', 'HbO2x0.5Epix0.4',
    'HbO2x1.0Epix0.025', 'HbO2x1.0Epix0.05', 'HbO2x1.0Epix0.1',
    'HbO2x1.0Epix0.15', 'HbO2x1.0Epix0.25', 'HbO2x1.0Epix0.4',
]

# HDRI lighting environments (19 lights, verified against actual .exr files)
HDRI_LIGHTS = [
    'bathroom_4k', 'bush_restaurant_4k', 'comfy_cafe_4k',
    'floral_tent_4k', 'graffiti_shelter_4k', 'hospital_room_4k',
    'kiara_interior_4k', 'lapa_4k', 'lythwood_room_4k',
    'reading_room_4k', 'reinforced_concrete_01_4k',
    'rural_asphalt_road_4k', 'school_hall_4k', 'st_fagans_interior_4k',
    'surgery_4k', 'veranda_4k', 'vintage_measuring_lab_4k',
    'vulture_hide_4k', 'yaris_interior_garage_4k',
]

# Camera tilt: continuous sampling in [0, 30] degrees with random azimuth.
# Most real derm photos are roughly top-down with slight tilt,
# so we use beta(2, 5) distribution peaking ~8-10° with a tail to 30°.
MAX_TILT_DEG = 30


def sample_lesion_config(rng):
    """Sample a random lesion configuration."""
    version = rng.choice(['ver0', 'ver1'])
    if version == 'ver1':
        lesion_id = rng.choice(LESION_IDS_VER1)
        time_point = rng.choice(LESION_TIMEPOINTS_VER1)
    else:
        lesion_id = rng.choice(LESION_IDS_VER0)
        time_point = rng.choice(LESION_TIMEPOINTS_VER0)

    return {
        'lesion_id': lesion_id,
        'time_point': time_point,
        'lesion_mat': rng.choice(LESION_MATERIALS),
        'scale': rng.uniform(0.8, 2.0),
        'position': (rng.uniform(-5, 5), rng.uniform(-5, 5)),
        'version': version,
        'y_offset': -2,
    }


def sample_parameters(num_samples, seed=42):
    """
    Generate stratified parameter combinations for dataset.

    Ensures equal representation across Fitzpatrick groups.
    """
    rng = np.random.default_rng(seed)
    samples = []

    # Equal samples per Fitzpatrick group
    samples_per_group = num_samples // len(FITZPATRICK_GROUPS)
    remainder = num_samples % len(FITZPATRICK_GROUPS)

    for group_idx, (group_name, melanin_range) in enumerate(FITZPATRICK_GROUPS.items()):
        n = samples_per_group + (1 if group_idx < remainder else 0)

        for i in range(n):
            melanin = rng.choice(melanin_range)
            blood_frac = rng.choice(BLOOD_FRACTIONS)
            model_id = rng.choice(SKIN_MODELS)
            light_name = rng.choice(HDRI_LIGHTS)
            # Continuous camera tilt: magnitude from beta distribution, uniform azimuth
            tilt_mag = float(rng.beta(2, 5) * MAX_TILT_DEG)  # peaks ~8-10°, tail to 30°
            tilt_dir = float(rng.uniform(0, 360))  # random azimuth
            tilt_x = tilt_mag * np.cos(np.radians(tilt_dir))
            tilt_z = tilt_mag * np.sin(np.radians(tilt_dir))
            angle = (round(tilt_x, 2), round(tilt_z, 2))
            skin_scale = rng.uniform(0.8, 1.5)
            # ~25% chance no hair, otherwise random from 100 models
            if rng.random() < 0.25:
                hair_model = -1
            else:
                hair_model = int(rng.integers(0, 100))

            # Number of lesions (1-3)
            num_lesions = rng.choice([1, 1, 2, 2, 3])
            lesion_configs = [sample_lesion_config(rng) for _ in range(num_lesions)]
            # Ensure center lesion is at (0,0)
            lesion_configs[0]['position'] = (0, 0)

            camera_height = rng.uniform(12, 20)
            spp = 128

            samples.append({
                'sample_id': len(samples),
                'fitzpatrick_group': group_name,
                'melanin': melanin,
                'blood_frac': blood_frac,
                'model_id': model_id,
                'light_name': light_name,
                'angle_offset': angle,
                'skin_scale': skin_scale,
                'hair_model': hair_model,
                'camera_height': camera_height,
                'lesion_configs': lesion_configs,
                'spp': spp,
            })

    # Shuffle to mix groups
    rng.shuffle(samples)
    return samples


def render_single_sample(params, output_dir, fov=75):
    """
    Render a single sample. Designed for multiprocessing.

    Returns dict with sample info and status.
    """
    import mitsuba as mi

    sample_id = params['sample_id']
    sample_name = f"sample_{sample_id:06d}"
    sample_dir = os.path.join(output_dir, sample_name)

    # Skip if already rendered
    if os.path.exists(os.path.join(sample_dir, "image.png")) and \
       os.path.exists(os.path.join(sample_dir, "depth.png")):
        return {'sample_id': sample_id, 'status': 'skipped', 'name': sample_name}

    try:
        from render_extended_features import render_sample_with_depth

        result = render_sample_with_depth(
            model_id=params['model_id'],
            lesion_configs=params['lesion_configs'],
            melanin=params['melanin'],
            blood_frac=params['blood_frac'],
            light_name=params['light_name'],
            output_dir=sample_dir,
            sample_name="render",
            camera_height=params['camera_height'],
            fov=fov,
            angle_offset=tuple(params['angle_offset']),
            skin_scale=params['skin_scale'],
            hair_model=params['hair_model'],
            spp=params['spp'],
        )

        # Rename files to MoGe-compatible names
        for src_name, dst_name in [
            (f"render_rgb.png", "image.png"),
            (f"render_depth.png", "depth.png"),
            (f"render_meta.json", "meta.json"),
        ]:
            src = os.path.join(sample_dir, src_name)
            dst = os.path.join(sample_dir, dst_name)
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)

        # Save generation parameters (convert numpy types for JSON)
        params_clean = {}
        for k, v in params.items():
            if isinstance(v, (np.integer,)):
                params_clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                params_clean[k] = float(v)
            elif isinstance(v, (tuple, list)):
                params_clean[k] = [float(x) if isinstance(x, (np.floating,)) else
                                   int(x) if isinstance(x, (np.integer,)) else x for x in v]
            else:
                params_clean[k] = v
        params_clean['angle_offset'] = list(params['angle_offset'])
        with open(os.path.join(sample_dir, "generation_params.json"), 'w') as f:
            json.dump(params_clean, f, indent=2, default=str)

        return {
            'sample_id': sample_id,
            'status': 'success',
            'name': sample_name,
            'depth_validation': result.get('depth_validation', {}),
        }

    except Exception as e:
        return {
            'sample_id': sample_id,
            'status': 'error',
            'name': sample_name,
            'error': str(e),
        }


def generate_dataset(output_dir, num_samples, seed=42, num_workers=1, fov=75):
    """Generate the full DermDepth training dataset."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} samples...")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {num_workers}")
    print(f"  Seed: {seed}")

    # Sample parameters
    params_list = sample_parameters(num_samples, seed=seed)

    # Save parameter manifest
    manifest_path = os.path.join(output_dir, "generation_manifest.json")
    with open(manifest_path, 'w') as f:
        manifest = {
            'num_samples': num_samples,
            'seed': seed,
            'fitzpatrick_distribution': {
                group: sum(1 for p in params_list if p['fitzpatrick_group'] == group)
                for group in FITZPATRICK_GROUPS
            },
        }
        json.dump(manifest, f, indent=2)
    print(f"  Fitzpatrick distribution: {manifest['fitzpatrick_distribution']}")

    # Render samples
    start_time = time.time()
    results = []

    if num_workers <= 1:
        # Sequential rendering
        for i, params in enumerate(params_list):
            result = render_single_sample(params, output_dir, fov=fov)
            results.append(result)
            status = result['status']
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}] {result['name']}: {status} "
                  f"({rate:.1f} samples/min, ETA: {eta/60:.0f}min)")
    else:
        # Parallel rendering
        render_fn = partial(render_single_sample, output_dir=output_dir, fov=fov)
        with Pool(num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(render_fn, params_list)):
                results.append(result)
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                    print(f"  [{i+1}/{num_samples}] {rate:.1f} samples/min")

    # Summary
    total_time = time.time() - start_time
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    summary = {
        'total_requested': num_samples,
        'success': success,
        'skipped': skipped,
        'errors': errors,
        'total_time_min': total_time / 60,
        'rate_per_min': num_samples / (total_time / 60) if total_time > 0 else 0,
    }

    print(f"\nGeneration complete:")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Time: {total_time/60:.1f} min")

    # Create index file for MoGe compatibility
    successful_names = sorted([r['name'] for r in results if r['status'] in ('success', 'skipped')])
    index_path = os.path.join(output_dir, ".index.txt")
    with open(index_path, 'w') as f:
        f.write('\n'.join(successful_names))
    print(f"  Index: {len(successful_names)} entries")

    # Save summary
    with open(os.path.join(output_dir, "generation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate DermDepth training dataset")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for generated dataset')
    parser.add_argument('--num_samples', '-n', type=int, default=10000,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', '-j', type=int, default=1,
                        help='Number of parallel workers (default: 1, sequential)')
    parser.add_argument('--fov', type=float, default=75,
                        help='Camera FOV in degrees (default: 75)')
    parser.add_argument('--params_only', action='store_true',
                        help='Only generate parameter manifest, do not render')
    args = parser.parse_args()

    if args.params_only:
        params = sample_parameters(args.num_samples, seed=args.seed)
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "generation_params.json"), 'w') as f:
            # Convert numpy types for JSON
            clean = []
            for p in params:
                cp = {k: v for k, v in p.items()}
                cp['angle_offset'] = list(cp['angle_offset'])
                clean.append(cp)
            json.dump(clean, f, indent=2)
        print(f"Parameters saved to {args.output}/generation_params.json")
    else:
        generate_dataset(args.output, args.num_samples, args.seed, args.num_workers, args.fov)


if __name__ == "__main__":
    main()
