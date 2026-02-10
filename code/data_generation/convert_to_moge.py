#!/usr/bin/env python3
"""
Convert S-SYNTH output to MoGe-2 training format.

S-SYNTH generates images in a nested directory structure:
    .../skin_XXX/hairModel_XXX/.../image.png, mask.png, depth.png

MoGe-2 expects a flat structure:
    dataset/instance_XXXXX/image.jpg, depth.png, meta.json

This script converts between these formats.

Usage:
    python convert_to_moge.py --input /path/to/ssynth/output --output /path/to/moge/dataset
"""

import os
import glob
import shutil
import argparse
import json
import numpy as np
from PIL import Image

# Add local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_utils import save_meta_json, load_depth_moge, validate_depth


def find_ssynth_instances(ssynth_output_dir):
    """
    Find all rendered instances in S-SYNTH output directory.

    Args:
        ssynth_output_dir: Root directory of S-SYNTH output

    Returns:
        List of tuples: (folder_path, has_image, has_mask, has_depth)
    """
    instances = []

    # Find all image.png files recursively
    image_files = glob.glob(f"{ssynth_output_dir}/**/image.png", recursive=True)

    for img_path in image_files:
        folder = os.path.dirname(img_path)
        has_image = True
        has_mask = os.path.exists(os.path.join(folder, "mask.png"))
        has_depth = os.path.exists(os.path.join(folder, "depth.png"))
        has_meta = os.path.exists(os.path.join(folder, "meta.json"))

        instances.append({
            'folder': folder,
            'has_image': has_image,
            'has_mask': has_mask,
            'has_depth': has_depth,
            'has_meta': has_meta
        })

    return instances


def convert_ssynth_to_moge(ssynth_output_dir, moge_dataset_dir,
                           require_depth=True, convert_to_jpg=True,
                           jpg_quality=95, validate=True):
    """
    Convert S-SYNTH nested output structure to flat MoGe training format.

    Args:
        ssynth_output_dir: Root directory containing S-SYNTH output
        moge_dataset_dir: Target directory for MoGe-format dataset
        require_depth: If True, skip instances without depth.png
        convert_to_jpg: If True, convert images to JPEG (MoGe default)
        jpg_quality: JPEG quality (1-100, default 95)
        validate: If True, validate depth values

    Returns:
        List of successfully converted instance names
    """
    os.makedirs(moge_dataset_dir, exist_ok=True)

    instances_found = find_ssynth_instances(ssynth_output_dir)
    print(f"Found {len(instances_found)} S-SYNTH instances")

    converted = []
    skipped = []
    errors = []

    for idx, instance_info in enumerate(instances_found):
        folder = instance_info['folder']

        # Skip if missing required depth
        if require_depth and not instance_info['has_depth']:
            skipped.append(folder)
            continue

        instance_name = f"instance_{idx:06d}"
        instance_dir = os.path.join(moge_dataset_dir, instance_name)

        try:
            os.makedirs(instance_dir, exist_ok=True)

            # Convert and copy image
            img_src = os.path.join(folder, "image.png")
            if convert_to_jpg:
                img = Image.open(img_src).convert('RGB')
                img.save(os.path.join(instance_dir, "image.jpg"), quality=jpg_quality)
            else:
                shutil.copy(img_src, os.path.join(instance_dir, "image.png"))

            # Copy depth (already in MoGe format from S-SYNTH)
            if instance_info['has_depth']:
                depth_src = os.path.join(folder, "depth.png")
                depth_dst = os.path.join(instance_dir, "depth.png")
                shutil.copy(depth_src, depth_dst)

                # Validate depth if requested
                if validate:
                    depth_array, _ = load_depth_moge(depth_dst)
                    validation = validate_depth(depth_array)
                    if validation['warnings']:
                        print(f"  Warning in {instance_name}: {validation['warnings']}")

            # Copy or generate meta.json
            meta_src = os.path.join(folder, "meta.json")
            meta_dst = os.path.join(instance_dir, "meta.json")
            if instance_info['has_meta']:
                shutil.copy(meta_src, meta_dst)
            else:
                # Generate default meta.json for S-SYNTH camera settings
                save_meta_json(meta_dst, fov_deg=75, width=1024, height=1024)

            # Copy mask as optional segmentation
            if instance_info['has_mask']:
                mask_src = os.path.join(folder, "mask.png")
                shutil.copy(mask_src, os.path.join(instance_dir, "mask.png"))

            # Save source path for reference
            with open(os.path.join(instance_dir, "source.txt"), 'w') as f:
                f.write(folder)

            converted.append(instance_name)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(instances_found)} instances...")

        except Exception as e:
            errors.append((folder, str(e)))
            print(f"  Error processing {folder}: {e}")

    # Create index file
    index_path = os.path.join(moge_dataset_dir, ".index.txt")
    with open(index_path, 'w') as f:
        f.write('\n'.join(converted))

    # Create summary file
    summary = {
        'total_found': len(instances_found),
        'converted': len(converted),
        'skipped_no_depth': len(skipped),
        'errors': len(errors),
        'source_directory': ssynth_output_dir
    }

    with open(os.path.join(moge_dataset_dir, "conversion_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nConversion complete:")
    print(f"  Total found: {len(instances_found)}")
    print(f"  Converted: {len(converted)}")
    print(f"  Skipped (no depth): {len(skipped)}")
    print(f"  Errors: {len(errors)}")
    print(f"\nDataset saved to: {moge_dataset_dir}")

    return converted


def validate_moge_dataset(dataset_dir, num_samples=20):
    """
    Validate MoGe-format dataset before training.

    Args:
        dataset_dir: Path to MoGe-format dataset
        num_samples: Number of samples to validate (0 for all)

    Returns:
        List of error messages (empty if valid)
    """
    index_path = os.path.join(dataset_dir, ".index.txt")
    if not os.path.exists(index_path):
        return ["Missing .index.txt file"]

    with open(index_path) as f:
        instances = [line.strip() for line in f if line.strip()]

    if num_samples > 0:
        instances = instances[:num_samples]

    errors = []

    for instance in instances:
        instance_dir = os.path.join(dataset_dir, instance)

        if not os.path.isdir(instance_dir):
            errors.append(f"Missing directory: {instance}")
            continue

        # Check required files
        required_files = ['depth.png', 'meta.json']
        image_files = ['image.jpg', 'image.png']

        has_image = any(os.path.exists(os.path.join(instance_dir, f)) for f in image_files)
        if not has_image:
            errors.append(f"Missing image in {instance}")

        for fname in required_files:
            if not os.path.exists(os.path.join(instance_dir, fname)):
                errors.append(f"Missing {fname} in {instance}")
                continue

        # Validate depth
        depth_path = os.path.join(instance_dir, 'depth.png')
        if os.path.exists(depth_path):
            try:
                depth, _ = load_depth_moge(depth_path)
                valid_depth = depth[np.isfinite(depth)]
                if len(valid_depth) == 0:
                    errors.append(f"No valid depth values in {instance}")
                elif valid_depth.min() < 0:
                    errors.append(f"Negative depth in {instance}")
            except Exception as e:
                errors.append(f"Failed to read depth in {instance}: {e}")

        # Validate intrinsics
        meta_path = os.path.join(instance_dir, 'meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if 'intrinsics' not in meta:
                    errors.append(f"Missing intrinsics in {instance}/meta.json")
                else:
                    K = np.array(meta['intrinsics'])
                    if K.shape != (3, 3):
                        errors.append(f"Invalid intrinsics shape in {instance}")
                    elif K[0, 0] < 100 or K[0, 0] > 2000:
                        errors.append(f"Unusual focal length in {instance}: {K[0, 0]}")
            except Exception as e:
                errors.append(f"Failed to read meta.json in {instance}: {e}")

    return errors


def create_train_val_split(dataset_dir, val_ratio=0.1, seed=42):
    """
    Create train/validation split files for MoGe training.

    Args:
        dataset_dir: Path to MoGe-format dataset
        val_ratio: Fraction of data for validation (default 0.1)
        seed: Random seed for reproducibility
    """
    import random

    index_path = os.path.join(dataset_dir, ".index.txt")
    with open(index_path) as f:
        instances = [line.strip() for line in f if line.strip()]

    random.seed(seed)
    random.shuffle(instances)

    split_idx = int(len(instances) * (1 - val_ratio))
    train_instances = instances[:split_idx]
    val_instances = instances[split_idx:]

    with open(os.path.join(dataset_dir, "train.txt"), 'w') as f:
        f.write('\n'.join(train_instances))

    with open(os.path.join(dataset_dir, "val.txt"), 'w') as f:
        f.write('\n'.join(val_instances))

    print(f"Split created: {len(train_instances)} train, {len(val_instances)} val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert S-SYNTH output to MoGe-2 training format"
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='S-SYNTH output directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='MoGe dataset output directory')
    parser.add_argument('--no-depth-required', action='store_true',
                        help='Include instances without depth')
    parser.add_argument('--keep-png', action='store_true',
                        help='Keep images as PNG instead of converting to JPEG')
    parser.add_argument('--jpg-quality', type=int, default=95,
                        help='JPEG quality (1-100, default 95)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip depth validation')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing dataset')
    parser.add_argument('--split', action='store_true',
                        help='Create train/val split after conversion')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation ratio for split (default 0.1)')

    args = parser.parse_args()

    if args.validate_only:
        print(f"Validating dataset at: {args.output}")
        errors = validate_moge_dataset(args.output, num_samples=0)
        if errors:
            print(f"\nFound {len(errors)} errors:")
            for err in errors[:20]:
                print(f"  - {err}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
        else:
            print("Dataset validation passed!")
    else:
        convert_ssynth_to_moge(
            args.input,
            args.output,
            require_depth=not args.no_depth_required,
            convert_to_jpg=not args.keep_png,
            jpg_quality=args.jpg_quality,
            validate=not args.no_validate
        )

        if args.split:
            create_train_val_split(args.output, val_ratio=args.val_ratio)
