#!/usr/bin/env python3
"""
Render a 3x3 grid of S-SYNTH images with diverse parameters including depth maps.

Grid layout:
- Rows: Light (5%), Medium (25%), Dark (45%) skin tones
- Columns: Different conditions with 50% hair probability

Each cell renders: image.png, mask.png, depth.png, meta.json
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set up mitsuba BEFORE importing other modules
import mitsuba as mi
mi.set_variant('scalar_spectral')

import config
import util
import depth_utils

# Output directory
OUTPUT_DIR = "/workspace/hector/ssynth-release/test_output/diverse_grid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the 3x3 grid parameters
# Rows: skin tones (melanin fraction)
SKIN_TONES = [
    (0.05, "Light"),
    (0.25, "Medium"),
    (0.45, "Dark"),
]

# Columns: different conditions
CONDITIONS = [
    {"blood": 0.002, "lesion_mat": 1, "lesion_scale": 1.5, "label": "Small Lesion"},
    {"blood": 0.02, "lesion_mat": 10, "lesion_scale": 2.0, "label": "Medium Lesion"},
    {"blood": 0.05, "lesion_mat": 17, "lesion_scale": 2.5, "label": "Large Lesion"},
]

# Fixed parameters
LESION_DIR = config.sDir_lesion_ver1
LIGHT_ID = 0  # rural_asphalt_road_4k
MODEL_ID = 30  # skin model
LESION_ID = 12  # lesion shape
TIME_POINT = 15  # lesion growth stage
CAMERA_HEIGHT = 15  # mm
SPP_IMAGE = 64  # samples per pixel for image (lower for faster testing)
SPP_MASK = 16


def render_single_sample(row, col, mel, skin_label, condition, has_hair, output_subdir):
    """Render a single sample with image, mask, and depth."""
    print(f"\n{'='*60}")
    print(f"Rendering [{row},{col}]: {skin_label} skin, {condition['label']}, hair={has_hair}")
    print(f"{'='*60}")

    os.makedirs(output_subdir, exist_ok=True)

    # Get hair model (-1 for no hair)
    hair_model = random.choice(util.get_l_hairModel()) if has_hair else -1
    hair_albedo_idx = random.choice([0, 1, 2]) if has_hair else 0

    # Get material names
    lesion_mat, light_name, hair_albedo = util.get_materials_names(
        condition['lesion_mat'], LIGHT_ID, hair_albedo_idx
    )

    # Calculate lesion offset based on scale
    lesion_offset = -2.0 - (condition['lesion_scale'] - 1.5) * 0.5

    print(f"  Melanin: {mel:.0%}")
    print(f"  Blood fraction: {condition['blood']}")
    print(f"  Lesion material: {lesion_mat}")
    print(f"  Lesion scale: {condition['lesion_scale']}")
    print(f"  Hair model: {hair_model}")

    # === Render Mask ===
    print("\n  Rendering mask...")
    start = time.time()
    mi.set_variant('scalar_spectral')

    scene_mask = util.render_image(
        MODEL_ID, hair_model, LESION_ID, lesion_mat,
        condition['blood'], mel, TIME_POINT, light_name, hair_albedo,
        IMAGE=False,
        lesion_directory=LESION_DIR,
        lesionScale=condition['lesion_scale'],
        yOffset_lesion=lesion_offset,
        verbose=False
    )
    sensor = util.get_sensor(id_origin_y=CAMERA_HEIGHT)
    mask_img = mi.render(scene_mask, sensor=sensor, spp=SPP_MASK)
    mi.util.write_bitmap(os.path.join(output_subdir, "mask.png"), mask_img)
    print(f"    Done in {time.time()-start:.1f}s")

    # === Render Image ===
    print("  Rendering image...")
    start = time.time()
    mi.set_variant('scalar_spectral')

    scene_rgb = util.render_image(
        MODEL_ID, hair_model, LESION_ID, lesion_mat,
        condition['blood'], mel, TIME_POINT, light_name, hair_albedo,
        IMAGE=True,
        lesion_directory=LESION_DIR,
        lesionScale=condition['lesion_scale'],
        yOffset_lesion=lesion_offset,
        verbose=False
    )
    sensor = util.get_sensor(id_origin_y=CAMERA_HEIGHT)
    rgb_img = mi.render(scene_rgb, sensor=sensor, spp=SPP_IMAGE)
    mi.util.write_bitmap(os.path.join(output_subdir, "image.png"), rgb_img)
    print(f"    Done in {time.time()-start:.1f}s")

    # === Render Depth ===
    print("  Rendering depth...")
    start = time.time()
    mi.set_variant('scalar_rgb')

    scene_depth = util.render_depth_scene(
        MODEL_ID, LESION_ID, TIME_POINT,
        lesion_directory=LESION_DIR,
        lesionScale=condition['lesion_scale'],
        yOffset_lesion=lesion_offset
    )
    sensor_depth = util.get_sensor_rgb(id_origin_y=CAMERA_HEIGHT)
    depth_img = mi.render(scene_depth, sensor=sensor_depth, spp=1)
    depth_array = np.array(depth_img)

    # Extract depth channel
    if depth_array.ndim == 3 and depth_array.shape[2] >= 4:
        depth_channel = depth_array[:, :, 3]
    else:
        depth_channel = depth_array[:, :, 0] if depth_array.ndim == 3 else depth_array

    # Save depth and metadata
    depth_utils.save_depth_moge(os.path.join(output_subdir, "depth.png"), depth_channel, CAMERA_HEIGHT)
    depth_utils.save_meta_json(
        os.path.join(output_subdir, "meta.json"),
        fov_deg=75, width=1024, height=1024,
        additional_meta={
            'camera_height': CAMERA_HEIGHT,
            'melanin': mel,
            'blood_fraction': condition['blood'],
            'lesion_material': lesion_mat,
            'lesion_scale': condition['lesion_scale'],
            'has_hair': has_hair,
            'hair_model': hair_model,
            'skin_tone': skin_label,
            'condition': condition['label']
        }
    )
    print(f"    Done in {time.time()-start:.1f}s")

    # Switch back to spectral
    mi.set_variant('scalar_spectral')

    return {
        'image_path': os.path.join(output_subdir, "image.png"),
        'mask_path': os.path.join(output_subdir, "mask.png"),
        'depth_path': os.path.join(output_subdir, "depth.png"),
        'params': {
            'melanin': mel,
            'skin_tone': skin_label,
            'condition': condition['label'],
            'has_hair': has_hair
        }
    }


def create_visualization(samples):
    """Create a 3x3 visualization with RGB, depth, and annotations."""
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    fig.suptitle("S-SYNTH Diverse Grid: RGB + Depth\n(Light/Medium/Dark Skin × Small/Medium/Large Lesion)",
                 fontsize=16, fontweight='bold')

    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            sample = samples[idx]

            # RGB image
            ax_rgb = axes[row, col * 2]
            img = Image.open(sample['image_path'])
            ax_rgb.imshow(img)

            params = sample['params']
            hair_str = "Hair" if params['has_hair'] else "No Hair"
            title = f"{params['skin_tone']} Skin\n{params['condition']}\n{hair_str}"
            ax_rgb.set_title(title, fontsize=9)
            ax_rgb.axis('off')

            # Add border based on hair
            border_color = 'green' if params['has_hair'] else 'gray'
            for spine in ax_rgb.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
                spine.set_visible(True)

            # Depth map
            ax_depth = axes[row, col * 2 + 1]
            depth, _ = depth_utils.load_depth_moge(sample['depth_path'])
            valid_depth = depth[np.isfinite(depth) & (depth > 0)]
            if len(valid_depth) > 0:
                vmin, vmax = valid_depth.min(), valid_depth.max()
            else:
                vmin, vmax = 0, 20

            im = ax_depth.imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
            ax_depth.set_title(f"Depth ({vmin:.1f}-{vmax:.1f}mm)", fontsize=9)
            ax_depth.axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Depth (mm)')

    # Legend
    fig.text(0.5, 0.02, "Green border = Has Hair | Gray border = No Hair",
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.04, 0.9, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "grid_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("S-SYNTH Diverse Grid Rendering with Depth")
    print("=" * 60)

    random.seed(42)  # For reproducibility

    samples = []
    total_start = time.time()

    for row, (mel, skin_label) in enumerate(SKIN_TONES):
        for col, condition in enumerate(CONDITIONS):
            # 50% chance of having hair
            has_hair = random.random() < 0.5

            output_subdir = os.path.join(OUTPUT_DIR, f"r{row}_c{col}_{skin_label}_{condition['label'].replace(' ', '_')}")

            sample = render_single_sample(row, col, mel, skin_label, condition, has_hair, output_subdir)
            samples.append(sample)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total rendering time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")

    # Create visualization
    print("\nCreating visualization...")
    create_visualization(samples)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
