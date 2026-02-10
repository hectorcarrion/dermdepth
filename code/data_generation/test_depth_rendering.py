#!/usr/bin/env python3
"""
Test script for depth rendering pipeline.

Creates synthetic 3D geometry (sphere on a plane, simulating lesion on skin)
and tests the depth rendering + MoGe-compatible encoding.

Usage:
    conda activate ssynth
    python test_depth_rendering.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set up mitsuba
import mitsuba as mi

# Import depth utilities
from depth_utils import (
    save_depth_moge, load_depth_moge, save_meta_json,
    compute_intrinsics, validate_depth
)

# Output directory
OUTPUT_DIR = "/workspace/hector/ssynth-release/test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_test_scene_rgb():
    """
    Create a test scene with a sphere (lesion) on a plane (skin) for RGB rendering.
    Uses scalar_rgb variant.
    """
    mi.set_variant('scalar_rgb')

    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 8},

        # Plane representing skin surface at y=0
        'skin': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([10, 1, 10]).rotate([1, 0, 0], -90),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.8, 0.6, 0.5]}  # Skin-like color
            }
        },

        # Sphere representing lesion, slightly above/on the skin
        'lesion': {
            'type': 'sphere',
            'center': [0, 0.5, 0],
            'radius': 1.5,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.4, 0.2, 0.2]}  # Darker lesion color
            }
        },

        # Another smaller bump
        'lesion2': {
            'type': 'sphere',
            'center': [3, 0.3, 2],
            'radius': 0.8,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.5, 0.3, 0.25]}
            }
        },

        # Environment light
        'light': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': 1.0}
        }
    })
    return scene


def create_test_scene_depth():
    """
    Create a test scene for depth rendering using AOV integrator.
    Uses scalar_rgb variant with AOV for depth capture.
    """
    mi.set_variant('scalar_rgb')

    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'aov',
            'aovs': 'dd.y:depth',
            'nested': {'type': 'direct'}
        },

        # Plane representing skin surface at y=0
        'skin': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([10, 1, 10]).rotate([1, 0, 0], -90),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}}
        },

        # Sphere representing lesion
        'lesion': {
            'type': 'sphere',
            'center': [0, 0.5, 0],
            'radius': 1.5,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}}
        },

        # Another smaller bump
        'lesion2': {
            'type': 'sphere',
            'center': [3, 0.3, 2],
            'radius': 0.8,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}}
        },

        # Light for depth pass
        'light': {
            'type': 'directional',
            'direction': [0, -1, 0],
            'irradiance': {'type': 'rgb', 'value': 1.0}
        }
    })
    return scene


def get_test_sensor(camera_height=15, fov=75, width=512, height=512):
    """
    Create a sensor looking down from above (similar to S-SYNTH setup).
    """
    sensor = mi.load_dict({
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(
            target=[0, 0, 0],
            origin=[0, camera_height, 0],
            up=[0, 0, 1]
        ),
        'fov': fov,
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height,
        }
    })
    return sensor


def render_and_save():
    """
    Render RGB image and depth map, save in MoGe-compatible format.
    """
    print("=" * 60)
    print("Testing S-SYNTH Depth Rendering Pipeline")
    print("=" * 60)

    camera_height = 15  # mm (same as S-SYNTH default)
    fov = 75  # degrees
    width, height = 512, 512  # Lower res for testing

    # --- Render RGB image ---
    print("\n1. Rendering RGB image...")
    mi.set_variant('scalar_rgb')
    scene_rgb = create_test_scene_rgb()
    sensor = get_test_sensor(camera_height, fov, width, height)

    image_rgb = mi.render(scene_rgb, sensor=sensor, spp=64)
    image_array = np.array(image_rgb)[:, :, :3]  # RGB only

    # Save RGB image
    image_path = os.path.join(OUTPUT_DIR, "image.png")
    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(image_path)
    print(f"   Saved: {image_path}")

    # --- Render depth ---
    print("\n2. Rendering depth map...")
    mi.set_variant('scalar_rgb')
    scene_depth = create_test_scene_depth()
    sensor = get_test_sensor(camera_height, fov, width, height)

    depth_image = mi.render(scene_depth, sensor=sensor, spp=1)
    depth_array = np.array(depth_image)

    print(f"   Depth image shape: {depth_array.shape}")
    print(f"   Depth image channels: {depth_array.shape[2] if depth_array.ndim == 3 else 1}")

    # Extract depth channel (channel 3 for AOV, after RGB)
    if depth_array.ndim == 3 and depth_array.shape[2] >= 4:
        depth_channel = depth_array[:, :, 3]
        print("   Using channel 3 (AOV depth)")
    else:
        depth_channel = depth_array[:, :, 0] if depth_array.ndim == 3 else depth_array
        print("   Using channel 0 (fallback)")

    # --- Save depth in MoGe format ---
    print("\n3. Saving depth in MoGe-compatible format...")
    depth_path = os.path.join(OUTPUT_DIR, "depth.png")
    save_depth_moge(depth_path, depth_channel, camera_height)
    print(f"   Saved: {depth_path}")

    # --- Save camera intrinsics ---
    meta_path = os.path.join(OUTPUT_DIR, "meta.json")
    save_meta_json(meta_path, fov_deg=fov, width=width, height=height,
                   additional_meta={'camera_height': camera_height, 'depth_unit_mm': 1.0})
    print(f"   Saved: {meta_path}")

    # --- Validate depth ---
    print("\n4. Validating depth data...")
    validation = validate_depth(depth_channel)
    print(f"   Valid pixels: {validation['valid_pixels']}/{validation['total_pixels']} ({validation['valid_ratio']*100:.1f}%)")
    if 'min' in validation:
        print(f"   Depth range: {validation['min']:.2f} - {validation['max']:.2f} mm")
        print(f"   Mean depth: {validation['mean']:.2f} mm")
    if validation['warnings']:
        for warn in validation['warnings']:
            print(f"   WARNING: {warn}")

    # --- Test loading depth back ---
    print("\n5. Testing depth load/decode...")
    depth_loaded, metadata = load_depth_moge(depth_path)
    print(f"   Loaded depth shape: {depth_loaded.shape}")
    print(f"   Metadata: near={metadata.get('near', 'N/A'):.4f}, far={metadata.get('far', 'N/A'):.4f}")

    # Check round-trip accuracy
    valid_mask = np.isfinite(depth_channel) & (depth_channel > 0)
    if valid_mask.any():
        error = np.abs(depth_loaded[valid_mask] - depth_channel[valid_mask])
        print(f"   Round-trip error: max={error.max():.6f}, mean={error.mean():.6f}")

    # --- Create visualization ---
    print("\n6. Creating visualization...")
    create_visualization(image_array, depth_channel, depth_loaded, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"Test complete! Outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)

    return image_array, depth_channel


def create_visualization(image_rgb, depth_raw, depth_loaded, output_dir):
    """
    Create a side-by-side visualization of RGB, depth, and loaded depth.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # RGB image
    axes[0, 0].imshow(np.clip(image_rgb, 0, 1))
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')

    # Raw depth
    valid_depth = depth_raw[np.isfinite(depth_raw) & (depth_raw > 0)]
    if len(valid_depth) > 0:
        vmin, vmax = valid_depth.min(), valid_depth.max()
    else:
        vmin, vmax = 0, 1

    im1 = axes[0, 1].imshow(depth_raw, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Raw Depth (range: {vmin:.1f}-{vmax:.1f} mm)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Loaded depth (after encode/decode)
    im2 = axes[1, 0].imshow(depth_loaded, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Loaded Depth (after MoGe encode/decode)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Depth as 3D surface
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')

    # Subsample for 3D plot
    step = 8
    h, w = depth_raw.shape
    x = np.arange(0, w, step)
    z = np.arange(0, h, step)
    X, Z = np.meshgrid(x, z)
    Y = depth_raw[::step, ::step]

    # Mask invalid values
    Y_masked = np.ma.masked_where(~np.isfinite(Y), Y)

    ax3d.plot_surface(X, Z, -Y_masked, cmap='viridis', alpha=0.8)
    ax3d.set_xlabel('X (pixels)')
    ax3d.set_ylabel('Z (pixels)')
    ax3d.set_zlabel('Depth (mm)')
    ax3d.set_title('3D Depth Surface')

    plt.tight_layout()

    vis_path = os.path.join(output_dir, "visualization.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {vis_path}")

    # Also save depth as a normalized grayscale for easy viewing
    depth_norm = (depth_raw - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    depth_gray_path = os.path.join(output_dir, "depth_grayscale.png")
    Image.fromarray(depth_gray).save(depth_gray_path)
    print(f"   Saved: {depth_gray_path}")


def test_intrinsics():
    """
    Test camera intrinsics computation.
    """
    print("\n" + "=" * 60)
    print("Testing Camera Intrinsics")
    print("=" * 60)

    # S-SYNTH default: 75 deg FOV, 1024x1024
    K = compute_intrinsics(fov_deg=75, width=1024, height=1024)
    print(f"\nS-SYNTH default camera (75° FOV, 1024x1024):")
    print(f"  fx = fy = {K[0,0]:.2f} pixels")
    print(f"  cx = {K[0,2]:.2f}, cy = {K[1,2]:.2f}")
    print(f"\nIntrinsics matrix K:")
    print(K)


if __name__ == "__main__":
    test_intrinsics()
    render_and_save()
