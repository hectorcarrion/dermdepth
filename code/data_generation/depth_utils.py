"""
Depth utility functions for S-SYNTH depth rendering.

This module provides utilities for:
- Computing camera intrinsics from FOV and resolution
- Saving depth maps in MoGe-compatible 16-bit PNG format with logarithmic encoding
- Saving camera metadata in JSON format

MoGe depth encoding:
- 16-bit PNG with logarithmic encoding
- Values: 0=NaN, 1-65534=depth, 65535=inf
- Metadata stored in PNG chunks (near/far bounds)
"""

import numpy as np
import json
from PIL import Image, PngImagePlugin


def compute_intrinsics(fov_deg=75, width=1024, height=1024):
    """
    Compute camera intrinsics matrix from FOV and resolution.

    Args:
        fov_deg: Field of view in degrees (default 75, matching S-SYNTH)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        3x3 numpy array representing the camera intrinsics matrix K:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    For S-SYNTH default (75 deg, 1024x1024):
        fx = fy = 687.55 pixels
        cx = cy = 512 pixels
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = (width / 2) / np.tan(fov_rad / 2)
    fy = (height / 2) / np.tan(fov_rad / 2)
    cx = width / 2
    cy = height / 2

    return np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)


def save_depth_moge(path, depth_array, camera_height=None):
    """
    Save depth in MoGe-compatible 16-bit PNG with logarithmic encoding.

    The encoding maps depth values to 16-bit integers:
    - 0: NaN / invalid
    - 1-65534: valid depth (logarithmically encoded)
    - 65535: infinity

    Args:
        path: Output file path (.png)
        depth_array: 2D numpy array of depth values (camera-to-surface distance in mm)
        camera_height: Camera Y position for reference (optional, stored in metadata)

    Raises:
        ValueError: If no valid depth values exist in the array
    """
    depth = depth_array.astype(np.float32)

    # Handle invalid values
    mask_valid = np.isfinite(depth) & (depth > 0)
    if not mask_valid.any():
        raise ValueError("No valid depth values in depth array")

    # Compute near/far bounds from actual depth range
    near = max(depth[mask_valid].min(), 1e-5)
    far = max(near * 1.1, depth[mask_valid].max())

    # Logarithmic encoding to 16-bit
    # Maps [near, far] to [1, 65534] logarithmically
    depth_norm = (np.log(np.clip(depth, near, far) / near) / np.log(far / near))
    depth_encoded = (1 + np.round(depth_norm.clip(0, 1) * 65533)).astype(np.uint16)

    # Mark special values
    depth_encoded[~mask_valid] = 0  # Mark invalid as NaN
    depth_encoded[np.isinf(depth_array) & (depth_array > 0)] = 65535  # Mark positive infinity

    # Save with metadata in PNG chunks
    pil_image = Image.fromarray(depth_encoded)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text('near', str(near))
    pnginfo.add_text('far', str(far))
    if camera_height is not None:
        pnginfo.add_text('camera_height', str(camera_height))
    pil_image.save(path, pnginfo=pnginfo, compress_level=7)


def load_depth_moge(path):
    """
    Load depth from MoGe-format 16-bit PNG with logarithmic decoding.

    Args:
        path: Path to depth PNG file

    Returns:
        tuple: (depth_array, metadata_dict)
            - depth_array: 2D numpy array of depth values (NaN for invalid)
            - metadata_dict: Dictionary with 'near', 'far', and optional 'camera_height'
    """
    pil_image = Image.open(path)
    depth_encoded = np.array(pil_image, dtype=np.uint16)

    # Extract metadata from PNG chunks
    metadata = {}
    if 'near' in pil_image.info:
        metadata['near'] = float(pil_image.info['near'])
    if 'far' in pil_image.info:
        metadata['far'] = float(pil_image.info['far'])
    if 'camera_height' in pil_image.info:
        metadata['camera_height'] = float(pil_image.info['camera_height'])

    near = metadata.get('near', 1e-5)
    far = metadata.get('far', 100.0)

    # Decode: reverse the logarithmic encoding
    # 0 -> NaN, 1-65534 -> depth, 65535 -> inf
    depth_norm = (depth_encoded.astype(np.float32) - 1) / 65533.0
    depth = near * np.exp(depth_norm * np.log(far / near))

    # Handle special values
    depth[depth_encoded == 0] = np.nan
    depth[depth_encoded == 65535] = np.inf

    return depth, metadata


def save_meta_json(path, fov_deg=75, width=1024, height=1024, additional_meta=None):
    """
    Save camera intrinsics and metadata in MoGe-compatible JSON format.

    Args:
        path: Output file path (.json)
        fov_deg: Field of view in degrees
        width: Image width in pixels
        height: Image height in pixels
        additional_meta: Optional dictionary of additional metadata to include
    """
    intrinsics = compute_intrinsics(fov_deg, width, height)
    meta = {
        'intrinsics': intrinsics.tolist(),
        'fov_deg': fov_deg,
        'width': width,
        'height': height
    }
    if additional_meta:
        meta.update(additional_meta)

    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


def save_depth_raw(path, depth_array):
    """
    Save depth as raw 32-bit float NPY file (for debugging/validation).

    Args:
        path: Output file path (.npy)
        depth_array: 2D numpy array of depth values
    """
    np.save(path, depth_array.astype(np.float32))


def validate_depth(depth_array, expected_min=5.0, expected_max=50.0):
    """
    Validate depth array values are within expected range for S-SYNTH.

    S-SYNTH typical depth range:
    - Camera at Y = 15mm (default)
    - Skin surface at Y ≈ 0 to -5mm
    - Expected depth: 10-25mm typically

    Args:
        depth_array: 2D numpy array of depth values
        expected_min: Minimum expected valid depth (default 5mm)
        expected_max: Maximum expected valid depth (default 50mm)

    Returns:
        dict: Validation results with statistics and any warnings
    """
    valid_mask = np.isfinite(depth_array) & (depth_array > 0)
    valid_depth = depth_array[valid_mask]

    results = {
        'valid_pixels': int(valid_mask.sum()),
        'total_pixels': depth_array.size,
        'valid_ratio': float(valid_mask.sum() / depth_array.size),
        'warnings': []
    }

    if len(valid_depth) > 0:
        results['min'] = float(valid_depth.min())
        results['max'] = float(valid_depth.max())
        results['mean'] = float(valid_depth.mean())
        results['std'] = float(valid_depth.std())

        if valid_depth.min() < expected_min:
            results['warnings'].append(
                f"Minimum depth {valid_depth.min():.2f} below expected {expected_min}"
            )
        if valid_depth.max() > expected_max:
            results['warnings'].append(
                f"Maximum depth {valid_depth.max():.2f} above expected {expected_max}"
            )
    else:
        results['warnings'].append("No valid depth values found")

    return results
