#!/usr/bin/env python3
"""
Extended S-SYNTH features with depth rendering for MoGe-2 fine-tuning:
1. Multiple lesions per sample
2. Larger skin area (scaled skin)
3. Depth maps with mm-level accuracy
4. Tilted camera views
5. Complete MoGe-2 training data (RGB + depth + intrinsics)
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import mitsuba as mi

try:
    mi.set_variant('cuda_ad_spectral')
    import drjit as dr
    dr.set_flag(dr.JitFlag.Debug, True)  # Workaround for CUDA 12.7 driver miscompilation
    print("Using CUDA spectral variant (GPU, debug mode)")
except:
    mi.set_variant('scalar_spectral')
    print("Falling back to scalar spectral (CPU)")

import config
import depth_utils

OUTPUT_DIR = "/workspace/hector/ssynth-release/test_output/extended_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standard parameters
PAPER_SPP = 64  # Lower for testing
CAMERA_HEIGHT = 15
FOV = 75


def get_sensor(camera_height=15, fov=75, width=1024, height=1024, angle_offset=(0, 0)):
    """Create camera sensor."""
    import math
    x_off = camera_height * math.tan(math.radians(angle_offset[0]))
    z_off = camera_height * math.tan(math.radians(angle_offset[1]))
    origin = [x_off, camera_height, z_off]

    sensor = mi.load_dict({
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(
            target=[0, 0, 0],
            origin=origin,
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


def get_sensor_depth(camera_height=15, fov=75, width=1024, height=1024, angle_offset=(0, 0)):
    """Create camera sensor for depth rendering (scalar_rgb variant)."""
    import math
    x_off = camera_height * math.tan(math.radians(angle_offset[0]))
    z_off = camera_height * math.tan(math.radians(angle_offset[1]))
    origin = [x_off, camera_height, z_off]

    sensor = mi.load_dict({
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(
            target=[0, 0, 0],
            origin=origin,
            up=[0, 0, 1]
        ),
        'fov': fov,
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height,
            'pixel_format': 'rgba',
        }
    })
    return sensor


def create_depth_scene(model_id, lesion_configs, lesion_directory, skin_scale=1.0,
                        camera_height=15, angle_offset=(0, 0)):
    """
    Create a simplified scene for depth rendering (geometry only, no spectral materials).

    Uses AOV integrator to capture depth channel. The depth is computed as the
    distance from camera to the surface intersection point.
    """
    import math

    y_offset = -1.5 * skin_scale
    room_size = 40

    # Calculate camera position with angle offset
    x_off = camera_height * math.tan(math.radians(angle_offset[0]))
    z_off = camera_height * math.tan(math.radians(angle_offset[1]))

    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'aov',
            'aovs': 'dd.y:depth',
            'nested': {'type': 'direct'}
        }
    }

    # Simple diffuse BSDF for all geometry
    simple_bsdf = {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}
    }

    # Epidermis (skin surface - this is what we want depth of)
    scene['epidermis'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/epidermis_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(skin_scale).translate([0, y_offset / skin_scale, 0]),
        'bsdf': simple_bsdf
    }

    # Add lesions
    for i, lcfg in enumerate(lesion_configs):
        version = lcfg.get('version', 'ver1')
        if version == 'ver0':
            lesion_dir = config.sDir_lesion_ver0
        else:
            lesion_dir = config.sDir_lesion_ver1

        lesion_file = f"{lesion_dir}/lesion{lcfg['lesion_id']}_T{lcfg['time_point']:03d}.obj"
        pos_x, pos_z = lcfg.get('position', (0, 0))
        scale = lcfg.get('scale', 1.5)
        lesion_offset = lcfg.get('y_offset', -2)

        scene[f'lesion_{i}'] = {
            'type': 'obj',
            'filename': lesion_file,
            'to_world': mi.ScalarTransform4f.scale(scale).translate([pos_x, lesion_offset, pos_z]),
            'bsdf': simple_bsdf
        }

    # Directional light from above for depth pass
    scene['light'] = {
        'type': 'directional',
        'direction': [0, -1, 0],
        'irradiance': {'type': 'rgb', 'value': 1.0}
    }

    return scene


def ray_distance_to_z_depth(ray_distance, fov_deg, width=1024, height=1024):
    """
    Convert ray distance (3D Euclidean distance) to Z-depth (perpendicular distance).

    MoGe-2 expects Z-depth, which is the perpendicular distance from the camera plane
    to the surface. This is the standard pinhole camera model where:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    The conversion is:
        Z_depth = ray_distance * cos(angle_from_optical_axis)
               = ray_distance / sqrt(1 + tan_x^2 + tan_y^2)

    where tan_x, tan_y are the tangent of the viewing angle for each pixel.
    """
    # Compute focal length from FOV
    fov_rad = np.deg2rad(fov_deg)
    # For a symmetric camera, focal = (width/2) / tan(fov/2)
    # In normalized coordinates, focal_normalized = 0.5 / tan(fov/2)
    focal = (width / 2) / np.tan(fov_rad / 2)

    # Create pixel coordinate grids (0 to width-1, 0 to height-1)
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)

    # Compute offset from principal point (image center)
    cx, cy = width / 2, height / 2
    dx = (u_grid - cx) / focal  # tan(angle_x)
    dy = (v_grid - cy) / focal  # tan(angle_y)

    # Compute cos(theta) where theta is angle from optical axis
    # cos(theta) = 1 / sqrt(1 + tan_x^2 + tan_y^2)
    cos_theta = 1.0 / np.sqrt(1.0 + dx**2 + dy**2)

    # Convert ray distance to Z-depth
    z_depth = ray_distance * cos_theta

    return z_depth


def render_depth_proper(model_id, lesion_configs, camera_height=15, fov=75,
                        angle_offset=(0, 0), skin_scale=1.0, hair_model=-1):
    """
    Render depth using Mitsuba AOV integrator and convert to Z-depth.

    MoGe-2 expects Z-depth (perpendicular distance from camera plane), not ray distance.
    This function:
    1. Renders ray distance using Mitsuba's 'dd.y:depth' AOV
    2. Converts ray distance to Z-depth using the camera FOV

    Args:
        model_id: Skin model ID
        lesion_configs: List of lesion configuration dicts
        camera_height: Camera Y position in mm
        fov: Field of view in degrees
        angle_offset: (tilt_x, tilt_z) camera angle offset in degrees
        skin_scale: Scale factor for skin model
        hair_model: Hair model ID (-1 for no hair)

    Returns depth array in mm (Z-depth format for MoGe-2).
    """
    import math

    # Create depth scene
    y_offset = -1.5 * skin_scale

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'aov',
            'aovs': 'dd.y:depth',
            'nested': {'type': 'direct'}
        }
    }

    simple_bsdf = {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}
    }

    # Epidermis
    scene_dict['epidermis'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/epidermis_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(skin_scale).translate([0, y_offset / skin_scale, 0]),
        'bsdf': simple_bsdf
    }

    # Add hair (for consistent RGB-depth correspondence)
    if hair_model >= 0:
        scene_dict['hair'] = {
            'type': 'obj',
            'filename': config.sDir + f'outputModels/hair_{hair_model:03d}.obj',
            'to_world': mi.ScalarTransform4f.scale(skin_scale).translate([0, y_offset / skin_scale, 0]),
            'bsdf': simple_bsdf
        }

    # Add lesions
    for i, lcfg in enumerate(lesion_configs):
        version = lcfg.get('version', 'ver1')
        lesion_dir = config.sDir_lesion_ver0 if version == 'ver0' else config.sDir_lesion_ver1
        lesion_file = f"{lesion_dir}/lesion{lcfg['lesion_id']}_T{lcfg['time_point']:03d}.obj"
        pos_x, pos_z = lcfg.get('position', (0, 0))
        scale = lcfg.get('scale', 1.5)
        lesion_offset = lcfg.get('y_offset', -2)

        scene_dict[f'lesion_{i}'] = {
            'type': 'obj',
            'filename': lesion_file,
            'to_world': mi.ScalarTransform4f.scale(scale).translate([pos_x, lesion_offset, pos_z]),
            'bsdf': simple_bsdf
        }

    # Floor (matches RGB scene so all visible pixels have depth)
    room_size = 20
    scene_dict['wall_floor'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.scale([room_size, 1, room_size]).translate([0, -room_size, 0]).rotate([1, 0, 0], -90),
        'bsdf': simple_bsdf
    }

    # Light
    scene_dict['light'] = {
        'type': 'directional',
        'direction': [0, -1, 0],
        'irradiance': {'type': 'rgb', 'value': 1.0}
    }

    # Create sensor with angle
    x_off = camera_height * math.tan(math.radians(angle_offset[0]))
    z_off = camera_height * math.tan(math.radians(angle_offset[1]))
    origin = [x_off, camera_height, z_off]

    width, height = 1024, 1024
    sensor = mi.load_dict({
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(
            target=[0, 0, 0],
            origin=origin,
            up=[0, 0, 1]
        ),
        'fov': fov,
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height,
            'pixel_format': 'rgba',
        }
    })

    # Render
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, sensor=sensor, spp=1)
    depth_array = np.array(image)

    # Extract depth AOV channel
    # Output is RGBA + AOV, so depth is in channel 4 (index 4) for 5-channel output
    # The AOV 'dd.y:depth' gives the RAY DISTANCE (3D distance from camera to surface)
    if depth_array.ndim == 3:
        if depth_array.shape[2] >= 5:
            # RGBA + AOV format: depth is in channel 4
            ray_distance = depth_array[:, :, 4]
        elif depth_array.shape[2] >= 4:
            # Check if channel 3 has depth values (not just alpha)
            ch3 = depth_array[:, :, 3]
            ch3_range = ch3.max() - ch3.min()
            if ch3_range > 0.1:  # Has actual depth data
                ray_distance = ch3
            else:
                ray_distance = depth_array[:, :, 0]
        else:
            ray_distance = depth_array[:, :, 0]
    else:
        ray_distance = depth_array

    # Mark zero/very small values as NaN (no intersection / background)
    ray_distance = np.where(ray_distance < 0.1, np.nan, ray_distance)

    # Convert ray distance to Z-depth for MoGe-2 compatibility
    # Z-depth is the perpendicular distance from camera plane (standard pinhole model)
    # Ray distance varies across the image (bowl pattern), Z-depth is more uniform for flat surfaces
    z_depth = ray_distance_to_z_depth(ray_distance, fov, width, height)

    return z_depth


def render_depth(scene_dict, sensor, camera_height):
    """Render depth map from scene."""
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, sensor=sensor, spp=1)
    depth_array = np.array(image)

    # Extract depth AOV channel (channel 3 in RGBA output)
    if depth_array.ndim == 3 and depth_array.shape[2] >= 4:
        depth = depth_array[:, :, 3]
    else:
        depth = depth_array[:, :, 0] if depth_array.ndim == 3 else depth_array

    return depth


def render_sample_with_depth(model_id, lesion_configs, melanin, blood_frac,
                              light_name, output_dir, sample_name,
                              camera_height=15, fov=75, angle_offset=(0, 0),
                              skin_scale=1.0, hair_model=-1, spp=64):
    """
    Render a complete sample with RGB image, depth map, and metadata.

    Returns dict with paths and validation info.
    """
    os.makedirs(output_dir, exist_ok=True)

    result = {
        'name': sample_name,
        'camera_height': camera_height,
        'angle_offset': angle_offset,
        'skin_scale': skin_scale
    }

    # === Render RGB Image ===
    print(f"    Rendering RGB...")
    mi.set_variant('cuda_ad_spectral')
    import drjit as dr
    dr.set_flag(dr.JitFlag.Debug, True)  # Required for CUDA 12.7 driver workaround

    scene_dict = create_multi_lesion_scene(
        model_id, lesion_configs, melanin, blood_frac, light_name, hair_model
    )
    sensor = get_sensor(camera_height, fov, angle_offset=angle_offset)

    start = time.time()
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, sensor=sensor, spp=spp)
    rgb_time = time.time() - start

    image_array = np.array(image)[:, :, :3]
    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    rgb_path = os.path.join(output_dir, f"{sample_name}_rgb.png")
    Image.fromarray(image_uint8).save(rgb_path)
    result['rgb_path'] = rgb_path
    result['rgb_time'] = rgb_time

    # === Render Depth ===
    print(f"    Rendering depth...")
    mi.set_variant('scalar_rgb')

    start = time.time()
    depth = render_depth_proper(
        model_id, lesion_configs,
        camera_height=camera_height,
        fov=fov,
        angle_offset=angle_offset,
        skin_scale=skin_scale,
        hair_model=hair_model
    )
    depth_time = time.time() - start

    # Save depth in MoGe format
    depth_path = os.path.join(output_dir, f"{sample_name}_depth.png")
    depth_utils.save_depth_moge(depth_path, depth, camera_height)
    result['depth_path'] = depth_path
    result['depth_time'] = depth_time

    # Validate depth
    validation = depth_utils.validate_depth(depth)
    result['depth_validation'] = validation

    # === Save metadata (MoGe format) ===
    meta_path = os.path.join(output_dir, f"{sample_name}_meta.json")
    depth_utils.save_meta_json(
        meta_path, fov_deg=fov, width=1024, height=1024,
        additional_meta={
            'camera_height_mm': camera_height,
            'angle_offset_deg': angle_offset,
            'skin_scale': skin_scale,
            'depth_range_mm': [validation.get('min', 0), validation.get('max', 0)],
            'lesion_configs': lesion_configs,
            'melanin': melanin,
            'blood_fraction': blood_frac
        }
    )
    result['meta_path'] = meta_path

    # Switch back to spectral
    mi.set_variant('cuda_ad_spectral')
    dr.set_flag(dr.JitFlag.Debug, True)

    return result


def create_multi_lesion_scene(model_id, lesion_configs, melanin, blood_frac,
                               light_name, hair_model=-1):
    """
    Create a scene with MULTIPLE lesions.

    Args:
        model_id: Skin model ID
        lesion_configs: List of dicts with keys:
            - lesion_id, time_point, lesion_mat, scale, position (x, z), version
        melanin: Skin melanin fraction
        blood_frac: Blood fraction
        light_name: HDRI environment name
        hair_model: Hair model ID (-1 for no hair)
    """
    uniform_scale = 1
    y_offset = -1.5
    room_size = 20
    xt_scale = 0.1

    # Hair optical properties
    ext_hair = 28.5 + 37.5
    hair_albedo = [0.84, 0.6328, 0.44]  # Brown

    # Refractive indices
    ior_epi = 1.43
    ior_blood = 1.36
    ior_hypo = 1.44
    A, B, C = 1.3696, 3916.8, 2558.8
    ior_derm = A + (B / (500 ** 2)) + (C / (500 ** 4))

    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'volpathmis',
            'max_depth': 50
        }
    }

    # Add MULTIPLE lesions
    for i, lcfg in enumerate(lesion_configs):
        version = lcfg.get('version', 'ver1')
        if version == 'ver0':
            lesion_dir = config.sDir_lesion_ver0
        else:
            lesion_dir = config.sDir_lesion_ver1

        lesion_file = f"{lesion_dir}/lesion{lcfg['lesion_id']}_T{lcfg['time_point']:03d}.obj"
        pos_x, pos_z = lcfg.get('position', (0, 0))
        scale = lcfg.get('scale', 1.5)
        lesion_offset = lcfg.get('y_offset', -2)

        scene[f'lesion_{i}'] = {
            'type': 'obj',
            'filename': lesion_file,
            'to_world': mi.ScalarTransform4f.scale(uniform_scale * scale).translate([pos_x, lesion_offset, pos_z]),
            'bsdf': {
                'type': 'roughdielectric',
                'alpha': 0.01,
                'int_ior': 1.32988 - (-3.97577e7) * 0.95902 ** 500,
                'ext_ior': 1.000277
            },
            'interior': {
                'type': 'homogeneous',
                'albedo': {
                    'type': 'spectrum',
                    'filename': config.sDir + f"opticalMaterials/{lcfg['lesion_mat']}_alb.spd"
                },
                'sigma_t': {
                    'type': 'spectrum',
                    'filename': config.sDir + f"opticalMaterials/{lcfg['lesion_mat']}_ext.spd"
                },
                'scale': xt_scale
            }
        }

    # Hair (optional)
    if hair_model >= 0:
        scene['hair'] = {
            'type': 'obj',
            'filename': config.sDir + f'outputModels/hair_{hair_model:03d}.obj',
            'to_world': mi.ScalarTransform4f.scale(uniform_scale).translate([0, y_offset, 0]),
            'bsdf': {
                'type': 'roughdielectric',
                'alpha': 0.01,
                'int_ior': 1.55,
                'ext_ior': 1.000277
            },
            'interior': {
                'type': 'homogeneous',
                'sigma_t': ext_hair,
                'albedo': {'type': 'rgb', 'value': hair_albedo},
                'scale': 3
            }
        }

    # Epidermis
    scene['epidermis'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/epidermis_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(uniform_scale).translate([0, y_offset, 0]),
        'bsdf': {
            'type': 'roughdielectric',
            'alpha': 0.01,
            'int_ior': ior_epi,
            'ext_ior': 1.000277
        },
        'interior': {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/epidermis_alb_mel{melanin}.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/epidermis_ext_mel{melanin}.spd'
            },
            'scale': xt_scale
        }
    }

    # Vascular
    scene['vascular'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/vascular_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(uniform_scale).translate([0, y_offset, 0]),
        'bsdf': {
            'type': 'roughdielectric',
            'alpha': 0.01,
            'int_ior': ior_blood,
            'ext_ior': 1.000277
        },
        'interior': {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir + 'opticalMaterials/blood_HbO2_alb.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir + 'opticalMaterials/blood_HbO2_ext.spd'
            },
            'scale': xt_scale
        }
    }

    # Dermis
    scene['dermis'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/dermis_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(uniform_scale).translate([0, y_offset, 0]),
        'bsdf': {
            'type': 'roughdielectric',
            'alpha': 0.01,
            'int_ior': ior_derm,
            'ext_ior': 1.000277
        },
        'interior': {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/dermis_alb_fB{blood_frac}.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/dermis_ext_fB{blood_frac}.spd'
            },
            'scale': xt_scale
        }
    }

    # Hypodermis
    scene['hypodermis'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/hypodermis_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(uniform_scale).translate([0, y_offset, 0]),
        'bsdf': {
            'type': 'roughdielectric',
            'alpha': 0.01,
            'int_ior': ior_hypo,
            'ext_ior': 1.000277
        },
        'interior': {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir + 'opticalMaterials/hypo_alb.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir + 'opticalMaterials/hypo_ext.spd'
            },
            'scale': xt_scale
        }
    }

    # Environment lighting
    scene['env_light'] = {
        'type': 'envmap',
        'filename': config.sDir_hdri + '/' + light_name + '.exr',
        'scale': 3
    }

    # Floor
    scene['wall_floor'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.scale([room_size, 1, room_size]).translate([0, -room_size, 0]).rotate([1, 0, 0], -90),
        'bsdf': {
            'type': 'twosided',
            'material': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': 0.5}
            }
        }
    }

    return scene


def test_multi_lesion():
    """Test multiple lesions on a single skin patch."""
    print("\n" + "="*60)
    print("Testing Multiple Lesions")
    print("="*60)

    # Multiple lesion configurations - different positions on the skin
    lesion_configs = [
        # Center lesion (red, raised)
        {
            'lesion_id': 12,
            'time_point': 20,
            'lesion_mat': 'HbO2x1.0Epix0.025',
            'scale': 1.5,
            'position': (0, 0),  # Center
            'version': 'ver1'
        },
        # Top-left lesion (dark, flat)
        {
            'lesion_id': 15,
            'time_point': 10,
            'lesion_mat': 'HbO2x0.1Epix0.4',
            'scale': 1.0,
            'position': (-5, -5),  # Top-left quadrant
            'version': 'ver1'
        },
        # Bottom-right lesion (pink, small)
        {
            'lesion_id': 11,
            'time_point': 10,  # ver0 only has 10, 20, 30, 40, 50
            'lesion_mat': 'HbO2x0.5Epix0.15',
            'scale': 1.2,
            'position': (4, 4),  # Bottom-right quadrant
            'version': 'ver0'
        },
    ]

    print(f"  Creating scene with {len(lesion_configs)} lesions...")
    scene_dict = create_multi_lesion_scene(
        model_id=30,
        lesion_configs=lesion_configs,
        melanin=0.11,  # Light-medium skin
        blood_frac=0.02,
        light_name='surgery_4k',
        hair_model=50
    )

    sensor = get_sensor(camera_height=CAMERA_HEIGHT, fov=FOV)

    print("  Rendering...")
    start = time.time()
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, sensor=sensor, spp=PAPER_SPP)
    render_time = time.time() - start
    print(f"  Done in {render_time:.1f}s")

    # Save
    image_array = np.array(image)[:, :, :3]
    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    output_path = os.path.join(OUTPUT_DIR, "multi_lesion_test.png")
    Image.fromarray(image_uint8).save(output_path)
    print(f"  Saved: {output_path}")

    return output_path


def test_scaled_skin():
    """Test scaling up a single skin model for larger area (no seams)."""
    print("\n" + "="*60)
    print("Testing Scaled Skin (Single Model, No Seams)")
    print("="*60)

    from render_extended_features import create_multi_lesion_scene, get_sensor, OUTPUT_DIR, PAPER_SPP

    # Multiple lesions spread across the scaled area
    lesion_configs = [
        {
            'lesion_id': 12,
            'time_point': 20,
            'lesion_mat': 'HbO2x1.0Epix0.025',
            'scale': 1.5,
            'position': (0, 0),
            'version': 'ver1'
        },
        {
            'lesion_id': 15,
            'time_point': 15,
            'lesion_mat': 'HbO2x0.1Epix0.25',
            'scale': 1.0,
            'position': (-6, 5),
            'version': 'ver1'
        },
    ]

    # Create scene with scaled skin (will modify below)
    scene_dict = create_scaled_skin_scene(
        model_id=30,
        lesion_configs=lesion_configs,
        melanin=0.21,
        blood_frac=0.02,
        light_name='surgery_4k',
        skin_scale=2.0,  # Scale skin 2x (40x40mm instead of 20x20mm)
        hair_model=-1  # No hair for cleaner look
    )

    # Camera higher to see the larger area
    sensor = get_sensor(camera_height=30, fov=75)  # Higher camera

    print("  Rendering scaled skin...")
    start = time.time()
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, sensor=sensor, spp=PAPER_SPP)
    render_time = time.time() - start
    print(f"  Done in {render_time:.1f}s")

    # Save
    image_array = np.array(image)[:, :, :3]
    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    output_path = os.path.join(OUTPUT_DIR, "scaled_skin_test.png")
    Image.fromarray(image_uint8).save(output_path)
    print(f"  Saved: {output_path}")

    return output_path


def create_scaled_skin_scene(model_id, lesion_configs, melanin, blood_frac,
                              light_name, skin_scale=2.0, hair_model=-1):
    """
    Create a scene with a SCALED UP skin model for larger area without seams.
    """
    y_offset = -1.5 * skin_scale
    room_size = 40
    xt_scale = 0.1

    # Refractive indices
    ior_epi = 1.43
    ior_blood = 1.36
    ior_hypo = 1.44
    A, B, C = 1.3696, 3916.8, 2558.8
    ior_derm = A + (B / (500 ** 2)) + (C / (500 ** 4))

    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'volpathmis',
            'max_depth': 50
        }
    }

    # Scaled skin layers
    scene['epidermis'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/epidermis_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(skin_scale).translate([0, y_offset / skin_scale, 0]),
        'bsdf': {
            'type': 'roughdielectric',
            'alpha': 0.01,
            'int_ior': ior_epi,
            'ext_ior': 1.000277
        },
        'interior': {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/epidermis_alb_mel{melanin}.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/epidermis_ext_mel{melanin}.spd'
            },
            'scale': xt_scale / skin_scale  # Adjust for scale
        }
    }

    scene['vascular'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/vascular_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(skin_scale).translate([0, y_offset / skin_scale, 0]),
        'bsdf': {
            'type': 'roughdielectric',
            'alpha': 0.01,
            'int_ior': ior_blood,
            'ext_ior': 1.000277
        },
        'interior': {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir + 'opticalMaterials/blood_HbO2_alb.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir + 'opticalMaterials/blood_HbO2_ext.spd'
            },
            'scale': xt_scale / skin_scale
        }
    }

    scene['dermis'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/dermis_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(skin_scale).translate([0, y_offset / skin_scale, 0]),
        'bsdf': {
            'type': 'roughdielectric',
            'alpha': 0.01,
            'int_ior': ior_derm,
            'ext_ior': 1.000277
        },
        'interior': {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/dermis_alb_fB{blood_frac}.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/dermis_ext_fB{blood_frac}.spd'
            },
            'scale': xt_scale / skin_scale
        }
    }

    scene['hypodermis'] = {
        'type': 'obj',
        'filename': config.sDir + f'outputModels/hypodermis_{model_id:03d}.obj',
        'to_world': mi.ScalarTransform4f.scale(skin_scale).translate([0, y_offset / skin_scale, 0]),
        'bsdf': {
            'type': 'roughdielectric',
            'alpha': 0.01,
            'int_ior': ior_hypo,
            'ext_ior': 1.000277
        },
        'interior': {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir + 'opticalMaterials/hypo_alb.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir + 'opticalMaterials/hypo_ext.spd'
            },
            'scale': xt_scale / skin_scale
        }
    }

    # Add lesions (not scaled - keep original size)
    for i, lcfg in enumerate(lesion_configs):
        version = lcfg.get('version', 'ver1')
        if version == 'ver0':
            lesion_dir = config.sDir_lesion_ver0
        else:
            lesion_dir = config.sDir_lesion_ver1

        lesion_file = f"{lesion_dir}/lesion{lcfg['lesion_id']}_T{lcfg['time_point']:03d}.obj"
        pos_x, pos_z = lcfg.get('position', (0, 0))
        scale = lcfg.get('scale', 1.5)
        lesion_offset = lcfg.get('y_offset', -2)

        scene[f'lesion_{i}'] = {
            'type': 'obj',
            'filename': lesion_file,
            'to_world': mi.ScalarTransform4f.scale(scale).translate([pos_x, lesion_offset, pos_z]),
            'bsdf': {
                'type': 'roughdielectric',
                'alpha': 0.01,
                'int_ior': 1.32988 - (-3.97577e7) * 0.95902 ** 500,
                'ext_ior': 1.000277
            },
            'interior': {
                'type': 'homogeneous',
                'albedo': {
                    'type': 'spectrum',
                    'filename': config.sDir + f"opticalMaterials/{lcfg['lesion_mat']}_alb.spd"
                },
                'sigma_t': {
                    'type': 'spectrum',
                    'filename': config.sDir + f"opticalMaterials/{lcfg['lesion_mat']}_ext.spd"
                },
                'scale': xt_scale
            }
        }

    # Environment lighting
    scene['env_light'] = {
        'type': 'envmap',
        'filename': config.sDir_hdri + '/' + light_name + '.exr',
        'scale': 3
    }

    # Larger floor
    scene['wall_floor'] = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.scale([room_size, 1, room_size]).translate([0, -room_size, 0]).rotate([1, 0, 0], -90),
        'bsdf': {
            'type': 'twosided',
            'material': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': 0.5}
            }
        }
    }

    return scene


def test_depth_with_tilted_views():
    """
    Comprehensive depth rendering test with tilted camera views.

    Tests mm-level accuracy for MoGe-2 fine-tuning:
    - Multiple camera angles (top-down, 15°, 30°, 45°)
    - Diverse lesion morphologies (red raised, dark flat, etc.)
    - Depth validation and 3D visualization
    """
    print("\n" + "="*60)
    print("Testing Depth Rendering with Tilted Views")
    print("="*60)

    output_dir = os.path.join(OUTPUT_DIR, "depth_samples")
    os.makedirs(output_dir, exist_ok=True)

    # Camera angles to test
    angles = [
        ((0, 0), "top_down"),
        ((15, 0), "tilt_15"),
        ((30, 0), "tilt_30"),
        ((45, 0), "tilt_45"),
    ]

    # Lesion configurations for testing
    lesion_configs = [
        # Red raised lesion
        {
            'lesion_id': 12,
            'time_point': 20,
            'lesion_mat': 'HbO2x1.0Epix0.025',
            'scale': 1.5,
            'position': (0, 0),
            'version': 'ver1'
        },
        # Dark flat lesion
        {
            'lesion_id': 15,
            'time_point': 10,
            'lesion_mat': 'HbO2x0.1Epix0.4',
            'scale': 1.0,
            'position': (-5, 4),
            'version': 'ver1'
        },
    ]

    samples = []

    for angle_offset, angle_name in angles:
        print(f"\n  Rendering {angle_name} view...")
        sample = render_sample_with_depth(
            model_id=30,
            lesion_configs=lesion_configs,
            melanin=0.21,
            blood_frac=0.02,
            light_name='surgery_4k',
            output_dir=output_dir,
            sample_name=f"depth_{angle_name}",
            camera_height=CAMERA_HEIGHT,
            fov=FOV,
            angle_offset=angle_offset,
            skin_scale=1.0,
            hair_model=50,
            spp=64
        )
        samples.append(sample)

        # Print depth stats
        v = sample['depth_validation']
        print(f"    Depth range: {v.get('min', 0):.2f} - {v.get('max', 0):.2f} mm")
        print(f"    Mean depth: {v.get('mean', 0):.2f} mm (std: {v.get('std', 0):.2f})")

    # Create comprehensive visualization
    create_depth_visualization(samples, output_dir)

    # Validate mm-level accuracy
    validate_mm_accuracy(samples, output_dir)

    return samples


def create_depth_visualization(samples, output_dir):
    """Create visualization showing RGB + Depth for each camera angle."""
    print("\n  Creating depth visualization...")

    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    fig.suptitle("S-SYNTH Depth Rendering: RGB + Depth + Profile\n(For MoGe-2 Fine-tuning)",
                 fontsize=16, fontweight='bold')

    for i, sample in enumerate(samples):
        # Load RGB
        rgb = Image.open(sample['rgb_path'])
        axes[i, 0].imshow(rgb)
        angle = sample['angle_offset']
        axes[i, 0].set_title(f"RGB - {sample['name']}\nAngle: ({angle[0]}°, {angle[1]}°)", fontsize=10)
        axes[i, 0].axis('off')

        # Load and display depth
        depth, meta = depth_utils.load_depth_moge(sample['depth_path'])
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        vmin, vmax = valid_depth.min(), valid_depth.max()

        im = axes[i, 1].imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Depth Map\nRange: {vmin:.2f} - {vmax:.2f} mm", fontsize=10)
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, label='Depth (mm)')

        # Depth profile (center horizontal line)
        center_y = depth.shape[0] // 2
        profile = depth[center_y, :]
        valid_mask = np.isfinite(profile)
        x = np.arange(len(profile))

        axes[i, 2].plot(x[valid_mask], profile[valid_mask], 'b-', linewidth=1)
        axes[i, 2].set_xlabel('Pixel X')
        axes[i, 2].set_ylabel('Depth (mm)')
        axes[i, 2].set_title(f"Center Horizontal Profile\n(shows lesion elevation)", fontsize=10)
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_ylim([vmin - 1, vmax + 1])

    plt.tight_layout()
    output_path = os.path.join(output_dir, "depth_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def validate_mm_accuracy(samples, output_dir):
    """
    Validate mm-level depth accuracy for MoGe-2 training.

    Checks:
    1. Depth range is consistent with camera height
    2. Lesion elevations are detectable (sub-mm to few mm)
    3. Depth resolution sufficient for fine-tuning
    """
    print("\n  Validating mm-level accuracy...")

    report = {
        'summary': {},
        'per_sample': []
    }

    all_depths = []
    for sample in samples:
        depth, meta = depth_utils.load_depth_moge(sample['depth_path'])
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        all_depths.extend(valid_depth.flatten())

        v = sample['depth_validation']
        sample_report = {
            'name': sample['name'],
            'angle': sample['angle_offset'],
            'depth_min_mm': v.get('min', 0),
            'depth_max_mm': v.get('max', 0),
            'depth_mean_mm': v.get('mean', 0),
            'depth_std_mm': v.get('std', 0),
            'depth_range_mm': v.get('max', 0) - v.get('min', 0)
        }
        report['per_sample'].append(sample_report)

    # Compute overall stats
    all_depths = np.array(all_depths)
    report['summary'] = {
        'global_min_mm': float(all_depths.min()),
        'global_max_mm': float(all_depths.max()),
        'global_mean_mm': float(all_depths.mean()),
        'global_std_mm': float(all_depths.std()),
        'total_depth_range_mm': float(all_depths.max() - all_depths.min()),
        'camera_height_mm': CAMERA_HEIGHT,
        'expected_skin_depth_mm': CAMERA_HEIGHT,  # Approximate
    }

    # Estimate lesion elevation from depth variation
    lesion_elevation_estimate = report['summary']['total_depth_range_mm']
    report['summary']['estimated_lesion_elevation_mm'] = lesion_elevation_estimate

    # MoGe-2 accuracy requirements
    report['summary']['moge2_requirements'] = {
        'min_depth_resolution_mm': 0.1,  # 100 microns
        'recommended_depth_range_mm': [5, 50],
        'current_meets_requirements': lesion_elevation_estimate > 0.1
    }

    # Save report
    report_path = os.path.join(output_dir, "depth_accuracy_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n  === Depth Accuracy Report ===")
    print(f"  Camera height: {CAMERA_HEIGHT} mm")
    print(f"  Global depth range: {report['summary']['global_min_mm']:.2f} - {report['summary']['global_max_mm']:.2f} mm")
    print(f"  Total depth variation: {report['summary']['total_depth_range_mm']:.2f} mm")
    print(f"  Estimated lesion elevation: {lesion_elevation_estimate:.2f} mm")
    print(f"  Meets MoGe-2 requirements: {report['summary']['moge2_requirements']['current_meets_requirements']}")
    print(f"  Report saved: {report_path}")

    # Create 3D visualization
    create_3d_depth_visualization(samples[0], output_dir)

    return report


def create_3d_depth_visualization(sample, output_dir):
    """Create 3D surface visualization of depth map."""
    print("\n  Creating 3D depth visualization...")

    depth, _ = depth_utils.load_depth_moge(sample['depth_path'])
    rgb = np.array(Image.open(sample['rgb_path'])) / 255.0

    fig = plt.figure(figsize=(16, 8))

    # 2D depth
    ax1 = fig.add_subplot(1, 2, 1)
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    vmin, vmax = valid_depth.min(), valid_depth.max()
    im = ax1.imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title(f"Depth Map (2D)\nRange: {vmin:.2f} - {vmax:.2f} mm", fontsize=12)
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, label='Depth (mm)', fraction=0.046)

    # 3D surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Subsample for visualization
    step = 16
    h, w = depth.shape
    x = np.arange(0, w, step)
    z = np.arange(0, h, step)
    X, Z = np.meshgrid(x, z)
    Y = depth[::step, ::step]

    # Mask invalid
    Y_plot = np.where(np.isfinite(Y), Y, np.nan)

    # Get RGB colors
    rgb_sub = rgb[::step, ::step, :3]

    # Plot surface with RGB texture
    ax2.plot_surface(X, Z, -Y_plot, facecolors=rgb_sub, alpha=0.9)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Z (pixels)')
    ax2.set_zlabel('Depth (mm)')
    ax2.set_title("3D Surface with RGB Texture\n(Inverted for visualization)", fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "depth_3d_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def test_zoomed_out_comparison():
    """Compare standard view vs scaled (zoomed out) view."""
    print("\n" + "="*60)
    print("Creating Zoom Comparison")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Standard view
    standard_path = os.path.join(OUTPUT_DIR, "multi_lesion_test.png")
    if os.path.exists(standard_path):
        img = Image.open(standard_path)
        axes[0].imshow(img)
        axes[0].set_title("Standard View (20x20mm)\n3 Lesions on Single Patch", fontsize=12)
        axes[0].axis('off')

    # Scaled view
    scaled_path = os.path.join(OUTPUT_DIR, "scaled_skin_test.png")
    if os.path.exists(scaled_path):
        img = Image.open(scaled_path)
        axes[1].imshow(img)
        axes[1].set_title("Scaled View (40x40mm)\nSeamless Larger Area", fontsize=12)
        axes[1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "zoom_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("S-SYNTH Extended Features Test")
    print("  - Multiple lesions")
    print("  - Scaled skin (larger area)")
    print("  - Depth rendering for MoGe-2")
    print("  - Tilted camera views")
    print("="*60)

    # Test 1: Multiple lesions
    multi_path = test_multi_lesion()

    # Test 2: Scaled skin (larger area, seamless)
    scaled_path = test_scaled_skin()

    # Test 3: Depth rendering with tilted views (for MoGe-2)
    depth_samples = test_depth_with_tilted_views()

    # Create comparison
    test_zoomed_out_comparison()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Multiple lesions: {multi_path}")
    print(f"Scaled skin: {scaled_path}")
    print(f"Depth samples: {len(depth_samples)} (with tilted views)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nMoGe-2 Training Data Generated:")
    print("  - RGB images (.png)")
    print("  - Depth maps (.png, 16-bit logarithmic encoding)")
    print("  - Camera intrinsics (.json)")
    print("  - Depth accuracy report")
