#!/usr/bin/env python3
"""
GPU-accelerated S-SYNTH rendering with full parameter exploration.

Matches paper settings:
- SPP: 124
- Resolution: 1024x1024
- GPU: cuda_ad_spectral variant

Explores:
- Lesion shapes (regular vs irregular, ver0 vs ver1)
- Lesion growth stages (time points)
- Lesion materials (HbO2 × melanin combinations)
- Lighting environments (20 HDRIs)
- Camera positions/angles
- Skin tones and hair variations
"""

import os
import sys
import time
import random
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from PIL import Image

# Set up Mitsuba with CUDA variant
import mitsuba as mi

# Check available variants and set GPU
print("Available Mitsuba variants:", mi.variants())

# Try CUDA spectral first, fall back to scalar
try:
    mi.set_variant('cuda_ad_spectral')
    GPU_AVAILABLE = True
    print("Using CUDA spectral variant (GPU)")
except Exception as e:
    print(f"CUDA not available: {e}")
    mi.set_variant('scalar_spectral')
    GPU_AVAILABLE = False
    print("Falling back to scalar spectral (CPU)")

import config
import depth_utils

# ============================================================================
# PAPER SETTINGS
# ============================================================================
PAPER_SPP = 124  # Samples per pixel (paper value)
PAPER_RESOLUTION = (1024, 1024)  # Width x Height
CAMERA_HEIGHT = 15  # mm (1.5 cm above skin)
FOV = 75  # degrees

# Output directory
OUTPUT_DIR = "/workspace/hector/ssynth-release/test_output/gpu_exploration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# PARAMETER SPACE
# ============================================================================

# Lesion versions and their time points (verified from actual files)
LESION_VERSIONS = {
    'ver0': {
        'path': config.sDir_lesion_ver0,
        'time_points': [10, 20, 30, 40, 50],  # Regular lesions: T010, T020, T030, T040, T050
        'lesion_ids': list(range(10, 21)),  # lesion10 through lesion20
        'description': 'Regular (smooth boundary)'
    },
    'ver1': {
        'path': config.sDir_lesion_ver1,
        'time_points': [2, 5, 10, 15, 20, 25, 30],  # Irregular: T002, T005, T010, T015, T020, T025, T030
        'lesion_ids': list(range(10, 21)),  # lesion10 through lesion20 (same IDs as ver0)
        'description': 'Irregular (jagged boundary)'
    }
}

# Lesion materials (optical properties)
LESION_MATERIALS = [
    "melDermEpi",  # Melanin in dermis and epidermis
    "HbO2x0.1Epix0.025", "HbO2x0.1Epix0.05", "HbO2x0.1Epix0.1",
    "HbO2x0.1Epix0.15", "HbO2x0.1Epix0.25", "HbO2x0.1Epix0.4",
    "HbO2x0.5Epix0.025", "HbO2x0.5Epix0.05", "HbO2x0.5Epix0.1",
    "HbO2x0.5Epix0.15", "HbO2x0.5Epix0.25", "HbO2x0.5Epix0.4",
    "HbO2x1.0Epix0.025", "HbO2x1.0Epix0.05", "HbO2x1.0Epix0.1",
    "HbO2x1.0Epix0.15", "HbO2x1.0Epix0.25", "HbO2x1.0Epix0.4"
]

# Lighting environments
LIGHTING_ENVIRONMENTS = [
    'rural_asphalt_road_4k', 'comfy_cafe_4k', 'reading_room_4k',
    'school_hall_4k', 'bathroom_4k', 'floral_tent_4k',
    'st_fagans_interior_4k', 'vulture_hide_4k', 'lapa_4k',
    'surgery_4k', 'veranda_4k', 'vintage_measuring_lab_4k',
    'yaris_interior_garage_4k', 'hospital_room_4k', 'bush_restaurant_4k',
    'lythwood_room_4k', 'kiara_interior_4k', 'reinforced_concrete_01_4k',
    'graffiti_shelter_4k'
]

# Skin tones (melanosome fraction) - values must match available .spd files
SKIN_TONES = {
    'very_light': 0.01,
    'light': 0.05,
    'light_medium': 0.11,
    'medium': 0.21,
    'medium_dark': 0.31,
    'dark': 0.41,
    'very_dark': 0.5  # Note: .spd file uses "0.5" not "0.50"
}

# Blood fractions
BLOOD_FRACTIONS = [0.002, 0.005, 0.02, 0.05]

# Hair albedos
HAIR_ALBEDOS = {
    'gray': [0.57, 0.57, 0.57],
    'light': [0.9, 0.9, 0.9],
    'brown': [0.84, 0.6328, 0.44]
}


def get_sensor(camera_height=15, fov=75, width=1024, height=1024,
               look_at=(0, 0, 0), angle_offset=(0, 0)):
    """
    Create camera sensor with optional angle offset for varied views.

    Args:
        camera_height: Height above skin in mm
        fov: Field of view in degrees
        width, height: Resolution
        look_at: Target point (x, y, z)
        angle_offset: (x_angle, z_angle) in degrees for tilted views
    """
    # Calculate camera position with angle offset
    import math
    x_off = camera_height * math.tan(math.radians(angle_offset[0]))
    z_off = camera_height * math.tan(math.radians(angle_offset[1]))

    origin = [x_off, camera_height, z_off]

    # Use ScalarTransform4f for compatibility with all variants
    sensor = mi.load_dict({
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(
            target=list(look_at),
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


def render_scene_gpu(scene_dict, sensor, spp=PAPER_SPP):
    """Render a scene using GPU if available."""
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, sensor=sensor, spp=spp)
    return np.array(image)


def create_full_scene(model_id, hair_model, lesion_id, lesion_mat, blood_frac,
                      melanin, time_point, light_name, hair_albedo,
                      lesion_directory, lesion_scale=1.5, lesion_offset=-2,
                      for_image=True):
    """
    Create a complete S-SYNTH scene dictionary.

    This is a GPU-compatible version that builds the scene dict directly.
    """
    uniform_scale = 1
    y_offset = -1.5
    room_size = 20
    xt_scale = 0.1

    # Hair optical properties
    mua_hair = 28.5
    mus_hair = 37.5
    ext_hair = mua_hair + mus_hair

    # Refractive indices
    ior_epi = 1.43
    ior_blood = 1.36
    ior_hypo = 1.44

    # Dermis IOR (wavelength dependent, normalized to 500nm)
    A, B, C = 1.3696, 3916.8, 2558.8
    ior_derm = A + (B / (500 ** 2)) + (C / (500 ** 4))

    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'volpathmis',
            'max_depth': 1000
        }
    }

    # Lesion
    lesion_file = f"{lesion_directory}/lesion{lesion_id}_T{time_point:03d}.obj"
    scene['lesion'] = {
        'type': 'obj',
        'filename': lesion_file,
        'to_world': mi.ScalarTransform4f.scale(uniform_scale * lesion_scale).translate([0, lesion_offset, 0])
    }

    if for_image:
        scene['lesion']['bsdf'] = {
            'type': 'roughdielectric',
            'alpha': 0.01,
            'int_ior': 1.32988 - (-3.97577e7) * 0.95902 ** 500,
            'ext_ior': 1.000277
        }
        scene['lesion']['interior'] = {
            'type': 'homogeneous',
            'albedo': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/{lesion_mat}_alb.spd'
            },
            'sigma_t': {
                'type': 'spectrum',
                'filename': config.sDir + f'opticalMaterials/{lesion_mat}_ext.spd'
            },
            'scale': xt_scale
        }

        # Hair (if enabled)
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

        # Vascular (blood network)
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

        # Hypodermis (subcutfat)
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

        # HDRI Environment lighting
        if light_name != 'diffuse':
            scene['envmap'] = {
                'type': 'envmap',
                'filename': config.sDir_hdri + light_name + '.exr',
                'scale': 1.0
            }
        else:
            # Diffuse lighting (area light from above)
            scene['shape_light'] = {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f.scale([room_size, 1, room_size]).translate([0, room_size, 0]).rotate([1, 0, 0], 90),
                'emitter': {
                    'type': 'area',
                    'radiance': {'type': 'd65', 'scale': 3}
                }
            }
    else:
        # Mask rendering (simple diffuse materials)
        scene['lesion']['bsdf'] = {
            'type': 'twosided',
            'material': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
            }
        }
        scene['epidermis'] = {
            'type': 'obj',
            'filename': config.sDir + f'outputModels/epidermis_{model_id:03d}.obj',
            'to_world': mi.ScalarTransform4f.scale(uniform_scale).translate([0, y_offset, 0]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.0, 0.0, 0.0]}
            }
        }
        scene['shape_light'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([room_size, 1, room_size]).translate([0, room_size, 0]).rotate([1, 0, 0], 90),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'd65', 'scale': 3}
            }
        }
        scene['wall_floor'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([room_size, 1, room_size]).translate([0, -room_size, 0]).rotate([1, 0, 0], -90),
            'bsdf': {
                'type': 'twosided',
                'material': {
                    'type': 'diffuse',
                    'reflectance': {'type': 'rgb', 'value': [0.0, 0.0, 0.0]}
                }
            }
        }

    return scene


def benchmark_single_render(spp=PAPER_SPP):
    """Benchmark a single render to measure GPU performance."""
    print(f"\n{'='*60}")
    print(f"Benchmarking single render (SPP={spp}, {PAPER_RESOLUTION[0]}x{PAPER_RESOLUTION[1]})")
    print(f"{'='*60}")

    # Use a simple scene for benchmarking
    model_id = 30
    lesion_version = 'ver1'
    lesion_id = 12
    time_point = 15
    lesion_mat = "HbO2x0.5Epix0.15"
    melanin = 0.21
    blood_frac = 0.02
    hair_model = 63
    hair_albedo = HAIR_ALBEDOS['brown']
    light_name = 'surgery_4k'

    # Create sensor
    sensor = get_sensor(CAMERA_HEIGHT, FOV, PAPER_RESOLUTION[0], PAPER_RESOLUTION[1])

    # Create scene
    scene_dict = create_full_scene(
        model_id, hair_model, lesion_id, lesion_mat, blood_frac,
        melanin, time_point, light_name, hair_albedo,
        LESION_VERSIONS[lesion_version]['path'],
        lesion_scale=1.5, lesion_offset=-2, for_image=True
    )

    # Compile and render
    print("\nCompiling scene...")
    start = time.time()
    scene = mi.load_dict(scene_dict)
    compile_time = time.time() - start
    print(f"Compilation time: {compile_time:.2f}s")

    print(f"\nRendering with SPP={spp}...")
    start = time.time()
    image = mi.render(scene, sensor=sensor, spp=spp)
    render_time = time.time() - start

    print(f"Render time: {render_time:.2f}s")
    print(f"Pixels/sec: {(PAPER_RESOLUTION[0] * PAPER_RESOLUTION[1]) / render_time:,.0f}")

    # Save benchmark image
    image_array = np.array(image)[:, :, :3]
    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(os.path.join(OUTPUT_DIR, "benchmark_render.png"))

    return render_time, compile_time


def explore_lesion_parameters():
    """Generate samples exploring different lesion parameters."""
    print(f"\n{'='*60}")
    print("Exploring Lesion Parameters")
    print(f"{'='*60}")

    output_subdir = os.path.join(OUTPUT_DIR, "lesion_exploration")
    os.makedirs(output_subdir, exist_ok=True)

    # Fixed parameters
    model_id = 30
    melanin = 0.21
    blood_frac = 0.02
    hair_model = -1  # No hair for clearer lesion visibility
    hair_albedo = HAIR_ALBEDOS['brown']
    light_name = 'surgery_4k'  # Medical lighting
    lesion_scale = 2.0

    samples = []

    # Explore ver0 (regular) vs ver1 (irregular) with different time points
    for version_name, version_info in LESION_VERSIONS.items():
        lesion_dir = version_info['path']
        time_points = version_info['time_points']
        lesion_ids = version_info['lesion_ids']

        # Sample a few combinations
        for lesion_id in lesion_ids[:3]:  # First 3 lesion shapes
            for tp in time_points[::2]:  # Every other time point
                sample_name = f"{version_name}_lesion{lesion_id}_T{tp:03d}"
                print(f"  Rendering {sample_name}...")

                sensor = get_sensor(CAMERA_HEIGHT, FOV, PAPER_RESOLUTION[0], PAPER_RESOLUTION[1])

                try:
                    scene_dict = create_full_scene(
                        model_id, hair_model, lesion_id, "HbO2x0.5Epix0.15",
                        blood_frac, melanin, tp, light_name, hair_albedo,
                        lesion_dir, lesion_scale=lesion_scale, lesion_offset=-2,
                        for_image=True
                    )

                    start = time.time()
                    scene = mi.load_dict(scene_dict)
                    image = mi.render(scene, sensor=sensor, spp=64)  # Lower SPP for exploration
                    render_time = time.time() - start

                    # Save
                    image_array = np.array(image)[:, :, :3]
                    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
                    img_path = os.path.join(output_subdir, f"{sample_name}.png")
                    Image.fromarray(image_uint8).save(img_path)

                    samples.append({
                        'name': sample_name,
                        'version': version_name,
                        'lesion_id': lesion_id,
                        'time_point': tp,
                        'render_time': render_time,
                        'path': img_path
                    })
                    print(f"    Done in {render_time:.1f}s")

                except Exception as e:
                    print(f"    Failed: {e}")

    return samples


def explore_lighting():
    """Generate samples with different lighting environments."""
    print(f"\n{'='*60}")
    print("Exploring Lighting Environments")
    print(f"{'='*60}")

    output_subdir = os.path.join(OUTPUT_DIR, "lighting_exploration")
    os.makedirs(output_subdir, exist_ok=True)

    # Fixed parameters
    model_id = 30
    lesion_id = 12
    time_point = 15
    melanin = 0.21
    blood_frac = 0.02
    hair_model = 63
    hair_albedo = HAIR_ALBEDOS['brown']
    lesion_mat = "HbO2x0.5Epix0.15"

    samples = []

    for light_name in LIGHTING_ENVIRONMENTS[:6]:  # First 6 environments
        print(f"  Rendering with {light_name}...")

        sensor = get_sensor(CAMERA_HEIGHT, FOV, PAPER_RESOLUTION[0], PAPER_RESOLUTION[1])

        try:
            scene_dict = create_full_scene(
                model_id, hair_model, lesion_id, lesion_mat,
                blood_frac, melanin, time_point, light_name, hair_albedo,
                LESION_VERSIONS['ver1']['path'],
                lesion_scale=1.5, lesion_offset=-2, for_image=True
            )

            start = time.time()
            scene = mi.load_dict(scene_dict)
            image = mi.render(scene, sensor=sensor, spp=64)
            render_time = time.time() - start

            # Save
            image_array = np.array(image)[:, :, :3]
            image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
            img_path = os.path.join(output_subdir, f"light_{light_name}.png")
            Image.fromarray(image_uint8).save(img_path)

            samples.append({
                'light': light_name,
                'render_time': render_time,
                'path': img_path
            })
            print(f"    Done in {render_time:.1f}s")

        except Exception as e:
            print(f"    Failed: {e}")

    return samples


def test_parallel_rendering(num_scenes=4):
    """
    Test parallel rendering capabilities.

    On GPU, we test:
    1. Sequential rendering (baseline)
    2. Note: Mitsuba 3 GPU rendering is inherently parallel within a single render
    """
    print(f"\n{'='*60}")
    print(f"Testing Rendering Throughput ({num_scenes} scenes)")
    print(f"{'='*60}")

    # Create test scenes with different parameters
    test_params = [
        {'melanin': 0.05, 'light': 'surgery_4k', 'lesion_id': 12},
        {'melanin': 0.21, 'light': 'hospital_room_4k', 'lesion_id': 15},
        {'melanin': 0.41, 'light': 'reading_room_4k', 'lesion_id': 10},
        {'melanin': 0.31, 'light': 'bathroom_4k', 'lesion_id': 18},
    ][:num_scenes]

    model_id = 30
    blood_frac = 0.02
    hair_model = 63
    hair_albedo = HAIR_ALBEDOS['brown']
    time_point = 15
    lesion_mat = "HbO2x0.5Epix0.15"

    # Sequential rendering
    print(f"\nSequential rendering (SPP={PAPER_SPP})...")
    total_start = time.time()
    render_times = []

    for i, params in enumerate(test_params):
        sensor = get_sensor(CAMERA_HEIGHT, FOV, PAPER_RESOLUTION[0], PAPER_RESOLUTION[1])

        scene_dict = create_full_scene(
            model_id, hair_model, params['lesion_id'], lesion_mat,
            blood_frac, params['melanin'], time_point, params['light'],
            hair_albedo, LESION_VERSIONS['ver1']['path'],
            lesion_scale=1.5, lesion_offset=-2, for_image=True
        )

        start = time.time()
        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, sensor=sensor, spp=PAPER_SPP)
        render_time = time.time() - start
        render_times.append(render_time)

        print(f"  Scene {i+1}: {render_time:.2f}s")

        # Save
        image_array = np.array(image)[:, :, :3]
        image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image_uint8).save(os.path.join(OUTPUT_DIR, f"parallel_test_{i}.png"))

    total_time = time.time() - total_start

    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per scene: {np.mean(render_times):.2f}s")
    print(f"  Throughput: {num_scenes / total_time * 3600:.1f} images/hour")
    print(f"  Estimated time for 10K images: {10000 * np.mean(render_times) / 3600:.1f} hours")

    return {
        'num_scenes': num_scenes,
        'total_time': total_time,
        'avg_time': np.mean(render_times),
        'throughput_per_hour': num_scenes / total_time * 3600
    }


def print_moge2_resolution_analysis():
    """Analyze and recommend resolution for MoGe-2 training."""
    print(f"\n{'='*60}")
    print("MoGe-2 Training Resolution Analysis")
    print(f"{'='*60}")

    print("""
MoGe-2 Training Configuration (from v2.json):
- area_range: [250000, 1000000] pixels
  - 250,000 = 500x500
  - 1,000,000 = 1000x1000

- aspect_ratio_range: [0.5, 2.0]
  - Supports various aspect ratios

S-SYNTH Paper Settings:
- Resolution: 1024x1024 = 1,048,576 pixels
- This is slightly above MoGe's area_range upper bound

RECOMMENDATIONS:
1. Render at 1024x1024 (paper settings) for highest quality
2. MoGe training will randomly crop/resize within area_range
3. Higher resolution source images = better quality after augmentation
4. For fine-tuning, consider matching S-SYNTH's 1024x1024 exactly

For MoGe-2 fine-tuning config, use:
  "area_range": [900000, 1100000]  # ~950x950 to ~1050x1050
  "aspect_ratio_range": [0.95, 1.05]  # Nearly square (matching S-SYNTH)
""")


def generate_fig1b_samples():
    """
    Generate samples matching Fig 1b from the paper:
    - Camera angle variations (top-down and tilted views)
    - Different lesion morphologies:
      - Red lesions (high HbO2, low melanin)
      - Dark lesions (low HbO2, high melanin)
      - Raised/bumpy (late time points)
      - Flat (early time points)
      - Round vs irregular shapes
    """
    print(f"\n{'='*60}")
    print("Generating Fig 1b Style Samples")
    print(f"{'='*60}")

    output_subdir = os.path.join(OUTPUT_DIR, "fig1b_samples")
    os.makedirs(output_subdir, exist_ok=True)

    # Fixed parameters
    model_id = 30
    blood_frac = 0.02
    hair_model = 50  # With hair for realism
    light_name = 'surgery_4k'  # Medical lighting (bright, even)

    # Camera angles to test (x_angle, z_angle in degrees)
    CAMERA_ANGLES = {
        'top_down': (0, 0),       # Looking straight down
        'tilted_15': (15, 0),     # 15 degree tilt
        'tilted_30': (30, 0),     # 30 degree tilt (like fig 1b bottom)
        'tilted_45': (45, 0),     # 45 degree tilt (dramatic angle)
        'corner_view': (25, 25),  # Tilted in both directions
    }

    # Lesion appearance configurations matching Fig 1b
    LESION_CONFIGS = [
        # Red, raised lesions (high HbO2, late time point)
        {
            'name': 'red_raised',
            'lesion_mat': 'HbO2x1.0Epix0.025',  # High blood, low melanin = RED
            'melanin': 0.11,  # Light-medium skin to show red color
            'time_point': 30,
            'version': 'ver1',
            'lesion_id': 12,
            'lesion_scale': 2.0,
            'description': 'Red raised lesion (high HbO2)'
        },
        # Dark, flat lesion (high melanin, early time point)
        {
            'name': 'dark_flat',
            'lesion_mat': 'HbO2x0.1Epix0.4',  # Low blood, high melanin = DARK
            'melanin': 0.21,  # Medium skin
            'time_point': 10,
            'version': 'ver1',
            'lesion_id': 15,
            'lesion_scale': 1.5,
            'description': 'Dark flat lesion (high melanin)'
        },
        # Pink/reddish medium lesion
        {
            'name': 'pink_medium',
            'lesion_mat': 'HbO2x0.5Epix0.1',  # Medium blood, medium melanin
            'melanin': 0.05,  # Light skin to show pink
            'time_point': 20,
            'version': 'ver0',  # Smoother boundary
            'lesion_id': 11,
            'lesion_scale': 1.8,
            'description': 'Pink medium lesion'
        },
        # Brown raised bumpy lesion
        {
            'name': 'brown_bumpy',
            'lesion_mat': 'HbO2x0.1Epix0.25',  # Brown coloring
            'melanin': 0.21,
            'time_point': 50,  # Very late = very raised
            'version': 'ver0',
            'lesion_id': 10,
            'lesion_scale': 2.5,
            'description': 'Brown raised bumpy lesion'
        },
        # Small early lesion (barely visible)
        {
            'name': 'small_early',
            'lesion_mat': 'HbO2x0.5Epix0.15',
            'melanin': 0.31,  # Medium-dark skin
            'time_point': 2,  # Very early
            'version': 'ver1',
            'lesion_id': 14,
            'lesion_scale': 1.2,
            'description': 'Small early lesion'
        },
    ]

    samples = []
    spp = 64  # Lower SPP for faster exploration

    # Generate samples for each lesion config with different camera angles
    for lesion_cfg in LESION_CONFIGS:
        version_info = LESION_VERSIONS[lesion_cfg['version']]
        lesion_dir = version_info['path']

        for angle_name, (x_angle, z_angle) in CAMERA_ANGLES.items():
            sample_name = f"{lesion_cfg['name']}_{angle_name}"
            print(f"  Rendering {sample_name}...")

            try:
                # Create sensor with angle
                sensor = get_sensor(
                    camera_height=CAMERA_HEIGHT,
                    fov=FOV,
                    width=PAPER_RESOLUTION[0],
                    height=PAPER_RESOLUTION[1],
                    angle_offset=(x_angle, z_angle)
                )

                # Create scene
                scene_dict = create_full_scene(
                    model_id, hair_model, lesion_cfg['lesion_id'],
                    lesion_cfg['lesion_mat'], blood_frac, lesion_cfg['melanin'],
                    lesion_cfg['time_point'], light_name, HAIR_ALBEDOS['brown'],
                    lesion_dir, lesion_scale=lesion_cfg['lesion_scale'],
                    lesion_offset=-2, for_image=True
                )

                start = time.time()
                scene = mi.load_dict(scene_dict)
                image = mi.render(scene, sensor=sensor, spp=spp)
                render_time = time.time() - start

                # Save
                image_array = np.array(image)[:, :, :3]
                image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
                img_path = os.path.join(output_subdir, f"{sample_name}.png")
                Image.fromarray(image_uint8).save(img_path)

                samples.append({
                    'name': sample_name,
                    'lesion_config': lesion_cfg['name'],
                    'camera_angle': angle_name,
                    'render_time': render_time,
                    'path': img_path,
                    'description': lesion_cfg['description']
                })
                print(f"    Done in {render_time:.1f}s")

            except Exception as e:
                print(f"    Failed: {e}")

    # Create visualization grid
    create_fig1b_visualization(samples, output_subdir)

    return samples


def create_fig1b_visualization(samples, output_dir):
    """Create a visualization grid matching Fig 1b layout."""
    print("\n  Creating Fig 1b visualization grid...")

    # Group samples by lesion config
    lesion_configs = {}
    for s in samples:
        cfg = s['lesion_config']
        if cfg not in lesion_configs:
            lesion_configs[cfg] = {}
        lesion_configs[cfg][s['camera_angle']] = s

    # Create grid: rows = lesion types, cols = camera angles
    configs = list(lesion_configs.keys())
    angles = ['top_down', 'tilted_30', 'tilted_45']  # Select 3 angles for display

    fig, axes = plt.subplots(len(configs), len(angles), figsize=(15, 25))
    fig.suptitle("S-SYNTH Fig 1b Style: Lesion Morphology × Camera Angle",
                 fontsize=16, fontweight='bold')

    for row, cfg in enumerate(configs):
        for col, angle in enumerate(angles):
            ax = axes[row, col] if len(configs) > 1 else axes[col]

            if cfg in lesion_configs and angle in lesion_configs[cfg]:
                sample = lesion_configs[cfg][angle]
                img = Image.open(sample['path'])
                ax.imshow(img)

                if col == 0:
                    ax.set_ylabel(sample['description'], fontsize=10, fontweight='bold')
                if row == 0:
                    ax.set_title(f"Camera: {angle.replace('_', ' ')}", fontsize=11)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')

            ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "fig1b_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {output_path}")

    # Also create a side-by-side view comparison (top-down vs tilted)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Scene Simulation Views (matching Fig 1b)", fontsize=16, fontweight='bold')

    # Find red_raised samples for the main demo
    demo_samples = [s for s in samples if s['lesion_config'] == 'red_raised']

    view_pairs = [
        ('top_down', 'Top-Down View'),
        ('tilted_30', '30° Tilted View'),
        ('tilted_45', '45° Tilted View'),
    ]

    for col, (angle, title) in enumerate(view_pairs):
        matching = [s for s in demo_samples if s['camera_angle'] == angle]
        if matching:
            img = Image.open(matching[0]['path'])
            axes[0, col].imshow(img)
            axes[0, col].set_title(f"Red Raised Lesion\n{title}", fontsize=11)
            axes[0, col].axis('off')

    # Dark flat for comparison
    demo_samples2 = [s for s in samples if s['lesion_config'] == 'dark_flat']
    for col, (angle, title) in enumerate(view_pairs):
        matching = [s for s in demo_samples2 if s['camera_angle'] == angle]
        if matching:
            img = Image.open(matching[0]['path'])
            axes[1, col].imshow(img)
            axes[1, col].set_title(f"Dark Flat Lesion\n{title}", fontsize=11)
            axes[1, col].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "camera_angle_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved camera comparison: {output_path}")


def main():
    print("=" * 60)
    print("S-SYNTH GPU Rendering Exploration")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Variant: {mi.variant()}")
    print("=" * 60)

    # 1. Benchmark single render
    render_time, compile_time = benchmark_single_render(spp=PAPER_SPP)

    # 2. Explore lesion parameters
    lesion_samples = explore_lesion_parameters()

    # 3. Explore lighting
    lighting_samples = explore_lighting()

    # 4. Test parallel rendering throughput
    throughput = test_parallel_rendering(num_scenes=4)

    # 5. Generate Fig 1b style samples
    fig1b_samples = generate_fig1b_samples()

    # 6. MoGe-2 resolution analysis
    print_moge2_resolution_analysis()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"GPU: {'Yes (CUDA)' if GPU_AVAILABLE else 'No (CPU)'}")
    print(f"Single render time (SPP=124): {render_time:.2f}s")
    print(f"Throughput: {throughput['throughput_per_hour']:.0f} images/hour")
    print(f"Lesion samples generated: {len(lesion_samples)}")
    print(f"Lighting samples generated: {len(lighting_samples)}")
    print(f"Fig 1b samples generated: {len(fig1b_samples)}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
