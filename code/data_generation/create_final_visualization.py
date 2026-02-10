#!/usr/bin/env python3
"""
Create a final comparison visualization showing:
1. Light/Medium/Dark skin tone
2. With/Without hair comparison
3. RGB + Depth + Mask for each
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_utils import load_depth_moge

OUTPUT_DIR = "/workspace/hector/ssynth-release/test_output/diverse_grid"
FINAL_OUTPUT = "/workspace/hector/ssynth-release/test_output"


def create_skin_tone_comparison():
    """Create 3x4 grid: Light/Medium/Dark × RGB/Depth/Mask/Params"""

    samples = [
        ("r0_c1_Light_Medium_Lesion", "Light Skin (5% melanin)"),
        ("r1_c1_Medium_Medium_Lesion", "Medium Skin (25% melanin)"),
        ("r2_c1_Dark_Medium_Lesion", "Dark Skin (45% melanin)"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("S-SYNTH: Skin Tone Comparison with Depth Maps\n(Same lesion size and type across skin tones)",
                 fontsize=16, fontweight='bold')

    for row, (folder, label) in enumerate(samples):
        base_path = os.path.join(OUTPUT_DIR, folder)

        # Load files
        img = Image.open(os.path.join(base_path, "image.png"))
        mask = Image.open(os.path.join(base_path, "mask.png"))
        depth, meta = load_depth_moge(os.path.join(base_path, "depth.png"))

        with open(os.path.join(base_path, "meta.json")) as f:
            params = json.load(f)

        # RGB
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{label}\nRGB Image", fontsize=11)
        axes[row, 0].axis('off')

        # Mask
        axes[row, 1].imshow(mask, cmap='gray')
        axes[row, 1].set_title("Segmentation Mask", fontsize=11)
        axes[row, 1].axis('off')

        # Depth
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        vmin, vmax = valid_depth.min(), valid_depth.max()
        im = axes[row, 2].imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, 2].set_title(f"Depth Map ({vmin:.1f}-{vmax:.1f}mm)", fontsize=11)
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Parameters
        axes[row, 3].axis('off')
        has_hair = params.get('has_hair', False)
        param_text = f"Parameters:\n\n"
        param_text += f"• Melanin: {params.get('melanin', 'N/A'):.0%}\n"
        param_text += f"• Blood fraction: {params.get('blood_fraction', 'N/A')}\n"
        param_text += f"• Lesion material: {params.get('lesion_material', 'N/A')}\n"
        param_text += f"• Lesion scale: {params.get('lesion_scale', 'N/A')}\n"
        param_text += f"• Hair: {'Yes (model ' + str(params.get('hair_model', 'N/A')) + ')' if has_hair else 'No'}\n"
        param_text += f"• Camera height: {params.get('camera_height', 'N/A')}mm\n"

        axes[row, 3].text(0.1, 0.9, param_text, transform=axes[row, 3].transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[row, 3].set_title("Parameters", fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(FINAL_OUTPUT, "skin_tone_comparison_with_depth.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_hair_comparison():
    """Show comparison with/without hair on same skin tone."""

    # Find samples with and without hair for each skin tone
    samples_with_hair = []
    samples_no_hair = []

    for folder in os.listdir(OUTPUT_DIR):
        if not folder.startswith('r'):
            continue

        base_path = os.path.join(OUTPUT_DIR, folder)
        meta_path = os.path.join(base_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                params = json.load(f)
            if params.get('has_hair', False):
                samples_with_hair.append((folder, params))
            else:
                samples_no_hair.append((folder, params))

    # Create comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("S-SYNTH: Hair vs No Hair Comparison\n(Effect of hair follicles on skin appearance)",
                 fontsize=16, fontweight='bold')

    # Row 0: With hair
    if samples_with_hair:
        folder, params = samples_with_hair[0]
        base_path = os.path.join(OUTPUT_DIR, folder)

        img = Image.open(os.path.join(base_path, "image.png"))
        mask = Image.open(os.path.join(base_path, "mask.png"))
        depth, _ = load_depth_moge(os.path.join(base_path, "depth.png"))

        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f"WITH HAIR (Model {params.get('hair_model', 'N/A')})\n{params.get('skin_tone', '')} Skin", fontsize=11)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title("Segmentation Mask", fontsize=11)
        axes[0, 1].axis('off')

        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        vmin, vmax = valid_depth.min(), valid_depth.max()
        im = axes[0, 2].imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title(f"Depth ({vmin:.1f}-{vmax:.1f}mm)", fontsize=11)
        axes[0, 2].axis('off')

        # Zoomed in on hair
        img_arr = np.array(img)
        axes[0, 3].imshow(img_arr[200:500, 200:500])
        axes[0, 3].set_title("Zoomed: Hair Follicles Visible", fontsize=11)
        axes[0, 3].axis('off')

    # Row 1: Without hair
    if samples_no_hair:
        folder, params = samples_no_hair[0]
        base_path = os.path.join(OUTPUT_DIR, folder)

        img = Image.open(os.path.join(base_path, "image.png"))
        mask = Image.open(os.path.join(base_path, "mask.png"))
        depth, _ = load_depth_moge(os.path.join(base_path, "depth.png"))

        axes[1, 0].imshow(img)
        axes[1, 0].set_title(f"NO HAIR\n{params.get('skin_tone', '')} Skin", fontsize=11)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title("Segmentation Mask", fontsize=11)
        axes[1, 1].axis('off')

        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        vmin, vmax = valid_depth.min(), valid_depth.max()
        im = axes[1, 2].imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 2].set_title(f"Depth ({vmin:.1f}-{vmax:.1f}mm)", fontsize=11)
        axes[1, 2].axis('off')

        # Zoomed in - no hair
        img_arr = np.array(img)
        axes[1, 3].imshow(img_arr[200:500, 200:500])
        axes[1, 3].set_title("Zoomed: No Hair Follicles", fontsize=11)
        axes[1, 3].axis('off')

    plt.tight_layout()
    output_path = os.path.join(FINAL_OUTPUT, "hair_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_depth_3d_visualization():
    """Create 3D visualization of depth map."""

    sample_path = os.path.join(OUTPUT_DIR, "r1_c1_Medium_Medium_Lesion")
    depth, _ = load_depth_moge(os.path.join(sample_path, "depth.png"))
    img = Image.open(os.path.join(sample_path, "image.png"))

    fig = plt.figure(figsize=(16, 8))

    # 2D depth map
    ax1 = fig.add_subplot(1, 2, 1)
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    vmin, vmax = valid_depth.min(), valid_depth.max()
    im = ax1.imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title("Depth Map (2D View)", fontsize=14)
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

    # Get RGB colors for surface
    img_arr = np.array(img.resize((w//step, h//step))) / 255.0
    colors = img_arr[::1, ::1, :3]

    # Plot surface with colors
    ax2.plot_surface(X, Z, -Y_plot, facecolors=colors, alpha=0.9)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Z (pixels)')
    ax2.set_zlabel('Depth (mm)')
    ax2.set_title("3D Depth Surface with RGB Texture", fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(FINAL_OUTPUT, "depth_3d_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_figure():
    """Create a summary figure showing all parameters."""

    print("\n" + "="*60)
    print("S-SYNTH Parameter Summary")
    print("="*60)

    print("""
S-SYNTH supports the following parameters for synthetic skin generation:

SKIN APPEARANCE:
  • Melanin fraction (0.01-0.50): Controls skin tone
    - Light skin: 0.01-0.10
    - Medium skin: 0.15-0.30
    - Dark skin: 0.35-0.50

  • Blood fraction (0.002-0.05): Controls skin redness/vascularity
    - Low: 0.002 (pale)
    - Medium: 0.02 (normal)
    - High: 0.05 (flushed/irritated)

HAIR:
  • Hair model (0-99 or -1): 3D hair follicle geometry
    - -1 = no hair
    - 0-99 = different hair patterns

  • Hair albedo (3 presets):
    - Gray [0.57, 0.57, 0.57]
    - Light [0.9, 0.9, 0.9]
    - Brown [0.84, 0.63, 0.44]

LESION:
  • Lesion ID (1-20): Different lesion shapes
  • Lesion scale (1.0-3.0): Size multiplier
  • Lesion material (19 types): Optical properties
    - HbO2 levels: 0.1, 0.5, 1.0
    - Epidermis melanin: 0.025-0.4
  • Time point (15-55): Growth stage

GEOMETRY:
  • Skin model (0-99): Different skin surface textures
  • Camera height (10-20mm): Viewing distance

LIGHTING:
  • 20 HDRI environments including:
    - Medical settings (surgery, hospital)
    - Indoor scenes (cafe, bathroom)
    - Outdoor scenes (road, shelter)
    """)


if __name__ == "__main__":
    create_summary_figure()
    create_skin_tone_comparison()
    create_hair_comparison()
    create_depth_3d_visualization()

    print(f"\nAll visualizations saved to: {FINAL_OUTPUT}")
