#!/usr/bin/env python3
"""
Visualize S-SYNTH pre-rendered samples showing skin tone and parameter diversity.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re

# Paths
DATA_DIR = "/workspace/hector/ssynth-release/data/synthetic_dataset/mel_variation_selected/output/output"
OUTPUT_DIR = "/workspace/hector/ssynth-release/test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_path(path):
    """Extract parameters from S-SYNTH folder path."""
    params = {}

    # Extract skin model
    match = re.search(r'skin_(\d+)', path)
    if match:
        params['skin_model'] = int(match.group(1))

    # Extract hair model
    match = re.search(r'hairModel_(\d+)', path)
    if match:
        params['hair_model'] = int(match.group(1))

    # Extract melanin level (skin tone)
    match = re.search(r'mel_([\d.]+)', path)
    if match:
        params['melanin'] = float(match.group(1))

    # Extract blood fraction
    match = re.search(r'fB_([\d.]+)', path)
    if match:
        params['blood_fraction'] = float(match.group(1))

    # Extract lesion ID
    match = re.search(r'lesion_(\d+)', path)
    if match:
        params['lesion_id'] = int(match.group(1))

    # Extract time point
    match = re.search(r'T_(\d+)', path)
    if match:
        params['time_point'] = int(match.group(1))

    # Extract lesion material
    match = re.search(r'(HbO2x[\d.]+Epix[\d.]+|melDermEpi)', path)
    if match:
        params['lesion_material'] = match.group(1)

    # Extract hair albedo
    match = re.search(r'hairAlb_([\d.]+-[\d.]+-[\d.]+)', path)
    if match:
        params['hair_albedo'] = match.group(1)

    # Extract light
    match = re.search(r'light_([^/]+)', path)
    if match:
        params['light'] = match.group(1)

    return params


def get_skin_tone_label(melanin):
    """Convert melanin fraction to skin tone label."""
    if melanin <= 0.1:
        return "Light"
    elif melanin <= 0.25:
        return "Medium"
    else:
        return "Dark"


def find_diverse_samples():
    """Find samples with diverse parameters."""
    # Find all image files
    image_files = glob.glob(f"{DATA_DIR}/**/image.png", recursive=True)

    print(f"Found {len(image_files)} images")

    # Parse all paths
    samples = []
    for img_path in image_files:
        params = parse_path(img_path)
        params['image_path'] = img_path
        params['mask_path'] = img_path.replace('image.png', 'mask.png')
        samples.append(params)

    # Get unique melanin levels
    melanin_levels = sorted(set(s.get('melanin', 0) for s in samples))
    print(f"Melanin levels: {melanin_levels}")

    return samples, melanin_levels


def create_skin_tone_grid(samples, melanin_levels):
    """Create a grid showing skin tone variation."""
    # Select 3 melanin levels: light, medium, dark
    light_mel = min(m for m in melanin_levels if m <= 0.1) if any(m <= 0.1 for m in melanin_levels) else melanin_levels[0]
    dark_mel = max(m for m in melanin_levels if m >= 0.35) if any(m >= 0.35 for m in melanin_levels) else melanin_levels[-1]
    medium_mel = min(melanin_levels, key=lambda x: abs(x - 0.21))

    selected_mels = [light_mel, medium_mel, dark_mel]
    print(f"Selected melanin levels: {selected_mels}")

    # For each skin tone, find 3 different conditions
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("S-SYNTH Skin Tone & Condition Diversity", fontsize=16, fontweight='bold')

    conditions = [
        ("Low Blood", lambda s: s.get('blood_fraction', 0) <= 0.005),
        ("Medium Blood", lambda s: 0.005 < s.get('blood_fraction', 0) <= 0.02),
        ("High Blood", lambda s: s.get('blood_fraction', 0) > 0.02),
    ]

    for row, mel in enumerate(selected_mels):
        skin_tone = get_skin_tone_label(mel)
        mel_samples = [s for s in samples if abs(s.get('melanin', 0) - mel) < 0.01]

        for col, (cond_name, cond_func) in enumerate(conditions):
            ax = axes[row, col]

            # Find matching sample
            matching = [s for s in mel_samples if cond_func(s)]

            if matching:
                sample = matching[0]

                # Load and display image
                img = Image.open(sample['image_path'])
                ax.imshow(img)

                # Create title with parameters
                title = f"{skin_tone} Skin (mel={mel:.0%})\n{cond_name}"
                if 'hair_model' in sample and sample['hair_model'] >= 0:
                    title += f"\nWith Hair (model {sample['hair_model']})"
                else:
                    title += "\nNo Hair"

                ax.set_title(title, fontsize=10)
            else:
                ax.text(0.5, 0.5, f"No sample\n{skin_tone}\n{cond_name}",
                       ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "skin_tone_diversity.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_parameter_showcase(samples):
    """Create a detailed showcase of all parameters."""
    # Find samples with diverse parameters
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("S-SYNTH Parameter Showcase", fontsize=16, fontweight='bold')

    # Row 0: Skin tones (light, medium-light, medium, medium-dark, dark)
    # Row 1: Different lesion materials/sizes
    # Row 2: With/without hair, different lighting

    # Get unique parameter values
    all_mels = sorted(set(s.get('melanin', 0) for s in samples))
    all_bloods = sorted(set(s.get('blood_fraction', 0) for s in samples))
    all_lesion_mats = sorted(set(s.get('lesion_material', '') for s in samples))

    # Row 0: Skin tones
    for col, mel in enumerate(all_mels[:4]):
        ax = axes[0, col]
        matching = [s for s in samples if abs(s.get('melanin', 0) - mel) < 0.01]
        if matching:
            img = Image.open(matching[0]['image_path'])
            ax.imshow(img)
            ax.set_title(f"Melanin: {mel:.0%}\n({get_skin_tone_label(mel)} Skin)", fontsize=10)
        ax.axis('off')

    # Row 1: Different blood fractions on medium skin
    medium_mel = min(all_mels, key=lambda x: abs(x - 0.21))
    med_samples = [s for s in samples if abs(s.get('melanin', 0) - medium_mel) < 0.01]

    for col, bf in enumerate(all_bloods[:4]):
        ax = axes[1, col]
        matching = [s for s in med_samples if abs(s.get('blood_fraction', 0) - bf) < 0.001]
        if matching:
            img = Image.open(matching[0]['image_path'])
            ax.imshow(img)
            lesion_mat = matching[0].get('lesion_material', 'N/A')
            ax.set_title(f"Blood Fraction: {bf}\nLesion: {lesion_mat[:15]}...", fontsize=9)
        ax.axis('off')

    # Row 2: Show image + mask pairs
    for col in range(4):
        ax = axes[2, col]
        if col < len(samples):
            sample = samples[col * 2]
            if col % 2 == 0:
                img = Image.open(sample['image_path'])
                ax.set_title(f"RGB Image\nLesion {sample.get('lesion_id', 'N/A')}", fontsize=10)
            else:
                mask_path = sample['mask_path']
                if os.path.exists(mask_path):
                    img = Image.open(mask_path)
                    ax.set_title("Segmentation Mask", fontsize=10)
                else:
                    img = Image.open(sample['image_path'])
                    ax.set_title("(No mask found)", fontsize=10)
            ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "parameter_showcase.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_3x3_grid_diverse(samples, melanin_levels):
    """
    Create 3x3 grid with:
    - Rows: Light, Medium, Dark skin
    - Cols: Different conditions
    - 50% with hair annotation
    """
    # Select representative melanin levels
    light_mel = min(m for m in melanin_levels if m <= 0.1) if any(m <= 0.1 for m in melanin_levels) else melanin_levels[0]
    dark_mel = max(m for m in melanin_levels if m >= 0.35) if any(m >= 0.35 for m in melanin_levels) else melanin_levels[-1]
    medium_mel = min(melanin_levels, key=lambda x: abs(x - 0.21))

    selected_mels = [light_mel, medium_mel, dark_mel]
    tone_names = ["Light", "Medium", "Dark"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("S-SYNTH: 3×3 Diversity Grid\n(Skin Tone × Blood Fraction)",
                 fontsize=16, fontweight='bold')

    # Column labels (conditions)
    blood_conditions = [
        (0.005, "Low Blood (0.5%)"),
        (0.02, "Medium Blood (2%)"),
        (0.05, "High Blood (5%)"),
    ]

    for row, (mel, tone) in enumerate(zip(selected_mels, tone_names)):
        mel_samples = [s for s in samples if abs(s.get('melanin', 0) - mel) < 0.01]

        for col, (target_blood, blood_label) in enumerate(blood_conditions):
            ax = axes[row, col]

            # Find sample closest to target blood fraction
            matching = sorted(mel_samples,
                            key=lambda s: abs(s.get('blood_fraction', 0) - target_blood))

            if matching:
                sample = matching[0]

                # Load image
                img = Image.open(sample['image_path'])
                ax.imshow(img)

                # Build detailed title
                has_hair = sample.get('hair_model', -1) >= 0
                hair_str = "With Hair" if has_hair else "No Hair"

                title = f"{tone} Skin (mel={mel:.0%})\n"
                title += f"{blood_label}\n"
                title += f"{hair_str}"

                # Add lesion info
                lesion_id = sample.get('lesion_id', 'N/A')
                time_pt = sample.get('time_point', 'N/A')
                title += f" | Lesion #{lesion_id}"

                ax.set_title(title, fontsize=9)

                # Add border color based on hair
                border_color = 'green' if has_hair else 'gray'
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            else:
                ax.text(0.5, 0.5, f"No sample", ha='center', va='center')

            ax.set_xticks([])
            ax.set_yticks([])

    # Add column headers
    for col, (_, blood_label) in enumerate(blood_conditions):
        axes[0, col].set_xlabel('')

    # Add legend
    fig.text(0.5, 0.02, "Green border = Has Hair | Gray border = No Hair",
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "ssynth_3x3_diversity.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def show_single_sample_detail(samples):
    """Show a single sample in detail with all its parameters."""
    sample = samples[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("S-SYNTH Sample Detail", fontsize=14, fontweight='bold')

    # Image
    img = Image.open(sample['image_path'])
    axes[0].imshow(img)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')

    # Mask
    if os.path.exists(sample['mask_path']):
        mask = Image.open(sample['mask_path'])
        axes[1].imshow(mask)
        axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')

    # Parameters text
    axes[2].axis('off')
    params_text = "Parameters:\n\n"
    for key, value in sorted(sample.items()):
        if key not in ['image_path', 'mask_path']:
            params_text += f"• {key}: {value}\n"

    axes[2].text(0.1, 0.9, params_text, transform=axes[2].transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].set_title("Extracted Parameters")

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "sample_detail.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("S-SYNTH Sample Visualization")
    print("=" * 60)

    samples, melanin_levels = find_diverse_samples()

    if samples:
        print(f"\nCreating visualizations...")
        show_single_sample_detail(samples)
        create_3x3_grid_diverse(samples, melanin_levels)
        create_parameter_showcase(samples)

        print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    else:
        print("No samples found!")
