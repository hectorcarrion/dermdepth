#!/usr/bin/env python3
"""
DDI Ruler Annotation Tool.

Allows annotating ruler markings in DDI images for metric scale evaluation.
Records pixel coordinates of ruler endpoints and their real-world distance.

Two modes:
1. Interactive (matplotlib): click two ruler points, enter real distance
2. Batch from JSON: load pre-annotated ruler positions

Output format:
{
    "annotations": [
        {
            "image_id": "000001",
            "DDI_file": "000001.png",
            "point1_px": [x1, y1],
            "point2_px": [x2, y2],
            "real_distance_mm": 10.0,
            "pixel_distance": 142.3,
            "skin_tone": "56"
        }
    ]
}

Usage:
    # Interactive annotation
    python annotate_ddi_rulers.py --mode interactive --output annotations.json

    # List images likely containing rulers (for pre-screening)
    python annotate_ddi_rulers.py --mode screen --output ruler_candidates.json
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def screen_for_rulers(ddi_dir, output_path):
    """
    Screen DDI images for likely ruler presence using simple heuristics.

    Heuristics:
    - Linear edge patterns (rulers tend to have strong straight edges)
    - High-contrast thin lines
    - Specific color patterns (black/white ruler markings)

    This is a rough pre-screening -- manual verification needed.
    """
    print("Screening DDI images for rulers...")

    csv_path = ddi_dir / "map.csv"
    metadata = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    images_dir = ddi_dir / "images"
    candidates = []

    for i, row in enumerate(metadata):
        img_path = images_dir / row['DDI_file']
        if not img_path.exists():
            continue

        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            h, w = img.shape[:2]

            # Heuristic 1: Check for high-contrast thin features near edges
            # Rulers are often at the borders of the image
            border_width = max(h, w) // 10

            # Check all four borders
            borders = [
                img[:border_width, :, :],      # top
                img[-border_width:, :, :],      # bottom
                img[:, :border_width, :],       # left
                img[:, -border_width:, :],      # right
            ]

            max_contrast = 0
            for border in borders:
                if border.size == 0:
                    continue
                gray = border.mean(axis=-1)
                # Local contrast: std in small patches
                contrast = gray.std()
                max_contrast = max(max_contrast, contrast)

            # Heuristic 2: Very high std in grayscale suggests markings
            gray_full = img.mean(axis=-1)
            has_markings = gray_full.std() > 60

            # Score
            score = max_contrast / 100.0 + (0.5 if has_markings else 0)

            if score > 0.5:
                candidates.append({
                    'DDI_file': row['DDI_file'],
                    'DDI_ID': row.get('DDI_ID', ''),
                    'skin_tone': row.get('skin_tone', ''),
                    'disease': row.get('disease', ''),
                    'ruler_score': round(float(score), 3),
                })

        except Exception as e:
            continue

        if (i + 1) % 100 == 0:
            print(f"  Screened {i+1}/{len(metadata)} images...")

    # Sort by ruler score
    candidates.sort(key=lambda x: -x['ruler_score'])

    print(f"  Found {len(candidates)} candidate ruler images")

    with open(output_path, 'w') as f:
        json.dump({'candidates': candidates, 'total_screened': len(metadata)}, f, indent=2)

    return candidates


def interactive_annotate(ddi_dir, output_path, image_list=None, max_images=50):
    """
    Interactive annotation using matplotlib.

    For each image:
    1. Display the image
    2. User clicks two ruler endpoints
    3. User enters real-world distance
    4. Record annotation
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    csv_path = ddi_dir / "map.csv"
    metadata = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['DDI_file']] = row

    images_dir = ddi_dir / "images"

    # Load existing annotations if any
    annotations = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
            annotations = existing.get('annotations', [])
        annotated_files = {a['DDI_file'] for a in annotations}
    else:
        annotated_files = set()

    # Determine which images to annotate
    if image_list:
        # Use provided list (e.g., from ruler screening)
        files_to_annotate = [f for f in image_list if f not in annotated_files]
    else:
        # Use all images
        files_to_annotate = sorted(metadata.keys())
        files_to_annotate = [f for f in files_to_annotate if f not in annotated_files]

    files_to_annotate = files_to_annotate[:max_images]
    print(f"  Annotating {len(files_to_annotate)} images ({len(annotations)} already done)")

    for img_file in files_to_annotate:
        img_path = images_dir / img_file
        if not img_path.exists():
            continue

        img = Image.open(img_path)
        meta = metadata.get(img_file, {})

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img)
        ax.set_title(f"{img_file} | Tone: {meta.get('skin_tone', '?')} | {meta.get('disease', '?')}\n"
                     f"Click two ruler endpoints (or close window to skip)")

        points = []

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                points.append((event.xdata, event.ydata))
                ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
                if len(points) == 2:
                    ax.plot([points[0][0], points[1][0]],
                            [points[0][1], points[1][1]], 'r-', linewidth=2)
                    pixel_dist = np.sqrt((points[1][0] - points[0][0])**2 +
                                         (points[1][1] - points[0][1])**2)
                    ax.set_title(f"Pixel distance: {pixel_dist:.1f}px | Enter real distance in terminal")
                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        if len(points) >= 2:
            pixel_dist = np.sqrt((points[1][0] - points[0][0])**2 +
                                 (points[1][1] - points[0][1])**2)
            try:
                real_dist = float(input(f"  Enter real distance in mm for {img_file} (pixel dist={pixel_dist:.1f}): "))
                annotation = {
                    'image_id': meta.get('DDI_ID', img_file.replace('.png', '')),
                    'DDI_file': img_file,
                    'point1_px': [round(points[0][0], 1), round(points[0][1], 1)],
                    'point2_px': [round(points[1][0], 1), round(points[1][1], 1)],
                    'real_distance_mm': real_dist,
                    'pixel_distance': round(float(pixel_dist), 1),
                    'skin_tone': meta.get('skin_tone', ''),
                    'disease': meta.get('disease', ''),
                }
                annotations.append(annotation)
                print(f"    Annotated: {pixel_dist:.1f}px = {real_dist}mm")
            except (ValueError, EOFError):
                print(f"    Skipped (invalid input)")
        else:
            print(f"    Skipped (no points)")

        # Save after each annotation
        with open(output_path, 'w') as f:
            json.dump({'annotations': annotations}, f, indent=2)

    print(f"\nTotal annotations: {len(annotations)}")
    print(f"Saved to: {output_path}")
    return annotations


def create_template_annotations(ddi_dir, output_path, num_per_tone=20):
    """
    Create a template annotation file for manual editing.

    Selects images stratified by skin tone for annotation.
    Users can fill in point coordinates and distances manually.
    """
    csv_path = ddi_dir / "map.csv"
    metadata = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    tone_groups = defaultdict(list)
    for row in metadata:
        tone_groups[row.get('skin_tone', 'unknown')].append(row)

    template = {'annotations': []}
    for tone in sorted(tone_groups.keys()):
        selected = tone_groups[tone][:num_per_tone]
        for row in selected:
            template['annotations'].append({
                'image_id': row.get('DDI_ID', ''),
                'DDI_file': row['DDI_file'],
                'point1_px': [0, 0],
                'point2_px': [0, 0],
                'real_distance_mm': 0,
                'pixel_distance': 0,
                'skin_tone': row.get('skin_tone', ''),
                'disease': row.get('disease', ''),
                'annotated': False,
            })

    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Template with {len(template['annotations'])} entries saved to {output_path}")
    print(f"Skin tone distribution: {dict((k, min(len(v), num_per_tone)) for k, v in tone_groups.items())}")
    return template


def main():
    parser = argparse.ArgumentParser(description="DDI ruler annotation tool")
    parser.add_argument('--mode', type=str, default='template',
                        choices=['interactive', 'screen', 'template'],
                        help='Annotation mode')
    parser.add_argument('--output', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "eval_data" / "ddi_ruler_annotations.json"),
                        help='Output annotation file')
    parser.add_argument('--max_images', type=int, default=50,
                        help='Max images to annotate (interactive mode)')
    parser.add_argument('--num_per_tone', type=int, default=20,
                        help='Images per skin tone group (template mode)')
    args = parser.parse_args()

    ddi_dir = DATA_DIR / "DDI"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.mode == 'screen':
        screen_for_rulers(ddi_dir, args.output)
    elif args.mode == 'interactive':
        interactive_annotate(ddi_dir, args.output, max_images=args.max_images)
    elif args.mode == 'template':
        create_template_annotations(ddi_dir, args.output, num_per_tone=args.num_per_tone)


if __name__ == "__main__":
    main()
