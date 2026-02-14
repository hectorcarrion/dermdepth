#!/usr/bin/env python3
"""
Dataset analysis and visualization for DermDepth.

Explores WoundsDB, SKINL2, and DDI datasets:
- Sample photos, depth maps, PLY meshes
- Dataset statistics and coverage
- Depth histograms and distributions
- Identifies ruler images in DDI

Usage:
    python analyze_datasets.py [--output_dir OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import csv

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def analyze_woundsdb(output_dir):
    """Analyze WoundsDB (DB_ALL) dataset."""
    print("\n" + "=" * 60)
    print("Analyzing WoundsDB (DB_ALL)")
    print("=" * 60)

    db_dir = DATA_DIR / "DB_ALL"
    csv_path = db_dir / "WoundsDB_Description.csv"

    # Parse CSV metadata
    metadata = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata.append(row)
        print(f"  CSV metadata: {len(metadata)} rows")

    # Discover all scenes
    scenes = []
    for case_dir in sorted(db_dir.glob("case_*")):
        case_id = case_dir.name
        for day_dir in sorted(case_dir.glob("day_*")):
            day_id = day_dir.name
            results_dir = day_dir / "results"
            if not results_dir.exists():
                continue
            for scene_dir in sorted(results_dir.glob("scene_*")):
                scene_id = scene_dir.name
                files = {f.name for f in scene_dir.iterdir()}
                scenes.append({
                    'case': case_id,
                    'day': day_id,
                    'scene': scene_id,
                    'path': str(scene_dir),
                    'has_photo': 'photo.png' in files,
                    'has_depth': 'depth.png' in files,
                    'has_stereo_mesh': 'stereo-mesh.ply' in files,
                    'has_depth_mesh': 'depth-mesh.ply' in files,
                    'has_registration': 'registration.json' in files,
                    'has_thermal': 'thermal.png' in files,
                    'has_stereo': 'stereo.png' in files,
                })

    # Statistics
    cases = set(s['case'] for s in scenes)
    stats = {
        'total_scenes': len(scenes),
        'total_cases': len(cases),
        'scenes_with_photo': sum(1 for s in scenes if s['has_photo']),
        'scenes_with_depth': sum(1 for s in scenes if s['has_depth']),
        'scenes_with_stereo_mesh': sum(1 for s in scenes if s['has_stereo_mesh']),
        'scenes_with_depth_mesh': sum(1 for s in scenes if s['has_depth_mesh']),
        'scenes_with_registration': sum(1 for s in scenes if s['has_registration']),
        'scenes_with_thermal': sum(1 for s in scenes if s['has_thermal']),
    }

    print(f"  Cases: {stats['total_cases']}")
    print(f"  Total scenes: {stats['total_scenes']}")
    print(f"  With photo: {stats['scenes_with_photo']}")
    print(f"  With stereo-mesh: {stats['scenes_with_stereo_mesh']}")
    print(f"  With depth-mesh: {stats['scenes_with_depth_mesh']}")
    print(f"  With registration: {stats['scenes_with_registration']}")

    # Temporal analysis - visits per patient
    visits_per_case = defaultdict(set)
    for s in scenes:
        visits_per_case[s['case']].add(s['day'])
    visit_counts = [len(v) for v in visits_per_case.values()]
    stats['visits_per_case_mean'] = float(np.mean(visit_counts))
    stats['visits_per_case_max'] = max(visit_counts)

    # Visualize sample scenes
    save_dir = Path(output_dir) / "woundsdb"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sample grid: up to 12 scenes with photo + depth
    eval_scenes = [s for s in scenes if s['has_photo'] and s['has_stereo_mesh']]
    sample_scenes = eval_scenes[:min(12, len(eval_scenes))]

    if sample_scenes:
        n = len(sample_scenes)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols * 2, figsize=(5 * cols, 3 * rows))
        if rows == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(f"WoundsDB: Sample Scenes ({stats['total_scenes']} total, {len(eval_scenes)} with photo+mesh)",
                     fontsize=14, fontweight='bold')

        for i, scene in enumerate(sample_scenes):
            r, c = i // cols, i % cols

            # Photo
            photo = Image.open(os.path.join(scene['path'], 'photo.png'))
            axes[r, c * 2].imshow(photo)
            axes[r, c * 2].set_title(f"{scene['case']}/{scene['day']}", fontsize=8)
            axes[r, c * 2].axis('off')

            # Depth
            if scene['has_depth']:
                depth_img = Image.open(os.path.join(scene['path'], 'depth.png'))
                axes[r, c * 2 + 1].imshow(depth_img, cmap='viridis')
                axes[r, c * 2 + 1].set_title("depth", fontsize=8)
            axes[r, c * 2 + 1].axis('off')

        # Hide unused axes
        for i in range(n, rows * cols):
            r, c = i // cols, i % cols
            axes[r, c * 2].axis('off')
            axes[r, c * 2 + 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_dir / "sample_scenes.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Save stats
    with open(save_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Output saved to {save_dir}")
    return stats, scenes


def analyze_skinl2(output_dir):
    """Analyze SKINL2 dataset."""
    print("\n" + "=" * 60)
    print("Analyzing SKINL2")
    print("=" * 60)

    skinl2_dir = DATA_DIR / "SKINL2"
    save_dir = Path(output_dir) / "skinl2"
    save_dir.mkdir(parents=True, exist_ok=True)

    stats = {}

    # Analyze v1 (has Central View + DepthMap structure)
    v1_dir = skinl2_dir / "SKINL2_v1"
    cv_dir = v1_dir / "Central View"
    dm_dir = v1_dir / "DepthMap"

    samples = []
    if cv_dir.exists():
        for disease_dir in sorted(cv_dir.iterdir()):
            if not disease_dir.is_dir():
                continue
            disease = disease_dir.name
            for sample_dir in sorted(disease_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sample_id = sample_dir.name
                # Find central view PNG
                pngs = list(sample_dir.glob("*_TotalFocus.png"))
                if not pngs:
                    pngs = list(sample_dir.glob("*.png"))
                # Find depth TIFF
                depth_dir = dm_dir / disease / sample_id
                tiffs = list(depth_dir.glob("*_DepthMap.tiff")) if depth_dir.exists() else []

                samples.append({
                    'disease': disease,
                    'sample_id': sample_id,
                    'central_view': str(pngs[0]) if pngs else None,
                    'depth_tiff': str(tiffs[0]) if tiffs else None,
                    'has_both': bool(pngs and tiffs),
                })

    disease_counts = defaultdict(int)
    for s in samples:
        disease_counts[s['disease']] += 1

    stats['v1'] = {
        'total_samples': len(samples),
        'with_both': sum(1 for s in samples if s['has_both']),
        'diseases': dict(disease_counts),
    }

    print(f"  V1 samples: {stats['v1']['total_samples']}")
    print(f"  V1 with central view + depth: {stats['v1']['with_both']}")
    print(f"  Diseases: {dict(disease_counts)}")

    # Count v2 and v3
    for version in ['v2', 'v3']:
        vdir = skinl2_dir / f"SKINL2_{version}"
        if vdir.exists():
            count = sum(1 for _ in vdir.glob("*") if _.is_dir())
            stats[version] = {'total_samples': count}
            print(f"  {version.upper()} samples: {count}")

    # Visualize sample central views + depth maps
    paired_samples = [s for s in samples if s['has_both']]
    sample_set = paired_samples[:min(8, len(paired_samples))]

    if sample_set:
        n = len(sample_set)
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
        if n == 1:
            axes = axes[:, np.newaxis]
        fig.suptitle(f"SKINL2 v1: Central View + Depth ({len(paired_samples)} paired samples)",
                     fontsize=14, fontweight='bold')

        for i, s in enumerate(sample_set):
            # Central view
            img = Image.open(s['central_view'])
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"{s['disease']}\n{s['sample_id']}", fontsize=7)
            axes[0, i].axis('off')

            # Depth TIFF
            try:
                depth_img = Image.open(s['depth_tiff'])
                depth_arr = np.array(depth_img, dtype=np.float32)
                valid = depth_arr[depth_arr > 0]
                if len(valid) > 0:
                    vmin, vmax = np.percentile(valid, [2, 98])
                    axes[1, i].imshow(depth_arr, cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[1, i].set_title(f"Depth [{vmin:.1f}-{vmax:.1f}]", fontsize=7)
                else:
                    axes[1, i].imshow(depth_arr, cmap='viridis')
                    axes[1, i].set_title("Depth", fontsize=7)
            except Exception as e:
                axes[1, i].text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', fontsize=6)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(save_dir / "sample_pairs.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Depth histogram for paired samples
    if paired_samples:
        all_depths = []
        for s in paired_samples[:50]:  # Limit to 50 for speed
            try:
                d = np.array(Image.open(s['depth_tiff']), dtype=np.float32)
                valid = d[(d > 0) & np.isfinite(d)]
                if len(valid) > 0:
                    all_depths.extend(valid.flatten()[:10000])  # Subsample
            except:
                pass

        if all_depths:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(all_depths, bins=100, alpha=0.7, color='steelblue')
            ax.set_xlabel('Depth Value (raw units)')
            ax.set_ylabel('Count')
            ax.set_title(f'SKINL2 Depth Distribution ({len(paired_samples)} samples)')
            plt.tight_layout()
            plt.savefig(save_dir / "depth_histogram.png", dpi=150, bbox_inches='tight')
            plt.close()

    with open(save_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Output saved to {save_dir}")
    return stats, samples


def analyze_ddi(output_dir):
    """Analyze DDI dataset."""
    print("\n" + "=" * 60)
    print("Analyzing DDI")
    print("=" * 60)

    ddi_dir = DATA_DIR / "DDI"
    save_dir = Path(output_dir) / "ddi"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata CSV
    csv_path = ddi_dir / "map.csv"
    metadata = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    print(f"  Total images: {len(metadata)}")

    # Analyze by skin tone
    tone_counts = defaultdict(int)
    disease_counts = defaultdict(int)
    malignant_counts = defaultdict(int)
    for row in metadata:
        tone = row.get('skin_tone', 'unknown')
        tone_counts[tone] += 1
        disease_counts[row.get('disease', 'unknown')] += 1
        if row.get('malignant', 'False') == 'True':
            malignant_counts[tone] += 1

    stats = {
        'total_images': len(metadata),
        'skin_tone_distribution': dict(tone_counts),
        'num_diseases': len(disease_counts),
        'top_diseases': dict(sorted(disease_counts.items(), key=lambda x: -x[1])[:10]),
        'malignant_by_tone': dict(malignant_counts),
        'malignant_total': sum(1 for m in metadata if m.get('malignant', 'False') == 'True'),
        'benign_total': sum(1 for m in metadata if m.get('malignant', 'False') == 'False'),
    }

    print(f"  Skin tone distribution: {dict(tone_counts)}")
    print(f"  Diseases: {stats['num_diseases']}")
    print(f"  Malignant: {stats['malignant_total']}, Benign: {stats['benign_total']}")

    # Visualize sample images per skin tone
    images_dir = ddi_dir / "images"
    tone_groups = defaultdict(list)
    for row in metadata:
        tone_groups[row.get('skin_tone', 'unknown')].append(row)

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle("DDI: Sample Images by Skin Tone Group", fontsize=14, fontweight='bold')

    for row_idx, tone in enumerate(sorted(tone_groups.keys())):
        samples = tone_groups[tone][:6]
        for col_idx, sample in enumerate(samples):
            img_path = images_dir / sample['DDI_file']
            if img_path.exists():
                img = Image.open(img_path)
                axes[row_idx, col_idx].imshow(img)
                mal_str = "MAL" if sample.get('malignant', 'False') == 'True' else "BEN"
                axes[row_idx, col_idx].set_title(
                    f"T{tone} {mal_str}\n{sample['disease'][:20]}", fontsize=6)
            axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / "sample_by_tone.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Skin tone distribution bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    tones = sorted(tone_counts.keys())
    ax1.bar(tones, [tone_counts[t] for t in tones], color=['#f4c2a1', '#d4956b', '#8b5e3c'])
    ax1.set_xlabel('Skin Tone Group')
    ax1.set_ylabel('Count')
    ax1.set_title('DDI: Skin Tone Distribution')

    # Malignancy by tone
    mal = [malignant_counts.get(t, 0) for t in tones]
    ben = [tone_counts[t] - malignant_counts.get(t, 0) for t in tones]
    ax2.bar(tones, mal, label='Malignant', color='salmon')
    ax2.bar(tones, ben, bottom=mal, label='Benign', color='steelblue')
    ax2.set_xlabel('Skin Tone Group')
    ax2.set_ylabel('Count')
    ax2.set_title('DDI: Malignancy by Skin Tone')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    with open(save_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Output saved to {save_dir}")
    return stats, metadata


def main():
    parser = argparse.ArgumentParser(description="Analyze DermDepth evaluation datasets")
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "analysis"),
                        help='Output directory for analysis results')
    parser.add_argument('--datasets', nargs='+', default=['woundsdb', 'skinl2', 'ddi'],
                        choices=['woundsdb', 'skinl2', 'ddi'],
                        help='Datasets to analyze')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    if 'woundsdb' in args.datasets:
        results['woundsdb'] = analyze_woundsdb(args.output_dir)
    if 'skinl2' in args.datasets:
        results['skinl2'] = analyze_skinl2(args.output_dir)
    if 'ddi' in args.datasets:
        results['ddi'] = analyze_ddi(args.output_dir)

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
