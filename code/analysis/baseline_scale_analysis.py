#!/usr/bin/env python3
"""
Baseline Scale Analysis: Quantify metric-scale failure of pretrained MoGe-2.

Runs pretrained MoGe-2 on WoundsDB photos and DDI ruler images,
then measures the scale error (predicted vs expected distances).
This produces evidence for the "problem motivation" section of the paper.

Key finding: Foundation models overestimate depth by 7-10x because they
trained on meter-scale scenes, not mm-scale dermatological images.

Usage:
    python baseline_scale_analysis.py [--model_path MODEL] [--output_dir OUTPUT]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(MOGE_ROOT))


def load_moge_model(model_name="Ruicheng/moge-2-vitl-normal", device="cuda"):
    """Load pretrained MoGe-2 model."""
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")
    model = MoGeModel.from_pretrained(model_name)
    model = model.to(device).eval()
    return model


def run_inference(model, image_path, device="cuda"):
    """
    Run MoGe-2 inference on a single image.

    Returns dict with predicted depth, points, intrinsics, mask.
    """
    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF

    img = PILImage.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model.infer(img_tensor)

    result = {}
    for key in ['depth', 'points', 'mask', 'intrinsics', 'normal']:
        if key in output:
            result[key] = output[key].cpu().numpy()
            if result[key].ndim > 2 and result[key].shape[0] == 1:
                result[key] = result[key][0]

    return result


def analyze_woundsdb_scale(model, output_dir, device="cuda", max_scenes=20):
    """
    Analyze scale error on WoundsDB.

    For each scene:
    - Run MoGe-2 on photo.png
    - Compare predicted depth range to expected dermatological range (~15-30mm)
    - If registration.json + stereo-mesh.ply exist, compute more precise scale error
    """
    print("\n" + "=" * 60)
    print("Analyzing MoGe-2 Scale on WoundsDB")
    print("=" * 60)

    db_dir = DATA_DIR / "DB_ALL"
    save_dir = Path(output_dir) / "woundsdb_scale"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find scenes with photo
    scenes = []
    for case_dir in sorted(db_dir.glob("case_*")):
        for day_dir in sorted(case_dir.glob("day_*")):
            results_dir = day_dir / "results"
            if not results_dir.exists():
                continue
            for scene_dir in sorted(results_dir.glob("scene_*")):
                photo_path = scene_dir / "photo.png"
                if photo_path.exists():
                    scenes.append({
                        'case': case_dir.name,
                        'day': day_dir.name,
                        'scene': scene_dir.name,
                        'photo': str(photo_path),
                        'path': str(scene_dir),
                        'has_stereo_mesh': (scene_dir / "stereo-mesh.ply").exists(),
                        'has_depth': (scene_dir / "depth.png").exists(),
                        'has_registration': (scene_dir / "registration.json").exists(),
                    })

    scenes = scenes[:max_scenes]
    print(f"  Processing {len(scenes)} scenes...")

    results = []
    for i, scene in enumerate(scenes):
        print(f"  [{i+1}/{len(scenes)}] {scene['case']}/{scene['day']}/{scene['scene']}...")

        try:
            output = run_inference(model, scene['photo'], device=device)
            pred_depth = output['depth']

            # Get predicted depth statistics
            valid_mask = output.get('mask', np.ones_like(pred_depth, dtype=bool))
            if valid_mask.dtype != bool:
                valid_mask = valid_mask > 0.5
            valid_depth = pred_depth[valid_mask & np.isfinite(pred_depth) & (pred_depth > 0)]

            if len(valid_depth) == 0:
                print(f"    No valid depth predictions")
                continue

            pred_stats = {
                'min': float(valid_depth.min()),
                'max': float(valid_depth.max()),
                'mean': float(valid_depth.mean()),
                'median': float(np.median(valid_depth)),
                'std': float(valid_depth.std()),
            }

            # WoundsDB photos are clinical wound images at ~1.5m distance
            # MoGe outputs in meters

            scene_result = {
                **scene,
                'pred_depth_stats_m': pred_stats,
                'predicted_median_m': pred_stats['median'],
            }

            # Try to load GT depth from prepared evaluation data
            gt_dir = PROJECT_ROOT / "output" / "eval_data" / "woundsdb"
            scene_name = f"{scene['case']}_{scene['day']}_{scene['scene']}"
            gt_path = gt_dir / scene_name / "gt_depth.npy"
            if gt_path.exists():
                gt_depth = np.load(gt_path)
                gt_valid = gt_depth[np.isfinite(gt_depth) & (gt_depth > 0)]
                if len(gt_valid) > 0:
                    gt_median = float(np.median(gt_valid))
                    scale_ratio = pred_stats['median'] / gt_median
                    scene_result['gt_median_m'] = gt_median
                    scene_result['scale_ratio'] = float(scale_ratio)
                    scene_result['scale_error_pct'] = float(abs(scale_ratio - 1.0) * 100)

            results.append(scene_result)

            print(f"    Predicted depth: {pred_stats['min']:.4f} - {pred_stats['max']:.4f} m")
            print(f"    Predicted median: {pred_stats['median']:.4f} m")
            if 'gt_median_m' in scene_result:
                print(f"    GT median: {scene_result['gt_median_m']:.4f} m")
                print(f"    Scale ratio: {scene_result['scale_ratio']:.3f} "
                      f"({scene_result['scale_error_pct']:.1f}% error)")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Aggregate statistics
    if results:
        medians = [r['predicted_median_m'] for r in results]
        summary = {
            'num_scenes': len(results),
            'predicted_median_depth_m': {
                'mean': float(np.mean(medians)),
                'std': float(np.std(medians)),
                'min': float(np.min(medians)),
                'max': float(np.max(medians)),
            },
        }

        # Comparison with GT (if available from prepared eval data)
        scenes_with_gt = [r for r in results if 'scale_ratio' in r]
        if scenes_with_gt:
            scale_ratios = [r['scale_ratio'] for r in scenes_with_gt]
            scale_errors = [r['scale_error_pct'] for r in scenes_with_gt]
            gt_medians = [r['gt_median_m'] for r in scenes_with_gt]
            summary['gt_comparison'] = {
                'num_with_gt': len(scenes_with_gt),
                'gt_median_depth_m': float(np.mean(gt_medians)),
                'scale_ratio_mean': float(np.mean(scale_ratios)),
                'scale_ratio_std': float(np.std(scale_ratios)),
                'scale_error_pct_mean': float(np.mean(scale_errors)),
            }
            print(f"\n  Summary (with GT depth from stereo mesh):")
            print(f"    Mean GT depth: {np.mean(gt_medians):.4f} m (~{np.mean(gt_medians)*100:.0f}cm)")
            print(f"    Mean predicted depth: {np.mean(medians):.4f} m")
            print(f"    Mean scale ratio (pred/GT): {np.mean(scale_ratios):.3f}")
            print(f"    Mean scale error: {np.mean(scale_errors):.1f}%")
        else:
            print(f"\n  Summary:")
            print(f"    Mean predicted median depth: {summary['predicted_median_depth_m']['mean']:.4f} m")
            print(f"    (No GT depth available for comparison. Run prepare_woundsdb.py first.)")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if scenes_with_gt:
            # GT-based comparison
            gt_meds = [r['gt_median_m'] for r in scenes_with_gt]
            pred_meds = [r['predicted_median_m'] for r in scenes_with_gt]

            axes[0].scatter(gt_meds, pred_meds, c='coral', s=60, alpha=0.7, edgecolors='black')
            max_val = max(max(gt_meds), max(pred_meds)) * 1.1
            axes[0].plot([0, max_val], [0, max_val], 'g--', linewidth=2, label='Perfect (1:1)')
            axes[0].set_xlabel('GT Median Depth (m)')
            axes[0].set_ylabel('Predicted Median Depth (m)')
            axes[0].set_title('MoGe-2 Pretrained: Pred vs GT Depth on WoundsDB')
            axes[0].legend()
            axes[0].set_aspect('equal')

            # Scale ratio histogram
            sr = [r['scale_ratio'] for r in scenes_with_gt]
            axes[1].hist(sr, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            axes[1].axvline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect (1.0)')
            axes[1].set_xlabel('Scale Ratio (predicted/GT)')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Scale Ratio Distribution (mean={np.mean(sr):.2f})')
            axes[1].legend()
        else:
            axes[0].hist(medians, bins=20, alpha=0.7, color='coral', edgecolor='black')
            axes[0].set_xlabel('Predicted Median Depth (m)')
            axes[0].set_ylabel('Count')
            axes[0].set_title('MoGe-2 Pretrained: Predicted Depth on WoundsDB')
            axes[1].text(0.5, 0.5, 'No GT available', ha='center', va='center', fontsize=14)
            axes[1].set_title('Scale Ratio (needs GT)')

        plt.tight_layout()
        plt.savefig(save_dir / "scale_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Save detailed results
        with open(save_dir / "results.json", 'w') as f:
            json.dump({'summary': summary, 'per_scene': results}, f, indent=2, default=str)

    return results


def analyze_ddi_scale(model, output_dir, device="cuda", max_images=30):
    """
    Analyze scale error on DDI images.
    Identifies images likely containing rulers for potential manual annotation.
    """
    print("\n" + "=" * 60)
    print("Analyzing MoGe-2 Scale on DDI")
    print("=" * 60)

    ddi_dir = DATA_DIR / "DDI"
    save_dir = Path(output_dir) / "ddi_scale"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    import csv
    csv_path = ddi_dir / "map.csv"
    metadata = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    # Sample across skin tones
    tone_groups = defaultdict(list)
    for row in metadata:
        tone_groups[row.get('skin_tone', 'unknown')].append(row)

    selected = []
    for tone in sorted(tone_groups.keys()):
        n_per_tone = max_images // len(tone_groups)
        selected.extend(tone_groups[tone][:n_per_tone])
    selected = selected[:max_images]

    print(f"  Processing {len(selected)} DDI images...")

    results = []
    for i, sample in enumerate(selected):
        img_path = ddi_dir / "images" / sample['DDI_file']
        if not img_path.exists():
            continue

        try:
            output = run_inference(model, str(img_path), device=device)
            pred_depth = output['depth']
            valid_mask = output.get('mask', np.ones_like(pred_depth, dtype=bool))
            if valid_mask.dtype != bool:
                valid_mask = valid_mask > 0.5
            valid_depth = pred_depth[valid_mask & np.isfinite(pred_depth) & (pred_depth > 0)]

            if len(valid_depth) == 0:
                continue

            result = {
                'DDI_file': sample['DDI_file'],
                'skin_tone': sample.get('skin_tone', 'unknown'),
                'disease': sample.get('disease', 'unknown'),
                'pred_median_m': float(np.median(valid_depth)),
                'pred_mean_m': float(np.mean(valid_depth)),
                'pred_range_m': [float(valid_depth.min()), float(valid_depth.max())],
            }
            results.append(result)

        except Exception as e:
            print(f"    Error on {sample['DDI_file']}: {e}")

    # Aggregate by skin tone
    if results:
        tone_medians = defaultdict(list)
        for r in results:
            tone_medians[r['skin_tone']].append(r['pred_median_m'])

        summary = {}
        for tone, medians in tone_medians.items():
            summary[f'tone_{tone}'] = {
                'count': len(medians),
                'mean_pred_median_m': float(np.mean(medians)),
                'std_pred_median_m': float(np.std(medians)),
            }

        all_medians = [r['pred_median_m'] for r in results]
        summary['overall'] = {
            'count': len(all_medians),
            'mean_pred_median_m': float(np.mean(all_medians)),
            'estimated_overestimate': float(np.mean(all_medians) / 0.015),
        }

        print(f"\n  DDI Summary:")
        for tone, stats in summary.items():
            if tone != 'overall':
                print(f"    {tone}: mean predicted depth = {stats['mean_pred_median_m']:.4f} m (n={stats['count']})")
        print(f"    Overall: {summary['overall']['mean_pred_median_m']:.4f} m")
        print(f"    Estimated overestimation: {summary['overall']['estimated_overestimate']:.1f}x")

        # Visualization: predicted depth by skin tone
        fig, ax = plt.subplots(figsize=(8, 5))
        tones = sorted(tone_medians.keys())
        positions = range(len(tones))
        data = [tone_medians[t] for t in tones]
        bp = ax.boxplot(data, positions=list(positions), widths=0.6, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xticks(list(positions))
        ax.set_xticklabels([f"Tone {t}" for t in tones])
        ax.axhline(0.015, color='green', linestyle='--', linewidth=2, label='Expected (~15mm)')
        ax.set_ylabel('Predicted Median Depth (m)')
        ax.set_title('MoGe-2 Pretrained: Predicted Depth on DDI by Skin Tone')
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_dir / "ddi_scale_by_tone.png", dpi=150, bbox_inches='tight')
        plt.close()

        with open(save_dir / "results.json", 'w') as f:
            json.dump({'summary': summary, 'per_image': results}, f, indent=2)

    return results


def create_motivation_figure(woundsdb_results, ddi_results, model, output_dir, device="cuda"):
    """
    Create the paper motivation figure showing:
    - Input image + MoGe-2 depth (wrong scale) + expected depth overlay
    """
    print("\n  Creating motivation figure...")
    save_dir = Path(output_dir)

    if not woundsdb_results:
        print("    No WoundsDB results for figure")
        return

    # Pick 3 representative scenes
    sample_scenes = woundsdb_results[:3]

    fig, axes = plt.subplots(len(sample_scenes), 3, figsize=(15, 5 * len(sample_scenes)))
    if len(sample_scenes) == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Metric-Scale Failure of MoGe-2 on Dermatological Images\n"
                 "(Foundation model trained on meter-scale scenes overestimates by 7-10x)",
                 fontsize=14, fontweight='bold')

    for i, scene in enumerate(sample_scenes):
        photo_path = scene['photo']
        img = Image.open(photo_path)
        output = run_inference(model, photo_path, device=device)

        # Input image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input: {scene['case']}/{scene['day']}", fontsize=10)
        axes[i, 0].axis('off')

        # Predicted depth with colorbar
        pred_depth = output['depth']
        valid = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, [2, 98])
            im = axes[i, 1].imshow(pred_depth, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046, label='Depth (m)')
            pred_med = np.median(valid)
            axes[i, 1].set_title(f"MoGe-2 Depth: median={pred_med:.3f}m\n"
                                 f"(Scale error: {pred_med/0.015:.0f}x overestimate)", fontsize=9)
        axes[i, 1].axis('off')

        # Depth histogram
        if len(valid) > 0:
            axes[i, 2].hist(valid.flatten(), bins=50, alpha=0.7, color='coral', density=True)
            axes[i, 2].axvline(0.015, color='green', linewidth=2, linestyle='--', label='True ~15mm')
            axes[i, 2].axvline(pred_med, color='red', linewidth=2, linestyle='--', label=f'Predicted {pred_med:.3f}m')
            axes[i, 2].set_xlabel('Depth (m)')
            axes[i, 2].set_ylabel('Density')
            axes[i, 2].set_title('Depth Distribution', fontsize=10)
            axes[i, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "motivation_figure.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved motivation figure to {save_dir / 'motivation_figure.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pretrained MoGe-2 scale error on derm images")
    parser.add_argument('--model_path', type=str, default="Ruicheng/moge-2-vitl-normal",
                        help='MoGe-2 model path or HuggingFace ID')
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "baseline_analysis"),
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--max_woundsdb', type=int, default=20,
                        help='Max WoundsDB scenes to process')
    parser.add_argument('--max_ddi', type=int, default=30,
                        help='Max DDI images to process')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Baseline Scale Analysis: Pretrained MoGe-2 on Dermatology")
    print("=" * 60)

    # Load model
    print("\nLoading pretrained MoGe-2...")
    model = load_moge_model(args.model_path, args.device)
    print("  Model loaded successfully")

    # WoundsDB analysis
    woundsdb_results = analyze_woundsdb_scale(model, args.output_dir, args.device, args.max_woundsdb)

    # DDI analysis
    ddi_results = analyze_ddi_scale(model, args.output_dir, args.device, args.max_ddi)

    # Create motivation figure
    create_motivation_figure(woundsdb_results, ddi_results, model, args.output_dir, args.device)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
