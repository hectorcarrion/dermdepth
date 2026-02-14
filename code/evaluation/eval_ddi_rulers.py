#!/usr/bin/env python3
"""
DDI Ruler-Based Scale Evaluation.

For each annotated DDI image with ruler markings:
1. Run model inference to get 3D point map
2. Measure Euclidean distance between annotated ruler endpoints in 3D
3. Compare predicted distance to known real distance
4. Compute scale ratio and error, stratified by skin tone

Metrics:
- Scale Ratio: predicted_distance / true_distance (target = 1.0)
- Scale Error (%): |scale_ratio - 1.0| * 100
- Stratified by Fitzpatrick skin tone (12, 34, 56)

Usage:
    python eval_ddi_rulers.py --model MODEL --annotations ANNOTATIONS_JSON
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(MOGE_ROOT))


def load_model(model_path, device="cuda"):
    """Load MoGe-2 model."""
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model_config = checkpoint.get('model_config', None)
        if model_config:
            model = MoGeModel(**model_config)
        else:
            model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model = MoGeModel.from_pretrained(model_path)

    return model.to(device).eval()


def run_inference_points(model, image_path, device="cuda"):
    """Run inference and return 3D point map."""
    import torchvision.transforms.functional as TF

    img = Image.open(image_path).convert('RGB')
    img_w, img_h = img.size
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model.infer(img_tensor)

    points = output['points'].cpu().numpy()
    if points.ndim == 4 and points.shape[0] == 1:
        points = points[0]  # HxWx3

    depth = output['depth'].cpu().numpy()
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]

    return points, depth, (img_w, img_h)


def measure_ruler_distance(points, point1_px, point2_px, image_size, points_shape):
    """
    Measure 3D Euclidean distance between two pixel locations.

    Args:
        points: HxWx3 point map
        point1_px: [x, y] pixel coordinates of first ruler point
        point2_px: [x, y] pixel coordinates of second ruler point
        image_size: (width, height) of original image
        points_shape: (H, W) of points map

    Returns:
        distance_3d: Euclidean distance in model's units (meters)
        distance_2d_px: Pixel distance
    """
    img_w, img_h = image_size
    pts_h, pts_w = points_shape[:2]

    # Scale pixel coordinates to points map resolution
    scale_x = pts_w / img_w
    scale_y = pts_h / img_h

    x1 = int(round(point1_px[0] * scale_x))
    y1 = int(round(point1_px[1] * scale_y))
    x2 = int(round(point2_px[0] * scale_x))
    y2 = int(round(point2_px[1] * scale_y))

    # Clamp to valid range
    x1 = max(0, min(x1, pts_w - 1))
    y1 = max(0, min(y1, pts_h - 1))
    x2 = max(0, min(x2, pts_w - 1))
    y2 = max(0, min(y2, pts_h - 1))

    # Get 3D points
    p1 = points[y1, x1, :]  # xyz
    p2 = points[y2, x2, :]  # xyz

    # Check validity
    if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
        # Try averaging a small neighborhood
        r = 3
        y1s, y1e = max(0, y1-r), min(pts_h, y1+r+1)
        x1s, x1e = max(0, x1-r), min(pts_w, x1+r+1)
        y2s, y2e = max(0, y2-r), min(pts_h, y2+r+1)
        x2s, x2e = max(0, x2-r), min(pts_w, x2+r+1)

        patch1 = points[y1s:y1e, x1s:x1e, :]
        patch2 = points[y2s:y2e, x2s:x2e, :]

        valid1 = np.isfinite(patch1).all(axis=-1)
        valid2 = np.isfinite(patch2).all(axis=-1)

        if valid1.sum() > 0 and valid2.sum() > 0:
            p1 = patch1[valid1].mean(axis=0)
            p2 = patch2[valid2].mean(axis=0)
        else:
            return None, None

    distance_3d = float(np.linalg.norm(p2 - p1))
    distance_2d = float(np.sqrt((point2_px[0] - point1_px[0])**2 + (point2_px[1] - point1_px[1])**2))

    return distance_3d, distance_2d


def evaluate_ddi_rulers(model, annotations_path, output_dir, device="cuda"):
    """
    Evaluate metric scale accuracy using DDI ruler annotations.
    """
    print("=" * 60)
    print("DDI Ruler-Based Scale Evaluation")
    print("=" * 60)

    with open(annotations_path) as f:
        data = json.load(f)
    annotations = data['annotations']

    # Filter to annotated entries
    annotated = [a for a in annotations if a.get('real_distance_mm', 0) > 0
                 and a.get('annotated', True) is not False]
    print(f"  {len(annotated)} annotated ruler images")

    if not annotated:
        print("  No annotations found. Run annotate_ddi_rulers.py first.")
        return

    ddi_images_dir = DATA_DIR / "DDI" / "images"
    results = []
    tone_results = defaultdict(list)

    for i, ann in enumerate(annotated):
        img_path = ddi_images_dir / ann['DDI_file']
        if not img_path.exists():
            continue

        print(f"  [{i+1}/{len(annotated)}] {ann['DDI_file']}...", end="")

        try:
            points, depth, img_size = run_inference_points(model, str(img_path), device)

            dist_3d, dist_2d = measure_ruler_distance(
                points,
                ann['point1_px'], ann['point2_px'],
                img_size, points.shape
            )

            if dist_3d is None:
                print(f" invalid points")
                continue

            # Convert to mm: MoGe predicts in meters
            predicted_mm = dist_3d * 1000
            true_mm = ann['real_distance_mm']
            scale_ratio = predicted_mm / true_mm
            scale_error_pct = abs(scale_ratio - 1.0) * 100

            result = {
                'DDI_file': ann['DDI_file'],
                'skin_tone': ann.get('skin_tone', 'unknown'),
                'disease': ann.get('disease', 'unknown'),
                'true_distance_mm': true_mm,
                'predicted_distance_mm': float(predicted_mm),
                'scale_ratio': float(scale_ratio),
                'scale_error_pct': float(scale_error_pct),
                'pixel_distance': float(dist_2d) if dist_2d else 0,
            }
            results.append(result)
            tone_results[ann.get('skin_tone', 'unknown')].append(result)

            print(f" pred={predicted_mm:.1f}mm true={true_mm:.1f}mm ratio={scale_ratio:.2f}")

        except Exception as e:
            print(f" Error: {e}")

    # Aggregate results
    if results:
        scale_ratios = [r['scale_ratio'] for r in results]
        scale_errors = [r['scale_error_pct'] for r in results]

        summary = {
            'total_evaluated': len(results),
            'scale_ratio': {
                'mean': float(np.mean(scale_ratios)),
                'std': float(np.std(scale_ratios)),
                'median': float(np.median(scale_ratios)),
            },
            'scale_error_pct': {
                'mean': float(np.mean(scale_errors)),
                'std': float(np.std(scale_errors)),
                'median': float(np.median(scale_errors)),
            },
        }

        # Per skin tone
        summary['per_tone'] = {}
        for tone, tone_res in sorted(tone_results.items()):
            ratios = [r['scale_ratio'] for r in tone_res]
            errors = [r['scale_error_pct'] for r in tone_res]
            summary['per_tone'][f'tone_{tone}'] = {
                'count': len(tone_res),
                'scale_ratio_mean': float(np.mean(ratios)),
                'scale_ratio_std': float(np.std(ratios)),
                'scale_error_pct_mean': float(np.mean(errors)),
            }

        # Fairness gap
        tone_means = [v['scale_error_pct_mean'] for v in summary['per_tone'].values()]
        if len(tone_means) > 1:
            summary['fairness_gap_pct'] = float(max(tone_means) - min(tone_means))

        print(f"\n  Summary:")
        print(f"    Scale Ratio: {summary['scale_ratio']['mean']:.3f} +/- {summary['scale_ratio']['std']:.3f}")
        print(f"    Scale Error: {summary['scale_error_pct']['mean']:.1f}% +/- {summary['scale_error_pct']['std']:.1f}%")
        for tone, stats in summary['per_tone'].items():
            print(f"    {tone}: ratio={stats['scale_ratio_mean']:.3f} error={stats['scale_error_pct_mean']:.1f}% (n={stats['count']})")
        if 'fairness_gap_pct' in summary:
            print(f"    Fairness gap: {summary['fairness_gap_pct']:.1f}%")

    else:
        summary = {'note': 'No results'}

    # Save
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'summary': summary,
        'per_image': results,
    }
    with open(save_dir / "ddi_ruler_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    # Visualization
    if results:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Scatter: predicted vs true
        true_dists = [r['true_distance_mm'] for r in results]
        pred_dists = [r['predicted_distance_mm'] for r in results]
        tones = [r['skin_tone'] for r in results]
        tone_colors = {'12': '#f4c2a1', '34': '#d4956b', '56': '#8b5e3c', 'unknown': 'gray'}

        for tone in sorted(set(tones)):
            mask = [t == tone for t in tones]
            t = [true_dists[j] for j in range(len(results)) if mask[j]]
            p = [pred_dists[j] for j in range(len(results)) if mask[j]]
            axes[0].scatter(t, p, c=tone_colors.get(tone, 'gray'), label=f'Tone {tone}', alpha=0.7, s=60)

        max_val = max(max(true_dists), max(pred_dists)) * 1.1
        axes[0].plot([0, max_val], [0, max_val], 'k--', label='Perfect (1:1)')
        axes[0].set_xlabel('True Distance (mm)')
        axes[0].set_ylabel('Predicted Distance (mm)')
        axes[0].set_title('Predicted vs True Ruler Distance')
        axes[0].legend()
        axes[0].set_aspect('equal')

        # Scale ratio histogram
        axes[1].hist(scale_ratios, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].axvline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect (1.0)')
        axes[1].set_xlabel('Scale Ratio (predicted/true)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Scale Ratio Distribution\n(mean={np.mean(scale_ratios):.2f})')
        axes[1].legend()

        # Per-tone comparison
        tone_keys = sorted(summary['per_tone'].keys())
        tone_labels = [k.replace('tone_', 'T') for k in tone_keys]
        tone_means = [summary['per_tone'][k]['scale_ratio_mean'] for k in tone_keys]
        tone_stds = [summary['per_tone'][k].get('scale_ratio_std', 0) for k in tone_keys]
        axes[2].bar(tone_labels, tone_means, yerr=tone_stds, capsize=5,
                    color=[tone_colors.get(k.replace('tone_', ''), 'gray') for k in tone_keys],
                    edgecolor='black')
        axes[2].axhline(1.0, color='green', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Skin Tone Group')
        axes[2].set_ylabel('Mean Scale Ratio')
        axes[2].set_title('Scale Ratio by Skin Tone')

        plt.tight_layout()
        plt.savefig(save_dir / "ddi_ruler_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nResults saved to {save_dir}")
    return output


def main():
    parser = argparse.ArgumentParser(description="DDI ruler-based scale evaluation")
    parser.add_argument('--model', type=str, required=True,
                        help='Model path or HuggingFace ID')
    parser.add_argument('--annotations', type=str,
                        default=str(PROJECT_ROOT / "output" / "eval_data" / "ddi_ruler_annotations.json"),
                        help='Path to ruler annotations JSON')
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "evaluation"),
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name for output path')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.model_name)
    model = load_model(args.model, args.device)
    evaluate_ddi_rulers(model, args.annotations, output_dir, args.device)


if __name__ == "__main__":
    main()
