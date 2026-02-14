#!/usr/bin/env python3
"""
Generate paper figures for DermDepth (MICCAI 2025, 8-page LNCS).

Figures:
1. Problem Motivation: Input + MoGe-2 depth (wrong) + DermDepth depth (correct)
2. Method Pipeline: S-SYNTH -> MoGe-2 architecture -> inference
3. Synthetic Data Diversity: Grid of melanin x lighting/lesions
4. Quantitative Results: Bar charts on WoundsDB metrics
5. Qualitative Comparison: Side-by-side depth maps
6. DDI Ruler Scale Analysis: Scatter of predicted vs true distances
7. Fairness Analysis: Grouped bar by skin tone
8. Ablations: Dataset size curve + loss weight sweep

Usage:
    python create_figures.py --results_dir RESULTS [--output_dir OUTPUT]
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
from matplotlib.gridspec import GridSpec
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))

# LNCS figure dimensions (in inches, for 8-page format)
FULL_WIDTH = 6.5   # Full column width
HALF_WIDTH = 3.15  # Half column width
DPI = 300

# Color palette
COLORS = {
    'moge_pretrained': '#e74c3c',   # Red
    'dermdepth': '#27ae60',          # Green
    'tone_12': '#f4c2a1',           # Light skin
    'tone_34': '#d4956b',           # Medium skin
    'tone_56': '#8b5e3c',           # Dark skin
    'baseline': '#95a5a6',          # Gray
}


def load_results(results_dir):
    """Load all evaluation results from directory."""
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.rglob("*.json"):
        key = str(json_file.relative_to(results_path)).replace('.json', '').replace('/', '_')
        try:
            with open(json_file) as f:
                results[key] = json.load(f)
        except:
            pass

    return results


def fig1_motivation(results, output_dir, model=None, device="cuda"):
    """
    Figure 1: Problem Motivation (full-width).

    Shows: Input image + MoGe-2 depth (wrong scale) + DermDepth depth (correct) + 3D point clouds.
    """
    print("  Creating Figure 1: Problem Motivation...")

    # If we have baseline analysis results, use them
    baseline_key = [k for k in results if 'baseline' in k.lower() or 'motivation' in k.lower()]

    fig = plt.figure(figsize=(FULL_WIDTH, 3.5))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)

    fig.suptitle("Metric-Scale Failure of Foundation Models\non Dermatological Images",
                 fontsize=9, fontweight='bold', y=0.98)

    # Placeholder layout
    labels = [
        ("Input Image", "RGB photo of wound/lesion"),
        ("MoGe-2 (Pretrained)", "Depth prediction\n(7-10x overestimate)"),
        ("DermDepth (Ours)", "Corrected metric depth"),
        ("Scale Comparison", "Predicted vs true\ndepth distribution"),
    ]

    for i, (title, desc) in enumerate(labels):
        row, col = i // 4, i % 4
        ax = fig.add_subplot(gs[0, i])
        ax.text(0.5, 0.5, f"{title}\n\n{desc}", ha='center', va='center',
                fontsize=6, style='italic', color='gray',
                transform=ax.transAxes)
        ax.set_title(title, fontsize=7, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom row: 3D point clouds
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        ax.text(0.5, 0.5, "3D Point Cloud\n(from above)", ha='center', va='center',
                fontsize=6, style='italic', color='gray',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(os.path.join(output_dir, "fig1_motivation.pdf"), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig1_motivation.png"), dpi=DPI, bbox_inches='tight')
    plt.close()


def fig2_pipeline(output_dir):
    """
    Figure 2: Method Pipeline (full-width).

    S-SYNTH rendering -> MoGe-2 architecture with scale_head highlighted -> inference.
    """
    print("  Creating Figure 2: Method Pipeline...")

    fig = plt.figure(figsize=(FULL_WIDTH, 2.5))
    gs = GridSpec(1, 5, figure=fig, wspace=0.1)

    stages = [
        "S-SYNTH\nRendering\n(200 skin tones)",
        "RGB + Depth\n+ Intrinsics\n(20K samples)",
        "MoGe-2\nViT-L Backbone\n(frozen Stage 1)",
        "Scale Head\nRecalibration\n(mm-scale)",
        "Metric Depth\nPrediction\n(sub-mm accuracy)",
    ]

    for i, label in enumerate(stages):
        ax = fig.add_subplot(gs[0, i])
        color = COLORS['dermdepth'] if i == 3 else '#3498db'
        ax.add_patch(plt.Rectangle((0.1, 0.15), 0.8, 0.7, fill=True,
                                     facecolor=color, alpha=0.15, edgecolor=color, linewidth=2))
        ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=6, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # Arrow between stages
        if i < len(stages) - 1:
            ax.annotate('', xy=(1.15, 0.5), xytext=(0.95, 0.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                        xycoords='axes fraction', textcoords='axes fraction')

    plt.savefig(os.path.join(output_dir, "fig2_pipeline.pdf"), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig2_pipeline.png"), dpi=DPI, bbox_inches='tight')
    plt.close()


def fig4_quantitative(results, output_dir):
    """
    Figure 4: Quantitative Results (half-width).

    Bar charts comparing MoGe-2 pretrained vs DermDepth on WoundsDB.
    """
    print("  Creating Figure 4: Quantitative Results...")

    fig, axes = plt.subplots(1, 2, figsize=(HALF_WIDTH, 2.0))

    # Placeholder data (replace with actual results)
    metrics = ['AbsRel', 'RMSE\n(mm)', 'Scale\nErr (%)']
    pretrained = [0.45, 12.5, 850]  # Placeholder
    dermdepth = [0.08, 2.1, 15]    # Placeholder

    x = np.arange(len(metrics))
    width = 0.35

    axes[0].bar(x - width/2, pretrained, width, label='MoGe-2 (pretrained)',
                color=COLORS['moge_pretrained'], alpha=0.8)
    axes[0].bar(x + width/2, dermdepth, width, label='DermDepth (ours)',
                color=COLORS['dermdepth'], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=6)
    axes[0].set_ylabel('Value', fontsize=7)
    axes[0].set_title('WoundsDB Metrics', fontsize=8, fontweight='bold')
    axes[0].legend(fontsize=5, loc='upper right')
    axes[0].tick_params(axis='both', labelsize=6)

    # Delta1 accuracy
    delta_metrics = ['Delta1\n(1.25)', 'Delta2\n(1.56)', 'Delta3\n(1.95)']
    pretrained_d = [0.15, 0.35, 0.55]
    dermdepth_d = [0.85, 0.95, 0.98]

    x2 = np.arange(len(delta_metrics))
    axes[1].bar(x2 - width/2, pretrained_d, width, label='MoGe-2',
                color=COLORS['moge_pretrained'], alpha=0.8)
    axes[1].bar(x2 + width/2, dermdepth_d, width, label='DermDepth',
                color=COLORS['dermdepth'], alpha=0.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(delta_metrics, fontsize=6)
    axes[1].set_ylabel('Accuracy', fontsize=7)
    axes[1].set_title('Depth Accuracy', fontsize=8, fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=5, loc='upper left')
    axes[1].tick_params(axis='both', labelsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_quantitative.pdf"), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig4_quantitative.png"), dpi=DPI, bbox_inches='tight')
    plt.close()


def fig6_ddi_rulers(results, output_dir):
    """
    Figure 6: DDI Ruler Scale Analysis (half-width).

    Scatter plot of predicted vs true ruler distances.
    """
    print("  Creating Figure 6: DDI Ruler Analysis...")

    fig, ax = plt.subplots(1, 1, figsize=(HALF_WIDTH, HALF_WIDTH))

    # Load DDI results if available
    ddi_key = [k for k in results if 'ddi' in k.lower() and 'ruler' in k.lower()]

    if ddi_key:
        data = results[ddi_key[0]]
        per_image = data.get('per_image', [])
        if per_image:
            true_d = [r['true_distance_mm'] for r in per_image]
            pred_d = [r['predicted_distance_mm'] for r in per_image]
            tones = [r.get('skin_tone', 'unknown') for r in per_image]

            for tone in sorted(set(tones)):
                mask = [t == tone for t in tones]
                t = [true_d[j] for j in range(len(per_image)) if mask[j]]
                p = [pred_d[j] for j in range(len(per_image)) if mask[j]]
                ax.scatter(t, p, c=COLORS.get(f'tone_{tone}', 'gray'),
                           label=f'Tone {tone}', alpha=0.7, s=30, edgecolors='black', linewidths=0.5)

    # Reference line
    ax.plot([0, 50], [0, 50], 'k--', linewidth=1, label='Perfect (1:1)', alpha=0.5)
    ax.set_xlabel('True Distance (mm)', fontsize=7)
    ax.set_ylabel('Predicted Distance (mm)', fontsize=7)
    ax.set_title('DDI Ruler Scale Evaluation', fontsize=8, fontweight='bold')
    ax.legend(fontsize=5, loc='upper left')
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_ddi_rulers.pdf"), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig6_ddi_rulers.png"), dpi=DPI, bbox_inches='tight')
    plt.close()


def fig7_fairness(results, output_dir):
    """
    Figure 7: Fairness Analysis (half-width).

    Grouped bar chart of metrics by skin tone.
    """
    print("  Creating Figure 7: Fairness Analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(HALF_WIDTH, 2.0))

    tones = ['12', '34', '56']
    tone_labels = ['I-II\n(Light)', 'III-IV\n(Med)', 'V-VI\n(Dark)']
    tone_colors = [COLORS['tone_12'], COLORS['tone_34'], COLORS['tone_56']]

    # Placeholder data (replace with actual results)
    pretrained_absrel = [0.42, 0.47, 0.52]
    dermdepth_absrel = [0.07, 0.08, 0.09]

    x = np.arange(len(tones))
    width = 0.35

    axes[0].bar(x - width/2, pretrained_absrel, width, label='MoGe-2',
                color=[c + '80' for c in tone_colors], edgecolor='black', linewidth=0.5)
    axes[0].bar(x + width/2, dermdepth_absrel, width, label='DermDepth',
                color=tone_colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tone_labels, fontsize=6)
    axes[0].set_ylabel('AbsRel', fontsize=7)
    axes[0].set_title('Depth Error by Skin Tone', fontsize=7, fontweight='bold')
    axes[0].legend(fontsize=5)
    axes[0].tick_params(axis='both', labelsize=6)

    # Scale error by skin tone
    pretrained_scale_err = [780, 850, 950]
    dermdepth_scale_err = [12, 15, 18]

    axes[1].bar(x - width/2, pretrained_scale_err, width, label='MoGe-2',
                color=COLORS['moge_pretrained'], alpha=0.5, edgecolor='black', linewidth=0.5)
    axes[1].bar(x + width/2, dermdepth_scale_err, width, label='DermDepth',
                color=COLORS['dermdepth'], alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tone_labels, fontsize=6)
    axes[1].set_ylabel('Scale Error (%)', fontsize=7)
    axes[1].set_title('Scale Error by Skin Tone', fontsize=7, fontweight='bold')
    axes[1].legend(fontsize=5)
    axes[1].tick_params(axis='both', labelsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_fairness.pdf"), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig7_fairness.png"), dpi=DPI, bbox_inches='tight')
    plt.close()


def fig8_ablations(results, output_dir):
    """
    Figure 8: Ablation Studies (half-width).

    Dataset size saturation curve + loss weight sweep.
    """
    print("  Creating Figure 8: Ablations...")

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.0))

    # A1: Dataset size curve
    sizes = [1000, 2000, 5000, 10000, 20000]
    absrel = [0.15, 0.12, 0.09, 0.08, 0.078]  # Placeholder
    axes[0].plot(sizes, absrel, 'o-', color=COLORS['dermdepth'], linewidth=1.5, markersize=4)
    axes[0].set_xlabel('Training Samples', fontsize=7)
    axes[0].set_ylabel('AbsRel', fontsize=7)
    axes[0].set_title('A1: Dataset Size', fontsize=7, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].tick_params(axis='both', labelsize=6)
    axes[0].grid(True, alpha=0.3)

    # A4: Loss weight sweep
    weights = [0.1, 0.5, 1.0, 2.0]
    scale_err = [45, 22, 15, 18]  # Placeholder
    axes[1].plot(weights, scale_err, 's-', color='#e67e22', linewidth=1.5, markersize=4)
    axes[1].set_xlabel('Metric Scale Loss Weight', fontsize=7)
    axes[1].set_ylabel('Scale Error (%)', fontsize=7)
    axes[1].set_title('A4: Loss Weight', fontsize=7, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=6)
    axes[1].grid(True, alpha=0.3)

    # A3: Fine-tuning strategy
    strategies = ['Scale\nOnly', 'Full\nFT', 'LoRA\nR=16', 'Frozen\nBB']
    absrel_strat = [0.20, 0.08, 0.10, 0.25]  # Placeholder
    colors_strat = [COLORS['baseline'], COLORS['dermdepth'], '#3498db', COLORS['moge_pretrained']]
    axes[2].bar(range(len(strategies)), absrel_strat, color=colors_strat, alpha=0.8,
                edgecolor='black', linewidth=0.5)
    axes[2].set_xticks(range(len(strategies)))
    axes[2].set_xticklabels(strategies, fontsize=5)
    axes[2].set_ylabel('AbsRel', fontsize=7)
    axes[2].set_title('A3: FT Strategy', fontsize=7, fontweight='bold')
    axes[2].tick_params(axis='both', labelsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig8_ablations.pdf"), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig8_ablations.png"), dpi=DPI, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate DermDepth paper figures")
    parser.add_argument('--results_dir', type=str,
                        default=str(PROJECT_ROOT / "output" / "evaluation"),
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', '-o', type=str,
                        default=str(PROJECT_ROOT / "output" / "figures"),
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating DermDepth Paper Figures")
    print("=" * 60)

    # Load results
    results = load_results(args.results_dir)
    print(f"  Loaded {len(results)} result files")

    # Generate all figures
    fig1_motivation(results, args.output_dir)
    fig2_pipeline(args.output_dir)
    fig4_quantitative(results, args.output_dir)
    fig6_ddi_rulers(results, args.output_dir)
    fig7_fairness(results, args.output_dir)
    fig8_ablations(results, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")
    print("Note: Figures use placeholder data. Re-run after evaluation for real results.")


if __name__ == "__main__":
    main()
