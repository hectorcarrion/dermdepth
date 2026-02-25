#!/usr/bin/env python3
"""Generate publication-quality figures for DermDepth paper.

Creates:
1. Figure 2: Scale correction ablation (3-panel: scale, absrel, geometry)
2. Figure 3: Per-version and per-disease breakdown
3. Figure 4: Summary bar chart of best models
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# MICCAI style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUT_DIR = Path("output/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data from evaluations (hardcoded for reproducibility)
# ============================================================

# Exp A: Scale head only (EMA checkpoints)
exp_a_steps = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000, 4000]
exp_a_skinl2_scale = [16.097, 8.593, 3.881, 1.916, 1.109, 0.738, 0.543, 0.423, 0.344, 0.223, 0.188]
exp_a_skinl2_absrel = [15.110, 7.600, 2.893, 0.991, 0.424, 0.405, 0.498, 0.587, 0.658, 0.776, 0.812]
exp_a_skinl2_sid1 = [1.000] * 11  # Geometry perfectly preserved
exp_a_woundsdb_scale = [0.618, 0.525, 0.428, 0.346, 0.279, 0.229, 0.191, 0.162, 0.140, 0.094, 0.076]
exp_a_woundsdb_absrel = [0.376, 0.468, 0.566, 0.649, 0.716, 0.768, 0.806, 0.835, 0.858, 0.905, 0.923]
exp_a_woundsdb_sid1 = [0.911] * 11  # Geometry preserved (different baseline)

# Exp B: Decoder fine-tune (EMA, starts from warmup step 1000)
# Total steps = warmup 1000 + decoder steps
exp_b_steps_total = [1000 + s for s in [2000, 4000, 6000, 8000]]  # 3000, 5000, 7000, 9000 total
exp_b_skinl2_scale = [0.144, 0.153, 0.146, 0.137]
exp_b_skinl2_absrel = [0.856, 0.847, 0.855, 0.864]
exp_b_skinl2_sid1 = [0.999, 0.994, 0.992, 0.981]
exp_b_woundsdb_scale = [0.060, 0.059, 0.053, 0.046]
exp_b_woundsdb_absrel = [0.941, 0.943, 0.949, 0.956]
exp_b_woundsdb_sid1 = [0.858, 0.787, 0.763, 0.698]

# Try to load Exp C results if available
exp_c_steps_total = []
exp_c_skinl2_scale = []
exp_c_skinl2_absrel = []
exp_c_skinl2_sid1 = []
exp_c_woundsdb_scale = []
exp_c_woundsdb_absrel = []
exp_c_woundsdb_sid1 = []

for step in [3000, 6000, 9000, 12000, 15000]:
    result_dir = Path(f"output/evaluation/exp_c_step{step}_ema")
    skinl2_f = result_dir / "skinl2" / "results.json"
    woundsdb_f = result_dir / "woundsdb" / "results.json"
    if skinl2_f.exists() and woundsdb_f.exists():
        with open(skinl2_f) as f:
            sr = json.load(f)
        with open(woundsdb_f) as f:
            wr = json.load(f)
        # Handle both 'overall' and 'summary' key formats
        ss = sr.get("overall", sr.get("summary", {}))
        ws = wr.get("overall", wr.get("summary", {}))
        def _val(d, key):
            v = d.get(key)
            return v.get("mean") if isinstance(v, dict) else v
        total_step = 1000 + step  # warmup 1000 + exp_c steps
        exp_c_steps_total.append(total_step)
        exp_c_skinl2_scale.append(_val(ss, "scale_ratio"))
        exp_c_skinl2_absrel.append(_val(ss, "absrel") or _val(ss, "abs_rel"))
        exp_c_skinl2_sid1.append(_val(ss, "si_delta1"))
        exp_c_woundsdb_scale.append(_val(ws, "scale_ratio"))
        exp_c_woundsdb_absrel.append(_val(ws, "absrel") or _val(ws, "abs_rel"))
        exp_c_woundsdb_sid1.append(_val(ws, "si_delta1"))

has_exp_c = len(exp_c_steps_total) > 0

# ============================================================
# Figure 2: Ablation comparison (3-panel)
# ============================================================
def make_ablation_figure():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))  # MICCAI column width

    colors = {'A': '#2166ac', 'B': '#b2182b', 'C': '#4daf4a'}
    markers = {'A': 'o', 'B': 's', 'C': '^'}

    # --- Panel 1: SKINL2 Scale Ratio ---
    ax = axes[0]
    ax.semilogy(exp_a_steps, exp_a_skinl2_scale, f'-{markers["A"]}', color=colors['A'],
                label='Exp A: Scale head (2.1M)', markersize=4, linewidth=1.2)
    ax.semilogy(exp_b_steps_total, exp_b_skinl2_scale, f'-{markers["B"]}', color=colors['B'],
                label='Exp B: Decoder (22M)', markersize=4, linewidth=1.2)
    if has_exp_c:
        ax.semilogy(exp_c_steps_total, exp_c_skinl2_scale, f'-{markers["C"]}', color=colors['C'],
                    label='Exp C: Full (326M)', markersize=4, linewidth=1.2)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvspan(900, 1100, alpha=0.15, color='green')
    ax.annotate('Best\n(1K)', xy=(1000, 1.109), fontsize=6.5, color='#2166ac',
                ha='center', va='bottom', xytext=(1500, 2.5),
                arrowprops=dict(arrowstyle='->', color='#2166ac', lw=0.8))
    ax.set_xlabel('Training Step')
    ax.set_ylabel('SKINL2 Scale Ratio (pred/gt)')
    ax.set_title('(a) Scale Correction', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=6.5)
    ax.set_ylim(0.03, 25)
    ax.set_xlim(-200, max(exp_a_steps[-1], exp_b_steps_total[-1] if exp_b_steps_total else 0,
                          exp_c_steps_total[-1] if exp_c_steps_total else 0) + 500)

    # --- Panel 2: SKINL2 AbsRel ---
    ax = axes[1]
    ax.plot(exp_a_steps, exp_a_skinl2_absrel, f'-{markers["A"]}', color=colors['A'],
            label='Exp A', markersize=4, linewidth=1.2)
    ax.plot(exp_b_steps_total, exp_b_skinl2_absrel, f'-{markers["B"]}', color=colors['B'],
            label='Exp B', markersize=4, linewidth=1.2)
    if has_exp_c:
        ax.plot(exp_c_steps_total, exp_c_skinl2_absrel, f'-{markers["C"]}', color=colors['C'],
                label='Exp C', markersize=4, linewidth=1.2)

    # Mark best
    best_idx = np.argmin(exp_a_skinl2_absrel)
    ax.plot(exp_a_steps[best_idx], exp_a_skinl2_absrel[best_idx], '*',
            color=colors['A'], markersize=10, zorder=5)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('SKINL2 Abs. Relative Error')
    ax.set_title('(b) Depth Accuracy', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=6.5)
    ax.set_ylim(0, 1.1)

    # --- Panel 3: Geometry Preservation (SI-Delta1) ---
    ax = axes[2]
    ax.plot(exp_a_steps, exp_a_skinl2_sid1, f'-{markers["A"]}', color=colors['A'],
            label='Exp A (SKINL2)', markersize=4, linewidth=1.2)
    ax.plot(exp_b_steps_total, exp_b_skinl2_sid1, f'--{markers["B"]}', color=colors['B'],
            label='Exp B (SKINL2)', markersize=4, linewidth=1.2, alpha=0.7)
    ax.plot(exp_b_steps_total, exp_b_woundsdb_sid1, f'-{markers["B"]}', color=colors['B'],
            label='Exp B (WoundsDB)', markersize=4, linewidth=1.2)
    if has_exp_c:
        ax.plot(exp_c_steps_total, exp_c_skinl2_sid1, f'--{markers["C"]}', color=colors['C'],
                label='Exp C (SKINL2)', markersize=4, linewidth=1.2, alpha=0.7)
        ax.plot(exp_c_steps_total, exp_c_woundsdb_sid1, f'-{markers["C"]}', color=colors['C'],
                label='Exp C (WoundsDB)', markersize=4, linewidth=1.2)

    ax.axhline(y=0.911, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(200, 0.905, 'Baseline WoundsDB', fontsize=5.5, color='gray')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('SI-Delta1 (geometry quality)')
    ax.set_title('(c) Geometry Preservation', fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=5.5, ncol=1)
    ax.set_ylim(0.6, 1.02)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig2_ablation.png')
    fig.savefig(OUT_DIR / 'fig2_ablation.pdf')
    plt.close(fig)
    print(f"Saved fig2_ablation.{{png,pdf}}")


# ============================================================
# Figure 3: Per-version breakdown
# ============================================================
def make_version_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))

    # Data: per-version
    versions = ['v1\n(close-up)', 'v2\n(medium)', 'v3\n(distant)']
    baseline_scale = [9.17, 26.26, 38.41]
    dermdepth_scale = [0.82, 1.37, 2.46]
    dermdepth_absrel = [0.272, 0.407, 1.467]

    x = np.arange(len(versions))
    width = 0.35

    # Panel 1: Scale ratio comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, baseline_scale, width, label='Baseline MoGe-2',
                   color='#d73027', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, dermdepth_scale, width, label='DermDepth (Ours)',
                   color='#4575b4', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Perfect (1.0)')
    ax.set_ylabel('Scale Ratio (pred/gt)')
    ax.set_title('(a) Scale Correction by Version', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_yscale('log')
    ax.set_ylim(0.5, 60)

    # Add value labels
    for bar, val in zip(bars1, baseline_scale):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val:.0f}x', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    for bar, val in zip(bars2, dermdepth_scale):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    # Panel 2: Per-disease breakdown (v1 only)
    ax = axes[1]
    diseases = ['BCC', 'DFib', 'Hem', 'Mel', 'Nev', 'Other', 'Psor', 'SebK']
    disease_scale = [0.716, 0.952, 0.768, 0.793, 0.840, 0.746, 0.917, 0.954]
    disease_n = [31, 14, 28, 16, 94, 31, 5, 31]
    disease_colors = plt.cm.Set2(np.linspace(0, 1, len(diseases)))

    bars = ax.bar(np.arange(len(diseases)), disease_scale, color=disease_colors,
                  edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylabel('Scale Ratio (pred/gt)')
    ax.set_title('(b) Per-Disease Scale (SKINL2 v1)', fontweight='bold')
    ax.set_xticks(np.arange(len(diseases)))
    ax.set_xticklabels(diseases, rotation=45, ha='right')
    ax.set_ylim(0.5, 1.15)

    # Add N labels
    for bar, n in zip(bars, disease_n):
        ax.text(bar.get_x() + bar.get_width()/2, 0.52,
                f'n={n}', ha='center', va='bottom', fontsize=5.5, color='gray')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig3_version_disease.png')
    fig.savefig(OUT_DIR / 'fig3_version_disease.pdf')
    plt.close(fig)
    print(f"Saved fig3_version_disease.{{png,pdf}}")


# ============================================================
# Figure 4: Summary comparison bar chart
# ============================================================
def make_summary_figure():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))

    models = ['Baseline\nMoGe-2', 'Exp A\n(Scale, 1K)', 'Exp B\n(Decoder, 3K)']
    colors = ['#999999', '#2166ac', '#b2182b']

    if has_exp_c:
        # Find best Exp C by closest scale to 1.0
        best_c_idx = np.argmin([abs(s - 1.0) for s in exp_c_skinl2_scale])
        best_c_step = exp_c_steps_total[best_c_idx]
        models.append(f'Exp C\n(Full, {best_c_step-1000})')
        colors.append('#4daf4a')

    # SKINL2 Scale
    skinl2_scale = [16.097, 1.109, 0.144]
    if has_exp_c:
        skinl2_scale.append(exp_c_skinl2_scale[best_c_idx])

    # SKINL2 AbsRel
    skinl2_absrel = [15.110, 0.424, 0.856]
    if has_exp_c:
        skinl2_absrel.append(exp_c_skinl2_absrel[best_c_idx])

    # WoundsDB SI-Delta1
    woundsdb_sid1 = [0.911, 0.911, 0.858]
    if has_exp_c:
        woundsdb_sid1.append(exp_c_woundsdb_sid1[best_c_idx])

    x = np.arange(len(models))

    # Panel 1: Scale
    ax = axes[0]
    bars = ax.bar(x, skinl2_scale, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylabel('SKINL2 Scale Ratio')
    ax.set_title('Scale Correction', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6.5)
    ax.set_yscale('log')
    ax.set_ylim(0.05, 25)
    for bar, val in zip(bars, skinl2_scale):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    # Panel 2: AbsRel
    ax = axes[1]
    bars = ax.bar(x, skinl2_absrel, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('SKINL2 Abs. Rel. Error')
    ax.set_title('Depth Accuracy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6.5)
    for bar, val in zip(bars, skinl2_absrel):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax.set_ylim(0, max(skinl2_absrel) * 1.3)

    # Panel 3: Geometry
    ax = axes[2]
    bars = ax.bar(x, woundsdb_sid1, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('WoundsDB SI-Delta1')
    ax.set_title('Geometry Preservation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=6.5)
    ax.set_ylim(0.6, 1.02)
    for bar, val in zip(bars, woundsdb_sid1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig4_summary.png')
    fig.savefig(OUT_DIR / 'fig4_summary.pdf')
    plt.close(fig)
    print(f"Saved fig4_summary.{{png,pdf}}")


# ============================================================
# Figure 5: WoundsDB vs SKINL2 trade-off (scatter)
# ============================================================
def make_tradeoff_figure():
    """Shows the inherent trade-off: improving derm scale degrades general-distance."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Exp A trajectory
    ax.plot(exp_a_skinl2_scale, exp_a_woundsdb_scale, '-o', color='#2166ac',
            markersize=4, linewidth=1.2, label='Exp A: Scale head', zorder=3)
    # Label a few key points
    for i, step in enumerate(exp_a_steps):
        if step in [0, 1000, 4000]:
            offset = (8, 5) if step != 0 else (-15, -10)
            ax.annotate(f'{step}', (exp_a_skinl2_scale[i], exp_a_woundsdb_scale[i]),
                       fontsize=5.5, textcoords='offset points', xytext=offset,
                       color='#2166ac')

    # Exp B trajectory
    ax.plot(exp_b_skinl2_scale, exp_b_woundsdb_scale, '-s', color='#b2182b',
            markersize=4, linewidth=1.2, label='Exp B: Decoder FT', zorder=3)

    if has_exp_c:
        ax.plot(exp_c_skinl2_scale, exp_c_woundsdb_scale, '-^', color='#4daf4a',
                markersize=4, linewidth=1.2, label='Exp C: Full FT', zorder=3)

    # Perfect point
    ax.plot(1.0, 1.0, '*', color='gold', markersize=12, markeredgecolor='black',
            markeredgewidth=0.5, zorder=5, label='Perfect scale')

    ax.set_xlabel('SKINL2 Scale Ratio')
    ax.set_ylabel('WoundsDB Scale Ratio')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.03, 25)
    ax.set_ylim(0.03, 1.5)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('Scale Trade-off Trajectory', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig5_tradeoff.png')
    fig.savefig(OUT_DIR / 'fig5_tradeoff.pdf')
    plt.close(fig)
    print(f"Saved fig5_tradeoff.{{png,pdf}}")


if __name__ == '__main__':
    print(f"Exp C data available: {has_exp_c}" + (f" ({len(exp_c_steps_total)} checkpoints)" if has_exp_c else ""))
    make_ablation_figure()
    make_version_figure()
    make_summary_figure()
    make_tradeoff_figure()
    print("All figures generated!")
