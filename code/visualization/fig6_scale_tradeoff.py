#!/usr/bin/env python3
"""Fig 6: Scale Tradeoff Trajectory — SKINL2 vs WoundsDB scale ratio.

Shows how Exp A (synth-only) trades off between datasets,
while Exp G (real data) and Exp H (real+DDI) converge toward perfect scale.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# MICCAI style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUT_DIR = Path("output/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data: Scale ratios on STRATIFIED TEST SPLITS
# ============================================================

# Exp A: Scale head only, synth-only training (from Exp A trajectory on test splits)
# Steps: 0=base, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000, 4000
# Note: step 0 = base MoGe-2, evaluated on stratified test splits
exp_a_steps = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000]
exp_a_sk = [17.163, 8.593, 3.881, 1.916, 1.153, 0.543, 0.344, 0.223, 0.188]
exp_a_wdb = [0.610, 0.525, 0.428, 0.346, 0.279, 0.191, 0.140, 0.094, 0.076]

# Exp G: Scale head, synth + real (WoundsDB + SKINL2), from Exp A s1000 EMA
exp_g_steps = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
exp_g_sk = [1.153, 1.103, 1.046, 0.994, 0.951, 0.923, 0.902, 0.890, 0.888, 0.886]
exp_g_wdb = [0.279, 0.339, 0.421, 0.506, 0.593, 0.676, 0.751, 0.813, 0.868, 0.916]

# Exp H: Scale head, synth + real + DDI pseudo-GT, from Exp A s1000 EMA
exp_h_steps = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
exp_h_sk = [1.153, 1.065, 1.009, 0.960, 0.931, 0.909, 0.895, 0.883, 0.877, 0.872]
exp_h_wdb = [0.279, 0.321, 0.391, 0.469, 0.553, 0.644, 0.727, 0.798, 0.858, 0.910]

# Baselines (on stratified test splits)
baselines = {
    'DA3-Nested': (4.157, 0.715),
    'MapAnything': (10.999, 0.773),
    'PPD': (16.065, 0.651),
}

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))

# Shaded "ideal zone" around (1,1)
from matplotlib.patches import FancyBboxPatch
rect = FancyBboxPatch((0.8, 0.8), 0.4, 0.4, boxstyle="round,pad=0.02",
                       facecolor='#2ecc71', alpha=0.08, edgecolor='#2ecc71',
                       linewidth=1, linestyle='--', zorder=1)
ax.add_patch(rect)

# Perfect scale star
ax.plot(1.0, 1.0, marker='*', color='gold', markersize=24, zorder=10,
        markeredgecolor='#333', markeredgewidth=1.0, label='Perfect scale')

# Exp A trajectory (synth only) — traverses diagonally, never reaches (1,1)
ax.plot(exp_a_sk, exp_a_wdb, 'o-', color='#3498db', markersize=4, linewidth=2.0,
        label='Synth only', zorder=5, alpha=0.85)
# Label key steps for Exp A
ax.annotate('Base\nMoGe-2', (exp_a_sk[0], exp_a_wdb[0]), fontsize=7, color='#3498db',
            textcoords='offset points', xytext=(-8, -14), fontweight='bold', ha='center')
ax.annotate('s1000', (exp_a_sk[4], exp_a_wdb[4]), fontsize=7, color='#3498db',
            textcoords='offset points', xytext=(-12, -10), fontweight='bold')
ax.annotate('s4000', (exp_a_sk[-1], exp_a_wdb[-1]), fontsize=7, color='#3498db',
            textcoords='offset points', xytext=(-12, -10), fontweight='bold')

# Exp G trajectory — bends toward star
ax.plot(exp_g_sk, exp_g_wdb, 's-', color='#e67e22', markersize=4, linewidth=2.0,
        label='+WoundsDB +SKINL2', zorder=6, alpha=0.85)
ax.annotate('s1800', (exp_g_sk[-1], exp_g_wdb[-1]), fontsize=7, color='#e67e22',
            textcoords='offset points', xytext=(6, -2), fontweight='bold')

# Exp H trajectory — also converges, nearly overlaps G
ax.plot(exp_h_sk, exp_h_wdb, 'D-', color='#27ae60', markersize=4, linewidth=2.0,
        label='+WoundsDB +SKINL2 +DDI', zorder=7, alpha=0.85)
ax.annotate('s1800', (exp_h_sk[-1], exp_h_wdb[-1]), fontsize=7, color='#27ae60',
            textcoords='offset points', xytext=(6, 5), fontweight='bold')

# Connecting arrow from Exp A s1000 to G/H start
ax.annotate('', xy=(exp_g_sk[0]*1.02, exp_g_wdb[0]*1.05),
            xytext=(exp_a_sk[4], exp_a_wdb[4]),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5, ls='--'))
ax.annotate('Fine-tune\nwith real data', (0.65, 0.31), fontsize=7, color='#7f8c8d',
            ha='center', style='italic', fontweight='bold')

# Baselines as distinct markers
baseline_markers = {
    'DA3-Nested': ('^', '#e74c3c'),
    'MapAnything': ('v', '#9b59b6'),
    'PPD': ('P', '#795548'),
}
for name, (sk, wdb) in baselines.items():
    marker, color = baseline_markers[name]
    ax.plot(sk, wdb, marker=marker, color=color, markersize=10, markeredgecolor='white',
            markeredgewidth=0.8, label=name, zorder=8)
    ax.annotate(name, (sk, wdb), fontsize=6.5, color=color,
                textcoords='offset points', xytext=(8, -3))

# Reference lines at scale=1.0
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.4)
ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.4)

# Formatting
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('SKINL2 Scale Ratio (pred / GT)', fontsize=11)
ax.set_ylabel('WoundsDB Scale Ratio (pred / GT)', fontsize=11)
ax.set_title('Scale Correction Trajectory', fontsize=13, fontweight='bold')

# Axis limits
ax.set_xlim(0.05, 25)
ax.set_ylim(0.05, 1.5)

# Region annotations
ax.text(0.07, 0.07, 'Underestimates\nboth', fontsize=7, color='#95a5a6', ha='left', style='italic')
ax.text(5, 0.07, 'Overestimates skin depth\nunderestimates wound depth', fontsize=7, color='#95a5a6', ha='center', style='italic')

ax.legend(loc='upper left', framealpha=0.95, edgecolor='#bdc3c7', fontsize=8,
          title='Training Data', title_fontsize=8.5)
ax.grid(True, alpha=0.15, which='both')

plt.tight_layout()

for ext in ['pdf', 'png']:
    fig.savefig(OUT_DIR / f'fig6_scale_tradeoff_v2.{ext}')
    print(f'Saved: {OUT_DIR / f"fig6_scale_tradeoff_v2.{ext}"}')

plt.close()
print('Done!')
