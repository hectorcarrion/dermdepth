#!/usr/bin/env python3
"""Fig 6: Scale Tradeoff Trajectory v3 — better spacing, less overlap.

Shows how Exp A (synth-only) trades off between datasets,
while Exp G (real data) and Exp H (real+DDI) converge toward perfect scale.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# MICCAI style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
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

exp_a_steps = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000]
exp_a_sk = [17.163, 8.593, 3.881, 1.916, 1.153, 0.543, 0.344, 0.223, 0.188]
exp_a_wdb = [0.610, 0.525, 0.428, 0.346, 0.279, 0.191, 0.140, 0.094, 0.076]

exp_g_steps = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
exp_g_sk = [1.153, 1.103, 1.046, 0.994, 0.951, 0.923, 0.902, 0.890, 0.888, 0.886]
exp_g_wdb = [0.279, 0.339, 0.421, 0.506, 0.593, 0.676, 0.751, 0.813, 0.868, 0.916]

exp_h_steps = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
exp_h_sk = [1.153, 1.065, 1.009, 0.960, 0.931, 0.909, 0.895, 0.883, 0.877, 0.872]
exp_h_wdb = [0.279, 0.321, 0.391, 0.469, 0.553, 0.644, 0.727, 0.798, 0.858, 0.910]

baselines = {
    'DA$^3$': (4.157, 0.715),
    'MapAnything': (10.999, 0.773),
    'PPD': (16.065, 0.651),
}

# ============================================================
# Plot — wider figure with more breathing room
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

# Shaded "ideal zone" around (1,1)
rect = FancyBboxPatch((0.8, 0.8), 0.4, 0.4, boxstyle="round,pad=0.03",
                       facecolor='#2ecc71', alpha=0.10, edgecolor='#2ecc71',
                       linewidth=1.2, linestyle='--', zorder=1)
ax.add_patch(rect)

# Perfect scale star
ax.plot(1.0, 1.0, marker='*', color='gold', markersize=28, zorder=10,
        markeredgecolor='#333', markeredgewidth=1.2)

# ---- Exp A trajectory (synth only) ----
ax.plot(exp_a_sk, exp_a_wdb, 'o-', color='#3498db', markersize=5, linewidth=2.2,
        zorder=5, alpha=0.9)
# Label start and key steps — keep "Base MoGe-2" inside plot
ax.annotate('Base MoGe-2', (exp_a_sk[0], exp_a_wdb[0]),
            fontsize=7.5, color='#3498db', fontweight='bold', ha='right',
            textcoords='offset points', xytext=(-6, 6))
ax.annotate('step 1000', (exp_a_sk[4], exp_a_wdb[4]),
            fontsize=7, color='#3498db', fontweight='bold',
            textcoords='offset points', xytext=(8, -12))
ax.annotate('step 4000', (exp_a_sk[-1], exp_a_wdb[-1]),
            fontsize=7, color='#3498db', fontweight='bold',
            textcoords='offset points', xytext=(-8, -14), ha='center')

# ---- Connecting arrow from Exp A s1000 to G/H start ----
ax.annotate('', xy=(exp_g_sk[1], exp_g_wdb[1]),
            xytext=(exp_a_sk[4]*0.98, exp_a_wdb[4]*1.02),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5,
                           connectionstyle='arc3,rad=-0.15', ls='--'))
ax.annotate('+ real data', (0.75, 0.33), fontsize=7.5, color='#7f8c8d',
            ha='center', style='italic', fontweight='bold')

# ---- Exp G trajectory ----
ax.plot(exp_g_sk, exp_g_wdb, 's-', color='#e67e22', markersize=4.5, linewidth=2.2,
        zorder=6, alpha=0.9)
ax.annotate('Exp G', (exp_g_sk[-1], exp_g_wdb[-1]),
            fontsize=7.5, color='#e67e22', fontweight='bold',
            textcoords='offset points', xytext=(-22, -12), ha='center')

# ---- Exp H trajectory ----
ax.plot(exp_h_sk, exp_h_wdb, 'D-', color='#27ae60', markersize=4.5, linewidth=2.2,
        zorder=7, alpha=0.9)
ax.annotate('Exp H', (exp_h_sk[-1], exp_h_wdb[-1]),
            fontsize=7.5, color='#27ae60', fontweight='bold',
            textcoords='offset points', xytext=(10, 4))

# ---- Baselines ----
baseline_markers = {
    'DA$^3$': ('^', '#e74c3c'),
    'MapAnything': ('v', '#9b59b6'),
    'PPD': ('P', '#795548'),
}
baseline_offsets = {
    'DA$^3$': (8, 6),
    'MapAnything': (0, 8),
    'PPD': (-10, -12),
}
for name, (sk, wdb) in baselines.items():
    marker, color = baseline_markers[name]
    ax.plot(sk, wdb, marker=marker, color=color, markersize=11, markeredgecolor='white',
            markeredgewidth=1.0, zorder=8)
    ox, oy = baseline_offsets[name]
    ax.annotate(name, (sk, wdb), fontsize=7.5, color=color, fontweight='bold',
                textcoords='offset points', xytext=(ox, oy), ha='center')

# Reference lines at scale=1.0
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.35)
ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.35)

# Formatting
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('SKINL2 Scale Ratio (pred / GT)\n$\\it{Dermoscopic\\ scale}$', fontsize=11)
ax.set_ylabel('WoundsDB Scale Ratio (pred / GT)\n$\\it{Macroscopic\\ scale}$', fontsize=11)
ax.set_title('Scale Correction Trajectory', fontsize=13, fontweight='bold', pad=10)

# Wider limits for breathing room
ax.set_xlim(0.1, 25)
ax.set_ylim(0.05, 1.5)

# Custom ticks for cleaner labels
ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2, 5, 10, 20])
ax.set_xticklabels(['0.1', '0.2', '0.5', '1.0', '2', '5', '10', '20'])
ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0])
ax.set_yticklabels(['0.05', '0.1', '0.2', '0.5', '1.0'])

# Region annotations (corners)
ax.text(0.12, 0.06, 'Underestimates both', fontsize=7.5, color='#95a5a6',
        ha='left', style='italic')
ax.text(12, 0.06, 'Over-predicts skin\nunder-predicts wound', fontsize=7,
        color='#95a5a6', ha='center', style='italic')

# Custom legend — bottom right, out of the way
legend_handles = [
    Line2D([0], [0], marker='*', color='gold', markersize=12,
           markeredgecolor='#333', linestyle='None', label='Perfect scale'),
    Line2D([0], [0], marker='o', color='#3498db', markersize=5,
           linewidth=2, label='Synth only (Exp A)'),
    Line2D([0], [0], marker='s', color='#e67e22', markersize=5,
           linewidth=2, label='+Real depth (Exp G)'),
    Line2D([0], [0], marker='D', color='#27ae60', markersize=5,
           linewidth=2, label='+Real+DDI (Exp H)'),
    Line2D([0], [0], marker='^', color='#e74c3c', markersize=7,
           linestyle='None', label='DA$^3$-Nested'),
    Line2D([0], [0], marker='v', color='#9b59b6', markersize=7,
           linestyle='None', label='MapAnything'),
    Line2D([0], [0], marker='P', color='#795548', markersize=7,
           linestyle='None', label='PPD'),
]
ax.legend(handles=legend_handles, loc='center left', framealpha=0.95,
          edgecolor='#bdc3c7', fontsize=7.5, ncol=1,
          bbox_to_anchor=(1.02, 0.5))

ax.grid(True, alpha=0.12, which='both')

plt.tight_layout()

for ext in ['pdf', 'png']:
    fig.savefig(OUT_DIR / f'fig6_scale_tradeoff_v3.{ext}')
    print(f'Saved: {OUT_DIR / f"fig6_scale_tradeoff_v3.{ext}"}')

plt.close()
print('Done!')
