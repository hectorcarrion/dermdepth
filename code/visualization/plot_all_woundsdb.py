#!/usr/bin/env python3
"""Plot all prepared WoundsDB scenes: photo + depth overlay grid."""
import numpy as np
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

base = Path('/workspace/hector/dermdepth/output/eval_data/woundsdb')
out_dir = Path('/workspace/hector/dermdepth/output/verification')
out_dir.mkdir(parents=True, exist_ok=True)

# Collect all dense GT scenes
scenes = []
for d in sorted(base.iterdir()):
    meta_path = d / 'meta.json'
    if not meta_path.exists():
        continue
    meta = json.load(open(meta_path))
    if meta.get('gt_type') == 'dense':
        scenes.append(d.name)

print(f'Plotting {len(scenes)} scenes')

COLS = 4
ROWS_PER_PAGE = 4
SCENES_PER_PAGE = COLS * ROWS_PER_PAGE
n_pages = math.ceil(len(scenes) / SCENES_PER_PAGE)

for page in range(n_pages):
    page_scenes = scenes[page * SCENES_PER_PAGE : (page + 1) * SCENES_PER_PAGE]
    n_rows = math.ceil(len(page_scenes) / COLS)

    fig, axes = plt.subplots(n_rows, COLS * 2, figsize=(COLS * 2 * 3.2, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, scene in enumerate(page_scenes):
        row = idx // COLS
        col = idx % COLS
        ax_photo = axes[row, col * 2]
        ax_depth = axes[row, col * 2 + 1]

        scene_dir = base / scene
        img = np.array(Image.open(scene_dir / 'image.png').convert('RGB'))
        gt = np.load(scene_dir / 'gt_depth.npy')
        mask = np.load(scene_dir / 'gt_mask.npy')

        gt_valid = gt[mask]
        vmin, vmax = gt_valid.min(), gt_valid.max()
        coverage = mask.sum() / mask.size * 100

        # Photo
        ax_photo.imshow(img)
        label = scene.replace('_scene_1', '').replace('_scene_2', '_s2').replace('_scene_4', '_s4')
        ax_photo.set_title(label, fontsize=7)
        ax_photo.axis('off')

        # Depth overlay on photo
        alpha = 0.55
        img_f = img.astype(np.float32) / 255.0
        gt_vis = gt.copy()
        gt_vis[~mask] = np.nan
        depth_norm = (gt_vis - vmin) / (vmax - vmin + 1e-8)
        depth_rgb = plt.cm.turbo(depth_norm)[:, :, :3]
        overlay = np.where(mask[:, :, None],
                           alpha * depth_rgb + (1 - alpha) * img_f,
                           img_f)
        ax_depth.imshow(np.clip(overlay, 0, 1))
        ax_depth.set_title(f'{coverage:.0f}% | {gt_valid.mean():.2f}m', fontsize=7)
        ax_depth.axis('off')

    # Hide unused axes
    for idx in range(len(page_scenes), n_rows * COLS):
        row = idx // COLS
        col = idx % COLS
        axes[row, col * 2].axis('off')
        axes[row, col * 2 + 1].axis('off')

    plt.suptitle(f'WoundsDB Dense ToF GT — All Scenes (Page {page+1}/{n_pages})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f'fig19_all_scenes_page{page+1}.png'
    plt.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname} ({len(page_scenes)} scenes)')

print('Done')
