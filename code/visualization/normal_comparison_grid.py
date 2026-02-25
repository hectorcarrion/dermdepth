#!/usr/bin/env python3
"""Side-by-side normal map comparison: GT, Base, Exp A, D1, D2, D3.

Exports multi-page PDFs with ALL samples from SKINL2, WoundsDB,
and 6 synthetic training samples. Uses MoGe's standard normal colormap
(blue/purple for camera-facing surfaces).
"""

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
sys.path.insert(0, str(MOGE_ROOT))
import utils3d

SKINL2_DIR = PROJECT_ROOT / "output" / "eval_data" / "skinl2"
WOUNDSDB_DIR = PROJECT_ROOT / "output" / "eval_data" / "woundsdb"
SYNTH_DIR = PROJECT_ROOT / "data" / "dermdepth_train" / "colab_gen" / "DermDepthSynth"
OUT_DIR = PROJECT_ROOT / "output" / "figures" / "normal_comparison"

MODELS = {
    'Base MoGe-2': 'Ruicheng/moge-2-vitl-normal',
    'Exp A (scale)': str(PROJECT_ROOT / "output/training/exp_a/checkpoint/00001000_ema.pt"),
    'D1 (norm\\nfrom A)': str(PROJECT_ROOT / "output/training/exp_d1/checkpoint/00002500_ema.pt"),
    'D2 (norm\\nfrom base)': str(PROJECT_ROOT / "output/training/exp_d2/checkpoint/00002500_ema.pt"),
    'D3 (joint)': str(PROJECT_ROOT / "output/training/exp_d3/checkpoint/00002500_ema.pt"),
}


def load_model(model_path, device='cuda'):
    import os
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")
    if os.path.isfile(model_path):
        ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
        cfg = ckpt.get('model_config', None)
        model = MoGeModel(**cfg) if cfg else MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model = MoGeModel.from_pretrained(model_path)
    return model.to(device).eval()


def infer_normal(model, image_path, device='cuda'):
    import torchvision.transforms.functional as TF
    img = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model.infer(img_tensor)
    n = out['normal']
    if isinstance(n, torch.Tensor):
        n = n.cpu().numpy()
        if n.ndim > 3 and n.shape[0] == 1:
            n = n.squeeze(0)
    return n


def colorize_normal(normal, mask=None):
    """MoGe standard normal colormap: R=X, G=-Y, B=-Z (blue = camera-facing).

    Exactly matches MoGe/moge/utils/vis.py:colorize_normal().
    """
    n = normal.copy()
    if mask is not None:
        n = np.where(mask[..., None], n, 0)
    else:
        invalid = ~np.all(np.isfinite(n), axis=-1)
        n[invalid] = 0
    n = n * [0.5, -0.5, -0.5] + 0.5
    return (n.clip(0, 1) * 255).astype(np.uint8)


def derive_gt_normal(gt_depth, height, width, mask=None, intrinsics=None,
                     edge_threshold=88):
    """Derive GT normal map from depth."""
    if intrinsics is None:
        fx = fy = width / (2.0 * np.tan(np.radians(30.0)))
        cx, cy = width / 2.0, height / 2.0
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if mask is None:
        mask = np.isfinite(gt_depth) & (gt_depth > 0)
    kwargs = dict(intrinsics=intrinsics, mask=mask)
    if edge_threshold is not None:
        kwargs['edge_threshold'] = edge_threshold
    normal, normal_mask = utils3d.np.depth_map_to_normal_map(gt_depth, **kwargs)
    normal = np.where(normal_mask[..., None], normal, np.nan)
    return normal


def resize_to(arr, target_h, target_w):
    from scipy.ndimage import zoom
    h, w = arr.shape[:2]
    if (h, w) == (target_h, target_w):
        return arr
    if arr.ndim == 2:
        return zoom(arr, (target_h / h, target_w / w), order=1)
    return zoom(arr, (target_h / h, target_w / w, 1), order=1)


def load_synth_depth(sample_dir):
    """Load depth from 16-bit PNG (MoGe encoding) + meta.json intrinsics."""
    from moge.utils.io import read_depth
    depth = read_depth(str(sample_dir / "depth.png"))
    with open(sample_dir / "meta.json") as f:
        meta = json.load(f)
    # Denormalize intrinsics from meta
    intr = np.array(meta['intrinsics'], dtype=np.float64)
    w, h = meta['width'], meta['height']
    pixel_intr = intr.copy()
    pixel_intr[0] *= w  # fx, cx
    pixel_intr[1] *= h  # fy, cy
    # Keep depth in mm (normals are scale-invariant; mm avoids numerical issues)
    return depth, pixel_intr, h, w


def make_page(pdf, samples, dataset_dir, loaded_models, device, dataset_name,
              rows_per_page=6, is_synth=False):
    """Write pages to a PdfPages object for a batch of samples."""
    n_cols = 2 + len(loaded_models)  # Image + GT + models
    col_labels = ['Image', 'GT Normal'] + list(loaded_models.keys())

    for page_start in range(0, len(samples), rows_per_page):
        page_samples = samples[page_start:page_start + rows_per_page]
        n_rows = len(page_samples)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 2.8, n_rows * 2.5),
                                 squeeze=False)

        for si, sample_name in enumerate(page_samples):
            sample_dir = dataset_dir / sample_name
            image_path = sample_dir / "image.png"
            if not image_path.exists():
                for ax in axes[si]:
                    ax.axis('off')
                continue

            # Load GT depth and derive normals
            if is_synth:
                gt_depth, pixel_intr, h, w = load_synth_depth(sample_dir)
                gt_mask = np.isfinite(gt_depth) & (gt_depth > 0)
                gt_normal = derive_gt_normal(gt_depth, h, w, mask=gt_mask,
                                             intrinsics=pixel_intr,
                                             edge_threshold=None)
            else:
                gt_depth = np.load(sample_dir / "gt_depth.npy")
                gt_mask_path = sample_dir / "gt_mask.npy"
                gt_mask = np.load(gt_mask_path) if gt_mask_path.exists() else None
                h, w = gt_depth.shape[:2]
                gt_normal = derive_gt_normal(gt_depth, h, w, mask=gt_mask)

            img = np.array(Image.open(image_path).convert('RGB'))

            # Input image
            axes[si, 0].imshow(img)
            axes[si, 0].axis('off')
            if si == 0:
                axes[si, 0].set_title(col_labels[0], fontsize=9, fontweight='bold')

            # GT normal
            gt_rgb = colorize_normal(gt_normal)
            axes[si, 1].imshow(gt_rgb)
            axes[si, 1].axis('off')
            if si == 0:
                axes[si, 1].set_title(col_labels[1], fontsize=9, fontweight='bold')

            # Model predictions
            for mi, (name, model) in enumerate(loaded_models.items()):
                col = 2 + mi
                pred_normal = infer_normal(model, str(image_path), device)
                if pred_normal.shape[:2] != (h, w):
                    pred_normal = resize_to(pred_normal, h, w)

                pred_rgb = colorize_normal(pred_normal)
                axes[si, col].imshow(pred_rgb)
                axes[si, col].axis('off')
                if si == 0:
                    axes[si, col].set_title(col_labels[col], fontsize=9, fontweight='bold')

            # Row label
            short_name = sample_name[:30]
            axes[si, 0].set_ylabel(short_name, fontsize=6, rotation=0,
                                   labelpad=5, ha='right', va='center')

        page_num = page_start // rows_per_page + 1
        total_pages = (len(samples) + rows_per_page - 1) // rows_per_page
        fig.suptitle(f'{dataset_name} — Normal Comparison (page {page_num}/{total_pages})',
                     fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Page {page_num}/{total_pages} ({len(page_samples)} samples)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['skinl2', 'woundsdb', 'synth', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Load all models once
    print("Loading models...")
    loaded_models = {}
    for name, path in MODELS.items():
        print(f"  {name}...", end="", flush=True)
        loaded_models[name] = load_model(path, device)
        print(" OK")

    if args.dataset in ('skinl2', 'all'):
        skinl2_samples = sorted([d.name for d in SKINL2_DIR.iterdir() if d.is_dir()])
        print(f"\n=== SKINL2 ({len(skinl2_samples)} samples) ===")
        pdf_path = OUT_DIR / "skinl2_normal_comparison.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            make_page(pdf, skinl2_samples, SKINL2_DIR, loaded_models, device,
                      'SKINL2', rows_per_page=6)
        print(f"Saved: {pdf_path}")

    if args.dataset in ('woundsdb', 'all'):
        wdb_samples = sorted([d.name for d in WOUNDSDB_DIR.iterdir() if d.is_dir()])
        print(f"\n=== WoundsDB ({len(wdb_samples)} samples) ===")
        pdf_path = OUT_DIR / "woundsdb_normal_comparison.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            make_page(pdf, wdb_samples, WOUNDSDB_DIR, loaded_models, device,
                      'WoundsDB', rows_per_page=6)
        print(f"Saved: {pdf_path}")

    if args.dataset in ('synth', 'all'):
        synth_all = sorted([d.name for d in SYNTH_DIR.iterdir()
                            if d.is_dir() and (d / "image.png").exists()])
        synth_samples = random.sample(synth_all, min(6, len(synth_all)))
        print(f"\n=== Synthetic ({len(synth_samples)} samples) ===")
        pdf_path = OUT_DIR / "synthetic_normal_comparison.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            make_page(pdf, synth_samples, SYNTH_DIR, loaded_models, device,
                      'DermDepth-Synth (Training)', rows_per_page=6, is_synth=True)
        print(f"Saved: {pdf_path}")

    print("\nAll done!")


if __name__ == '__main__':
    main()
