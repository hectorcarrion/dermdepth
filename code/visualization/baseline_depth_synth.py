#!/usr/bin/env python3
"""
Save depth predictions from baseline methods on synthetic samples,
then plot comparison with GT and D3.

Step 1: Run with --method to save predictions (each in its own conda env)
Step 2: Run with --plot to create the figure (in MoGe env)

Usage:
    # In da3 env:
    CUDA_VISIBLE_DEVICES=3 conda run -n da3 python -u code/visualization/baseline_depth_synth.py --method da3nested
    # In mapanything env:
    CUDA_VISIBLE_DEVICES=3 conda run -n mapanything python -u code/visualization/baseline_depth_synth.py --method mapanything
    # In ppd env:
    CUDA_VISIBLE_DEVICES=3 conda run -n ppd python -u code/visualization/baseline_depth_synth.py --method ppd
    # In MoGe env (plot):
    CUDA_VISIBLE_DEVICES=3 conda run -n MoGe python -u code/visualization/baseline_depth_synth.py --plot
"""

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTH_DIR = PROJECT_ROOT / "data" / "dermdepth_train" / "colab_gen" / "DermDepthSynth"
CACHE_DIR = PROJECT_ROOT / "output" / "figures" / "normal_comparison" / "_baseline_cache"

# Same 6 samples as other comparisons (seed=42)
random.seed(42)
_all = sorted([d.name for d in SYNTH_DIR.iterdir()
               if d.is_dir() and (d / "image.png").exists()])
SAMPLES = random.sample(_all, min(6, len(_all)))


def save_da3nested(device='cuda'):
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "Depth-Anything-3" / "src"))
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(device).eval()
    for name in SAMPLES:
        out = CACHE_DIR / name / "da3nested_depth.npy"
        if out.exists():
            print(f"  {name}: da3nested exists, skip")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        pred = model.inference([str(SYNTH_DIR / name / "image.png")])
        d = pred.depth[0]
        if isinstance(d, torch.Tensor):
            d = d.cpu().numpy()
        np.save(out, d.astype(np.float32))
        print(f"  {name}: da3nested med={np.median(d)*1000:.1f}mm")


def save_mapanything(device='cuda'):
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "map-anything"))
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    for name in SAMPLES:
        out = CACHE_DIR / name / "mapanything_depth.npy"
        if out.exists():
            print(f"  {name}: mapanything exists, skip")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        views = load_images([str(SYNTH_DIR / name / "image.png")])
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype='bf16')
        d = preds[0]['depth_z'][0, :, :, 0].cpu().numpy()
        np.save(out, d.astype(np.float32))
        print(f"  {name}: mapanything med={np.median(d)*1000:.1f}mm")


def save_ppd(device='cuda'):
    import cv2
    PPD_DIR = PROJECT_ROOT / "baseline_methods" / "pixel-perfect-depth"
    sys.path.insert(0, str(PPD_DIR))
    from ppd.models.ppd import PixelPerfectDepth
    from ppd.moge.model.v2 import MoGeModel
    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    moge = MoGeModel.from_pretrained(str(PPD_DIR / 'checkpoints' / 'moge2.pt')).to(device).eval()
    ppd_model = PixelPerfectDepth(
        semantics_model='DA2',
        semantics_pth=str(PPD_DIR / 'checkpoints' / 'depth_anything_v2_vitl.pth'),
        sampling_steps=20
    )
    ppd_model.load_state_dict(
        torch.load(str(PPD_DIR / 'checkpoints' / 'ppd.pth'), map_location='cpu'),
        strict=False
    )
    ppd_model = ppd_model.to(device).eval()

    for name in SAMPLES:
        out = CACHE_DIR / name / "ppd_depth.npy"
        if out.exists():
            print(f"  {name}: ppd exists, skip")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(SYNTH_DIR / name / "image.png"))
        depth, resize_image = ppd_model.infer_image(image)
        depth = depth.squeeze().cpu().numpy()
        moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        moge_image = torch.tensor(moge_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        moge_depth, mask, intrinsic = moge.infer(moge_image)
        moge_depth[~mask] = moge_depth[mask].max()
        metric_depth = recover_metric_depth_ransac(depth, moge_depth, mask)
        np.save(out, metric_depth.astype(np.float32))
        print(f"  {name}: ppd med={np.median(metric_depth)*1000:.1f}mm")


def save_d3(device='cuda'):
    """Save D3 predictions using MoGe env."""
    MOGE_ROOT = PROJECT_ROOT / "MoGe"
    sys.path.insert(0, str(MOGE_ROOT))
    from moge.model import import_model_class_by_version
    import torchvision.transforms.functional as TF

    MoGeModel = import_model_class_by_version("v2")
    ckpt_path = str(PROJECT_ROOT / "output/training/exp_d3/checkpoint/00002500_ema.pt")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = ckpt.get('model_config', None)
    model = MoGeModel(**cfg) if cfg else MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    model.load_state_dict(ckpt['model'], strict=False)
    model = model.to(device).eval()

    for name in SAMPLES:
        out = CACHE_DIR / name / "d3_depth.npy"
        if out.exists():
            print(f"  {name}: d3 exists, skip")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(SYNTH_DIR / name / "image.png").convert('RGB')
        img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model.infer(img_tensor)
        d = output['depth'].cpu().numpy()
        if d.ndim > 2 and d.shape[0] == 1:
            d = d.squeeze(0)
        np.save(out, d.astype(np.float32))
        print(f"  {name}: d3 med={np.median(d)*1000:.1f}mm")


def plot(device='cuda'):
    """Create the comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.ndimage import zoom

    MOGE_ROOT = PROJECT_ROOT / "MoGe"
    sys.path.insert(0, str(MOGE_ROOT))
    from moge.utils.io import read_depth
    from moge.utils.vis import colorize_depth

    methods = ['d3', 'da3nested', 'mapanything', 'ppd']
    method_labels = ['D3 (joint)', 'DA3-Nested', 'MapAnything', 'PPD']

    # Check which methods have cached predictions
    available = []
    for m, label in zip(methods, method_labels):
        if all((CACHE_DIR / s / f"{m}_depth.npy").exists() for s in SAMPLES):
            available.append((m, label))
        else:
            missing = [s for s in SAMPLES if not (CACHE_DIR / s / f"{m}_depth.npy").exists()]
            print(f"  WARNING: {label} missing {len(missing)} samples, skipping")

    n_rows = len(SAMPLES)
    n_cols = 2 + len(available)  # Image + GT + methods
    col_labels = ['Image', 'GT Depth'] + [l for _, l in available]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.5),
                             squeeze=False)

    for si, sample_name in enumerate(SAMPLES):
        sample_dir = SYNTH_DIR / sample_name
        img = np.array(Image.open(sample_dir / "image.png").convert('RGB'))

        gt_depth_mm = read_depth(str(sample_dir / "depth.png"))
        gt_depth_m = gt_depth_mm * 0.001
        with open(sample_dir / "meta.json") as f:
            meta = json.load(f)
        depth_range = meta.get('depth_range_mm', [0, 0])
        h, w = gt_depth_m.shape[:2]

        # Image
        axes[si, 0].imshow(img)
        axes[si, 0].axis('off')
        if si == 0:
            axes[si, 0].set_title(col_labels[0], fontsize=9, fontweight='bold')

        # GT depth
        gt_vis = colorize_depth(gt_depth_m)
        axes[si, 1].imshow(gt_vis)
        axes[si, 1].axis('off')
        if si == 0:
            axes[si, 1].set_title(col_labels[1], fontsize=9, fontweight='bold')
        axes[si, 1].set_xlabel(f'{depth_range[0]:.1f}–{depth_range[1]:.1f}mm', fontsize=7)

        # Each method
        for mi, (method, label) in enumerate(available):
            col = 2 + mi
            pred = np.load(CACHE_DIR / sample_name / f"{method}_depth.npy")
            # Resize to GT if needed
            if pred.shape[:2] != (h, w):
                pred = zoom(pred, (h / pred.shape[0], w / pred.shape[1]), order=1)
            pred_vis = colorize_depth(pred)
            med_mm = np.nanmedian(pred) * 1000
            scale = np.nanmedian(pred) / np.nanmedian(gt_depth_m)
            axes[si, col].imshow(pred_vis)
            axes[si, col].axis('off')
            if si == 0:
                axes[si, col].set_title(label, fontsize=9, fontweight='bold')
            axes[si, col].set_xlabel(f'med={med_mm:.1f}mm ({scale:.1f}x)', fontsize=7)

        axes[si, 0].set_ylabel(sample_name[:20], fontsize=6, rotation=0,
                               labelpad=5, ha='right', va='center')

    fig.suptitle('DermDepth-Synth — Baseline Depth Comparison',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    pdf_path = PROJECT_ROOT / "output" / "figures" / "normal_comparison" / "synthetic_baseline_depth.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['da3nested', 'mapanything', 'ppd', 'd3'])
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    print(f"Samples: {SAMPLES}")

    if args.method == 'da3nested':
        save_da3nested(args.device)
    elif args.method == 'mapanything':
        save_mapanything(args.device)
    elif args.method == 'ppd':
        save_ppd(args.device)
    elif args.method == 'd3':
        save_d3(args.device)
    elif args.plot:
        plot(args.device)
    else:
        print("Specify --method or --plot")
