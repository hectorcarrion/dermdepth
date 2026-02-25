#!/usr/bin/env python3
"""Metric depth comparison grid: GT vs DermDepth (D3) vs baselines.

Two-phase workflow (each baseline needs its own conda env):

  Phase 1 — save predictions (run each in its conda env, all on GPU 3):
    CUDA_VISIBLE_DEVICES=3 conda run -n MoGe python -u code/visualization/depth_comparison_grid.py --save --method d3 --dataset all
    CUDA_VISIBLE_DEVICES=3 conda run -n da3 python -u code/visualization/depth_comparison_grid.py --save --method da3nested --dataset all
    CUDA_VISIBLE_DEVICES=3 conda run -n mapanything python -u code/visualization/depth_comparison_grid.py --save --method mapanything --dataset all
    CUDA_VISIBLE_DEVICES=3 conda run -n ppd python -u code/visualization/depth_comparison_grid.py --save --method ppd --dataset all

  Phase 2 — plot (in MoGe env):
    CUDA_VISIBLE_DEVICES=3 conda run -n MoGe python -u code/visualization/depth_comparison_grid.py --plot --dataset all
"""

import sys
import json
import random
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"

SYNTH_DIR = PROJECT_ROOT / "data" / "dermdepth_train" / "colab_gen" / "DermDepthSynth"
SKINL2_DIR = PROJECT_ROOT / "output" / "eval_data" / "skinl2"
WOUNDSDB_DIR = PROJECT_ROOT / "output" / "eval_data" / "woundsdb"
CACHE_DIR = PROJECT_ROOT / "output" / "figures" / "depth_comparison" / "_cache"
OUT_DIR = PROJECT_ROOT / "output" / "figures" / "depth_comparison"

METHODS = ['d3', 'da3nested', 'mapanything', 'ppd']
METHOD_LABELS = {
    'd3': 'DermDepth (Ours)',
    'da3nested': 'DA3-Nested',
    'mapanything': 'MapAnything',
    'ppd': 'PPD',
}


def get_samples(dataset):
    if dataset == 'synth':
        random.seed(42)
        all_s = sorted([d.name for d in SYNTH_DIR.iterdir()
                        if d.is_dir() and (d / "image.png").exists()])
        chosen = random.sample(all_s, min(12, len(all_s)))
        return chosen[6:]  # second batch of 6, different from normal comparison
    elif dataset == 'woundsdb':
        return sorted([d.name for d in WOUNDSDB_DIR.iterdir() if d.is_dir()])
    elif dataset == 'skinl2':
        return sorted([d.name for d in SKINL2_DIR.iterdir() if d.is_dir()])


def get_data_dir(dataset):
    return {'synth': SYNTH_DIR, 'woundsdb': WOUNDSDB_DIR, 'skinl2': SKINL2_DIR}[dataset]


def get_gt_depth_m(dataset, sample_dir):
    """Load GT depth in meters."""
    if dataset == 'synth':
        sys.path.insert(0, str(MOGE_ROOT))
        from moge.utils.io import read_depth
        return read_depth(str(sample_dir / "depth.png")) * 0.001  # mm → m
    else:
        gt = np.load(sample_dir / "gt_depth.npy")
        if dataset == 'skinl2':
            return gt * 0.001  # mm → m
        return gt  # woundsdb already in meters


# ========== Method save functions ==========

def save_d3(samples, dataset, device='cuda'):
    sys.path.insert(0, str(MOGE_ROOT))
    from moge.model import import_model_class_by_version
    import torchvision.transforms.functional as TF
    from PIL import Image

    MoGeModel = import_model_class_by_version("v2")
    ckpt_path = str(PROJECT_ROOT / "output/training/exp_d3/checkpoint/00002500_ema.pt")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = ckpt.get('model_config', None)
    model = MoGeModel(**cfg) if cfg else MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    model.load_state_dict(ckpt['model'], strict=False)
    model = model.to(device).eval()

    data_dir = get_data_dir(dataset)
    for name in samples:
        out = CACHE_DIR / dataset / name / "d3_depth.npy"
        if out.exists():
            print(f"  {name}: d3 exists, skip")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(data_dir / name / "image.png").convert('RGB')
        img_t = TF.to_tensor(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model.infer(img_t)
        d = output['depth'].cpu().numpy()
        if d.ndim > 2 and d.shape[0] == 1:
            d = d.squeeze(0)
        np.save(out, d.astype(np.float32))
        print(f"  {name}: d3 med={np.median(d)*1000:.1f}mm")


def save_da3nested(samples, dataset, device='cuda'):
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "Depth-Anything-3" / "src"))
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(device).eval()

    data_dir = get_data_dir(dataset)
    for name in samples:
        out = CACHE_DIR / dataset / name / "da3nested_depth.npy"
        if out.exists():
            print(f"  {name}: da3nested exists, skip")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        pred = model.inference([str(data_dir / name / "image.png")])
        d = pred.depth[0]
        if isinstance(d, torch.Tensor):
            d = d.cpu().numpy()
        np.save(out, d.astype(np.float32))
        print(f"  {name}: da3nested med={np.median(d)*1000:.1f}mm")


def save_mapanything(samples, dataset, device='cuda'):
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "map-anything"))
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    data_dir = get_data_dir(dataset)
    for name in samples:
        out = CACHE_DIR / dataset / name / "mapanything_depth.npy"
        if out.exists():
            print(f"  {name}: mapanything exists, skip")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        views = load_images([str(data_dir / name / "image.png")])
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype='bf16')
        d = preds[0]['depth_z'][0, :, :, 0].cpu().numpy()
        np.save(out, d.astype(np.float32))
        print(f"  {name}: mapanything med={np.median(d)*1000:.1f}mm")


def save_ppd(samples, dataset, device='cuda'):
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

    data_dir = get_data_dir(dataset)
    for name in samples:
        out = CACHE_DIR / dataset / name / "ppd_depth.npy"
        if out.exists():
            print(f"  {name}: ppd exists, skip")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(data_dir / name / "image.png"))
        depth, resize_image = ppd_model.infer_image(image)
        depth = depth.squeeze().cpu().numpy()
        moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        moge_image = torch.tensor(moge_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        moge_depth, mask, intrinsic = moge.infer(moge_image)
        moge_depth[~mask] = moge_depth[mask].max()
        metric_depth = recover_metric_depth_ransac(depth, moge_depth, mask)
        np.save(out, metric_depth.astype(np.float32))
        print(f"  {name}: ppd med={np.median(metric_depth)*1000:.1f}mm")


SAVE_FUNCS = {
    'd3': save_d3,
    'da3nested': save_da3nested,
    'mapanything': save_mapanything,
    'ppd': save_ppd,
}


# ========== Plotting ==========

def plot(dataset):
    """Create multi-page PDF with colorbars showing depth in mm."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.ndimage import zoom
    from PIL import Image

    samples = get_samples(dataset)
    data_dir = get_data_dir(dataset)

    # Check which methods have all predictions cached
    available = []
    for m in METHODS:
        missing = sum(1 for s in samples
                      if not (CACHE_DIR / dataset / s / f"{m}_depth.npy").exists())
        if missing == 0:
            available.append(m)
        else:
            print(f"  WARNING: {METHOD_LABELS[m]} missing {missing}/{len(samples)} samples, skipping")

    if not available:
        print(f"  No methods available for {dataset}, skipping")
        return

    n_cols = 2 + len(available)  # Input + GT + methods
    col_labels = ['Input', 'GT Depth'] + [METHOD_LABELS[m] for m in available]
    rows_per_page = 6
    cmap = 'turbo'

    pdf_path = OUT_DIR / f"{dataset}_depth_comparison.pdf"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(pdf_path)) as pdf:
        for page_start in range(0, len(samples), rows_per_page):
            page_samples = samples[page_start:page_start + rows_per_page]
            n_rows = len(page_samples)

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 3.0, n_rows * 3.2),
                                     squeeze=False)

            for si, sample_name in enumerate(page_samples):
                sample_dir = data_dir / sample_name
                if not (sample_dir / "image.png").exists():
                    for ax in axes[si]:
                        ax.axis('off')
                    continue

                img = np.array(Image.open(sample_dir / "image.png").convert('RGB'))
                gt_depth_m = get_gt_depth_m(dataset, sample_dir)
                h, w = gt_depth_m.shape[:2]

                gt_mask = None
                if (sample_dir / "gt_mask.npy").exists():
                    gt_mask = np.load(sample_dir / "gt_mask.npy")

                # GT in mm, masked
                gt_mm = gt_depth_m * 1000.0
                if gt_mask is not None:
                    gt_mm = np.where(gt_mask, gt_mm, np.nan)

                # Input image
                axes[si, 0].imshow(img)
                axes[si, 0].axis('off')
                if si == 0:
                    axes[si, 0].set_title(col_labels[0], fontsize=10, fontweight='bold')

                # GT depth with colorbar
                valid_gt = gt_mm[np.isfinite(gt_mm)]
                if len(valid_gt) > 0:
                    vmin_gt = float(np.percentile(valid_gt, 2))
                    vmax_gt = float(np.percentile(valid_gt, 98))
                else:
                    vmin_gt, vmax_gt = 0, 1
                im_gt = axes[si, 1].imshow(gt_mm, cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
                axes[si, 1].axis('off')
                if si == 0:
                    axes[si, 1].set_title(col_labels[1], fontsize=10, fontweight='bold')
                divider = make_axes_locatable(axes[si, 1])
                cax = divider.append_axes("bottom", size="5%", pad=0.05)
                cb = fig.colorbar(im_gt, cax=cax, orientation='horizontal')
                cb.ax.tick_params(labelsize=6)
                cb.set_label('mm', fontsize=6)

                # Each method
                for mi, method in enumerate(available):
                    col = 2 + mi
                    pred = np.load(CACHE_DIR / dataset / sample_name / f"{method}_depth.npy")
                    if pred.shape[:2] != (h, w):
                        pred = zoom(pred, (h / pred.shape[0], w / pred.shape[1]), order=1)
                    pred_mm = pred * 1000.0

                    valid_pred = pred_mm[np.isfinite(pred_mm)]
                    if len(valid_pred) > 0:
                        vmin_p = float(np.percentile(valid_pred, 2))
                        vmax_p = float(np.percentile(valid_pred, 98))
                    else:
                        vmin_p, vmax_p = 0, 1

                    im_pred = axes[si, col].imshow(pred_mm, cmap=cmap,
                                                    vmin=vmin_p, vmax=vmax_p)
                    axes[si, col].axis('off')
                    if si == 0:
                        axes[si, col].set_title(col_labels[col], fontsize=10, fontweight='bold')

                    divider = make_axes_locatable(axes[si, col])
                    cax = divider.append_axes("bottom", size="5%", pad=0.05)
                    cb = fig.colorbar(im_pred, cax=cax, orientation='horizontal')
                    cb.ax.tick_params(labelsize=6)
                    cb.set_label('mm', fontsize=6)

                # Row label
                axes[si, 0].set_ylabel(sample_name[:30], fontsize=5, rotation=0,
                                       labelpad=5, ha='right', va='center')

            page_num = page_start // rows_per_page + 1
            total_pages = (len(samples) + rows_per_page - 1) // rows_per_page
            ds_label = {'synth': 'Synthetic', 'woundsdb': 'WoundsDB', 'skinl2': 'SKINL2'}
            fig.suptitle(f'{ds_label[dataset]} — Metric Depth Comparison '
                         f'(p.{page_num}/{total_pages})',
                         fontsize=12, fontweight='bold', y=1.01)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Page {page_num}/{total_pages}")

    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='Save predictions')
    parser.add_argument('--plot', action='store_true', help='Create figure')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--dataset', default='all',
                        choices=['synth', 'woundsdb', 'skinl2', 'all'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    datasets = ['synth', 'woundsdb', 'skinl2'] if args.dataset == 'all' else [args.dataset]

    if args.save:
        if not args.method:
            print("Specify --method when using --save")
            sys.exit(1)
        for ds in datasets:
            samples = get_samples(ds)
            print(f"\n=== {ds} ({len(samples)} samples) — saving {args.method} ===")
            SAVE_FUNCS[args.method](samples, ds, args.device)
        print("\nDone saving.")

    elif args.plot:
        for ds in datasets:
            samples = get_samples(ds)
            print(f"\n=== {ds} ({len(samples)} samples) — plotting ===")
            plot(ds)
        print("\nAll plots done.")

    else:
        print("Specify --save or --plot")
