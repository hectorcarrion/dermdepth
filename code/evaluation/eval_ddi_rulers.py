#!/usr/bin/env python3
"""Evaluate metric scale accuracy on DDI images using ruler segmentation masks.

Uses FEDD segmentation labels (class 3 = Ruler) with known physical ruler size
(6cm x 1.1cm = 6.6 cm²) to evaluate 3D surface area accuracy of depth methods.

For each DDI image with a fully-visible ruler:
1. Load ruler segmentation mask (256x256) and upscale to image resolution
2. For each method's depth prediction, compute 3D surface area of ruler region
3. Compare to GT area = 6.6 cm² (6cm x 1.1cm)
4. Report per-method and per-skin-tone results

Two-phase workflow (baselines need their own conda envs):

  Phase 1 — save predictions:
    CUDA_VISIBLE_DEVICES=3 conda run -n MoGe python -u code/evaluation/eval_ddi_rulers.py --save --method moge2
    CUDA_VISIBLE_DEVICES=3 conda run -n MoGe python -u code/evaluation/eval_ddi_rulers.py --save --method dermdepth
    CUDA_VISIBLE_DEVICES=3 conda run -n da3 python -u code/evaluation/eval_ddi_rulers.py --save --method da3nested
    CUDA_VISIBLE_DEVICES=3 conda run -n mapanything python -u code/evaluation/eval_ddi_rulers.py --save --method mapanything
    CUDA_VISIBLE_DEVICES=3 conda run -n ppd python -u code/evaluation/eval_ddi_rulers.py --save --method ppd

  Phase 2 — evaluate:
    conda run -n MoGe python code/evaluation/eval_ddi_rulers.py --evaluate
"""

import sys
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"
DDI_IMAGES = PROJECT_ROOT / "data" / "DDI" / "images"
DDI_MAP = PROJECT_ROOT / "data" / "DDI" / "map.csv"
LABELS_DIR = PROJECT_ROOT / "data" / "DDI" / "FEDD" / "ddi_labels"
CACHE_DIR = PROJECT_ROOT / "output" / "evaluation" / "ddi_rulers" / "_cache"
OUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "ddi_rulers"

sys.path.insert(0, str(MOGE_ROOT))

# Known ruler dimensions
RULER_LENGTH_CM = 6.0
RULER_WIDTH_CM = 1.1
GT_AREA_CM2 = RULER_LENGTH_CM * RULER_WIDTH_CM  # 6.6 cm²

# Manually excluded samples (visual inspection)
EXCLUDED = {'000186', '000559'}

# Methods and display names
METHODS = ['moge2', 'dermdepth', 'da3nested', 'mapanything', 'ppd']
METHOD_LABELS = {
    'moge2': 'MoGe-2 (base)',
    'dermdepth': 'DermDepth',
    'da3nested': 'DA3-Nested',
    'mapanything': 'MapAnything',
    'ppd': 'PPD',
}

CHECKPOINTS = {
    'moge2': str(MOGE_ROOT / "pretrained_moge2.pt"),
    'dermdepth': str(PROJECT_ROOT / "output" / "training" / "exp_a" / "checkpoint" / "00001000_ema.pt"),
}

# Dynamically-set checkpoint override (via --checkpoint CLI)
_checkpoint_override = None
_eval_methods_filter = None


# ==================== Data loading ====================

def load_ddi_metadata():
    """Load DDI map.csv into dict keyed by DDI_file."""
    meta = {}
    with open(DDI_MAP) as f:
        for row in csv.DictReader(f):
            meta[row['DDI_file']] = row
    return meta


def collect_label_files():
    """Collect all unique label .npy files across all FEDD splits."""
    label_files = {}
    for f in LABELS_DIR.rglob('*.npy'):
        label_files.setdefault(f.name, f)
    return label_files


def get_ruler_samples():
    """Get filtered list of DDI samples with fully-visible rulers.

    Filters:
    1. Ruler mask (class 3) must exist
    2. Ruler must not touch image border (not cropped)
    3. Bounding box aspect ratio between 3:1 and 10:1 (full ruler shape)
    4. Not in manual exclusion list
    """
    label_files = collect_label_files()
    meta = load_ddi_metadata()
    samples = []

    for name, path in sorted(label_files.items()):
        stem = name.replace('.npy', '')
        if stem in EXCLUDED:
            continue

        mask = np.load(path)
        if 3 not in mask:
            continue

        ruler = (mask == 3)

        # Filter: ruler must not touch border
        if (ruler[0, :].any() or ruler[-1, :].any() or
                ruler[:, 0].any() or ruler[:, -1].any()):
            continue

        # Filter: aspect ratio of bounding box
        rows = np.where(ruler.any(axis=1))[0]
        cols = np.where(ruler.any(axis=0))[0]
        bb_h = rows[-1] - rows[0] + 1
        bb_w = cols[-1] - cols[0] + 1
        ar = max(bb_h, bb_w) / max(min(bb_h, bb_w), 1)
        if ar < 3.0 or ar > 10.0:
            continue

        filename = stem + '.png'
        img_path = DDI_IMAGES / filename
        if not img_path.exists():
            continue

        m = meta.get(filename, {})
        samples.append({
            'stem': stem,
            'filename': filename,
            'label_path': str(path),
            'skin_tone': m.get('skin_tone', '?'),
            'disease': m.get('disease', '?'),
            'malignant': m.get('malignant', '?'),
            'ruler_pixels_256': int(ruler.sum()),
            'aspect_ratio': float(ar),
        })

    return samples


def load_ruler_mask(label_path, img_h, img_w):
    """Load ruler mask and upscale to image resolution."""
    from scipy.ndimage import zoom
    mask_256 = np.load(label_path)
    ruler_256 = (mask_256 == 3).astype(np.uint8)
    scale_h = img_h / ruler_256.shape[0]
    scale_w = img_w / ruler_256.shape[1]
    ruler_full = zoom(ruler_256, (scale_h, scale_w), order=0)  # nearest-neighbor
    return ruler_full.astype(bool)


# ==================== Surface area computation ====================

def estimate_intrinsics(height, width, fov_deg=60.0):
    """Estimate pinhole intrinsics."""
    fx = fy = width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def compute_surface_area(depth, mask, intrinsics):
    """Compute 3D surface area within mask using depth + intrinsics.

    Returns area in the square of the depth's units (m² if depth in meters).
    """
    h, w = depth.shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    X = (jj - cx) * depth / fx
    Y = (ii - cy) * depth / fy
    Z = depth

    # Tangent vectors (forward differences)
    dXdx = np.zeros_like(X); dYdx = np.zeros_like(Y); dZdx = np.zeros_like(Z)
    dXdx[:, :-1] = X[:, 1:] - X[:, :-1]
    dYdx[:, :-1] = Y[:, 1:] - Y[:, :-1]
    dZdx[:, :-1] = Z[:, 1:] - Z[:, :-1]

    dXdy = np.zeros_like(X); dYdy = np.zeros_like(Y); dZdy = np.zeros_like(Z)
    dXdy[:-1, :] = X[1:, :] - X[:-1, :]
    dYdy[:-1, :] = Y[1:, :] - Y[:-1, :]
    dZdy[:-1, :] = Z[1:, :] - Z[:-1, :]

    # Cross product magnitude = area element
    nx = dYdx * dZdy - dZdx * dYdy
    ny = dZdx * dXdy - dXdx * dZdy
    nz = dXdx * dYdy - dYdx * dXdy
    area_element = np.sqrt(nx**2 + ny**2 + nz**2)

    # Valid: within mask AND finite depth AND neighbors exist
    valid = mask & np.isfinite(depth) & (depth > 0)
    valid[:-1, :] &= np.isfinite(depth[1:, :])
    valid[:, :-1] &= np.isfinite(depth[:, 1:])

    total_area = float(np.sum(area_element[valid]))
    n_pixels = int(valid.sum())
    return total_area, n_pixels


# ==================== Save predictions ====================

def save_moge(method, samples, device='cuda', cache_name=None):
    """Save MoGe-2 or DermDepth depth predictions."""
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from moge.model import import_model_class_by_version

    checkpoint = _checkpoint_override or CHECKPOINTS[method]
    print(f"Loading {cache_name or method} from {checkpoint}")
    MoGeModel = import_model_class_by_version("v2")
    model = MoGeModel.from_pretrained(checkpoint).to(device).eval()

    cache = CACHE_DIR / (cache_name or method)
    cache.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(samples):
        out_path = cache / f"{s['stem']}_depth.npy"
        if out_path.exists():
            continue
        img = Image.open(DDI_IMAGES / s['filename']).convert('RGB')
        img_t = TF.to_tensor(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model.infer(img_t)
        d = out['depth'].cpu().numpy()
        if d.ndim > 2 and d.shape[0] == 1:
            d = d.squeeze(0)
        np.save(out_path, d.astype(np.float32))
        if i < 3 or i % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] {s['stem']}: {d.shape}, med={np.median(d)*1000:.1f}mm")

    print(f"Done: {cache}")


def save_da3nested(samples, device='cuda'):
    """Save DA3-Nested predictions."""
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "Depth-Anything-3" / "src"))
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(device).eval()

    cache = CACHE_DIR / "da3nested"
    cache.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(samples):
        out_path = cache / f"{s['stem']}_depth.npy"
        if out_path.exists():
            continue
        pred = model.inference([str(DDI_IMAGES / s['filename'])])
        d = pred.depth[0]
        if isinstance(d, torch.Tensor):
            d = d.cpu().numpy()
        np.save(out_path, d.astype(np.float32))
        if i < 3 or i % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] {s['stem']}: med={np.median(d)*1000:.1f}mm")

    print(f"Done: {cache}")


def save_mapanything(samples, device='cuda'):
    """Save MapAnything predictions."""
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "map-anything"))
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    cache = CACHE_DIR / "mapanything"
    cache.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(samples):
        out_path = cache / f"{s['stem']}_depth.npy"
        if out_path.exists():
            continue
        views = load_images([str(DDI_IMAGES / s['filename'])])
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype='bf16')
        d = preds[0]['depth_z'][0, :, :, 0].cpu().numpy()
        np.save(out_path, d.astype(np.float32))
        if i < 3 or i % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] {s['stem']}: med={np.median(d)*1000:.1f}mm")

    print(f"Done: {cache}")


def save_ppd(samples, device='cuda'):
    """Save Pixel-Perfect Depth predictions."""
    import torch, cv2
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

    cache = CACHE_DIR / "ppd"
    cache.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(samples):
        out_path = cache / f"{s['stem']}_depth.npy"
        if out_path.exists():
            continue
        image = cv2.imread(str(DDI_IMAGES / s['filename']))
        depth, resize_image = ppd_model.infer_image(image)
        depth = depth.squeeze().cpu().numpy()
        moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        moge_image = torch.tensor(moge_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        moge_depth, mask, intrinsic = moge.infer(moge_image)
        moge_depth[~mask] = moge_depth[mask].max()
        metric_depth = recover_metric_depth_ransac(depth, moge_depth, mask)
        np.save(out_path, metric_depth.astype(np.float32))
        if i < 3 or i % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] {s['stem']}: med={np.median(metric_depth)*1000:.1f}mm")

    print(f"Done: {cache}")


SAVE_FUNCS = {
    'moge2': lambda samps, dev: save_moge('moge2', samps, dev),
    'dermdepth': lambda samps, dev: save_moge('dermdepth', samps, dev),
    'da3nested': lambda samps, dev: save_da3nested(samps, dev),
    'mapanything': lambda samps, dev: save_mapanything(samps, dev),
    'ppd': lambda samps, dev: save_ppd(samps, dev),
}


# ==================== Evaluate ====================

def evaluate(fov_deg=60.0):
    """Evaluate all methods on DDI ruler samples."""
    from PIL import Image
    from scipy.ndimage import zoom as scipy_zoom

    samples = get_ruler_samples()
    print(f"DDI Ruler Area Evaluation")
    print(f"  Samples: {len(samples)} (fully-visible rulers)")
    print(f"  GT ruler area: {GT_AREA_CM2:.1f} cm² ({RULER_LENGTH_CM}cm x {RULER_WIDTH_CM}cm)")
    print(f"  Assumed FoV: {fov_deg}°")

    # Find available methods (check hardcoded list + any extra cache dirs)
    available = []
    for m in METHODS:
        cache = CACHE_DIR / m
        if cache.exists() and len(list(cache.glob("*.npy"))) > 0:
            available.append(m)
    # Also detect custom methods from cache subdirectories
    if CACHE_DIR.exists():
        for d in sorted(CACHE_DIR.iterdir()):
            if d.is_dir() and d.name not in available and len(list(d.glob("*.npy"))) > 0:
                available.append(d.name)
    # Filter to --methods if specified
    if _eval_methods_filter:
        available = [m for m in available if m in _eval_methods_filter]
    print(f"  Methods: {', '.join(available)}")

    if not available:
        print("\nNo predictions cached. Run --save --method X first.")
        return

    # Evaluate
    results_by_method = defaultdict(lambda: defaultdict(list))
    per_sample = []

    for si, s in enumerate(samples):
        img = Image.open(DDI_IMAGES / s['filename']).convert('RGB')
        img_w, img_h = img.size
        ruler_mask = load_ruler_mask(s['label_path'], img_h, img_w)
        intrinsics = estimate_intrinsics(img_h, img_w, fov_deg=fov_deg)

        sample_result = {
            'stem': s['stem'], 'filename': s['filename'],
            'skin_tone': s['skin_tone'], 'disease': s['disease'],
            'image_size': [img_w, img_h],
            'ruler_pixels': int(ruler_mask.sum()),
            'methods': {},
        }

        for method in available:
            depth_path = CACHE_DIR / method / f"{s['stem']}_depth.npy"
            if not depth_path.exists():
                continue

            depth = np.load(depth_path)

            # Resize depth to image resolution if needed
            if depth.shape[:2] != (img_h, img_w):
                depth = scipy_zoom(depth, (img_h / depth.shape[0], img_w / depth.shape[1]), order=1)

            area_m2, n_valid = compute_surface_area(depth, ruler_mask, intrinsics)
            area_cm2 = area_m2 * 1e4  # m² → cm²
            ratio = area_cm2 / GT_AREA_CM2

            results_by_method[method]['all'].append(ratio)
            results_by_method[method][f'tone_{s["skin_tone"]}'].append(ratio)

            sample_result['methods'][method] = {
                'area_cm2': float(area_cm2),
                'ratio': float(ratio),
                'n_valid_pixels': n_valid,
            }

        per_sample.append(sample_result)

        if si < 3 or si % 10 == 0:
            areas_str = ", ".join(
                f"{m}={sample_result['methods'][m]['area_cm2']:.1f}"
                for m in available if m in sample_result['methods']
            )
            print(f"  [{si+1}/{len(samples)}] {s['stem']} (tone {s['skin_tone']}): {areas_str} cm²")

    # ==================== Print results ====================
    print(f"\n{'='*95}")
    print(f"DDI RULER AREA EVALUATION — GT = {GT_AREA_CM2:.1f} cm²")
    print(f"{'='*95}")

    # Area ratio table (method x skin tone)
    print(f"\n{'Method':<20} {'All':>10} {'FP I-II':>10} {'FP III-IV':>10} {'FP V-VI':>10} {'n':>5}")
    print("-" * 67)

    summary = {}
    for method in available:
        all_ratios = results_by_method[method]['all']
        if not all_ratios:
            continue

        row = f"{METHOD_LABELS.get(method, method):<20}"
        method_summary = {}

        for key, label in [('all', 'All'), ('tone_12', 'FP I-II'), ('tone_34', 'FP III-IV'), ('tone_56', 'FP V-VI')]:
            ratios = results_by_method[method][key]
            if ratios:
                med = float(np.median(ratios))
                row += f" {med:>9.2f}x"
                method_summary[key] = {
                    'median_ratio': med,
                    'mean_ratio': float(np.mean(ratios)),
                    'std_ratio': float(np.std(ratios)),
                    'median_area_cm2': float(np.median(np.array(ratios) * GT_AREA_CM2)),
                    'n': len(ratios),
                }
            else:
                row += f" {'N/A':>10}"

        row += f" {len(all_ratios):>5}"
        print(row)
        summary[method] = method_summary

    print(f"\n  Ratio = predicted_ruler_area / {GT_AREA_CM2:.1f} cm² (target = 1.0)")
    print(f"  Values shown are medians")

    # Absolute area table
    print(f"\n{'Method':<20} {'Med Area':>10} {'Mean Area':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 62)
    for method in available:
        all_ratios = results_by_method[method]['all']
        if not all_ratios:
            continue
        areas = [r * GT_AREA_CM2 for r in all_ratios]
        errors = [a - GT_AREA_CM2 for a in areas]
        abs_errors = [abs(e) for e in errors]
        rmse = float(np.sqrt(np.mean([e**2 for e in errors])))
        print(f"{METHOD_LABELS.get(method, method):<20} {np.median(areas):>9.1f}cm² "
              f"{np.mean(areas):>9.1f}cm² {np.mean(abs_errors):>9.1f}cm² {rmse:>9.1f}cm²")

    # Fairness gap
    print(f"\n{'Method':<20} {'Max-Min Gap':>12} {'Max Tone':>10} {'Min Tone':>10}")
    print("-" * 54)
    for method in available:
        tone_medians = {}
        for tone in ['12', '34', '56']:
            ratios = results_by_method[method][f'tone_{tone}']
            if ratios:
                tone_medians[tone] = float(np.median(ratios))
        if len(tone_medians) >= 2:
            max_t = max(tone_medians, key=tone_medians.get)
            min_t = min(tone_medians, key=tone_medians.get)
            gap = tone_medians[max_t] - tone_medians[min_t]
            print(f"{METHOD_LABELS.get(method, method):<20} {gap:>11.2f}x {f'FP {max_t}':>10} {f'FP {min_t}':>10}")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        'gt_area_cm2': GT_AREA_CM2,
        'ruler_dimensions_cm': [RULER_LENGTH_CM, RULER_WIDTH_CM],
        'fov_deg': fov_deg,
        'n_samples': len(samples),
        'excluded': list(EXCLUDED),
        'summary': summary,
        'per_sample': per_sample,
    }
    out_path = OUT_DIR / "ddi_ruler_results.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    _make_plots(results_by_method, summary, available)


def _make_plots(results_by_method, summary, available):
    """Generate analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Area ratio box plot by method
    data, labels = [], []
    for m in available:
        ratios = results_by_method[m]['all']
        if ratios:
            data.append(ratios)
            labels.append(METHOD_LABELS.get(m, m))
    if data:
        bp = axes[0].boxplot(data, labels=labels, patch_artist=True)
        colors = ['steelblue', '#e94560', '#2ecc71', '#f39c12', '#9b59b6']
        for patch, c in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        axes[0].axhline(1.0, color='green', linestyle='--', linewidth=2)
        axes[0].set_ylabel(f'Area Ratio (pred / {GT_AREA_CM2:.1f} cm²)')
        axes[0].set_title('Ruler Area Accuracy')
        axes[0].tick_params(axis='x', rotation=25)

    # 2. Per-skin-tone grouped bars
    tones = ['12', '34', '56']
    tone_labels = ['FP I-II', 'FP III-IV', 'FP V-VI']
    x = np.arange(len(tones))
    width = 0.8 / max(len(available), 1)
    colors = ['steelblue', '#e94560', '#2ecc71', '#f39c12', '#9b59b6']
    for idx, method in enumerate(available):
        means = []
        for t in tones:
            ratios = results_by_method[method][f'tone_{t}']
            means.append(float(np.median(ratios)) if ratios else 0)
        short = METHOD_LABELS.get(method, method)[:12]
        axes[1].bar(x + idx * width, means, width, label=short, color=colors[idx % len(colors)], alpha=0.8)
    axes[1].axhline(1.0, color='green', linestyle='--', linewidth=2)
    axes[1].set_xticks(x + width * (len(available) - 1) / 2)
    axes[1].set_xticklabels(tone_labels)
    axes[1].set_ylabel('Median Area Ratio')
    axes[1].set_title('Scale by Skin Tone')
    axes[1].legend(fontsize=7)

    # 3. DermDepth vs MoGe-2 scatter
    if 'dermdepth' in results_by_method and 'moge2' in results_by_method:
        dd = results_by_method['dermdepth']['all']
        base = results_by_method['moge2']['all']
        if dd and base and len(dd) == len(base):
            axes[2].scatter(base, dd, alpha=0.6, s=40, c='#e94560')
            lim = max(max(base), max(dd)) * 1.1
            axes[2].plot([0, lim], [0, lim], 'k--', alpha=0.3)
            axes[2].axhline(1.0, color='green', linestyle='--', alpha=0.5)
            axes[2].axvline(1.0, color='green', linestyle='--', alpha=0.5)
            axes[2].set_xlabel('MoGe-2 Area Ratio')
            axes[2].set_ylabel('DermDepth Area Ratio')
            axes[2].set_title('DermDepth vs MoGe-2')

    plt.tight_layout()
    plot_path = OUT_DIR / "ddi_ruler_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDI Ruler Area Evaluation')
    parser.add_argument('--save', action='store_true', help='Save depth predictions')
    parser.add_argument('--method', type=str, help='Method for --save (or custom name with --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path (uses MoGe loader). Saves to cache under --method name.')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to include in --evaluate (default: auto-detect from cache)')
    parser.add_argument('--fov', type=float, default=60.0, help='Assumed FoV (degrees)')
    parser.add_argument('--device', default='cuda', help='Device for inference')
    args = parser.parse_args()

    # Apply global overrides from CLI
    if args.checkpoint:
        _checkpoint_override = args.checkpoint
    if args.methods:
        _eval_methods_filter = set(args.methods)

    if args.save:
        if not args.method:
            print("Specify --method when using --save")
            sys.exit(1)
        if _checkpoint_override:
            # Custom checkpoint: use MoGe loader, save under --method name
            samples = get_ruler_samples()
            print(f"Saving MoGe predictions ({args.method}) from {_checkpoint_override}")
            print(f"  for {len(samples)} ruler samples")
            save_moge('dermdepth', samples, args.device, cache_name=args.method)
        elif args.method in SAVE_FUNCS:
            samples = get_ruler_samples()
            print(f"Saving {args.method} predictions for {len(samples)} ruler samples")
            SAVE_FUNCS[args.method](samples, args.device)
        else:
            print(f"Unknown method: {args.method}. Use --checkpoint for custom models.")
            sys.exit(1)
    elif args.evaluate:
        evaluate(fov_deg=args.fov)
    else:
        print("Usage:")
        print("  --save --method X   Save depth predictions")
        print("  --evaluate          Run evaluation on cached predictions")
        print(f"\nAvailable methods: {', '.join(METHODS)}")
        samples = get_ruler_samples()
        print(f"Ruler samples: {len(samples)}")
