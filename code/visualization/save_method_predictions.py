#!/usr/bin/env python3
"""Save per-pixel depth (and normal) predictions for method comparison figures.

Runs inference for one or more methods on a fixed set of samples from SKINL2
and WoundsDB. Each method must be run from its own conda environment.

Usage:
    # MoGe-2 + DermDepth (MoGe env)
    python save_method_predictions.py --method moge2 --method dermdepth --device cuda

    # DA3-Nested (da3 env)
    python save_method_predictions.py --method da3nested --device cuda

    # MapAnything (mapanything env)
    python save_method_predictions.py --method mapanything --device cuda

    # PPD (ppd env)
    python save_method_predictions.py --method ppd --device cuda
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOGE_ROOT = PROJECT_ROOT / "MoGe"

SKINL2_DIR = PROJECT_ROOT / "output" / "eval_data" / "skinl2"
WOUNDSDB_DIR = PROJECT_ROOT / "output" / "eval_data" / "woundsdb"
OUT_DIR = PROJECT_ROOT / "output" / "figures" / "method_comparison" / "predictions"

# Fixed samples
SAMPLES = {
    'skinl2': [
        'v1_Melanoma_0203',
        'v2_Seborrheic Keratosis_0051',
        'v3_Hemangioma_0010',
    ],
    'woundsdb': [
        'case_22_day_1_scene_1',
        'case_39_day_1_scene_1',
    ],
}

DERMDEPTH_CKPT = str(PROJECT_ROOT / "output" / "training" / "exp_a" / "checkpoint" / "00001000_ema.pt")
DERMDEPTH_D1_CKPT = str(PROJECT_ROOT / "output" / "training" / "exp_d1" / "checkpoint" / "00003000_ema.pt")
DERMDEPTH_D2_CKPT = str(PROJECT_ROOT / "output" / "training" / "exp_d2" / "checkpoint" / "00003000_ema.pt")
DERMDEPTH_D3_CKPT = str(PROJECT_ROOT / "output" / "training" / "exp_d3" / "checkpoint" / "00003000_ema.pt")
MOGE2_HF_ID = "Ruicheng/moge-2-vitl-normal"


def get_sample_paths():
    """Return list of (sample_name, image_path, dataset) for all fixed samples."""
    paths = []
    for name in SAMPLES['skinl2']:
        img = SKINL2_DIR / name / 'image.png'
        if img.exists():
            paths.append((name, img, 'skinl2'))
        else:
            print(f"  WARNING: {img} not found, skipping")
    for name in SAMPLES['woundsdb']:
        img = WOUNDSDB_DIR / name / 'image.png'
        if img.exists():
            paths.append((name, img, 'woundsdb'))
        else:
            print(f"  WARNING: {img} not found, skipping")
    return paths


def resize_to_gt(pred, gt_shape):
    """Resize prediction to GT resolution."""
    from scipy.ndimage import zoom
    h, w = pred.shape[:2]
    th, tw = gt_shape[:2]
    if (h, w) == (th, tw):
        return pred
    if pred.ndim == 2:
        return zoom(pred, (th / h, tw / w), order=1)
    else:
        return zoom(pred, (th / h, tw / w, 1), order=1)


# ---- MoGe-2 / DermDepth ----

def load_moge(model_path, device='cuda'):
    sys.path.insert(0, str(MOGE_ROOT))
    from moge.model import import_model_class_by_version
    import os
    MoGeModel = import_model_class_by_version("v2")

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model_config = checkpoint.get('model_config', None)
        if model_config:
            model = MoGeModel(**model_config)
        else:
            model = MoGeModel.from_pretrained(MOGE2_HF_ID)
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model = MoGeModel.from_pretrained(model_path)

    return model.to(device).eval()


def infer_moge(model, image_path, device='cuda'):
    """Returns (depth, normal) — both numpy arrays."""
    import torchvision.transforms.functional as TF
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model.infer(img_tensor)

    depth = output['depth']
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
        if depth.ndim > 2 and depth.shape[0] == 1:
            depth = depth.squeeze(0)

    normal = output.get('normal', None)
    if normal is not None and isinstance(normal, torch.Tensor):
        normal = normal.cpu().numpy()
        if normal.ndim > 3 and normal.shape[0] == 1:
            normal = normal.squeeze(0)

    return depth.astype(np.float32), normal.astype(np.float32) if normal is not None else None


# ---- DA3-Nested ----

def load_da3nested(device='cuda'):
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "Depth-Anything-3" / "src"))
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    return model.to(device).eval()


def infer_da3nested(model, image_path, device='cuda'):
    pred = model.inference([str(image_path)])
    depth = pred.depth[0]  # (H, W) in meters
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    return depth.astype(np.float32), None


# ---- MapAnything ----

def load_mapanything(device='cuda'):
    sys.path.insert(0, str(PROJECT_ROOT / "baseline_methods" / "map-anything"))
    from mapanything.models import MapAnything
    return MapAnything.from_pretrained("facebook/map-anything").to(device)


def infer_mapanything(model, image_path, device='cuda'):
    from mapanything.utils.image import load_images
    views = load_images([str(image_path)])
    preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype='bf16')
    depth = preds[0]['depth_z'][0, :, :, 0].cpu().numpy()
    return depth.astype(np.float32), None


# ---- Pixel-Perfect Depth ----

PPD_DIR = PROJECT_ROOT / "baseline_methods" / "pixel-perfect-depth"


def load_ppd(device='cuda'):
    sys.path.insert(0, str(PPD_DIR))
    from ppd.models.ppd import PixelPerfectDepth
    from ppd.moge.model.v2 import MoGeModel

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

    return {'ppd': ppd_model, 'moge': moge, 'device': device}


def infer_ppd(model_dict, image_path, device='cuda'):
    import cv2
    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    ppd_model = model_dict['ppd']
    moge = model_dict['moge']
    dev = model_dict['device']

    image = cv2.imread(str(image_path))
    depth, resize_image = ppd_model.infer_image(image)
    depth = depth.squeeze().cpu().numpy()

    moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    moge_image = torch.tensor(moge_image / 255, dtype=torch.float32, device=dev).permute(2, 0, 1)
    moge_depth, mask, intrinsic = moge.infer(moge_image)
    moge_depth[~mask] = moge_depth[mask].max()

    metric_depth = recover_metric_depth_ransac(depth, moge_depth, mask)
    return metric_depth.astype(np.float32), None


# ---- Dispatch table ----

METHOD_TABLE = {
    'moge2': {
        'load': lambda device: load_moge(MOGE2_HF_ID, device),
        'infer': infer_moge,
        'has_normal': True,
    },
    'dermdepth': {
        'load': lambda device: load_moge(DERMDEPTH_CKPT, device),
        'infer': infer_moge,
        'has_normal': True,
    },
    'dermdepth_d1': {
        'load': lambda device: load_moge(DERMDEPTH_D1_CKPT, device),
        'infer': infer_moge,
        'has_normal': True,
    },
    'dermdepth_d2': {
        'load': lambda device: load_moge(DERMDEPTH_D2_CKPT, device),
        'infer': infer_moge,
        'has_normal': True,
    },
    'dermdepth_d3': {
        'load': lambda device: load_moge(DERMDEPTH_D3_CKPT, device),
        'infer': infer_moge,
        'has_normal': True,
    },
    'da3nested': {
        'load': load_da3nested,
        'infer': infer_da3nested,
        'has_normal': False,
    },
    'mapanything': {
        'load': load_mapanything,
        'infer': infer_mapanything,
        'has_normal': False,
    },
    'ppd': {
        'load': load_ppd,
        'infer': infer_ppd,
        'has_normal': False,
    },
}


def save_predictions(method_name, device='cuda'):
    """Run inference for a single method on all fixed samples."""
    entry = METHOD_TABLE[method_name]
    print(f"\nLoading {method_name}...")
    model = entry['load'](device)

    sample_paths = get_sample_paths()
    print(f"Running {method_name} on {len(sample_paths)} samples...")

    for sample_name, image_path, dataset in sample_paths:
        out_dir = OUT_DIR / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)

        depth_path = out_dir / f'{method_name}_depth.npy'
        normal_path = out_dir / f'{method_name}_normal.npy'

        if depth_path.exists():
            print(f"  {sample_name}: {method_name} already exists, skipping")
            continue

        print(f"  {sample_name}...", end="", flush=True)

        depth, normal = entry['infer'](model, image_path, device)

        # Get GT shape to resize
        if dataset == 'skinl2':
            gt_shape = np.load(SKINL2_DIR / sample_name / 'gt_depth.npy').shape
        else:
            gt_shape = np.load(WOUNDSDB_DIR / sample_name / 'gt_depth.npy').shape

        # Resize depth to GT resolution
        if depth.shape[:2] != gt_shape[:2]:
            depth = resize_to_gt(depth, gt_shape)

        np.save(depth_path, depth)
        print(f" depth {depth.shape}", end="")

        if normal is not None:
            if normal.shape[:2] != gt_shape[:2]:
                normal = resize_to_gt(normal, gt_shape)
            np.save(normal_path, normal)
            print(f" + normal {normal.shape}", end="")

        print(f" [{np.nanmedian(depth):.4f}m]")

    print(f"  {method_name} done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save method predictions for comparison figure')
    parser.add_argument('--method', action='append', required=True,
                        choices=list(METHOD_TABLE.keys()),
                        help='Method(s) to run (can specify multiple)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    for method in args.method:
        save_predictions(method, args.device)

    print("\nAll done. Predictions saved to:", OUT_DIR)
