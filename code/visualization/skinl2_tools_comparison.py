#!/usr/bin/env python3
"""
Compare our Gaussian σ=15 denoising against the SKINL2 third-party tools
(morphological pipeline from Lourenço 2022 + Faria 2021).

Runs on the same 15 cases (5 per version) used in prior analysis.
Produces per-version figures with columns:
  RGB | Raw | Gaussian σ=15 | Morph Pipeline | Normals (Gauss) | Normals (Morph)
Plus cross-section comparison and quantitative noise metrics.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "data" / "SKINL2" / "SKINL2_tools"))
from skinl2_depth_enhance import enhance_depth_map, METHODS

DATA_ROOT = PROJECT_ROOT / "data" / "SKINL2"
OUTPUT_DIR = PROJECT_ROOT / "output" / "verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

APPROX_FOV_MM = 40.0


def compute_normals_physical(depth_mm, fov_mm=APPROX_FOV_MM):
    """Compute normals with proper physical pixel spacing."""
    h, w = depth_mm.shape
    pixel_spacing = fov_mm / w
    dz_dy, dz_dx = np.gradient(depth_mm)
    dz_dx_phys = dz_dx / pixel_spacing
    dz_dy_phys = dz_dy / pixel_spacing
    normals = np.stack([-dz_dx_phys, -dz_dy_phys, np.ones_like(depth_mm)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(norm, 1e-8)


def normal_to_rgb(normals):
    return np.clip((normals + 1.0) / 2.0, 0, 1)


def noise_metrics(depth):
    """Compute noise-related metrics for a depth map."""
    # High-frequency energy: Laplacian magnitude
    from scipy.ndimage import laplace
    lap = laplace(depth)
    hf_energy = np.mean(lap ** 2)

    # Gradient smoothness: mean absolute second derivative
    dx = np.diff(depth, axis=1)
    dy = np.diff(depth, axis=0)
    ddx = np.diff(dx, axis=1)
    ddy = np.diff(dy, axis=0)
    grad_smooth = np.mean(np.abs(ddx)) + np.mean(np.abs(ddy))

    # Local variance in small patches
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(depth, size=5)
    local_var = uniform_filter((depth - local_mean) ** 2, size=5)
    mean_local_var = np.mean(local_var)

    return {
        'hf_energy': hf_energy,
        'grad_smooth': grad_smooth,
        'local_var': mean_local_var,
    }


# ── Dataset discovery ──

def discover_v1():
    cases = []
    cv_root = DATA_ROOT / 'SKINL2_v1' / 'Central View'
    dm_root = DATA_ROOT / 'SKINL2_v1' / 'DepthMap'
    for cat in sorted(os.listdir(cv_root)):
        cat_cv = cv_root / cat
        cat_dm = dm_root / cat
        if not cat_cv.is_dir() or not cat_dm.is_dir():
            continue
        for sid in sorted(os.listdir(cat_cv)):
            cv_files = list((cat_cv / sid).glob('*.png'))
            dm_files = list((cat_dm / sid).glob('*.tiff'))
            if cv_files and dm_files:
                cases.append(('v1', cat, sid, cv_files[0], dm_files[0]))
    return cases


def discover_v2v3(version):
    cases = []
    root = DATA_ROOT / f'SKINL2_{version}'
    for case_id in sorted(os.listdir(root)):
        case_dir = root / case_id
        if not case_dir.is_dir():
            continue
        for cat in os.listdir(case_dir):
            cv_dir = case_dir / cat / 'Light Field' / 'Central View'
            dm_dir = case_dir / cat / 'Light Field' / 'Depth Map'
            if cv_dir.is_dir() and dm_dir.is_dir():
                cv_files = list(cv_dir.glob('*TotalFocus*.png'))
                dm_files = list(dm_dir.glob('*DepthMap.tiff'))
                if cv_files and dm_files:
                    cases.append((version, cat, case_id, cv_files[0], dm_files[0]))
    return cases


def select_diverse(cases_list, n=5):
    by_cat = {}
    for c in cases_list:
        by_cat.setdefault(c[1], []).append(c)
    selected = []
    for cat in sorted(by_cat.keys()):
        if len(selected) >= n:
            break
        selected.append(random.choice(by_cat[cat]))
    remaining = [c for c in cases_list if c not in selected]
    while len(selected) < n and remaining:
        choice = random.choice(remaining)
        selected.append(choice)
        remaining.remove(choice)
    return selected[:n]


def main():
    v1_cases = discover_v1()
    v2_cases = discover_v2v3('v2')
    v3_cases = discover_v2v3('v3')

    all_selected = (select_diverse(v1_cases, 5) +
                    select_diverse(v2_cases, 5) +
                    select_diverse(v3_cases, 5))

    print(f"Selected {len(all_selected)} cases across v1/v2/v3\n")

    pages = [('v1', all_selected[:5]),
             ('v2', all_selected[5:10]),
             ('v3', all_selected[10:15])]

    all_noise = []  # collect noise metrics for summary table

    for page_name, page_cases in pages:
        n = len(page_cases)
        # 7 cols: RGB | Raw depth | Gaussian | Morph | Normals(Gauss) | Normals(Morph) | Cross-section
        fig, axes = plt.subplots(n, 7, figsize=(42, 6 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, (ver, cat, sid, cv_path, dm_path) in enumerate(page_cases):
            label = f"{ver}/{cat}/{sid}"
            print(f"Processing {label}...")

            cv_img = np.array(Image.open(cv_path).convert('RGB'))
            depth_raw = np.array(Image.open(dm_path), dtype=np.float32)

            # v2: crop black borders
            if ver == 'v2':
                pct = 0.03
                h, w = cv_img.shape[:2]
                t, b, l, r = int(h*pct), h-int(h*pct), int(w*pct), w-int(w*pct)
                cv_img = cv_img[t:b, l:r]
                dh, dw = depth_raw.shape
                dt = int(t * dh / h)
                db = int(b * dh / h)
                dl = int(l * dw / w)
                dr = int(r * dw / w)
                depth_raw = depth_raw[dt:db, dl:dr]

            # Resize central view to depth resolution
            cv_small = np.array(Image.fromarray(cv_img).resize(
                (depth_raw.shape[1], depth_raw.shape[0]), Image.LANCZOS))

            # Method 1: Our Gaussian σ=15
            depth_gauss = np.abs(gaussian_filter(depth_raw, sigma=15))

            # Method 2: SKINL2 Tools morphological pipeline
            print(f"  Running morphological pipeline...")
            result = enhance_depth_map(
                depth_raw.copy(), cv_small.copy(),
                method='morphological', verbose=False)
            depth_morph = np.abs(result['enhanced'])

            # Normals
            normals_gauss = compute_normals_physical(depth_gauss)
            normals_morph = compute_normals_physical(depth_morph)

            # Noise metrics
            raw_abs = np.abs(depth_raw)
            nm_raw = noise_metrics(raw_abs)
            nm_gauss = noise_metrics(depth_gauss)
            nm_morph = noise_metrics(depth_morph)
            all_noise.append({
                'version': ver, 'category': cat, 'id': sid,
                'raw': nm_raw, 'gaussian': nm_gauss, 'morph': nm_morph
            })

            # Depth normalization for display
            vmin = min(depth_gauss.min(), depth_morph.min(), raw_abs.min())
            vmax = max(depth_gauss.max(), depth_morph.max(), raw_abs.max())

            # Col 0: RGB
            axes[row, 0].imshow(cv_small)
            axes[row, 0].set_title(f'{cat}\n({ver}, {sid})', fontsize=10, fontweight='bold')
            axes[row, 0].axis('off')

            # Col 1: Raw depth
            im = axes[row, 1].imshow(raw_abs, cmap='jet', vmin=vmin, vmax=vmax)
            span = raw_abs.max() - raw_abs.min()
            axes[row, 1].set_title(f'Raw Depth\nspan={span:.1f}mm', fontsize=10)
            axes[row, 1].axis('off')

            # Col 2: Gaussian σ=15
            axes[row, 2].imshow(depth_gauss, cmap='jet', vmin=vmin, vmax=vmax)
            axes[row, 2].set_title(f'Gaussian σ=15\nHF={nm_gauss["hf_energy"]:.4f}', fontsize=10)
            axes[row, 2].axis('off')

            # Col 3: Morphological pipeline
            axes[row, 3].imshow(depth_morph, cmap='jet', vmin=vmin, vmax=vmax)
            axes[row, 3].set_title(f'Morph Pipeline\nHF={nm_morph["hf_energy"]:.4f}', fontsize=10)
            axes[row, 3].axis('off')

            # Col 4: Normals (Gaussian)
            axes[row, 4].imshow(normal_to_rgb(normals_gauss))
            axes[row, 4].set_title('Normals (Gaussian)', fontsize=10)
            axes[row, 4].axis('off')

            # Col 5: Normals (Morph)
            axes[row, 5].imshow(normal_to_rgb(normals_morph))
            axes[row, 5].set_title('Normals (Morph)', fontsize=10)
            axes[row, 5].axis('off')

            # Col 6: Cross-section comparison
            mid = depth_raw.shape[0] // 2
            axes[row, 6].plot(raw_abs[mid], 'gray', alpha=0.3, lw=0.5, label='Raw')
            axes[row, 6].plot(depth_gauss[mid], 'b-', lw=1.0, label='Gauss σ=15')
            axes[row, 6].plot(depth_morph[mid], 'r-', lw=1.0, label='Morph')
            axes[row, 6].set_title('Cross-section (mid row)', fontsize=10)
            axes[row, 6].legend(fontsize=7)
            axes[row, 6].set_ylabel('Depth (mm)', fontsize=8)
            axes[row, 6].tick_params(labelsize=7)
            axes[row, 6].grid(alpha=0.2)

            # Noise reduction stats
            raw_hf = nm_raw['hf_energy']
            g_reduction = (1 - nm_gauss['hf_energy'] / raw_hf) * 100 if raw_hf > 0 else 0
            m_reduction = (1 - nm_morph['hf_energy'] / raw_hf) * 100 if raw_hf > 0 else 0
            print(f"  HF noise reduction: Gauss={g_reduction:.1f}%, Morph={m_reduction:.1f}%")

        plt.suptitle(
            f'SKINL2 {page_name.upper()} — Gaussian σ=15 vs Morphological Pipeline\n'
            'RGB | Raw | Gaussian σ=15 | Morph Pipeline | Normals (Gauss) | Normals (Morph) | Cross-section',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_num = {'v1': 46, 'v2': 47, 'v3': 48}[page_name]
        path = OUTPUT_DIR / f'fig{fig_num}_skinl2_{page_name}_gauss_vs_morph.png'
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"Saved {path.name}\n")

    # ── Summary table ──
    print(f"\n{'='*90}")
    print("Noise Metrics Summary: Raw vs Gaussian σ=15 vs Morphological Pipeline")
    print(f"{'='*90}")
    print(f"{'Ver':<4} {'Category':<22} {'ID':<6} {'Metric':<12} {'Raw':<12} {'Gaussian':<12} {'Morph':<12} {'G %red':<8} {'M %red':<8}")
    print("-" * 90)
    for entry in all_noise:
        for metric_name in ['hf_energy', 'local_var']:
            r = entry['raw'][metric_name]
            g = entry['gaussian'][metric_name]
            m = entry['morph'][metric_name]
            g_red = (1 - g / r) * 100 if r > 0 else 0
            m_red = (1 - m / r) * 100 if r > 0 else 0
            print(f"{entry['version']:<4} {entry['category']:<22} {entry['id']:<6} "
                  f"{metric_name:<12} {r:<12.6f} {g:<12.6f} {m:<12.6f} "
                  f"{g_red:<8.1f} {m_red:<8.1f}")

    # Averages
    print(f"\n{'='*90}")
    print("Averages by version:")
    for ver in ['v1', 'v2', 'v3']:
        ver_entries = [e for e in all_noise if e['version'] == ver]
        if not ver_entries:
            continue
        for metric_name in ['hf_energy', 'local_var']:
            r_avg = np.mean([e['raw'][metric_name] for e in ver_entries])
            g_avg = np.mean([e['gaussian'][metric_name] for e in ver_entries])
            m_avg = np.mean([e['morph'][metric_name] for e in ver_entries])
            g_red = (1 - g_avg / r_avg) * 100 if r_avg > 0 else 0
            m_red = (1 - m_avg / r_avg) * 100 if r_avg > 0 else 0
            print(f"  {ver} {metric_name:<12}: Raw={r_avg:.6f}  Gauss={g_avg:.6f} ({g_red:.1f}% red)  "
                  f"Morph={m_avg:.6f} ({m_red:.1f}% red)")

    print(f"\nOverall:")
    for metric_name in ['hf_energy', 'local_var']:
        r_avg = np.mean([e['raw'][metric_name] for e in all_noise])
        g_avg = np.mean([e['gaussian'][metric_name] for e in all_noise])
        m_avg = np.mean([e['morph'][metric_name] for e in all_noise])
        g_red = (1 - g_avg / r_avg) * 100 if r_avg > 0 else 0
        m_red = (1 - m_avg / r_avg) * 100 if r_avg > 0 else 0
        print(f"  {metric_name:<12}: Raw={r_avg:.6f}  Gauss={g_avg:.6f} ({g_red:.1f}% red)  "
              f"Morph={m_avg:.6f} ({m_red:.1f}% red)")


if __name__ == "__main__":
    main()
