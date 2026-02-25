#!/usr/bin/env python3
"""Consolidate all evaluation results into a single JSON file.

Reads results from all experiment evaluation directories and produces
a unified JSON suitable for table generation and figure plotting.
"""

import json
from pathlib import Path
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "output" / "evaluation"
OUT_FILE = EVAL_DIR / "all_results.json"


def find_results(exp_prefix, step, dataset, ema=True):
    """Find results.json for a given experiment/step/dataset combination."""
    suffix = "ema" if ema else "raw"
    base_dir = EVAL_DIR / f"{exp_prefix}_step{step}_{suffix}"

    # Try various nesting patterns (due to --model_name inconsistencies)
    candidates = [
        base_dir / dataset / "results.json",
        base_dir / "model" / dataset / "results.json",
        base_dir / dataset / dataset / "results.json",
        base_dir / dataset / "model" / dataset / "results.json",
        base_dir / dataset / dataset / dataset / "results.json",
    ]

    for p in candidates:
        if p.exists():
            return p
    return None


def extract_metrics(results_path):
    """Extract key metrics from a results.json file."""
    with open(results_path) as f:
        data = json.load(f)

    # Handle both flat and nested formats
    overall = data.get("overall", data.get("summary", {}))

    metrics = {}
    for key in ["abs_rel", "absrel", "scale_ratio", "si_delta1", "si_absrel",
                 "delta1", "rmse", "scale_error_pct"]:
        if key in overall:
            val = overall[key]
            if isinstance(val, dict):
                metrics[key] = val.get("mean", val)
            else:
                metrics[key] = val

    # Normalize key names
    if "absrel" in metrics and "abs_rel" not in metrics:
        metrics["abs_rel"] = metrics.pop("absrel")

    metrics["n_samples"] = data.get("num_evaluated", data.get("n_evaluated", None))

    # Per-version breakdown if available
    if "per_version" in data:
        metrics["per_version"] = data["per_version"]
    if "per_disease" in data:
        metrics["per_disease"] = data["per_disease"]

    return metrics


def main():
    all_results = OrderedDict()

    # ===================== Baseline =====================
    print("Collecting baseline results...")
    baseline = {"name": "MoGe-2 (baseline)", "trainable_params": 0}
    for ds in ["skinl2", "woundsdb"]:
        path = find_results("exp_a", 0, ds, ema=True)
        if path is None:
            # Try baseline dir
            path = EVAL_DIR / "baseline" / ds / "results.json"
        if path and path.exists():
            baseline[ds] = extract_metrics(path)
            print(f"  {ds}: {path}")
    all_results["baseline"] = baseline

    # ===================== Exp A =====================
    print("\nCollecting Exp A results...")
    exp_a = {"name": "Exp A: Scale head only", "trainable_params": "2.1M"}
    exp_a["checkpoints"] = OrderedDict()
    for step in [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000, 4000]:
        ckpt = {}
        for ds in ["skinl2", "woundsdb"]:
            path = find_results("exp_a", step, ds, ema=True)
            if path:
                ckpt[ds] = extract_metrics(path)
        if ckpt:
            exp_a["checkpoints"][str(step)] = ckpt
            print(f"  step {step}: {list(ckpt.keys())}")
    exp_a["best_step"] = 1000
    exp_a["best_criterion"] = "SKINL2 scale closest to 1.0"
    all_results["exp_a"] = exp_a

    # ===================== Exp B =====================
    print("\nCollecting Exp B results...")
    exp_b = {"name": "Exp B: Decoder fine-tune", "trainable_params": "22M"}
    exp_b["checkpoints"] = OrderedDict()
    for step in [2000, 4000, 6000, 8000, 10000]:
        ckpt = {}
        for ds in ["skinl2", "woundsdb"]:
            path = find_results("exp_b", step, ds, ema=True)
            if path:
                ckpt[ds] = extract_metrics(path)
        if ckpt:
            exp_b["checkpoints"][str(step)] = ckpt
            print(f"  step {step}: {list(ckpt.keys())}")
    all_results["exp_b"] = exp_b

    # ===================== Exp C =====================
    print("\nCollecting Exp C results...")
    exp_c = {"name": "Exp C: Full fine-tune", "trainable_params": "326M"}
    exp_c["checkpoints"] = OrderedDict()
    for step in [3000, 6000, 9000, 12000, 15000]:
        ckpt = {}
        for ds in ["skinl2", "woundsdb"]:
            path = find_results("exp_c", step, ds, ema=True)
            if path:
                ckpt[ds] = extract_metrics(path)
        if ckpt:
            exp_c["checkpoints"][str(step)] = ckpt
            print(f"  step {step}: {list(ckpt.keys())}")
    if exp_c["checkpoints"]:
        all_results["exp_c"] = exp_c
    else:
        print("  (no results yet)")

    # ===================== Save =====================
    with open(OUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved consolidated results to {OUT_FILE}")
    print(f"Experiments: {list(all_results.keys())}")


if __name__ == "__main__":
    main()
