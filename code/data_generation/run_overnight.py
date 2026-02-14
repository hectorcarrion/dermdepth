#!/usr/bin/env python3
"""
Overnight generation run for DermDepth dataset.

Generates samples with skip-existing logic for resumability.
Uses os._exit(0) to avoid drjit cleanup crash with CUDA 12.7 driver.

Usage:
    CUDA_VISIBLE_DEVICES=3 python run_overnight.py
"""
import os
import sys
import time
import atexit
import signal

# Force clean exit to avoid drjit cleanup crash
atexit.register(lambda: os._exit(0))

# Handle SIGTERM gracefully
def sigterm_handler(signum, frame):
    print(f"\nReceived signal {signum}, exiting cleanly...")
    os._exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dermdepth_dataset import generate_dataset

OUTPUT_DIR = '/workspace/hector/dermdepth/data/dermdepth_train'
NUM_SAMPLES = 2000
SEED = 42

print("=" * 60)
print("DermDepth Overnight Generation")
print("=" * 60)
print(f"  Output: {OUTPUT_DIR}")
print(f"  Samples: {NUM_SAMPLES}")
print(f"  Seed: {SEED}")
print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Estimated: ~180s/sample, ~{NUM_SAMPLES * 180 / 3600:.0f}h total")
print(f"  Supports resume (skip-existing)")
print("=" * 60)

t0 = time.time()
results = generate_dataset(OUTPUT_DIR, NUM_SAMPLES, seed=SEED, num_workers=1)
dt = time.time() - t0

success = sum(1 for r in results if r['status'] == 'success')
skipped = sum(1 for r in results if r['status'] == 'skipped')
errors = sum(1 for r in results if r['status'] == 'error')

print(f"\n{'=' * 60}")
print(f"FINAL RESULTS")
print(f"{'=' * 60}")
print(f"  Success: {success}")
print(f"  Skipped: {skipped}")
print(f"  Errors: {errors}")
print(f"  Total time: {dt/3600:.1f} hours")
if success > 0:
    print(f"  Rate: {success/(dt/60):.1f} samples/min = {success/(dt/3600):.0f} samples/hour")
print(f"  End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
