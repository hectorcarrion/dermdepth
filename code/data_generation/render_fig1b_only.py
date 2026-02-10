#!/usr/bin/env python3
"""
Quick script to generate Fig 1b style samples only.
"""
import os
import sys

# Set up Mitsuba with CUDA variant
import mitsuba as mi

try:
    mi.set_variant('cuda_ad_spectral')
    print("Using CUDA spectral variant (GPU)")
except:
    mi.set_variant('scalar_spectral')
    print("Falling back to scalar spectral (CPU)")

# Import the main script's functions
from render_gpu_exploration import (
    generate_fig1b_samples,
    OUTPUT_DIR
)

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Fig 1b Style Samples")
    print("=" * 60)

    samples = generate_fig1b_samples()

    print(f"\n{'='*60}")
    print(f"Generated {len(samples)} samples")
    print(f"Output: {OUTPUT_DIR}/fig1b_samples/")
    print("=" * 60)
