"""
BazarSetu V3.0 - Kaggle Dedicated Execution Runner
Optimized for Dual NVIDIA Tesla T4 (FP16, cuDNN benchmark)
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config import CONFIG
from core.device import DEVICE_MGR
from pipeline.orchestrator import System1Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BazarSetu.KaggleRunner")

def main():
    parser = argparse.ArgumentParser(description="BazarSetu V3.0 System 01 Kaggle Runner")
    parser.add_argument("--input_images", nargs="+", help="Paths to raw smartphone photographs")
    parser.add_argument("--product_id", type=str, default="kaggle_demo_product")
    parser.add_argument("--interactive_refine", action="store_true", help="Simulate a SAM 2 tap refinement")
    parser.add_argument("--tap_coord", nargs=2, type=float, default=[0.5, 0.5], help="Normalized (x, y) tap coordinates")
    parser.add_argument("--force", action="store_true", help="Bypass Quality Gate rejections (useful for evaluation/previews)")
    args = parser.parse_args()

    logger.info("Initializing BazarSetu System 01 on Kaggle...")
    logger.info(f"Target Storage: {CONFIG.storage_root}")
    
    pipeline = System1Pipeline()

    # Determine input files
    if args.input_images:
        input_paths = [Path(p) for p in args.input_images]
    else:
        # Check if Kaggle sample dir exists, otherwise generate synthetic test pattern for self-verification
        sample_dir = Path("/kaggle/input/bazarsetu-samples")
        if sample_dir.exists():
            input_paths = list(sample_dir.glob("*.jpg"))[:4]
        else:
            logger.info("No input specified. Creating a synthetic artisan pottery test image for validation...")
            test_dir = CONFIG.storage_root / "test_inputs"
            test_dir.mkdir(parents=True, exist_ok=True)
            test_img_path = test_dir / "artisan_pottery_sample.png"
            
            import cv2
            import numpy as np
            # Create a 800x800 sample image with neutral background and central terracotta pot
            canvas = np.full((800, 800, 3), (210, 215, 220), dtype=np.uint8)
            # Draw terracotta vase
            cv2.ellipse(canvas, (400, 480), (160, 200), 0, 0, 360, (45, 82, 185), -1)
            cv2.ellipse(canvas, (400, 280), (80, 40), 0, 0, 360, (35, 70, 165), -1)
            # Add subtle texture
            noise = np.random.normal(0, 5, canvas.shape).astype(np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(test_img_path), canvas)
            input_paths = [test_img_path]

    logger.info(f"Processing {len(input_paths)} image(s) using Dual-T4 worker pool (force={args.force})...")
    start_total = time.perf_counter()
    results = pipeline.process_batch(input_paths, product_id=args.product_id, force=args.force)
    total_time_s = time.perf_counter() - start_total

    print("\n" + "=" * 65)
    print(" BAZARSETU V3.0 - SYSTEM 01 EXECUTION SUMMARY (KAGGLE)")
    print("=" * 65)
    for r in results:
        print(f"Asset ID        : {r.asset_id}")
        print(f"Status          : {r.status} ({r.seller_message})")
        print(f"Quality Code    : {r.quality_code}")
        print(f"GPU Latency     : {r.gpu_inference_latency_ms:.1f} ms")
        print(f"Total Latency   : {r.total_latency_ms:.1f} ms")
        print(f"Marketplace Img : {r.marketplace_image_path}")
        print(f"Thumbnail       : {r.thumbnail_path}")
        print(f"Metadata Audit  : {r.metadata_path}")
        print("-" * 65)
    print(f"BATCH TURNAROUND TIME: {total_time_s:.2f} s across {len(input_paths)} image(s)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()

