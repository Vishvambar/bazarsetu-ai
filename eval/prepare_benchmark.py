"""
BazarSetu V3.0 - Benchmark Dataset Preparer
Stages a curated set of diverse artisan craft images (pottery, brass, textiles, woodcraft, jewellery)
with ground-truth masks for quantitative Definition of Done (DOD) verification.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BazarSetu.BenchmarkData")

def create_synthetic_artisan_benchmark(output_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Creates a benchmark set of complex artisan craft items with known ground-truth masks:
    1. Terracotta Pottery: Curved clay contours, dark workshop background.
    2. Brass/Metalcraft: High specular reflections and glare.
    3. Handloom Textiles: Fine weave edges and tassels.
    4. Woodcraft: Organic wooden grain and hollow cutouts.
    5. Intricate Jewellery: Thin filigree loops and chains (thin-edge boundary stress test).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_pairs = []

    # --- Sample 1: Terracotta Water Pot (Pottery) ---
    img1 = np.full((1000, 1000, 3), (120, 115, 110), dtype=np.uint8) # Workshop table
    mask1 = np.zeros((1000, 1000), dtype=np.uint8)
    # Pot belly
    cv2.ellipse(img1, (500, 560), (280, 320), 0, 0, 360, (38, 70, 168), -1) # Terracotta orange/brown
    cv2.ellipse(mask1, (500, 560), (280, 320), 0, 0, 360, 255, -1)
    # Pot neck
    cv2.ellipse(img1, (500, 260), (140, 70), 0, 0, 360, (30, 60, 150), -1)
    cv2.ellipse(mask1, (500, 260), (140, 70), 0, 0, 360, 255, -1)
    # Pot rim
    cv2.ellipse(img1, (500, 200), (160, 50), 0, 0, 360, (25, 50, 140), -1)
    cv2.ellipse(mask1, (500, 200), (160, 50), 0, 0, 360, 255, -1)
    
    p1_img, p1_mask = output_dir / "01_terracotta_pot.jpg", output_dir / "01_terracotta_pot_mask.png"
    cv2.imwrite(str(p1_img), img1)
    cv2.imwrite(str(p1_mask), mask1)
    generated_pairs.append((p1_img, p1_mask))

    # --- Sample 2: Brass Oil Diya / Bowl (Metalcraft with Specular Glare) ---
    img2 = np.full((1000, 1000, 3), (70, 65, 60), dtype=np.uint8) # Dark surface
    mask2 = np.zeros((1000, 1000), dtype=np.uint8)
    # Brass base
    cv2.ellipse(img2, (500, 500), (320, 160), 0, 0, 360, (50, 180, 215), -1) # Gold/brass tone
    cv2.ellipse(mask2, (500, 500), (320, 160), 0, 0, 360, 255, -1)
    # High glare highlight
    cv2.ellipse(img2, (440, 480), (100, 40), -15, 0, 360, (180, 240, 255), -1)
    
    p2_img, p2_mask = output_dir / "02_brass_diya.jpg", output_dir / "02_brass_diya_mask.png"
    cv2.imwrite(str(p2_img), img2)
    cv2.imwrite(str(p2_mask), mask2)
    generated_pairs.append((p2_img, p2_mask))

    # --- Sample 3: Handloom Silk Stole with Fringe (Textiles) ---
    img3 = np.full((1000, 1000, 3), (200, 205, 210), dtype=np.uint8)
    mask3 = np.zeros((1000, 1000), dtype=np.uint8)
    # Rectangular cloth
    cv2.rectangle(img3, (250, 200), (750, 750), (140, 35, 180), -1) # Crimson silk
    cv2.rectangle(mask3, (250, 200), (750, 750), 255, -1)
    # Tassels / fringe (loose threads)
    for tx in range(260, 740, 8):
        cv2.line(img3, (tx, 750), (tx + np.random.randint(-4, 4), 820), (100, 25, 140), 2)
        cv2.line(mask3, (tx, 750), (tx + np.random.randint(-4, 4), 820), 255, 2)
        
    p3_img, p3_mask = output_dir / "03_handloom_silk.jpg", output_dir / "03_handloom_silk_mask.png"
    cv2.imwrite(str(p3_img), img3)
    cv2.imwrite(str(p3_mask), mask3)
    generated_pairs.append((p3_img, p3_mask))

    # --- Sample 4: Carved Wooden Jewellery Box (Woodcraft) ---
    img4 = np.full((1000, 1000, 3), (160, 160, 160), dtype=np.uint8)
    mask4 = np.zeros((1000, 1000), dtype=np.uint8)
    # Wooden box body
    cv2.rectangle(img4, (280, 320), (720, 680), (35, 65, 115), -1) # Dark teak wood
    cv2.rectangle(mask4, (280, 320), (720, 680), 255, -1)
    # Brass latch on top
    cv2.rectangle(img4, (470, 480), (530, 530), (60, 190, 220), -1)

    p4_img, p4_mask = output_dir / "04_carved_woodbox.jpg", output_dir / "04_carved_woodbox_mask.png"
    cv2.imwrite(str(p4_img), img4)
    cv2.imwrite(str(p4_mask), mask4)
    generated_pairs.append((p4_img, p4_mask))

    # --- Sample 5: Filigree Earrings with Thin Chain (Jewellery Boundary Stress Test) ---
    img5 = np.full((1000, 1000, 3), (40, 40, 40), dtype=np.uint8) # Dark velvet cloth
    mask5 = np.zeros((1000, 1000), dtype=np.uint8)
    # Thin top hook
    cv2.ellipse(img5, (500, 220), (25, 45), 0, 0, 180, (80, 210, 235), 3)
    cv2.ellipse(mask5, (500, 220), (25, 45), 0, 0, 180, 255, 3)
    # Central intricate filigree pendant with hollow interior
    cv2.circle(img5, (500, 450), 160, (70, 200, 230), 6) # Outer ring
    cv2.circle(mask5, (500, 450), 160, 255, 6)
    cv2.circle(img5, (500, 450), 70, (80, 210, 235), -1) # Center gem
    cv2.circle(mask5, (500, 450), 70, 255, -1)
    # Dangling beads
    for bx in [400, 450, 500, 550, 600]:
        cv2.line(img5, (bx, 600), (bx, 680), (80, 210, 235), 2)
        cv2.line(mask5, (bx, 600), (bx, 680), 255, 2)
        cv2.circle(img5, (bx, 690), 10, (70, 190, 220), -1)
        cv2.circle(mask5, (bx, 690), 10, 255, -1)

    p5_img, p5_mask = output_dir / "05_filigree_jewellery.jpg", output_dir / "05_filigree_jewellery_mask.png"
    cv2.imwrite(str(p5_img), img5)
    cv2.imwrite(str(p5_mask), mask5)
    generated_pairs.append((p5_img, p5_mask))

    logger.info(f"Successfully staged {len(generated_pairs)} benchmark artisan image-mask pairs in {output_dir}")
    return generated_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/storage/benchmark_data")
    args = parser.parse_args()
    create_synthetic_artisan_benchmark(Path(args.output_dir))

