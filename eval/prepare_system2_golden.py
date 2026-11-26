"""
BazarSetu V3.0 - System 2 Golden Cutout Generator.

Runs every image in V3.0/dataset/dataset_manifest.json through System 1
(Stages A-H with force=True) and copies standardized Stage G marketplace
studio outputs into V3.0/storage/golden/ for Step 3/4 evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import System1Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BazarSetu.PrepareSystem2Golden")

MIN_STAGE_A_DIM = 256
UPSCALE_MIN_SIDE = 384


def ensure_min_stage_a_resolution(src_path: Path, temp_dir: Path) -> Path:
    """
    Stage A rejects images under 256x256. For legacy low-res evaluation
    thumbnails (e.g., 225x225 or 153x306), write a Lanczos-upscaled temporary
    copy so Stage G still produces clean studio cutouts without relaxing Stage A.
    """
    with Image.open(src_path) as im:
        w, h = im.size
        if min(w, h) >= MIN_STAGE_A_DIM:
            return src_path
        scale = UPSCALE_MIN_SIDE / float(min(w, h))
        new_size = (int(round(w * scale)), int(round(h * scale)))
        upscaled = im.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
        tmp_path = temp_dir / f"upscaled_{src_path.stem}.png"
        upscaled.save(tmp_path, format="PNG")
        logger.info("Upscaled %s (%dx%d -> %dx%d) for Stage A.", src_path.name, w, h, *new_size)
        return tmp_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage G studio cutouts for System 2 golden eval.")
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Skip views whose target file already exists in storage/golden/.",
    )
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / "dataset" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    golden_dir = PROJECT_ROOT / "storage" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing System 1 pipeline to generate clean Stage G studio cutouts...")
    pipeline = System1Pipeline()

    processed = 0
    skipped = 0
    failed = 0
    with tempfile.TemporaryDirectory(prefix="golden_prep_") as tmp_root:
        tmp_dir = Path(tmp_root)
        for prod in manifest["products"]:
            pid = prod["product_id"]
            for view_key, raw_rel in prod["raw_views"].items():
                raw_path = PROJECT_ROOT / raw_rel
                target_rel = prod["golden_views"][view_key]
                target_path = PROJECT_ROOT / target_rel
                target_path.parent.mkdir(parents=True, exist_ok=True)

                if args.missing_only and target_path.exists():
                    skipped += 1
                    continue

                ingest_input = ensure_min_stage_a_resolution(raw_path, tmp_dir)
                res = pipeline.process_single(ingest_input, product_id=pid, force=True)
                src_out = Path(res.marketplace_image_path) if res.marketplace_image_path else None
                if src_out and src_out.exists():
                    shutil.copy2(src_out, target_path)
                    processed += 1
                    logger.info("[OK] %s (%s) -> %s", pid, view_key, target_rel)
                else:
                    failed += 1
                    logger.error("[FAIL] %s (%s): status=%s msg=%s", pid, view_key, res.status, res.seller_message)

    logger.info(
        "Golden cutout generation finished: %d succeeded, %d skipped, %d failed.",
        processed,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
