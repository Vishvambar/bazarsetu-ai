"""
BazarSetu V3.0 - System 2 Golden Cutout Generator.

Runs every image in V3.0/dataset/dataset_manifest.json through System 1
(Stages A-H with force=True) and copies standardized Stage G marketplace
studio outputs into V3.0/storage/golden/ for Step 3/4 evaluation.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import System1Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BazarSetu.PrepareSystem2Golden")


def main() -> None:
    manifest_path = PROJECT_ROOT / "dataset" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    golden_dir = PROJECT_ROOT / "storage" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing System 1 pipeline to generate clean Stage G studio cutouts...")
    pipeline = System1Pipeline()

    processed = 0
    failed = 0
    for prod in manifest["products"]:
        pid = prod["product_id"]
        for view_key, raw_rel in prod["raw_views"].items():
            raw_path = PROJECT_ROOT / raw_rel
            target_rel = prod["golden_views"][view_key]
            target_path = PROJECT_ROOT / target_rel
            target_path.parent.mkdir(parents=True, exist_ok=True)

            res = pipeline.process_single(raw_path, product_id=pid, force=True)
            src_out = Path(res.marketplace_image_path) if res.marketplace_image_path else None
            if src_out and src_out.exists():
                shutil.copy2(src_out, target_path)
                processed += 1
                logger.info("[OK] %s (%s) -> %s", pid, view_key, target_rel)
            else:
                failed += 1
                logger.error("[FAIL] %s (%s): status=%s msg=%s", pid, view_key, res.status, res.seller_message)

    logger.info("Golden cutout generation finished: %d succeeded, %d failed.", processed, failed)


if __name__ == "__main__":
    main()

