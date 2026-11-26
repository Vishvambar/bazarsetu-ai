"""
BazarSetu V3.0 - System 2 Golden Cutout Generator & Archive Manager.

Runs every image in V3.0/dataset/dataset_manifest.json through System 1
(Stages A-H on cuda:0 with force=True) and copies standardized Stage G
marketplace studio outputs into V3.0/storage/golden/. Supports zip export
and restore so golden cutouts persist across Kaggle session restarts.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import System1Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BazarSetu.PrepareSystem2Golden")

MIN_STAGE_A_DIM = 256
UPSCALE_MIN_SIDE = 384

DEFAULT_RESTORE_CANDIDATES = (
    Path("/kaggle/input/bazarsetu-golden-cutouts/golden_cutouts.zip"),
    Path("/kaggle/working/golden_cutouts.zip"),
    PROJECT_ROOT / "dataset" / "golden_cutouts.zip",
)


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


def export_golden_zip(golden_dir: Path, zip_path: Path) -> int:
    """Archive all cutouts under storage/golden/ into a portable zip file."""
    files = sorted(golden_dir.glob("*.png"))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    size_mb = zip_path.stat().st_size / (1024.0 * 1024.0)
    logger.info("Exported %d golden cutouts -> %s (%.2f MB)", len(files), zip_path, size_mb)
    return len(files)


def restore_golden_zip(zip_path: Path, golden_dir: Path) -> int:
    """Restore cutouts from a zip archive safely into storage/golden/."""
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_root = golden_dir.resolve()
    restored = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            safe_name = Path(member.filename).name
            if not safe_name.endswith(".png"):
                continue
            dest = (golden_dir / safe_name).resolve()
            if not str(dest).startswith(str(golden_root)):
                raise ValueError(f"Unsafe zip entry skipped: {member.filename}")
            with zf.open(member, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            restored += 1
    logger.info("Restored %d golden cutouts from %s -> %s", restored, zip_path, golden_dir)
    return restored


def find_restore_archive(explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None
    for cand in DEFAULT_RESTORE_CANDIDATES:
        if cand.exists():
            return cand
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage G studio cutouts for System 2 golden eval.")
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Skip views whose target file already exists in storage/golden/.",
    )
    parser.add_argument(
        "--export-zip",
        type=str,
        default=None,
        help="Save generated storage/golden/*.png files into a zip archive (e.g., /kaggle/working/golden_cutouts.zip).",
    )
    parser.add_argument(
        "--restore-zip",
        type=str,
        default=None,
        help="Restore cutouts from a zip archive before checking --missing-only.",
    )
    parser.add_argument(
        "--auto-restore",
        action="store_true",
        help="Automatically check standard Kaggle Input / repo zip paths to restore without re-running BiRefNet.",
    )
    args = parser.parse_args()

    golden_dir = PROJECT_ROOT / "storage" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    if args.restore_zip or args.auto_restore:
        archive = find_restore_archive(args.restore_zip)
        if archive:
            restore_golden_zip(archive, golden_dir)
        elif args.restore_zip:
            raise FileNotFoundError(f"Restore archive not found: {args.restore_zip}")

    manifest_path = PROJECT_ROOT / "dataset" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Quick scan: if all target cutouts already exist (e.g., after restore-zip), avoid loading BiRefNet!
    pending = []
    for prod in manifest["products"]:
        pid = prod["product_id"]
        for view_key, raw_rel in prod["raw_views"].items():
            target_rel = prod["golden_views"][view_key]
            target_path = PROJECT_ROOT / target_rel
            if args.missing_only and target_path.exists():
                continue
            pending.append((pid, view_key, PROJECT_ROOT / raw_rel, target_rel, target_path))

    if not pending:
        logger.info("All %d golden cutouts already present in %s; skipping Stage G.", manifest["total_images"], golden_dir)
        if args.export_zip:
            export_golden_zip(golden_dir, Path(args.export_zip))
        return

    logger.info("Initializing System 1 pipeline (cuda:0 only) for %d image(s)...", len(pending))
    pipeline = System1Pipeline(gpu_ids=[0])

    processed = 0
    failed = 0
    with tempfile.TemporaryDirectory(prefix="golden_prep_") as tmp_root:
        tmp_dir = Path(tmp_root)
        for pid, view_key, raw_path, target_rel, target_path in pending:
            target_path.parent.mkdir(parents=True, exist_ok=True)
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

    logger.info("Golden cutout generation finished: %d succeeded, %d failed.", processed, failed)
    if args.export_zip:
        export_golden_zip(golden_dir, Path(args.export_zip))


if __name__ == "__main__":
    main()
