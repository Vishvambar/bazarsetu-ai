"""
BazarSetu V3.0 - Immutable Product Master (Stage E)
Manages the write-protected asset bundle and audit metadata.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("BazarSetu.ProductMaster")

@dataclass
class QualityMetricsRecord:
    sharpness_variance: float
    mean_luminance: float
    product_coverage_ratio: float
    edge_margin_px: int
    quality_code: str
    passed: bool

@dataclass
class FidelityMetricsRecord:
    delta_e00: float
    boundary_f1: float
    mask_iou: float
    verified: bool

@dataclass
class ProcessingAssetRecord:
    asset_id: str
    product_id: str
    original_sha256: str
    created_at_iso: str
    model_name: str
    model_version: str
    gpu_inference_latency_ms: float
    total_processing_latency_ms: float
    quality_metrics: QualityMetricsRecord
    fidelity_metrics: FidelityMetricsRecord
    file_paths: Dict[str, str] = field(default_factory=dict)
    seller_approval_status: str = "PENDING_SELLER_REVIEW"

class ProductMasterManager:
    """
    Ensures non-destructive storage and enforces directory layout:
    products/{product_id}/
      ├── original/
      ├── masks/
      ├── cutouts/
      ├── marketplace/
      ├── thumbnails/
      └── metadata/
    """
    def __init__(self, storage_root: Path):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def init_bundle(self, product_id: str) -> Dict[str, Path]:
        base = self.storage_root / "products" / product_id
        dirs = {
            "root": base,
            "original": base / "original",
            "masks": base / "masks",
            "cutouts": base / "cutouts",
            "marketplace": base / "marketplace",
            "thumbnails": base / "thumbnails",
            "metadata": base / "metadata",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def store_original(self, product_id: str, source_path: Path, asset_id: str) -> Tuple[Path, str]:
        """Copies original raw image to write-protected original/ dir and returns destination and hash."""
        dirs = self.init_bundle(product_id)
        ext = source_path.suffix.lower() if source_path.suffix else ".jpg"
        dest_path = dirs["original"] / f"{asset_id}_original{ext}"
        
        shutil.copy2(source_path, dest_path)
        # Mark read-only for protection (chmod 444)
        try:
            os.chmod(dest_path, 0o444)
        except Exception as e:
            logger.debug(f"Could not set read-only permission: {e}")
            
        file_hash = self.compute_sha256(dest_path)
        logger.info(f"Stored immutable original: {dest_path} (SHA-256: {file_hash[:12]}...)")
        return dest_path, file_hash

    def save_metadata(self, record: ProcessingAssetRecord) -> Path:
        dirs = self.init_bundle(record.product_id)
        meta_file = dirs["metadata"] / f"{record.asset_id}_metadata.json"
        
        # Write to JSON
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)
            
        logger.info(f"Persisted Product Master metadata: {meta_file}")
        return meta_file

