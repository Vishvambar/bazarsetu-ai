"""
BazarSetu V3.0 - Stage A: Image Ingestion
Validates input, checks constraints, and creates an immutable original.
"""

from __future__ import annotations
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

from core.product_master import ProductMasterManager

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_FILE_SIZE_BYTES = 2 * 1024        # 2 KB (lowered to support highly compressed WebP/JPEGs)
MAX_FILE_SIZE_BYTES = 35 * 1024 * 1024 # 35 MB
MIN_DIMENSION = 256                   # Minimum resolution

@dataclass
class IngestionResult:
    asset_id: str
    product_id: str
    original_path: Path
    original_hash: str
    width: int
    height: int
    format: str
    valid: bool
    error_message: Optional[str] = None

class IngestionStage:
    def __init__(self, master_manager: ProductMasterManager):
        self.master_manager = master_manager

    def process(self, source_image_path: Path, product_id: Optional[str] = None) -> IngestionResult:
        source_path = Path(source_image_path)
        product_id = product_id or f"prod_{uuid.uuid4().hex[:10]}"
        asset_id = f"asset_{uuid.uuid4().hex[:12]}"

        if not source_path.exists():
            return IngestionResult(
                asset_id=asset_id, product_id=product_id, original_path=source_path,
                original_hash="", width=0, height=0, format="", valid=False,
                error_message=f"Source file does not exist: {source_path}"
            )

        # Check file extension
        ext = source_path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return IngestionResult(
                asset_id=asset_id, product_id=product_id, original_path=source_path,
                original_hash="", width=0, height=0, format="", valid=False,
                error_message=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # Check file size
        file_size = source_path.stat().st_size
        if file_size < MIN_FILE_SIZE_BYTES or file_size > MAX_FILE_SIZE_BYTES:
            return IngestionResult(
                asset_id=asset_id, product_id=product_id, original_path=source_path,
                original_hash="", width=0, height=0, format="", valid=False,
                error_message=f"File size {file_size / 1024:.1f} KB outside limits (20KB - 30MB)"
            )

        # Validate with PIL
        try:
            with Image.open(source_path) as img:
                w, h = img.size
                img_format = img.format or ext.replace(".", "").upper()
                if w < MIN_DIMENSION or h < MIN_DIMENSION:
                    return IngestionResult(
                        asset_id=asset_id, product_id=product_id, original_path=source_path,
                        original_hash="", width=w, height=h, format=img_format, valid=False,
                        error_message=f"Resolution {w}x{h} is below minimum {MIN_DIMENSION}x{MIN_DIMENSION}"
                    )
        except Exception as e:
            return IngestionResult(
                asset_id=asset_id, product_id=product_id, original_path=source_path,
                original_hash="", width=0, height=0, format="", valid=False,
                error_message=f"Corrupt or unreadable image file: {e}"
            )

        # Store in immutable Product Master original/
        dest_path, file_hash = self.master_manager.store_original(product_id, source_path, asset_id)

        return IngestionResult(
            asset_id=asset_id,
            product_id=product_id,
            original_path=dest_path,
            original_hash=file_hash,
            width=w,
            height=h,
            format=img_format,
            valid=True,
            error_message=None
        )

