"""
BazarSetu V3.0 - Step 6: SigLIP Duplicate Search (GPU 0) & Gemma-Native High-Res Label Escalation (GPU 1).

Zero PaddleOCR dependencies:
  - Duplicate Listing Search compares Stage G front_view 768-d SigLIP embeddings
    against persisted seller embeddings in storage/.
  - High-Res Label Escalation triggers strictly when has_visible_text=True AND
    (needs_highres_label_crop=True OR text_legibility_confidence < 0.75), reading
    System 1's archived original photo from storage/products/{product_id}/original/.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from config import CONFIG
from system2.engine import PATCH_SIZE, System2Engine, compute_patch_aligned_size
from system2.schemas import ProductIntelligenceOutput
from system2.siglip_encoder import SigLIPEncoder, l2_normalize

logger = logging.getLogger("BazarSetu.VisualSearch")

DEFAULT_DUPLICATE_THRESHOLD = 0.88


def should_escalate_label(output: ProductIntelligenceOutput) -> bool:
    """Return True strictly when label escalation is justified per Step 6 gate."""
    seals = output.text_and_seals
    return bool(
        seals.has_visible_text
        and (seals.needs_highres_label_crop or float(seals.text_legibility_confidence) < 0.75)
    )


@dataclass
class DuplicateMatch:
    product_id: str
    similarity: float
    title_en: str = ""
    category: str = ""
    domain_code: str = ""
    thumbnail_url: str = ""


@dataclass
class DuplicateSearchResult:
    product_id: str
    threshold: float
    has_duplicates: bool
    matches: List[DuplicateMatch] = field(default_factory=list)


class VisualDuplicateIndex:
    """Thread-safe persisted JSON index of 768-d SigLIP product embeddings."""

    def __init__(
        self,
        index_path: Optional[Union[str, Path]] = None,
        encoder: Optional[SigLIPEncoder] = None,
    ):
        self.index_path = Path(index_path) if index_path else (CONFIG.storage_root / "system2_visual_index.json")
        self.encoder = encoder or SigLIPEncoder(lazy_load=True)
        self._lock = threading.Lock()
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load_unlocked()

    def _load_unlocked(self) -> None:
        if not self.index_path.exists():
            self._entries = {}
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._entries = data
        except Exception as exc:
            logger.warning("Could not load visual index %s: %s", self.index_path, exc)
            self._entries = {}

    def _save_unlocked(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._entries), encoding="utf-8")
        tmp.replace(self.index_path)

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def upsert(
        self,
        product_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        normed = l2_normalize(np.asarray(embedding, dtype=np.float32))
        with self._lock:
            self._entries[product_id] = {
                "embedding": [round(float(x), 6) for x in normed.tolist()],
                "metadata": dict(metadata or {}),
            }
            self._save_unlocked()

    def search_similar(
        self,
        query_embedding: np.ndarray,
        query_product_id: str = "",
        threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        top_k: int = 5,
    ) -> DuplicateSearchResult:
        query_vec = l2_normalize(np.asarray(query_embedding, dtype=np.float32))
        matches: List[DuplicateMatch] = []

        with self._lock:
            for pid, entry in self._entries.items():
                if query_product_id and pid == query_product_id:
                    continue
                stored = entry.get("embedding")
                if not stored:
                    continue
                vec = l2_normalize(np.asarray(stored, dtype=np.float32))
                if vec.shape != query_vec.shape:
                    continue
                sim = float(np.dot(query_vec, vec))
                if sim >= threshold:
                    meta = entry.get("metadata") or {}
                    matches.append(
                        DuplicateMatch(
                            product_id=pid,
                            similarity=round(sim, 4),
                            title_en=str(meta.get("title_en") or ""),
                            category=str(meta.get("category") or ""),
                            domain_code=str(meta.get("domain_code") or ""),
                            thumbnail_url=str(meta.get("thumbnail_url") or ""),
                        )
                    )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        trimmed = matches[:top_k]
        return DuplicateSearchResult(
            product_id=query_product_id,
            threshold=threshold,
            has_duplicates=len(trimmed) > 0,
            matches=trimmed,
        )

    def index_and_search(
        self,
        product_id: str,
        image_source: Optional[Union[str, Path, Image.Image]] = None,
        image_embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        top_k: int = 5,
    ) -> Tuple[DuplicateSearchResult, np.ndarray]:
        """Compute embedding if needed, find existing duplicates, and index this item."""
        if image_embedding is None:
            if image_source is None:
                raise ValueError("Either image_source or image_embedding is required.")
            emb = self.encoder.encode_image(image_source)
        else:
            emb = l2_normalize(image_embedding)

        search_res = self.search_similar(
            query_embedding=emb,
            query_product_id=product_id,
            threshold=threshold,
            top_k=top_k,
        )
        self.upsert(product_id=product_id, embedding=emb, metadata=metadata)
        return search_res, emb


def resolve_original_image_path(
    product_id: str,
    storage_root: Optional[Path] = None,
    explicit_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Locate System 1's immutable original image on disk for high-res escalation."""
    if explicit_path:
        cand = Path(explicit_path)
        if cand.exists():
            return cand

    root = Path(storage_root) if storage_root else CONFIG.storage_root
    orig_dir = root / "products" / product_id / "original"
    if orig_dir.exists():
        files = sorted(
            [p for p in orig_dir.iterdir() if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0]
    return None


def prepare_highres_label_crop(
    source_image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    max_dim: int = 960,
    center_focus_ratio: float = 0.85,
) -> Path:
    """
    Create a 48px-patch-aligned high-resolution crop from the original photograph
    for Gemma-native label/seal verification (no external OCR engine).
    """
    src = Path(source_image_path)
    if not src.exists():
        raise FileNotFoundError(f"Original image for label escalation not found: {src}")

    with Image.open(src) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        # Tight gentle crop only when border margins dominate; preserves hangtags near edges
        if 0.5 <= center_focus_ratio < 0.99 and min(w, h) >= 320:
            crop_w = int(round(w * center_focus_ratio))
            crop_h = int(round(h * center_focus_ratio))
            left = max(0, (w - crop_w) // 2)
            top = max(0, (h - crop_h) // 2)
            rgb = rgb.crop((left, top, left + crop_w, top + crop_h))

        aligned = compute_patch_aligned_size(rgb.width, rgb.height, max_dim=max_dim)
        if rgb.size != aligned:
            rgb = rgb.resize(aligned, Image.Resampling.LANCZOS)

        dest = Path(output_path) if output_path else src.with_name(f"{src.stem}_esc960.jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dest, format="JPEG", quality=94)
        return dest


def escalate_label_inspection(
    product_id: str,
    initial_output: ProductIntelligenceOutput,
    engine: System2Engine,
    storage_root: Optional[Path] = None,
    explicit_original_path: Optional[Union[str, Path]] = None,
    fallback_view_path: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> Tuple[ProductIntelligenceOutput, bool]:
    """
    Execute a single Gemma-native high-res escalation pass when visible text/seals
    need sharper inspection. Returns (merged_output, escalated_bool).
    """
    if not force and not should_escalate_label(initial_output):
        return initial_output, False

    source_path = resolve_original_image_path(
        product_id=product_id,
        storage_root=storage_root,
        explicit_path=explicit_original_path,
    )
    if source_path is None and fallback_view_path:
        cand = Path(fallback_view_path)
        if cand.exists():
            source_path = cand

    if source_path is None:
        logger.info("Escalation requested for %s, but no original/label image found.", product_id)
        return initial_output, False

    root = Path(storage_root) if storage_root else CONFIG.storage_root
    crop_dest = root / "temp_uploads" / f"{product_id}_escalation.jpg"
    crop_path = prepare_highres_label_crop(source_path, output_path=crop_dest)

    try:
        esc_dict = engine.predict(
            product_id=product_id,
            views={"escalation_crop_view": str(crop_path)},
            escalation_mode=True,
        )
        esc_out = ProductIntelligenceOutput.model_validate(esc_dict)
    finally:
        if crop_path.exists():
            try:
                crop_path.unlink()
            except OSError:
                pass

    merged = initial_output.model_dump()
    # Union seals and snippets preserving order
    existing_seals = list(merged["text_and_seals"].get("certification_seals") or [])
    for s in esc_out.text_and_seals.certification_seals:
        if s and s not in existing_seals:
            existing_seals.append(s)
    merged["text_and_seals"]["certification_seals"] = existing_seals

    existing_snippets = list(merged["text_and_seals"].get("visible_text_snippets") or [])
    for snip in esc_out.text_and_seals.visible_text_snippets:
        if snip and snip not in existing_snippets and len(existing_snippets) < 3:
            existing_snippets.append(snip)
    merged["text_and_seals"]["visible_text_snippets"] = existing_snippets

    if esc_out.text_and_seals.brand_or_artisan_mark and not merged["text_and_seals"].get("brand_or_artisan_mark"):
        merged["text_and_seals"]["brand_or_artisan_mark"] = esc_out.text_and_seals.brand_or_artisan_mark

    merged["text_and_seals"]["text_legibility_confidence"] = max(
        float(merged["text_and_seals"].get("text_legibility_confidence", 0.0)),
        float(esc_out.text_and_seals.text_legibility_confidence),
    )
    merged["text_and_seals"]["needs_highres_label_crop"] = False
    merged["text_and_seals"]["has_visible_text"] = bool(
        merged["text_and_seals"]["has_visible_text"] or existing_seals or existing_snippets
    )

    views_proc = list(merged["audit_telemetry"].get("views_processed") or [])
    if "escalation_crop_view" not in views_proc:
        views_proc.append("escalation_crop_view")
    merged["audit_telemetry"]["views_processed"] = views_proc

    return ProductIntelligenceOutput.model_validate(merged), True

