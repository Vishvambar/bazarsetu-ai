"""
BazarSetu V3.0 - System 01 Full Orchestrator (Stages A through H)
Supports single image processing and multi-angle batch execution across dual Tesla T4s.
"""

from __future__ import annotations
import concurrent.futures
from datetime import datetime, timezone
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from config import PipelineConfig, CONFIG
from core.device import DEVICE_MGR
from core.product_master import (
    FidelityMetricsRecord,
    ProcessingAssetRecord,
    ProductMasterManager,
    QualityMetricsRecord,
)
from pipeline.stage_a_ingestion import IngestionResult, IngestionStage
from pipeline.stage_b_quality import QualityAssessmentResult, QualityGateStage
from pipeline.stage_cd_segmentation import BiRefNetSegmenter, SegmentationResult
from pipeline.stage_d_sam2 import SAM2InteractiveRefiner
from pipeline.stage_f_polish import PhotographicPolishStage, PolishResult
from pipeline.stage_g_rendering import MarketplaceRenderingStage, RenderingResult
from pipeline.stage_h_fidelity import FidelityVerificationResult, FidelityVerificationStage

logger = logging.getLogger("BazarSetu.Orchestrator")

@dataclass
class PipelineExecutionResult:
    asset_id: str
    product_id: str
    status: str # COMPLETED, ACTION_REQUIRED, FAILED
    seller_message: str
    quality_code: str
    gpu_inference_latency_ms: float
    total_latency_ms: float
    marketplace_image_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    cutout_path: Optional[str] = None
    mask_path: Optional[str] = None
    metadata_path: Optional[str] = None
    delta_e00: float = 0.0
    boundary_f1: float = 0.0

class System1Pipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or CONFIG
        self.master_manager = ProductMasterManager(self.cfg.storage_root)
        
        # Stages
        self.stage_a = IngestionStage(self.master_manager)
        self.stage_b = QualityGateStage(self.cfg.quality)
        self.stage_f = PhotographicPolishStage(self.cfg.polish)
        self.stage_g = MarketplaceRenderingStage(self.cfg.rendering)
        self.stage_h = FidelityVerificationStage(self.cfg.verification)
        
        # Segmentation workers for dual GPU
        self.segmenters: Dict[int, BiRefNetSegmenter] = {}
        self.sam2_refiners: Dict[int, SAM2InteractiveRefiner] = {}
        self._init_gpu_workers()

    def _init_gpu_workers(self):
        # Warmup and initialize primary and secondary GPU workers
        DEVICE_MGR.warmup()
        num_workers = max(1, DEVICE_MGR.device_count)
        for w_idx in range(num_workers):
            dev = DEVICE_MGR.get_device(w_idx)
            self.segmenters[w_idx] = BiRefNetSegmenter(device=dev)
            self.sam2_refiners[w_idx] = SAM2InteractiveRefiner(device=dev)

    def process_single(
        self,
        source_image_path: Path,
        product_id: Optional[str] = None,
        worker_id: int = 0,
        force: bool = False
    ) -> PipelineExecutionResult:
        """
        Executes full Stages A through H for a single photograph.
        """
        start_time = time.perf_counter()
        
        # --- Stage A: Ingestion ---
        ingest_res: IngestionResult = self.stage_a.process(source_image_path, product_id)
        if not ingest_res.valid:
            return PipelineExecutionResult(
                asset_id=ingest_res.asset_id,
                product_id=ingest_res.product_id,
                status="FAILED",
                seller_message=ingest_res.error_message or "Ingestion failed.",
                quality_code="CORRUPT_OR_INVALID",
                gpu_inference_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000.0
            )

        # Load stored original
        image_bgr = cv2.imread(str(ingest_res.original_path))
        if image_bgr is None:
            return PipelineExecutionResult(
                asset_id=ingest_res.asset_id,
                product_id=ingest_res.product_id,
                status="FAILED",
                seller_message="Could not read stored image.",
                quality_code="READ_ERROR",
                gpu_inference_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000.0
            )

        # --- Stage B: Actionable Quality Gate ---
        qa_res: QualityAssessmentResult = self.stage_b.evaluate(image_bgr)
        if not qa_res.passed and not force:
            logger.info(f"Quality Gate rejected: code={qa_res.feedback_code}")
            return PipelineExecutionResult(
                asset_id=ingest_res.asset_id,
                product_id=ingest_res.product_id,
                status="ACTION_REQUIRED",
                seller_message=qa_res.seller_message,
                quality_code=qa_res.feedback_code,
                gpu_inference_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000.0
            )
        elif not qa_res.passed and force:
            logger.info(f"Quality Gate advisory: code={qa_res.feedback_code} (bypassed with force=True)")

        # --- Stage C & D: Segmentation & Alpha Matting (GPU) ---
        segmenter = self.segmenters[worker_id % len(self.segmenters)]
        seg_res: SegmentationResult = segmenter.segment(image_bgr)
        gpu_latency = seg_res.inference_time_ms

        # --- Stage F: Deterministic Photographic Polish ---
        polish_res: PolishResult = self.stage_f.process(image_bgr, seg_res.mask)

        # --- Stage G: Marketplace Grounding & Rendering (1200x1200) ---
        render_res: RenderingResult = self.stage_g.render(polish_res.polished_cutout)

        # --- Stage H: Automated Fidelity Verification ---
        fid_res: FidelityVerificationResult = self.stage_h.verify(
            original_bgr=image_bgr,
            polished_bgr=polish_res.polished_bgr,
            alpha_mask=seg_res.mask
        )

        # --- Stage E: Persist to Immutable Product Master ---
        dirs = self.master_manager.init_bundle(ingest_res.product_id)
        asset_id = ingest_res.asset_id

        mask_dest = dirs["masks"] / f"{asset_id}_mask.png"
        cutout_dest = dirs["cutouts"] / f"{asset_id}_cutout.png"
        market_dest = dirs["marketplace"] / f"{asset_id}_marketplace.webp"
        thumb_dest = dirs["thumbnails"] / f"{asset_id}_thumb.webp"

        cv2.imwrite(str(mask_dest), seg_res.mask)
        cv2.imwrite(str(cutout_dest), polish_res.polished_cutout)
        cv2.imwrite(str(market_dest), render_res.marketplace_image, [cv2.IMWRITE_WEBP_QUALITY, 92])
        cv2.imwrite(str(thumb_dest), render_res.thumbnail_image, [cv2.IMWRITE_WEBP_QUALITY, 85])

        total_latency = (time.perf_counter() - start_time) * 1000.0

        # Create audit record
        record = ProcessingAssetRecord(
            asset_id=asset_id,
            product_id=ingest_res.product_id,
            original_sha256=ingest_res.original_hash,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            model_name=seg_res.model_name,
            model_version=seg_res.model_version,
            gpu_inference_latency_ms=gpu_latency,
            total_processing_latency_ms=total_latency,
            quality_metrics=QualityMetricsRecord(
                sharpness_variance=qa_res.sharpness_variance,
                mean_luminance=qa_res.mean_luminance,
                product_coverage_ratio=qa_res.product_coverage_ratio,
                edge_margin_px=qa_res.edge_margin_px,
                quality_code=qa_res.feedback_code,
                passed=qa_res.passed
            ),
            fidelity_metrics=FidelityMetricsRecord(
                delta_e00=fid_res.delta_e00,
                boundary_f1=fid_res.boundary_f1,
                mask_iou=fid_res.mask_iou,
                verified=fid_res.verified
            ),
            file_paths={
                "original": str(ingest_res.original_path),
                "mask": str(mask_dest),
                "cutout": str(cutout_dest),
                "marketplace": str(market_dest),
                "thumbnail": str(thumb_dest),
            },
            seller_approval_status="PENDING_SELLER_REVIEW"
        )
        meta_dest = self.master_manager.save_metadata(record)

        status = "COMPLETED" if fid_res.verified else "ACTION_REQUIRED"
        msg = "Ready for seller review (Sahi Hai)." if fid_res.verified else (fid_res.rejection_reason or "Fidelity verification warning.")

        return PipelineExecutionResult(
            asset_id=asset_id,
            product_id=ingest_res.product_id,
            status=status,
            seller_message=msg,
            quality_code=qa_res.feedback_code,
            gpu_inference_latency_ms=gpu_latency,
            total_latency_ms=total_latency,
            marketplace_image_path=str(market_dest),
            thumbnail_path=str(thumb_dest),
            cutout_path=str(cutout_dest),
            mask_path=str(mask_dest),
            metadata_path=str(meta_dest),
            delta_e00=fid_res.delta_e00,
            boundary_f1=fid_res.boundary_f1
        )

    def process_batch(
        self,
        image_paths: List[Path],
        product_id: Optional[str] = None,
        force: bool = False
    ) -> List[PipelineExecutionResult]:
        """
        Distributes 2-5 product images across dual Tesla T4s concurrently.
        """
        product_id = product_id or f"prod_{int(time.time())}"
        results: List[PipelineExecutionResult] = []
        
        num_workers = max(1, DEVICE_MGR.device_count)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self.process_single, path, product_id, idx, force): idx
                for idx, path in enumerate(image_paths)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                res = future.result()
                results.append(res)
                
        return sorted(results, key=lambda r: r.asset_id)

