"""
BazarSetu V3.0 - Stage D Fallback: Interactive SAM 2 Tap Refinement
Accepts single normalized (x, y) coordinate taps to refine intricate edges.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import torch

from config import CONFIG
from core.device import DEVICE_MGR

logger = logging.getLogger("BazarSetu.SAM2Refiner")

@dataclass
class RefinementResult:
    refined_mask: np.ndarray       # uint8 0-255 mask
    refined_cutout: np.ndarray     # 4-channel BGRA
    latency_ms: float
    method_used: str

class SAM2InteractiveRefiner:
    def __init__(self, device: Optional[torch.device] = None, weights_path: Optional[Path] = None):
        self.device = device or DEVICE_MGR.get_device(0)
        self.weights_path = Path(weights_path or CONFIG.models.sam2_weights_path)
        self.predictor = None
        self._init_predictor()

    def _init_predictor(self):
        if self.weights_path.exists() and DEVICE_MGR.has_cuda:
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                logger.info(f"Loading SAM 2 from {self.weights_path} onto {self.device}...")
                model = build_sam2(CONFIG.models.sam2_config_name, str(self.weights_path), device=self.device)
                self.predictor = SAM2ImagePredictor(model)
                logger.info("SAM 2 Interactive Predictor ready.")
            except Exception as e:
                logger.warning(f"Could not load SAM 2: {e}. Fallback interactive mode active.")
                self.predictor = None
        else:
            self.predictor = None

    def refine_tap(
        self,
        image_bgr: np.ndarray,
        current_mask: np.ndarray,
        norm_x: float,
        norm_y: float,
        is_foreground: bool = True
    ) -> RefinementResult:
        """
        Refines segmentation given a normalized tap point (0.0 to 1.0).
        """
        h, w = image_bgr.shape[:2]
        px = int(np.clip(norm_x * w, 0, w - 1))
        py = int(np.clip(norm_y * h, 0, h - 1))
        start_time = time.perf_counter()

        if self.predictor is not None and DEVICE_MGR.has_cuda:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)
            
            point_coords = np.array([[px, py]], dtype=np.float32)
            point_labels = np.array([1 if is_foreground else 0], dtype=np.int32)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
            refined_mask = (masks[0] * 255).astype(np.uint8)
            method = "SAM2_FP16"
        else:
            # Algorithmic interactive fallback: local region growing / floodFill on gradient
            refined_mask = current_mask.copy()
            radius = int(max(w, h) * 0.04) # 4% of dimension
            cv2.circle(
                refined_mask,
                (px, py),
                radius,
                255 if is_foreground else 0,
                thickness=-1
            )
            # Smooth transition
            refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
            refined_mask = np.where(refined_mask > 127, 255, 0).astype(np.uint8)
            method = "HEURISTIC_TAP_REFINEMENT"

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Update BGRA cutout
        b, g, r = cv2.split(image_bgr)
        refined_cutout = cv2.merge([b, g, r, refined_mask])

        return RefinementResult(
            refined_mask=refined_mask,
            refined_cutout=refined_cutout,
            latency_ms=latency_ms,
            method_used=method
        )

