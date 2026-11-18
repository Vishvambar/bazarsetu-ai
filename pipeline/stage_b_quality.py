"""
BazarSetu V3.0 - Stage B: Actionable Quality Gate
Computes blur, exposure, and framing metrics. Returns standardized seller feedback codes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from config import QualityGateConfig, CONFIG

@dataclass
class QualityAssessmentResult:
    passed: bool
    feedback_code: str  # OK, BLURRY, TOO_DARK, TOO_BRIGHT, TOO_SMALL, CUT_OFF_FRAME
    seller_message: str
    sharpness_variance: float
    mean_luminance: float
    product_coverage_ratio: float
    edge_margin_px: int

SELLER_MESSAGES = {
    "OK": "Photo quality is good.",
    "BLURRY": "The photograph is blurry. Please hold your phone steady and tap to focus.",
    "TOO_DARK": "The photograph is too dark. Please move to a brighter area or turn on lights.",
    "TOO_BRIGHT": "The photograph is too bright or has excessive glare. Please avoid direct harsh flash.",
    "TOO_SMALL": "The product appears too small. Please move the camera closer to the product.",
    "CUT_OFF_FRAME": "Parts of the product are cut off. Please capture the entire product in the frame."
}

class QualityGateStage:
    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.cfg = config or CONFIG.quality

    def evaluate(self, image_bgr: np.ndarray) -> QualityAssessmentResult:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness via Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Exposure & Luminance
        mean_lum = float(np.mean(gray))
        high_clip_ratio = float(np.sum(gray >= 250) / (h * w))

        # 3. Dynamic Foreground Isolation
        scale = 512.0 / max(h, w)
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        blurred = cv2.GaussianBlur(small_gray, (5, 5), 0)
        h_s, w_s = small_gray.shape[:2]

        # Determine background illuminance from perimeter pixels
        border_pixels = np.concatenate([
            small_gray[0, :], small_gray[-1, :],
            small_gray[:, 0], small_gray[:, -1]
        ])
        bg_median = float(np.median(border_pixels))
        center_roi = small_gray[int(h_s * 0.25):int(h_s * 0.75), int(w_s * 0.25):int(w_s * 0.75)]
        center_median = float(np.median(center_roi))

        # Dynamic threshold polarity based on background tone
        thresh_type = cv2.THRESH_BINARY_INV if bg_median >= center_median else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        product_area_ratio = 0.5
        touching_edges_count = 0
        min_margin = min(w, h)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            bx, by, bw, bh = int(bx / scale), int(by / scale), int(bw / scale), int(bh / scale)
            bbox_area = bw * bh
            product_area_ratio = float(bbox_area / (h * w))
            
            # Check edge intersections (touching within 3px)
            if bx <= 3: touching_edges_count += 1
            if by <= 3: touching_edges_count += 1
            if (w - (bx + bw)) <= 3: touching_edges_count += 1
            if (h - (by + bh)) <= 3: touching_edges_count += 1
            min_margin = min(bx, by, max(0, w - (bx + bw)), max(0, h - (by + bh)))

        # Gate Evaluation logic in priority order
        if laplacian_var < self.cfg.min_laplacian_variance:
            code = "BLURRY"
        elif mean_lum < self.cfg.min_mean_luminance:
            code = "TOO_DARK"
        elif mean_lum > self.cfg.max_mean_luminance or high_clip_ratio > self.cfg.max_clipped_highlight_ratio:
            code = "TOO_BRIGHT"
        elif product_area_ratio < self.cfg.min_product_coverage_ratio:
            code = "TOO_SMALL"
        # Only trigger CUT_OFF_FRAME if product is clipped on multiple sides or fills entire frame
        elif touching_edges_count >= 3 or (touching_edges_count >= 2 and product_area_ratio > 0.88):
            code = "CUT_OFF_FRAME"
        else:
            code = "OK"

        passed = (code == "OK")
        return QualityAssessmentResult(
            passed=passed,
            feedback_code=code,
            seller_message=SELLER_MESSAGES[code],
            sharpness_variance=laplacian_var,
            mean_luminance=mean_lum,
            product_coverage_ratio=product_area_ratio,
            edge_margin_px=min_margin
        )

    def evaluate_file(self, file_path: Path) -> QualityAssessmentResult:
        img = cv2.imread(str(file_path))
        if img is None:
            return QualityAssessmentResult(
                passed=False, feedback_code="BLURRY",
                seller_message="Unable to read image.",
                sharpness_variance=0.0, mean_luminance=0.0,
                product_coverage_ratio=0.0, edge_margin_px=0
            )
        return self.evaluate(img)

