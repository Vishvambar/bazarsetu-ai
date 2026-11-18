"""
BazarSetu V3.0 - Stage F: Deterministic Photographic Polish
Fixes indoor lighting casts, lifts dark midtones, removes sensor noise, and sharpens details.
Strictly non-generative (deterministic pixel math).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from config import PolishConfig, CONFIG

logger = logging.getLogger("BazarSetu.Polish")

@dataclass
class PolishResult:
    polished_bgr: np.ndarray
    polished_cutout: np.ndarray
    white_balance_applied: bool
    gamma_applied: float
    denoise_applied: bool
    unsharp_applied: bool

class PhotographicPolishStage:
    def __init__(self, config: Optional[PolishConfig] = None):
        self.cfg = config or CONFIG.polish

    def apply_auto_white_balance(self, image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Illuminant cast correction.
        Estimates indoor light cast primarily from ambient background pixels (if available)
        so that product's natural dyes/clay colors are not neutralized.
        """
        result = image_bgr.astype(np.float32)
        
        # Estimate illuminant from ambient background where possible, to preserve foreground pigments
        if mask is not None and np.sum(mask == 0) > 500:
            sample_pixels = result[mask == 0]
        elif mask is not None and np.sum(mask > 0) > 100:
            sample_pixels = result[mask > 0]
        else:
            sample_pixels = result.reshape(-1, 3)

        # Shades-of-Gray (p=6 norm)
        p = 6.0
        norm_per_channel = np.power(np.mean(np.power(sample_pixels, p), axis=0), 1.0 / p)
        
        gray_val = np.mean(norm_per_channel)
        scale = gray_val / (norm_per_channel + 1e-6)
        
        # Gentle scale dampening (max +- 6% shift to protect delicate dyes)
        scale = np.clip(scale, 0.94, 1.06)
        
        for c in range(3):
            result[:, :, c] *= scale[c]
            
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_adaptive_gamma(self, image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Lifts dark midtones adaptively while preserving highlights.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if mask is not None and np.sum(mask > 0) > 100:
            mean_lum = float(np.mean(gray[mask > 0]))
        else:
            mean_lum = float(np.mean(gray))

        # Compute target gamma
        target = self.cfg.target_midtones_luminance
        if mean_lum < 10.0 or mean_lum > 240.0:
            gamma = 1.0
        else:
            gamma = np.log(target / 255.0) / np.log(mean_lum / 255.0)
            gamma = float(np.clip(gamma, 0.85, 1.25))

        if abs(gamma - 1.0) < 0.03:
            return image_bgr, 1.0

        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected = cv2.LUT(image_bgr, table)
        return corrected, gamma

    def apply_bilateral_denoise(self, image_bgr: np.ndarray) -> np.ndarray:
        """Reduces camera sensor noise while keeping boundaries sharp."""
        return cv2.bilateralFilter(
            image_bgr,
            d=self.cfg.denoise_diameter,
            sigmaColor=self.cfg.denoise_sigma_color,
            sigmaSpace=self.cfg.denoise_sigma_space
        )

    def apply_unsharp_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Enhances micro-textures (handloom weave, metal engraving, wood grain)."""
        gaussian = cv2.GaussianBlur(image_bgr, (0, 0), self.cfg.unsharp_radius)
        sharpened = cv2.addWeighted(
            image_bgr,
            1.0 + self.cfg.unsharp_amount,
            gaussian,
            -self.cfg.unsharp_amount,
            0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def enforce_color_fidelity(self, original_bgr: np.ndarray, polished_bgr: np.ndarray, mask: np.ndarray, max_delta: float = 2.5) -> np.ndarray:
        """
        Safety governor: blends polished foreground with original foreground if color shift
        approaches the Delta E00 threshold.
        """
        fg = mask > 128
        if not np.any(fg):
            return polished_bgr

        # Fast LAB delta estimation
        orig_fg = original_bgr[fg].reshape(-1, 1, 3)
        pol_fg = polished_bgr[fg].reshape(-1, 1, 3)
        lab_orig = cv2.cvtColor(orig_fg, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_pol = cv2.cvtColor(pol_fg, cv2.COLOR_BGR2LAB).astype(np.float32)
        current_delta = float(np.mean(np.linalg.norm(lab_orig - lab_pol, axis=2)))

        if current_delta <= max_delta:
            return polished_bgr

        # Attenuate adjustment towards original
        alpha = max_delta / (current_delta + 1e-6)
        blended = cv2.addWeighted(polished_bgr, alpha, original_bgr, 1.0 - alpha, 0)
        return blended

    def process(self, image_bgr: np.ndarray, alpha_mask: np.ndarray) -> PolishResult:
        polished = image_bgr.copy()

        wb_done = False
        if self.cfg.enable_auto_white_balance:
            polished = self.apply_auto_white_balance(polished, alpha_mask)
            wb_done = True

        gamma_applied = 1.0
        if self.cfg.enable_adaptive_gamma:
            # Gentle gamma adjustment
            polished, gamma_applied = self.apply_adaptive_gamma(polished, alpha_mask)

        denoise_done = False
        if self.cfg.enable_bilateral_denoise:
            polished = self.apply_bilateral_denoise(polished)
            denoise_done = True

        unsharp_done = False
        if self.cfg.enable_unsharp_mask:
            polished = self.apply_unsharp_mask(polished)
            unsharp_done = True

        # Enforce Color Fidelity Gate
        polished = self.enforce_color_fidelity(image_bgr, polished, alpha_mask, max_delta=2.2)

        # Recombine into 4-channel cutout
        b, g, r = cv2.split(polished)
        polished_cutout = cv2.merge([b, g, r, alpha_mask])

        return PolishResult(
            polished_bgr=polished,
            polished_cutout=polished_cutout,
            white_balance_applied=wb_done,
            gamma_applied=gamma_applied,
            denoise_applied=denoise_done,
            unsharp_applied=unsharp_done
        )
