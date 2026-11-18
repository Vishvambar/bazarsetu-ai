"""
BazarSetu V3.0 - Stage G: Marketplace Grounding & Rendering
Places product on 1200x1200 canvas with 60%-80% dominance and a grounded contact shadow.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from config import RenderingConfig, CONFIG

logger = logging.getLogger("BazarSetu.Rendering")

@dataclass
class RenderingResult:
    marketplace_image: np.ndarray  # 1200x1200 BGR
    thumbnail_image: np.ndarray    # 300x300 BGR
    visual_dominance_ratio: float
    canvas_size: Tuple[int, int]
    shadow_applied: bool

class MarketplaceRenderingStage:
    def __init__(self, config: Optional[RenderingConfig] = None):
        self.cfg = config or CONFIG.rendering

    def _render_contact_shadow(
        self,
        canvas_alpha: np.ndarray,
        base_x: int,
        base_y: int,
        scaled_bw: int
    ):
        """
        Draws a multi-layered mathematical contact occlusion shadow:
        1. Dark, narrow contact core right at the base line.
        2. Softer diffuse ambient occlusion spreading outward.
        """
        h, w = canvas_alpha.shape[:2]
        shadow_map = np.zeros((h, w), dtype=np.float32)

        # 1. Contact core ellipse
        core_w = int(scaled_bw * 0.78)
        core_h = int(max(6, scaled_bw * 0.06))
        cv2.ellipse(
            shadow_map,
            (base_x, base_y - 2),
            (core_w // 2, core_h // 2),
            0, 0, 360,
            0.50, # Core opacity
            -1
        )

        # 2. Diffuse ambient shadow ellipse
        diffuse_w = int(scaled_bw * self.cfg.shadow_horizontal_scale)
        diffuse_h = int(max(14, scaled_bw * self.cfg.shadow_vertical_scale))
        cv2.ellipse(
            shadow_map,
            (base_x, base_y + 4),
            (diffuse_w // 2, diffuse_h // 2),
            0, 0, 360,
            0.25, # Diffuse opacity
            -1
        )

        # Gaussian blur for soft realistic falloff
        ksize = self.cfg.shadow_blur_ksize
        if ksize % 2 == 0:
            ksize += 1
        blurred_shadow = cv2.GaussianBlur(shadow_map, (ksize, ksize), 0)
        
        # Merge with canvas alpha
        return np.clip(blurred_shadow * self.cfg.shadow_opacity, 0.0, 1.0)

    def render(self, cutout_bgra: np.ndarray) -> RenderingResult:
        """
        Composites cutout onto a 1200x1200 white canvas with contact grounding.
        """
        cw = self.cfg.canvas_width
        ch = self.cfg.canvas_height
        
        # 1. Extract tight bounding box of foreground
        alpha = cutout_bgra[:, :, 3]
        pts = cv2.findNonZero(alpha)
        if pts is None:
            # Empty mask fallback
            blank = np.full((ch, cw, 3), 255, dtype=np.uint8)
            thumb = cv2.resize(blank, (300, 300))
            return RenderingResult(blank, thumb, 0.0, (cw, ch), False)

        bx, by, bw, bh = cv2.boundingRect(pts)
        tight_cutout = cutout_bgra[by : by + bh, bx : bx + bw]

        # 2. Scale to target visual dominance (60% to 80% of canvas)
        max_box = int(min(cw, ch) * self.cfg.target_visual_dominance)
        scale = min(max_box / bw, max_box / bh)
        
        nw = int(round(bw * scale))
        nh = int(round(bh * scale))
        scaled_cutout = cv2.resize(tight_cutout, (nw, nh), interpolation=cv2.INTER_AREA)

        # Calculate visual dominance ratio (product area vs canvas area)
        visual_dominance = (nw * nh) / (cw * ch)

        # 3. Compute placement (centered horizontally, grounded vertically)
        pos_x = (cw - nw) // 2
        # Grounding: leave balanced vertical padding
        pos_y = (ch - nh) // 2

        # 4. Create pure white canvas
        canvas = np.full((ch, cw, 3), self.cfg.canvas_bg_color, dtype=np.float32)

        # 5. Generate and blend contact shadow
        base_x = pos_x + (nw // 2)
        base_y = pos_y + nh
        shadow_mask = self._render_contact_shadow(np.zeros((ch, cw), dtype=np.float32), base_x, base_y, nw)

        # Apply dark shadow on white canvas: canvas = canvas * (1 - shadow_alpha)
        for c in range(3):
            canvas[:, :, c] = canvas[:, :, c] * (1.0 - shadow_mask)

        # 6. Composite product cutout over shadow and canvas
        fg_bgr = scaled_cutout[:, :, :3].astype(np.float32)
        fg_alpha = (scaled_cutout[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

        roi = canvas[pos_y : pos_y + nh, pos_x : pos_x + nw]
        canvas[pos_y : pos_y + nh, pos_x : pos_x + nw] = fg_bgr * fg_alpha + roi * (1.0 - fg_alpha)

        final_image = np.clip(canvas, 0, 255).astype(np.uint8)
        
        # 7. Generate Thumbnail (300x300)
        thumbnail = cv2.resize(final_image, (300, 300), interpolation=cv2.INTER_AREA)

        return RenderingResult(
            marketplace_image=final_image,
            thumbnail_image=thumbnail,
            visual_dominance_ratio=visual_dominance,
            canvas_size=(cw, ch),
            shadow_applied=True
        )

