"""
BazarSetu V3.0 - Stages C & D: Aspect-Ratio-Safe Segmentation & Matting
Uses BiRefNet-HR in FP16 with letterbox padding to preserve product aspect ratios.
"""

from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CONFIG
from core.device import DEVICE_MGR

logger = logging.getLogger("BazarSetu.Segmentation")

@dataclass
class SegmentationResult:
    mask: np.ndarray       # uint8 0-255 binary/alpha mask of original dimensions
    cutout: np.ndarray     # BGRA 4-channel image
    bounding_box: Tuple[int, int, int, int] # (x, y, w, h)
    inference_time_ms: float
    model_name: str
    model_version: str

def letterbox_pad(image_rgb: np.ndarray, target_size: Tuple[int, int] = (1024, 1024)) -> Tuple[np.ndarray, float, int, int]:
    """Pads image to target_size preserving original aspect ratio."""
    h, w = image_rgb.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = tw - nw
    pad_h = th - nh
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[128, 128, 128]
    )
    return padded, scale, pad_left, pad_top

def unpad_mask(padded_mask: np.ndarray, orig_shape: Tuple[int, int], scale: float, pad_left: int, pad_top: int) -> np.ndarray:
    """Reverts letterbox padding and restores mask to original resolution."""
    orig_h, orig_w = orig_shape
    tw, th = padded_mask.shape[1], padded_mask.shape[0]
    
    nw = int(round(orig_w * scale))
    nh = int(round(orig_h * scale))
    
    cropped = padded_mask[pad_top : pad_top + nh, pad_left : pad_left + nw]
    restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return restored

class BiRefNetSegmenter:
    """
    BiRefNet-HR inference runner with FP16 precision.
    Includes fallback heuristics when model weights are not loaded locally.
    """
    def __init__(self, device: Optional[torch.device] = None, weights_path: Optional[Path] = None):
        self.device = device or DEVICE_MGR.get_device(0)
        self.weights_path = Path(weights_path or CONFIG.models.birefnet_weights_path)
        self.model = None
        self.model_name = "BiRefNet-HR"
        self.model_version = "epoch_244_fp16"
        self._load_model()

    def _load_model(self):
        if not DEVICE_MGR.has_cuda:
            logger.info("CUDA not available; using fallback segmentation.")
            self.model = None
            return

        # Candidate locations for BiRefNet weights
        candidates = [
            self.weights_path,
            Path("/kaggle/working/weights/birefnet"),
            Path("/kaggle/input/birefnet"),
            Path(__file__).resolve().parent.parent / "weights" / "birefnet"
        ]

        # Check if any candidate path exists with model files
        local_target = None
        for cand in candidates:
            if cand.exists():
                if cand.is_file():
                    local_target = cand
                    break
                elif cand.is_dir() and any(cand.glob("*.safetensors")) or any(cand.glob("*.pth")) or any(cand.glob("*.bin")):
                    local_target = cand
                    break

        repo_or_path = str(local_target) if local_target else "ZhengPeng7/BiRefNet"

        try:
            logger.info(f"Loading BiRefNet from '{repo_or_path}' onto {self.device} (FP16)...")
            from transformers import AutoModelForImageSegmentation
            self.model = AutoModelForImageSegmentation.from_pretrained(
                repo_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)
            self.model.eval()
            logger.info(f"BiRefNet-HR successfully loaded in FP16 on {self.device}.")
            self._warmup()
        except Exception as e:
            logger.warning(f"Could not load BiRefNet via transformers: {e}. Trying direct checkpoint...")
            if local_target and Path(local_target).is_file():
                try:
                    self.model = torch.load(local_target, map_location=self.device)
                    self.model = self.model.to(dtype=torch.float16).eval()
                    logger.info("BiRefNet loaded via direct torch checkpoint.")
                    self._warmup()
                except Exception as ex:
                    logger.warning(f"Direct checkpoint failed: {ex}. Using fallback.")
                    self.model = None
            else:
                logger.warning("Using algorithmic fallback segmentation.")
                self.model = None

    def _warmup(self):
        try:
            dummy = torch.randn(1, 3, *CONFIG.models.birefnet_input_size, dtype=torch.float16, device=self.device)
            with torch.inference_mode():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = self.model(dummy)
            torch.cuda.synchronize(self.device)
            logger.info(f"BiRefNet CUDA kernels compiled & cached on {self.device}.")
        except Exception as e:
            logger.debug(f"Warmup exception: {e}")

    def segment(self, image_bgr: np.ndarray) -> SegmentationResult:
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        start_time = time.perf_counter()
        gpu_inference_time_ms = 0.0

        if self.model is not None and DEVICE_MGR.has_cuda:
            # 1. Aspect-Ratio Safe Letterbox Pad
            padded_rgb, scale, pad_left, pad_top = letterbox_pad(image_rgb, CONFIG.models.birefnet_input_size)
            
            # 2. Normalize and prepare Tensor (FP16)
            tensor_in = torch.from_numpy(padded_rgb).permute(2, 0, 1).unsqueeze(0).to(
                device=self.device, dtype=torch.float16
            ) / 255.0
            
            # Standard ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
            tensor_in = (tensor_in - mean) / std

            # 3. Dedicated GPU Forward Pass Timing via CUDA Events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.inference_mode():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(tensor_in)
                    pred = outputs[-1].sigmoid() if isinstance(outputs, (list, tuple)) else outputs.sigmoid()
            end_event.record()
            torch.cuda.synchronize(self.device)
            gpu_inference_time_ms = float(start_event.elapsed_time(end_event))

            pred_np = pred.squeeze().cpu().numpy()

            # 4. Unpad back to original dimensions
            raw_mask = (pred_np * 255.0).astype(np.uint8)
            alpha_mask = unpad_mask(raw_mask, (h, w), scale, pad_left, pad_top)
            
        else:
            # Fallback algorithmic segmentation for environment without weights
            # Uses GrabCut initialized with Otsu foreground bounding box
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Saliency / Central prior
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(c)
                # Expand slightly
                x = max(0, x - 5)
                y = max(0, y - 5)
                bw = min(w - x, bw + 10)
                bh = min(h - y, bh + 10)
            else:
                x, y, bw, bh = int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)

            mask_gc = np.zeros(image_bgr.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (x, y, bw, bh)
            cv2.grabCut(image_bgr, mask_gc, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            alpha_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype('uint8')
            gpu_inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Compute Bounding Box from Alpha Mask
        fg_points = cv2.findNonZero(alpha_mask)
        if fg_points is not None:
            bx, by, bw, bh = cv2.boundingRect(fg_points)
        else:
            bx, by, bw, bh = (0, 0, w, h)

        # Create 4-channel BGRA cutout
        b, g, r = cv2.split(image_bgr)
        cutout_bgra = cv2.merge([b, g, r, alpha_mask])

        return SegmentationResult(
            mask=alpha_mask,
            cutout=cutout_bgra,
            bounding_box=(bx, by, bw, bh),
            inference_time_ms=gpu_inference_time_ms,
            model_name=self.model_name,
            model_version=self.model_version
        )

