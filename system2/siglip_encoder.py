"""
BazarSetu V3.0 - Shared SigLIP-base-patch16-384 Feature Encoder (GPU 0).

Loads google/siglip-base-patch16-384 once onto cuda:0 (FP16) so both:
  1. Step 5 Two-Stage ONDC Taxonomy Grounder (static ONDC leaf text priors)
  2. Step 6 SigLIP Duplicate Visual Search Index
share a single 768-d embedding backbone without touching GPU 1 (Gemma 4 12B).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

logger = logging.getLogger("BazarSetu.SigLIPEncoder")

DEFAULT_SIGLIP_MODEL = os.getenv("SYSTEM2_SIGLIP_MODEL", "google/siglip-base-patch16-384")
SIGLIP_DIM = 768


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D vector or rows of a 2D matrix into float32."""
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return arr / norms


@dataclass
class SigLIPConfig:
    model_id: str = DEFAULT_SIGLIP_MODEL
    device: Optional[str] = None
    embedding_dim: int = SIGLIP_DIM


class SigLIPEncoder:
    """Thread-safe shared SigLIP encoder pinned to cuda:0 (or CPU when no CUDA)."""

    def __init__(
        self,
        config: Optional[SigLIPConfig] = None,
        lazy_load: bool = True,
        image_encode_fn: Optional[Callable[[Union[str, Path, Image.Image]], np.ndarray]] = None,
        text_encode_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
    ):
        self.cfg = config or SigLIPConfig()
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device: Optional[str] = self.cfg.device
        self._image_encode_fn = image_encode_fn
        self._text_encode_fn = text_encode_fn

        if not lazy_load and self._image_encode_fn is None:
            self.load()

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        if self._model is not None or self._image_encode_fn is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoProcessor

            device = self._resolve_device()
            self._device = device
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            logger.info("Loading SigLIP backbone (%s) on %s (%s)...", self.cfg.model_id, device, dtype)

            self._processor = AutoProcessor.from_pretrained(self.cfg.model_id)
            self._model = AutoModel.from_pretrained(self.cfg.model_id, torch_dtype=dtype)
            self._model.to(device)
            self._model.eval()
            logger.info("SigLIP loaded cleanly on %s.", device)

    def encode_image(self, image_source: Union[str, Path, Image.Image]) -> np.ndarray:
        """Encode an image path or PIL.Image into an L2-normalized float32 vector."""
        if self._image_encode_fn is not None:
            return l2_normalize(np.asarray(self._image_encode_fn(image_source), dtype=np.float32))

        if self._model is None:
            self.load()

        if isinstance(image_source, Image.Image):
            pil_img = image_source.convert("RGB")
        else:
            path = Path(image_source)
            if not path.exists():
                raise FileNotFoundError(f"SigLIP image not found: {path}")
            with Image.open(path) as opened:
                pil_img = opened.convert("RGB")

        import torch

        with self._lock:
            inputs = self._processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            if str(self._device).startswith("cuda") and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            with torch.inference_mode():
                features = self._model.get_image_features(**inputs)
            vec = features.detach().float().cpu().numpy()[0]
        return l2_normalize(vec)

    def encode_texts(self, prompts: Sequence[str]) -> np.ndarray:
        """Encode text prompts into an L2-normalized (N, D) float32 matrix."""
        if not prompts:
            return np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)

        if self._text_encode_fn is not None:
            return l2_normalize(np.asarray(self._text_encode_fn(prompts), dtype=np.float32))

        if self._model is None:
            self.load()

        import torch

        with self._lock:
            inputs = self._processor(text=list(prompts), padding="max_length", truncation=True, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                features = self._model.get_text_features(**inputs)
            mat = features.detach().float().cpu().numpy()
        return l2_normalize(mat)

