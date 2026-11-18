"""
BazarSetu V3.0 - Central Configuration
Optimized for Dual NVIDIA Tesla T4 (Turing, Compute 7.5, FP16)
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple

# Environment detection: Kaggle vs Local
IS_KAGGLE = os.path.exists("/kaggle")
BASE_STORAGE_DIR = Path("/kaggle/working/storage") if IS_KAGGLE else Path(__file__).resolve().parent / "storage"
KAGGLE_INPUT_DIR = Path("/kaggle/input") if IS_KAGGLE else Path(__file__).resolve().parent / "weights"

@dataclass
class QualityGateConfig:
    # Laplacian variance threshold for blur detection
    min_laplacian_variance: float = 100.0
    
    # Exposure thresholds (0-255 luminance scale)
    min_mean_luminance: float = 40.0   # Below this triggers TOO_DARK
    max_mean_luminance: float = 220.0  # Above this triggers TOO_BRIGHT
    max_clipped_highlight_ratio: float = 0.15 # Severe glare
    
    # Framing thresholds
    min_product_coverage_ratio: float = 0.15 # Bbox area / image area < 15% -> TOO_SMALL
    min_edge_margin_px: int = 12             # Distance to image boundary < 12px -> CUT_OFF_FRAME

@dataclass
class ModelConfig:
    # Model Precision: Tesla T4 REQUIRES float16. Never use bfloat16 on Turing (compute 7.5)
    precision: str = "float16"
    
    # BiRefNet settings (896x896 aligns with Tensor Core 32-byte memory tiles on Turing T4)
    birefnet_input_size: Tuple[int, int] = (896, 896)
    birefnet_weights_path: Path = field(
        default_factory=lambda: (
            Path("/kaggle/working/weights/birefnet")
            if IS_KAGGLE and Path("/kaggle/working/weights/birefnet").exists()
            else KAGGLE_INPUT_DIR / "birefnet"
        )
    )
    
    # SAM 2 Fallback settings
    sam2_config_name: str = "sam2.1_hiera_b+.yaml"
    sam2_weights_path: Path = field(default_factory=lambda: KAGGLE_INPUT_DIR / "sam2" / "sam2.1_hiera_base_plus.pt")
    
    # Dual GPU allocation
    gpu_primary: int = 0    # cuda:0
    gpu_secondary: int = 1  # cuda:1

@dataclass
class PolishConfig:
    # Stage F: Deterministic Photographic Polish
    enable_auto_white_balance: bool = True
    white_balance_percentile: float = 95.0
    
    enable_adaptive_gamma: bool = True
    target_midtones_luminance: float = 128.0
    
    enable_bilateral_denoise: bool = True
    denoise_diameter: int = 5
    denoise_sigma_color: float = 40.0
    denoise_sigma_space: float = 40.0
    
    enable_unsharp_mask: bool = True
    unsharp_radius: float = 1.2
    unsharp_amount: float = 0.6

@dataclass
class RenderingConfig:
    # Stage G: Marketplace Grounding & Canvas
    canvas_width: int = 1200
    canvas_height: int = 1200
    canvas_bg_color: Tuple[int, int, int] = (255, 255, 255) # Pure White
    target_visual_dominance: float = 0.72 # 72% coverage (within 60%-80% requirement)
    
    # Grounded mathematical contact occlusion shadow
    shadow_opacity: float = 0.38
    shadow_vertical_scale: float = 0.18 # Flattened ellipse ratio relative to product width
    shadow_horizontal_scale: float = 0.85
    shadow_blur_ksize: int = 31

@dataclass
class VerificationConfig:
    # Stage H: Acceptance gates
    max_delta_e00: float = 3.0       # Color fidelity must be < 3.0
    min_boundary_f1: float = 0.92     # Boundary F1-score >= 0.92
    min_segmentation_iou: float = 0.94 # Segmentation IoU >= 0.94
    max_gpu_latency_ms: float = 400.0 # GPU inference latency < 400 ms
    max_e2e_turnaround_s: float = 3.5  # End-to-end time < 3.5 s

@dataclass
class PipelineConfig:
    storage_root: Path = BASE_STORAGE_DIR
    quality: QualityGateConfig = field(default_factory=QualityGateConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    polish: PolishConfig = field(default_factory=PolishConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

CONFIG = PipelineConfig()

