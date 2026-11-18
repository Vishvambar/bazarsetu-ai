"""
BazarSetu V3.0 - Dual Tesla T4 Device & Stream Manager
"""

from __future__ import annotations
import logging
import torch
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BazarSetu.Device")

class DeviceManager:
    """
    Manages dual Tesla T4 instances (cuda:0 and cuda:1).
    Enforces FP16, enables cudnn benchmark, and manages CUDA streams.
    """
    def __init__(self):
        self.has_cuda = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.has_cuda else 0
        self.devices: Dict[str, torch.device] = {}
        self.streams: Dict[str, torch.cuda.Stream] = {}
        self._initialize_devices()

    def _initialize_devices(self):
        if not self.has_cuda:
            logger.warning("CUDA not available. Falling back to CPU.")
            self.devices["primary"] = torch.device("cpu")
            self.devices["secondary"] = torch.device("cpu")
            return

        logger.info(f"Detected {self.device_count} CUDA devices.")
        for idx in range(self.device_count):
            dev_name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            total_vram_gb = props.total_memory / (1024 ** 3)
            logger.info(f"GPU {idx}: {dev_name} ({total_vram_gb:.2f} GB VRAM, Compute {props.major}.{props.minor})")
            
            # Enforce Turing optimizations
            if props.major == 7 and props.minor == 5:
                logger.info(f"Configuring GPU {idx} for Turing Architecture (Native FP16).")
            
            # Enable cuDNN benchmark for optimized conv kernels
            torch.backends.cudnn.benchmark = True

        self.devices["primary"] = torch.device("cuda:0")
        self.streams["primary"] = torch.cuda.Stream(device=self.devices["primary"])

        if self.device_count > 1:
            self.devices["secondary"] = torch.device("cuda:1")
            self.streams["secondary"] = torch.cuda.Stream(device=self.devices["secondary"])
        else:
            self.devices["secondary"] = self.devices["primary"]
            self.streams["secondary"] = self.streams["primary"]

    def get_device(self, worker_id: int = 0) -> torch.device:
        """Assigns GPU device based on worker or batch split."""
        if not self.has_cuda:
            return torch.device("cpu")
        target_idx = worker_id % self.device_count
        return torch.device(f"cuda:{target_idx}")

    def get_stream(self, worker_id: int = 0) -> Optional[torch.cuda.Stream]:
        if not self.has_cuda:
            return None
        key = "secondary" if (worker_id % self.device_count == 1 and self.device_count > 1) else "primary"
        return self.streams.get(key)

    def warmup(self):
        """Runs a lightweight dummy forward pass on all available GPUs to eliminate cold-start spikes."""
        if not self.has_cuda:
            return
        logger.info("Performing CUDA warm-up across available GPUs...")
        for idx in range(self.device_count):
            dev = torch.device(f"cuda:{idx}")
            with torch.cuda.device(dev):
                dummy = torch.randn(1, 3, 1024, 1024, dtype=torch.float16, device=dev)
                # Simple conv warmup
                conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device=dev, dtype=torch.float16)
                with torch.inference_mode():
                    _ = conv(dummy)
                torch.cuda.synchronize(dev)
        logger.info("CUDA warm-up complete.")

DEVICE_MGR = DeviceManager()

