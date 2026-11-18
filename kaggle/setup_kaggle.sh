#!/bin/bash
# BazarSetu V3.0 - Kaggle Environment Initialization
set -e

echo "=== Initializing BazarSetu V3.0 on Kaggle (Dual Tesla T4) ==="
nvidia-smi

echo "=== Checking Python Environment ==="
python3 --version

echo "=== Installing Dependencies ==="
pip install -q --upgrade pip
pip install -q einops timm scikit-image opencv-python-headless pydantic

echo "=== Verifying CUDA & Torch Devices ==="
python3 -c '
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB)")
'

echo "=== Setup Complete. Ready for Inference ==="

