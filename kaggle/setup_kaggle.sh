#!/bin/bash
# BazarSetu V3.0 - Kaggle Environment Initialization (Dual Tesla T4)
set -e

echo "=== Initializing BazarSetu V3.0 on Kaggle (Dual Tesla T4) ==="
nvidia-smi

echo "=== Checking Python Environment ==="
python3 --version

echo "=== Installing System 1 + FastAPI Dependencies ==="
pip install -q --upgrade pip
pip install -q einops timm scikit-image opencv-python-headless pydantic fastapi uvicorn python-multipart huggingface_hub

echo "=== Installing llama-cpp-python (Prebuilt CUDA 12 Wheel for Gemma 4) ==="
if ! python3 -c "from llama_cpp.llama_chat_format import Gemma4ChatHandler" 2>/dev/null; then
    pip install -q --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
      || pip install -q --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125 \
      || CMAKE_ARGS="-DGGML_CUDA=on" pip install -q --upgrade --no-cache-dir llama-cpp-python
fi

echo "=== Verifying Gemma4ChatHandler & CUDA Devices ==="
python3 -c '
import torch
from llama_cpp.llama_chat_format import Gemma4ChatHandler
print("Gemma4ChatHandler available:", Gemma4ChatHandler.__name__)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB)")
'

echo "=== Setup Complete. Ready for Inference ==="
