#!/bin/bash
# BazarSetu V3.0 - Kaggle Environment Initialization (Dual Tesla T4)
set -e

echo "=== Initializing BazarSetu V3.0 on Kaggle (Dual Tesla T4) ==="
nvidia-smi

echo "=== Checking Python & CUDA Compiler ==="
python3 --version
which nvcc && nvcc --version || echo "WARNING: nvcc not in PATH"

echo "=== Installing System 1 + FastAPI Dependencies ==="
pip install -q --upgrade pip
pip install -q einops timm scikit-image opencv-python-headless pydantic fastapi uvicorn python-multipart huggingface_hub

echo "=== Ensuring llama-cpp-python with CUDA Support (Turing sm_75) ==="
if ! python3 -c "import llama_cpp; from llama_cpp.llama_chat_format import Gemma4ChatHandler; assert llama_cpp.llama_supports_gpu_offload()" 2>/dev/null; then
    echo "Installing llama-cpp-python with CUDA enabled (skipping prebuilt CPU-only fallback)..."
    # Only try prebuilt wheel if Python <= 3.12; Python 3.13 compiles CUDA natively for Turing (sm_75)
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    INSTALLED_WHEEL=0
    if [ "$PY_MINOR" -le 12 ]; then
        if pip install -q --only-binary=:all: --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 2>/dev/null; then
            if python3 -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload()" 2>/dev/null; then
                INSTALLED_WHEEL=1
            fi
        fi
    fi

    if [ "$INSTALLED_WHEEL" -eq 0 ]; then
        echo "Building llama-cpp-python from source with GGML_CUDA=ON for Tesla T4 (sm_75)..."
        CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" FORCE_CMAKE=1 \
            pip install --no-cache-dir --force-reinstall llama-cpp-python
    fi
fi

echo "=== Verifying Gemma4ChatHandler, GPU Offload & CUDA Devices ==="
python3 -c '
import torch
import llama_cpp
from llama_cpp.llama_chat_format import Gemma4ChatHandler
assert llama_cpp.llama_supports_gpu_offload(), "FATAL: llama-cpp-python built without CUDA GPU offload!"
print("llama_supports_gpu_offload:", llama_cpp.llama_supports_gpu_offload())
print("Gemma4ChatHandler available:", Gemma4ChatHandler.__name__)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB)")
'

echo "=== Setup Complete. Ready for Inference ==="
