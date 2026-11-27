#!/bin/bash
# BazarSetu V3.0 - Kaggle Environment Initialization (Dual Tesla T4)
set -e

echo "=== Initializing BazarSetu V3.0 on Kaggle (Dual Tesla T4) ==="
nvidia-smi

echo "=== Checking Python & CUDA Compiler ==="
python3 --version
which nvcc && nvcc --version || echo "WARNING: nvcc not in PATH"

echo "=== Installing System 1, SigLIP & FastAPI Dependencies ==="
pip install -q --upgrade pip
pip install -q einops timm scikit-image opencv-python-headless pydantic fastapi uvicorn python-multipart huggingface_hub transformers sentencepiece protobuf

echo "=== Ensuring llama-cpp-python with CUDA Support (Prebuilt Wheel) ==="
if ! python3 -c '
import ctypes
from pathlib import Path
import llama_cpp
try:
    from llama_cpp._ggml import ggml_backend_load_all_from_path
    llama_cpp.llama_backend_init()
    lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
    if lib_dir.exists():
        ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))
except ImportError:
    pass
try:
    from llama_cpp.llama_multimodal import Gemma4ChatHandler
except ImportError:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
assert llama_cpp.llama_supports_gpu_offload()
' 2>/dev/null; then
    WHEEL_URL=$(python3 -c '
import json, sys, urllib.request, subprocess, re

py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
cuda_tag = "cu130"
try:
    out = subprocess.check_output(["nvcc", "--version"], text=True)
    m = re.search(r"release (\d+)\.(\d+)", out)
    if m:
        maj, min_v = int(m.group(1)), int(m.group(2))
        if maj == 13:
            cuda_tag = "cu130" if min_v == 0 else f"cu13{min_v}"
        elif maj == 12:
            cuda_tag = f"cu12{min_v}"
except Exception:
    pass

# Query JamePeng releases for exact cp313/cp312/cp311/cp310 linux_x86_64 CUDA wheel
fallback_tags = [cuda_tag, "cu130", "cu128", "cu126", "cu124"]
try:
    req = urllib.request.Request(
        "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=10",
        headers={"User-Agent": "bazarsetu-setup"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        releases = json.loads(resp.read().decode("utf-8"))
    for tag_want in fallback_tags:
        found = ""
        for rel in releases:
            for asset in rel.get("assets", []):
                name = asset.get("name", "")
                if py_tag in name and "linux_x86_64" in name and tag_want in name and name.endswith(".whl"):
                    found = asset["browser_download_url"]
                    break
            if found:
                print(found)
                sys.exit(0)
except Exception:
    pass

if py_tag == "cp313":
    print("https://github.com/JamePeng/llama-cpp-python/releases/download/v0.4.8-cu130-linux-20261122/llama_cpp_python-0.4.8%2Bcu130-cp313-cp313-linux_x86_64.whl")
')

    if [ -n "$WHEEL_URL" ]; then
        echo "Installing prebuilt CUDA wheel: $WHEEL_URL"
        pip install -q --no-cache-dir --force-reinstall "$WHEEL_URL"
    else
        echo "No prebuilt wheel matched; falling back to abetlen cu124 / source CUDA build..."
        pip install -q --only-binary=:all: --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
          || CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" FORCE_CMAKE=1 pip install --no-cache-dir --force-reinstall llama-cpp-python
    fi
fi

echo "=== Verifying Gemma4ChatHandler, GPU Offload & CUDA Devices ==="
python3 -c '
import ctypes
from pathlib import Path
import torch
import llama_cpp
try:
    from llama_cpp._ggml import ggml_backend_load_all_from_path
    llama_cpp.llama_backend_init()
    lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
    if lib_dir.exists():
        ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))
except ImportError:
    pass
try:
    from llama_cpp.llama_multimodal import Gemma4ChatHandler
except ImportError:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
assert llama_cpp.llama_supports_gpu_offload(), "FATAL: llama-cpp-python lacks CUDA GPU offload!"
print("llama-cpp-python version:", getattr(llama_cpp, "__version__", "unknown"))
print("llama_supports_gpu_offload:", llama_cpp.llama_supports_gpu_offload())
print("Gemma4ChatHandler available:", Gemma4ChatHandler.__name__)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB)")
'

echo "=== Setup Complete. Ready for Inference ==="
