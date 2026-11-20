"""
BazarSetu V3.0 - Kaggle API Server & Tunnel Launcher
Exposes the Dual-T4 FastAPI backend to Amar's Flutter application over a secure public HTTPS tunnel.

Supports:
  1. Cloudflare Tunnel (Default - 100% Free, No token or signup required)
  2. ngrok (Optional via --ngrok_token)
"""

from __future__ import annotations
import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
import urllib.request

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BazarSetu.KaggleTunnel")

def download_cloudflared() -> Path:
    """Downloads standalone cloudflared binary if not present."""
    cf_path = Path("/tmp/cloudflared") if os.path.exists("/tmp") else PROJECT_ROOT / "cloudflared"
    if cf_path.exists() and os.access(cf_path, os.X_OK):
        return cf_path

    logger.info("Downloading Cloudflared tunnel binary (standalone Linux x86_64)...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    try:
        urllib.request.urlretrieve(url, cf_path)
        cf_path.chmod(0o755)
        logger.info("Cloudflared downloaded successfully.")
        return cf_path
    except Exception as e:
        logger.error(f"Failed to download cloudflared: {e}")
        raise

def start_uvicorn(port: int = 8000):
    """Starts the Uvicorn ASGI server hosting System 1 & BPP API."""
    import uvicorn
    from api.server import app
    logger.info(f"Starting FastAPI server on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

def launch_cloudflare_tunnel(port: int = 8000) -> str:
    """Spawns cloudflared tunnel and captures public HTTPS URL."""
    cf_bin = download_cloudflared()
    logger.info(f"Opening Cloudflare tunnel to http://127.0.0.1:{port}...")
    
    cmd = [str(cf_bin), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    tunnel_url = None
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    
    start_time = time.time()
    while time.time() - start_time < 35:
        line = process.stdout.readline()
        if not line:
            break
        match = pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if not tunnel_url:
        logger.error("Could not obtain trycloudflare.com tunnel URL.")
        raise RuntimeError("Cloudflare tunnel failed to initialize.")

    return tunnel_url

def launch_ngrok_tunnel(token: str, port: int = 8000) -> str:
    """Spawns ngrok tunnel using pyngrok."""
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(token)
        tunnel = ngrok.connect(port)
        return tunnel.public_url.replace("http://", "https://")
    except ImportError:
        logger.error("pyngrok is not installed. Run 'pip install pyngrok'.")
        raise

def print_banner(public_url: str, port: int):
    print("\n" + "=" * 70)
    print("  🚀 BAZARSETU SYSTEM 01 IS LIVE & READY FOR AMAR'S FLUTTER APP! 🚀")
    print("=" * 70)
    print(f" Public HTTPS URL   : {public_url}")
    print(f" Interactive Docs   : {public_url}/docs")
    print(f" Health Check       : {public_url}/health")
    print(f" BPP Upload Bridge  : {public_url}/api/v1/bpp/upload-image")
    print(f" BPP Studio Image   : {public_url}/api/v1/bpp/studio-image")
    print(f" BPP Publish Route  : {public_url}/api/v1/bpp/publish")
    print("-" * 70)
    print(" 📱 AMAR'S FLUTTER APP (.env) CONFIGURATION:")
    print(" Copy & paste these exact lines into project-zerotype/.env :")
    print("-" * 70)
    print(f" BACKEND_URL={public_url}")
    print(f" BPP_BACKEND_URL={public_url}")
    print(" USE_MOCK_DATA=false")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="BazarSetu V3.0 Kaggle API Tunnel Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Local port for FastAPI server")
    parser.add_argument("--ngrok_token", type=str, default=None, help="Optional ngrok authtoken")
    parser.add_argument("--provider", type=str, choices=["cloudflare", "ngrok"], default="cloudflare")
    args = parser.parse_args()

    # 1. Start Uvicorn in background thread
    server_thread = threading.Thread(target=start_uvicorn, args=(args.port,), daemon=True)
    server_thread.start()
    
    # Wait for server to bind port
    time.sleep(2.5)

    # 2. Establish Tunnel
    if args.provider == "ngrok" or args.ngrok_token:
        if not args.ngrok_token:
            logger.error("--ngrok_token is required when using ngrok provider.")
            sys.exit(1)
        public_url = launch_ngrok_tunnel(args.ngrok_token, args.port)
    else:
        public_url = launch_cloudflare_tunnel(args.port)

    # 3. Print integration banner
    print_banner(public_url, args.port)

    # 4. Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping Kaggle API tunnel server...")

if __name__ == "__main__":
    main()

