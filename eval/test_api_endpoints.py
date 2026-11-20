"""
BazarSetu V3.0 - Comprehensive API Endpoints Integration Test
Validates all native System 1 and Flutter BPP routes.
"""

import io
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from fastapi.testclient import TestClient
from api.server import app

def create_dummy_image_bytes():
    # 300x300 sample clay pot on neutral background
    img = np.full((300, 300, 3), (200, 205, 210), dtype=np.uint8)
    cv2.ellipse(img, (150, 160), (70, 90), 0, 0, 360, (40, 75, 175), -1)
    is_success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer.tobytes())

def run_tests():
    client = TestClient(app)
    print("=== 1. Testing Root & Health Checks ===")
    r = client.get("/")
    assert r.status_code == 200, f"Root failed: {r.text}"
    print("GET / ->", r.json()["service"])

    r = client.get("/health")
    assert r.status_code == 200
    print("GET /health ->", r.json()["status"])

    r = client.get("/api/v1/bpp/health")
    assert r.status_code == 200
    print("GET /api/v1/bpp/health ->", r.json())

    print("\n=== 2. Testing Synchronous Studio Image Generation ===")
    img_bytes = create_dummy_image_bytes()
    r = client.post(
        "/api/v1/bpp/studio-image",
        files={"file": ("pot.jpg", img_bytes, "image/jpeg")},
        data={"catalog_id": "test_craft_001", "force": "true"}
    )
    assert r.status_code == 200, f"Studio image failed: {r.text}"
    data = r.json()
    print("POST /api/v1/bpp/studio-image -> status:", data["status"])
    print("  Studio URL:", data["studio_image_url"])
    print("  Thumbnail URL:", data["thumbnail_url"])
    print("  Bounding Box:", data["bounding_box"])

    print("\n=== 3. Testing Flutter upload-image Route (Studio Mode) ===")
    img_bytes = create_dummy_image_bytes()
    r = client.post(
        "/api/v1/bpp/upload-image",
        files={"file": ("raw_camera.jpg", img_bytes, "image/jpeg")},
        data={"catalog_id": "test_craft_001", "image_type": "studio", "force": "true"},
        headers={"bypass-tunnel-reminder": "true"}
    )
    assert r.status_code == 200, f"upload-image failed: {r.text}"
    data = r.json()
    print("POST /api/v1/bpp/upload-image (studio) -> status:", data["status"])
    print("  Public URL (for Flutter Image.network):", data["url"])
    assert data["url"].startswith("http"), "URL must be absolute for Flutter"

    print("\n=== 4. Testing Flutter upload-image Route (Raw Mode) ===")
    img_bytes = create_dummy_image_bytes()
    r = client.post(
        "/api/v1/bpp/upload-image",
        files={"file": ("raw_camera.jpg", img_bytes, "image/jpeg")},
        data={"catalog_id": "test_craft_001", "image_type": "raw", "force": "true"}
    )
    assert r.status_code == 200
    data = r.json()
    print("POST /api/v1/bpp/upload-image (raw) ->", data["url"])

    print("\n=== 5. Testing ONDC BPP Publish (ONDC par Prakashan Karein) ===")
    payload = {
        "catalog_id": "test_craft_001",
        "artisan_name": "Savita Patil",
        "cluster_name": "Pune Pottery Cluster",
        "title_en": "Handmade Terracotta Water Pot",
        "title_hi": "हाथ से बना मिट्टी का घड़ा",
        "description_en": "Authentic terracotta water pot made from natural clay.",
        "description_hi": "प्राकृतिक मिट्टी से बना पारंपरिक घड़ा।",
        "craft_category": "Terracotta Pottery",
        "category_id": "ONDC:RET16",
        "materials": ["Natural Clay", "Terracotta"],
        "dimensions": "10x10 inches",
        "floor_price": 200.0,
        "fair_price": 350.0,
        "premium_price": 450.0,
        "raw_image_url": "http://testserver/storage/raw.jpg",
        "studio_image_url": "http://testserver/storage/studio.webp",
        "gallery_images": [],
        "available_qty": 5
    }
    r = client.post("/api/v1/bpp/publish", json=payload)
    assert r.status_code == 200, f"Publish failed: {r.text}"
    print("POST /api/v1/bpp/publish ->", r.json()["message"])

    print("\n=== 6. Testing Listings Retrieval ===")
    r = client.get("/api/v1/bpp/listings")
    assert r.status_code == 200
    listings_data = r.json()
    print(f"GET /api/v1/bpp/listings -> {listings_data['count']} active listing(s)")

    r = client.get("/api/v1/bpp/listings/test_craft_001")
    assert r.status_code == 200
    print("GET /api/v1/bpp/listings/test_craft_001 -> Title:", r.json()["title_en"])

    print("\n=== 7. Testing Seller Approval Handshake ('Sahi Hai') ===")
    r = client.post(
        "/api/v1/bpp/products/test_craft_001/approve",
        json={"seller_approved": True, "feedback_note": "Sahi Hai - Approved by artisan"}
    )
    assert r.status_code == 200
    print("POST /api/v1/bpp/products/test_craft_001/approve ->", r.json()["status"])

    print("\n=== 8. Testing Async System 1 Job Upload ===")
    img_bytes = create_dummy_image_bytes()
    r = client.post(
        "/v1/products/upload",
        files={"file": ("pot.jpg", img_bytes, "image/jpeg")},
        data={"product_id": "test_async_002", "force": "true"}
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    print("POST /v1/products/upload (HTTP 202) -> Job ID:", job_id)

    r = client.get(f"/v1/jobs/{job_id}")
    assert r.status_code == 200
    print(f"GET /v1/jobs/{job_id} -> Status:", r.json()["status"])

    print("\n" + "=" * 50)
    print(" ALL API INTEGRATION TESTS PASSED (100% GREEN) ")
    print("=" * 50)

if __name__ == "__main__":
    run_tests()
