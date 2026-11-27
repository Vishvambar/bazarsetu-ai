"""
BazarSetu V3.0 - End-to-End Kaggle Dual-T4 Live Verification (Steps 5, 6 & 7).

Verifies on live GPU hardware:
  1. GPU 0 (cuda:0) SigLIP-base-patch16-384 FP16 text priors + image embeddings
  2. Step 5 Two-Stage ONDC Taxonomy Grounding (RET12 / RET16 -> RET15 export)
  3. Step 6 Duplicate Search + Gemma-Native High-Res Label Escalation
  4. Step 7 Thread-Safe FastAPI /api/v1/system2/analyze + /status + /search-duplicates
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.server as api_server
from config import CONFIG
from eval.prepare_system2_golden import find_restore_archive, restore_golden_zip
from system2.engine import System2Config, System2Engine
from system2.siglip_encoder import SigLIPEncoder
from system2.taxonomy_grounder import TaxonomyGrounder
from system2.visual_search import VisualDuplicateIndex


async def run_live_check() -> int:
    golden_dir = PROJECT_ROOT / "storage" / "golden"
    archive = find_restore_archive(None)
    if archive:
        restore_golden_zip(archive, golden_dir)

    sample1 = golden_dir / "metalware6_front.png"
    sample2 = golden_dir / "seal4_front.png"
    if not sample1.exists() or not sample2.exists():
        print(f"Missing golden samples in {golden_dir}. Run prepare_system2_golden.py first.")
        return 1

    with tempfile.TemporaryDirectory() as td:
        orig_root = CONFIG.storage_root
        CONFIG.storage_root = Path(td)
        api_server.SYSTEM2_JOBS_DB.clear()

        try:
            t0 = time.perf_counter()
            siglip = SigLIPEncoder(lazy_load=False)
            grounder = TaxonomyGrounder(encoder=siglip, preload_priors=True)
            vindex = VisualDuplicateIndex(
                index_path=CONFIG.storage_root / "system2_visual_index.json",
                encoder=siglip,
            )
            s2_engine = System2Engine(config=System2Config(verbose=False))
            print(f"[INIT] SigLIP (cuda:0) + Gemma 4 12B (cuda:1) ready in {time.perf_counter() - t0:.2f}s")

            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    free_b, total_b = torch.cuda.mem_get_info(idx)
                    print(
                        f"  cuda:{idx} -> {(total_b - free_b) / (1024**3):.2f} / {total_b / (1024**3):.2f} GB used"
                    )

            api_server.system2_engine = s2_engine
            api_server.siglip_encoder = siglip
            api_server.taxonomy_grounder = grounder
            api_server.visual_index = vindex

            transport = httpx.ASGITransport(app=api_server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                r1, r2 = await asyncio.gather(
                    client.post(
                        "/api/v1/system2/analyze",
                        json={
                            "product_id": "live_kalash",
                            "views": {"front_view": str(sample1)},
                            "translate_home_decor_for_gateway": True,
                        },
                    ),
                    client.post(
                        "/api/v1/system2/analyze",
                        json={
                            "product_id": "live_seal_jar",
                            "views": {"front_view": str(sample2)},
                        },
                    ),
                )
                assert r1.status_code == 202 and r2.status_code == 202

                done = {}
                deadline = time.monotonic() + 60.0
                while len(done) < 2 and time.monotonic() < deadline:
                    await asyncio.sleep(0.25)
                    for pid in ("live_kalash", "live_seal_jar"):
                        if pid in done:
                            continue
                        resp = await client.get(f"/api/v1/system2/status/{pid}")
                        body = resp.json()
                        if body["status"] in ("completed", "failed"):
                            done[pid] = body

                for pid, body in done.items():
                    assert body["status"] == "completed", f"{pid} failed: {body.get('error')}"
                    res = body["result"]
                    print(
                        f"[RESULT {pid}] domain={res['ondc_taxonomy']['domain_code']} "
                        f"export={body['export_domain_code']} "
                        f"cat={res['ondc_taxonomy']['category']} "
                        f"seals={res['text_and_seals']['certification_seals']} "
                        f"escalated={body['escalated']} review={body['needs_seller_review']}"
                    )

                dup = await client.post(
                    "/api/v1/system2/search-duplicates",
                    json={"image_path": str(sample1), "threshold": 0.90, "top_k": 3},
                )
                assert dup.status_code == 200
                dup_json = dup.json()
                print(
                    f"[DUP SEARCH] has_duplicates={dup_json['has_duplicates']} "
                    f"top={dup_json['matches'][:1]}"
                )
                assert dup_json["has_duplicates"]
                assert dup_json["matches"][0]["product_id"] == "live_kalash"
        finally:
            CONFIG.storage_root = orig_root

    print("=== LIVE SYSTEM 2 FULL-STACK (STEPS 5-7) PASSED ===")
    return 0


def main() -> int:
    return asyncio.run(run_live_check())


if __name__ == "__main__":
    raise SystemExit(main())
