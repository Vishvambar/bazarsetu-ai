"""
Offline integration & concurrency tests for Step 7 (api/server.py System 2 routes).

Verifies:
  1. Concurrent /api/v1/system2/analyze calls serialize strictly behind GPU 1 lock (max_in_flight == 1)
  2. Duplicate in-flight analyze calls for the same product_id are deduplicated safely
  3. End-to-end high-res label escalation + Step 5 grounding + Step 6 duplicate search via FastAPI
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, Sequence

import httpx
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.server as api_server
from config import CONFIG
from system2.engine import System2Config, System2Engine
from system2.siglip_encoder import SigLIPEncoder
from system2.taxonomy_grounder import CURATED_ONDC_LEAVES, TaxonomyGrounder
from system2.visual_search import VisualDuplicateIndex


def _vec(seed: int, dim: int = 768) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(dim,)).astype(np.float32)
    return v / float(np.linalg.norm(v))


class System2ApiConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_storage_root = CONFIG.storage_root
        CONFIG.storage_root = Path(self._tmpdir.name)
        api_server.SYSTEM2_JOBS_DB.clear()

        self._orig_s2_engine = api_server.system2_engine
        self._orig_siglip = api_server.siglip_encoder
        self._orig_grounder = api_server.taxonomy_grounder
        self._orig_vindex = api_server.visual_index

    async def asyncTearDown(self) -> None:
        api_server.system2_engine = self._orig_s2_engine
        api_server.siglip_encoder = self._orig_siglip
        api_server.taxonomy_grounder = self._orig_grounder
        api_server.visual_index = self._orig_vindex
        CONFIG.storage_root = self._orig_storage_root
        self._tmpdir.cleanup()

    def _create_product_image(self, product_id: str, make_original: bool = False) -> Path:
        root = CONFIG.storage_root / "products" / product_id
        market_dir = root / "marketplace"
        market_dir.mkdir(parents=True, exist_ok=True)
        img_path = market_dir / "front.png"
        Image.new("RGB", (600, 600), (255, 255, 255)).save(img_path)
        if make_original:
            orig_dir = root / "original"
            orig_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1200, 900), (255, 255, 255)).save(orig_dir / "raw.jpg")
        return img_path

    async def test_concurrent_analyze_serializes_gpu1_and_runs_escalation_and_search(self) -> None:
        counter_lock = threading.Lock()
        state = {"in_flight": 0, "max_in_flight": 0, "calls": 0}
        leaf_vecs = {
            leaf.prompt_text: _vec(100 + idx) for idx, leaf in enumerate(CURATED_ONDC_LEAVES)
        }
        shared_img_vec = leaf_vecs[CURATED_ONDC_LEAVES[0].prompt_text].copy()

        def fake_text_encode(prompts: Sequence[str]) -> np.ndarray:
            return np.stack([leaf_vecs[p] for p in prompts], axis=0)

        def fake_image_encode(_src: Any) -> np.ndarray:
            return shared_img_vec.copy()

        fake_siglip = SigLIPEncoder(
            lazy_load=True,
            image_encode_fn=fake_image_encode,
            text_encode_fn=fake_text_encode,
        )
        fake_grounder = TaxonomyGrounder(encoder=fake_siglip, preload_priors=True)
        fake_vindex = VisualDuplicateIndex(
            index_path=CONFIG.storage_root / "system2_visual_index.json",
            encoder=fake_siglip,
        )

        def fake_completion(**kwargs: Any) -> Dict[str, Any]:
            with counter_lock:
                state["in_flight"] += 1
                state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
                state["calls"] += 1
            try:
                time.sleep(0.05)
                user_text = ""
                for part in kwargs["messages"][1]["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_text += part.get("text", "")
                is_esc = "escalation_crop_view" in user_text
                trigger_esc = ("prod_esc" in user_text) and not is_esc
                payload = {
                    "product": "Copper Kalash",
                    "title": "Hand-Hammered Copper Kalash Vessel",
                    "desc": "Handcrafted copper ceremonial kalash made by Pune artisans.",
                    "comp": "single_item",
                    "scale": "tabletop",
                    "domain": "ONDC:RET16",
                    "cat": "Metalware",
                    "subcat": "Hammered Copper Kalash",
                    "color": "Burnished Copper",
                    "color_fam": "Orange",
                    "finish": "hammered_mathar",
                    "mat": "Copper",
                    "mat_fam": "Metal",
                    "craft": ["hand_hammered_mathar"],
                    "seals": ["GI Tag"] if is_esc else [],
                    "text": ("prod_esc" in user_text),
                    "legib": 0.95 if is_esc else (0.55 if trigger_esc else 1.0),
                    "crop": trigger_esc,
                    "conf": 0.94,
                    "review": [],
                }
                return {"choices": [{"message": {"content": json.dumps(payload)}}]}
            finally:
                with counter_lock:
                    state["in_flight"] -= 1

        api_server.system2_engine = System2Engine(
            config=System2Config(), lazy_load=True, completion_fn=fake_completion
        )
        api_server.siglip_encoder = fake_siglip
        api_server.taxonomy_grounder = fake_grounder
        api_server.visual_index = fake_vindex

        for pid in ("prod_1", "prod_2", "prod_esc"):
            self._create_product_image(pid, make_original=(pid == "prod_esc"))

        transport = httpx.ASGITransport(app=api_server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Fire 3 overlapping analyze requests simultaneously + 1 duplicate for prod_1
            r1, r2, r3, r1_dup = await asyncio.gather(
                client.post(
                    "/api/v1/system2/analyze",
                    json={"product_id": "prod_1", "translate_home_decor_for_gateway": True},
                ),
                client.post(
                    "/api/v1/system2/analyze",
                    json={"product_id": "prod_2", "translate_home_decor_for_gateway": True},
                ),
                client.post(
                    "/api/v1/system2/analyze",
                    json={"product_id": "prod_esc", "translate_home_decor_for_gateway": True},
                ),
                client.post(
                    "/api/v1/system2/analyze",
                    json={"product_id": "prod_1", "translate_home_decor_for_gateway": True},
                ),
            )
            self.assertEqual(r1.status_code, 202)
            self.assertEqual(r2.status_code, 202)
            self.assertEqual(r3.status_code, 202)
            self.assertEqual(r1_dup.status_code, 202)
            self.assertTrue(r1_dup.json()["deduplicated"])

            # 2. Poll until all 3 complete
            statuses: Dict[str, Dict[str, Any]] = {}
            deadline = time.monotonic() + 10.0
            while len(statuses) < 3 and time.monotonic() < deadline:
                await asyncio.sleep(0.03)
                for pid in ("prod_1", "prod_2", "prod_esc"):
                    if pid in statuses:
                        continue
                    resp = await client.get(f"/api/v1/system2/status/{pid}")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()
                    if body["status"] in ("completed", "failed"):
                        statuses[pid] = body

            self.assertEqual(len(statuses), 3)
            for pid, body in statuses.items():
                self.assertEqual(body["status"], "completed", msg=f"{pid} error={body.get('error')}")
                self.assertEqual(body["export_domain_code"], "ONDC:RET15")
                meta_file = CONFIG.storage_root / "products" / pid / "metadata" / "system2_intelligence.json"
                self.assertTrue(meta_file.exists())

            # Strict concurrency guarantee: never more than 1 call in flight on GPU 1
            self.assertEqual(state["max_in_flight"], 1)
            # 3 normal passes + 1 high-res escalation pass for prod_esc = 4 total LLM calls
            self.assertEqual(state["calls"], 4)

            # Confirm high-res escalation merged GI Tag on prod_esc
            esc_body = statuses["prod_esc"]
            self.assertTrue(esc_body["escalated"])
            self.assertIn("GI Tag", esc_body["result"]["text_and_seals"]["certification_seals"])

            # Confirm duplicate search finds previously indexed matching products
            dup_resp = await client.post(
                "/api/v1/system2/search-duplicates",
                json={"product_id": "prod_1", "threshold": 0.90, "top_k": 5},
            )
            self.assertEqual(dup_resp.status_code, 200)
            dup_data = dup_resp.json()
            self.assertTrue(dup_data["has_duplicates"])
            returned_ids = {m["product_id"] for m in dup_data["matches"]}
            self.assertNotIn("prod_1", returned_ids)
            self.assertTrue("prod_2" in returned_ids or "prod_esc" in returned_ids)


if __name__ == "__main__":
    unittest.main()

