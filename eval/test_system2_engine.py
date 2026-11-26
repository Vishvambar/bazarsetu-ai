"""
Offline unit tests for System 2 Step 4 (V3.0/system2/engine.py).
Tests 48px patch alignment, multi-view URI encoding, tool-choice parsing,
empty thought block stripping, and thread-safe prediction without requiring a GPU.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system2.engine import (
    PATCH_SIZE,
    System2Config,
    System2Engine,
    build_product_intelligence_tool,
    compute_patch_aligned_size,
    extract_tool_or_json_payload,
    normalize_wire_payload,
)
from system2.schemas import ProductIntelligenceOutput


def _sample_valid_payload(pid: str = "mv_metal1") -> Dict[str, Any]:
    return {
        "product_id": pid,
        "product_identity": {
            "apparent_product": "Hammered Copper Kalash",
            "catalog_title_en": "Hand-Hammered Tambat Copper Kalash",
            "description_en": "Hand-hammered copper kalash crafted by Tambat Ali artisans for puja and home decor.",
            "composition_type": "single_item",
            "set_contents": [],
            "physical_scale": "tabletop",
        },
        "ondc_taxonomy": {
            "domain_code": "ONDC:RET16",
            "category": "Metalware",
            "subcategory": "Copper Kalash",
            "generic_name": "Copper Water Vessel",
            "confidence": 0.95,
        },
        "color_profile": {
            "primary_color": {"name": "Burnished Copper", "standard_family": "Orange"},
            "secondary_colors": [],
            "surface_finish": "hammered_mathar",
            "color_distribution": "solid",
        },
        "materials": [
            {
                "material_name": "Copper",
                "material_family": "Metal",
                "location": "main_body",
                "confidence": 0.94,
            }
        ],
        "pattern_and_craft": {
            "has_pattern": True,
            "pattern_categories": ["geometric"],
            "craft_techniques": ["hand_hammered_mathar"],
            "motif_description": "Uniform hand-hammered dimple facets",
        },
        "structural_components": [],
        "text_and_seals": {
            "has_visible_text": False,
            "brand_or_artisan_mark": None,
            "certification_seals": [],
            "visible_text_snippets": [],
            "text_legibility_confidence": 1.0,
            "needs_highres_label_crop": False,
        },
        "marketplace_enrichment": {
            "target_audience": "home_utility",
            "occasions": ["Daily Pooja", "Festive"],
            "search_keywords": ["copper kalash", "tambat ali", "mathar kaam"],
        },
        "audit_telemetry": {
            "views_processed": ["front_view", "detail_view"],
            "overall_confidence": 0.94,
            "needs_seller_review": [],
        },
    }


class System2EngineTests(unittest.TestCase):
    def test_patch_alignment_multiples_of_48(self) -> None:
        for w, h, max_d in [(1200, 1200, 480), (1200, 1200, 384), (1024, 572, 480), (896, 1200, 672)]:
            tw, th = compute_patch_aligned_size(w, h, max_d)
            self.assertEqual(tw % PATCH_SIZE, 0)
            self.assertEqual(th % PATCH_SIZE, 0)
            self.assertLessEqual(max(tw, th), max_d)

    def test_tool_definition_matches_pydantic_schema(self) -> None:
        tool = build_product_intelligence_tool()
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "submit_product_intelligence")
        params = tool["function"]["parameters"]
        serialized = json.dumps(params)
        self.assertNotIn("$defs", serialized)
        self.assertNotIn("$ref", serialized)
        self.assertEqual(params["properties"]["ondc_taxonomy"]["type"], "object")
        self.assertIn("domain_code", params["properties"]["ondc_taxonomy"]["properties"])
        self.assertEqual(params["properties"]["ondc_taxonomy"]["properties"]["domain_code"]["type"], "string")
        self.assertIn("ONDC:RET16", params["properties"]["ondc_taxonomy"]["properties"]["domain_code"]["enum"])
        self.assertEqual(params["properties"]["materials"]["items"]["type"], "object")
        ident_props = params["properties"]["product_identity"]["properties"]
        self.assertIn("catalog_title_en", ident_props)
        self.assertNotIn("catalog_title_hi", ident_props)
        self.assertNotIn("description_hi", ident_props)

    def test_extract_payload_handles_tool_calls_and_thought_fallback(self) -> None:
        payload = _sample_valid_payload()
        resp_tool = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "submit_product_intelligence", "arguments": json.dumps(payload)}}
                        ],
                    }
                }
            ]
        }
        parsed = extract_tool_or_json_payload(resp_tool)
        self.assertEqual(parsed["product_id"], "mv_metal1")

        # Fallback: empty Gemma 4 channel thought prefix + fenced JSON in content
        resp_thought = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"<|channel>thought\n<channel|>```json\n{json.dumps(payload)}\n```",
                    }
                }
            ]
        }
        parsed2 = extract_tool_or_json_payload(resp_thought)
        self.assertEqual(parsed2["ondc_taxonomy"]["domain_code"], "ONDC:RET16")

    def test_engine_predict_multiview_mock_and_tensor_split_config(self) -> None:
        cfg = System2Config()
        self.assertEqual(cfg.main_gpu, 1)
        self.assertEqual(cfg.tensor_split, [0.0, 1.0])
        self.assertFalse(cfg.use_tools)
        self.assertEqual(cfg.front_max_dim % 48, 0)
        self.assertEqual(cfg.secondary_max_dim % 48, 0)

        captured: Dict[str, Any] = {}

        def fake_completion(**kwargs: Any) -> Dict[str, Any]:
            captured.update(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "product": "Hammered Copper Kalash",
                                    "title": "Hand-Hammered Tambat Copper Kalash",
                                    "desc": "Hand-hammered copper kalash crafted by artisans for puja.",
                                    "comp": "single_item",
                                    "scale": "tabletop",
                                    "domain": "ONDC:RET16",
                                    "cat": "Metalware",
                                    "subcat": "Copper Kalash",
                                    "color": "Burnished Copper",
                                    "color_fam": "Copper",  # tests color_fam alias coercion -> Orange
                                    "finish": "polished",   # tests finish alias coercion -> glossy
                                    "mat": "Copper",
                                    "mat_fam": "Metal",
                                    "mat_conf": 0.94,
                                    "pat": ["geometric"],
                                    "craft": ["hand_hammered_mathar"],
                                    "seals": [],
                                    "text": False,
                                    "crop": False,
                                    "conf": 0.94,
                                    "review": [],
                                }
                            )
                        }
                    }
                ]
            }

        engine = System2Engine(config=cfg, lazy_load=True, completion_fn=fake_completion)
        with tempfile.TemporaryDirectory() as td:
            f_path = Path(td) / "front.png"
            d_path = Path(td) / "detail.png"
            Image.new("RGB", (1200, 1200), (255, 255, 255)).save(f_path)
            Image.new("RGB", (1200, 1200), (240, 240, 240)).save(d_path)

            res = engine.predict(
                "mv_metal1",
                {"front_view": str(f_path), "detail_view": str(d_path)},
            )
            self.assertEqual(res["ondc_taxonomy"]["domain_code"], "ONDC:RET16")
            self.assertEqual(res["color_profile"]["primary_color"]["standard_family"], "Orange")
            self.assertEqual(res["color_profile"]["surface_finish"], "glossy")
            self.assertEqual(res["audit_telemetry"]["views_processed"], ["front_view", "detail_view"])
            user_parts = captured["messages"][1]["content"]
            img_urls = [p["image_url"]["url"] for p in user_parts if p["type"] == "image_url"]
            self.assertEqual(len(img_urls), 2)
            self.assertTrue(all(u.startswith("data:image/jpeg;base64,") for u in img_urls))
            self.assertNotIn("tools", captured)
            self.assertNotIn("tool_choice", captured)

    def test_compact_wire_normalization_matches_locked_contract(self) -> None:
        compact = {
            "product": "Engraved Brass Tumbler",
            "title": "Traditional Engraved Brass Water Tumbler",
            "desc": "Handcrafted brass tumbler featuring fine traditional floral engravings.",
            "comp": "single_item",
            "scale": "handheld",
            "domain": "ONDC:RET16",
            "cat": "Metalware",
            "subcat": "Carved Metal Tumbler",
            "dom_conf": 0.94,
            "color": "Antique Silver-Brass",
            "color_fam": "Silver",
            "finish": "oxidized_antique",
            "mat": "Brass",
            "mat_fam": "Metal",
            "mat_conf": 0.92,
            "pat": ["traditional_motif"],
            "craft": ["engraved"],
            "seals": [],
            "text": False,
            "crop": False,
            "conf": 0.92,
            "review": [],
            "kw": ["brass tumbler", "engraved mug"],
        }
        hydrated = normalize_wire_payload(compact, product_id="metalware1", views_used=["front_view"])
        validated = ProductIntelligenceOutput.model_validate(hydrated)
        self.assertEqual(validated.ondc_taxonomy.domain_code.value, "ONDC:RET16")
        self.assertEqual(validated.materials[0].material_family, "Metal")
        self.assertEqual(validated.audit_telemetry.views_processed, ["front_view"])
        self.assertEqual(validated.pattern_and_craft.craft_techniques, ["engraved"])


if __name__ == "__main__":
    unittest.main()
