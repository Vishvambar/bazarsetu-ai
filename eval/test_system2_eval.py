"""
Unit & smoke tests for System 2 Step 2 (schemas.py) and Step 3 (system2_eval.py + golden_dataset.jsonl).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.system2_eval import evaluate_pipeline, load_golden_records
from system2.schemas import (
    ONDCDomainCode,
    ProductIntelligenceOutput,
    TextAndSeals,
)


class System2ContractAndEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_path = PROJECT_ROOT / "eval" / "golden_dataset.jsonl"
        cls.manifest_path = PROJECT_ROOT / "dataset" / "dataset_manifest.json"
        cls.records = load_golden_records(cls.golden_path)
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_json_schema_export_and_gateway_translation(self) -> None:
        schema = ProductIntelligenceOutput.model_json_schema()
        self.assertIn("properties", schema)
        self.assertIn("text_and_seals", schema["properties"])
        self.assertEqual(
            ONDCDomainCode.HOME_DECOR.to_gateway_domain(translate_home_decor=False),
            "ONDC:RET16",
        )
        self.assertEqual(
            ONDCDomainCode.HOME_DECOR.to_gateway_domain(translate_home_decor=True),
            "ONDC:RET15",
        )
        seals = TextAndSeals(has_visible_text=True, text_legibility_confidence=0.45, needs_highres_label_crop=True)
        self.assertTrue(seals.needs_highres_label_crop)
        self.assertAlmostEqual(seals.text_legibility_confidence, 0.45)

    def test_golden_dataset_aligns_with_manifest(self) -> None:
        self.assertEqual(len(self.records), self.manifest["total_products"])
        manifest_by_id = {p["product_id"]: p for p in self.manifest["products"]}
        for row in self.records:
            pid = row["product_id"]
            self.assertIn(pid, manifest_by_id)
            for vkey, golden_rel in row["views"].items():
                self.assertEqual(golden_rel, manifest_by_id[pid]["golden_views"][vkey])

    def test_oracle_mock_clears_all_release_gates(self) -> None:
        gt_map = {r["product_id"]: r["ground_truth"] for r in self.records}

        def oracle_predictor(pid: str, views: Dict[str, str]) -> Dict[str, Any]:
            gt = gt_map[pid]
            return {
                "product_id": pid,
                "product_identity": {
                    "apparent_product": gt["subcategory"],
                    "catalog_title_en": f"Handcrafted {gt['subcategory']}",
                    "catalog_title_hi": f"हस्तनिर्मित {gt['subcategory']}",
                    "description_en": "Authentic handmade artisan product crafted using traditional regional techniques.",
                    "description_hi": "पारंपरिक विधि से कारीगर द्वारा हाथ से निर्मित सुंदर उत्पाद।",
                    "composition_type": gt["composition_type"],
                    "set_contents": [],
                    "physical_scale": gt["physical_scale"],
                },
                "ondc_taxonomy": {
                    "domain_code": gt["domain_code"],
                    "category": gt["category"],
                    "subcategory": gt["subcategory"],
                    "generic_name": gt["subcategory"],
                    "confidence": 0.94,
                },
                "color_profile": {
                    "primary_color": {
                        "name": gt["primary_color_family"],
                        "standard_family": gt["primary_color_family"],
                    },
                    "secondary_colors": [],
                    "surface_finish": "matte",
                    "color_distribution": "solid",
                },
                "materials": [
                    {
                        "material_name": gt["primary_material_family"],
                        "material_family": gt["primary_material_family"],
                        "location": "main_body",
                        "confidence": 0.65 if gt["expect_seller_review"] else 0.92,
                    }
                ],
                "pattern_and_craft": {
                    "has_pattern": bool(gt["expected_craft_techniques"]),
                    "pattern_categories": ["traditional_motif"],
                    "craft_techniques": gt["expected_craft_techniques"],
                    "motif_description": None,
                },
                "structural_components": [],
                "text_and_seals": {
                    "has_visible_text": bool(gt["expected_seals"]),
                    "brand_or_artisan_mark": None,
                    "certification_seals": gt["expected_seals"],
                    "visible_text_snippets": [],
                    "text_legibility_confidence": 0.95,
                    "needs_highres_label_crop": False,
                },
                "marketplace_enrichment": {
                    "target_audience": "home_utility",
                    "occasions": ["Festive"],
                    "search_keywords": [gt["subcategory"]],
                },
                "audit_telemetry": {
                    "views_processed": list(views.keys()),
                    "overall_confidence": 0.68 if gt["expect_seller_review"] else 0.93,
                    "needs_seller_review": ["materials[0].material_name"] if gt["expect_seller_review"] else [],
                },
            }

        report = evaluate_pipeline(self.golden_path, oracle_predictor)
        self.assertTrue(report["all_gates_passed"], msg=f"Failures: {report['failures']}")
        self.assertEqual(report["total_samples"], 43)
        self.assertEqual(report["seal_hallucination_count"], 0)
        self.assertEqual(report["review_false_positive_pct"], 0.0)
        self.assertEqual(report["seal_recall_pct"], 100.0)

    def test_false_positive_review_and_seal_hallucination_fail_gates(self) -> None:
        gt_map = {r["product_id"]: r["ground_truth"] for r in self.records}

        def noisy_predictor(pid: str, views: Dict[str, str]) -> Dict[str, Any]:
            if pid == "metalware1":
                raise RuntimeError("Simulated inference glitch")
            gt = gt_map[pid]
            return {
                "product_id": pid,
                "product_identity": {
                    "apparent_product": gt["subcategory"],
                    "catalog_title_en": f"Handcrafted {gt['subcategory']}",
                    "catalog_title_hi": f"हस्तनिर्मित {gt['subcategory']}",
                    "description_en": "Craft item.",
                    "description_hi": "हस्तशिल्प उत्पाद।",
                    "composition_type": gt["composition_type"],
                    "set_contents": [],
                    "physical_scale": gt["physical_scale"],
                },
                "ondc_taxonomy": {
                    "domain_code": gt["domain_code"],
                    "category": gt["category"],
                    "subcategory": gt["subcategory"],
                    "generic_name": gt["subcategory"],
                    "confidence": 0.70,
                },
                "color_profile": {
                    "primary_color": {
                        "name": gt["primary_color_family"],
                        "standard_family": gt["primary_color_family"],
                    },
                    "secondary_colors": [],
                    "surface_finish": "matte",
                    "color_distribution": "solid",
                },
                "materials": [
                    {
                        "material_name": gt["primary_material_family"],
                        "material_family": gt["primary_material_family"],
                        "location": "main_body",
                        "confidence": 0.60,
                    }
                ],
                "pattern_and_craft": {
                    "has_pattern": False,
                    "pattern_categories": [],
                    "craft_techniques": gt["expected_craft_techniques"],
                },
                "structural_components": [],
                "text_and_seals": {
                    "has_visible_text": True,
                    "certification_seals": ["Silk Mark"],
                    "visible_text_snippets": [],
                    "text_legibility_confidence": 0.5,
                    "needs_highres_label_crop": True,
                },
                "marketplace_enrichment": {},
                "audit_telemetry": {
                    "views_processed": list(views.keys()),
                    "overall_confidence": 0.6,
                    "needs_seller_review": ["materials[0].material_name"],
                },
            }

        report = evaluate_pipeline(self.golden_path, noisy_predictor)
        self.assertFalse(report["all_gates_passed"])
        self.assertGreater(report["seal_hallucination_count"], 0)
        self.assertEqual(report["review_false_positive_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()

