"""
Offline unit tests for System 2 Step 5 (taxonomy_grounder.py) and Step 6 (visual_search.py).
Uses synthetic 768-d vectors and mocked System2Engine so no GPU weights download is required.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.test_system2_engine import _sample_valid_payload
from system2.engine import PATCH_SIZE, System2Config, System2Engine
from system2.schemas import ProductIntelligenceOutput
from system2.siglip_encoder import SigLIPEncoder
from system2.taxonomy_grounder import CURATED_ONDC_LEAVES, TaxonomyGrounder
from system2.visual_search import (
    VisualDuplicateIndex,
    escalate_label_inspection,
    prepare_highres_label_crop,
    should_escalate_label,
)


def _deterministic_vec(seed: int, dim: int = 768) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(dim,)).astype(np.float32)
    return v / np.linalg.norm(v)


def _fake_encoder() -> Tuple[SigLIPEncoder, Dict[str, int], Dict[str, np.ndarray]]:
    text_calls = {"count": 0}
    leaf_vecs: Dict[str, np.ndarray] = {
        leaf.prompt_text: _deterministic_vec(idx + 10) for idx, leaf in enumerate(CURATED_ONDC_LEAVES)
    }

    def fake_text_encode(prompts: Sequence[str]) -> np.ndarray:
        text_calls["count"] += 1
        rows = [leaf_vecs.get(p, _deterministic_vec(abs(hash(p)) % 1000)) for p in prompts]
        return np.stack(rows, axis=0)

    def fake_image_encode(src: Any) -> np.ndarray:
        # Return vector closely aligned with leaf 0 (ret16_copper_kalash) by default
        return leaf_vecs[CURATED_ONDC_LEAVES[0].prompt_text].copy()

    return (
        SigLIPEncoder(lazy_load=True, image_encode_fn=fake_image_encode, text_encode_fn=fake_text_encode),
        text_calls,
        leaf_vecs,
    )


class Step5AndStep6Tests(unittest.TestCase):
    def test_step5_zero_shot_priors_domain_lock_and_low_conf_review(self) -> None:
        encoder, text_calls, leaf_vecs = _fake_encoder()
        grounder = TaxonomyGrounder(encoder=encoder, preload_priors=True)
        self.assertEqual(text_calls["count"], 1)

        # Verify Day-1 priors reuse cached text matrix without re-encoding texts
        kalash_dict = _sample_valid_payload("mv_metal1")
        pred = ProductIntelligenceOutput.model_validate(kalash_dict)
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "front.png"
            Image.new("RGB", (480, 480), (255, 255, 255)).save(img_path)

            res = grounder.ground(
                pred,
                front_image_path=img_path,
                translate_home_decor_for_gateway=True,
            )
            self.assertEqual(text_calls["count"], 1)
            self.assertEqual(res.output.ondc_taxonomy.domain_code.value, "ONDC:RET16")
            self.assertEqual(res.export_domain_code, "ONDC:RET15")
            self.assertFalse(res.review_flagged_by_grounder)
            self.assertEqual(res.stage1_priors[0].leaf_id, "ret16_copper_kalash")

            # Layer seller exemplar image embedding and confirm Stage 1 prior blending works
            grounder.add_seller_exemplar("ret16_copper_kalash", leaf_vecs[CURATED_ONDC_LEAVES[0].prompt_text])
            priors, _ = grounder.compute_stage1_priors(image_embedding=leaf_vecs[CURATED_ONDC_LEAVES[0].prompt_text])
            self.assertEqual(priors[0].leaf_id, "ret16_copper_kalash")

        # Domain mismatch (e.g. saree accidentally tagged RET16) gets locked to RET12 + review flag
        mismatch = _sample_valid_payload("saree_err")
        mismatch["product_identity"]["apparent_product"] = "Paithani Silk Saree"
        mismatch["ondc_taxonomy"] = {
            "domain_code": "ONDC:RET16",
            "category": "Sarees",
            "subcategory": "Paithani Silk Saree",
            "generic_name": "Handloom Silk Saree",
            "confidence": 0.90,
        }
        res2 = grounder.ground(ProductIntelligenceOutput.model_validate(mismatch), compute_priors=False)
        self.assertEqual(res2.output.ondc_taxonomy.domain_code.value, "ONDC:RET12")
        self.assertEqual(res2.export_domain_code, "ONDC:RET12")
        self.assertTrue(res2.review_flagged_by_grounder)
        self.assertIn("ondc_taxonomy.category", res2.output.audit_telemetry.needs_seller_review)

    def test_step6_duplicate_index_and_highres_label_escalation(self) -> None:
        base_vec = _deterministic_vec(42)
        near_dup = l2_normalize_vec(base_vec + 0.05 * _deterministic_vec(43))
        different_vec = _deterministic_vec(99)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            idx_path = root / "visual_index.json"
            index = VisualDuplicateIndex(index_path=idx_path)
            index.upsert("prod_a", base_vec, {"title_en": "Brass Kalash", "category": "Metalware", "domain_code": "ONDC:RET16"})
            index.upsert("prod_b", different_vec, {"title_en": "Silk Saree", "category": "Sarees", "domain_code": "ONDC:RET12"})

            # Reload from disk to confirm persistence
            reloaded = VisualDuplicateIndex(index_path=idx_path)
            self.assertEqual(reloaded.count(), 2)
            dup_res = reloaded.search_similar(near_dup, query_product_id="prod_new", threshold=0.88)
            self.assertTrue(dup_res.has_duplicates)
            self.assertEqual(dup_res.matches[0].product_id, "prod_a")
            self.assertGreaterEqual(dup_res.matches[0].similarity, 0.88)

            # High-res escalation predicate check
            no_esc_dict = _sample_valid_payload("clear_item")
            self.assertFalse(should_escalate_label(ProductIntelligenceOutput.model_validate(no_esc_dict)))

            esc_dict = _sample_valid_payload("seal_blur")
            esc_dict["text_and_seals"] = {
                "has_visible_text": True,
                "brand_or_artisan_mark": None,
                "certification_seals": ["FSSAI"],
                "visible_text_snippets": ["Mango Pickle"],
                "text_legibility_confidence": 0.62,
                "needs_highres_label_crop": True,
            }
            initial_pred = ProductIntelligenceOutput.model_validate(esc_dict)
            self.assertTrue(should_escalate_label(initial_pred))

            # Store System 1 archived original image on disk under storage/products/seal_blur/original/
            orig_dir = root / "products" / "seal_blur" / "original"
            orig_dir.mkdir(parents=True, exist_ok=True)
            orig_img = orig_dir / "asset01_original.jpg"
            Image.new("RGB", (1400, 900), (255, 255, 255)).save(orig_img)

            crop_out = prepare_highres_label_crop(orig_img, output_path=root / "crop_check.jpg", max_dim=960)
            with Image.open(crop_out) as im:
                self.assertEqual(im.width % PATCH_SIZE, 0)
                self.assertEqual(im.height % PATCH_SIZE, 0)

            seen_kwargs: Dict[str, Any] = {}

            def fake_escalation_llm(**kwargs: Any) -> Dict[str, Any]:
                seen_kwargs.update(kwargs)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "product": "Mango Pickle",
                                        "title": "Homemade Mango Pickle Jar",
                                        "desc": "Homemade spiced mango pickle packed in sealed glass jar.",
                                        "domain": "ONDC:RET10",
                                        "cat": "Packaged Food",
                                        "subcat": "Artisanal Mango Pickle",
                                        "color": "Amber Brown",
                                        "color_fam": "Brown",
                                        "finish": "glossy",
                                        "mat": "Glass",
                                        "mat_fam": "Glass",
                                        "seals": ["FSSAI", "GI Tag"],
                                        "text": True,
                                        "brand": "Aai Kitchen",
                                        "snippets": ["Lic No 11524001000412"],
                                        "legib": 0.96,
                                        "crop": False,
                                        "conf": 0.95,
                                        "review": [],
                                    }
                                )
                            }
                        }
                    ]
                }

            engine = System2Engine(config=System2Config(), lazy_load=True, completion_fn=fake_escalation_llm)
            merged_pred, did_escalate = escalate_label_inspection(
                product_id="seal_blur",
                initial_output=initial_pred,
                engine=engine,
                storage_root=root,
            )
            self.assertTrue(did_escalate)
            self.assertIn("GI Tag", merged_pred.text_and_seals.certification_seals)
            self.assertIn("FSSAI", merged_pred.text_and_seals.certification_seals)
            self.assertEqual(merged_pred.text_and_seals.brand_or_artisan_mark, "Aai Kitchen")
            self.assertFalse(merged_pred.text_and_seals.needs_highres_label_crop)
            self.assertGreaterEqual(merged_pred.text_and_seals.text_legibility_confidence, 0.96)
            self.assertIn("escalation_crop_view", merged_pred.audit_telemetry.views_processed)
            self.assertIn("messages", seen_kwargs)


def l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    return v / float(np.linalg.norm(v))


if __name__ == "__main__":
    unittest.main()
