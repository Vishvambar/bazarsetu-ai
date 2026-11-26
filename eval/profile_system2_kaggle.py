"""
BazarSetu V3.0 - Kaggle Profiler for System 2 (Compact Wire Speed Test).

Compares on metalware1 (cuda:1):
  - TEST D1: Compact flat English JSON without GBNF grammar
  - TEST D2: Compact flat English JSON with response_format={"type": "json_object"}
and validates both through normalize_wire_payload -> ProductIntelligenceOutput.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.prepare_system2_golden import find_restore_archive, restore_golden_zip
from system2.engine import (
    System2Config,
    System2Engine,
    extract_tool_or_json_payload,
    normalize_wire_payload,
    prepare_view_data_uri,
)
from system2.schemas import ProductIntelligenceOutput

COMPACT_PROMPT = """Output ONLY a compact single-line JSON object (no markdown fences, English only, omit optional empty fields).
Required keys & exact allowed enums:
- product: plain object name
- title: 40-70 char English ONDC title
- desc: one concise sentence (10-18 words)
- comp: "single_item"|"matched_set"|"multipack"|"product_with_packaging"
- scale: "pocket_wearable"|"handheld"|"tabletop"|"floor_standing"|"draped_fabric"
- domain: "ONDC:RET16" (pottery/metal/wood/decor) | "ONDC:RET12" (textile/apparel/footwear) | "ONDC:RET10" (food)
- cat: category e.g. "Metalware", "Pottery & Claycraft", "Woodcraft", "Sarees", "Home Decor"
- subcat: specific product type
- color: descriptive color; color_fam: Red|Blue|Green|Yellow|Orange|Pink|Purple|Grey|Black|White|Brown|Gold|Silver|Beige|Multicolor
- finish: "matte"|"glossy"|"metallic"|"oxidized_antique"|"raw_natural"|"woven_textile"|"hammered_mathar"|"glazed"
- mat: primary material; mat_fam: Metal|Textile|Wood|Clay_Ceramic|Leather|Stone|Glass|Natural_Fiber|Paper_Bamboo; mat_conf: 0.0-1.0
- pat: array from ["solid","geometric","floral","traditional_motif","warli_tribal","striped","checked","abstract"]
- craft: array from ["hand_hammered_mathar","lost_wax_cast","engraved","embossed","hand_painted","glazed","hand_carved","pit_fired","zari_brocade","embroidery","block_print","warli_painted"]
- seals: ONLY clearly visible logos from ["Silk Mark","Handloom Mark","GI Tag","India Handmade","FSSAI","BIS Hallmark"], else []
- text: bool; legib: 0.0-1.0; crop: bool (true only if seal/label blurry)
- conf: 0.0-1.0; review: ["materials[0].material_name"] if material/purity ambiguous, else []
- kw: 2-3 short search keywords"""


def run_variant(engine: System2Engine, data_uri: str, label: str, use_json_rf: bool) -> None:
    kwargs = {
        "messages": [
            {"role": "system", "content": COMPACT_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Analyze front_view for metalware1 and output the compact JSON now."},
                ],
            },
        ],
        "max_tokens": 300,
        "temperature": 0.1,
    }
    if use_json_rf:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.perf_counter()
    res = engine._llm.create_chat_completion(**kwargs)
    dt = time.perf_counter() - t0
    usage = res.get("usage", {})
    comp_tok = usage.get("completion_tokens", 0)
    raw_content = res["choices"][0]["message"].get("content") or ""
    status = "OK"
    try:
        parsed = extract_tool_or_json_payload(res)
        hydrated = normalize_wire_payload(parsed, product_id="metalware1", views_used=["front_view"])
        validated = ProductIntelligenceOutput.model_validate(hydrated)
        summary = (
            f"domain={validated.ondc_taxonomy.domain_code.value} "
            f"cat={validated.ondc_taxonomy.category} "
            f"color={validated.color_profile.primary_color.standard_family} "
            f"craft={validated.pattern_and_craft.craft_techniques}"
        )
    except Exception as exc:
        status = f"ERR: {exc}"
        summary = repr(raw_content[:200])

    print(
        f"[{label}] {dt:.2f}s | prompt={usage.get('prompt_tokens')} | comp={comp_tok} "
        f"({comp_tok / max(dt, 1e-6):.1f} tok/s) | {status} | {summary}",
        flush=True,
    )


def main() -> int:
    golden_dir = PROJECT_ROOT / "storage" / "golden"
    archive = find_restore_archive(None)
    if archive:
        restore_golden_zip(archive, golden_dir)

    cfg = System2Config(verbose=False, max_tokens=768)
    engine = System2Engine(config=cfg)
    sample_img = golden_dir / "metalware1_front.png"
    data_uri = prepare_view_data_uri(sample_img, max_dim=cfg.front_max_dim)

    run_variant(engine, data_uri, "TEST D1 (Compact NO-grammar)", use_json_rf=False)
    run_variant(engine, data_uri, "TEST D2 (Compact json_object)", use_json_rf=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
