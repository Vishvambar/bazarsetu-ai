"""
BazarSetu V3.0 - Product-Specific Seller Verification & Catalog MCQ Generator.

Generates 2 to 5 zero-fluff Multiple Choice Questions tailored strictly to the
product System 2 just identified:
  1. purpose="verify_prediction": pre-selects Gemma 4 / SigLIP predictions for
     fast 1-tap seller confirmation on uncertain/critical fields.
  2. purpose="complete_catalog": gathers non-visual ONDC catalog / Legal Metrology
     specs (e.g. saree length, pickle net weight, vessel capacity, footwear sizes)
     relevant ONLY to that specific product category.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from system2.schemas import (
    CompositionType,
    ONDCDomainCode,
    PhysicalScale,
    ProductIntelligenceOutput,
    ProductMCQ,
    QuestionOption,
)
from system2.taxonomy_grounder import TaxonomyGroundingResult

_MATERIAL_CHOICES_BY_FAMILY: Dict[str, List[str]] = {
    "Metal": ["Brass", "Copper", "Bell Metal (Kansa)", "Brass-Plated Alloy", "Silver / German Silver"],
    "Textile": ["Pure Silk", "Cotton-Silk Blend", "Pure Cotton", "Handspun Linen", "Art Silk / Viscose Blend"],
    "Clay_Ceramic": ["Terracotta Clay", "Glazed Ceramic", "Stoneware", "Ceramic & Woven Cane"],
    "Wood": ["Sheesham Rosewood", "Teak Wood", "Mango Wood", "MDF / Engineered Wood"],
    "Stone": ["Marble Stone", "Soapstone", "Sandstone"],
    "Leather": ["Vegetable-Tanned Leather", "Synthetic Leather"],
    "Glass": ["Food-Grade Glass Jar", "Artisanal Handblown Glass"],
    "Natural_Fiber": ["Natural Cane / Wicker", "Bamboo", "Jute"],
}


def _slug_option(idx: int) -> str:
    return f"opt_{idx + 1}"


def _build_options_with_default(
    candidates: Sequence[str],
    predicted_value: Optional[str] = None,
    limit: int = 5,
) -> Tuple[List[QuestionOption], Optional[str]]:
    """Build clean deduplicated options and mark whichever matches Gemma's prediction."""
    ordered: List[str] = []
    pred_clean = (predicted_value or "").strip()
    seen_low = set()

    if pred_clean:
        ordered.append(pred_clean)
        seen_low.add(pred_clean.lower())

    for c in candidates:
        c_clean = str(c).strip()
        low = c_clean.lower()
        if not c_clean:
            continue
        # Avoid near-duplicate if predicted value already covers or equals option
        if low in seen_low:
            continue
        ordered.append(c_clean)
        seen_low.add(low)
        if len(ordered) >= limit:
            break

    if len(ordered) < 2:
        ordered.append("Other / Specify")

    options: List[QuestionOption] = []
    default_id: Optional[str] = None
    for idx, label in enumerate(ordered[:limit]):
        oid = _slug_option(idx)
        is_pred = bool(pred_clean and label.lower() == pred_clean.lower())
        if is_pred and default_id is None:
            default_id = oid
        options.append(
            QuestionOption(
                option_id=oid,
                label=label,
                value=label,
                is_model_prediction=is_pred,
            )
        )
    return options, default_id


def _build_verify_category_mcq(
    output: ProductIntelligenceOutput,
    grounding: Optional[TaxonomyGroundingResult],
) -> Optional[ProductMCQ]:
    review_set = set(output.audit_telemetry.needs_seller_review)
    needs_cat = "ondc_taxonomy.category" in review_set or float(output.ondc_taxonomy.confidence) < 0.85
    if not needs_cat:
        return None

    pred_label = f"{output.ondc_taxonomy.category} — {output.ondc_taxonomy.subcategory}"
    cands: List[str] = [pred_label]
    if grounding and grounding.stage1_priors:
        for prior in grounding.stage1_priors:
            cand = f"{prior.category} — {prior.subcategory}"
            if cand not in cands:
                cands.append(cand)

    options, default_id = _build_options_with_default(cands, predicted_value=pred_label, limit=4)
    return ProductMCQ(
        question_id="verify_category",
        target_field="ondc_taxonomy.category",
        purpose="verify_prediction",
        question_text=f"Confirm the right ONDC category for '{output.product_identity.apparent_product}':",
        options=options,
        allow_multiple=False,
        allow_custom_input=True,
        default_option_id=default_id,
        required=True,
    )


def _build_verify_material_mcq(output: ProductIntelligenceOutput) -> Optional[ProductMCQ]:
    primary_mat = output.materials[0]
    review_blob = " ".join(output.audit_telemetry.needs_seller_review).lower()
    flagged = (
        "material" in review_blob
        or float(primary_mat.confidence) < 0.85
        or (
            primary_mat.material_family in ("Metal", "Textile")
            and not output.text_and_seals.certification_seals
            and float(primary_mat.confidence) < 0.94
        )
    )
    if not flagged:
        return None

    fam_choices = _MATERIAL_CHOICES_BY_FAMILY.get(
        primary_mat.material_family,
        ["Natural Handcrafted Material", "Blended / Composite Material"],
    )
    options, default_id = _build_options_with_default(
        fam_choices, predicted_value=primary_mat.material_name, limit=5
    )
    return ProductMCQ(
        question_id="verify_primary_material",
        target_field="materials[0].material_name",
        purpose="verify_prediction",
        question_text=f"Confirm the primary material / purity of this {output.ondc_taxonomy.subcategory}:",
        options=options,
        allow_multiple=False,
        allow_custom_input=True,
        default_option_id=default_id,
        required=True,
    )


def _build_verify_composition_mcq(output: ProductIntelligenceOutput) -> Optional[ProductMCQ]:
    comp = output.product_identity.composition_type
    if comp not in (CompositionType.MATCHED_SET, CompositionType.MULTIPACK):
        return None

    label_map = {
        CompositionType.SINGLE_ITEM.value: "Single Item (1 piece)",
        CompositionType.MATCHED_SET.value: "Matched Set (as pictured)",
        CompositionType.MULTIPACK.value: "Multipack Bundle (multiple identical units)",
    }
    pred_val = comp.value
    opts = [
        QuestionOption(
            option_id=_slug_option(idx),
            label=lbl,
            value=val,
            is_model_prediction=(val == pred_val),
        )
        for idx, (val, lbl) in enumerate(label_map.items())
    ]
    default_id = next((o.option_id for o in opts if o.is_model_prediction), None)
    return ProductMCQ(
        question_id="verify_pack_type",
        target_field="product_identity.composition_type",
        purpose="verify_prediction",
        question_text="Are buyers ordering everything pictured together or one individual piece?",
        options=opts,
        allow_multiple=False,
        allow_custom_input=False,
        default_option_id=default_id,
        required=True,
    )


def _build_verify_seals_mcq(output: ProductIntelligenceOutput) -> Optional[ProductMCQ]:
    seals = output.text_and_seals
    cat_low = output.ondc_taxonomy.category.lower()
    sub_low = output.ondc_taxonomy.subcategory.lower()
    relevant = (
        seals.has_visible_text
        or bool(seals.certification_seals)
        or seals.needs_highres_label_crop
        or "saree" in cat_low
        or "pickle" in sub_low
        or output.ondc_taxonomy.domain_code == ONDCDomainCode.GROCERY
    )
    if not relevant:
        return None

    if output.ondc_taxonomy.domain_code == ONDCDomainCode.GROCERY or "pickle" in sub_low:
        base_opts = ["FSSAI", "GI Tag", "Organic India", "No Certification Logo"]
    elif output.ondc_taxonomy.domain_code == ONDCDomainCode.FASHION:
        base_opts = ["Silk Mark", "Handloom Mark", "GI Tag", "No Certification Seal"]
    else:
        base_opts = ["GI Tag", "BIS Hallmark", "Handicraft Mark", "No Certification Seal"]

    detected = set(s.strip() for s in seals.certification_seals if s and s.strip())
    opts: List[QuestionOption] = []
    default_id: Optional[str] = None
    for idx, seal_name in enumerate(base_opts):
        oid = _slug_option(idx)
        is_pred = seal_name in detected
        if is_pred and default_id is None:
            default_id = oid
        opts.append(
            QuestionOption(
                option_id=oid,
                label=seal_name,
                value=seal_name,
                is_model_prediction=is_pred,
            )
        )
    if not detected:
        default_id = opts[-1].option_id

    return ProductMCQ(
        question_id="verify_certification_seals",
        target_field="text_and_seals.certification_seals",
        purpose="verify_prediction",
        question_text="Select any valid authenticity seals/marks included on this product:",
        options=opts,
        allow_multiple=True,
        allow_custom_input=False,
        default_option_id=default_id,
        required=False,
    )


def _build_product_specific_catalog_mcqs(output: ProductIntelligenceOutput) -> List[ProductMCQ]:
    """Build non-visual, zero-fluff catalog completion MCQs strictly for this craft type."""
    domain = output.ondc_taxonomy.domain_code
    cat = output.ondc_taxonomy.category.lower()
    sub = output.ondc_taxonomy.subcategory.lower()
    prod = output.product_identity.apparent_product.lower()
    blob = f"{cat} {sub} {prod}"
    scale = output.product_identity.physical_scale
    questions: List[ProductMCQ] = []

    # 1. Packaged Food / Pickles (ONDC:RET10)
    if domain == ONDCDomainCode.GROCERY or any(k in blob for k in ("pickle", "achar", "honey", "food", "spice")):
        opts_wt, _ = _build_options_with_default(["250g", "500g", "1 kg"])
        questions.append(
            ProductMCQ(
                question_id="food_net_weight",
                target_field="catalog_specs.net_quantity",
                purpose="complete_catalog",
                question_text=f"What is the net jar weight of this {output.ondc_taxonomy.subcategory}?",
                options=opts_wt,
                allow_multiple=False,
                allow_custom_input=True,
            )
        )
        opts_shelf, _ = _build_options_with_default(["3 Months", "6 Months", "12 Months"])
        questions.append(
            ProductMCQ(
                question_id="food_shelf_life",
                target_field="catalog_specs.shelf_life",
                purpose="complete_catalog",
                question_text="Shelf life / best-before duration from packing:",
                options=opts_shelf,
                allow_multiple=False,
                allow_custom_input=True,
            )
        )
        return questions

    # 2. Sarees (ONDC:RET12)
    if "saree" in blob or "sari" in blob or cat == "sarees":
        opts_len, _ = _build_options_with_default(
            ["5.5m (Saree Only)", "6.2m (With Unstitched Blouse Piece)", "9 Yard Nauvari (8.2m)"]
        )
        questions.append(
            ProductMCQ(
                question_id="saree_length_blouse",
                target_field="catalog_specs.dimensions",
                purpose="complete_catalog",
                question_text="Select exact saree length & blouse piece inclusion:",
                options=opts_len,
                allow_multiple=False,
                allow_custom_input=True,
            )
        )
        opts_care, _ = _build_options_with_default(["Dry Clean Only", "Gentle Cold Hand Wash"])
        questions.append(
            ProductMCQ(
                question_id="textile_wash_care",
                target_field="catalog_specs.care_instructions",
                purpose="complete_catalog",
                question_text="Recommended wash care for this saree:",
                options=opts_care,
                allow_multiple=False,
                allow_custom_input=False,
            )
        )
        return questions

    # 3. Footwear / Chappals
    if "footwear" in blob or "chappal" in blob or "sandal" in blob:
        opts_sz, _ = _build_options_with_default(["IND/UK 6", "IND/UK 7", "IND/UK 8", "IND/UK 9", "IND/UK 10"])
        questions.append(
            ProductMCQ(
                question_id="footwear_sizes",
                target_field="catalog_specs.available_sizes",
                purpose="complete_catalog",
                question_text="Which footwear sizes are available in stock?",
                options=opts_sz,
                allow_multiple=True,
                allow_custom_input=True,
            )
        )
        return questions

    # 4. Ethnic Wear (Kurta, Dhoti, Veshti, Kaftan)
    if any(k in blob for k in ("kurta", "dupatta", "kaftan", "dhoti", "veshti")):
        if any(k in blob for k in ("dhoti", "veshti")):
            opts_dhoti, _ = _build_options_with_default(["2.0 Meters (Single)", "4.0 Meters (Double)"])
            questions.append(
                ProductMCQ(
                    question_id="dhoti_length",
                    target_field="catalog_specs.dimensions",
                    purpose="complete_catalog",
                    question_text="Select dhoti/veshti fabric length:",
                    options=opts_dhoti,
                    allow_multiple=False,
                    allow_custom_input=True,
                )
            )
        else:
            opts_apparel, _ = _build_options_with_default(["S", "M", "L", "XL", "XXL", "Free Size"])
            questions.append(
                ProductMCQ(
                    question_id="apparel_sizes",
                    target_field="catalog_specs.available_sizes",
                    purpose="complete_catalog",
                    question_text="Select available garment sizes:",
                    options=opts_apparel,
                    allow_multiple=True,
                    allow_custom_input=True,
                )
            )
        return questions

    # 5. Unstitched Fabrics
    if domain == ONDCDomainCode.FASHION or "fabric" in blob:
        opts_cut, _ = _build_options_with_default(["1 Meter", "2.5 Meters (Standard Cut)", "3 Meters", "5 Meters"])
        questions.append(
            ProductMCQ(
                question_id="fabric_cut_length",
                target_field="catalog_specs.dimensions",
                purpose="complete_catalog",
                question_text="Cut length supplied per ordered unit:",
                options=opts_cut,
                allow_multiple=False,
                allow_custom_input=True,
            )
        )
        return questions

    # 6. Diyas / Small Clay Lamps
    if "diya" in blob:
        opts_pack, _ = _build_options_with_default(["Set of 4", "Set of 6", "Set of 12", "Single Large Diya"])
        questions.append(
            ProductMCQ(
                question_id="diya_pack_count",
                target_field="catalog_specs.net_quantity",
                purpose="complete_catalog",
                question_text="How many diyas are packed in one unit?",
                options=opts_pack,
                allow_multiple=False,
                allow_custom_input=True,
            )
        )
        return questions

    # 7. Vessels (Kalash, Urli, Handi, Tumbler, Mug, Flask, Teapot, Bowl)
    if any(k in blob for k in ("kalash", "urli", "handi", "tumbler", "mug", "flask", "teapot", "bowl", "cup")):
        opts_cap, _ = _build_options_with_default(
            ["Small (< 350 ml / 4-5 in)", "Medium (500 ml - 1.2 L / 6-8 in)", "Large (> 1.5 L / 9+ in)"]
        )
        questions.append(
            ProductMCQ(
                question_id="vessel_capacity",
                target_field="catalog_specs.dimensions",
                purpose="complete_catalog",
                question_text=f"Approximate capacity / diameter of this {output.ondc_taxonomy.subcategory}:",
                options=opts_cap,
                allow_multiple=False,
                allow_custom_input=True,
            )
        )
        opts_safe, _ = _build_options_with_default(["Yes - Food / Water Safe", "Pooja & Decorative Display Only"])
        questions.append(
            ProductMCQ(
                question_id="vessel_food_safe",
                target_field="catalog_specs.food_safe",
                purpose="complete_catalog",
                question_text="Is the interior finish safe for drinking water or food?",
                options=opts_safe,
                allow_multiple=False,
                allow_custom_input=False,
            )
        )
        return questions

    # 8. Figurines, Lamps, Woodcraft, Stone, Home Decor
    if scale == PhysicalScale.FLOOR_STANDING or "samai" in blob:
        dim_opts = ["2 - 3 Feet", "3 - 4 Feet", "Above 4 Feet"]
    else:
        dim_opts = ["Compact (< 6 in)", "Medium Tabletop (6 - 12 in)", "Large (12 - 20 in)"]
    opts_dim, _ = _build_options_with_default(dim_opts)
    questions.append(
        ProductMCQ(
            question_id="decor_dimensions",
            target_field="catalog_specs.dimensions",
            purpose="complete_catalog",
            question_text=f"Approximate height / overall size of this {output.ondc_taxonomy.subcategory}:",
            options=opts_dim,
            allow_multiple=False,
            allow_custom_input=True,
        )
    )
    return questions


def generate_product_mcqs(
    output: ProductIntelligenceOutput,
    grounding: Optional[TaxonomyGroundingResult] = None,
    max_questions: int = 5,
) -> List[ProductMCQ]:
    """
    Generate ordered product-specific MCQs (verify_prediction first, then
    complete_catalog), strictly capped with zero generic fluff.
    """
    verify_builders = (
        _build_verify_category_mcq(output, grounding),
        _build_verify_material_mcq(output),
        _build_verify_composition_mcq(output),
        _build_verify_seals_mcq(output),
    )
    ordered: List[ProductMCQ] = [q for q in verify_builders if q is not None]
    for q in _build_product_specific_catalog_mcqs(output):
        ordered.append(q)

    deduped: List[ProductMCQ] = []
    seen_ids = set()
    for q in ordered:
        if q.question_id in seen_ids:
            continue
        seen_ids.add(q.question_id)
        deduped.append(q)
        if len(deduped) >= max_questions:
            break
    return deduped


def apply_mcq_answers(
    output: ProductIntelligenceOutput,
    answers: Dict[str, Any],
) -> Tuple[ProductIntelligenceOutput, Dict[str, Any]]:
    """
    Apply seller MCQ responses back onto ProductIntelligenceOutput, clear resolved
    review flags, and return (confirmed_output, catalog_specs).
    """
    data = output.model_dump()
    review_paths: List[str] = list(data["audit_telemetry"].get("needs_seller_review") or [])
    catalog_specs: Dict[str, Any] = {}

    def _clear_review_prefix(prefix: str) -> None:
        nonlocal review_paths
        review_paths = [p for p in review_paths if prefix not in p.lower()]

    for qid, val in answers.items():
        if val is None or val == "":
            continue

        if qid == "verify_category":
            text = str(val)
            if "—" in text:
                cat_part, sub_part = [p.strip() for p in text.split("—", 1)]
                data["ondc_taxonomy"]["category"] = cat_part
                data["ondc_taxonomy"]["subcategory"] = sub_part
            else:
                data["ondc_taxonomy"]["category"] = text.strip()
            data["ondc_taxonomy"]["confidence"] = 1.0
            _clear_review_prefix("ondc_taxonomy")

        elif qid == "verify_primary_material":
            data["materials"][0]["material_name"] = str(val).strip()
            data["materials"][0]["confidence"] = 1.0
            _clear_review_prefix("material")

        elif qid == "verify_pack_type":
            valid_comps = {c.value for c in CompositionType}
            clean_comp = str(val).strip()
            if clean_comp in valid_comps:
                data["product_identity"]["composition_type"] = clean_comp

        elif qid == "verify_certification_seals":
            items = val if isinstance(val, list) else [val]
            cleaned = [
                str(s).strip()
                for s in items
                if s and "no certification" not in str(s).lower() and "none" != str(s).strip().lower()
            ]
            data["text_and_seals"]["certification_seals"] = cleaned
            data["text_and_seals"]["text_legibility_confidence"] = 1.0
            data["text_and_seals"]["needs_highres_label_crop"] = False
            _clear_review_prefix("seals")

        else:
            catalog_specs[qid] = val

    data["audit_telemetry"]["needs_seller_review"] = review_paths
    confirmed = ProductIntelligenceOutput.model_validate(data)
    return confirmed, catalog_specs

