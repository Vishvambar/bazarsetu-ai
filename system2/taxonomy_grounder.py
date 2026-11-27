"""
BazarSetu V3.0 - Step 5: Two-Stage ONDC Enum Grounding with Day-1 Zero-Shot Priors.

Stage 1: Pre-embeds static ONDC leaf category text prompts on GPU 0 (SigLIP) at
startup for zero-shot image-to-text cosine priors (plus optional seller exemplar
image-to-image priors as catalog grows).
Stage 2: Reconciles Gemma 4's raw (domain_code, category, subcategory) against the
curated ONDC craft/retail leaf dictionary, locks ONDC:RET12 vs RET16/RET15
consistency, and flags 'ondc_taxonomy.category' in audit_telemetry.needs_seller_review
when grounded confidence drops below 0.75.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from system2.schemas import ONDCDomainCode, ProductIntelligenceOutput
from system2.siglip_encoder import SigLIPEncoder, l2_normalize

logger = logging.getLogger("BazarSetu.TaxonomyGrounder")

REVIEW_CONFIDENCE_THRESHOLD = 0.75


@dataclass(frozen=True)
class ONDCLeafEntry:
    leaf_id: str
    domain_code: ONDCDomainCode
    category: str
    subcategory: str
    generic_name: str
    prompt_phrase: str
    aliases: Tuple[str, ...] = ()

    @property
    def prompt_text(self) -> str:
        return f"a studio product photo of {self.prompt_phrase}"


CURATED_ONDC_LEAVES: Tuple[ONDCLeafEntry, ...] = (
    # ── ONDC:RET16 (Internal Home & Decor / Indian Crafts; exportable as RET15) ──
    ONDCLeafEntry(
        "ret16_copper_kalash",
        ONDCDomainCode.HOME_DECOR,
        "Metalware",
        "Hammered Copper Kalash",
        "Copper Water Vessel",
        "a hand-hammered copper kalash ceremonial pot",
        ("copper kalash", "tambat kalash", "copper lota", "hammered copper pot"),
    ),
    ONDCLeafEntry(
        "ret16_brass_kalash",
        ONDCDomainCode.HOME_DECOR,
        "Metalware",
        "Hammered Brass Kalash",
        "Brass Ceremonial Pot",
        "a traditional hammered brass kalash vessel",
        ("brass kalash", "brass handi", "hammered brass"),
    ),
    ONDCLeafEntry(
        "ret16_brass_urli",
        ONDCDomainCode.HOME_DECOR,
        "Metalware",
        "Polished Brass Urli Bowl",
        "Brass Decorative Bowl",
        "a polished golden brass urli or puja bowl",
        ("brass urli", "brass bowls", "engraved brass bowls", "enamel metal bowls"),
    ),
    ONDCLeafEntry(
        "ret16_metal_tumbler",
        ONDCDomainCode.HOME_DECOR,
        "Metalware",
        "Carved Metal Tumbler",
        "Metal Drinking Cup",
        "an engraved silver or white-metal tumbler cup",
        ("metal tumbler", "carved metal tumbler", "engraved mug"),
    ),
    ONDCLeafEntry(
        "ret16_dhokra_figurine",
        ONDCDomainCode.HOME_DECOR,
        "Metalware",
        "Dhokra Cast Metal Figurine",
        "Brass Metal Figurine",
        "a lost-wax cast dhokra brass figurine or ceremonial flask",
        ("dhokra", "nandi bull", "bell metal flask", "metal musicians"),
    ),
    ONDCLeafEntry(
        "ret16_brass_samai",
        ONDCDomainCode.HOME_DECOR,
        "Metalware",
        "Brass Peacock Samai Lamp",
        "Brass Oil Lamp Stand",
        "a tall brass peacock samai floor oil lamp stand",
        ("samai lamp", "peacock lamp", "floor lamp", "brass diya stand"),
    ),
    ONDCLeafEntry(
        "ret16_metal_bud_vase",
        ONDCDomainCode.HOME_DECOR,
        "Metalware",
        "Engraved Brass Bud Vases",
        "Decorative Metal Vase",
        "dark oxidized engraved brass decorative bud vases",
        ("bud vase", "metal vases", "oxidized vase"),
    ),
    ONDCLeafEntry(
        "ret16_terracotta_diya",
        ONDCDomainCode.HOME_DECOR,
        "Pottery & Claycraft",
        "Terracotta Oil Diyas",
        "Terracotta Diya",
        "rustic unglazed pit-fired terracotta clay oil diyas",
        ("terracotta diya", "clay diya", "earthen lamp", "oil diyas"),
    ),
    ONDCLeafEntry(
        "ret16_terracotta_flask",
        ONDCDomainCode.HOME_DECOR,
        "Pottery & Claycraft",
        "Terracotta Water Flask",
        "Clay Water Bottle",
        "a pit-fired terracotta clay water bottle or pot",
        ("water flask", "clay bottle", "matka", "terracotta pot"),
    ),
    ONDCLeafEntry(
        "ret16_ceramic_mug_bowl",
        ONDCDomainCode.HOME_DECOR,
        "Pottery & Claycraft",
        "Glazed Ceramic Tableware",
        "Ceramic Cup or Bowl",
        "a glazed hand-painted ceramic mug, bowl, or embossed teapot",
        ("ceramic mug", "ceramic bowl", "ceramic teapot"),
    ),
    ONDCLeafEntry(
        "ret16_ceramic_cane_vase",
        ONDCDomainCode.HOME_DECOR,
        "Pottery & Claycraft",
        "Ceramic & Cane Vase",
        "Handcrafted Ceramic Vase",
        "a hybrid ceramic and woven cane artisan flower vase",
        ("ceramic & cane vase", "cane vase", "wicker vase"),
    ),
    ONDCLeafEntry(
        "ret16_wood_carved",
        ONDCDomainCode.HOME_DECOR,
        "Woodcraft",
        "Hand-Carved Wooden Decor",
        "Carved Wooden Craft",
        "a hand-carved rosewood jewellery box, fruit bowl, or wooden wall art",
        ("wooden bowl", "jewellery box", "wall art", "rosewood", "woodcraft"),
    ),
    ONDCLeafEntry(
        "ret16_warli_tray",
        ONDCDomainCode.HOME_DECOR,
        "Woodcraft",
        "Warli Painted Wooden Serving Tray",
        "Wooden Serving Tray",
        "a dark wooden serving tray with white tribal warli paintings",
        ("warli tray", "serving tray", "warli painted"),
    ),
    ONDCLeafEntry(
        "ret16_stone_decor",
        ONDCDomainCode.HOME_DECOR,
        "Home Decor",
        "Hand-Painted Stone & Decor",
        "Decorative Home Accent",
        "a hand-painted white marble pillar candle stand or ornate gold jewellery box",
        ("marble stand", "pillar stand", "gold jewellery box", "home decor"),
    ),
    # ── ONDC:RET12 (Fashion, Handloom Sarees, Ethnic Wear, Footwear) ──
    ONDCLeafEntry(
        "ret12_paithani_saree",
        ONDCDomainCode.FASHION,
        "Sarees",
        "Paithani Silk Saree",
        "Handloom Silk Saree",
        "a traditional paithani silk saree with golden zari woven border",
        ("paithani", "silk saree", "zari saree", "handloom saree", "tissue saree"),
    ),
    ONDCLeafEntry(
        "ret12_striped_saree",
        ONDCDomainCode.FASHION,
        "Sarees",
        "Striped Handloom Saree",
        "Handwoven Saree",
        "a striped woven cotton or linen handloom saree",
        ("striped saree", "cotton saree"),
    ),
    ONDCLeafEntry(
        "ret12_cotton_dhoti",
        ONDCDomainCode.FASHION,
        "Ethnic Wear",
        "Handloom Cotton Dhoti",
        "Traditional Dhoti",
        "a folded white handloom cotton dhoti or veshti with border",
        ("dhoti", "veshti", "cotton dhoti"),
    ),
    ONDCLeafEntry(
        "ret12_kurta_set",
        ONDCDomainCode.FASHION,
        "Ethnic Wear",
        "Embroidered Kurta Set",
        "Artisan Ethnic Apparel",
        "an embroidered or block-printed ethnic kurta, dupatta set, or kaftan",
        ("kurta", "dupatta", "kaftan", "ethnic wear"),
    ),
    ONDCLeafEntry(
        "ret12_handloom_fabric",
        ONDCDomainCode.FASHION,
        "Fabrics",
        "Handloom Woven Fabric",
        "Unstitched Handloom Fabric",
        "folded unstitched handloom linen or textured woven fabric cloth",
        ("linen fabric", "woven fabric", "fabric bolt", "textile"),
    ),
    ONDCLeafEntry(
        "ret12_kolhapuri_footwear",
        ONDCDomainCode.FASHION,
        "Footwear",
        "Kolhapuri Leather Chappals",
        "Leather Footwear",
        "handcrafted brown leather kolhapuri chappals sandals",
        ("kolhapuri", "chappal", "sandal", "footwear"),
    ),
    # ── ONDC:RET10 (Grocery & Artisanal Food) ──
    ONDCLeafEntry(
        "ret10_mango_pickle",
        ONDCDomainCode.GROCERY,
        "Packaged Food",
        "Artisanal Mango Pickle",
        "Packaged Food Jar",
        "a glass jar of homemade Indian mango pickle with FSSAI label",
        ("mango pickle", "pickle", "achar", "packaged food"),
    ),
)

_TEXTILE_KEYWORDS = (
    "saree",
    "sari",
    "paithani",
    "kurta",
    "dupatta",
    "dhoti",
    "veshti",
    "kaftan",
    "shawl",
    "fabric",
    "handloom",
    "silk",
    "linen",
    "chappal",
    "footwear",
    "sandal",
)
_FOOD_KEYWORDS = ("pickle", "achar", "food", "honey", "spice", "masala", "jam")


@dataclass
class PriorCandidate:
    leaf_id: str
    domain_code: str
    category: str
    subcategory: str
    score: float


@dataclass
class TaxonomyGroundingResult:
    output: ProductIntelligenceOutput
    matched_leaf: ONDCLeafEntry
    export_domain_code: str
    stage1_priors: List[PriorCandidate]
    image_embedding: Optional[np.ndarray] = None
    review_flagged_by_grounder: bool = False


class TaxonomyGrounder:
    """Two-stage ONDC taxonomy validator & Day-1 zero-shot SigLIP grounder."""

    def __init__(
        self,
        encoder: Optional[SigLIPEncoder] = None,
        leaves: Sequence[ONDCLeafEntry] = CURATED_ONDC_LEAVES,
        preload_priors: bool = False,
    ):
        self.encoder = encoder or SigLIPEncoder(lazy_load=True)
        self.leaves: List[ONDCLeafEntry] = list(leaves)
        self._lock = threading.Lock()
        self._text_priors: Optional[np.ndarray] = None
        self._seller_exemplars: Dict[str, List[np.ndarray]] = {}

        if preload_priors:
            self.precompute_text_priors()

    def precompute_text_priors(self) -> np.ndarray:
        """Embed static ONDC leaf prompts once using SigLIP text encoder (GPU 0)."""
        with self._lock:
            if self._text_priors is not None:
                return self._text_priors
            prompts = [leaf.prompt_text for leaf in self.leaves]
            self._text_priors = self.encoder.encode_texts(prompts)
            logger.info("Precomputed %d static ONDC zero-shot text priors.", len(self.leaves))
            return self._text_priors

    def add_seller_exemplar(self, leaf_id: str, embedding: np.ndarray) -> None:
        """Layer a verified seller catalog image embedding onto a leaf prior."""
        normed = l2_normalize(np.asarray(embedding, dtype=np.float32))
        with self._lock:
            self._seller_exemplars.setdefault(leaf_id, []).append(normed)

    def compute_stage1_priors(
        self,
        image_source: Optional[Union[str, Path, Image.Image]] = None,
        image_embedding: Optional[np.ndarray] = None,
        top_k: int = 3,
    ) -> Tuple[List[PriorCandidate], np.ndarray]:
        """Compute Stage 1 cosine similarity priors across all ONDC craft leaves."""
        if image_embedding is None:
            if image_source is None:
                raise ValueError("Either image_source or image_embedding is required for Stage 1 priors.")
            image_embedding = self.encoder.encode_image(image_source)
        else:
            image_embedding = l2_normalize(image_embedding)

        text_mat = self.precompute_text_priors()
        text_sims = np.dot(text_mat, image_embedding)

        candidates: List[PriorCandidate] = []
        for idx, leaf in enumerate(self.leaves):
            score = float(text_sims[idx])
            exemplars = self._seller_exemplars.get(leaf.leaf_id)
            if exemplars:
                ex_mat = np.stack(exemplars, axis=0)
                ex_sim = float(np.max(np.dot(ex_mat, image_embedding)))
                score = 0.70 * score + 0.30 * ex_sim
            candidates.append(
                PriorCandidate(
                    leaf_id=leaf.leaf_id,
                    domain_code=leaf.domain_code.value,
                    category=leaf.category,
                    subcategory=leaf.subcategory,
                    score=round(score, 4),
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k], image_embedding

    def _find_best_dictionary_leaf(
        self,
        pred: ProductIntelligenceOutput,
        priors: Sequence[PriorCandidate],
    ) -> Tuple[ONDCLeafEntry, float]:
        """Match Gemma 4 output against CURATED_ONDC_LEAVES via lexical overlap + domain check."""
        raw_cat = pred.ondc_taxonomy.category.lower()
        raw_sub = pred.ondc_taxonomy.subcategory.lower()
        raw_gen = pred.ondc_taxonomy.generic_name.lower()
        raw_title = pred.product_identity.apparent_product.lower()
        haystack = f"{raw_cat} {raw_sub} {raw_gen} {raw_title}"

        best_leaf = self.leaves[0]
        best_score = -1.0
        for leaf in self.leaves:
            score = 0.0
            if leaf.domain_code == pred.ondc_taxonomy.domain_code:
                score += 1.0
            if leaf.category.lower() == raw_cat:
                score += 2.0
            if leaf.subcategory.lower() == raw_sub:
                score += 4.0
            for alias in leaf.aliases:
                if alias in haystack:
                    score += 3.0
            leaf_tokens = set(leaf.subcategory.lower().replace("&", " ").split()) | set(
                leaf.category.lower().replace("&", " ").split()
            )
            pred_tokens = set(haystack.replace("&", " ").split())
            overlap = len(leaf_tokens & pred_tokens)
            score += 0.8 * overlap
            if score > best_score:
                best_score = score
                best_leaf = leaf

        if best_score < 1.5 and priors:
            top_id = priors[0].leaf_id
            for leaf in self.leaves:
                if leaf.leaf_id == top_id:
                    best_leaf = leaf
                    break

        prior_agreement = 1.0
        if priors:
            top_domains = {p.domain_code for p in priors[:2]}
            if best_leaf.domain_code.value in top_domains:
                prior_agreement = 1.0
            elif priors[0].score > 0.25:
                prior_agreement = 0.85
        return best_leaf, prior_agreement

    def ground(
        self,
        output: ProductIntelligenceOutput,
        front_image_path: Optional[Union[str, Path, Image.Image]] = None,
        image_embedding: Optional[np.ndarray] = None,
        translate_home_decor_for_gateway: bool = False,
        compute_priors: bool = True,
    ) -> TaxonomyGroundingResult:
        """
        Reconcile Gemma 4 prediction against ONDC dictionary + Day-1 SigLIP priors.
        Enforces RET12 vs RET16 domain consistency and flags low-confidence taxonomy
        in audit_telemetry.needs_seller_review.
        """
        priors: List[PriorCandidate] = []
        emb: Optional[np.ndarray] = image_embedding
        if compute_priors and (front_image_path is not None or image_embedding is not None):
            priors, emb = self.compute_stage1_priors(
                image_source=front_image_path,
                image_embedding=image_embedding,
                top_k=3,
            )

        matched_leaf, prior_agreement = self._find_best_dictionary_leaf(output, priors)

        # Consistency guard between lexical textile/food cues and domain_code
        blob = " ".join(
            [
                output.product_identity.apparent_product,
                output.ondc_taxonomy.category,
                output.ondc_taxonomy.subcategory,
            ]
        ).lower()

        domain_consistent = True
        resolved_domain = matched_leaf.domain_code
        if any(tok in blob for tok in _FOOD_KEYWORDS):
            if resolved_domain != ONDCDomainCode.GROCERY:
                domain_consistent = False
                resolved_domain = ONDCDomainCode.GROCERY
        elif any(tok in blob for tok in _TEXTILE_KEYWORDS):
            if resolved_domain != ONDCDomainCode.FASHION:
                domain_consistent = False
                resolved_domain = ONDCDomainCode.FASHION

        base_conf = float(output.ondc_taxonomy.confidence)
        if output.ondc_taxonomy.domain_code != resolved_domain:
            # Penalize raw confidence when Gemma domain disagreed with leaf dictionary
            base_conf = min(base_conf, 0.72)

        grounded_conf = round(max(0.0, min(1.0, base_conf * prior_agreement)), 4)
        if not domain_consistent:
            grounded_conf = min(grounded_conf, 0.70)

        data = output.model_dump()
        data["ondc_taxonomy"]["domain_code"] = resolved_domain.value
        data["ondc_taxonomy"]["category"] = matched_leaf.category
        if not data["ondc_taxonomy"].get("generic_name"):
            data["ondc_taxonomy"]["generic_name"] = matched_leaf.generic_name
        data["ondc_taxonomy"]["confidence"] = grounded_conf

        review_paths = list(data["audit_telemetry"].get("needs_seller_review") or [])
        flagged = False
        if grounded_conf < REVIEW_CONFIDENCE_THRESHOLD:
            if "ondc_taxonomy.category" not in review_paths:
                review_paths.append("ondc_taxonomy.category")
            flagged = True
        data["audit_telemetry"]["needs_seller_review"] = review_paths

        grounded_output = ProductIntelligenceOutput.model_validate(data)
        export_domain = resolved_domain.to_gateway_domain(
            translate_home_decor=translate_home_decor_for_gateway
        )

        return TaxonomyGroundingResult(
            output=grounded_output,
            matched_leaf=matched_leaf,
            export_domain_code=export_domain,
            stage1_priors=priors,
            image_embedding=emb,
            review_flagged_by_grounder=flagged,
        )

