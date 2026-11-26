"""
BazarSetu V3.0 - System 2 Visual Catalog Intelligence Contract (Pydantic v2).

Locked schema shared between:
  1. Gemma 4 12B Unified (QAT GGUF) tool-choice structured output decoding (GPU 1)
  2. FastAPI System 2 endpoint & ONDC BPP mapper
  3. Golden evaluation harness (eval/system2_eval.py)
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ONDCDomainCode(str, Enum):
    """Internal ONDC domain codes aligned with BPP ondc_mapper.py & Flutter client."""

    GROCERY = "ONDC:RET10"
    FNB = "ONDC:RET11"
    FASHION = "ONDC:RET12"
    BPC = "ONDC:RET13"
    ELECTRONICS = "ONDC:RET14"
    APPLIANCES = "ONDC:RET15"
    HOME_DECOR = "ONDC:RET16"  # Internal code for Crafts, Pottery & Metalware (exportable as RET15)
    AGRICULTURE = "ONDC:RET17"
    HEALTH_WELLNESS = "ONDC:RET18"

    def to_gateway_domain(self, translate_home_decor: bool = False) -> str:
        """Translate internal RET16 to official Beckn v1.2.5 RET15 when exporting."""
        if translate_home_decor and self == ONDCDomainCode.HOME_DECOR:
            return "ONDC:RET15"
        return self.value


class CompositionType(str, Enum):
    SINGLE_ITEM = "single_item"
    MATCHED_SET = "matched_set"
    MULTIPACK = "multipack"
    PRODUCT_WITH_PACKAGING = "product_with_packaging"


class PhysicalScale(str, Enum):
    POCKET_WEARABLE = "pocket_wearable"  # Rings, earrings, wallets
    HANDHELD = "handheld"                # Diyas, bells, cups (< 6 inches)
    TABLETOP = "tabletop"                # Vases, 10-inch idols, brass urlis, kalash
    FLOOR_STANDING = "floor_standing"    # Large planters, tall floor lamps
    DRAPED_FABRIC = "draped_fabric"      # Sarees, dupattas, shawls (multi-meter)


class SurfaceFinish(str, Enum):
    MATTE = "matte"
    GLOSSY = "glossy"
    METALLIC = "metallic"
    OXIDIZED_ANTIQUE = "oxidized_antique"
    RAW_NATURAL = "raw_natural"
    WOVEN_TEXTILE = "woven_textile"
    HAMMERED_MATHAR = "hammered_mathar"  # Pune Tambat Ali copper/brass technique
    GLAZED = "glazed"


class PatternCategory(str, Enum):
    SOLID = "solid"
    GEOMETRIC = "geometric"
    FLORAL = "floral"
    TRADITIONAL_MOTIF = "traditional_motif"
    WARLI_TRIBAL = "warli_tribal"
    STRIPED = "striped"
    CHECKED = "checked"
    ABSTRACT = "abstract"


class ColorSwatch(BaseModel):
    name: str = Field(
        ...,
        description="Descriptive craft color, e.g., 'Terracotta Rust', 'Peacock Green', 'Antique Brass'",
    )
    standard_family: str = Field(
        ...,
        description="Standard ONDC filter chip: Red, Blue, Green, Yellow, Orange, Pink, Purple, Grey, Black, White, Brown, Gold, Silver, Beige, Multicolor",
    )


class ColorProfile(BaseModel):
    primary_color: ColorSwatch
    secondary_colors: List[ColorSwatch] = Field(default_factory=list, max_length=2)
    surface_finish: SurfaceFinish
    color_distribution: str = Field(
        ..., description="e.g., solid, dual_tone, multicolor, ombre, colorblock"
    )


class MaterialDetail(BaseModel):
    material_name: str = Field(
        ...,
        description="Specific craft material, e.g., 'Terracotta Clay', 'Brass', 'Copper', 'Paithani Silk', 'Sheesham Wood'",
    )
    material_family: str = Field(
        ...,
        description="Broad group: Metal, Textile, Wood, Clay_Ceramic, Leather, Stone, Glass, Natural_Fiber, Paper_Bamboo",
    )
    location: str = Field(
        default="main_body",
        description="Where on the product, e.g., 'main_body', 'handle', 'border', 'pallu'",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class PatternAndCraft(BaseModel):
    has_pattern: bool
    pattern_categories: List[PatternCategory] = Field(default_factory=list)
    craft_techniques: List[str] = Field(
        default_factory=list,
        description="e.g., ['pit_fired', 'hand_hammered_mathar', 'zari_brocade', 'hand_carved', 'block_print', 'warli_painted']",
    )
    motif_description: Optional[str] = Field(
        None, description="Short visual description, e.g., 'Peacock and lotus vine border'"
    )


class StructuralComponent(BaseModel):
    component_type: str = Field(
        ...,
        description="handle, lid, clasp, strap, spout, pedestal_base, zipper, tassel",
    )
    detail: str = Field(..., description="e.g., 'Curved brass handles on both sides'")


class TextAndSeals(BaseModel):
    has_visible_text: bool
    brand_or_artisan_mark: Optional[str] = None
    certification_seals: List[str] = Field(
        default_factory=list,
        description="Detected seals: 'Silk Mark', 'Handloom Mark', 'GI Tag', 'India Handmade', 'FSSAI', 'BIS Hallmark'",
    )
    visible_text_snippets: List[str] = Field(default_factory=list, max_length=3)
    text_legibility_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_highres_label_crop: bool = Field(
        default=False,
        description="True only when text or certification seals are visible but too small or blurry to read reliably.",
    )


class ONDCTaxonomy(BaseModel):
    domain_code: ONDCDomainCode
    category: str = Field(
        ..., description="e.g., 'Metalware', 'Pottery & Claycraft', 'Woodcraft', 'Sarees', 'Footwear'"
    )
    subcategory: str = Field(
        ..., description="e.g., 'Copper Kalash', 'Terracotta Diya', 'Paithani Saree', 'Brass Urli'"
    )
    generic_name: str = Field(
        ..., description="Legal Metrology common commodity name, e.g., 'Terracotta Diya'"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class ProductIdentity(BaseModel):
    apparent_product: str = Field(..., description="Plain-language name of the physical object")
    catalog_title_en: str = Field(..., description="English SEO title (40-70 chars)")
    catalog_title_hi: str = Field(..., description="Natural conversational Hindi title in Devanagari")
    description_en: str = Field(
        ..., description="Concise e-commerce storytelling description in English (15-30 words)"
    )
    description_hi: str = Field(
        ..., description="Concise conversational Hindi description in Devanagari (15-25 words)"
    )
    composition_type: CompositionType = CompositionType.SINGLE_ITEM
    set_contents: List[str] = Field(default_factory=list)
    physical_scale: PhysicalScale


class MarketplaceEnrichment(BaseModel):
    target_audience: str = Field(
        default="home_utility",
        description="women, men, unisex, kids, or home_utility",
    )
    occasions: List[str] = Field(default_factory=list, max_length=3)
    search_keywords: List[str] = Field(default_factory=list, max_length=6)


class AuditTelemetry(BaseModel):
    views_processed: List[str] = Field(
        ..., description="Subset of ['front_view', 'detail_view', 'back_or_label_view']"
    )
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    needs_seller_review: List[str] = Field(
        default_factory=list,
        description="Dot-paths for fields below 0.75 confidence, e.g., ['materials[0].material_name']",
    )


class ProductIntelligenceOutput(BaseModel):
    product_id: str
    product_identity: ProductIdentity
    ondc_taxonomy: ONDCTaxonomy
    color_profile: ColorProfile
    materials: List[MaterialDetail] = Field(..., min_length=1)
    pattern_and_craft: PatternAndCraft
    structural_components: List[StructuralComponent] = Field(default_factory=list)
    text_and_seals: TextAndSeals
    marketplace_enrichment: MarketplaceEnrichment
    audit_telemetry: AuditTelemetry
