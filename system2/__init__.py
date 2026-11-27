"""BazarSetu System 2: Visual Catalog Intelligence package."""

from system2.engine import System2Config, System2Engine
from system2.question_generator import apply_mcq_answers, generate_product_mcqs
from system2.schemas import (
    AuditTelemetry,
    ColorProfile,
    ColorSwatch,
    CompositionType,
    MarketplaceEnrichment,
    MaterialDetail,
    ONDCDomainCode,
    ONDCTaxonomy,
    PatternAndCraft,
    PatternCategory,
    PhysicalScale,
    ProductIdentity,
    ProductIntelligenceOutput,
    ProductMCQ,
    QuestionOption,
    StructuralComponent,
    SurfaceFinish,
    TextAndSeals,
)
from system2.siglip_encoder import SigLIPConfig, SigLIPEncoder
from system2.taxonomy_grounder import CURATED_ONDC_LEAVES, TaxonomyGrounder, TaxonomyGroundingResult
from system2.visual_search import (
    DuplicateMatch,
    DuplicateSearchResult,
    VisualDuplicateIndex,
    escalate_label_inspection,
    should_escalate_label,
)

__all__ = [
    "AuditTelemetry",
    "CURATED_ONDC_LEAVES",
    "ColorProfile",
    "ColorSwatch",
    "CompositionType",
    "DuplicateMatch",
    "DuplicateSearchResult",
    "MarketplaceEnrichment",
    "MaterialDetail",
    "ONDCDomainCode",
    "ONDCTaxonomy",
    "PatternAndCraft",
    "PatternCategory",
    "PhysicalScale",
    "ProductIdentity",
    "ProductIntelligenceOutput",
    "ProductMCQ",
    "QuestionOption",
    "SigLIPConfig",
    "SigLIPEncoder",
    "StructuralComponent",
    "SurfaceFinish",
    "System2Config",
    "System2Engine",
    "TaxonomyGrounder",
    "TaxonomyGroundingResult",
    "TextAndSeals",
    "VisualDuplicateIndex",
    "apply_mcq_answers",
    "escalate_label_inspection",
    "generate_product_mcqs",
    "should_escalate_label",
]

