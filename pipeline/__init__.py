"""BazarSetu V3.0 Pipeline Stages"""
from pipeline.orchestrator import System1Pipeline, PipelineExecutionResult
from pipeline.stage_a_ingestion import IngestionStage
from pipeline.stage_b_quality import QualityGateStage
from pipeline.stage_cd_segmentation import BiRefNetSegmenter
from pipeline.stage_d_sam2 import SAM2InteractiveRefiner
from pipeline.stage_f_polish import PhotographicPolishStage
from pipeline.stage_g_rendering import MarketplaceRenderingStage
from pipeline.stage_h_fidelity import FidelityVerificationStage

