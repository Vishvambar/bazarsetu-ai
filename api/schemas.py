"""
BazarSetu V3.0 - API Data Schemas
Pydantic contracts for Amar's Flutter application.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from system2.schemas import ProductIntelligenceOutput

class JobStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    ACTION_REQUIRED = "ACTION_REQUIRED"
    FAILED = "FAILED"

class UploadResponse(BaseModel):
    job_id: str
    product_id: str
    status: JobStatus = JobStatus.UPLOADED
    message: str = "Image upload accepted for background processing."
    estimated_wait_seconds: float = 2.5

class FidelityInfo(BaseModel):
    delta_e00: float
    boundary_f1: float
    mask_iou: float
    verified: bool

class LatencyInfo(BaseModel):
    gpu_inference_latency_ms: float
    total_processing_latency_ms: float

class JobStatusResponse(BaseModel):
    job_id: str
    product_id: str
    asset_id: Optional[str] = None
    status: JobStatus
    seller_message: str
    quality_code: str
    created_at_iso: str
    completed_at_iso: Optional[str] = None
    urls: Optional[Dict[str, str]] = None
    bounding_box: Optional[List[int]] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    fidelity: Optional[FidelityInfo] = None
    latencies: Optional[LatencyInfo] = None
    seller_approval_status: str = "PENDING_SELLER_REVIEW"

class BatchUploadResponse(BaseModel):
    batch_id: str
    product_id: str
    total_images: int
    jobs: List[UploadResponse]
    status: JobStatus = JobStatus.UPLOADED
    message: str = "Batch of multi-angle images accepted for background processing."

class TapRefineRequest(BaseModel):
    norm_x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate (0.0 to 1.0)")
    norm_y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate (0.0 to 1.0)")
    is_foreground: bool = Field(True, description="True if foreground tap, False if background tap")

class SellerApprovalRequest(BaseModel):
    seller_approved: bool = Field(..., description="True for 'Sahi Hai', False for rejection")
    feedback_note: Optional[str] = None

class SellerApprovalResponse(BaseModel):
    product_id: str
    asset_id: str
    status: str
    message: str
    audit_file_updated: bool

class HealthResponse(BaseModel):
    status: str
    version: str
    cuda_available: bool
    gpu_count: int
    devices: List[str]
    storage_root: str

class BppUploadImageResponse(BaseModel):
    status: str = "UPLOADED"
    url: str
    filename: str
    image_type: str = "studio"
    storage: str = "system1_product_master"
    asset_id: Optional[str] = None
    catalog_id: Optional[str] = None
    raw_image_url: Optional[str] = None
    studio_image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    quality_code: Optional[str] = "OK"
    seller_message: Optional[str] = "Photo processed into studio listing."

class SyncProcessResponse(BaseModel):
    status: str
    catalog_id: str
    asset_id: str
    raw_image_url: str
    studio_image_url: str
    thumbnail_url: str
    quality_code: str
    seller_message: str
    bounding_box: List[int]
    fidelity: FidelityInfo
    latency_ms: float

class ArtisanListingPayload(BaseModel):
    catalog_id: str = Field(..., description="Unique SKU / catalog ID from Flutter app")
    seller_id: str = Field(default="default_artisan")
    artisan_name: str = Field(default="BazaarSetu Artisan")
    cluster_name: str = Field(default="Chanderi Handloom Cluster")
    gps_coordinates: str = Field(default="24.7118,78.1098")
    area_code: str = Field(default="473446")
    title_en: str = Field(..., description="English product title")
    title_hi: str = Field(default="", description="Hindi product title")
    description_en: str = Field(..., description="English product description")
    description_hi: str = Field(default="", description="Hindi product description")
    craft_category: str = Field(default="Handloom Silk")
    category_id: str = Field(default="ONDC:RET12")
    materials: List[str] = Field(default_factory=list)
    dimensions: str = Field(default="1 unit")
    floor_price: float = Field(..., gt=0)
    fair_price: float = Field(..., gt=0)
    premium_price: float = Field(..., gt=0)
    raw_image_url: Optional[str] = None
    studio_image_url: str = ""
    gallery_images: List[str] = Field(default_factory=list)
    available_qty: int = Field(default=5, ge=0)
    needs_seller_review: List[str] = Field(default_factory=list)
    certification_seals: List[str] = Field(default_factory=list)
    craft_techniques: List[str] = Field(default_factory=list)


class PublishResponse(BaseModel):
    status: str = "PUBLISHED"
    message: str
    catalog_id: str
    artisan: str
    fair_price_inr: float
    raw_image_url: Optional[str] = None
    studio_image_url: Optional[str] = None


class ListingsListResponse(BaseModel):
    count: int
    listings: List[Dict]


class System2JobState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DuplicateCandidateInfo(BaseModel):
    product_id: str
    similarity: float
    title_en: str = ""
    category: str = ""
    domain_code: str = ""
    thumbnail_url: str = ""


class System2AnalyzeRequest(BaseModel):
    product_id: str = Field(..., min_length=1)
    views: Optional[Dict[str, str]] = None
    force: bool = False
    run_duplicate_check: bool = True
    run_escalation: bool = True
    translate_home_decor_for_gateway: bool = False


class System2AnalyzeResponse(BaseModel):
    product_id: str
    status: System2JobState
    deduplicated: bool = False
    message: str


class System2StatusResponse(BaseModel):
    product_id: str
    status: System2JobState
    created_at_iso: str
    completed_at_iso: Optional[str] = None
    error: Optional[str] = None
    escalated: bool = False
    export_domain_code: Optional[str] = None
    needs_seller_review: List[str] = Field(default_factory=list)
    duplicate_matches: List[DuplicateCandidateInfo] = Field(default_factory=list)
    result: Optional[ProductIntelligenceOutput] = None


class VisualDuplicateSearchRequest(BaseModel):
    product_id: Optional[str] = None
    image_path: Optional[str] = None
    threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


class VisualDuplicateSearchResponse(BaseModel):
    product_id: str
    threshold: float
    has_duplicates: bool
    matches: List[DuplicateCandidateInfo] = Field(default_factory=list)


