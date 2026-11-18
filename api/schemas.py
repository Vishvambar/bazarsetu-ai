"""
BazarSetu V3.0 - API Data Schemas
Pydantic contracts for Amar's Flutter application.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

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
    fidelity: Optional[FidelityInfo] = None
    latencies: Optional[LatencyInfo] = None
    seller_approval_status: str = "PENDING_SELLER_REVIEW"

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
