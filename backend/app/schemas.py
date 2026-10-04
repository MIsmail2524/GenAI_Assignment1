from pydantic import BaseModel
from typing import Dict, Optional


class UniversalResponse(BaseModel):
    recon_png_b64: str
    corrupted_png_b64: str
    applied_corruption: Optional[str] = None
    applied_severity: Optional[str] = None
    clean_bypass: bool = False
    inference_ms: float


class HardRoutingResponse(BaseModel):
    recon_png_b64: str
    corrupted_png_b64: str
    probabilities: Dict[str, float]
    predicted: str
    expert_used: str
    applied_corruption: Optional[str] = None
    applied_severity: Optional[str] = None
    clean_bypass: bool = False
    inference_ms: float


class SoftMoEResponse(BaseModel):
    recon_png_b64: str
    corrupted_png_b64: str
    routing_weights: Dict[str, float]
    dominant_expert: str          # NEW
    clean_bypass: bool = False
    applied_corruption: Optional[str] = None
    applied_severity: Optional[str] = None
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]