import os
from fastapi import APIRouter
from ..inference import ONNX_DIR, _MODELS
from ..schemas import HealthResponse

router = APIRouter()

REQUIRED = [
    'task1.onnx', 'classifier.onnx',
    'specialist_salt_pepper.onnx', 'specialist_gaussian_blur.onnx',
    'specialist_occlusion.onnx', 'task3.onnx', 'task4.onnx',
]


@router.get('/health', response_model=HealthResponse)
def health():
    present = {f: os.path.exists(os.path.join(ONNX_DIR, f))
               for f in REQUIRED}
    return HealthResponse(
        status='ok' if all(present.values()) else 'degraded',
        models_loaded=present,
    )