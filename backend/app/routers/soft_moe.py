import base64
import time
from fastapi import APIRouter, File, Form, UploadFile
from ..inference import (classifier, soft_moe, preprocess, run_image_model,
                         restored_output, image_bytes_to_png, tensor_to_png,
                         softmax)
from ..corruptions import apply_corruption_by_name
from ..schemas import SoftMoEResponse

router = APIRouter()

BRANCHES = ['clean', 'salt_pepper', 'gaussian_blur', 'occlusion']


@router.post('/soft-moe', response_model=SoftMoEResponse)
async def soft_moe_inference(
    file: UploadFile = File(...),
    force_corruption: str = Form('none'),
    severity: str = Form('medium'),
):
    image_bytes = await file.read()
    x = preprocess(image_bytes)
    if force_corruption and force_corruption != 'none':
        x = apply_corruption_by_name(x, force_corruption, severity=severity)
    corrupted_png = (image_bytes_to_png(image_bytes)
                     if force_corruption in ('none', 'clean')
                     else tensor_to_png(x))

    t0 = time.perf_counter()
    clean_bypass = force_corruption == 'clean'
    if force_corruption == 'none':
        logits = run_image_model(classifier(), x)['logits']
        clean_bypass = int(softmax(logits, axis=1)[0].argmax()) == 0

    if clean_bypass:
        weights = [1.0, 0.0, 0.0, 0.0]
        recon_png = image_bytes_to_png(image_bytes)
    else:
        out = run_image_model(soft_moe(), x)
        weights = out['routing_weights'][0].tolist()
        recon_png = tensor_to_png(restored_output(out))
    dt_ms = (time.perf_counter() - t0) * 1000

    w = {name: float(w) for name, w in zip(BRANCHES, weights)}
    dominant = max(w, key=w.get)

    return SoftMoEResponse(
        recon_png_b64=base64.b64encode(recon_png).decode(),
        corrupted_png_b64=base64.b64encode(corrupted_png).decode(),
        routing_weights=w,
        dominant_expert=dominant,
        clean_bypass=clean_bypass,
        applied_corruption=(force_corruption
                            if force_corruption not in ('none', 'clean')
                            else None),
        applied_severity=(severity
                          if force_corruption not in ('none', 'clean')
                          else None),
        inference_ms=dt_ms,
    )