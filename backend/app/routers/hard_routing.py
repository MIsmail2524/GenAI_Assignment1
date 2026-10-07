import base64
import time
from fastapi import APIRouter, File, Form, UploadFile
from ..inference import (read_validated_upload, classifier, specialist, preprocess,
                         run_image_model, restored_output, image_bytes_to_png,
                         tensor_to_png, softmax)
from ..corruptions import apply_corruption_by_name
from ..schemas import HardRoutingResponse

router = APIRouter()

NAMES = ['clean', 'salt_pepper', 'gaussian_blur', 'occlusion']


@router.post('/hard-routing', response_model=HardRoutingResponse)
async def hard_routing(
    file: UploadFile = File(...),
    force_corruption: str = Form('none'),
    severity: str = Form('medium'),
):
    image_bytes = await read_validated_upload(file)
    x = preprocess(image_bytes)

    applied = None
    applied_sev = None
    if force_corruption and force_corruption != 'none':
        x = apply_corruption_by_name(x, force_corruption, severity=severity)
        applied = force_corruption
        applied_sev = severity
    corrupted_png = (image_bytes_to_png(image_bytes) if applied is None
                     else tensor_to_png(x))

    t0 = time.perf_counter()
    logits = run_image_model(classifier(), x)['logits']
    probs = softmax(logits, axis=1)[0]
    pred_idx = int(probs.argmax())
    pred_name = NAMES[pred_idx]

    if pred_name == 'clean':
        recon = x
        expert = 'identity'
        recon_png = corrupted_png
    else:
        recon = restored_output(
            run_image_model(specialist(pred_name), x))
        expert = pred_name
        recon_png = tensor_to_png(recon)

    dt_ms = (time.perf_counter() - t0) * 1000

    return HardRoutingResponse(
        recon_png_b64=base64.b64encode(recon_png).decode(),
        corrupted_png_b64=base64.b64encode(corrupted_png).decode(),
        probabilities={n: float(p) for n, p in zip(NAMES, probs)},
        predicted=pred_name,
        expert_used=expert,
        applied_corruption=applied,
        applied_severity=applied_sev,
        clean_bypass=pred_name == 'clean',
        inference_ms=dt_ms,
    )