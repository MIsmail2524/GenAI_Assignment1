import base64
import time
from fastapi import APIRouter, File, Form, UploadFile
from ..inference import (classifier, universal_ae, preprocess,
                         run_image_model, restored_output, softmax,
                         image_bytes_to_png, tensor_to_png)
from ..corruptions import apply_corruption_by_name
from ..schemas import UniversalResponse

router = APIRouter()


@router.post('/universal-restoration', response_model=UniversalResponse)
async def universal_restoration(
    file: UploadFile = File(...),
    force_corruption: str = Form('none'),
    severity: str = Form('medium'),
):
    """Upload any image. Optionally apply a corruption first (for demos),
    then run the universal autoencoder on the result."""
    image_bytes = await file.read()
    x = preprocess(image_bytes)

    # If the user asked for a corruption, apply it before restoring
    if force_corruption and force_corruption not in ('none', 'clean'):
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
        recon_png = image_bytes_to_png(image_bytes)
    else:
        out = run_image_model(universal_ae(), x)
        recon_png = tensor_to_png(restored_output(out))
    dt_ms = (time.perf_counter() - t0) * 1000

    return UniversalResponse(
        recon_png_b64=base64.b64encode(recon_png).decode(),
        corrupted_png_b64=base64.b64encode(corrupted_png).decode(),
        applied_corruption=(force_corruption
                            if force_corruption not in ('none', 'clean')
                            else None),
        applied_severity=(severity
                          if force_corruption not in ('none', 'clean')
                          else None),
        clean_bypass=clean_bypass,
        inference_ms=dt_ms,
    )