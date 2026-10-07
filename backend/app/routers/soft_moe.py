import base64
import time
from fastapi import APIRouter, File, Form, UploadFile
from ..inference import (read_validated_upload, soft_moe, preprocess, run_image_model,
                         restored_output, image_bytes_to_png, tensor_to_png)
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
    image_bytes = await read_validated_upload(file)
    x = preprocess(image_bytes)
    if force_corruption and force_corruption != 'none':
        x = apply_corruption_by_name(x, force_corruption, severity=severity)
    corrupted_png = (image_bytes_to_png(image_bytes)
                     if force_corruption in ('none', 'clean')
                     else tensor_to_png(x))

    # The complete soft-MoE ONNX graph (gate + identity branch + 3 experts +
    # weighted sum) runs on every input; the identity branch handles clean images.
    t0 = time.perf_counter()
    out = run_image_model(soft_moe(), x)
    weights = out['routing_weights'][0].tolist()
    recon_png = tensor_to_png(restored_output(out))
    dt_ms = (time.perf_counter() - t0) * 1000

    w = {name: float(w) for name, w in zip(BRANCHES, weights)}
    dominant = max(w, key=w.get)
    clean_bypass = dominant == 'clean'   # identity branch carries most of the weight

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