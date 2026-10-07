import time
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from ..inference import generator, preprocess, tensor_to_png, read_validated_upload

router = APIRouter()


@router.post('/face-to-sketch')
async def face_to_sketch(
    file: UploadFile = File(...),
    style: int = Form(0),
):
    if style not in (0, 1, 2):
        raise HTTPException(status_code=400, detail='style must be 0, 1, or 2')

    x = preprocess(await read_validated_upload(file)) * 2.0 - 1.0   # [0,1] -> [-1,1] as in training
    style_ids = np.array([style], dtype=np.int64)

    t0 = time.perf_counter()
    model = generator()
    style_input_name = next(
        (name for name in ('style_id', 'style_idx', 'style')
         if name in model.input_names),
        None,
    )
    if style_input_name is None:
        raise HTTPException(
            status_code=500,
            detail=f'Unsupported sketch model inputs: {model.input_names}',
        )
    out = model.run(photo=x, **{style_input_name: style_ids})
    dt_ms = (time.perf_counter() - t0) * 1000

    png = tensor_to_png((out['sketch'] + 1.0) / 2.0)                # tanh [-1,1] -> [0,1]
    return Response(content=png, media_type='image/png',
                    headers={'X-Inference-Ms': f'{dt_ms:.1f}'})