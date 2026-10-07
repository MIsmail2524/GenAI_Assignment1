import io
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

ONNX_DIR = os.environ.get(
    'ONNX_DIR', str(Path(__file__).resolve().parents[2] / 'models'))
IMG_SIZE = 128


class ONNXModel:
    def __init__(self, filename, providers=None):
        path = os.path.join(ONNX_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ONNX model not found: {path}. "
                f"Run scripts/make_dummy_models.py to generate dummies, "
                f"or copy the real .onnx files into models/.")
        providers = providers or ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(path, providers=providers)
        self.input_names  = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.filename = filename

    def run(self, **inputs):
        outs = self.sess.run(self.output_names, inputs)
        return dict(zip(self.output_names, outs))


# Lazy-loaded singletons so a missing model doesn't crash the health check
_MODELS = {}

def _get(name, filename):
    if name not in _MODELS:
        _MODELS[name] = ONNXModel(filename)
    return _MODELS[name]


def universal_ae():
    return _get('universal_ae', 'task1.onnx')


def classifier():
    return _get('classifier', 'classifier.onnx')


def specialist(ctype):
    return _get(f'specialist_{ctype}', f'specialist_{ctype}.onnx')


def soft_moe():
    return _get('soft_moe', 'task3.onnx')


def generator():
    return _get('generator', 'task4.onnx')


# ---------- Image helpers ----------

ALLOWED_TYPES = {'image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp'}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def read_validated_upload(upload):
    """Reject non-image, empty, oversized or undecodable uploads with HTTP 400/413."""
    from fastapi import HTTPException
    if upload.content_type and upload.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=400,
                            detail=f'Unsupported file type {upload.content_type}; use PNG, JPEG, WEBP or BMP.')
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail='Empty file.')
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='File larger than 10 MB.')
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail='File is not a valid image.')
    return data


def preprocess(image_bytes, size=IMG_SIZE):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0     # HWC
    arr = np.transpose(arr, (2, 0, 1))[None, ...]       # NCHW
    return np.ascontiguousarray(arr)


def tensor_to_png(nchw_float):
    arr = np.clip(nchw_float[0].transpose(1, 2, 0), 0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()


def image_bytes_to_png(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def run_image_model(model, image):
    if len(model.input_names) != 1:
        raise ValueError(
            f'Expected one image input for {model.filename}, '
            f'found {model.input_names}')
    return model.run(**{model.input_names[0]: image})


def restored_output(outputs):
    for name in ('recon', 'restored_image'):
        if name in outputs:
            return outputs[name]
    raise KeyError(f'No restored image output found: {list(outputs)}')


def softmax(x, axis=1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)