"""Check that every ONNX model the app needs is present, loads in ONNX Runtime
and returns outputs of the expected shape and range.

    python scripts/check_models.py            # uses ./models
    ONNX_DIR=/path/to/models python scripts/check_models.py
"""
import os
import sys

import numpy as np
import onnxruntime as ort

MODELS_DIR = os.environ.get('ONNX_DIR', os.path.join(os.path.dirname(__file__), '..', 'models'))
x01 = np.random.default_rng(0).random((2, 3, 128, 128), dtype=np.float32)

CHECKS = {
    'task1.onnx': ({'corrupted_image': x01}, {'restored_image': (2, 3, 128, 128)}),
    'classifier.onnx': ({'corrupted_image': x01}, {'logits': (2, 4)}),
    'specialist_salt_pepper.onnx': ({'corrupted_image': x01}, {'restored_image': (2, 3, 128, 128)}),
    'specialist_gaussian_blur.onnx': ({'corrupted_image': x01}, {'restored_image': (2, 3, 128, 128)}),
    'specialist_occlusion.onnx': ({'corrupted_image': x01}, {'restored_image': (2, 3, 128, 128)}),
    'task3.onnx': ({'corrupted_image': x01}, {'restored_image': (2, 3, 128, 128), 'routing_weights': (2, 4)}),
    'task4.onnx': ({'photo': x01 * 2 - 1, 'style_idx': np.array([0, 2], dtype=np.int64)}, {'sketch': (2, 3, 128, 128)}),
}

failed = False
for name, (inputs, expected) in CHECKS.items():
    path = os.path.join(MODELS_DIR, name)
    try:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, inputs)))
        for out_name, shape in expected.items():
            assert outs[out_name].shape == shape, f'{out_name} shape {outs[out_name].shape} != {shape}'
        if 'routing_weights' in outs:
            assert np.allclose(outs['routing_weights'].sum(1), 1, atol=1e-5), 'routing weights do not sum to 1'
        ranges = ', '.join(f'{k} [{v.min():.3f}, {v.max():.3f}]' for k, v in outs.items())
        print(f'OK    {name:32s} {ranges}')
    except Exception as e:
        failed = True
        print(f'FAIL  {name:32s} {e}')
sys.exit(1 if failed else 0)
