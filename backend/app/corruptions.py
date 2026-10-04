import math
import cv2
import numpy as np

CORRUPTION_TYPES = ['clean', 'salt_pepper', 'gaussian_blur', 'occlusion']

# Matches Colab training ranges; here we use single fixed severities
# for the interactive demo.
SP_PROB_RANGE      = (0.02, 0.15)
BLUR_KERNELS       = [3, 5, 7]
BLUR_SIGMA_RANGE   = (0.5, 2.5)
OCC_NUM_RECTS_RANGE = (1, 3)
OCC_AREA_FRAC_RANGE = (0.10, 0.35)


def apply_salt_pepper(img, prob, rng):
    out = img.copy()
    h, w = out.shape[:2]
    corrupt = rng.random((h, w)) < prob
    salt    = rng.random((h, w)) < 0.5
    out[corrupt & salt]  = 255
    out[corrupt & ~salt] = 0
    return out


def apply_gaussian_blur(img, k, sigma):
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def sample_rectangles(h, w, n, target_frac, rng):
    total = h * w
    target = target_frac * total
    splits = rng.dirichlet(np.ones(n)) * target
    rects = []
    for area in splits:
        aspect = rng.uniform(0.5, 2.0)
        rw = max(1, min(int(round(math.sqrt(area * aspect))), w))
        rh = max(1, min(int(round(area / rw)), h))
        x0 = int(rng.integers(0, max(1, w - rw + 1)))
        y0 = int(rng.integers(0, max(1, h - rh + 1)))
        rects.append([x0, y0, rw, rh])
    return rects


def apply_occlusion(img, rects):
    out = img.copy()
    for (x0, y0, rw, rh) in rects:
        out[y0:y0 + rh, x0:x0 + rw] = 0
    return out


def apply_corruption_by_name(nchw_float, name, severity='medium', seed=None):
    """nchw_float: (1,3,128,128) float32 in [0,1]. Returns same shape."""
    rng = np.random.default_rng(seed)
    img = (nchw_float[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    h, w = img.shape[:2]

    if name == 'clean':
        out = img
    elif name == 'salt_pepper':
        prob = {'low': 0.03, 'medium': 0.08, 'high': 0.15}[severity]
        out = apply_salt_pepper(img, prob, rng)
    elif name == 'gaussian_blur':
        k, s = {'low': (3, 0.7), 'medium': (5, 1.5), 'high': (7, 2.5)}[severity]
        out = apply_gaussian_blur(img, k, s)
    elif name == 'occlusion':
        n, frac = {'low': (1, 0.10), 'medium': (2, 0.20), 'high': (3, 0.35)}[severity]
        rects = sample_rectangles(h, w, n, frac, rng)
        out = apply_occlusion(img, rects)
    else:
        raise ValueError(name)

    return (out.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]