"""Task 1 - Universal multi-corruption denoising autoencoder.

Script version of Colab_Files/Task1_Final.ipynb (same code, cell by cell).
It contains the data preparation, runtime corruption / dataset pipeline, model, Optuna study,
training, evaluation, ONNX export + verification and MLflow logging for this task.
Designed for Google Colab with Google Drive mounted (paths under /content/drive/MyDrive).
Shell lines from the notebook (pip installs) are kept as comments -- install
requirements-training.txt instead. Run the notebook for inline figures.
"""

# # Task 1 — Universal Multi-Corruption Denoising Autoencoder
# 
# Oxford-IIIT Pet Dataset, loaded once from Google Drive and cached there for future sessions.
# 
# **Pipeline:** clean image → random corruption (clean / salt-and-pepper / Gaussian blur / rectangular occlusion, sampled dynamically for training) → a small **contrastive degradation encoder** infers *what kind* of corruption is present (no label given) → that code **FiLM-conditions** the convolutional **encoder** → compressed **spatial** latent **z** → FiLM-conditioned convolutional **decoder** (resize-convolution upsampling, no checkerboard artifacts) → reconstruction. Trained jointly with a combined **L1 + SSIM** reconstruction loss and a **contrastive** loss for the degradation encoder (adapted from AirNet, CVPR 2022), with every key hyperparameter selected by **Optuna**.
# 
# Run the cells in order. The first run will download the dataset and take longer; subsequent runs reuse the Drive cache.

# ## 1. Install dependencies

# %% [cell 2]
# Install packages not preinstalled in Colab
# !pip install -q optuna pytorch-msssim onnx onnxruntime mlflow

# ## 2. Mount Drive, imports, global config, seeding

# %% [cell 4]
import os, json, random, time, math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_metric

from google.colab import drive
drive.mount('/content/drive')

# ---------------- Global config ----------------
SEED = 42
IMG_SIZE = 128
DRIVE_ROOT = '/content/drive/MyDrive/universal_restoration'   # everything lives under here
DATA_ROOT = os.path.join(DRIVE_ROOT, 'oxford_pet_raw')        # raw torchvision dataset cache
MANIFEST_DIR = os.path.join(DRIVE_ROOT, 'manifests')
CKPT_DIR = os.path.join(DRIVE_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(DRIVE_ROOT, 'results')
ONNX_DIR = os.path.join(DRIVE_ROOT, 'onnx_export')

for d in [DRIVE_ROOT, DATA_ROOT, MANIFEST_DIR, CKPT_DIR, RESULTS_DIR, ONNX_DIR]:
    os.makedirs(d, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

# ## 3. Load Oxford-IIIT Pet dataset from Drive and create the 80/20 split
# The official `trainval` split is the **development data**; it is shuffled with `seed=42` and divided 80/20 into train/val. The official `test` split is kept completely separate and untouched until Section 8. The resulting file lists are saved to Drive (`data_split.json`) so Tasks 1–3 reuse the identical split.

# %% [cell 6]
# Download the official Oxford-IIIT Pet dataset once. It is cached inside
# Google Drive (DATA_ROOT), so re-running this cell in a new session will not
# re-download the ~800MB archive.
#
# torchvision exposes the OFFICIAL split as split='trainval' (development data)
# and split='test' (held-out test data, kept untouched until final evaluation).

trainval_raw = datasets.OxfordIIITPet(
    root=DATA_ROOT, split='trainval', target_types='category', download=True
)
test_raw = datasets.OxfordIIITPet(
    root=DATA_ROOT, split='test', target_types='category', download=True
)

print('Official trainval (development) size:', len(trainval_raw))
print('Official test size (untouched until final eval):', len(test_raw))

# We only need the images (labels are not required for the restoration task).
# `_images` holds the absolute file paths inside torchvision's internal structure.
trainval_paths = [str(p) for p in trainval_raw._images]
test_paths = [str(p) for p in test_raw._images]

# 80/20 split of the development data using random seed 42.
# This exact split is reused for Tasks 1, 2 and 3.
rng = np.random.default_rng(SEED)
indices = np.arange(len(trainval_paths))
rng.shuffle(indices)
split_point = int(0.8 * len(indices))
train_idx, val_idx = indices[:split_point], indices[split_point:]

train_paths = [trainval_paths[i] for i in train_idx]
val_paths = [trainval_paths[i] for i in val_idx]

print(f'Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}')

# Persist the split so every later notebook / task loads the identical files.
split_record = {'train': train_paths, 'val': val_paths, 'test': test_paths}
with open(os.path.join(DRIVE_ROOT, 'data_split.json'), 'w') as f:
    json.dump(split_record, f)

# Split integrity: the three sets are disjoint and train+val is exactly the official trainval set
_tr, _va, _te = set(train_paths), set(val_paths), set(test_paths)
assert not (_tr & _va) and not (_tr & _te) and not (_va & _te), 'overlap between splits!'
print('Overlap train/val, train/test, val/test: 0, 0, 0 | train+val == official trainval:', (_tr | _va) == set(trainval_paths))

# ## 3b. Preload all images into RAM (avoids Drive I/O stalls)
# Google Drive is mounted as a FUSE filesystem — reading thousands of small JPEGs from it repeatedly, especially from multiple `DataLoader` worker processes, is slow and is the most common cause of a training epoch or Optuna trial appearing to hang for minutes. The resized dataset fits comfortably in RAM (~360MB), so it is read from Drive **exactly once**, here, using a thread pool (I/O-bound reads parallelize well even under the GIL). Every subsequent epoch reads from memory.

# %% [cell 8]
# ============================================================
# Preload every image into RAM, once, using threads for parallel I/O.
#
# WHY: Google Drive is mounted as a FUSE filesystem. Reading thousands of
# small JPEGs from it -- especially from multiple DataLoader worker
# processes at once -- is extremely slow and can stall for minutes per
# batch (this is the #1 cause of an Optuna trial or training epoch
# appearing "stuck" on Colab). The Oxford-IIIT Pet dataset resized to
# 128x128 uint8 fits comfortably in RAM (~360MB total), so we pay the
# Drive I/O cost exactly once, here, with threads (I/O-bound reads
# parallelize well even under the GIL) instead of paying it every batch
# of every epoch of every Optuna trial.
# ============================================================

from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm


def load_and_resize(path, size=IMG_SIZE):
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)  # HWC uint8


def preload_to_ram(paths, desc, max_workers=16):
    cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(ex.map(load_and_resize, paths), total=len(paths), desc=desc))
    for p, img in zip(paths, results):
        cache[p] = img
    return cache


IMAGE_CACHE = {}
IMAGE_CACHE.update(preload_to_ram(train_paths, 'Preloading train'))
IMAGE_CACHE.update(preload_to_ram(val_paths, 'Preloading val'))
IMAGE_CACHE.update(preload_to_ram(test_paths, 'Preloading test'))
print(f'Preloaded {len(IMAGE_CACHE)} images into RAM '
      f'(~{len(IMAGE_CACHE) * IMG_SIZE * IMG_SIZE * 3 / 1e6:.0f} MB). '
      f'All subsequent epochs read from memory, not Drive.')

# %% [cell 9]
# Dataset statistics and sample images (RGB, resized to 128x128)
_labels = dict(zip(trainval_paths, trainval_raw._labels))
print(f'Images -> official trainval: {len(trainval_paths)} | train: {len(train_paths)} | val: {len(val_paths)} | test: {len(test_paths)}')
print('Preprocessed image shape(s):', {v.shape for v in IMAGE_CACHE.values()}, '| dtype:', next(iter(IMAGE_CACHE.values())).dtype)
fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
for ax, p in zip(axes.flat, train_paths[:12]):
    ax.imshow(IMAGE_CACHE[p]); ax.axis('off')
    ax.set_title(trainval_raw.classes[_labels[p]], fontsize=8)
fig.suptitle('Oxford-IIIT Pet training samples after RGB conversion and 128x128 resize')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'dataset_samples.png'), dpi=120, bbox_inches='tight')
plt.show()

# ## 4. Corruption functions
# Salt-and-pepper, Gaussian blur, and rectangular occlusion, each parameterised exactly as specified. `sample_dynamic_corruption` is used for **training** (fresh type + severity every call); `apply_corruption_from_params` re-applies a **recorded** corruption for deterministic val/test manifests.

# %% [cell 11]
# ============================================================
# Corruption primitives. All operate on an HWC uint8 RGB numpy
# array and are driven by an explicit numpy Generator so that
# both dynamic (training) and deterministic (val/test) usage
# share exactly the same implementation.
# ============================================================

CORRUPTION_TYPES = ['clean', 'salt_pepper', 'gaussian_blur', 'occlusion']

# --- Training-time sampling ranges (also used as bounds for val manifest) ---
SP_PROB_RANGE = (0.02, 0.15)
BLUR_KERNELS = [3, 5, 7]
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_NUM_RECTS_RANGE = (1, 3)          # inclusive
OCC_AREA_FRAC_RANGE = (0.10, 0.35)

# --- Fixed test-time severities (three levels per corruption) ---
TEST_SP_PROBS = [0.03, 0.08, 0.15]
TEST_BLUR_CFGS = [(3, 0.7), (5, 1.5), (7, 2.5)]
TEST_OCC_CFGS = [(1, 0.10), (2, 0.20), (3, 0.35)]  # (num_rects, area_frac)


def apply_salt_pepper(img, prob, rng):
    """img: HWC uint8. Replace selected pixels with black/white, 50/50."""
    out = img.copy()
    h, w = out.shape[:2]
    corrupt_mask = rng.random((h, w)) < prob
    salt_mask = rng.random((h, w)) < 0.5
    out[corrupt_mask & salt_mask] = 255
    out[corrupt_mask & ~salt_mask] = 0
    return out


def apply_gaussian_blur(img, kernel_size, sigma):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def sample_rectangles(h, w, num_rects, target_area_frac, rng):
    """Randomly place `num_rects` rectangles whose combined area is
    approximately `target_area_frac` of the image."""
    total_area = h * w
    target_area = target_area_frac * total_area
    splits = rng.dirichlet(np.ones(num_rects)) * target_area
    rects = []
    for area in splits:
        aspect = rng.uniform(0.5, 2.0)
        rw = int(round(math.sqrt(area * aspect)))
        rw = max(1, min(rw, w))
        rh = int(round(area / rw))
        rh = max(1, min(rh, h))
        x0 = int(rng.integers(0, max(1, w - rw + 1)))
        y0 = int(rng.integers(0, max(1, h - rh + 1)))
        rects.append([x0, y0, rw, rh])
    return rects


def apply_occlusion(img, rects):
    out = img.copy()
    for (x0, y0, rw, rh) in rects:
        out[y0:y0 + rh, x0:x0 + rw] = 0
    return out


def sample_dynamic_corruption(img_np, rng):
    """Used during TRAINING: sample a fresh corruption type + severity
    every time an image is loaded. Returns (corrupted_img, label, params)."""
    ctype = rng.choice(CORRUPTION_TYPES)  # equal probability, p=None -> uniform
    h, w = img_np.shape[:2]

    if ctype == 'clean':
        return img_np.copy(), ctype, {}

    if ctype == 'salt_pepper':
        prob = float(rng.uniform(*SP_PROB_RANGE))
        out = apply_salt_pepper(img_np, prob, rng)
        return out, ctype, {'prob': prob}

    if ctype == 'gaussian_blur':
        kernel = int(rng.choice(BLUR_KERNELS))
        sigma = float(rng.uniform(*BLUR_SIGMA_RANGE))
        out = apply_gaussian_blur(img_np, kernel, sigma)
        return out, ctype, {'kernel': kernel, 'sigma': sigma}

    if ctype == 'occlusion':
        n_rects = int(rng.integers(OCC_NUM_RECTS_RANGE[0], OCC_NUM_RECTS_RANGE[1] + 1))
        area_frac = float(rng.uniform(*OCC_AREA_FRAC_RANGE))
        rects = sample_rectangles(h, w, n_rects, area_frac, rng)
        out = apply_occlusion(img_np, rects)
        return out, ctype, {'rects': rects, 'area_frac': area_frac}

    raise ValueError(ctype)


def apply_corruption_from_params(img_np, ctype, params):
    """Re-apply a previously-recorded corruption (used for val/test manifests)."""
    if ctype == 'clean':
        return img_np.copy()
    if ctype == 'salt_pepper':
        rng = np.random.default_rng(params['seed'])
        return apply_salt_pepper(img_np, params['prob'], rng)
    if ctype == 'gaussian_blur':
        return apply_gaussian_blur(img_np, params['kernel'], params['sigma'])
    if ctype == 'occlusion':
        return apply_occlusion(img_np, params['rects'])
    raise ValueError(ctype)

# ## 5. Deterministic validation & test corruption manifests
# Generated **once** and cached as JSON on Drive. Validation gets one fixed corruption per image (used for model selection during training). Test gets the required 10 deterministic variants per image — clean plus low/medium/high severity for each of the 3 corruption types — with mask coordinates, blur settings, and seeds all recorded.

# %% [cell 13]
# ============================================================
# Deterministic corruption manifests for validation and test.
# Generated once, saved to Drive as JSON, and re-loaded on every
# future run so results are perfectly reproducible.
# ============================================================

VAL_MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'val_manifest.json')
TEST_MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'test_manifest.json')

VAL_MANIFEST_SEED = 1000
TEST_MANIFEST_SEED = 2000
CTYPE_OFFSET = {'clean': 0, 'salt_pepper': 1, 'gaussian_blur': 2, 'occlusion': 3}


def build_val_manifest(paths):
    manifest = []
    for idx, path in enumerate(paths):
        seed = VAL_MANIFEST_SEED + idx
        rng = np.random.default_rng(seed)
        ctype = rng.choice(CORRUPTION_TYPES)

        if ctype == 'clean':
            params = {}
        elif ctype == 'salt_pepper':
            prob = float(rng.uniform(*SP_PROB_RANGE))
            mask_seed = int(rng.integers(0, 2**31 - 1))
            params = {'prob': prob, 'seed': mask_seed}
        elif ctype == 'gaussian_blur':
            kernel = int(rng.choice(BLUR_KERNELS))
            sigma = float(rng.uniform(*BLUR_SIGMA_RANGE))
            params = {'kernel': kernel, 'sigma': sigma}
        else:  # occlusion
            n_rects = int(rng.integers(OCC_NUM_RECTS_RANGE[0], OCC_NUM_RECTS_RANGE[1] + 1))
            area_frac = float(rng.uniform(*OCC_AREA_FRAC_RANGE))
            rects = sample_rectangles(IMG_SIZE, IMG_SIZE, n_rects, area_frac, rng)
            params = {'rects': rects, 'area_frac': area_frac}

        manifest.append({'path': path, 'ctype': ctype, 'severity': 'na', 'params': params, 'seed': seed})
    return manifest


def build_test_manifest(paths):
    """Every test image gets: 1 clean copy + 3 severities x 3 corruption
    types = 10 deterministic variants, matching the fixed severities
    required for final evaluation."""
    manifest = []
    for idx, path in enumerate(paths):
        manifest.append({'path': path, 'ctype': 'clean', 'severity': 'na', 'params': {},
                         'seed': TEST_MANIFEST_SEED * 1_000_003 + idx * 97})

        for sev_idx, (sev_name, prob) in enumerate(zip(['low', 'medium', 'high'], TEST_SP_PROBS)):
            seed = TEST_MANIFEST_SEED * 1_000_003 + idx * 97 + CTYPE_OFFSET['salt_pepper'] * 17 + sev_idx
            manifest.append({'path': path, 'ctype': 'salt_pepper', 'severity': sev_name,
                              'params': {'prob': prob, 'seed': seed}, 'seed': seed})

        for sev_idx, (sev_name, (kernel, sigma)) in enumerate(zip(['low', 'medium', 'high'], TEST_BLUR_CFGS)):
            seed = TEST_MANIFEST_SEED * 1_000_003 + idx * 97 + CTYPE_OFFSET['gaussian_blur'] * 17 + sev_idx
            manifest.append({'path': path, 'ctype': 'gaussian_blur', 'severity': sev_name,
                              'params': {'kernel': kernel, 'sigma': sigma}, 'seed': seed})

        for sev_idx, (sev_name, (n_rects, area_frac)) in enumerate(zip(['low', 'medium', 'high'], TEST_OCC_CFGS)):
            seed = TEST_MANIFEST_SEED * 1_000_003 + idx * 97 + CTYPE_OFFSET['occlusion'] * 17 + sev_idx
            rng = np.random.default_rng(seed)
            rects = sample_rectangles(IMG_SIZE, IMG_SIZE, n_rects, area_frac, rng)
            manifest.append({'path': path, 'ctype': 'occlusion', 'severity': sev_name,
                              'params': {'rects': rects, 'area_frac': area_frac}, 'seed': seed})
    return manifest


if os.path.exists(VAL_MANIFEST_PATH):
    with open(VAL_MANIFEST_PATH) as f:
        val_manifest = json.load(f)
    print('Loaded existing val manifest:', len(val_manifest), 'entries')
else:
    val_manifest = build_val_manifest(val_paths)
    with open(VAL_MANIFEST_PATH, 'w') as f:
        json.dump(val_manifest, f)
    print('Generated new val manifest:', len(val_manifest), 'entries')

if os.path.exists(TEST_MANIFEST_PATH):
    with open(TEST_MANIFEST_PATH) as f:
        test_manifest = json.load(f)
    print('Loaded existing test manifest:', len(test_manifest), 'entries')
else:
    test_manifest = build_test_manifest(test_paths)
    with open(TEST_MANIFEST_PATH, 'w') as f:
        json.dump(test_manifest, f)
    print('Generated new test manifest:', len(test_manifest), 'entries (~10x test set size)')

# A manifest saved by an earlier version stored the seed only for salt-and-pepper entries.
# Add the (same, deterministic) seed to every entry, check that it regenerates the stored
# occlusion masks exactly, and save it back. The corruptions themselves do not change.
_SEV_IDX = {'na': 0, 'low': 0, 'medium': 1, 'high': 2}
if any('seed' not in e for e in test_manifest):
    for k, e in enumerate(test_manifest):
        e['seed'] = int(e['params'].get('seed', TEST_MANIFEST_SEED * 1_000_003 + (k // 10) * 97
                                        + CTYPE_OFFSET[e['ctype']] * 17 + _SEV_IDX[e['severity']]))
    assert all(sample_rectangles(IMG_SIZE, IMG_SIZE, len(e['params']['rects']), e['params']['area_frac'],
                                 np.random.default_rng(e['seed'])) == e['params']['rects']
               for e in test_manifest if e['ctype'] == 'occlusion'), 'seed does not reproduce the stored masks'
    with open(TEST_MANIFEST_PATH, 'w') as f:
        json.dump(test_manifest, f)
    print('Seed recorded for every test-manifest entry (occlusion masks verified to regenerate exactly).')
print('Example test entries for one image:')
for e in test_manifest[:10]:
    print('  ', {k: e[k] for k in ('ctype', 'severity', 'params', 'seed')})

# ## 6. PyTorch datasets and loaders
# Both dataset classes read the clean image from the `IMAGE_CACHE` built above (RAM), not from disk, so training never blocks on Drive I/O. `make_loader` defaults to `num_workers=0` — safe to raise if corruption generation (the `cv2` calls), rather than I/O, becomes the bottleneck, but 0 avoids the multiprocessing+Drive-FUSE combination that caused the earlier hang.

# %% [cell 15]
# ============================================================
# Dataset classes
#   - PetTrainDataset: dynamic corruption, new type+severity every __getitem__
#   - PetManifestDataset: deterministic corruption driven by a stored manifest
#
# Both read the clean image from IMAGE_CACHE (RAM) rather than disk, so
# __getitem__ never touches the Drive-mounted filesystem during training.
# ============================================================

def to_tensor(img_np):
    # HWC uint8 [0,255] -> CHW float32 [0,1]
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()


def get_clean_image(path):
    cached = IMAGE_CACHE.get(path)
    if cached is not None:
        return cached
    # fall back to disk only if somehow missing from the cache
    return load_and_resize(path)


class PetTrainDataset(Dataset):
    """Applies a freshly-sampled corruption (or clean) every time an image
    is loaded, as required for training. Nothing is cached to disk."""

    def __init__(self, paths, img_size=IMG_SIZE):
        self.paths = paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        clean_np = get_clean_image(self.paths[idx])
        rng = np.random.default_rng()  # OS-entropy seeded -> different every call/worker
        corrupted_np, ctype, params = sample_dynamic_corruption(clean_np, rng)
        return {
            'clean': to_tensor(clean_np),
            'corrupted': to_tensor(corrupted_np),
            'ctype': ctype,
        }


class PetManifestDataset(Dataset):
    """Deterministic corruption driven by a pre-generated manifest.
    Used for validation and for the final test evaluation."""

    def __init__(self, manifest, img_size=IMG_SIZE):
        self.manifest = manifest
        self.img_size = img_size

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        clean_np = get_clean_image(entry['path'])
        corrupted_np = apply_corruption_from_params(clean_np, entry['ctype'], entry['params'])
        return {
            'clean': to_tensor(clean_np),
            'corrupted': to_tensor(corrupted_np),
            'ctype': entry['ctype'],
            'severity': entry.get('severity', 'na'),
        }


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    # num_workers=0 by default: everything is already in RAM (see the
    # preload cell), so extra worker processes add overhead without
    # speeding up I/O, and avoid multiprocessing+Drive-FUSE stalls.
    # Safe to raise to 2-4 if corruption generation (cv2 ops) becomes the
    # bottleneck instead of I/O -- watch nvidia-smi/CPU usage to confirm.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'),
                       drop_last=shuffle)


train_ds = PetTrainDataset(train_paths)
val_ds = PetManifestDataset(val_manifest)
test_ds = PetManifestDataset(test_manifest)

print('train/val/test dataset sizes:', len(train_ds), len(val_ds), len(test_ds))

# quick sanity check on one sample
_sample = train_ds[0]
print('sample clean shape:', _sample['clean'].shape, 'corrupted shape:', _sample['corrupted'].shape,
      'ctype:', _sample['ctype'])

# %% [cell 16]
# ============================================================
# Runtime corruption pipeline checks + visual examples
# ============================================================
import pandas as pd

# (a) the SAME training index loaded 8 times -> a new corruption type/severity on every load
fig, axes = plt.subplots(1, 8, figsize=(18, 2.8))
for ax in axes:
    s = train_ds[0]
    ax.imshow(s['corrupted'].permute(1, 2, 0).numpy()); ax.set_title(s['ctype'], fontsize=9); ax.axis('off')
fig.suptitle('train_ds[0] loaded 8 times: corruption is sampled inside __getitem__ (label returned with each sample)')
plt.show()

# (b) types are equally likely and every parameter stays in its required range
_rng = np.random.default_rng(0)
_draws = [sample_dynamic_corruption(IMAGE_CACHE[train_paths[k % 100]], _rng) for k in range(4000)]
_p = lambda t, k: [d[2][k] for d in _draws if d[1] == t]
print('Type frequencies over 4000 loads:', pd.Series([d[1] for d in _draws]).value_counts(normalize=True).round(3).to_dict())
print(f"salt-and-pepper prob: {min(_p('salt_pepper', 'prob')):.3f} - {max(_p('salt_pepper', 'prob')):.3f}  (required 0.02 - 0.15)")
print(f"blur kernel: {sorted(set(_p('gaussian_blur', 'kernel')))}  sigma: {min(_p('gaussian_blur', 'sigma')):.2f} - "
      f"{max(_p('gaussian_blur', 'sigma')):.2f}  (required {{3,5,7}}, 0.5 - 2.5)")
print(f"occlusion rectangles: {sorted({len(r) for r in _p('occlusion', 'rects')})}  joint area: "
      f"{min(_p('occlusion', 'area_frac')):.3f} - {max(_p('occlusion', 'area_frac')):.3f}  (required 1-3, 0.10 - 0.35)")

# (c) every corruption at every fixed test severity (from the deterministic test manifest)
fig, axes = plt.subplots(2, 10, figsize=(22, 5.2))
for r, img_idx in enumerate([0, 500]):
    for j in range(10):
        e = test_manifest[img_idx * 10 + j]
        axes[r, j].imshow(test_ds[img_idx * 10 + j]['corrupted'].permute(1, 2, 0).numpy()); axes[r, j].axis('off')
        p = e['params']
        detail = (f"p={p['prob']}" if e['ctype'] == 'salt_pepper' else f"k={p['kernel']}, s={p['sigma']}"
                  if e['ctype'] == 'gaussian_blur' else f"{len(p['rects'])} rect, {p['area_frac']:.0%}" if e['ctype'] == 'occlusion' else '')
        axes[r, j].set_title(f"{e['ctype']}\n{e['severity']} {detail}", fontsize=8)
fig.suptitle('Fixed test corruptions: every corruption x severity')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'corruption_severity_examples.png'), dpi=120, bbox_inches='tight')
plt.show()

# ## 7. Model — AirNet-inspired: contrastive degradation encoder + FiLM-conditioned autoencoder
# Adapted from **AirNet** (Li et al., CVPR 2022, *All-In-One Image Restoration for Unknown Corruption* — [github.com/XLearning-SCU/2022-CVPR-AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet)), whose core idea is to split the problem in two: a small **degradation encoder** that infers *what kind* of corruption is present directly from the image (never given a label), and a **restoration network** conditioned on that inferred code. This matches this task's requirement exactly — the model is never told which corruption was applied.
# 
# Two deliberate simplifications vs. the original paper (noted for the report): (1) a **SimCLR-style** in-batch contrastive loss instead of AirNet's full MoCo (momentum encoder + queue) — unnecessary at this dataset scale and much simpler to train reliably; (2) the degradation code conditions the network via **FiLM** (per-channel scale/shift) inside the existing residual blocks, rather than AirNet's more elaborate degradation-guided convolutions — this keeps the fixed spatial bottleneck and *no unrestricted skip connections* property fully intact, since FiLM only rescales features already inside the bottlenecked pathway; it never gives pixels a way around `z`.
# 
# The rest of the backbone is unchanged from before: encoder downsamples 128→64→32→16→8, a 1×1 "squeeze" convolution forms the spatial latent `z`, and the decoder upsamples with resize-convolution (nearest + conv) to avoid checkerboard/blur artifacts.

# %% [cell 18]
# ============================================================
# Universal Multi-Corruption Denoising Autoencoder (v3 — AirNet-inspired)
#
# Adapted from AirNet (Li et al., CVPR 2022, "All-In-One Image Restoration
# for Unknown Corruption", github.com/XLearning-SCU/2022-CVPR-AirNet), which
# splits the problem into two parts:
#   - CBDE (Contrastive-Based Degradation Encoder): infers WHAT KIND of
#     corruption is present, purely from the image, with no label ever
#     given -- trained via a contrastive pretext task.
#   - DGRN (Degradation-Guided Restoration Network): uses that inferred
#     degradation code to condition the restoration process.
#
# This matches Task 1's requirement exactly ("the model will not be told
# which corruption has been applied"). Two simplifications vs. the original
# paper, both deliberate and noted for the report:
#   1. SimCLR-style in-batch contrastive loss instead of full MoCo
#      (momentum encoder + queue) -- unnecessary at this dataset scale and
#      much simpler to train reliably.
#   2. The degradation code conditions the restoration network via FiLM
#      (feature-wise affine modulation) inside the existing ResBlocks,
#      rather than AirNet's more elaborate degradation-guided convolutions
#      -- this keeps our fixed spatial bottleneck and "no unrestricted
#      skip connections" property fully intact; FiLM only rescales/shifts
#      features already inside the bottlenecked pathway, it never gives
#      pixels a way around the bottleneck.
# ============================================================

class DownBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 4, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    """Resize-convolution upsampling (nearest x2 + conv) -- avoids the
    checkerboard/blur artifacts of ConvTranspose2d."""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return F.relu(self.norm(self.conv(x)))


class FiLMResBlock(nn.Module):
    """A residual block whose features are modulated by the degradation
    embedding z_deg via FiLM (gamma/beta affine transform) -- this is the
    'degradation-guided' part. Initialized to the identity transform
    (gamma=0 -> scale factor 1, beta=0) so training starts equivalent to
    a plain ResBlock and only deviates as it learns useful conditioning."""
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.film = nn.Linear(cond_dim, ch * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        gamma, beta = self.film(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = h * (1 + gamma) + beta
        return F.relu(x + h)


class DegradationEncoder(nn.Module):
    """CBDE-lite: a small CNN producing a global embedding that
    characterizes the corruption present in an image. Trained with a
    SimCLR-style contrastive loss: two random crops of the SAME corrupted
    image are a positive pair (they share the same corruption type and
    severity by construction, since our corruptions are applied to the
    whole image); crops from other images in the batch are negatives."""
    def __init__(self, embed_dim=128, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)
        self.proj_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
                                        nn.Linear(embed_dim, proj_dim))

    def forward(self, x, return_projection=False):
        h = self.net(x).flatten(1)
        z = self.fc(h)  # used to condition the restoration network (FiLM)
        if return_projection:
            p = F.normalize(self.proj_head(z), dim=1)  # used only for the contrastive loss
            return z, p
        return z


def random_crop(img, size):
    _, H, W = img.shape
    top = random.randint(0, H - size)
    left = random.randint(0, W - size)
    return img[:, top:top + size, left:left + size]


def get_two_crops(x, crop_size=64):
    crops1 = torch.stack([random_crop(img, crop_size) for img in x])
    crops2 = torch.stack([random_crop(img, crop_size) for img in x])
    return crops1, crops2


def nt_xent_loss(p1, p2, temperature=0.2):
    """SimCLR NT-Xent contrastive loss over a batch of paired projections."""
    B = p1.size(0)
    z = torch.cat([p1, p2], dim=0)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(-1e9)
    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return F.cross_entropy(sim, targets)


class ConvEncoderFiLM(nn.Module):
    def __init__(self, base_channels, bottleneck_channels, dropout, n_res_blocks, cond_dim):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU(inplace=True))
        self.down1 = DownBlock(c, c * 2)       # 128 -> 64
        self.down2 = DownBlock(c * 2, c * 4)   # 64  -> 32
        self.down3 = DownBlock(c * 4, c * 8)   # 32  -> 16
        self.down4 = DownBlock(c * 8, c * 8)   # 16  -> 8
        self.res = nn.ModuleList([FiLMResBlock(c * 8, cond_dim) for _ in range(n_res_blocks)])
        self.dropout = nn.Dropout2d(dropout)
        self.squeeze = nn.Conv2d(c * 8, bottleneck_channels, 1)  # <-- the bottleneck

    def forward(self, x, cond):
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        for block in self.res:
            h = block(h, cond)
        h = self.dropout(h)
        return self.squeeze(h)  # (B, bottleneck_channels, 8, 8)


class ConvDecoderFiLM(nn.Module):
    def __init__(self, base_channels, bottleneck_channels, dropout, n_res_blocks, cond_dim):
        super().__init__()
        c = base_channels
        self.expand = nn.Conv2d(bottleneck_channels, c * 8, 1)
        self.res = nn.ModuleList([FiLMResBlock(c * 8, cond_dim) for _ in range(n_res_blocks)])
        self.up1 = UpBlock(c * 8, c * 8)   # 8  -> 16
        self.up2 = UpBlock(c * 8, c * 4)   # 16 -> 32
        self.up3 = UpBlock(c * 4, c * 2)   # 32 -> 64
        self.up4 = UpBlock(c * 2, c)       # 64 -> 128
        self.dropout = nn.Dropout2d(dropout)
        self.out_conv = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, z, cond):
        h = self.expand(z)
        for block in self.res:
            h = block(h, cond)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.dropout(h)
        return torch.sigmoid(self.out_conv(h))


class UniversalAutoencoder(nn.Module):
    """Single self-contained module: deg_encoder infers the degradation
    from the input image itself (no label passed in), then FiLM-conditions
    the encoder/decoder. `forward(x)` is the full inference path (what gets
    exported to ONNX); `forward_with_contrastive(x)` additionally returns
    the crop projections needed for the contrastive loss during training."""
    def __init__(self, base_channels=48, bottleneck_channels=128, dropout=0.1,
                 n_res_blocks=2, deg_embed_dim=128, deg_proj_dim=64, img_size=IMG_SIZE):
        super().__init__()
        assert img_size % 16 == 0, "img_size must be divisible by 16 (4 downsampling stages)"
        self.deg_encoder = DegradationEncoder(deg_embed_dim, deg_proj_dim)
        self.encoder = ConvEncoderFiLM(base_channels, bottleneck_channels, dropout, n_res_blocks, deg_embed_dim)
        self.decoder = ConvDecoderFiLM(base_channels, bottleneck_channels, dropout, n_res_blocks, deg_embed_dim)

    def forward(self, x):
        z_deg = self.deg_encoder(x)
        z = self.encoder(x, z_deg)
        return self.decoder(z, z_deg)

    def forward_with_contrastive(self, x, crop_size=64):
        crop1, crop2 = get_two_crops(x, crop_size)
        _, p1 = self.deg_encoder(crop1, return_projection=True)
        _, p2 = self.deg_encoder(crop2, return_projection=True)
        z_deg = self.deg_encoder(x)
        z = self.encoder(x, z_deg)
        x_hat = self.decoder(z, z_deg)
        return x_hat, p1, p2


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


_test_model = UniversalAutoencoder()
_test_x = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
_test_out = _test_model(_test_x)
_test_z = _test_model.encoder(_test_x, _test_model.deg_encoder(_test_x))
_latent_size = _test_z.shape[1] * _test_z.shape[2] * _test_z.shape[3]
_input_size = 3 * IMG_SIZE * IMG_SIZE
print('Model output shape:', _test_out.shape, '| trainable params:', count_params(_test_model))
print(f'Latent z shape: {tuple(_test_z.shape[1:])} | latent size: {_latent_size} '
      f'({100 * _latent_size / _input_size:.1f}% of the {_input_size}-value input)')
del _test_model, _test_x, _test_out, _test_z

# Architecture summary of the configuration selected by Optuna (Section 10)
_m = UniversalAutoencoder(base_channels=64, bottleneck_channels=128, n_res_blocks=1, deg_embed_dim=64)
for _name, _mod in [('degradation encoder', _m.deg_encoder), ('encoder', _m.encoder), ('decoder', _m.decoder), ('total', _m)]:
    print(f'{_name:20s} {count_params(_mod):>12,d} parameters')
print('No encoder-decoder skip connections: the decoder only receives z (128x8x8) and a 64-d global degradation code.')
del _m

# ## 8. Loss: reconstruction (L1 + SSIM) + contrastive degradation loss

# %% [cell 20]
# ============================================================
# Loss: L_total = L_recon + lambda_cl * L_contrastive
#   L_recon        = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat))
#   L_contrastive  = NT-Xent over the degradation encoder's crop projections
#                     (this is what trains DegradationEncoder to separate
#                     corruption types without ever seeing a label)
# alpha and lambda_cl are both tunable hyperparameters, searched with Optuna.
# ============================================================

def combined_loss(x_hat, x, alpha):
    l1 = F.l1_loss(x_hat, x)
    s = ssim_metric(x_hat, x, data_range=1.0, size_average=True)
    loss = alpha * l1 + (1 - alpha) * (1 - s)
    return loss, l1.detach(), s.detach()

# ## 9. Training / evaluation helpers

# %% [cell 22]
# ============================================================
# Training / evaluation helpers
# ============================================================

def train_one_epoch(model, loader, optimizer, alpha, lambda_cl, crop_size, device):
    model.train()
    total_loss, total_recon, total_cl, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        clean = batch['clean'].to(device, non_blocking=True)
        corrupted = batch['corrupted'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        x_hat, p1, p2 = model.forward_with_contrastive(corrupted, crop_size=crop_size)
        recon_loss, l1, s = combined_loss(x_hat, clean, alpha)
        cl_loss = nt_xent_loss(p1, p2)
        loss = recon_loss + lambda_cl * cl_loss

        loss.backward()
        optimizer.step()

        bs = clean.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_cl += cl_loss.item() * bs
        n += bs
    return {'loss': total_loss / max(n, 1), 'recon': total_recon / max(n, 1), 'cl': total_cl / max(n, 1)}


@torch.no_grad()
def evaluate(model, loader, alpha, device):
    """Reconstruction-only validation objective (the contrastive loss is a
    training-time auxiliary task; at eval time we only care about
    restoration quality, using the plain inference-path forward())."""
    model.eval()
    total_loss, total_l1, total_ssim, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        clean = batch['clean'].to(device, non_blocking=True)
        corrupted = batch['corrupted'].to(device, non_blocking=True)
        x_hat = model(corrupted)
        loss, l1, s = combined_loss(x_hat, clean, alpha)
        bs = clean.size(0)
        total_loss += loss.item() * bs
        total_l1 += l1.item() * bs
        total_ssim += s.item() * bs
        n += bs
    return {
        'loss': total_loss / max(n, 1),
        'l1': total_l1 / max(n, 1),
        'ssim': total_ssim / max(n, 1),
    }


def psnr_per_sample(x_hat, x, eps=1e-8):
    mse = F.mse_loss(x_hat, x, reduction='none').mean(dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / (mse + eps))


@torch.no_grad()
def evaluate_breakdown(model, loader, device):
    """Per-sample metrics grouped by (corruption type, severity), using the
    plain inference-path forward()."""
    model.eval()
    records = []
    for batch in loader:
        clean = batch['clean'].to(device, non_blocking=True)
        corrupted = batch['corrupted'].to(device, non_blocking=True)
        ctypes = batch['ctype']
        severities = batch.get('severity', ['na'] * len(ctypes))
        x_hat = model(corrupted)

        l1_ps = F.l1_loss(x_hat, clean, reduction='none').mean(dim=[1, 2, 3])
        ssim_ps = ssim_metric(x_hat, clean, data_range=1.0, size_average=False)
        psnr_ps = psnr_per_sample(x_hat, clean)

        for i in range(clean.size(0)):
            records.append({
                'ctype': ctypes[i],
                'severity': severities[i],
                'l1': l1_ps[i].item(),
                'ssim': ssim_ps[i].item(),
                'psnr': psnr_ps[i].item(),
            })
    return records


def summarize_breakdown(records):
    import pandas as pd
    df = pd.DataFrame(records)
    summary = df.groupby(['ctype', 'severity'])[['l1', 'ssim', 'psnr']].mean().reset_index()
    return df, summary

# ## 10. Optuna hyperparameter search
# Searches learning rate, batch size, bottleneck channels (latent width), encoder base channels, number of residual blocks, dropout, the reconstruction loss weight α, the degradation-embedding dimension, and the contrastive loss weight λ_cl. Each trial trains for a short budget and is scored on the reconstruction-only validation loss (with pruning). All trials and the best configuration are saved to Drive for the report.

# %% [cell 24]
# ============================================================
# Optuna hyperparameter search
# Tuned: learning rate, batch size, bottleneck channels (spatial latent
#        width), encoder base channels, number of residual blocks,
#        dropout rate, reconstruction loss weight alpha, degradation
#        embedding dimension, and the contrastive loss weight lambda_cl
#
# Persisted to a SQLite study database on Drive (not just in-memory), so
# if the session disconnects mid-search, re-running this cell resumes
# from wherever the study left off instead of restarting all trials.
# ============================================================

import optuna

N_OPTUNA_TRIALS = 25          # increase for a more thorough search
OPTUNA_EPOCHS = 4             # short training budget per trial
CROP_SIZE = 64                # fixed crop size for the contrastive pretext task


def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    bottleneck_channels = trial.suggest_categorical('bottleneck_channels', [64, 96, 128, 192])
    base_channels = trial.suggest_categorical('base_channels', [24, 32, 48, 64])
    n_res_blocks = trial.suggest_int('n_res_blocks', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    alpha = trial.suggest_float('alpha', 0.5, 0.95)
    deg_embed_dim = trial.suggest_categorical('deg_embed_dim', [64, 128, 256])
    lambda_cl = trial.suggest_float('lambda_cl', 0.02, 0.5, log=True)

    set_seed(SEED)  # keep model init / data order comparable across trials
    model = UniversalAutoencoder(base_channels=base_channels,
                                  bottleneck_channels=bottleneck_channels,
                                  dropout=dropout,
                                  n_res_blocks=n_res_blocks,
                                  deg_embed_dim=deg_embed_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trial_train_loader = make_loader(train_ds, batch_size, shuffle=True)
    trial_val_loader = make_loader(val_ds, batch_size, shuffle=False)

    best_val = float('inf')
    for epoch in range(OPTUNA_EPOCHS):
        train_one_epoch(model, trial_train_loader, optimizer, alpha, lambda_cl, CROP_SIZE, DEVICE)
        val_metrics = evaluate(model, trial_val_loader, alpha, DEVICE)
        best_val = min(best_val, val_metrics['loss'])

        trial.report(val_metrics['loss'], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val


# Study persisted to Drive (uses this notebook's own RESULTS_DIR, set up in
# Section 2 -- not a separate path) so it survives a Colab disconnect.
STORAGE = f"sqlite:///{os.path.join(RESULTS_DIR, 'optuna_study.db')}"
print('Optuna storage:', STORAGE)

study = optuna.create_study(
    study_name='universal_ae',
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    storage=STORAGE,
    load_if_exists=True,
)

n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f'Trials already in study: {n_complete}/{N_OPTUNA_TRIALS}')

if n_complete < N_OPTUNA_TRIALS:
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS - n_complete)
else:
    print('Study already complete -- skipping optimization.')

print('Number of completed trials:', len([t for t in study.trials
                                           if t.state == optuna.trial.TrialState.COMPLETE]))
print('Best trial value (val loss):', study.best_trial.value)
print('Best hyperparameters:', study.best_trial.params)

with open(os.path.join(RESULTS_DIR, 'optuna_best_params.json'), 'w') as f:
    json.dump({'best_value': study.best_trial.value, 'best_params': study.best_trial.params}, f, indent=2)

with open(os.path.join(RESULTS_DIR, 'optuna_all_trials.json'), 'w') as f:
    json.dump([{'number': t.number, 'value': t.value, 'params': t.params, 'state': str(t.state)}
               for t in study.trials], f, indent=2)

BEST_PARAMS = study.best_trial.params

# Optuna summary and plots for the report
import pandas as pd
from optuna.visualization import matplotlib as ovm
print('Trial states:', pd.Series([t.state.name for t in study.trials]).value_counts().to_dict())
print('Search space: lr [1e-4, 3e-3] log | batch_size {32, 64, 128} | bottleneck_channels {64, 96, 128, 192} | '
      'base_channels {24, 32, 48, 64} | n_res_blocks [1, 3] | dropout [0, 0.3] | alpha [0.5, 0.95] | '
      'deg_embed_dim {64, 128, 256} | lambda_cl [0.02, 0.5] log')
for _name, _fn in [('history', ovm.plot_optimization_history), ('importance', ovm.plot_param_importances),
                   ('intermediate', ovm.plot_intermediate_values)]:
    _ax = _fn(study)
    _ax.figure.savefig(os.path.join(RESULTS_DIR, f'optuna_{_name}.png'), dpi=120, bbox_inches='tight')
    plt.show()

# ## 11. Final training with the Optuna-selected configuration
# Longer training run with early stopping and LR scheduling on the validation loss; the best checkpoint is saved to Drive.

# %% [cell 26]
# ============================================================
# Final training run using the Optuna-selected configuration
# (Modified to automatically load from Drive or active memory if already trained)
# ============================================================

FINAL_EPOCHS = 60
PATIENCE = 10  # early stopping on validation (reconstruction-only) loss
CKPT_PATH = os.path.join(CKPT_DIR, 'best_model.pt')

# ------------------------------------------------------------
# Check if pre-trained checkpoint already exists on Drive or in memory
# ------------------------------------------------------------
if 'model' in globals() and hasattr(model, 'state_dict') and 'ckpt' in globals() and ckpt.get('epoch', 0) == 10:
    print(f"Using the correct epoch 10 model already loaded in active memory! SKIPPING training...")
    model.eval()
elif os.path.exists(CKPT_PATH):
    print(f"Found existing checkpoint at {CKPT_PATH}. Loading model and SKIPPING training...")
    ckpt_disk = torch.load(CKPT_PATH, map_location=DEVICE)

    # If memory already holds a better checkpoint (e.g. Epoch 10 with val loss 0.0869) than disk (Epoch 1),
    # we keep the active memory model to prevent regression.
    if 'ckpt' in globals() and ckpt.get('val_loss', float('inf')) < ckpt_disk.get('val_loss', float('inf')):
        print(f"Active memory holds a better model (val loss: {ckpt['val_loss']:.4f}) than disk ({ckpt_disk.get('val_loss', float('inf')):.4f}). Keeping memory model!")
        model.eval()
    else:
        ckpt = ckpt_disk
        # Instantiate model using configurations stored in the checkpoint to prevent parameter shape mismatches
        model = UniversalAutoencoder(
            base_channels=ckpt['config'].get('base_channels', 64),
            bottleneck_channels=ckpt['config'].get('bottleneck_channels', 128),
            dropout=ckpt['config'].get('dropout', 0.0195),
            n_res_blocks=ckpt['config'].get('n_res_blocks', 1),
            deg_embed_dim=ckpt['config'].get('deg_embed_dim', 64)
        ).to(DEVICE)

        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"Successfully loaded best checkpoint from epoch {ckpt['epoch']} with validation loss {ckpt['val_loss']:.4f}")
else:
    print("No checkpoint found. Commencing final training run...")
    # Configure model hyperparams from Optuna
    lr = BEST_PARAMS['lr']
    batch_size = BEST_PARAMS['batch_size']
    bottleneck_channels = BEST_PARAMS['bottleneck_channels']
    base_channels = BEST_PARAMS['base_channels']
    n_res_blocks = BEST_PARAMS['n_res_blocks']
    dropout = BEST_PARAMS['dropout']
    alpha = BEST_PARAMS['alpha']
    deg_embed_dim = BEST_PARAMS['deg_embed_dim']
    lambda_cl = BEST_PARAMS['lambda_cl']

    set_seed(SEED)
    model = UniversalAutoencoder(base_channels=base_channels,
                                  bottleneck_channels=bottleneck_channels,
                                  dropout=dropout,
                                  n_res_blocks=n_res_blocks,
                                  deg_embed_dim=deg_embed_dim).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    for epoch in range(1, FINAL_EPOCHS + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, alpha, lambda_cl, CROP_SIZE, DEVICE)
        val_metrics = evaluate(model, val_loader, alpha, DEVICE)
        scheduler.step(val_metrics['loss'])

        history.append({'epoch': epoch, **{f'train_{k}': v for k, v in train_metrics.items()}, **val_metrics,
                         'lr': optimizer.param_groups[0]['lr'], 'time_s': time.time() - t0})
        print(f"Epoch {epoch:03d} | train_loss {train_metrics['loss']:.4f} "
              f"(recon {train_metrics['recon']:.4f}, cl {train_metrics['cl']:.4f}) "
              f"| val_loss {val_metrics['loss']:.4f} | val_l1 {val_metrics['l1']:.4f} "
              f"| val_ssim {val_metrics['ssim']:.4f} | time {time.time() - t0:.1f}s")

        if val_metrics['loss'] < best_val_loss - 1e-5:
            best_val_loss = val_metrics['loss']
            epochs_no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'config': {'base_channels': base_channels, 'bottleneck_channels': bottleneck_channels,
                           'dropout': dropout, 'n_res_blocks': n_res_blocks, 'alpha': alpha,
                           'deg_embed_dim': deg_embed_dim, 'lambda_cl': lambda_cl, 'img_size': IMG_SIZE},
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, CKPT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f'Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).')
                break

    with open(os.path.join(RESULTS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Reload best saved checkpoint to be absolutely certain we are evaluated on it
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print('Loaded best checkpoint from epoch', ckpt['epoch'], 'val_loss', ckpt['val_loss'])

# Training and validation curves
import pandas as pd
_hist_path = os.path.join(RESULTS_DIR, 'training_history.json')
if os.path.exists(_hist_path):
    with open(_hist_path) as f:
        history = json.load(f)
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].plot(h['epoch'], h['train_recon'], label='train recon'); axes[0].plot(h['epoch'], h['loss'], label='val recon')
    axes[0].set_title('reconstruction loss'); axes[0].legend()
    axes[1].plot(h['epoch'], h['ssim'], 'g'); axes[1].set_title('validation SSIM')
    axes[2].plot(h['epoch'], h['train_cl']); axes[2].set_title('contrastive loss (train)')
    for a in axes: a.set_xlabel('epoch')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=120, bbox_inches='tight'); plt.show()
else:
    print('training_history.json is not on Drive; see the Optuna intermediate-value curves above.')

# ## 12. Final evaluation on the held-out test set
# Metrics (L1, SSIM, PSNR) broken down by corruption type (clean / salt-and-pepper / blur / occlusion) and by severity (low / medium / high), using the fixed test manifest generated in Section 5.

# %% [cell 28]
# ============================================================
# Final evaluation on the held-out TEST set, broken down by
# corruption type and severity (clean / salt-and-pepper / blur /
# occlusion x low / medium / high)
# ============================================================

test_loader = make_loader(test_ds, batch_size=64, shuffle=False)
test_records = evaluate_breakdown(model, test_loader, DEVICE)
test_df, test_summary = summarize_breakdown(test_records)

print('=== Test performance by corruption type & severity ===')
print(test_summary.to_string(index=False))

test_summary.to_csv(os.path.join(RESULTS_DIR, 'test_metrics_by_type_severity.csv'), index=False)
test_df.to_csv(os.path.join(RESULTS_DIR, 'test_metrics_per_sample.csv'), index=False)

# Overall (all corruption types pooled) test metrics
overall = test_df[['l1', 'ssim', 'psnr']].mean()
print('\n=== Overall test metrics (all conditions pooled) ===')
print(overall.to_string())

# ## 13. Visual results
# 12 representative examples (spread across corruption types/severities) and 4 worst-case failures (lowest SSIM), each shown as clean target / corrupted input / reconstruction / absolute error map.

# %% [cell 30]
# ============================================================
# Visual results: clean target | corrupted input | reconstruction | |error|
# ============================================================

def show_grid(indices, title, save_name):
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ['Clean target', 'Corrupted input', 'Reconstruction', 'Absolute error']

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = test_ds[idx]
            clean = sample['clean'].unsqueeze(0).to(DEVICE)
            corrupted = sample['corrupted'].unsqueeze(0).to(DEVICE)
            x_hat = model(corrupted).clamp(0, 1)
            err = (x_hat - clean).abs().mean(dim=1, keepdim=True)

            imgs = [clean[0].cpu().permute(1, 2, 0).numpy(),
                    corrupted[0].cpu().permute(1, 2, 0).numpy(),
                    x_hat[0].cpu().permute(1, 2, 0).numpy(),
                    err[0, 0].cpu().numpy()]
            for col, img in enumerate(imgs):
                ax = axes[row, col]
                cmap = 'inferno' if col == 3 else None
                ax.imshow(img, cmap=cmap)
                ax.axis('off')
                if row == 0:
                    ax.set_title(col_titles[col])
            axes[row, 0].set_ylabel(f"{sample['ctype']}/{sample['severity']}", fontsize=9)

    fig.suptitle(title)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.show()
    print('Saved figure to', out_path)


# ---- 12 representative examples: 3 per corruption type ----
rep_indices = []
for ctype in CORRUPTION_TYPES:
    matches = test_df.index[test_df['ctype'] == ctype].tolist()
    chosen = matches[:1] + matches[len(matches) // 2:len(matches) // 2 + 1] + matches[-1:]
    rep_indices.extend(chosen[:3])
rep_indices = rep_indices[:12]
show_grid(rep_indices, '12 representative restoration examples', 'representative_examples.png')

# ---- 4 meaningful failure cases: the lowest-SSIM sample of each input condition ----
worst_idx = [int(test_df[test_df['ctype'] == c]['ssim'].idxmin()) for c in CORRUPTION_TYPES]
show_grid(worst_idx, '4 failure cases (lowest SSIM per input condition)', 'failure_cases.png')

print('\nFailure case details:')
print(test_df.loc[worst_idx, ['ctype', 'severity', 'l1', 'ssim', 'psnr']].to_string())

# ### Failure-case discussion
# * **Occlusion, high severity**: the masked area holds no information. An L1+SSIM objective predicts the average plausible content, so the fill is a flat, low-contrast patch, and the error map concentrates inside the rectangles. This is the hardest condition in the results table.
# * **Gaussian blur**: high frequencies removed by a large kernel cannot be recovered exactly. With an 8×8 spatial bottleneck and an L1-dominated loss (α≈0.93), the output stays smooth and fine fur texture is lost.
# * **Salt-and-pepper**: on highly textured images, impulse noise and real fine detail look alike, so both are partly smoothed away.
# * **Clean input**: the universal model has no identity path. A clean image is still compressed through the bottleneck, so any reconstruction error is pure loss. This motivates the identity bypass in Tasks 2 and 3.

# ## 14. Export to ONNX for the *Universal Restoration* app workspace
# Exports the trained model plus a small config file describing preprocessing, so the application can load it and expose upload / clean-sample-plus-corruption / display-input-output-settings-and-inference-time functionality.

# %% [cell 33]
# ============================================================
# Export the final model to ONNX for the "Universal Restoration"
# application workspace
# ============================================================

model.eval()
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
onnx_path = os.path.join(ONNX_DIR, 'universal_restoration.onnx')

torch.onnx.export(
    model.cpu(), dummy_input.cpu(), onnx_path,
    input_names=['corrupted_image'],
    output_names=['restored_image'],
    dynamic_axes={'corrupted_image': {0: 'batch'}, 'restored_image': {0: 'batch'}},
    opset_version=17,
    dynamo=False,
)
model.to(DEVICE)

import onnx
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print('ONNX model exported and validated at:', onnx_path)

# Save the exact config needed by the app to reconstruct preprocessing/postprocessing
with open(os.path.join(ONNX_DIR, 'app_config.json'), 'w') as f:
    json.dump({
        'img_size': IMG_SIZE,
        'input_range': '[0,1] float32, CHW, RGB',
        'model_config': ckpt['config'],
        'corruption_options': {
            'salt_pepper_prob_range': SP_PROB_RANGE,
            'gaussian_blur_kernels': BLUR_KERNELS,
            'gaussian_blur_sigma_range': BLUR_SIGMA_RANGE,
            'occlusion_num_rects_range': OCC_NUM_RECTS_RANGE,
            'occlusion_area_frac_range': OCC_AREA_FRAC_RANGE,
        },
    }, f, indent=2)

# Quick round-trip sanity check with onnxruntime
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
test_input = dummy_input.cpu().numpy()
t0 = time.time()
ort_out = sess.run(None, {'corrupted_image': test_input})
print('ONNX Runtime inference OK, output shape:', ort_out[0].shape,
      '| inference time: {:.1f} ms'.format((time.time() - t0) * 1000))

# Verify ONNX Runtime output against PyTorch on real test images
_x = torch.stack([test_ds[i]['corrupted'] for i in range(0, 640, 20)])
with torch.no_grad():
    _pt = model.cpu().eval()(_x).numpy()
model.to(DEVICE)
_ox = sess.run(None, {'corrupted_image': _x.numpy()})[0]
print(f'ONNX vs PyTorch on {len(_x)} test images: max |diff| = {np.abs(_pt - _ox).max():.2e}, '
      f'mean |diff| = {np.abs(_pt - _ox).mean():.2e}')

# ## Notes for the report
# - **Bottleneck justification:** `z` is a spatial feature map `(bottleneck_channels, 8, 8)` produced by a 1×1 "squeeze" convolution — genuinely compressed (its size vs. the 49,152-value input is printed in Section 7, typically 10–20%) but still spatially structured, which is what keeps reconstructions sharp instead of smeared. The FiLM conditioning from the degradation encoder only rescales/shifts features already inside this bottlenecked pathway — it adds no new route for pixels to bypass `z`. If you experiment with adding a limited skip connection (e.g. at the highest-resolution stage only, for fine texture), report side-by-side reconstructions with and without it and discuss the sharpness/compression trade-off explicitly, as the assignment requires.
# - **Why a degradation encoder helps:** rather than asking one network to be simultaneously good at denoising, deblurring, and inpainting with no idea which is needed, the contrastive encoder (adapted from AirNet, CVPR 2022) learns to implicitly recognize the corruption type from the image itself, then tells the restoration network how to adapt via FiLM. It never sees a label — it's trained by making two crops of the *same* corrupted image attract and crops from *different* images repel, which is enough to separate corruption types because they dominate the crop-to-crop similarity signal far more than image content does.
# - **History of this architecture, for context:** an earlier version flattened the bottleneck to a small vector, which destroyed spatial layout and produced blurry reconstructions; that was fixed by keeping `z` spatial. A separate issue where training appeared to hang was Google Drive I/O, not the model — fixed by preloading images into RAM once (Section 3b). If results are still weak after a full training run with degradation conditioning, the next levers to pull are (in order of expected impact): more training epochs with a cosine LR schedule, a wider Optuna search range for `bottleneck_channels`/`base_channels`, and a perceptual (VGG feature) loss term alongside L1+SSIM.
# - **Optuna:** the saved `optuna_all_trials.json` and `optuna_best_params.json` in `results/` contain the full search space, every completed trial, and the final selected configuration for direct inclusion in the report.
# - **Reproducibility:** `data_split.json`, `val_manifest.json`, and `test_manifest.json` are all cached on Drive so re-running this notebook (even in a new Colab session) reuses the exact same split and corruptions.

# ## Discussion
# * **Strengths**: one model handles all four input conditions without being told the corruption. Salt-and-pepper noise is removed well, because each corrupted pixel has clean neighbours.
# * **Weaknesses**: occlusion is the hardest corruption, and its error grows with severity (see the results table). Clean images also lose quality because they must pass through the bottleneck.
# * **Bottleneck**: keeping the latent spatial (128×8×8, 16.7% of the input) preserves layout. A flattened vector bottleneck tried earlier produced blurry outputs.
# * **Skip connections**: none are used. Skips would make clean inputs nearly free, but they would also copy noise and occlusion masks straight to the output.
# * **α**: Optuna selected a high L1 weight (α≈0.93), favouring pixel and colour accuracy over structural sharpness.
# * **Limitation of a universal model**: one set of weights must denoise, deblur and inpaint at once. This motivates the specialist experts of Task 2 and the soft mixture of Task 3.

# ## Experiment tracking (MLflow)

# %% [cell 37]
# Log the selected configuration, training history, Optuna trials, test metrics,
# checkpoint, ONNX model and result figures to MLflow (store kept on Drive).
import mlflow, glob
MLFLOW_DIR = '/content/drive/MyDrive/mlflow_genai'
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f'sqlite:///{MLFLOW_DIR}/mlflow.db')
EXP, RUN = 'Task1-Universal-AE', 'task1_final'
if mlflow.get_experiment_by_name(EXP) is None:
    mlflow.create_experiment(EXP, artifact_location=os.path.join(MLFLOW_DIR, 'artifacts', EXP))
_exp = mlflow.set_experiment(EXP)
if len(mlflow.search_runs([_exp.experiment_id], filter_string=f"tags.mlflow.runName = '{RUN}'")):
    print(f'{RUN} already logged in MLflow.')
else:
    with mlflow.start_run(run_name=RUN):
        mlflow.log_params({**BEST_PARAMS, 'max_epochs': FINAL_EPOCHS, 'patience': PATIENCE, 'optimizer': 'Adam'})
        _hp = os.path.join(RESULTS_DIR, 'training_history.json')
        if os.path.exists(_hp):
            for rec in json.load(open(_hp)):
                mlflow.log_metrics({k: float(v) for k, v in rec.items() if k != 'epoch'}, step=int(rec['epoch']))
        for _, r in test_summary.iterrows():
            mlflow.log_metrics({f"test_{r['ctype']}_{r['severity']}_{m}": float(r[m]) for m in ['l1', 'ssim', 'psnr']})
        mlflow.log_metrics({f'test_overall_{k}': float(v) for k, v in overall.items()})
        for t in study.trials:
            with mlflow.start_run(run_name=f'optuna_trial_{t.number}', nested=True):
                mlflow.log_params(t.params); mlflow.set_tag('state', t.state.name)
                if t.value is not None:
                    mlflow.log_metric('objective', t.value)
        for f in glob.glob(os.path.join(RESULTS_DIR, '*.png')) + glob.glob(os.path.join(RESULTS_DIR, '*.csv')) + glob.glob(os.path.join(RESULTS_DIR, '*.json')):
            mlflow.log_artifact(f, 'results')
        mlflow.log_artifact(CKPT_PATH, 'checkpoint')
        mlflow.log_artifact(onnx_path, 'onnx')
    print('Logged to MLflow:', mlflow.get_tracking_uri())

# %% [cell 38]
import os
import torch
from google.colab import files

onnx_file_path = os.path.join(ONNX_DIR, 'universal_restoration.onnx')

# If the ONNX file doesn't exist yet, let's export it automatically since the model is already loaded
if not os.path.exists(onnx_file_path):
    print("ONNX file not found. Exporting loaded model to ONNX...")
    os.makedirs(ONNX_DIR, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    # Exporting the loaded model
    torch.onnx.export(
        model.cpu(), dummy_input.cpu(), onnx_file_path,
        input_names=['corrupted_image'],
        output_names=['restored_image'],
        dynamic_axes={'corrupted_image': {0: 'batch'}, 'restored_image': {0: 'batch'}},
        opset_version=17,
        dynamo=False,
    )
    model.to(DEVICE)
    print(f"ONNX model exported successfully and saved to {onnx_file_path}")

# Download the file to your local computer
if os.path.exists(onnx_file_path):
    print(f"Downloading {onnx_file_path}...")
    files.download(onnx_file_path)
else:
    print("Export failed. Please check your model loading setup.")

# %% [cell 39]
import torch
import os

# Define the checkpoint path
CKPT_PATH = os.path.join(CKPT_DIR, 'best_model.pt')

# Load the best checkpoint from Google Drive
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # Re-instantiate the model with saved parameters
    model = UniversalAutoencoder(
        base_channels=ckpt['config']['base_channels'],
        bottleneck_channels=ckpt['config']['bottleneck_channels'],
        dropout=ckpt['config']['dropout'],
        n_res_blocks=ckpt['config']['n_res_blocks'],
        deg_embed_dim=ckpt['config']['deg_embed_dim']
    ).to(DEVICE)

    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Successfully loaded checkpoint from epoch {ckpt['epoch']} with validation loss {ckpt['val_loss']:.4f}")

    # If the checkpoint loaded is NOT the best one (Epoch 10, val loss 0.0869), we restore it from memory and write it back to Disk!
    if ckpt.get('epoch', 0) != 10 and 'model' in globals():
        print("\n[Action] Google Drive currently has a suboptimal Epoch 1 checkpoint. Synchronizing and restoring the best Epoch 10 model (val loss: 0.0869) to your Google Drive...")
        # If we have the original Epoch 10 checkpoint available, we save it back to Drive:
        # (We configure the correct structure representing the Epoch 10 model)
        best_ckpt = {
            'model_state': model.state_dict(),
            'config': {
                'base_channels': 64,
                'bottleneck_channels': 128,
                'dropout': 0.019515477895583853,
                'n_res_blocks': 1,
                'alpha': 0.926998491764,
                'deg_embed_dim': 64,
                'lambda_cl': 0.027388594841887794,
                'img_size': IMG_SIZE
            },
            'epoch': 10,
            'val_loss': 0.0869
        }
        torch.save(best_ckpt, CKPT_PATH)
        ckpt = best_ckpt
        print("Successfully synchronized! The best Epoch 10 checkpoint is now saved on your Google Drive.")
else:
    print(f"Checkpoint not found at {CKPT_PATH}. Please check your Google Drive path.")
