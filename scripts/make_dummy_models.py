"""
Generates dummy ONNX models that have the SAME I/O contract as the real
ones, so the FastAPI backend and React frontend can be developed and tested
without needing the trained weights.

When you swap in the real .onnx files (same filenames, same I/O names),
nothing in the app changes.

Run:
    pip install torch onnx
    python scripts/make_dummy_models.py
"""
import os
import torch
import torch.nn as nn

OUT = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(OUT, exist_ok=True)

IMG = 128


# ---------- Dummy Universal AE ----------
class DummyAE(nn.Module):
    def __init__(self):
        super().__init__()
        # A real (but untrained) tiny conv AE — outputs something plausible
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


# ---------- Dummy Classifier ----------
class DummyClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc(self.body(x))


# ---------- Dummy Soft MoE ----------
class DummySoftMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae_salt = DummyAE()
        self.ae_blur = DummyAE()
        self.ae_occ  = DummyAE()
        self.gate = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=1)                # (N, 4)
        y_id    = x
        y_salt  = self.ae_salt(x)
        y_blur  = self.ae_blur(x)
        y_occ   = self.ae_occ(x)
        recon = (w[:, 0:1, None, None] * y_id
                 + w[:, 1:2, None, None] * y_salt
                 + w[:, 2:3, None, None] * y_blur
                 + w[:, 3:4, None, None] * y_occ)
        return recon, w


# ---------- Dummy GAN Generator ----------
class DummyGen(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.embed = nn.Embedding(3, embed_dim)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
        )
        self.style_proj = nn.Linear(embed_dim, 32)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, photo, style_id):
        h = self.enc(photo)
        s = self.style_proj(self.embed(style_id))[:, :, None, None]
        h = h + s
        return self.dec(h)


def export(model, path, input_names, output_names, dummy_inputs):
    torch.onnx.export(
        model, dummy_inputs, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes={n: {0: 'batch'} for n in input_names + output_names},
        opset_version=17,
        do_constant_folding=True,
    )
    print('wrote', path)


dummy = torch.randn(1, 3, IMG, IMG)

export(DummyAE(), os.path.join(OUT, 'task1.onnx'),
    ['corrupted_image'], ['restored_image'], dummy)

export(DummyClf(), os.path.join(OUT, 'classifier.onnx'),
    ['corrupted_image'], ['logits'], dummy)

for ctype in ['salt_pepper', 'gaussian_blur', 'occlusion']:
    export(DummyAE(), os.path.join(OUT, f'specialist_{ctype}.onnx'),
        ['corrupted_image'], ['restored_image'], dummy)

export(DummySoftMoE(), os.path.join(OUT, 'task3.onnx'),
    ['corrupted_image'], ['restored_image', 'routing_weights'], dummy)

style_ids = torch.tensor([0], dtype=torch.long)
export(DummyGen(), os.path.join(OUT, 'task4.onnx'),
    ['photo', 'style'], ['sketch'], (dummy, style_ids))

print('\nAll dummy models written to', OUT)