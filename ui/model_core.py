"""
model_core.py -- model loading + inference logic for the OCT Model Explorer.

Kept separate from the Gradio UI (app.py) so it can be unit-tested and
reused on its own. Every "load_*" function does a STRICT state_dict load
(no silent partial matches) and raises with a clear message on failure,
following the same verification discipline used throughout this project:
a checkpoint that "loads" with strict=False but has mismatched shapes is
worse than an outright error, since it produces plausible-looking garbage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler
from PIL import Image

# Make the project root importable (ui/model_core.py lives one level down).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.classifiers import build_classifier as _project_build_classifier

CLASS_NAMES = ["NORMAL", "CNV", "DME", "DRUSEN"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# helpers

def _strict_load(model: torch.nn.Module, path: str, label: str) -> str:
    """Loads a checkpoint with strict=True. Returns a human-readable status
    string on success; raises RuntimeError with a clear message on any
    shape/key mismatch, rather than letting strict=False hide the problem."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # Some checkpoints save {"state_dict": ...} or {"model": ...} wrappers.
    if isinstance(sd, dict) and "state_dict" in sd and not any(
        k.count(".") for k in sd.keys()
    ):
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and set(sd.keys()) <= {"model", "optimizer", "epoch"}:
        sd = sd["model"]
    try:
        result = model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"[{label}] STRICT LOAD FAILED -- the checkpoint's architecture "
            f"does not match the model built here. This usually means the "
            f"constructor args (channels/num_res_blocks/etc.) are wrong for "
            f"this specific checkpoint.\n\nOriginal error:\n{e}"
        ) from e
    n_params = sum(p.numel() for p in model.parameters())
    model.eval()
    model.to(DEVICE)
    return f"[{label}] loaded OK -- {n_params:,} params, strict match confirmed."


def pil_to_model_input(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Grayscale-or-RGB PIL image -> (1,3,size,size) float32 tensor,
    z-score normalized per-image (matches the project's own preprocessing:
    resize to 224, z-score normalize, replicate grayscale to 3 channels)."""
    img = img.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    x3 = x.repeat(1, 3, 1, 1)
    return x3.to(DEVICE)


def tensor_to_display(x: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (1,1,H,W) normalized tensor -> uint8 grayscale array
    for display/saving. Channels are averaged (matches how we've been
    visualizing 3-channel-replicated grayscale outputs throughout this
    project) and the result is min-max stretched to 0-255 for a clean,
    report-ready image (z-score data has no fixed display range)."""
    arr = x.detach().cpu()[0].mean(dim=0).numpy()
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-8)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def latent_to_display(z: torch.Tensor) -> np.ndarray:
    """(1,3,h,w) latent -> RGB uint8 preview, each channel independently
    min-max stretched (same convention used earlier in this project)."""
    z_np = z.detach().cpu()[0].numpy()
    channels = []
    for c in range(z_np.shape[0]):
        ch = z_np[c]
        lo, hi = ch.min(), ch.max()
        channels.append((ch - lo) / (hi - lo + 1e-8))
    rgb = np.stack(channels[:3], axis=-1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


# autoencoder

def build_autoencoder() -> AutoencoderKL:
    return AutoencoderKL(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(64, 128, 256, 512),
        latent_channels=3,
        num_res_blocks=(2, 2, 2, 2),
        attention_levels=(False, False, True, True),
        norm_num_groups=32,
        with_encoder_nonlocal_attn=True,
        with_decoder_nonlocal_attn=True,
        use_checkpoint=False,
    )


def load_autoencoder(path: str) -> tuple[AutoencoderKL, str]:
    model = build_autoencoder()
    status = _strict_load(model, path, "Autoencoder")
    return model, status


@torch.no_grad()
def ae_encode_decode(model: AutoencoderKL, img: Image.Image):
    """Returns (input_display, latent_display, recon_display, mse)."""
    x = pil_to_model_input(img)
    z_mu, z_sigma = model.encode(x)
    z = model.sampling(z_mu, z_sigma)
    recon = model.decode(z)
    mse = F.mse_loss(recon, x).item()
    return (
        tensor_to_display(x),
        latent_to_display(z_mu),
        tensor_to_display(recon),
        mse,
    )


# diffusion

def build_diffusion_unet() -> DiffusionModelUNet:
    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(64, 128, 256),
        attention_levels=(False, True, True),
        num_res_blocks=(2, 2, 2),
        num_class_embeds=4,
        norm_num_groups=32,
    )


def load_diffusion_unet(path: str) -> tuple[DiffusionModelUNet, str]:
    model = build_diffusion_unet()
    status = _strict_load(model, path, "Diffusion UNet")
    return model, status


@torch.no_grad()
def diffusion_generate(
    unet: DiffusionModelUNet,
    ae: AutoencoderKL,
    class_idx: int,
    num_train_timesteps: int,
    num_inference_steps: int,
    seed: int,
):
    """Runs a full DDIM sampling loop and decodes through the autoencoder.
    Returns a display-ready uint8 image. Requires both models loaded."""
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    torch.manual_seed(seed)
    z_t = torch.randn(1, 3, 28, 28, device=DEVICE)
    label = torch.tensor([class_idx], device=DEVICE)
    for t in scheduler.timesteps:
        t_batch = t.unsqueeze(0) if t.dim() == 0 else t
        t_batch = t_batch.to(DEVICE)
        noise_pred = unet(z_t, timesteps=t_batch, class_labels=label)
        z_t, _ = scheduler.step(noise_pred, t, z_t)

    decoded = ae.decode(z_t)
    return tensor_to_display(decoded), latent_to_display(z_t)


# classifier

def build_classifier(arch: str, num_classes: int = 4) -> torch.nn.Module:
    """Build a classifier that EXACTLY matches the project's training code.

    The full-factorial checkpoints were saved via
    ``models/classifiers.py::build_classifier``, which returns a
    ``ClassifierWrapper`` holding the backbone under ``.backbone`` (so the
    state_dict keys are ``backbone.*``) and uses a **timm** ViT-Base/16
    (``backbone.blocks.*`` keys) -- NOT a torchvision ViT.  Matching the
    real builder here is what makes ``_strict_load`` succeed for all three
    architectures.  Note: project training used ``pretrained=False`` so we
    pass that here too (weights are overwritten by the checkpoint anyway)."""
    return _project_build_classifier(arch, num_classes=num_classes, pretrained=False)


def load_classifier(arch: str, path: str) -> tuple[torch.nn.Module, str]:
    model = build_classifier(arch)
    status = _strict_load(model, path, f"Classifier ({arch})")
    return model, status


@torch.no_grad()
def classify_image(model: torch.nn.Module, img: Image.Image):
    """Returns dict[class_name -> probability] for Gradio's Label component."""
    x = pil_to_model_input(img)
    logits = model(x)  # ClassifierWrapper.forward -> backbone(x)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
