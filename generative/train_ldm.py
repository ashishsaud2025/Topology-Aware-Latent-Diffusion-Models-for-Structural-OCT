"""Stage 2 of the pipeline: Fine-tune MONAI Generative Model.

Trains (or fine-tunes) a class-conditional Latent Diffusion Model on the real
OCT training split using the MONAI Generative Models framework:
  1. Autoencoder (KL-regularized) compresses images into a latent space.
  2. A conditional diffusion UNet is trained/fine-tuned in that latent space,
     conditioned on class label, to later synthesize per-class OCT images.

Reference conventions follow the MONAI Generative Models tutorials
(`monai.apps.generative` / `generative.networks.nets`), adapted here for
class-conditional 2D OCT B-scans.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from utils.logging_utils import get_logger
from utils.seed import set_global_seed

logger = get_logger(__name__)


def build_autoencoder(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate the KL-autoencoder used to compress OCT images to latents.

    TODO: instantiate `generative.networks.nets.AutoencoderKL` (or equivalent
    MONAI Generative class) with channels/latent_channels matching
    cfg['generative']['latent_channels'] and cfg['generative']['image_size'].
    Load `cfg['generative']['autoencoder_checkpoint']` if provided.
    """
    raise NotImplementedError("TODO: build MONAI AutoencoderKL")


def build_diffusion_unet(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate the conditional diffusion UNet operating in latent space.

    TODO: instantiate `generative.networks.nets.DiffusionModelUNet` with
    class-conditioning (e.g. via cross-attention on a class-embedding) so
    that `generate_synthetic.py` can later sample per-class images.
    """
    raise NotImplementedError("TODO: build MONAI DiffusionModelUNet with class conditioning")


def build_scheduler(cfg: Dict[str, Any]):
    """Instantiate the noise scheduler (DDPM or DDIM) per
    cfg['generative']['scheduler'].

    TODO: use `generative.networks.schedulers.DDPMScheduler` /
    `DDIMScheduler` with `num_train_timesteps` matching training config.
    """
    raise NotImplementedError("TODO: build noise scheduler")


def train_autoencoder_stage(
    autoencoder: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
) -> torch.nn.Module:
    """Stage-1 training: reconstruction + KL + adversarial/perceptual losses.

    TODO: implement the standard MONAI Generative autoencoder training loop
    (L1/L2 reconstruction loss, KL loss, optional PatchGAN discriminator +
    perceptual loss). Save checkpoint to
    cfg['generative']['autoencoder_checkpoint'].
    """
    raise NotImplementedError("TODO: implement autoencoder training loop")


def train_diffusion_stage(
    autoencoder: torch.nn.Module,
    diffusion_unet: torch.nn.Module,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
) -> torch.nn.Module:
    """Stage-2 training: fine-tune the conditional diffusion UNet in the
    frozen autoencoder's latent space.

    TODO: implement the standard MONAI Generative diffusion training loop:
      for each batch -> encode to latent -> sample timestep t -> add noise
      -> predict noise conditioned on class label -> MSE loss.
    Save checkpoint to cfg['generative']['diffusion_checkpoint'].
    """
    raise NotImplementedError("TODO: implement diffusion model training loop")


def main(cfg: Dict[str, Any]) -> None:
    set_global_seed(cfg["project"]["seed"])
    device = cfg["project"]["device"]

    logger.info("Building real-data training/validation loaders for LDM fine-tuning...")
    # TODO: build DataLoaders over the REAL train/val split only
    # (synthetic images obviously cannot be used to train the generator).
    train_loader: DataLoader = ...  # TODO
    val_loader: DataLoader = ...    # TODO

    autoencoder = build_autoencoder(cfg).to(device)
    autoencoder = train_autoencoder_stage(autoencoder, train_loader, val_loader, cfg)

    diffusion_unet = build_diffusion_unet(cfg).to(device)
    scheduler = build_scheduler(cfg)
    diffusion_unet = train_diffusion_stage(
        autoencoder, diffusion_unet, scheduler, train_loader, val_loader, cfg
    )

    logger.info("MONAI LDM fine-tuning complete.")


if __name__ == "__main__":
    import argparse

    from utils.seed import load_config

    parser = argparse.ArgumentParser(description="Fine-tune MONAI LDM on real OCT data")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    main(load_config(args.config))
