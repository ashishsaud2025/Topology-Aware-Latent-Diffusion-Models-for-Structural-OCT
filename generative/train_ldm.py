"""Stage 2 of the pipeline: Fine-tune MONAI Generative Model.

Trains (or fine-tunes) a class-conditional Latent Diffusion Model on the real
OCT training split using the MONAI Generative Models framework:
  1. Autoencoder (KL-regularized) compresses images into a latent space.
  2. A conditional diffusion UNet is trained/fine-tuned in that latent space,
     conditioned on class label, to later synthesize per-class OCT images.

This implementation uses:
  - `monai.networks.nets.AutoencoderKL` for the autoencoder
  - `monai.networks.nets.DiffusionModelUNet` for the denoising UNet
  - `monai.networks.schedulers.DDPMScheduler` / `DDIMScheduler` for noise scheduling

Reference: MONAI Generative Models tutorials
(https://github.com/Project-MONAI/GenerativeModels)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logging_utils import get_logger
from utils.seed import set_global_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_autoencoder(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the KL-autoencoder used to compress OCT images to latents.

    Architecture suitable for 224x224 grayscale OCT images:
      - 2D convolutions
      - 1 input channel (grayscale), 1 output channel
      - 3 latent channels (compressed representation)
      - Channel progression: 64 -> 128 -> 256 -> 512
      - Attention at 16x16 and 8x8 levels

    Args:
        cfg: Full pipeline config dict.

    Returns:
        AutoencoderKL instance (untrained unless checkpoint loaded below).
    """
    ae = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512),
        latent_channels=cfg["generative"]["latent_channels"],
        num_res_blocks=(2, 2, 2, 2),
        attention_levels=(False, False, True, True),
        norm_num_groups=32,
        with_encoder_nonlocal_attn=True,
        with_decoder_nonlocal_attn=True,
        use_checkpoint=False,
    )

    # Load checkpoint if provided
    checkpoint_path = cfg["generative"].get("autoencoder_checkpoint")
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading autoencoder checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        ae.load_state_dict(state_dict)
    elif checkpoint_path:
        logger.warning(
            f"Autoencoder checkpoint not found: {checkpoint_path}. "
            "Starting from scratch."
        )

    total_params = sum(p.numel() for p in ae.parameters())
    logger.info(
        f"Built AutoencoderKL: latent_channels={cfg['generative']['latent_channels']}, "
        f"{total_params:,} parameters"
    )
    return ae


def build_diffusion_unet(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the conditional diffusion UNet operating in latent space.

    Uses class-embedding conditioning via `num_class_embeds` so that
    `generate_synthetic.py` can later sample per-class images by passing
    the desired class label during inference.

    Args:
        cfg: Full pipeline config dict.

    Returns:
        DiffusionModelUNet instance (untrained unless checkpoint loaded).
    """
    latent_channels = cfg["generative"]["latent_channels"]
    num_classes = len(cfg["data"]["classes"])

    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=latent_channels,
        out_channels=latent_channels,
        channels=(64, 128, 256, 256),
        attention_levels=(False, False, True, True),
        num_res_blocks=(2, 2, 2, 2),
        num_head_channels=8,
        # Class-conditioning via num_class_embeds (learned class embeddings
        # added to timestep embedding internally).  cross_attention_dim and
        # with_conditioning are NOT needed because we do NOT use cross-attention.
        with_conditioning=False,
        cross_attention_dim=None,
        num_class_embeds=num_classes,
        upcast_attention=False,
        use_flash_attention=False,
    )

    # Load checkpoint if provided
    checkpoint_path = cfg["generative"].get("diffusion_checkpoint")
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading diffusion UNet checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        unet.load_state_dict(state_dict)
    elif checkpoint_path:
        logger.warning(
            f"Diffusion UNet checkpoint not found: {checkpoint_path}. "
            "Starting from scratch."
        )

    total_params = sum(p.numel() for p in unet.parameters())
    logger.info(
        f"Built DiffusionModelUNet: {total_params:,} parameters, "
        f"{num_classes} class embeddings"
    )
    return unet


def build_scheduler(cfg: Dict[str, Any]):
    """Instantiate the noise scheduler per config.

    Supports DDPM (for training) and DDIM (for faster inference sampling).
    Defaults to DDPM if not specified or recognized.

    Args:
        cfg: Full pipeline config dict.

    Returns:
        Scheduler instance (DDPMScheduler or DDIMScheduler).
    """
    scheduler_type = cfg["generative"].get("scheduler", "ddpm").lower()
    num_train_timesteps = cfg["generative"].get("num_inference_steps", 1000)

    if scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule="linear_beta",
            clip_sample=True,
            prediction_type="epsilon",
        )
        logger.info(f"Built DDIMScheduler ({num_train_timesteps} steps)")
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule="linear_beta",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )
        logger.info(f"Built DDPMScheduler ({num_train_timesteps} steps)")

    return scheduler


# ---------------------------------------------------------------------------
# Autoencoder training
# ---------------------------------------------------------------------------

def train_autoencoder_stage(
    autoencoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
) -> nn.Module:
    """Stage-1 training: reconstruction + KL loss.

    Training loop for the AutoencoderKL:
      - Forward: encode image -> latent sample -> decode
      - Loss: reconstruction (L1) + KL divergence
      - Optimizer: AdamW with cosine LR schedule
      - Checkpoints saved to autoencoder_checkpoint path

    Args:
        autoencoder: AutoencoderKL model.
        train_loader: DataLoader over real OCT training images.
        val_loader: DataLoader over real OCT validation images.
        cfg: Full pipeline config dict.

    Returns:
        Trained autoencoder model (best validation loss).
    """
    device = cfg["project"]["device"]
    num_epochs = cfg["generative"].get("num_train_epochs", 200)
    lr = cfg["generative"].get("learning_rate", 2.5e-5)
    batch_size = cfg["generative"].get("batch_size", 16)

    checkpoint_path = cfg["generative"].get("autoencoder_checkpoint")
    if checkpoint_path is None:
        checkpoint_path = str(Path(cfg["project"]["output_dir"]) / "models" / "autoencoder.pt")
        cfg["generative"]["autoencoder_checkpoint"] = checkpoint_path

    output_dir = Path(checkpoint_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    autoencoder = autoencoder.to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(enabled=(device == "cuda"))
    recon_criterion = nn.L1Loss()

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = cfg["training"].get("early_stopping_patience", 8)

    logger.info(f"Starting autoencoder training for up to {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # --- Training ---
        autoencoder.train()
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"AE Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

            optimizer.zero_grad()

            with autocast(enabled=(device == "cuda")):
                # Forward through autoencoder
                # NOTE: AutoencoderKL returns (reconstruction, z_mu, z_sigma)
                # where z_sigma is the standard deviation (not log variance)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                z_logvar = torch.log(z_sigma.pow(2) + 1e-8)  # compute log variance safely

                # Reconstruction loss
                recon_loss = recon_criterion(reconstruction, images)

                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
                kl_loss = kl_loss / images.shape[0]  # normalize by batch size

                # Combined loss (beta-VAE weighting: KL weight = 1e-6)
                total_loss = recon_loss + 1e-6 * kl_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_total_loss += total_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "recon": f"{recon_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}",
            })

        avg_train_loss = train_total_loss / max(num_batches, 1)

        # --- Validation ---
        autoencoder.eval()
        val_total_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

                with autocast(enabled=(device == "cuda")):
                    reconstruction, z_mu, z_sigma = autoencoder(images)
                    z_logvar = torch.log(z_sigma.pow(2) + 1e-8)
                    recon_loss = recon_criterion(reconstruction, images)
                    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
                    kl_loss = kl_loss / images.shape[0]
                    total_loss = recon_loss + 1e-6 * kl_loss

                val_total_loss += total_loss.item()
                val_batches += 1

        avg_val_loss = val_total_loss / max(val_batches, 1)

        # LR scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} "
            f"(recon={train_recon_loss/num_batches:.6f}, KL={train_kl_loss/num_batches:.6f}) | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Early stopping & checkpointing based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(autoencoder.state_dict(), checkpoint_path)
            logger.info(f"  -> Saved best autoencoder checkpoint ({avg_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {early_stop_patience} epochs)"
                )
                break

    # Load best checkpoint
    best_state = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(best_state)
    logger.info(f"Autoencoder training complete. Best val loss: {best_val_loss:.6f}")

    return autoencoder


# ---------------------------------------------------------------------------
# Diffusion model training
# ---------------------------------------------------------------------------

def train_diffusion_stage(
    autoencoder: nn.Module,
    diffusion_unet: nn.Module,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
) -> nn.Module:
    """Stage-2 training: fine-tune the conditional diffusion UNet in the
    frozen autoencoder's latent space.

    Training procedure:
      1. Freeze autoencoder (eval mode, no gradients)
      2. For each batch: encode images to latents
      3. Sample random timestep t for each latent
      4. Add noise according to scheduler (forward diffusion)
      5. Predict noise with UNet conditioned on class label
      6. MSE loss between predicted and actual noise

    Args:
        autoencoder: Frozen AutoencoderKL (eval mode).
        diffusion_unet: DiffusionModelUNet to train.
        scheduler: Noise scheduler (DDPMScheduler).
        train_loader: DataLoader over real OCT training images.
        val_loader: DataLoader over real OCT validation images.
        cfg: Full pipeline config dict.

    Returns:
        Trained diffusion UNet (best validation loss).
    """
    device = cfg["project"]["device"]
    num_epochs = cfg["generative"].get("num_train_epochs", 200)
    lr = cfg["generative"].get("learning_rate", 2.5e-5)

    checkpoint_path = cfg["generative"].get("diffusion_checkpoint")
    if checkpoint_path is None:
        checkpoint_path = str(Path(cfg["project"]["output_dir"]) / "models" / "diffusion_unet.pt")
        cfg["generative"]["diffusion_checkpoint"] = checkpoint_path

    output_dir = Path(checkpoint_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze autoencoder
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    diffusion_unet = diffusion_unet.to(device)
    optimizer = torch.optim.AdamW(diffusion_unet.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(enabled=(device == "cuda"))
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = cfg["training"].get("early_stopping_patience", 8)

    logger.info(f"Starting diffusion UNet training for up to {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # --- Training ---
        diffusion_unet.train()
        train_total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Diff Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                # Encode to latent space
                # NOTE: AutoencoderKL.encode returns (z_mu, z_sigma)
                z_mu, z_sigma = autoencoder.encode(images)
                # Reparameterization trick: z = mu + sigma * epsilon
                z = z_mu + z_sigma * torch.randn_like(z_sigma)

            # Sample random timesteps
            batch_size = z.shape[0]
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (batch_size,), device=device
            ).long()

            # Add noise
            noise = torch.randn_like(z)
            noisy_z = scheduler.add_noise(original_samples=z, noise=noise, timesteps=timesteps)

            with autocast(enabled=(device == "cuda")):
                # Predict noise
                noise_pred = diffusion_unet(
                    x=noisy_z, timesteps=timesteps, class_labels=labels
                )
                loss = mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_train_loss = train_total_loss / max(num_batches, 1)

        # --- Validation ---
        diffusion_unet.eval()
        val_total_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch[0].to(device), batch[1].to(device)

                with autocast(enabled=(device == "cuda")):
                    z_mu, z_sigma = autoencoder.encode(images)
                    z = z_mu  # use mean for validation (no sampling)
                    batch_size = z.shape[0]
                    timesteps = torch.randint(
                        0, scheduler.num_train_timesteps, (batch_size,), device=device
                    ).long()
                    noise = torch.randn_like(z)
                    noisy_z = scheduler.add_noise(
                        original_samples=z, noise=noise, timesteps=timesteps
                    )
                    noise_pred = diffusion_unet(
                        x=noisy_z, timesteps=timesteps, class_labels=labels
                    )
                    loss = mse_loss(noise_pred, noise)

                val_total_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_total_loss / max(val_batches, 1)

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train MSE: {avg_train_loss:.6f} | "
            f"Val MSE: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Early stopping & checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(diffusion_unet.state_dict(), checkpoint_path)
            logger.info(f"  -> Saved best diffusion checkpoint ({avg_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {early_stop_patience} epochs)"
                )
                break

    # Load best checkpoint
    best_state = torch.load(checkpoint_path, map_location=device)
    diffusion_unet.load_state_dict(best_state)
    logger.info(f"Diffusion UNet training complete. Best val loss: {best_val_loss:.6f}")

    return diffusion_unet


# ---------------------------------------------------------------------------
# Data loading helper for LDM training
# ---------------------------------------------------------------------------

def _build_ldm_data_loaders(cfg: Dict[str, Any]):
    """Build DataLoaders over the REAL train/val split for LDM fine-tuning.

    Uses OOCImageDataset + build_data_loaders from data.dataset,
    loading from the preprocessed real data index CSV files generated
    during the preprocessing stage.

    Returns:
        Tuple of (train_loader, val_loader) for LDM training.
    """
    import pandas as pd

    from data.dataset import OCTImageDataset, build_data_loaders, get_default_transforms

    processed_dir = Path(cfg["data"]["processed_dir"])
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["generative"].get("batch_size", 16)
    num_workers = cfg["training"].get("num_workers", 8)
    classes = cfg["data"]["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Load preprocessed split indices (saved by preprocessing stage)
    split_dir = processed_dir / "splits"
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"  # not used here, but exists

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Preprocessed split CSVs not found in {split_dir}. "
            "Run the preprocessing stage first: python main.py preprocess --config configs/config.yaml"
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_transform = get_default_transforms(image_size, train=True)
    val_transform = get_default_transforms(image_size, train=False)

    train_dataset = OCTImageDataset(
        index_df=train_df,
        class_to_idx=class_to_idx,
        transform=train_transform,
        image_size=image_size,
    )
    val_dataset = OCTImageDataset(
        index_df=val_df,
        class_to_idx=class_to_idx,
        transform=val_transform,
        image_size=image_size,
    )
    # Dummy test dataset (just for build_data_loaders API compatibility)
    test_dataset = OCTImageDataset(
        index_df=val_df.head(1),  # placeholder
        class_to_idx=class_to_idx,
        transform=val_transform,
        image_size=image_size,
    )

    loaders = build_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers
    )
    return loaders["train"], loaders["val"]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(cfg: Dict[str, Any]) -> None:
    """Run the full LDM fine-tuning pipeline.

    Args:
        cfg: Full pipeline config dict (loaded from config.yaml).
    """
    set_global_seed(cfg["project"]["seed"])
    device = cfg.get("project", {}).get("device", "cuda")

    # Auto-detect device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    cfg["project"]["device"] = device

    logger.info("Building real-data training/validation loaders for LDM fine-tuning...")
    train_loader, val_loader = _build_ldm_data_loaders(cfg)

    logger.info("Building autoencoder...")
    autoencoder = build_autoencoder(cfg).to(device)

    logger.info("Training autoencoder stage...")
    autoencoder = train_autoencoder_stage(autoencoder, train_loader, val_loader, cfg)

    logger.info("Building diffusion UNet and scheduler...")
    diffusion_unet = build_diffusion_unet(cfg).to(device)
    scheduler = build_scheduler(cfg)

    logger.info("Training diffusion UNet stage...")
    diffusion_unet = train_diffusion_stage(
        autoencoder, diffusion_unet, scheduler, train_loader, val_loader, cfg
    )

    logger.info("MONAI LDM fine-tuning complete.")
    logger.info(f"  Autoencoder checkpoint: {cfg['generative']['autoencoder_checkpoint']}")
    logger.info(f"  Diffusion checkpoint: {cfg['generative']['diffusion_checkpoint']}")


if __name__ == "__main__":
    import argparse

    from utils.seed import load_config

    parser = argparse.ArgumentParser(description="Fine-tune MONAI LDM on real OCT data")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    main(load_config(args.config))