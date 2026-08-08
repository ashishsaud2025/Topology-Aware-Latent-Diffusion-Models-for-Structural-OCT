"""Stage 3 of the pipeline: Generate Synthetic Images per Class.

Loads the fine-tuned MONAI LDM (autoencoder + diffusion UNet) and samples a
class-conditional pool of synthetic OCT images per class. This pool is
generated ONCE up-front, large enough to cover the maximum requirement of
any experimental cell in the 3x5x3 factorial grid (i.e. the union of Factor A
ratios x Factor B distribution strategies), and is then subsampled per-cell
by experiment/dataset_builder.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from utils.logging_utils import get_logger

logger = get_logger(__name__)


# Generator loading
def load_trained_generator(cfg: Dict[str, Any]):
    """Load the fine-tuned autoencoder + diffusion UNet + scheduler from
    checkpoints produced by generative/train_ldm.py.

    Re-uses the model builder functions from `train_ldm` so the architecture
    is identical to what was trained.

    Args:
        cfg: Full pipeline config dict.

    Returns:
        Tuple of (autoencoder, diffusion_unet, scheduler), all in eval mode
        and moved to the configured device.

    Raises:
        FileNotFoundError: If checkpoint files are missing.
    """
    from generative.train_ldm import build_autoencoder, build_diffusion_unet, build_scheduler

    device = cfg["project"]["device"]
    autoencoder_ckpt = cfg["generative"].get("autoencoder_checkpoint")
    diffusion_ckpt = cfg["generative"].get("diffusion_checkpoint")

    if not autoencoder_ckpt or not Path(autoencoder_ckpt).exists():
        raise FileNotFoundError(
            f"Autoencoder checkpoint not found: {autoencoder_ckpt}. "
            "Run the generative training stage first."
        )
    if not diffusion_ckpt or not Path(diffusion_ckpt).exists():
        raise FileNotFoundError(
            f"Diffusion UNet checkpoint not found: {diffusion_ckpt}. "
            "Run the generative training stage first."
        )

    # Build models (checkpoints will be loaded inside build functions)
    logger.info(f"Loading autoencoder from: {autoencoder_ckpt}")
    autoencoder = build_autoencoder(cfg).to(device)
    autoencoder.eval()

    logger.info(f"Loading diffusion UNet from: {diffusion_ckpt}")
    diffusion_unet = build_diffusion_unet(cfg).to(device)
    diffusion_unet.eval()

    scheduler = build_scheduler(cfg)

    logger.info("Generator loaded successfully.")
    return autoencoder, diffusion_unet, scheduler


# Class-conditional sampling
@torch.no_grad()
def sample_class_conditional_batch(
    autoencoder: torch.nn.Module,
    diffusion_unet: torch.nn.Module,
    scheduler: Any,
    class_label: int,
    batch_size: int,
    cfg: Dict[str, Any],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run the reverse diffusion process conditioned on `class_label` and
    decode the resulting latents back to image space via the autoencoder.

    Implements the standard DDPM/DDIM sampling loop:
      1. Start from Gaussian noise in latent space.
      2. Iteratively denoise using `diffusion_unet` conditioned on class_label.
      3. Decode the final latent to image space via `autoencoder.decode`.

    Args:
        autoencoder: Frozen AutoencoderKL in eval mode.
        diffusion_unet: Frozen DiffusionModelUNet in eval mode.
        scheduler: DDPMScheduler or DDIMScheduler instance.
        class_label: Integer class label for conditioning (0..num_classes-1).
        batch_size: Number of images to generate in this batch.
        cfg: Full pipeline config dict.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Tensor of shape (batch_size, 1, H, W) containing generated images
        in the same value range as the real data preprocessing
        (z-score normalized float32).
    """
    device = cfg["project"]["device"]
    latent_channels = cfg["generative"]["latent_channels"]
    image_size = cfg["data"]["image_size"]
    num_inference_steps = cfg["generative"].get("num_inference_steps", 1000)

    # Calculate latent spatial dimensions (typically 1/8 of image size for
    # a 4-level autoencoder with stride 2 each level)
    latent_spatial = image_size // 8  # 224 -> 28 for typical AutoencoderKL

    # Set scheduler timesteps for inference
    scheduler.set_timesteps(num_inference_steps)

    # Start from random noise in latent space
    z_shape = (batch_size, latent_channels, latent_spatial, latent_spatial)
    z = torch.randn(z_shape, device=device, generator=generator)
    class_labels = torch.full((batch_size,), fill_value=class_label, device=device, dtype=torch.long)

    # Create class label embeddings for timestep conditioning
    # DiffusionModelUNet adds class_emb to timestep embedding internally

    # Sampling loop
    timesteps = scheduler.timesteps
    pbar = tqdm(timesteps, desc=f"Sampling class {class_label}", leave=False)
    for t in pbar:
        timestep = t
        if hasattr(timestep, 'item'):
            timestep = timestep.item()
        timestep_tensor = torch.full((batch_size,), fill_value=timestep, device=device, dtype=torch.long)

        # Predict noise
        noise_pred = diffusion_unet(
            x=z,
            timesteps=timestep_tensor,
            class_labels=class_labels,
        )

        # Denoise step
        # scheduler.step returns (prev_sample, pred_original_sample)
        step_output = scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=z,
            generator=generator,
        )
        z = step_output[0]

    # Decode latent to image space
    images = autoencoder.decode(z)

    # Clamp to valid range (z-score normalized: typically [-3, 3])
    images = torch.clamp(images, min=-3.0, max=3.0)

    return images


# Synthetic pool generation

def compute_max_synthetic_requirement(
    real_class_counts: Dict[str, int],
    factor_a_ratios: List[float],
    factor_b_strategies: List[str],
) -> Dict[str, int]:
    """Compute, per class, the maximum number of synthetic images that ANY
    cell in the factorial grid could require, so we only need to run
    generation once for the whole study.

    For each (ratio, strategy) combination, computes how many synthetic images
    per class would be needed, then takes the maximum across all combinations.

    Strategies:
      - proportional: synthetic total = ratio * total_real, split by class proportion
      - minority_only: synthetic budget spent on below-average classes only
      - fully_balanced: synthetic images added to bring all classes up to majority count,
        then additional ratio budget split proportionally

    Args:
        real_class_counts: Dict mapping class name -> count in the real training set.
        factor_a_ratios: List of synthetic ratios to test (e.g. [0.0, 0.25, 0.5, 0.75, 1.0]).
        factor_b_strategies: List of distribution strategies to test.

    Returns:
        Dict mapping class name -> maximum synthetic images needed across all cells.
    """
    if not real_class_counts:
        raise ValueError("real_class_counts is empty. Cannot compute synthetic requirements.")

    total_real = sum(real_class_counts.values())
    classes = list(real_class_counts.keys())
    majority_count = max(real_class_counts.values())
    mean_count = total_real / len(classes)

    # Track maximum per class across all (ratio, strategy) combinations
    max_per_class: Dict[str, int] = {c: 0 for c in classes}

    for ratio in factor_a_ratios:
        if ratio == 0.0:
            continue  # no synthetic data for baseline cells

        synthetic_budget = int(round(ratio * total_real))

        for strategy in factor_b_strategies:
            per_class_counts: Dict[str, int] = {c: 0 for c in classes}

            if strategy == "proportional":
                # Distribute budget proportionally to real class shares
                allocated = 0
                for i, cls in enumerate(classes):
                    if i < len(classes) - 1:
                        # Proportional allocation
                        n = int(round(synthetic_budget * real_class_counts[cls] / total_real))
                    else:
                        # Last class gets remainder
                        n = synthetic_budget - allocated
                    per_class_counts[cls] = max(0, n)
                    allocated += n

            elif strategy == "minority_only":
                # Identify minority classes (below mean count)
                minority_classes = [
                    c for c in classes if real_class_counts[c] < mean_count
                ]
                if not minority_classes:
                    # If no minority classes (perfectly balanced), fall back to proportional
                    for cls in classes:
                        per_class_counts[cls] = int(
                            round(synthetic_budget * real_class_counts[cls] / total_real)
                        )
                else:
                    minority_total_real = sum(real_class_counts[c] for c in minority_classes)
                    allocated = 0
                    for i, cls in enumerate(minority_classes):
                        if i < len(minority_classes) - 1:
                            n = int(round(
                                synthetic_budget * real_class_counts[cls] / minority_total_real
                            ))
                        else:
                            n = synthetic_budget - allocated
                        per_class_counts[cls] = max(0, n)
                        allocated += n

            elif strategy == "fully_balanced":
                # First, bring all classes up to majority count
                remaining_budget = synthetic_budget
                for cls in classes:
                    deficit = majority_count - real_class_counts[cls]
                    if deficit > 0:
                        add = min(deficit, remaining_budget)
                        per_class_counts[cls] = add
                        remaining_budget -= add

                # If there's remaining budget, distribute it proportionally
                if remaining_budget > 0 and strategy == "fully_balanced":
                    # After balancing, all classes have majority_count, so distribute evenly
                    per_class_additional = remaining_budget // len(classes)
                    for cls in classes:
                        per_class_counts[cls] += per_class_additional

            # Update maximums
            for cls in classes:
                max_per_class[cls] = max(max_per_class[cls], per_class_counts[cls])

    # Add 20% buffer for safety (in case of rounding or config changes)
    max_per_class = {
        cls: int(round(count * 1.2)) + 1
        for cls, count in max_per_class.items()
    }

    logger.info(
        f"Max synthetic requirements per class: {dict(max_per_class)} "
        f"(total={sum(max_per_class.values())})"
    )
    return max_per_class


def generate_synthetic_pool(
    cfg: Dict[str, Any],
    per_class_target_counts: Dict[str, int],
    output_dir: str | Path,
) -> "pd.DataFrame":  # noqa: F821
    """Generate `per_class_target_counts[class]` synthetic images for each
    class and save them to `output_dir/<class>/synthetic_XXXXXX.png`.

    Images are saved as float32 PNG files using the same z-score normalization
    format as the preprocessed real images.

    Args:
        cfg: Full pipeline config dict.
        per_class_target_counts: Dict mapping class name -> number of images to
            generate for that class.
        output_dir: Directory to save generated images and the index CSV.

    Returns:
        DataFrame with columns [filepath, label, is_synthetic] mirroring the
        schema used by data/preprocessing.py's real-data index, so both can
        be concatenated directly in experiment/dataset_builder.py.
    """
    import pandas as pd

    # For saving images
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    autoencoder, diffusion_unet, scheduler = load_trained_generator(cfg)
    device = cfg["project"]["device"]

    # Create a generator for deterministic sampling if seed is available
    seed = cfg["project"].get("seed", 42)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    rows = []
    for class_name, target_count in per_class_target_counts.items():
        if target_count <= 0:
            logger.info(f"Skipping class '{class_name}' (target count = {target_count})")
            continue

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        class_label = cfg["data"]["classes"].index(class_name)
        batch_size = min(
            cfg["generative"].get("batch_size", 16),
            target_count,
        )

        n_generated = 0
        pbar = tqdm(total=target_count, desc=f"Generating {class_name}")

        while n_generated < target_count:
            cur_batch = min(batch_size, target_count - n_generated)

            # Generate a batch
            images = sample_class_conditional_batch(
                autoencoder=autoencoder,
                diffusion_unet=diffusion_unet,
                scheduler=scheduler,
                class_label=class_label,
                batch_size=cur_batch,
                cfg=cfg,
                generator=generator,
            )

            # Save each image in the batch
            for i in range(images.shape[0]):
                img = images[i].cpu().numpy()  # (3, H, W)
                # The autoencoder was trained on 3-channel input where all
                # channels are the replicated grayscale value (see _to_rgb in
                # data/dataset.py), so decode returns 3 identical channels.
                # Take only the first channel to save as single-channel
                # grayscale, matching the real preprocessing output format.
                img = img[0]  # (H, W)

                # Save as float32 TIFF (lossless; matches preprocessing format).
                # NOTE: PNG does NOT support float32 via OpenCV — it silently
                # falls back to uint8 and clamps z-score values.
                out_name = f"synthetic_{n_generated + i:06d}.tiff"
                out_path = class_dir / out_name
                cv2.imwrite(
                    str(out_path),
                    img.astype(np.float32),
                )

                rows.append({
                    "filepath": str(out_path.resolve()),
                    "label": class_name,
                    "patient_id": f"synthetic_{class_name}_{n_generated + i}",
                    "is_synthetic": True,
                })

            n_generated += cur_batch
            pbar.update(cur_batch)
            # Update generator seed for the next batch to avoid identical samples
            generator.manual_seed(seed + n_generated)

        pbar.close()
        logger.info(f"Generated {n_generated} synthetic images for class '{class_name}'")

    index_df = pd.DataFrame(rows)
    index_path = output_dir / "synthetic_index.csv"
    index_df.to_csv(index_path, index=False)
    logger.info(
        f"Synthetic pool index saved to {index_path} "
        f"({len(index_df)} total images)"
    )

    return index_df


# Main entrypoint

if __name__ == "__main__":
    import argparse

    from utils.seed import load_config, set_global_seed

    parser = argparse.ArgumentParser(description="Generate synthetic OCT images per class")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    # Load real class counts from the class-analysis report produced by
    # data/preprocessing.py::run_class_analysis_report
    processed_dir = Path(cfg["data"]["processed_dir"])
    report_path = processed_dir / "splits" / "class_analysis_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Class analysis report not found at {report_path}. "
            "Run the preprocessing stage first."
        )

    import pandas as pd
    report_df = pd.read_csv(report_path)
    train_row = report_df[report_df["split"] == "train"]
    if len(train_row) == 0:
        raise ValueError("No 'train' split found in class analysis report.")

    real_class_counts = {}
    for class_name in cfg["data"]["classes"]:
        if class_name in train_row.columns:
            real_class_counts[class_name] = int(train_row[class_name].values[0])
        else:
            real_class_counts[class_name] = 0

    logger.info(f"Real training class counts: {real_class_counts}")

    max_requirement = compute_max_synthetic_requirement(
        real_class_counts,
        cfg["experiment"]["factor_a_synthetic_ratio"],
        cfg["experiment"]["factor_b_distribution_strategy"],
    )

    generate_synthetic_pool(
        cfg,
        max_requirement,
        Path(cfg["data"]["processed_dir"]) / "synthetic",
    )