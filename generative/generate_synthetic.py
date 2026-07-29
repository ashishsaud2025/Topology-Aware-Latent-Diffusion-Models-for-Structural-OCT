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

import torch
from tqdm import tqdm

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_trained_generator(cfg: Dict[str, Any]):
    """Load the fine-tuned autoencoder + diffusion UNet + scheduler from the
    checkpoints produced by generative/train_ldm.py.

    TODO: mirror `train_ldm.build_autoencoder` / `build_diffusion_unet` /
    `build_scheduler`, then load_state_dict from the configured checkpoints
    and set both networks to eval() mode.
    """
    raise NotImplementedError("TODO: load fine-tuned MONAI LDM components")


@torch.no_grad()
def sample_class_conditional_batch(
    autoencoder,
    diffusion_unet,
    scheduler,
    class_label: int,
    batch_size: int,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Run the reverse diffusion process conditioned on `class_label` and
    decode the resulting latents back to image space via the autoencoder.

    TODO: implement the standard DDPM/DDIM sampling loop
    (start from Gaussian noise in latent space, iteratively denoise using
    `diffusion_unet` conditioned on class_label, then `autoencoder.decode`).
    Returns a tensor of shape (batch_size, C, H, W) in [0, 1] or [-1, 1]
    (be explicit and consistent with the real-data preprocessing range).
    """
    raise NotImplementedError("TODO: implement class-conditional DDPM/DDIM sampling")


def generate_synthetic_pool(
    cfg: Dict[str, Any],
    per_class_target_counts: Dict[str, int],
    output_dir: str | Path,
) -> "pd.DataFrame":  # noqa: F821 (pandas imported lazily to keep this module lightweight)
    """Generate `per_class_target_counts[class]` synthetic images for each
    class and save them to `output_dir/<class>/synthetic_XXXX.png`.

    Returns an index DataFrame [filepath, label, is_synthetic=True] mirroring
    the schema used by data/preprocessing.py's real-data index, so both can
    be concatenated directly in experiment/dataset_builder.py.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    autoencoder, diffusion_unet, scheduler = load_trained_generator(cfg)

    rows = []
    for class_name, target_count in per_class_target_counts.items():
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_label = cfg["data"]["classes"].index(class_name)

        n_generated = 0
        batch_size = cfg["generative"]["batch_size"]
        pbar = tqdm(total=target_count, desc=f"Generating {class_name}")
        while n_generated < target_count:
            cur_batch = min(batch_size, target_count - n_generated)
            images = sample_class_conditional_batch(
                autoencoder, diffusion_unet, scheduler, class_label, cur_batch, cfg
            )
            # TODO: save each image in `images` to class_dir with a unique
            # filename and append {filepath, label, is_synthetic: True} to rows.
            n_generated += cur_batch
            pbar.update(cur_batch)
        pbar.close()

    return pd.DataFrame(rows)


def compute_max_synthetic_requirement(
    real_class_counts: Dict[str, int],
    factor_a_ratios: List[float],
    factor_b_strategies: List[str],
) -> Dict[str, int]:
    """Compute, per class, the maximum number of synthetic images that ANY
    cell in the factorial grid could require, so we only need to run
    generation once for the whole study.

    TODO: implement the per-strategy arithmetic, e.g.:
      - proportional: synthetic count scales real class proportions by ratio
      - minority_only: synthetic images only added to below-average classes
      - fully_balanced: synthetic images added until all classes match the
        majority class count, then further scaled by ratio if ratio > that point
    Take the max across all (ratio, strategy) combinations per class.
    """
    raise NotImplementedError("TODO: implement max synthetic requirement computation")


if __name__ == "__main__":
    import argparse

    from utils.seed import load_config, set_global_seed

    parser = argparse.ArgumentParser(description="Generate synthetic OCT images per class")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    # TODO: load real_class_counts from the class-analysis report produced by
    # data/preprocessing.py::run_class_analysis_report
    real_class_counts: Dict[str, int] = {}  # TODO

    max_requirement = compute_max_synthetic_requirement(
        real_class_counts,
        cfg["experiment"]["factor_a_synthetic_ratio"],
        cfg["experiment"]["factor_b_distribution_strategy"],
    )
    generate_synthetic_pool(
        cfg, max_requirement, Path(cfg["data"]["processed_dir"]) / "synthetic"
    )
