"""Standalone entrypoint for Stage 2+3: fine-tune the MONAI LDM and generate
the synthetic image pool. Kept separate from run_full_pipeline.py since this
is typically the longest-running, most GPU-hungry stage and benefits from
independent scheduling/checkpointing.

Usage:
    python scripts/run_generation.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data.preprocessing import compute_class_distribution
from generative import generate_synthetic, train_ldm
from utils.logging_utils import get_logger
from utils.seed import load_config, set_global_seed

logger = get_logger("run_generation")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_global_seed(cfg["project"]["seed"])

    logger.info("Fine-tuning MONAI LDM on real training data...")
    train_ldm.main(cfg)

    logger.info("Generating synthetic image pool...")
    # TODO: load the real train split's class distribution (persisted from
    # the preprocessing stage) rather than recomputing/hardcoding it here.
    real_train_counts: dict = {}  # TODO
    max_requirement = generate_synthetic.compute_max_synthetic_requirement(
        real_train_counts,
        cfg["experiment"]["factor_a_synthetic_ratio"],
        cfg["experiment"]["factor_b_distribution_strategy"],
    )
    generate_synthetic.generate_synthetic_pool(
        cfg, max_requirement, Path(cfg["data"]["processed_dir"]) / "synthetic"
    )
    logger.info("Generation stage complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
