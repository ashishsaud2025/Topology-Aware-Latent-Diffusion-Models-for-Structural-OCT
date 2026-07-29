"""Reproducibility helpers: global seeding and config loading."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def set_global_seed(
    seed: int,
    deterministic: bool = False,
    benchmark: bool = True,
) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) for reproducibility.

    Called once at the start of every entrypoint (preprocessing, generation,
    training, analysis) so that seed-repeated experimental cells
    (`experiment.n_seeds_per_cell`) are truly independent runs.

    Args:
        seed: The random seed to use.
        deterministic: If True, enables torch deterministic mode (slower but
            fully reproducible). Disables cuDNN benchmark.
            Default: False (faster, but may have small numerical differences).
        benchmark: If True, enables cuDNN auto-tuner for best performance.
            Only effective when deterministic=False.
            Default: True.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(
                "PyTorch deterministic mode enabled (may reduce throughput)."
            )
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = benchmark
            logger.info(
                f"PyTorch seeded with seed={seed}, "
                f"cudnn.benchmark={benchmark}"
            )
    except ImportError:
        pass


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load the single YAML config that drives the entire pipeline."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Expand paths to absolute for consistency
    if "data" in cfg:
        for key in ("raw_dir", "processed_dir"):
            if key in cfg["data"] and cfg["data"][key] is not None:
                cfg["data"][key] = str(Path(cfg["data"][key]).resolve())

    logger.info(f"Loaded configuration from {config_path}")
    return cfg