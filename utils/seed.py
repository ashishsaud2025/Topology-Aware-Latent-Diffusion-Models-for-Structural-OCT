"""Reproducibility helpers: global seeding and config loading."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) for reproducibility.

    Called once at the start of every entrypoint (preprocessing, generation,
    training, analysis) so that seed-repeated experimental cells
    (`experiment.n_seeds_per_cell`) are truly independent runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # TODO: decide on determinism vs. throughput trade-off:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load the single YAML config that drives the entire pipeline."""
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
