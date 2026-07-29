"""Structured logging + experiment run naming conventions."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Return a configured logger that writes to stdout and optionally to file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated calls

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def cell_run_id(ratio: float, distribution: str, architecture: str, seed: int) -> str:
    """Canonical experimental-cell identifier used across datasets, checkpoints,
    result CSVs, and XAI outputs. E.g. `ratio0.50_minority_only_vit_base_seed1`.
    """
    return f"ratio{ratio:.2f}_{distribution}_{architecture}_seed{seed}"
