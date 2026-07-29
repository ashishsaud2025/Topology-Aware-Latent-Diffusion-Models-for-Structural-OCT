"""Standalone entrypoint for Stage 5+6: materialize experimental datasets and
train every cell in the 3x5x3 factorial grid.

Supports an optional `--cell-index` argument so individual cells can be
dispatched to separate GPU jobs/nodes (e.g. via a Slurm array job or simple
shell loop) rather than training all ~45+ cells sequentially in one process.

Usage:
    # Train all cells sequentially:
    python scripts/run_experiment_grid.py --config configs/config.yaml

    # Train a single cell (for cluster array-job dispatch):
    python scripts/run_experiment_grid.py --config configs/config.yaml --cell-index 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment.factorial_design import build_factorial_grid, deduplicate_baseline_cells
from training.train_classifier import train_experimental_cell
from utils.logging_utils import get_logger
from utils.seed import load_config

logger = get_logger("run_experiment_grid")


def main(config_path: str, cell_index: int | None) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg["project"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    index_dir = output_dir / "experimental_indices"

    cells = deduplicate_baseline_cells(build_factorial_grid(cfg))
    targets = cells if cell_index is None else [cells[cell_index]]

    for cell in targets:
        logger.info(f"Training cell: {cell.run_id}")
        # TODO: build train_loader/val_loader from
        # index_dir / f"{cell.run_id-with-safe-chars}.csv" via
        # data/dataset.py::OCTImageDataset, then call:
        # train_experimental_cell(cell, train_loader, val_loader, cfg, checkpoint_dir)
        raise NotImplementedError(
            "TODO: wire up DataLoader construction from the materialized "
            "per-cell index CSVs, then call train_experimental_cell(...)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--cell-index",
        type=int,
        default=None,
        help="Train only the Nth cell in the factorial grid (for cluster array-job dispatch)",
    )
    args = parser.parse_args()
    main(args.config, args.cell_index)
