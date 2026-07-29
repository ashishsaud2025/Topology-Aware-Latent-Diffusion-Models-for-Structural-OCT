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

import torch

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

    # Load fixed val + test data once (same across all cells)
    import pandas as pd

    from data.dataset import OCTImageDataset, build_data_loaders, get_default_transforms

    # We load val from CSVs — the val split never receives synthetic data
    split_dir = Path(cfg["data"]["processed_dir"]) / "splits"
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    class_to_idx = {c: i for i, c in enumerate(cfg["data"]["classes"])}
    val_transform = get_default_transforms(cfg["data"]["image_size"], train=False)
    val_dataset = OCTImageDataset(val_df, class_to_idx, val_transform, image_size=cfg["data"]["image_size"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    _ = test_df  # test_loader not used during training; kept for consistency

    for cell in targets:
        logger.info(f"Training cell: {cell.run_id}")
        # Load the per-cell mixed training index
        safe_name = cell.run_id.replace("|", "_").replace("=", "")
        train_index_csv = index_dir / f"{safe_name}.csv"
        if not train_index_csv.exists():
            raise FileNotFoundError(
                f"Materialized training index not found: {train_index_csv}. "
                f"Run experiment/dataset_builder.py::materialize_all_cells first."
            )
        train_df = pd.read_csv(train_index_csv)
        train_transform = get_default_transforms(cfg["data"]["image_size"], train=True)
        train_dataset = OCTImageDataset(
            train_df, class_to_idx, train_transform, image_size=cfg["data"]["image_size"]
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        train_experimental_cell(cell, train_loader, val_loader, cfg, checkpoint_dir)
        logger.info(f"Completed cell: {cell.run_id}")


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
