"""Train Classification Models stage: runs one experimental cell
(ResNet50 / EfficientNet-B0 / ViT, at a given synthetic ratio + distribution)
to convergence, checkpointing the best model for downstream evaluation and
explainability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from experiment.factorial_design import ExperimentalCell
from models.classifiers import build_classifier
from utils.logging_utils import cell_run_id, get_logger
from utils.seed import set_global_seed

logger = get_logger(__name__)


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: Dict[str, Any]):
    """Instantiate optimizer + LR scheduler per configs/config.yaml['training'].

    TODO: implement AdamW + cosine schedule (or configured alternatives).
    """
    raise NotImplementedError("TODO: build optimizer/scheduler from training config")


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device) -> float:
    """Single training epoch; returns average training loss.

    TODO: standard forward/backward/step loop with tqdm progress bar.
    """
    raise NotImplementedError("TODO: implement training epoch")


@torch.no_grad()
def validate_one_epoch(model, loader: DataLoader, criterion, device) -> Dict[str, float]:
    """Single validation epoch; returns at least {"loss": ..., "accuracy": ...}
    used for early stopping / best-checkpoint selection.

    TODO: implement validation loop.
    """
    raise NotImplementedError("TODO: implement validation epoch")


def train_experimental_cell(
    cell: ExperimentalCell,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    checkpoint_dir: str | Path,
) -> Path:
    """Full training run for one (ratio, distribution, architecture, seed)
    cell, with early stopping on validation loss/accuracy.

    Returns the path to the best model checkpoint for this run, to be
    consumed by evaluation/evaluate.py and explainability/*.py.
    """
    set_global_seed(cell.seed)
    device = cfg["project"]["device"]
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(cfg["data"]["classes"])
    model = build_classifier(cell.architecture, num_classes=num_classes).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_metric = float("-inf")
    patience_counter = 0
    checkpoint_path = checkpoint_dir / f"{cell_run_id(cell.ratio, cell.distribution_strategy, cell.architecture, cell.seed)}.pt"

    for epoch in range(cfg["training"]["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        logger.info(
            f"[{cell.run_id}] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_metric:
            best_val_metric = val_metrics["accuracy"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                logger.info(f"[{cell.run_id}] early stopping at epoch={epoch}")
                break

        # TODO: scheduler.step() placement depends on scheduler type (per-epoch vs per-batch)

    return checkpoint_path
