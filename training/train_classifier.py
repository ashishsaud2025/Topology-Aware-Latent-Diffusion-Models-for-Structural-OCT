"""Train Classification Models stage: runs one experimental cell
(ResNet50 / EfficientNet-B0 / ViT, at a given synthetic ratio + distribution)
to convergence, checkpointing the best model for downstream evaluation and
explainability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment.factorial_design import ExperimentalCell
from models.classifiers import build_classifier
from utils.logging_utils import cell_run_id, get_logger
from utils.seed import set_global_seed

logger = get_logger(__name__)


def build_optimizer_and_scheduler(
    model: torch.nn.Module, cfg: Dict[str, Any]
) -> Tuple[torch.optim.Optimizer, Any]:
    """Instantiate optimizer + LR scheduler per configs/config.yaml['training'].

    Supported optimizers: adamw, sgd
    Supported schedulers: cosine, step, plateau
    """
    train_cfg = cfg["training"]
    lr = train_cfg.get("learning_rate", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    scheduler_name = train_cfg.get("lr_scheduler", "cosine").lower()
    num_epochs = train_cfg.get("num_epochs", 50)

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == "step":
        step_size = train_cfg.get("lr_step_size", 10)
        gamma = train_cfg.get("lr_gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.1
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: GradScaler,
) -> float:
    """Single training epoch; returns average training loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Single validation epoch; returns at least {"loss": ..., "accuracy": ...}
    used for early stopping / best-checkpoint selection.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)

        with autocast(enabled=(device == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        num_batches += 1

    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(num_batches, 1)

    return {"loss": avg_loss, "accuracy": accuracy}


def train_experimental_cell(
    cell: ExperimentalCell,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    checkpoint_dir: str | Path,
) -> Path:
    """Full training run for one (ratio, distribution, architecture, seed)
    cell, with early stopping on validation accuracy.

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
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device == "cuda"))

    best_val_metric = float("-inf")
    patience_counter = 0
    early_stop_patience = cfg["training"].get("early_stopping_patience", 8)
    checkpoint_path = checkpoint_dir / f"{cell_run_id(cell.ratio, cell.distribution_strategy, cell.architecture, cell.seed)}.pt"

    num_epochs = cfg["training"].get("num_epochs", 50)
    scheduler_type = cfg["training"].get("lr_scheduler", "cosine").lower()

    logger.info(f"Training {cell.run_id} for up to {num_epochs} epochs (device={device})")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        # Step scheduler (cosine/step: per-epoch; plateau: per-epoch on val loss)
        if scheduler_type == "plateau":
            scheduler.step(val_metrics["loss"])
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[{cell.run_id}] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"lr={current_lr:.2e}"
        )

        if val_metrics["accuracy"] > best_val_metric:
            best_val_metric = val_metrics["accuracy"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  -> Saved best checkpoint ({best_val_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"[{cell.run_id}] early stopping at epoch={epoch}")
                break

    return checkpoint_path