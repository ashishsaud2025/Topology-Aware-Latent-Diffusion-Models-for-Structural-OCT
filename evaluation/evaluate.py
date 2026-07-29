"""Evaluate on Fixed Real Test Set stage.

Computes Accuracy, Precision, Recall, F1-score, ROC-AUC, and a full
per-class classification report for a trained model checkpoint, always
against the SAME fixed, 100%-real test split (see
data/preprocessing.py::stratified_patient_level_split) so results are
directly comparable across every cell of the 3x5x3 factorial grid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from experiment.factorial_design import ExperimentalCell
from models.classifiers import build_classifier


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, test_loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over the fixed test set, returning (y_true, y_pred,
    y_proba) as numpy arrays for metric computation.

    Softmax is applied to logits to produce y_proba for ROC-AUC computation.
    """
    model.to(device).eval()

    all_true: list[int] = []
    all_pred: list[int] = []
    all_proba: list[np.ndarray] = []

    for batch in test_loader:
        images, labels = batch[0].to(device), batch[1].to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        _, predicted = logits.max(1)

        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(predicted.cpu().numpy().tolist())
        all_proba.extend(probs.cpu().numpy())

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_proba),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute the full metric suite required by the pipeline:
    Accuracy, Precision, Recall, F1-score, ROC-AUC, per-class performance.

    - Precision/Recall/F1 reported as macro AND weighted averages (macro is
      what H2 specifically requires for class-balance evaluation).
    - ROC-AUC computed one-vs-rest (multi_class="ovr") for the multi-class
      setting.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "per_class_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
    }

    # Balanced accuracy (mean of per-class recall) -- explicitly required by H2
    per_class_recall = [
        metrics["per_class_report"][c]["recall"] for c in class_names
    ]
    metrics["balanced_accuracy"] = float(np.mean(per_class_recall))

    return metrics


def evaluate_experimental_cell(
    cell: ExperimentalCell,
    checkpoint_path: str | Path,
    test_loader: DataLoader,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Load a trained checkpoint for `cell` and evaluate it on the fixed real
    test set, returning a metrics dict tagged with the cell's factor levels
    (ready to be appended to the master results table consumed by
    stats/statistical_analysis.py and hypotheses/hypothesis_tests.py).
    """
    device = cfg["project"]["device"]
    num_classes = len(cfg["data"]["classes"])

    model = build_classifier(cell.architecture, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    y_true, y_pred, y_proba = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_proba, cfg["data"]["classes"])

    metrics.update(
        {
            "ratio": cell.ratio,
            "distribution_strategy": cell.distribution_strategy,
            "architecture": cell.architecture,
            "seed": cell.seed,
            "run_id": cell.run_id,
        }
    )
    return metrics


def evaluate_all_cells(
    cells: List[ExperimentalCell],
    checkpoint_dir: str | Path,
    test_loader: DataLoader,
    cfg: Dict[str, Any],
    output_csv: str | Path,
) -> "pd.DataFrame":  # noqa: F821
    """Evaluate every trained cell and persist the master results table,
    the single artifact consumed by all downstream statistical analysis and
    hypothesis testing.
    """
    import pandas as pd

    from utils.logging_utils import cell_run_id

    checkpoint_dir = Path(checkpoint_dir)
    rows = []
    for cell in cells:
        ckpt_path = checkpoint_dir / f"{cell_run_id(cell.ratio, cell.distribution_strategy, cell.architecture, cell.seed)}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {cell.run_id}: {ckpt_path}")
        rows.append(evaluate_experimental_cell(cell, ckpt_path, test_loader, cfg))

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_csv, index=False)
    return results_df
