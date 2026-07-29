"""Quantitative Explainability Analysis stage.

Converts qualitative Grad-CAM / Attention-Rollout heatmaps into quantitative,
statistically-testable metrics by comparing them against either:
  (a) ground-truth lesion/pathology annotation masks (if available,
      cfg['explainability']['lesion_mask_dir']), or
  (b) a reference/baseline model's attention map (e.g. the ratio=0.0,
      real-data-only model), when no clinical lesion masks exist -- used to
      measure how much synthetic augmentation SHIFTS attention relative to
      the un-augmented baseline (directly supporting H4).

Metrics: IoU, Dice Similarity Coefficient, Center-of-Mass (CoM) distance,
and (optional) Earth Mover's Distance (EMD / Wasserstein distance) treating
each heatmap as a 2D probability distribution over pixels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def binarize_heatmap(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize a normalized [0, 1] heatmap at `threshold`, used as the
    precursor to IoU / Dice computation."""
    return (heatmap >= threshold).astype(np.uint8)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice Similarity Coefficient (F1 over pixels) between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection / denom)


def compute_center_of_mass(heatmap: np.ndarray) -> np.ndarray:
    """Compute the (row, col) center-of-mass of a heatmap, weighting each
    pixel by its (non-negative) intensity.

    TODO: consider using the raw continuous heatmap (not the binarized mask)
    for a more precise centroid estimate, per the pipeline's
    "Center-of-Mass Distance" metric name.
    """
    heatmap = np.clip(heatmap, a_min=0, a_max=None)
    total_mass = heatmap.sum()
    if total_mass == 0:
        # Degenerate case: no attention anywhere -> default to image center.
        return np.array([heatmap.shape[0] / 2, heatmap.shape[1] / 2])

    rows = np.arange(heatmap.shape[0])[:, None]
    cols = np.arange(heatmap.shape[1])[None, :]
    row_com = (rows * heatmap).sum() / total_mass
    col_com = (cols * heatmap).sum() / total_mass
    return np.array([row_com, col_com])


def compute_center_of_mass_distance(heatmap_a: np.ndarray, heatmap_b: np.ndarray) -> float:
    """Euclidean distance between the centers-of-mass of two heatmaps
    (e.g. augmented-model attention vs. lesion mask, or vs. baseline-model
    attention). Lower = more spatially consistent attention.
    """
    com_a = compute_center_of_mass(heatmap_a)
    com_b = compute_center_of_mass(heatmap_b)
    return float(np.linalg.norm(com_a - com_b))


def compute_earth_movers_distance(heatmap_a: np.ndarray, heatmap_b: np.ndarray) -> float:
    """2D Earth Mover's Distance (Wasserstein distance) between two heatmaps
    treated as (normalized) probability distributions over pixel locations.

    TODO: implement via the `POT` (Python Optimal Transport) package, e.g.
    `ot.emd2` with a pixel-coordinate cost matrix, or `ot.sliced_wasserstein_distance`
    for a computationally cheaper approximation over larger heatmaps.
    """
    raise NotImplementedError("TODO: implement 2D EMD via the POT library")


def evaluate_heatmap_pair(
    model_heatmap: np.ndarray,
    reference_heatmap: np.ndarray,
    threshold: float = 0.5,
    compute_emd: bool = False,
) -> dict:
    """Compute the full quantitative-XAI metric suite for one (model
    heatmap, reference) pair -- reference being either a ground-truth lesion
    mask or a baseline model's attention map.
    """
    mask_a = binarize_heatmap(model_heatmap, threshold)
    mask_b = binarize_heatmap(reference_heatmap, threshold)

    metrics = {
        "iou": compute_iou(mask_a, mask_b),
        "dice": compute_dice(mask_a, mask_b),
        "center_of_mass_distance": compute_center_of_mass_distance(model_heatmap, reference_heatmap),
    }
    if compute_emd:
        metrics["earth_movers_distance"] = compute_earth_movers_distance(
            model_heatmap, reference_heatmap
        )
    return metrics


def aggregate_quantitative_xai_over_dataset(
    model_heatmaps: np.ndarray,
    reference_heatmaps: np.ndarray,
    threshold: float = 0.5,
    compute_emd: bool = False,
) -> "pd.DataFrame":  # noqa: F821
    """Compute per-image quantitative-XAI metrics over an entire evaluation
    set (e.g. the fixed real test set), returning a DataFrame ready to be
    merged into the master results table for H4 statistical testing.
    """
    import pandas as pd

    rows = []
    for i in range(model_heatmaps.shape[0]):
        rows.append(
            evaluate_heatmap_pair(
                model_heatmaps[i], reference_heatmaps[i], threshold, compute_emd
            )
        )
    return pd.DataFrame(rows)
