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
    pixel by its raw continuous (non-negative) intensity.
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


def compute_earth_movers_distance(
    heatmap_a: np.ndarray, heatmap_b: np.ndarray, max_samples: int = 512
) -> float:
    """2D Earth Mover's Distance (Wasserstein distance) between two heatmaps
    treated as (normalized) probability distributions over pixel locations.

    Implemented via the POT library: pixel locations are subsampled to at most
    `max_samples` points (stratified grid pattern, deterministic/reproducible)
    so the exact `ot.emd2` transport LP builds a cost matrix of at most
    max_samples x max_samples.
    """
    import ot

    if heatmap_a.shape != heatmap_b.shape:
        # Upsample the smaller to the larger for a fair comparison
        from skimage.transform import resize

        target = heatmap_b.shape if heatmap_a.size < heatmap_b.size else heatmap_a.shape
        heatmap_a = resize(heatmap_a, target, mode="reflect", preserve_range=True)
        heatmap_b = resize(heatmap_b, target, mode="reflect", preserve_range=True)

    h, w = heatmap_a.shape
    total = h * w

    # Subsample a deterministic grid of pixel locations (≈ max_samples points)
    if total <= max_samples:
        stride = 1
    else:
        stride = int(np.ceil(np.sqrt(total / max_samples)))
    rows_s = np.arange(0, h, stride)
    cols_s = np.arange(0, w, stride)
    grid_rows, grid_cols = np.meshgrid(rows_s, cols_s, indexing="ij")
    locs_a = np.stack([grid_rows.ravel(), grid_cols.ravel()], axis=1).astype(float)
    w_a = heatmap_a[np.ix_(rows_s, cols_s)].ravel()
    locs_b = locs_a.copy()
    w_b = heatmap_b[np.ix_(rows_s, cols_s)].ravel()

    # Normalize weights to probability distributions
    w_a = np.clip(w_a, 0, None)
    w_b = np.clip(w_b, 0, None)
    if w_a.sum() <= 0 or w_b.sum() <= 0:
        return float("nan")
    w_a = w_a / w_a.sum()
    w_b = w_b / w_b.sum()

    # Cost matrix: Euclidean distance between pixel locations
    M = ot.dist(locs_a, locs_b, metric="euclidean")
    return float(ot.emd2(w_a, w_b, M))


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
