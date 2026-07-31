"""Persistent homology for retinal layer boundary maps.

Computes Betti numbers (beta_0, beta_1) and persistence diagrams from binary
boundary maps produced by topology/layer_segmentation.py.

  * beta_0 = connected components. Healthy layer boundary = 1.
  * beta_1 = holes/loops. Healthy layer boundary = 0.

Uses ripser/gudhi when available; otherwise falls back to an exact
connected-components + hole-counting algorithm (scipy/scikit-image).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.ndimage as ndi

from utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

try:
    import gudhi as gd
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False


def _connected_components(mask: np.ndarray) -> int:
    """Number of 8-connected components in a binary mask."""
    if mask.sum() == 0:
        return 0
    labeled, _ = ndi.label(mask, structure=np.ones((3, 3)))
    return int(labeled.max())


def _holes(boundary: np.ndarray) -> int:
    """Count enclosed holes via background components not touching borders."""
    if boundary.sum() == 0:
        return 0
    bg = ~boundary
    labeled_bg, _ = ndi.label(bg, structure=np.ones((3, 3)))
    if labeled_bg.max() == 0:
        return 0
    border_labels = (
        set(labeled_bg[0, :]) | set(labeled_bg[-1, :])
        | set(labeled_bg[:, 0]) | set(labeled_bg[:, -1])
    )
    border_labels.discard(0)
    return max(0, int(labeled_bg.max()) - len(border_labels))


def compute_betti_numbers(binary_map: np.ndarray, use_persistence: bool = True) -> Tuple[int, int]:
    """Compute (beta_0, beta_1) of a binary boundary map."""
    mask = np.asarray(binary_map, dtype=bool)
    if mask.sum() == 0:
        return (0, 0)

    if use_persistence and HAS_GUDHI:
        try:
            from gudhi import CubicalComplex

            cells = mask.astype(np.float64)
            cub = CubicalComplex(dimensions=cells.shape, top_dimensional_cells=cells.flatten())
            cub.compute_persistence(homology_coeff_field=2)
            beta_0 = len(cub.persistence_intervals_in_dimension(0))
            p1 = cub.persistence_intervals_in_dimension(1)
            beta_1 = sum(1 for b, d in p1 if np.isinf(d) or d > 1e9)
            return (int(beta_0), int(beta_1))
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Gudhi persistence failed ({exc}); using exact fallback.")

    # Exact combinatorial fallback
    return (_connected_components(mask), _holes(mask))


def compute_boundary_topology(
    boundaries: Dict[str, np.ndarray],
    use_persistence: bool = True,
) -> Dict[str, Dict[str, int]]:
    """Compute (beta_0, beta_1) for every boundary map in an image."""
    return {
        bname: {
            "beta_0": compute_betti_numbers(bmap, use_persistence)[0],
            "beta_1": compute_betti_numbers(bmap, use_persistence)[1],
        }
        for bname, bmap in boundaries.items()
    }


def compute_persistence_diagram(binary_map: np.ndarray) -> List[Tuple[int, Tuple[float, float]]]:
    """Compute full persistence diagram for a boundary map."""
    mask = np.asarray(binary_map, dtype=bool)
    if mask.sum() == 0:
        return []

    if HAS_RIPSER:
        try:
            pts = np.argwhere(mask).astype(np.float64)
            if len(pts) < 2:
                return [(0, (0.0, np.inf))]
            dgms = ripser.ripser(pts, maxdim=1, silent=True)["dgms"]
            return [
                (dim, (float(birth), float(death)))
                for dim, dgm in enumerate(dgms)
                for birth, death in dgm
            ]
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Ripser persistence failed ({exc}); returning empty diagram.")
            return []
    elif HAS_GUDHI:
        try:
            from gudhi import CubicalComplex

            cells = mask.astype(np.float64)
            cub = CubicalComplex(dimensions=cells.shape, top_dimensional_cells=cells.flatten())
            cub.compute_persistence(homology_coeff_field=2)
            return [
                (dim, (float(birth), float(death)))
                for dim in (0, 1)
                for birth, death in cub.persistence_intervals_in_dimension(dim)
            ]
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Gudhi persistence failed ({exc}); returning empty diagram.")
            return []
    logger.warning("No persistent-homology library available; returning empty diagram.")
    return []