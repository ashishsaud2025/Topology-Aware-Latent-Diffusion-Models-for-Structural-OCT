"""Topology-aware validation stage (Stage 2B).

Implements retinal layer segmentation and persistent-homology based
topological validation of synthetic OCT images against the real-data
distribution, following the approach of Yamauchi, Wu, and Okada (2025):
synthetic images must preserve the same retinal layer-continuity topology
as real images before they are admitted into the classification pipeline.

Public API:
    RetinalLayerSegmenter   -- segments OCT B-scans into 8 layer regions and
                               produces binary boundary maps between layers
    compute_betti_numbers   -- exact (beta_0, beta_1) of a binary boundary map
    compute_boundary_topology -- per-boundary Betti numbers for one image
    TopologyValidator       -- compares synthetic vs real topological signatures
    run_topological_validation -- end-to-end Stage 2B pipeline
"""

from topology.layer_segmentation import BOUNDARY_NAMES, LAYER_NAMES, RetinalLayerSegmenter
from topology.persistent_homology import (
    compute_betti_numbers,
    compute_boundary_topology,
    compute_persistence_diagram,
)
from topology.topological_validation import TopologyValidator, run_topological_validation

__all__ = [
    "BOUNDARY_NAMES",
    "LAYER_NAMES",
    "RetinalLayerSegmenter",
    "compute_betti_numbers",
    "compute_boundary_topology",
    "compute_persistence_diagram",
    "TopologyValidator",
    "run_topological_validation",
]