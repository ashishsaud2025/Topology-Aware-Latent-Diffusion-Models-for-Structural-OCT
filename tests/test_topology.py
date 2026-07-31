"""Tests for the topology-aware validation stage (Stage 2B)."""
from __future__ import annotations

import numpy as np
import pytest

from topology.layer_segmentation import BOUNDARY_NAMES, RetinalLayerSegmenter
from topology.persistent_homology import compute_betti_numbers, compute_boundary_topology


def test_betti_single_component():
    """A single unbroken horizontal line has beta_0=1, beta_1=0."""
    b = np.zeros((20, 20), dtype=bool)
    b[10, :] = True
    assert compute_betti_numbers(b) == (1, 0)


def test_betti_two_components():
    """A broken line (gap) has beta_0=2."""
    b = np.zeros((20, 20), dtype=bool)
    b[10, :10] = True
    b[10, 12:] = True
    beta_0, _ = compute_betti_numbers(b)
    assert beta_0 == 2


def test_betti_loop():
    """A closed loop has beta_1=1."""
    b = np.zeros((20, 20), dtype=bool)
    b[5, 5:15] = True
    b[15, 5:15] = True
    b[5:16, 5] = True
    b[5:16, 15] = True
    _, beta_1 = compute_betti_numbers(b)
    assert beta_1 == 1


def test_empty_boundary():
    assert compute_betti_numbers(np.zeros((10, 10), dtype=bool)) == (0, 0)


def test_profile_segmenter_outputs_all_boundaries():
    """The profile segmenter must produce all 7 boundary maps."""
    img = np.zeros((112, 112), dtype=np.float32)
    for i in range(8):
        val = 0.5 + 0.5 * ((-1) ** i) * 0.4
        img[i * 14 : (i + 1) * 14, :] = val
    rng = np.random.default_rng(0)
    img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)

    segmenter = RetinalLayerSegmenter(backend="profile")
    result = segmenter.segment(img)

    assert set(result.boundaries.keys()) == set(BOUNDARY_NAMES)
    for bmap in result.boundaries.values():
        assert bmap.shape == img.shape


def test_boundary_topology_shape():
    """compute_boundary_topology returns entries for all 7 boundaries."""
    b = np.zeros((20, 20), dtype=bool)
    b[10, :] = True
    boundaries = {name: b for name in BOUNDARY_NAMES}
    sig = compute_boundary_topology(boundaries)
    assert len(sig) == 7
    for name in BOUNDARY_NAMES:
        assert "beta_0" in sig[name]
        assert "beta_1" in sig[name]


def test_validator_compare_distributions(tmp_path):
    """Validator produces a report with expected columns."""
    from topology.topological_validation import TopologyValidator

    rng = np.random.default_rng(1)

    def make_layered(name, n=6):
        paths = []
        for i in range(n):
            img = np.zeros((56, 56), dtype=np.float32)
            for j in range(4):
                val = 0.5 + 0.5 * ((-1) ** j) * 0.4
                img[j * 14 : (j + 1) * 14, :] = val
            img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)
            # Save as uint8 PNG (cv2 cannot read .npy)
            img_u8 = ((img - img.min()) / (np.ptp(img) + 1e-8) * 255.0).astype(np.uint8)
            p = tmp_path / f"{name}_{i}.png"
            import cv2
            cv2.imwrite(str(p), img_u8)
            paths.append(str(p))
        return paths

    real_paths = make_layered("real")
    synth_paths = make_layered("synth")

    validator = TopologyValidator(
        segmenter=RetinalLayerSegmenter(backend="profile"),
        use_persistence=False,
    )
    df = validator.compare_distributions(real_paths, synth_paths)
    assert list(df.columns) == [
        "boundary", "mean_beta0_real", "mean_beta0_synth",
        "mean_beta1_real", "mean_beta1_synth",
        "p_value_beta0", "p_value_beta1", "topology_passed",
    ]
    assert len(df) == 7