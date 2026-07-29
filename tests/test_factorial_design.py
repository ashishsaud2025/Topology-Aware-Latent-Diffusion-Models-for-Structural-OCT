"""Tests for experiment/factorial_design.py -- the only module with fully
implemented (non-TODO) logic, so it's covered now; add tests for other
modules as their TODOs are filled in.
"""
from experiment.factorial_design import (
    build_factorial_grid,
    deduplicate_baseline_cells,
    grid_summary,
)

TEST_CFG = {
    "experiment": {
        "factor_a_synthetic_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "factor_b_distribution_strategy": ["proportional", "minority_only", "fully_balanced"],
        "factor_c_architecture": ["resnet50", "efficientnet_b0", "vit_base"],
        "n_seeds_per_cell": 3,
    }
}


def test_build_factorial_grid_full_size():
    cells = build_factorial_grid(TEST_CFG)
    # 5 ratios x 3 distributions x 3 architectures x 3 seeds
    assert len(cells) == 5 * 3 * 3 * 3


def test_deduplicate_baseline_cells_collapses_ratio_zero():
    cells = build_factorial_grid(TEST_CFG)
    deduped = deduplicate_baseline_cells(cells)

    # Non-baseline cells (ratio > 0) should be untouched
    non_baseline_original = [c for c in cells if c.ratio != 0.0]
    non_baseline_deduped = [c for c in deduped if c.ratio != 0.0]
    assert len(non_baseline_original) == len(non_baseline_deduped)

    # Baseline cells (ratio == 0.0) should collapse to architectures x seeds only
    baseline_deduped = [c for c in deduped if c.ratio == 0.0]
    assert len(baseline_deduped) == 3 * 3  # architectures x seeds


def test_grid_summary_keys():
    cells = deduplicate_baseline_cells(build_factorial_grid(TEST_CFG))
    summary = grid_summary(cells)
    assert set(summary.keys()) == {
        "total_runs",
        "unique_cell_keys",
        "architectures",
        "ratios",
        "distribution_strategies",
    }
    assert summary["architectures"] == 3
    assert summary["ratios"] == 5
    assert summary["distribution_strategies"] == 3
