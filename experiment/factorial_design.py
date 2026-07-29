"""Experimental Design stage: builds the full 3x5x3 factorial grid.

Factor A: Synthetic Ratio            (5 levels, e.g. 0%, 25%, 50%, 75%, 100%)
Factor B: Synthetic Distribution      (3 levels: proportional, minority_only, fully_balanced)
Factor C: Classification Architecture (3 levels: ResNet50, EfficientNet-B0, ViT)

Note: the doc's diagram lists "3x5x3" in the research-gap text (3 architectures
x 5 ratios x 3 distributions), matching Factor A having 5 levels here even
though it's drawn first in the pipeline diagram -- the *design* is the same
regardless of factor ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List


@dataclass(frozen=True)
class ExperimentalCell:
    """A single fully-specified experimental run: one combination of
    (ratio, distribution strategy, architecture, seed)."""

    ratio: float
    distribution_strategy: str
    architecture: str
    seed: int

    @property
    def cell_key(self) -> str:
        """Identifier excluding seed -- groups seed-repeats of the same cell
        together for later aggregation/statistics."""
        return f"ratio={self.ratio:.2f}|dist={self.distribution_strategy}|arch={self.architecture}"

    @property
    def run_id(self) -> str:
        return f"{self.cell_key}|seed={self.seed}"


def build_factorial_grid(cfg: Dict[str, Any]) -> List[ExperimentalCell]:
    """Enumerate every (Factor A x Factor B x Factor C x seed) combination
    from the experiment config, producing the full list of runs required for
    `Train Classification Models` in the pipeline.
    """
    exp_cfg = cfg["experiment"]
    ratios = exp_cfg["factor_a_synthetic_ratio"]
    strategies = exp_cfg["factor_b_distribution_strategy"]
    architectures = exp_cfg["factor_c_architecture"]
    n_seeds = exp_cfg["n_seeds_per_cell"]

    cells = [
        ExperimentalCell(ratio=r, distribution_strategy=d, architecture=a, seed=s)
        for r, d, a, s in product(ratios, strategies, architectures, range(n_seeds))
    ]
    return cells


def deduplicate_baseline_cells(cells: List[ExperimentalCell]) -> List[ExperimentalCell]:
    """At ratio=0.0, the distribution strategy (Factor B) is a no-op since
    there is no synthetic data to distribute. This collapses redundant
    ratio=0.0 cells across the 3 distribution-strategy levels down to a
    single baseline-per-architecture-per-seed set, avoiding wasted compute
    on identical training runs.

    TODO: decide whether to keep this collapsing (compute-efficient) or run
    them independently (simpler bookkeeping, cleaner ANOVA balance) --
    a fully balanced 3x5x3 factorial design for classical ANOVA assumes equal
    cell sizes, so collapsing the ratio=0 cells creates an UNBALANCED design
    that the statistics module must account for (see stats/statistical_analysis.py).
    """
    seen = set()
    deduped = []
    for cell in cells:
        if cell.ratio == 0.0:
            key = (cell.architecture, cell.seed)  # distribution irrelevant at ratio 0
            if key in seen:
                continue
            seen.add(key)
        deduped.append(cell)
    return deduped


def grid_summary(cells: List[ExperimentalCell]) -> Dict[str, int]:
    """Quick sanity-check summary: total cells, unique architectures, etc."""
    return {
        "total_runs": len(cells),
        "unique_cell_keys": len({c.cell_key for c in cells}),
        "architectures": len({c.architecture for c in cells}),
        "ratios": len({c.ratio for c in cells}),
        "distribution_strategies": len({c.distribution_strategy for c in cells}),
    }
