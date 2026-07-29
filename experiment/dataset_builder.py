"""Create Experimental Datasets stage: for each ExperimentalCell, materialize
the exact real+synthetic training set implied by (ratio, distribution_strategy).

The FIXED REAL TEST SET (see data/preprocessing.py) is never touched here --
only the training (and optionally validation) split is augmented.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from experiment.factorial_design import ExperimentalCell


def allocate_synthetic_counts(
    real_train_counts: Dict[str, int],
    ratio: float,
    distribution_strategy: str,
) -> Dict[str, int]:
    """Compute how many synthetic images per class to inject, given the real
    training class counts, the target ratio, and the distribution strategy.

    - proportional: synthetic images added to every class in proportion to
      its existing real share, i.e. total_synthetic = ratio * total_real,
      split across classes by their real-class proportions.
    - minority_only: total_synthetic budget (ratio * total_real) is spent
      entirely on classes below the mean class count, prioritizing the most
      under-represented classes first.
    - fully_balanced: synthetic images are added first to equalize all
      classes to the majority-class count; any additional ratio budget
      beyond that point is then split proportionally (TODO: confirm this
      tie-breaking rule matches your intended H2 operationalization).

    TODO: implement the three branches precisely; unit-test them against
    known small examples (see tests/test_factorial_design.py for scaffolding).
    """
    raise NotImplementedError("TODO: implement per-strategy synthetic allocation")


def build_mixed_training_index(
    real_train_df: pd.DataFrame,
    synthetic_pool_df: pd.DataFrame,
    cell: ExperimentalCell,
    label_col: str = "label",
) -> pd.DataFrame:
    """Construct the final training index for one experimental cell by
    sampling `synthetic_pool_df` per-class according to
    `allocate_synthetic_counts`, then concatenating with `real_train_df`.

    Uses `cell.seed` to control which specific synthetic images are sampled
    (important for the seed-repeat variance estimates in the factorial design).
    """
    real_counts = real_train_df[label_col].value_counts().to_dict()
    synth_counts = allocate_synthetic_counts(
        real_counts, cell.ratio, cell.distribution_strategy
    )

    sampled_frames = [real_train_df]
    for class_name, n in synth_counts.items():
        class_pool = synthetic_pool_df[synthetic_pool_df[label_col] == class_name]
        if n > len(class_pool):
            raise ValueError(
                f"Requested {n} synthetic images for class '{class_name}' but only "
                f"{len(class_pool)} available in the pre-generated pool. "
                f"Re-run generative/generate_synthetic.py with a larger target count."
            )
        sampled_frames.append(class_pool.sample(n=n, random_state=cell.seed))

    return pd.concat(sampled_frames, ignore_index=True)


def materialize_all_cells(
    real_train_df: pd.DataFrame,
    synthetic_pool_df: pd.DataFrame,
    cells: list,
    output_index_dir: str,
) -> None:
    """Convenience driver: build + persist the mixed training index CSV for
    every cell in the factorial grid, so training/train_classifier.py can
    simply read `output_index_dir/<run_id>.csv` per run.
    """
    from pathlib import Path

    output_index_dir = Path(output_index_dir)
    output_index_dir.mkdir(parents=True, exist_ok=True)

    for cell in cells:
        mixed_df = build_mixed_training_index(real_train_df, synthetic_pool_df, cell)
        safe_name = cell.run_id.replace("|", "_").replace("=", "")
        mixed_df.to_csv(output_index_dir / f"{safe_name}.csv", index=False)
