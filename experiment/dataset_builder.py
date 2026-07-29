"""Create Experimental Datasets stage: for each ExperimentalCell, materialize
the exact real+synthetic training set implied by (ratio, distribution_strategy).

The FIXED REAL TEST SET (see data/preprocessing.py) is never touched here --
only the training (and optionally validation) split is augmented.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
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
      beyond that point is then split proportionally.
    """
    if not real_train_counts:
        return {}
    if ratio <= 0.0:
        return {c: 0 for c in real_train_counts}

    total_real = sum(real_train_counts.values())
    total_synth_budget = int(round(ratio * total_real))

    classes = sorted(real_train_counts.keys())
    counts = np.array([real_train_counts[c] for c in classes])
    majority_count = int(counts.max())
    mean_count = counts.mean()

    if distribution_strategy == "proportional":
        # Allocate in proportion to existing real class sizes
        proportions = counts / counts.sum()
        raw = proportions * total_synth_budget
        synth_per_class = {c: max(0, int(round(r))) for c, r in zip(classes, raw)}
        # Adjust rounding errors to hit the budget exactly
        allocated = sum(synth_per_class.values())
        diff = total_synth_budget - allocated
        if diff != 0:
            # Add/subtract from the largest class
            largest = max(synth_per_class, key=synth_per_class.get)
            synth_per_class[largest] = max(0, synth_per_class[largest] + diff)
        return synth_per_class

    elif distribution_strategy == "minority_only":
        # Only classes below the mean count get synthetic images
        minority_mask = counts < mean_count
        if not minority_mask.any():
            return {c: 0 for c in classes}
        minority_counts = counts[minority_mask]
        # Allocate budget proportional to how far below the mean each minority class is
        deficits = mean_count - minority_counts
        deficit_proportions = deficits / deficits.sum()
        raw = deficit_proportions * total_synth_budget
        result = {c: 0 for c in classes}
        for idx, c in enumerate(classes):
            if minority_mask[idx]:
                result[c] = max(0, int(round(raw[sum(minority_mask[:idx])])))
        # Adjust rounding
        allocated = sum(result.values())
        diff = total_synth_budget - allocated
        if diff != 0:
            # Give remaining to the most under-represented class
            most_minority = min(result, key=lambda c: (real_train_counts[c], -result[c]))
            result[most_minority] = max(0, result[most_minority] + diff)
        return result

    elif distribution_strategy == "fully_balanced":
        # 1) First pass: equalize all classes to the majority count
        deficits_to_majority = np.maximum(majority_count - counts, 0)
        equalize_cost = int(deficits_to_majority.sum())
        remaining_budget = total_synth_budget - equalize_cost

        # Allocate the equalization tokens first
        synth_per_class = {c: max(0, int(deficits_to_majority[i])) for i, c in enumerate(classes)}
        synth_per_class = {c: min(s, total_synth_budget) for c, s in synth_per_class.items()}

        # 2) Any remaining budget is split proportionally
        if remaining_budget > 0:
            # After equalization, all classes have at least majority_count
            # Counts after equalization
            post_equalize = np.array([
                max(counts[i], majority_count)
                for i in range(len(classes))
            ])
            proportions = post_equalize / post_equalize.sum()
            extra_raw = proportions * remaining_budget
            extra_alloc = {c: max(0, int(round(extra_raw[i]))) for i, c in enumerate(classes)}
            for c in classes:
                synth_per_class[c] += extra_alloc[c]
            # Adjust rounding
            allocated = sum(synth_per_class.values())
            diff = total_synth_budget - allocated
            if diff != 0:
                largest = max(synth_per_class, key=synth_per_class.get)
                synth_per_class[largest] = max(0, synth_per_class[largest] + diff)
        else:
            # Not enough budget to fully equalize; allocate proportionally
            # to the deficit magnitude
            deficits = deficits_to_majority
            if deficits.sum() > 0:
                proportions = deficits / deficits.sum()
                raw = proportions * total_synth_budget
                synth_per_class = {
                    c: max(0, int(round(raw[i])))
                    for i, c in enumerate(classes)
                }
                allocated = sum(synth_per_class.values())
                diff = total_synth_budget - allocated
                if diff != 0:
                    largest = max(synth_per_class, key=synth_per_class.get)
                    synth_per_class[largest] = max(0, synth_per_class[largest] + diff)
            else:
                synth_per_class = {c: 0 for c in classes}

        return synth_per_class

    else:
        raise ValueError(f"Unknown distribution_strategy: {distribution_strategy}")


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
        if n <= 0:
            continue
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