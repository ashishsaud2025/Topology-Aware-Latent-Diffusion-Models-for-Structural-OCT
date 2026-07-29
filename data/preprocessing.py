"""Stage 1 of the pipeline: Data Preprocessing & Class Analysis.

Responsibilities:
  - Load the raw real OCT dataset (e.g. Kermany et al. OCT2017, OCT-C8, or a
    private clinical dataset).
  - Standardize image format, resolution, and intensity normalization.
  - Perform a fixed, stratified train/val/test split. The TEST split must
    remain 100% real and untouched by synthetic data for the entire study
    (this is the "Fixed Real Test Set" referenced later in evaluation).
  - Quantify class imbalance to inform Factor B (distribution strategy) and
    to decide which classes are "minority" for the minority_only strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class ClassDistribution:
    """Per-class sample counts for a given data split."""
    counts: Dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def minority_classes(self) -> List[str]:
        """Classes below the mean class frequency; used to target Factor B
        `minority_only` synthetic allocation."""
        mean_count = self.total / len(self.counts)
        return [c for c, n in self.counts.items() if n < mean_count]

    def imbalance_ratio(self) -> float:
        """Majority:minority ratio, a standard imbalance severity metric."""
        return max(self.counts.values()) / min(self.counts.values())


def load_raw_dataset_index(raw_dir: str | Path, classes: List[str]) -> pd.DataFrame:
    """Walk `raw_dir` and build an index DataFrame with columns
    [filepath, label, patient_id].

    TODO: implement actual directory-walking logic matching your dataset's
    folder convention (e.g. raw_dir/<class>/<image>.jpeg). Ensure `patient_id`
    is extracted/parsed if available, since patient-level splitting (rather
    than image-level) is critical to avoid data leakage between train/val/test.
    """
    raise NotImplementedError("TODO: implement dataset indexing for your raw OCT source")


def compute_class_distribution(index_df: pd.DataFrame, label_col: str = "label") -> ClassDistribution:
    """Compute per-class counts from an indexed dataset DataFrame."""
    counts = index_df[label_col].value_counts().to_dict()
    return ClassDistribution(counts=counts)


def stratified_patient_level_split(
    index_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> Dict[str, pd.DataFrame]:
    """Split at the PATIENT level (not image level) to avoid leakage, while
    preserving class stratification within the split.

    Returns a dict: {"train": df, "val": df, "test": df}
    The "test" split is the FIXED REAL TEST SET used throughout evaluation;
    it must never receive synthetic images and must remain identical across
    every experimental cell for fair comparison.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "splits must sum to 1.0"
    # TODO: implement patient-level stratified split, e.g. via
    # sklearn.model_selection.StratifiedGroupKFold or a custom grouping routine.
    raise NotImplementedError("TODO: implement patient-level stratified split")


def preprocess_images(
    index_df: pd.DataFrame,
    output_dir: str | Path,
    image_size: int,
) -> pd.DataFrame:
    """Resize, normalize, and re-encode raw images into `output_dir`,
    returning an updated index pointing at the preprocessed files.

    TODO: implement resizing/normalization (consider CLAHE or standard
    z-score normalization commonly used for OCT preprocessing) and speckle
    noise handling if relevant to your OCT source.
    """
    raise NotImplementedError("TODO: implement image preprocessing pipeline")


def run_class_analysis_report(splits: Dict[str, pd.DataFrame], output_path: str | Path) -> None:
    """Generate and persist a class-balance report (counts, imbalance ratio,
    minority classes) per split, to be consulted when configuring
    Factor B (distribution strategy) in configs/config.yaml.
    """
    rows = []
    for split_name, df in splits.items():
        dist = compute_class_distribution(df)
        rows.append(
            {
                "split": split_name,
                "total": dist.total,
                "imbalance_ratio": dist.imbalance_ratio(),
                "minority_classes": ",".join(dist.minority_classes),
                **dist.counts,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)
