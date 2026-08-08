"""Standalone entrypoint for Stage 2+3: fine-tune the MONAI LDM and generate
the synthetic image pool. Kept separate from run_full_pipeline.py since this
is typically the longest-running, most GPU-hungry stage and benefits from
independent scheduling/checkpointing.

Usage:
    python scripts/run_generation.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data.preprocessing import compute_class_distribution
from generative import generate_synthetic, train_ldm
from utils.logging_utils import get_logger
from utils.seed import load_config, set_global_seed

logger = get_logger("run_generation")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_global_seed(cfg["project"]["seed"])

    logger.info("Fine-tuning MONAI LDM on real training data...")
    train_ldm.main(cfg)

    logger.info("Generating synthetic image pool...")
    import pandas as pd

    # Load the REAL train split's per-class counts from the class-analysis
    # report persisted by the preprocessing stage (data/preprocessing.py::
    # run_class_analysis_report). This mirrors the logic in
    # generate_synthetic.py::__main__ and fixes the previous ValueError
    # from passing an empty dict.
    processed_dir = Path(cfg["data"]["processed_dir"])
    report_path = processed_dir / "splits" / "class_analysis_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Class analysis report not found at {report_path}. "
            "Run the preprocessing stage first."
        )
    report_df = pd.read_csv(report_path)
    train_rows = report_df[report_df["split"] == "train"]
    if len(train_rows) == 0:
        raise ValueError("No 'train' split found in class analysis report.")

    real_train_counts = {}
    for class_name in cfg["data"]["classes"]:
        if class_name in train_rows.columns:
            real_train_counts[class_name] = int(train_rows[class_name].values[0])
        else:
            real_train_counts[class_name] = 0
    logger.info(f"Real training class counts: {real_train_counts}")

    max_requirement = generate_synthetic.compute_max_synthetic_requirement(
        real_train_counts,
        cfg["experiment"]["factor_a_synthetic_ratio"],
        cfg["experiment"]["factor_b_distribution_strategy"],
    )
    generate_synthetic.generate_synthetic_pool(
        cfg, max_requirement, processed_dir / "synthetic"
    )
    logger.info("Generation stage complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
