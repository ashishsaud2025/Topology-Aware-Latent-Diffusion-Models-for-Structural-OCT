"""Standalone entrypoint for Stages 7-11: evaluation, explainability,
quantitative XAI, statistical analysis, and H1-H4 hypothesis testing.

Usage:
    python scripts/run_analysis.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment.factorial_design import build_factorial_grid, deduplicate_baseline_cells
from hypotheses.hypothesis_tests import run_all_hypothesis_tests
from stats.statistical_analysis import summarize_main_and_interaction_effects
from utils.logging_utils import get_logger
from utils.seed import load_config

logger = get_logger("run_analysis")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg["project"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"

    cells = deduplicate_baseline_cells(build_factorial_grid(cfg))

    logger.info("Stage 7: evaluating all cells on the fixed real test set...")
    # TODO: build the fixed test_loader once (identical across all cells) via
    # data/dataset.py::OCTImageDataset over the FIXED real test split, then:
    # from evaluation.evaluate import evaluate_all_cells
    # results_df = evaluate_all_cells(cells, checkpoint_dir, test_loader, cfg,
    #                                  output_dir / "master_results.csv")
    raise NotImplementedError(
        "TODO: wire up the fixed test DataLoader, then call evaluate_all_cells(...) "
        "followed by the explainability + quantitative-XAI stages, and finally "
        "summarize_main_and_interaction_effects(...) / run_all_hypothesis_tests(...)."
    )

    # ---- Stage 8-9: Explainability + Quantitative XAI (per cell) -----------
    # TODO: for each cell, load its checkpoint, run gradcam/attention_rollout
    # per architecture branch, then explainability.quantitative_xai to produce
    # per-image IoU/Dice/CoM-distance/EMD, aggregated into xai_metrics_df.

    # ---- Stage 10: Statistical Analysis -------------------------------------
    # anova_summaries = summarize_main_and_interaction_effects(
    #     results_df, ["accuracy", "f1_macro", "roc_auc_ovr", "balanced_accuracy"], cfg
    # )
    # for metric, table in anova_summaries.items():
    #     table.to_csv(output_dir / f"anova_{metric}.csv")

    # ---- Stage 11: Experimental Findings (H1-H4) ----------------------------
    # findings = run_all_hypothesis_tests(results_df, xai_metrics_df, alpha=cfg["statistics"]["alpha"])
    # with open(output_dir / "hypothesis_findings.json", "w") as f:
    #     json.dump({k: v.statement for k, v in findings.items()}, f, indent=2)

    logger.info("Analysis stage complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
