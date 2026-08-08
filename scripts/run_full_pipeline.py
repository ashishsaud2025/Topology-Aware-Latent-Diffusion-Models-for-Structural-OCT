"""End-to-end orchestrator mirroring the full pipeline diagram:

Real OCT Dataset -> Preprocessing/Class Analysis -> Fine-tune MONAI LDM ->
Generate Synthetic Images -> Build Factorial Grid -> Create Experimental
Datasets -> Train Classifiers -> Evaluate -> Explainability ->
Quantitative XAI -> Statistical Analysis -> Experimental Findings (H1-H4)

Intended as a reference/orchestration script; for actual large-scale runs,
prefer the individual stage scripts (run_generation.py, run_experiment_grid.py,
run_analysis.py) so long-running stages (LDM fine-tuning, full factorial
training) can be run/resumed independently (e.g. across multiple GPU jobs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data.preprocessing import (
    compute_class_distribution,
    load_raw_dataset_index,
    preprocess_images,
    run_class_analysis_report,
    stratified_patient_level_split,
)
from experiment.dataset_builder import materialize_all_cells
from experiment.factorial_design import build_factorial_grid, deduplicate_baseline_cells, grid_summary
from generative.generate_synthetic import compute_max_synthetic_requirement, generate_synthetic_pool
from hypotheses.hypothesis_tests import run_all_hypothesis_tests
from stats.statistical_analysis import summarize_main_and_interaction_effects
from utils.logging_utils import get_logger
from utils.seed import load_config, set_global_seed

logger = get_logger("run_full_pipeline")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_global_seed(cfg["project"]["seed"])
    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Data Preprocessing & Class Analysis 
    logger.info("Stage 1: preprocessing + class analysis")
    raw_index = load_raw_dataset_index(
        cfg["data"]["raw_dir"],
        cfg["data"]["classes"],
        max_samples_per_class=cfg["data"].get("max_samples_per_class"),
        seed=cfg["project"]["seed"],
    )
    processed_index = preprocess_images(raw_index, cfg["data"]["processed_dir"], cfg["data"]["image_size"])
    splits = stratified_patient_level_split(
        processed_index,
        train_frac=cfg["data"]["train_split"],
        val_frac=cfg["data"]["val_split"],
        test_frac=cfg["data"]["test_split"],
        seed=cfg["project"]["seed"],
    )
    run_class_analysis_report(splits, output_dir / "class_analysis_report.csv")

    # Stage 2: Fine-tune MONAI Generative Model 
    logger.info("Stage 2: fine-tune MONAI LDM (see generative/train_ldm.py for full loop)")
    # NOTE: run separately in practice -- `python -m generative.train_ldm --config ...`
    # since this is a long-running, checkpointed process.

    # Stage 3: Generate Synthetic Images per Class 
    logger.info("Stage 3: generate synthetic image pool")
    real_train_counts = compute_class_distribution(splits["train"]).counts
    max_requirement = compute_max_synthetic_requirement(
        real_train_counts,
        cfg["experiment"]["factor_a_synthetic_ratio"],
        cfg["experiment"]["factor_b_distribution_strategy"],
    )
    synthetic_pool_df = generate_synthetic_pool(
        cfg, max_requirement, Path(cfg["data"]["processed_dir"]) / "synthetic"
    )

    # Stage 2B: Topological Validation of Synthetic Images 
    logger.info("Stage 2B: topological validation (real vs synthetic layer structure)")
    from topology.topological_validation import run_topological_validation

    topology_summary = run_topological_validation(
        cfg,
        real_index=splits["train"],
        synthetic_index=synthetic_pool_df,
        output_dir=output_dir / "topology",
    )
    n_failed = (
        int((~topology_summary["topology_passed"]).sum())
        if not topology_summary.empty else -1
    )
    if n_failed > 0:
        logger.warning(
            f"Topology gate: {n_failed} boundary-class comparisons FAILED. "
            "Review topology/topology_report.csv before proceeding."
        )
    else:
        logger.info("Topology gate PASSED: synthetic images preserve real layer topology.")

    # Stage 4: Experimental Design (3x5x3 factorial grid) 
    logger.info("Stage 4: build factorial design")
    cells = build_factorial_grid(cfg)
    cells = deduplicate_baseline_cells(cells)
    logger.info(f"Factorial grid summary: {grid_summary(cells)}")

    # Stage 5: Create Experimental Datasets 
    logger.info("Stage 5: materialize per-cell training datasets")
    materialize_all_cells(
        splits["train"], synthetic_pool_df, cells, output_dir / "experimental_indices"
    )

    # Stage 6: Train Classification Models 
    logger.info("Stage 6: train classifiers (see scripts/run_experiment_grid.py)")
    # NOTE: run separately in practice, parallelized across cells/GPUs.

    # Stage 7: Evaluate on Fixed Real Test Set 
    # Stage 8: Explainability Analysis 
    # Stage 9: Quantitative Explainability Analysis 
    logger.info("Stages 7-9: evaluation + explainability (see scripts/run_analysis.py)")

    # Stage 10: Statistical Analysis 
    # Stage 11: Experimental Findings (H1-H4) 
    logger.info("Stages 10-11: statistical analysis + hypothesis testing")
    # TODO: load master results_df + xai_metrics_df produced by earlier stages
    # results_df = pd.read_csv(output_dir / "master_results.csv")
    # xai_metrics_df = pd.read_csv(output_dir / "xai_metrics.csv")
    # anova_summaries = summarize_main_and_interaction_effects(results_df, ["accuracy", "f1_macro"], cfg)
    # findings = run_all_hypothesis_tests(results_df, xai_metrics_df, alpha=cfg["statistics"]["alpha"])

    logger.info("Pipeline scaffold complete. Fill in TODOs stage-by-stage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full OCT LDM augmentation pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
