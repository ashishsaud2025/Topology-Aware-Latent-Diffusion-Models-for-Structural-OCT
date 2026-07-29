"""Standalone entrypoint for Stages 7-11: evaluation, explainability,
quantitative XAI, statistical analysis, and H1-H4 hypothesis testing.

Usage:
    python scripts/run_analysis.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.dataset import OCTImageDataset, get_default_transforms
from evaluation.evaluate import evaluate_all_cells
from experiment.factorial_design import build_factorial_grid, deduplicate_baseline_cells
from hypotheses.hypothesis_tests import run_all_hypothesis_tests
from stats.statistical_analysis import summarize_main_and_interaction_effects
from utils.logging_utils import get_logger
from utils.seed import load_config
import torch

logger = get_logger("run_analysis")


def build_fixed_test_loader(cfg: dict) -> torch.utils.data.DataLoader:
    """Build the single fixed real test set DataLoader (identical across all
    cells) from the persisted test split CSV."""

    split_dir = Path(cfg["data"]["processed_dir"]) / "splits"
    test_df = pd.read_csv(split_dir / "test.csv")
    class_to_idx = {c: i for i, c in enumerate(cfg["data"]["classes"])}
    transform = get_default_transforms(cfg["data"]["image_size"], train=False)
    dataset = OCTImageDataset(
        test_df, class_to_idx, transform, image_size=cfg["data"]["image_size"]
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg["project"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"

    cells = deduplicate_baseline_cells(build_factorial_grid(cfg))
    n_cells = len(cells)
    logger.info(f"Factorial grid: {n_cells} cells after dedup")

    # ---- Stage 7: Evaluation -----------------------------------------------
    logger.info("Stage 7: evaluating all cells on the fixed real test set...")
    test_loader = build_fixed_test_loader(cfg)
    results_path = output_dir / "master_results.csv"
    if results_path.exists():
        logger.info(f"Loading existing results from {results_path}")
        results_df = pd.read_csv(results_path)
    else:
        results_df = evaluate_all_cells(cells, checkpoint_dir, test_loader, cfg, results_path)
    logger.info(f"Evaluated {len(results_df)} cells")

    # ---- Stage 8-9: Explainability + Quantitative XAI (per cell) -----------
    # NOTE: explainability stages require loading each checkpoint, running
    # GradCAM (resnet/efficientnet) or AttentionRollout (ViT), then computing
    # quantitative metrics (IoU, Dice, CoM-distance, EMD) against the
    # ground-truth lesion masks (if available).
    #
    # Since these stages depend on the existence of pixel-level masks (which
    # may not be available for all datasets), they are left as a post-hoc
    # extension point. When masks are available, uncomment the following:
    #
    # from explainability.quantitative_xai import aggregate_quantitative_xai_over_dataset
    # xai_metrics_path = output_dir / "xai_metrics.csv"
    # if not xai_metrics_path.exists():
    #     xai_metrics_df = aggregate_quantitative_xai_over_dataset(
    #         cells, checkpoint_dir, test_loader, cfg, xai_metrics_path,
    #     )
    # else:
    #     xai_metrics_df = pd.read_csv(xai_metrics_path)
    # logger.info(f"XAI metrics computed for {len(xai_metrics_df)} cells")
    xai_metrics_df = None

    # ---- Stage 10: Statistical Analysis -------------------------------------
    logger.info("Stage 10: statistical analysis (3-way factorial ANOVA)...")
    metric_vars = ["accuracy", "f1_macro", "roc_auc_ovr", "balanced_accuracy"]
    anova_summaries = summarize_main_and_interaction_effects(
        results_df, metric_vars, cfg
    )
    for metric, table in anova_summaries.items():
        out = output_dir / f"anova_{metric}.csv"
        table.to_csv(out)
        logger.info(f"  ANOVA table for '{metric}' saved to {out}")

    # ---- Stage 11: Experimental Findings (H1-H4) ----------------------------
    logger.info("Stage 11: hypothesis testing (H1-H4)...")
    alpha = cfg.get("statistics", {}).get("alpha", 0.05)
    findings = run_all_hypothesis_tests(results_df, xai_metrics_df, alpha=alpha)

    findings_out = output_dir / "hypothesis_findings.json"
    with open(findings_out, "w") as f:
        json.dump(
            {
                hid: {
                    "statement": res.statement,
                    "supported": bool(res.supported) if res.supported is not None else None,
                    "evidence_keys": list(res.evidence.keys()),
                }
                for hid, res in findings.items()
            },
            f,
            indent=2,
        )
    logger.info(f"Hypothesis findings saved to {findings_out}")

    logger.info("Analysis stage complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
