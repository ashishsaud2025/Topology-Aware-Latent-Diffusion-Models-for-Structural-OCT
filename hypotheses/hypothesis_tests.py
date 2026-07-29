"""Hypothesis test runners: translates the statistical machinery in
stats/statistical_analysis.py into direct, reportable answers to H1-H4 from
the research proposal.

H1: Optimal synthetic ratio exists (diminishing/negative returns beyond it).
H2: Minority-only distribution > proportional/fully-balanced for
    class-balanced performance (macro-F1, per-class recall, balanced accuracy).
H3: CNNs and ViTs respond differently to augmentation (architecture x
    ratio/distribution interaction effects).
H4: Augmentation strategy affects explainability; the optimal strategy
    preserves/enhances attention localization quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from stats.statistical_analysis import (
    find_optimal_ratio,
    run_pairwise_posthoc,
    run_three_way_anova,
)


@dataclass
class HypothesisResult:
    hypothesis_id: str
    statement: str
    supported: Optional[bool]  # None until the TODO analysis logic is implemented
    evidence: Dict[str, Any]


def test_h1_synthetic_ratio_effect(results_df: pd.DataFrame, alpha: float = 0.05) -> HypothesisResult:
    """H1: There exists an optimal synthetic ratio maximizing performance,
    beyond which additional synthetic data yields no significant improvement
    or degrades performance.

    Approach:
      1. ANOVA on ratio's main effect for accuracy / F1 / AUC.
      2. Tukey HSD pairwise comparison across ratio levels.
      3. find_optimal_ratio() to identify the plateau/degradation point.
    """
    anova_accuracy = run_three_way_anova(results_df, dependent_var="accuracy", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")
    anova_f1 = run_three_way_anova(results_df, dependent_var="f1_macro", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")
    anova_auc = run_three_way_anova(results_df, dependent_var="roc_auc_ovr", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")

    tukey_accuracy = run_pairwise_posthoc(results_df, dependent_var="accuracy", grouping_col="ratio", alpha=alpha)

    # TODO: implement find_optimal_ratio in stats/statistical_analysis.py, then
    # use its output to set `supported` based on whether a plateau/degradation
    # point is statistically identifiable.
    optimal_ratio_summary = None  # find_optimal_ratio(results_df, ["accuracy", "f1_macro", "roc_auc_ovr"])

    return HypothesisResult(
        hypothesis_id="H1",
        statement=(
            "An optimal synthetic ratio exists beyond which additional synthetic "
            "data yields no significant improvement or causes degradation."
        ),
        supported=None,  # TODO: derive from optimal_ratio_summary once implemented
        evidence={
            "anova_accuracy": anova_accuracy,
            "anova_f1_macro": anova_f1,
            "anova_roc_auc": anova_auc,
            "tukey_accuracy_by_ratio": tukey_accuracy,
            "optimal_ratio_summary": optimal_ratio_summary,
        },
    )


def test_h2_distribution_strategy_effect(results_df: pd.DataFrame, alpha: float = 0.05) -> HypothesisResult:
    """H2: minority_only distribution yields significantly better
    class-balanced performance (macro-F1, per-class recall, balanced
    accuracy) than proportional / fully_balanced strategies.
    """
    anova_macro_f1 = run_three_way_anova(results_df, dependent_var="f1_macro", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")
    anova_balanced_acc = run_three_way_anova(results_df, dependent_var="balanced_accuracy", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")

    tukey_macro_f1 = run_pairwise_posthoc(
        results_df, dependent_var="f1_macro", grouping_col="distribution_strategy", alpha=alpha
    )

    # TODO: additionally unpack `per_class_report` (nested dict, currently
    # stored per-row in evaluation/evaluate.py's results) into long-format
    # per-class recall rows for a dedicated per-class ANOVA/visualization.

    return HypothesisResult(
        hypothesis_id="H2",
        statement=(
            "Minority-only synthetic distribution yields significantly better "
            "class-balanced performance than proportional or fully-balanced strategies."
        ),
        supported=None,  # TODO: derive from tukey_macro_f1 p-values (minority_only vs. others)
        evidence={
            "anova_f1_macro": anova_macro_f1,
            "anova_balanced_accuracy": anova_balanced_acc,
            "tukey_f1_macro_by_distribution": tukey_macro_f1,
        },
    )


def test_h3_architecture_interaction(results_df: pd.DataFrame) -> HypothesisResult:
    """H3: CNNs (ResNet50, EfficientNet-B0) and ViT respond differently to
    augmentation, producing architecture-specific optimal ratios/strategies
    -- evidenced by significant architecture x ratio and/or architecture x
    distribution interaction terms in the ANOVA.
    """
    anova_accuracy = run_three_way_anova(results_df, dependent_var="accuracy", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")

    # TODO: extract specifically the "C(ratio):C(architecture)",
    # "C(distribution_strategy):C(architecture)", and 3-way interaction rows
    # from `anova_accuracy` and test their p-values against alpha.

    return HypothesisResult(
        hypothesis_id="H3",
        statement=(
            "CNNs and ViTs respond differently to synthetic augmentation, "
            "producing architecture-specific optimal ratios/distribution strategies."
        ),
        supported=None,  # TODO: derive from interaction-term significance
        evidence={"anova_accuracy_full_table": anova_accuracy},
    )


def test_h4_explainability_effect(
    results_df: pd.DataFrame, xai_metrics_df: pd.DataFrame
) -> HypothesisResult:
    """H4: synthetic augmentation influences attention patterns; the optimal
    augmentation strategy (per H1/H2) preserves or enhances clinically
    relevant attention localization (higher IoU/Dice + lower CoM distance
    and EMD, relative to baseline/lesion-mask reference).

    `xai_metrics_df` is expected to be the per-cell-aggregated output of
    explainability/quantitative_xai.py::aggregate_quantitative_xai_over_dataset,
    merged with the cell identifiers (ratio, distribution_strategy, architecture, seed).
    """
    merged = results_df.merge(
        xai_metrics_df, on=["ratio", "distribution_strategy", "architecture", "seed"]
    )

    anova_iou = run_three_way_anova(merged, dependent_var="iou", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")
    anova_dice = run_three_way_anova(merged, dependent_var="dice", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")
    anova_com_dist = run_three_way_anova(merged, dependent_var="center_of_mass_distance", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")

    # TODO: cross-reference the "optimal" cell identified by H1/H2 against
    # its explainability metrics vs. the ratio=0.0 baseline's explainability
    # metrics, to directly answer "does the optimal strategy preserve/enhance
    # attention localization" rather than just reporting main/interaction effects.

    return HypothesisResult(
        hypothesis_id="H4",
        statement=(
            "Synthetic augmentation influences attention patterns; the optimal "
            "strategy preserves or enhances clinically relevant attention localization."
        ),
        supported=None,  # TODO: derive from optimal-vs-baseline XAI metric comparison
        evidence={
            "anova_iou": anova_iou,
            "anova_dice": anova_dice,
            "anova_center_of_mass_distance": anova_com_dist,
        },
    )


def run_all_hypothesis_tests(
    results_df: pd.DataFrame,
    xai_metrics_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> Dict[str, HypothesisResult]:
    """Convenience driver running H1-H4 in sequence and returning a dict
    keyed by hypothesis id, the final artifact for the "Experimental
    Findings" stage of the pipeline.
    """
    findings = {
        "H1": test_h1_synthetic_ratio_effect(results_df, alpha=alpha),
        "H2": test_h2_distribution_strategy_effect(results_df, alpha=alpha),
        "H3": test_h3_architecture_interaction(results_df),
    }
    if xai_metrics_df is not None:
        findings["H4"] = test_h4_explainability_effect(results_df, xai_metrics_df)
    return findings
