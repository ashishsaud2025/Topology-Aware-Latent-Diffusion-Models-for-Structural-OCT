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
from typing import Any, Dict, List, Optional

import pandas as pd

from stats.statistical_analysis import (
    extract_pvalue,
    find_optimal_ratio,
    run_pairwise_posthoc,
    run_three_way_anova,
    tukey_results_to_dataframe,
)


@dataclass
class HypothesisResult:
    hypothesis_id: str
    statement: str
    supported: Optional[bool]  # None until decision logic is available
    evidence: Dict[str, Any]


def _pairwise_comparison(
    results_df: pd.DataFrame, dependent_var: str, grouping_col: str, g1: str, g2: str
) -> Dict[str, Any]:
    """Run Tukey HSD for `dependent_var` across `grouping_col` and return the
    row for the g1-vs-g2 comparison as a plain dict ({} when not found)."""
    pairwise = tukey_results_to_dataframe(
        run_pairwise_posthoc(results_df, dependent_var=dependent_var, grouping_col=grouping_col)
    )
    mask = (
        ((pairwise["group1"] == g1) & (pairwise["group2"] == g2))
        | ((pairwise["group1"] == g2) & (pairwise["group2"] == g1))
    )
    matching = pairwise[mask]
    if len(matching) == 0:
        return {}
    row = matching.iloc[0]
    return {
        "group1": row["group1"],
        "group2": row["group2"],
        "meandiff": float(row["meandiff"]),
        "p_adj": float(row["p-adj"]),
        "reject": bool(row["reject"]),
    }


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

    tukey_accuracy = tukey_results_to_dataframe(
        run_pairwise_posthoc(results_df, dependent_var="accuracy", grouping_col="ratio", alpha=alpha)
    )

    optimal_ratio_summary = find_optimal_ratio(
        results_df, ["accuracy", "f1_macro", "roc_auc_ovr"], alpha=alpha
    )

    # Decision: supported only if ratio is a statistically significant factor
    # (ANOVA p < alpha) AND at least one metric shows an identifiable
    # plateau / degradation point (i.e. an optimal ratio between 0 and 1,
    # or no significant gain beyond the most efficient ratio).
    p_ratio_accuracy = extract_pvalue(anova_accuracy, "C(ratio)")
    p_ratio_f1 = extract_pvalue(anova_f1, "C(ratio)")
    p_ratio_auc = extract_pvalue(anova_auc, "C(ratio)")
    ratio_significant = any(
        pd.notna(p) and p < alpha for p in (p_ratio_accuracy, p_ratio_f1, p_ratio_auc)
    )

    has_optimal_point = False
    for metric, summ in optimal_ratio_summary.items():
        if "error" in summ:
            continue
        # An optimal ratio strictly between the lowest and highest levels, or
        # a plateau of size >= 2 with NO significant degradation beyond the peak.
        if (
            0.0 < summ["peak_ratio"] < 1.0
            or (
                len(summ["plateau_ratios"]) >= 2
                and not summ["degradation_beyond_peak"]
                and not (summ["peak_ratio"] == 0.0 and len(summ["plateau_ratios"]) == 1)
            )
        ):
            has_optimal_point = True
            break

    supported = bool(ratio_significant and has_optimal_point)

    return HypothesisResult(
        hypothesis_id="H1",
        statement=(
            "An optimal synthetic ratio exists beyond which additional synthetic "
            "data yields no significant improvement or causes degradation."
        ),
        supported=supported,
        evidence={
            "anova_accuracy": anova_accuracy,
            "anova_f1_macro": anova_f1,
            "anova_roc_auc": anova_auc,
            "tukey_accuracy_by_ratio": tukey_accuracy,
            "optimal_ratio_summary": {
                k: (
                    {kk: vv for kk, vv in v.items() if kk != "pairwise_table"}
                    if isinstance(v, dict)
                    else v
                )
                for k, v in optimal_ratio_summary.items()
            },
            "decision": {
                "ratio_significant": ratio_significant,
                "p_ratio_accuracy": p_ratio_accuracy,
                "p_ratio_f1_macro": p_ratio_f1,
                "p_ratio_roc_auc": p_ratio_auc,
                "has_optimal_point": has_optimal_point,
                "supported": supported,
                "method": (
                    "3-way ANOVA main effect of ratio (typ=2) AND optimal-ratio "
                    "identification via Tukey-HSD plateau search; supported when "
                    "ratio is significant and a plateau/degradation point is "
                    "identifiable."
                ),
            },
        },
    )


def test_h2_distribution_strategy_effect(results_df: pd.DataFrame, alpha: float = 0.05) -> HypothesisResult:
    """H2: minority_only distribution yields significantly better
    class-balanced performance (macro-F1, per-class recall, balanced
    accuracy) than proportional / fully_balanced strategies.

    Decision: supported only when minority_only's macro-F1 (and balanced
    accuracy) is significantly higher than BOTH proportional and
    fully_balanced in Tukey HSD (p_adj < alpha, positive mean difference).
    The ANOVA strategy main effect is reported as context but is not
    required (an interaction may mask the main effect).
    """
    anova_macro_f1 = run_three_way_anova(results_df, dependent_var="f1_macro", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")
    anova_balanced_acc = run_three_way_anova(results_df, dependent_var="balanced_accuracy", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")

    tukey_f1_table = tukey_results_to_dataframe(
        run_pairwise_posthoc(results_df, dependent_var="f1_macro", grouping_col="distribution_strategy", alpha=alpha)
    )
    tukey_balanced_table = tukey_results_to_dataframe(
        run_pairwise_posthoc(results_df, dependent_var="balanced_accuracy", grouping_col="distribution_strategy", alpha=alpha)
    )

    f1_vs_prop = _pairwise_comparison(
        results_df, "f1_macro", "distribution_strategy", "minority_only", "proportional"
    )
    f1_vs_bal = _pairwise_comparison(
        results_df, "f1_macro", "distribution_strategy", "minority_only", "fully_balanced"
    )
    bal_vs_prop = _pairwise_comparison(
        results_df, "balanced_accuracy", "distribution_strategy", "minority_only", "proportional"
    )
    bal_vs_bal = _pairwise_comparison(
        results_df, "balanced_accuracy", "distribution_strategy", "minority_only", "fully_balanced"
    )

    # Decision: both class-balanced metrics must be significantly better
    # than both alternatives for a decisive SUPPORT; otherwise NOT_SUPPORTED
    # (single-metric evidence listed in the report).
    min_p_adj = max(
        f1_vs_prop.get("p_adj", 1.0),
        f1_vs_bal.get("p_adj", 1.0),
        bal_vs_prop.get("p_adj", 1.0),
        bal_vs_bal.get("p_adj", 1.0),
    )
    all_positive_diffs = all(
        d.get("meandiff", -1.0) > 0
        for d in (f1_vs_prop, f1_vs_bal, bal_vs_prop, bal_vs_bal)
        if d
    )
    supported = bool(min_p_adj < alpha and all_positive_diffs)

    return HypothesisResult(
        hypothesis_id="H2",
        statement=(
            "Minority-only synthetic distribution yields significantly better "
            "class-balanced performance than proportional or fully-balanced strategies."
        ),
        supported=supported,
        evidence={
            "anova_f1_macro": anova_macro_f1,
            "anova_balanced_accuracy": anova_balanced_acc,
            "tukey_f1_macro_by_distribution": tukey_f1_table,
            "tukey_balanced_accuracy_by_distribution": tukey_balanced_table,
            "pairwise_f1_macro_minority_vs_proportional": f1_vs_prop,
            "pairwise_f1_macro_minority_vs_fully_balanced": f1_vs_bal,
            "pairwise_balanced_acc_minority_vs_proportional": bal_vs_prop,
            "pairwise_balanced_acc_minority_vs_fully_balanced": bal_vs_bal,
            "decision": {
                "supported": supported,
                "min_p_adj": min_p_adj,
                "all_positive_mean_diffs": all_positive_diffs,
                "method": (
                    "Both macro-F1 and balanced-accuracy Tukey HSD must show "
                    "minority_only significantly better than both proportional "
                    "and fully_balanced (p_adj < alpha, positive meandiff)."
                ),
            },
        },
    )


def test_h3_architecture_interaction(results_df: pd.DataFrame, alpha: float = 0.05) -> HypothesisResult:
    """H3: CNNs (ResNet50, EfficientNet-B0) and ViT respond differently to
    augmentation, producing architecture-specific optimal ratios/strategies
    -- evidenced by significant architecture x ratio and/or architecture x
    distribution interaction terms in the ANOVA.

    Decision: supported when at least one architecture interaction term
    (C(ratio):C(architecture) or C(strategy):C(architecture)) is significant
    (p < alpha) in the accuracy or f1_macro ANOVA.
    """
    anova_accuracy = run_three_way_anova(results_df, dependent_var="accuracy", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")
    anova_f1 = run_three_way_anova(results_df, dependent_var="f1_macro", factor_a_col="ratio", factor_b_col="distribution_strategy", factor_c_col="architecture")

    interaction_terms = [
        "C(ratio):C(architecture)",
        "C(distribution_strategy):C(architecture)",
    ]
    p_vals = {
        f"accuracy__{term}": extract_pvalue(anova_accuracy, term)
        for term in interaction_terms
    }
    p_vals.update(
        {f"f1_macro__{term}": extract_pvalue(anova_f1, term) for term in interaction_terms}
    )

    significant_terms = {
        k: float(p) for k, p in p_vals.items() if pd.notna(p) and p < alpha
    }

    supported = bool(significant_terms)

    return HypothesisResult(
        hypothesis_id="H3",
        statement=(
            "CNNs and ViTs respond differently to synthetic augmentation, "
            "producing architecture-specific optimal ratios/distribution strategies."
        ),
        supported=supported,
        evidence={
            "anova_accuracy_full_table": anova_accuracy,
            "anova_f1_macro_full_table": anova_f1,
            "interaction_p_values": {k: (None if pd.isna(v) else float(v)) for k, v in p_vals.items()},
            "significant_interaction_terms": {
                k: v for k, v in significant_terms.items()
            },
            "decision": {
                "supported": supported,
                "alpha": alpha,
                "method": (
                    "Supported when C(ratio):C(architecture) or "
                    "C(strategy):C(architecture) is significant (p < alpha) in "
                    "accuracy or f1_macro ANOVA."
                ),
            },
        },
    )


def test_h4_explainability_effect(
    results_df: pd.DataFrame,
    xai_metrics_df: pd.DataFrame,
    alpha: float = 0.05,
    optimal_cell: Optional[Dict[str, Any]] = None,
) -> HypothesisResult:
    """H4: synthetic augmentation influences attention/explanation patterns;
    the optimal augmentation strategy (per H1/H2) preserves or enhances
    clinically relevant attention localization.

    Two-part decision rule:
      (1) ATTENTION CHANGES: the `ratio` main effect is significant (p < alpha)
          for at least one quantitative-XAI metric in a 3-way ANOVA.
      (2) OPTIMAL PRESERVES/ENHANCES: the H1/H2-optimal cell's mean XAI metric
          is not significantly WORSE than the ratio=0.0 baseline (per Tukey HSD
          on the ratio factor), with IoU/Dice >= baseline and CoM/EMD <= baseline.

    `xai_metrics_df` is expected to be the per-cell-aggregated output of
    explainability/quantitative_xai.py::aggregate_quantitative_xai_over_dataset,
    merged with the cell identifiers (ratio, distribution_strategy, architecture, seed).
    `optimal_cell` optionally supplies {ratio, distribution_strategy, architecture}
    for the H1-optimal cell (defaults to the ratio=0.0 baseline only).
    """
    merged = results_df.merge(
        xai_metrics_df, on=["ratio", "distribution_strategy", "architecture", "seed"]
    )

    metric_vars = ["iou", "dice", "center_of_mass_distance", "earth_movers_distance"]
    present_metrics = [m for m in metric_vars if m in merged.columns]

    anovas = {
        m: run_three_way_anova(
            merged,
            dependent_var=m,
            factor_a_col="ratio",
            factor_b_col="distribution_strategy",
            factor_c_col="architecture",
        )
        for m in present_metrics
    }
    p_vals = {
        m: extract_pvalue(anovas[m], "C(ratio)") for m in present_metrics
    }
    attention_changed = any(pd.notna(p) and p < alpha for p in p_vals.values())

    # Optimal-vs-baseline preservation: Tukey HSD on ratio for each metric.
    tukey_ratio = {
        m: tukey_results_to_dataframe(
            run_pairwise_posthoc(merged, dependent_var=m, grouping_col="ratio", alpha=alpha)
        )
        for m in present_metrics
    }

    optimal_ratio = float((optimal_cell or {}).get("ratio", 0.0))
    preservation: Dict[str, Any] = {}
    for m in present_metrics:
        table = tukey_ratio[m]
        optimal_mean = float(merged.loc[merged["ratio"] == optimal_ratio, m].mean())
        baseline_mean = float(merged.loc[merged["ratio"] == 0.0, m].mean())
        sig_vs_baseline = False
        for _, row in table.iterrows():
            g1, g2 = float(row["group1"]), float(row["group2"])
            if {g1, g2} == {optimal_ratio, 0.0} and bool(row["reject"]):
                sig_vs_baseline = True
                break
        # Higher-is-better metrics (IoU, Dice) must not drop below baseline;
        # lower-is-better (CoM distance, EMD) must not rise above baseline.
        if m in ("iou", "dice"):
            preserved = (not sig_vs_baseline) or optimal_mean >= baseline_mean - 1e-9
        else:
            preserved = (not sig_vs_baseline) or optimal_mean <= baseline_mean + 1e-9
        preservation[m] = {
            "optimal_ratio": optimal_ratio,
            "optimal_mean": optimal_mean,
            "baseline_mean": baseline_mean,
            "significant_vs_baseline": sig_vs_baseline,
            "attention_preserved": bool(preserved),
        }

    all_preserved = all(v["attention_preserved"] for v in preservation.values())
    supported = bool(attention_changed and all_preserved)

    return HypothesisResult(
        hypothesis_id="H4",
        statement=(
            "Synthetic augmentation influences attention patterns; the optimal "
            "strategy preserves or enhances clinically relevant attention localization."
        ),
        supported=supported,
        evidence={
            "anova_iou": anovas.get("iou"),
            "anova_dice": anovas.get("dice"),
            "anova_center_of_mass_distance": anovas.get("center_of_mass_distance"),
            "anova_earth_movers_distance": anovas.get("earth_movers_distance"),
            "p_ratio_by_metric": {
                m: (None if pd.isna(p) else float(p)) for m, p in p_vals.items()
            },
            "tukey_ratio_by_metric": {
                m: table for m, table in tukey_ratio.items()
            },
            "preservation": preservation,
            "decision": {
                "supported": supported,
                "alpha": alpha,
                "attention_changed": attention_changed,
                "all_metrics_preserved": all_preserved,
                "method": (
                    "Supported when (1) the ratio main effect is significant "
                    "(p < alpha) for >=1 quantitative-XAI metric in a 3-way ANOVA, "
                    "AND (2) the optimal cell's mean XAI metric is not "
                    "significantly worse than the ratio=0.0 baseline (Tukey HSD, "
                    "IoU/Dice >= baseline, CoM/EMD <= baseline)."
                ),
            },
        },
    )


def run_all_hypothesis_tests(
    results_df: pd.DataFrame,
    xai_metrics_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
    optimal_cell: Optional[Dict[str, Any]] = None,
) -> Dict[str, HypothesisResult]:
    """Convenience driver running H1-H4 in sequence and returning a dict
    keyed by hypothesis id, the final artifact for the "Experimental
    Findings" stage of the pipeline.
    """
    findings = {
        "H1": test_h1_synthetic_ratio_effect(results_df, alpha=alpha),
        "H2": test_h2_distribution_strategy_effect(results_df, alpha=alpha),
        "H3": test_h3_architecture_interaction(results_df, alpha=alpha),
    }
    if xai_metrics_df is not None:
        findings["H4"] = test_h4_explainability_effect(
            results_df, xai_metrics_df, alpha=alpha, optimal_cell=optimal_cell
        )
    return findings