"""Statistical Analysis stage: main effects, interaction effects, and
significance tests over the master results table produced by
evaluation/evaluate.py (+ explainability/quantitative_xai.py for H4).

Implements a three-way factorial ANOVA (Factor A: ratio, Factor B:
distribution strategy, Factor C: architecture) with post-hoc pairwise tests,
following standard factorial-design practice (main effects for each factor,
all 2-way interactions, and the 3-way interaction).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def run_three_way_anova(
    results_df: pd.DataFrame,
    dependent_var: str,
    factor_a_col: str = "ratio",
    factor_b_col: str = "distribution_strategy",
    factor_c_col: str = "architecture",
    include_three_way: bool = False,
) -> pd.DataFrame:
    """Fit a three-way factorial ANOVA model and return the ANOVA table for
    `dependent_var` (e.g. "accuracy", "f1_macro", "balanced_accuracy").

    Model specification
    -------------------
    By default (`include_three_way=False`) the model contains the three main
    effects plus all three 2-way interactions:

        y ~ C(ratio) + C(strategy) + C(architecture)
          + C(ratio):C(strategy) + C(ratio):C(architecture)
          + C(strategy):C(architecture)

    This is the standard model for UNREPLICATED factorial designs (the
    current grid: 1 seed per cell, 39 unique cells after
    `deduplicate_baseline_cells`). The full saturated model

        y ~ C(ratio) * C(strategy) * C(architecture)

    would require 1 + 4 + 2 + 2 + 8 + 8 + 4 + 16 = 45 estimable parameters
    but only has 39 observations, giving df_resid = 0, which makes
    statsmodels produce divide-by-zero / NaN covariance errors. In such
    designs the 3-way interaction is conventionally pooled into the residual
    error term (Montgomery, "Design and Analysis of Experiments"). The main
    effects and 2-way interactions -- exactly what H1, H2, and H3 require --
    remain estimable.

    Pass `include_three_way=True` to explicitly fit the saturated model on a
    replicated design (n_seeds_per_cell > 1, which supplies residual df).

    NOTE: `factor_a_col` (ratio) is continuous by nature but treated as
    CATEGORICAL here (5 discrete levels by design) -- wrapped in C(). Type II
    sums of squares (typ=2) are used, which are robust to the unbalanced
    ratio=0.0 cells introduced by `deduplicate_baseline_cells`.

    If the 2-way model is still rank-deficient (or produces non-finite
    values), the function transparently falls back to a main-effects-only
    model so the analysis pipeline can always produce a valid ANOVA table.
    """
    fa, fb, fc = factor_a_col, factor_b_col, factor_c_col

    if include_three_way:
        formulas = [
            f"{dependent_var} ~ C({fa}) * C({fb}) * C({fc})",
        ]
    else:
        formulas = [
            f"{dependent_var} ~ C({fa}) + C({fb}) + C({fc})"
            f" + C({fa}):C({fb}) + C({fa}):C({fc}) + C({fb}):C({fc})",
            # Fallback if the 2-way model is rank-deficient / non-finite:
            f"{dependent_var} ~ C({fa}) + C({fb}) + C({fc})",
        ]

    last_error: Exception | None = None
    for formula in formulas:
        try:
            model = ols(formula, data=results_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            # NOTE: statsmodels always leaves the Residual row's F / PR(>F)
            # columns NaN (there is no comparison group for the residuals),
            # which is expected and valid. Only the factor rows must be finite.
            factor_rows = anova_table.drop(index="Residual", errors="ignore")
            numeric = factor_rows.select_dtypes(include=["number"])
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError("ANOVA table contains non-finite values")
            return anova_table
        except Exception as exc:  # noqa: BLE001 - try next simpler spec
            last_error = exc
            continue

    raise RuntimeError(
        f"Three-way ANOVA failed for '{dependent_var}' with all model "
        f"specifications. Last error: {last_error}"
    )


def run_pairwise_posthoc(
    results_df: pd.DataFrame,
    dependent_var: str,
    grouping_col: str,
    alpha: float = 0.05,
) -> Any:
    """Tukey's HSD post-hoc pairwise comparison across the levels of a single
    factor (e.g. compare all 5 ratio levels pairwise on f1_macro), used to
    identify the "optimal ratio, beyond which no significant improvement"
    referenced in H1.
    """
    tukey_result = pairwise_tukeyhsd(
        endog=results_df[dependent_var],
        groups=results_df[grouping_col],
        alpha=alpha,
    )
    return tukey_result


def tukey_results_to_dataframe(tukey_result) -> pd.DataFrame:
    """Convert a statsmodels TukeyHSDResults object into a tidy DataFrame
    with columns [group1, group2, meandiff, p-adj, lower, upper, reject].

    Uses the stable ``_results_frame`` property when available and falls back
    to reconstructing the pair table from the raw result attributes.
    """
    frame = getattr(tukey_result, "_results_frame", None)
    if frame is not None and len(frame) > 0:
        out = frame.copy()
        for col in ("meandiff", "p-adj", "lower", "upper"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out["reject"] = out["reject"].astype(bool)
        return out

    # Fallback: rebuild from scalar result attributes.
    import itertools

    groups = list(tukey_result.groupsunique)
    confint = tukey_result.confint
    rows = []
    for (i, j), md, p_val, ci, rj in zip(
        itertools.combinations(range(len(groups)), 2),
        tukey_result.meandiffs,
        tukey_result.pvalues,
        confint,
        tukey_result.reject,
    ):
        rows.append(
            {
                "group1": groups[i],
                "group2": groups[j],
                "meandiff": float(md),
                "p-adj": float(p_val),
                "lower": float(ci[0]),
                "upper": float(ci[1]),
                "reject": bool(rj),
            }
        )
    return pd.DataFrame(rows)


def extract_pvalue(anova_table: pd.DataFrame, term: str) -> float:
    """Return the PR(>F) p-value for the ANOVA-table row named `term`
    (e.g. "C(ratio)", "C(ratio):C(architecture)"), or NaN when the term is
    absent from the table."""
    if term in anova_table.index and "PR(>F)" in anova_table.columns:
        return float(anova_table.loc[term, "PR(>F)"])
    return float("nan")


def find_optimal_ratio(
    results_df: pd.DataFrame,
    dependent_vars: List[str],
    ratio_col: str = "ratio",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """H1 support routine: for each metric in `dependent_vars`, compute the
    mean performance per ratio level, identify the ratio with peak
    performance, and use Tukey-HSD pairwise tests to determine the SMALLEST
    ratio that is NOT statistically distinguishable from the peak (i.e. the
    plateau start / "optimal" ratio beyond which gains are not significant).

    Returns {metric: {...}} with the peak ratio, the optimal (most
    efficient) ratio, the full plateau, whether any ratio beyond the peak is
    significantly degraded, the per-ratio means, and the pairwise table.
    """
    summary: Dict[str, Any] = {}
    for metric in dependent_vars:
        if metric not in results_df.columns:
            summary[metric] = {"error": f"column '{metric}' not found in results"}
            continue

        means = results_df.groupby(ratio_col)[metric].mean().sort_index()
        if len(means) < 3:
            summary[metric] = {
                "error": f"need >=3 ratio levels for Tukey HSD, got {len(means)}"
            }
            continue

        peak_ratio = float(means.idxmax())
        pairwise = tukey_results_to_dataframe(
            run_pairwise_posthoc(
                results_df, dependent_var=metric, grouping_col=ratio_col, alpha=alpha
            )
        )

        plateau = {peak_ratio}
        degradation_beyond_peak = False
        for _, row in pairwise.iterrows():
            if row["group1"] == peak_ratio:
                other = float(row["group2"])
            elif row["group2"] == peak_ratio:
                other = float(row["group1"])
            else:
                continue
            if other == peak_ratio or pd.isna(row["p-adj"]):
                continue
            if not row["reject"]:
                plateau.add(other)
            elif other > peak_ratio:
                degradation_beyond_peak = True

        optimal_ratio = float(min(plateau))
        summary[metric] = {
            "peak_ratio": peak_ratio,
            "peak_mean": float(means.loc[peak_ratio]),
            "optimal_ratio": optimal_ratio,
            "optimal_ratio_mean": float(means.loc[optimal_ratio]),
            "plateau_ratios": sorted(float(r) for r in plateau),
            "degradation_beyond_peak": degradation_beyond_peak,
            "ratio_means": {str(k): float(v) for k, v in means.items()},
            "pairwise_table": pairwise,
        }
    return summary


def summarize_main_and_interaction_effects(
    results_df: pd.DataFrame, dependent_vars: List[str], cfg: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Run `run_three_way_anova` for every metric in `dependent_vars` and
    return a dict {metric_name: anova_table}, the primary artifact consumed
    by hypotheses/hypothesis_tests.py.
    """
    return {
        metric: run_three_way_anova(results_df, dependent_var=metric)
        for metric in dependent_vars
    }
