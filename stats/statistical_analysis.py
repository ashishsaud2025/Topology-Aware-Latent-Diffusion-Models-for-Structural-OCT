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
) -> pd.DataFrame:
    """Fit a three-way factorial ANOVA model and return the ANOVA table
    (main effects + all 2-way + 3-way interactions) for `dependent_var`
    (e.g. "accuracy", "f1_macro", "balanced_accuracy").

    NOTE: `factor_a_col` (ratio) is continuous by nature but should be
    treated as CATEGORICAL for this factorial ANOVA (it has 5 discrete
    levels by design) -- wrapped with C() in the formula below.

    TODO: if experiment/factorial_design.py's `deduplicate_baseline_cells`
    was used, this design is UNBALANCED (ratio=0.0 cells don't vary by
    distribution_strategy). Standard `anova_lm` assumes balance for Type I
    sums of squares -- switch to `typ=2` or `typ=3` (Type II/III SS, robust
    to imbalance) accordingly.
    """
    formula = (
        f"{dependent_var} ~ C({factor_a_col}) * C({factor_b_col}) * C({factor_c_col})"
    )
    model = ols(formula, data=results_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


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


def find_optimal_ratio(
    results_df: pd.DataFrame,
    dependent_vars: List[str],
    ratio_col: str = "ratio",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """H1 support routine: for each metric in `dependent_vars`, compute the
    mean performance per ratio level, identify the ratio with peak
    performance, and use pairwise post-hoc tests to determine the smallest
    ratio that is NOT statistically distinguishable from the peak (i.e. the
    "optimal" ratio beyond which gains are not significant).

    TODO: implement the "no further significant improvement" logic precisely
    -- e.g. walk ratios in ascending order, stop at the first ratio whose
    Tukey-HSD comparison to the peak ratio is non-significant.
    """
    raise NotImplementedError("TODO: implement optimal-ratio identification logic")


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
