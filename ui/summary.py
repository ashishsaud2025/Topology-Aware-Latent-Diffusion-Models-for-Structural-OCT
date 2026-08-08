"""
summary.py -- loads and renders the project's research-results artifacts
(ANOVA tables, hypothesis findings, XAI metrics, topology report) as clean
tables/figures for the app's Summary tab.

Every loader is tolerant of a missing file (returns None / an empty
placeholder + a note) rather than raising, since a results folder may only
have some of these artifacts at any given point in the pipeline.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find(folder: str, name: str) -> str | None:
    """Recursively searches every subfolder of `folder` for a file named
    exactly `name`. Returns the most recently modified match's full path,
    or None if not found anywhere. Searching subfolders (not just the top
    level) matters because pipelines often write results into subfolders
    like outputs/anova/, outputs/xai/, outputs/topology/ rather than one
    flat directory."""
    matches = glob.glob(os.path.join(folder, "**", name), recursive=True)
    if not matches:
        return None
    if len(matches) > 1:
        # Multiple copies somewhere under the tree (e.g. an old run left
        # behind) -- take the newest one rather than an arbitrary one.
        matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _exists(folder: str, name: str) -> bool:
    return _find(folder, name) is not None


# hypotheses

def load_hypothesis_findings(folder: str) -> dict | None:
    """Prefers h4_findings.json (has the complete, corrected H4 section);
    falls back to hypothesis_findings.json (older/partial, no H4) if
    that's all that's present."""
    for name in ["h4_findings.json", "hypothesis_findings.json"]:
        if _exists(folder, name):
            with open(_find(folder, name)) as f:
                return json.load(f)
    return None


def render_hypothesis_html(findings: dict | None) -> str:
    if not findings:
        return "<p><i>No hypothesis findings file found in this folder "
        "(expected h4_findings.json or hypothesis_findings.json).</i></p>"

    order = ["H1", "H2", "H3", "H4"]
    cards = []
    for key in order:
        if key not in findings:
            cards.append(
                f"<div style='padding:10px;margin:6px 0;border-radius:8px;"
                f"background:#f2f2f2;color:#888;'><b>{key}</b>: not present "
                f"in this findings file.</div>"
            )
            continue
        h = findings[key]
        supported = h.get("supported", False)
        color = "#000000" if supported else "#000000"
        badge_color = "#1a7f37" if supported else "#c0362c"
        badge_text = "SUPPORTED" if supported else "NOT SUPPORTED"
        statement = h.get("statement", "")
        decision = h.get("decision", {})
        # pull out a couple of the most informative numbers if present
        detail_bits = []
        for k in ["p_ratio_accuracy", "p_ratio_f1_macro", "p_ratio_roc_auc", "min_p_adj"]:
            if k in decision:
                detail_bits.append(f"{k}={decision[k]:.2e}")
        detail = " | ".join(detail_bits)
        cards.append(
            f"<div style='padding:12px 14px;margin:8px 0;border-radius:10px;"
            f"background:{color};'>"
            f"<span style='display:inline-block;padding:2px 10px;border-radius:12px;"
            f"background:{badge_color};color:white;font-size:12px;font-weight:600;'>"
            f"{badge_text}</span>"
            f"<b style='margin-left:8px;'>{key}</b>"
            f"<div style='margin-top:6px;'>{statement}</div>"
            + (f"<div style='margin-top:4px;font-size:12px;color:#555;'>{detail}</div>" if detail else "")
            + "</div>"
        )
    return "<div>" + "".join(cards) + "</div>"


# ANOVA

ANOVA_FILES = {
    "Accuracy": "anova_accuracy.csv",
    "Balanced Accuracy": "anova_balanced_accuracy.csv",
    "F1 (macro)": "anova_f1_macro.csv",
    "ROC-AUC (OvR)": "anova_roc_auc_ovr.csv",
}


def load_anova_tables(folder: str) -> dict[str, pd.DataFrame]:
    out = {}
    for label, fname in ANOVA_FILES.items():
        path = _find(folder, fname)
        if path:
            out[label] = pd.read_csv(path)
    return out


# master results

def plot_marginal_means(folder: str):
    """3-panel figure: mean accuracy/F1/ROC-AUC by ratio, by architecture,
    by distribution strategy. Returns a matplotlib Figure, or None if
    master_results.csv isn't present."""
    master_path = _find(folder, "master_results.csv")
    if master_path is None:
        return None
    df = pd.read_csv(master_path)
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"]
    metrics = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for m in metrics:
        by_ratio = df.groupby("ratio")[m].mean()
        axes[0].plot(by_ratio.index, by_ratio.values, marker="o", label=m)
    axes[0].set_xlabel("synthetic ratio")
    axes[0].set_ylabel("mean score")
    axes[0].set_title("By synthetic ratio")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    if "architecture" in df.columns:
        x = np.arange(len(metrics))
        width = 0.25
        archs = sorted(df["architecture"].unique())
        for i, arch in enumerate(archs):
            means = [df[df.architecture == arch][m].mean() for m in metrics]
            axes[1].bar(x + i * width, means, width, label=arch)
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(metrics, rotation=20, ha="right", fontsize=8)
        axes[1].set_title("By architecture")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, axis="y")

    if "distribution_strategy" in df.columns:
        x = np.arange(len(metrics))
        width = 0.25
        strategies = sorted(df["distribution_strategy"].unique())
        for i, strat in enumerate(strategies):
            means = [df[df.distribution_strategy == strat][m].mean() for m in metrics]
            axes[2].bar(x + i * width, means, width, label=strat)
        axes[2].set_xticks(x + width)
        axes[2].set_xticklabels(metrics, rotation=20, ha="right", fontsize=8)
        axes[2].set_title("By distribution strategy")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# XAI

def plot_xai_heatmap_grid(folder: str):
    """Recreates the ratio x architecture XAI-metric grid. Prefers
    xai_metrics_summary.csv (has distribution_strategy, averaged over it
    here for a compact grid); falls back to xai_confusion_matrix.csv,
    which -- despite its name -- is the SAME kind of IoU/Dice/CoM/EMD
    data, not an actual class-prediction confusion matrix, filtered to
    the proportional strategy only. Returns None if neither is present."""
    src = None
    for fname in ["xai_metrics_summary.csv", "xai_confusion_matrix.csv"]:
        path = _find(folder, fname)
        if path:
            src = path
            break
    if src is None:
        return None

    df = pd.read_csv(src)
    metrics = ["iou", "dice", "center_of_mass_distance", "earth_movers_distance"]
    titles = ["IoU (higher=better)", "Dice (higher=better)",
              "Center-of-mass dist (lower=better)", "Earth mover's dist (lower=better)"]
    archs = sorted(df["architecture"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        pivot = df.pivot_table(index="ratio", columns="architecture", values=metric, aggfunc="mean")
        pivot = pivot[[a for a in archs if a in pivot.columns]]
        cmap = "RdYlGn" if i < 2 else "RdYlGn_r"
        im = axes[i].imshow(pivot.values, cmap=cmap, aspect="auto")
        axes[i].set_xticks(range(len(pivot.columns)))
        axes[i].set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
        axes[i].set_yticks(range(len(pivot.index)))
        axes[i].set_yticklabels([f"{r:.2f}" for r in pivot.index])
        axes[i].set_ylabel("ratio")
        axes[i].set_title(title, fontsize=10)
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    axes[i].text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.suptitle("XAI metrics vs (ratio, architecture) -- averaged over distribution strategy", y=1.03)
    plt.tight_layout()
    return fig


# topology

def load_topology_report(folder: str) -> tuple[pd.DataFrame | None, str]:
    topo_path = _find(folder, "topology_report.csv")
    if topo_path is None:
        return None, "topology_report.csv not found."
    df = pd.read_csv(topo_path)
    n_pass = int(df["topology_passed"].sum()) if "topology_passed" in df.columns else None
    n_total = len(df)
    note = f"{n_pass}/{n_total} boundary checks passed" if n_pass is not None else f"{n_total} rows"
    return df, note
