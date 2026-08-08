"""Visualize H4 explainability results.

Produces three artifacts under ``outputs/xai/``:

1. ``heatmap_comparison_resnet50.png`` -- side-by-side comparison for the
   ResNet-50 ratio=0.0 baseline: original OCT B-scan | Grad-CAM | independent
   anatomical reference (ONL+RPE) | difference (Grad-CAM - reference).

2. ``xai_confusion_matrix.png`` -- a 4x3 confusion-style grid
   (rows: synthetic ratio 0.25/0.50/0.75/1.00, columns: architecture) where
   each cell is subdivided into four horizontal bands holding the quantitative
   XAI metrics (IoU, Dice, CoM, EMD). Upper x-axis labels the metric bands,
   lower x-axis labels the models, and every band displays its actual value.

3. ``xai_confusion_matrix.csv`` -- the numeric values of the same grid.

Both are built from the saved H4 artifacts (no model re-inference).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.seed import load_config

# Constants
OUT_DIR = PROJECT_ROOT / "outputs" / "xai"
HEATMAP_DIR = OUT_DIR / "heatmaps"

METRICS = ["iou", "dice", "center_of_mass_distance", "earth_movers_distance"]
METRIC_LABELS = ["IoU (\u2191 good)", "Dice (\u2191 good)", "CoM px (\u2193 good)", "EMD (\u2193 good)"]
RATIOS = [0.25, 0.50, 0.75, 1.00]
ARCHS = ["efficientnet_b0", "resnet50", "vit_base"]
ARCH_LABELS = ["EfficientNet-B0", "ResNet-50", "ViT-Base/16"]
STRATEGY = "proportional"


def _norm01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2-D map to [0, 1] for display."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _load_source_images(cfg: dict, indices) -> np.ndarray:
    """Load the raw OCT B-scans for the given dataset indices as [0,1] tensors."""
    from scripts.run_h4_explainability import build_fixed_test_loader

    loader = build_fixed_test_loader(cfg, batch_size=32)
    ds = loader.dataset
    imgs = []
    for i in indices:
        t, _, _ = ds[int(i)]
        gray = t[0].numpy()
        imgs.append(_norm01(gray))
    return np.stack(imgs, axis=0)


# 1. Side-by-side comparison (ResNet-50 ratio=0.0 baseline)
def make_side_by_side(cfg: dict, n_samples: int = 6) -> None:
    npz_path = HEATMAP_DIR / "ratio0.00_proportional_resnet50_seed0.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}; run run_h4_explainability.py first")
    d = np.load(npz_path)
    grad_cam = np.asarray(d["model_heatmaps"])           # (N,224,224)
    reference = np.asarray(d["anatomical_reference"])    # (N,224,224)
    indices = np.asarray(d["indices"]).tolist()

    n = min(n_samples, grad_cam.shape[0])
    samples = list(range(n))
    idxs = [indices[s] for s in samples]

    src = _load_source_images(cfg, idxs)

    fig, axes = plt.subplots(n, 4, figsize=(15, 3.4 * n))
    if n == 1:
        axes = axes[None, :]

    for r in range(n):
        gcam = _norm01(grad_cam[samples[r]])
        diff = gcam - reference[samples[r]]  # signed: + model-only, - anatomy-missed

        axes[r, 0].imshow(src[r], cmap="gray")
        axes[r, 1].imshow(gcam, cmap="jet")
        axes[r, 2].imshow(reference[samples[r]], cmap="Greens", vmin=0.0, vmax=1.0)
        im = axes[r, 3].imshow(diff, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"sample {idxs[r]}", fontsize=11)

    cols = ["OCT B-scan", "Grad-CAM (ratio=0.0)", "Anatomical reference (ONL+RPE)", "Grad-CAM \u2212 reference"]
    for c, label in enumerate(cols):
        axes[0, c].set_title(label, fontsize=12, fontweight="bold")

    cb = fig.colorbar(im, ax=axes[:, 3], fraction=0.03, pad=0.02)
    cb.set_label("Grad-CAM \u2212 anatomical reference (signed)", fontsize=10)

    out = OUT_DIR / "heatmap_comparison_resnet50.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# 2. Confusion-matrix-style grid (4 ratios x 3 architectures)
def _load_matrix_vals(csv_path: Path) -> np.ndarray:
    """Return (4, 3, 4) array indexed [ratio_idx, arch_idx, metric_idx]."""
    df = pd.read_csv(csv_path)
    df = df[(df["distribution_strategy"] == STRATEGY)]
    vals = np.zeros((len(RATIOS), len(ARCHS), len(METRICS)))
    for ri, ratio in enumerate(RATIOS):
        for ai, arch in enumerate(ARCHS):
            row = df[(df["ratio"] - ratio).abs() < 1e-9]
            row = row[row["architecture"] == arch]
            if row.empty:
                raise ValueError(f"missing cell ratio={ratio} arch={arch} strategy={STRATEGY}")
            for mi, m in enumerate(METRICS):
                v = float(row.iloc[0][m])
                if not np.isfinite(v):
                    raise ValueError(f"non-finite {m} for ratio={ratio} arch={arch}")
                vals[ri, ai, mi] = v
    return vals


def make_confusion_matrix(csv_path: Path) -> None:
    vals = _load_matrix_vals(csv_path)  # (4,3,4)

    # Per-metric normalization across all 12 cells.
    norms = [Normalize(vmin=float(vals[:, :, k].min()), vmax=float(vals[:, :, k].max()))
             for k in range(4)]
    cmaps = [plt.get_cmap("viridis"), plt.get_cmap("viridis"),
             plt.get_cmap("plasma"), plt.get_cmap("plasma")]

    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(
        6, 5,
        height_ratios=[0.25, 1, 1, 1, 1, 0.25],
        width_ratios=[0.20, 1, 1, 1, 0.16],
        hspace=0.16, wspace=0.06,
    )

    # Top label strip: metric band labels 
    top_ax = fig.add_subplot(gs[0, 1:4])
    top_ax.axis("off")
    top_ax.set_xlim(0, 3)
    top_ax.set_ylim(0, 4)
    for k in range(4):
        rect = Rectangle((0, k), 3, 1, facecolor=cmaps[k](0.55),
                         edgecolor="white", lw=1.0)
        top_ax.add_patch(rect)
        top_ax.text(1.5, k + 0.5, METRIC_LABELS[k], ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
    top_ax.set_title("Quantitative XAI metric bands", fontsize=11, loc="left", pad=6)

    # Per-metric colorbars (right margin) 
    cb_gs = mgs.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1:5, 4], hspace=0.55)
    for k in range(4):
        cax = fig.add_subplot(cb_gs[k])
        sm = ScalarMappable(norm=norms[k], cmap=cmaps[k])
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="vertical")
        cax.set_title(METRIC_LABELS[k].split(" ")[0], fontsize=9, pad=2)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7)

    # Cells 
    cell_axes = [[fig.add_subplot(gs[r + 1, c + 1]) for c in range(3)] for r in range(4)]
    for ri in range(4):
        for ai in range(3):
            ax = cell_axes[ri][ai]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("white")
                spine.set_linewidth(1.2)
            for k in range(4):
                v = float(vals[ri, ai, k])
                color = cmaps[k](norms[k](v))
                ax.add_patch(Rectangle((0, k), 1, 1, facecolor=color,
                                       edgecolor="white", lw=0.8))
                r_, g_, b_, _ = color
                lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                txt_color = "white" if lum < 0.55 else "black"
                label = f"{v:.3f}" if k < 2 else f"{v:.1f}"
                ax.text(0.5, k + 0.5, label, ha="center", va="center",
                        fontsize=10.5, color=txt_color, fontweight="bold")

    # Row labels: synthetic ratio 
    for ri in range(4):
        yax = fig.add_subplot(gs[ri + 1, 0])
        yax.axis("off")
        yax.text(0.5, 0.5, f"{RATIOS[ri]:.2f}", ha="center", va="center",
                 fontsize=12, fontweight="bold")
    fig.text(0.028, 0.5, "Synthetic ratio", rotation=90, va="center", ha="center",
             fontsize=12, fontweight="bold")

    # Bottom label strip: architecture names 
    bot_ax = fig.add_subplot(gs[5, 1:4])
    bot_ax.axis("off")
    bot_ax.set_xlim(0, 3)
    bot_ax.set_ylim(0, 1)
    for ai in range(3):
        rect = Rectangle((ai, 0), 1, 1, facecolor="#d9d9d9",
                         edgecolor="white", lw=1.0)
        bot_ax.add_patch(rect)
        bot_ax.text(ai + 0.5, 0.5, ARCH_LABELS[ai], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="black")
    bot_ax.set_xlabel("Architecture", fontsize=12, labelpad=4)

    fig.suptitle(
        f"Quantitative Explainability (H4) \u2014 ratio \u00d7 architecture grid\n"
        f"strategy = {STRATEGY}; values = mean over 120 test images vs independent "
        "anatomical (ONL+RPE) reference",
        fontsize=12.5, fontweight="bold",
    )

    out = OUT_DIR / "xai_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # CSV companion
    records = []
    for ri in range(4):
        for ai in range(3):
            records.append({
                "ratio": RATIOS[ri],
                "architecture": ARCHS[ai],
                **{METRICS[k]: round(float(vals[ri, ai, k]), 6) for k in range(4)},
            })
    csv_out = OUT_DIR / "xai_confusion_matrix.csv"
    pd.DataFrame(records).to_csv(csv_out, index=False)
    print(f"saved {csv_out}")


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "config_full_factorial.yaml"))
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    make_side_by_side(cfg, n_samples=6)
    make_confusion_matrix(OUT_DIR / "xai_metrics.csv")


if __name__ == "__main__":
    main()