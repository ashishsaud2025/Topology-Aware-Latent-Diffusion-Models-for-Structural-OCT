"""H4 explainability driver: Grad-CAM + Attention Rollout + quantitative XAI.

Every cell's heatmaps (including the ratio=0.0 baseline) are scored against an
INDEPENDENT anatomical reference -- the outer-retina (ONL + RPE) ROI derived
per-image from the model-free intensity-profile layer segmenter
(topology/layer_segmentation.py). No cell is ever compared to its own
attention, so the quantitative metrics and the H4 preservation test are
non-circular.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import OCTImageDataset, get_default_transforms
from explainability.attention_rollout import batch_attention_rollout_heatmaps
from explainability.gradcam import compute_batch_gradcam_heatmaps
from explainability.quantitative_xai import evaluate_heatmap_pair
from experiment.factorial_design import (
    build_factorial_grid,
    deduplicate_baseline_cells,
)
from hypotheses.hypothesis_tests import run_all_hypothesis_tests
from models.classifiers import build_classifier
from stats.statistical_analysis import find_optimal_ratio
from topology.layer_segmentation import RetinalLayerSegmenter
from utils.logging_utils import cell_run_id, get_logger
from utils.seed import load_config, set_global_seed

logger = get_logger("run_h4_explainability")

# Clinically relevant outer-retina layers for DME pathology (ONL=7, RPE=8).
# A good model should focus its attention here; used as the INDEPENDENT
# anatomical reference ROI, so no model is ever scored against itself.
OUTER_RETINA_LAYERS = (7, 8)


def build_fixed_test_loader(cfg: dict, batch_size: int):
    split_dir = Path(cfg["data"]["processed_dir"]) / "splits"
    test_csv = split_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test split missing: {test_csv}")
    test_df = pd.read_csv(test_csv)
    class_to_idx = {c: i for i, c in enumerate(cfg["data"]["classes"])}
    transform = get_default_transforms(cfg["data"]["image_size"], train=False)
    dataset = OCTImageDataset(
        test_df, class_to_idx, transform, image_size=cfg["data"]["image_size"]
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=cfg["training"]["num_workers"], pin_memory=True, drop_last=False,
    )


def select_test_indices(loader, max_samples: int) -> List[int]:
    if max_samples is None or max_samples <= 0:
        return list(range(len(loader.dataset)))
    df = loader.dataset.df
    if "label" not in df.columns:
        return list(range(min(max_samples, len(df))))
    pooled = []
    rng = np.random.default_rng(42)
    for label in df["label"].unique():
        class_idx = list(df.index[df["label"] == label].values)
        n = max(1, round(max_samples * len(class_idx) / len(df)))
        picked = rng.choice(class_idx, size=min(n, len(class_idx)), replace=False)
        pooled.extend(int(i) for i in picked)
    return sorted(pooled)


def anatomically_mask(tensor_img: torch.Tensor, target_size=(224, 224)) -> np.ndarray:
    """Derive the outer-retina (ONL + RPE) binary ROI for one OCT B-scan.

    Uses the dependency-free intensity-profile segmenter
    (topology/layer_segmentation.py) which needs no external annotations,
    making the reference fully independent of every classifier checkpoint.

    Args:
        tensor_img: (3, H, W) RGB replicate of a z-score normalized B-scan.

    Returns:
        (224, 224) float32 binary ROI (1 = outer-retina pathology zone).
    """
    try:
        gray = tensor_img[0].numpy()  # channel replicate = grayscale
        segmenter = RetinalLayerSegmenter(backend="profile")
        seg = segmenter.segment(gray)
        label_map = np.asarray(seg.label_map, dtype=np.uint8)
        roi = np.isin(label_map, OUTER_RETINA_LAYERS).astype(np.float32)
        if roi.sum() == 0:  # degenerate segmentation; fall back to full image
            roi = np.ones_like(roi, dtype=np.float32)
            logger.warning("Segmenter returned empty outer-retina ROI; using full image mask.")
        t = torch.from_numpy(roi).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
        return t.squeeze(0).squeeze(0).numpy()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Segmentation failed ({exc}); using full-image ROI.")
        return np.ones(target_size, dtype=np.float32)


def build_anatomical_reference(loader, indices: List[int]) -> np.ndarray:
    """Independent per-image anatomical reference ROI (N, 224, 224).

    Computed ONCE from the raw test images and used as the ground-truth
    comparison target for every cell, so ratio=0.0 is also scored against
    anatomy -- never against its own attention.
    """
    ds = loader.dataset
    refs = []
    for idx in indices:
        tensor_img, _, _ = ds[idx]
        refs.append(anatomically_mask(tensor_img))
    refs = np.stack(refs, axis=0)
    logger.info(f"Anatomical reference ROIs (outer retina): {refs.shape}")
    return refs


def heatmaps_for_samples(
    model, arch, loader, indices, device, target_size=(224, 224)
) -> np.ndarray:
    """Compute saliency heatmaps for the given samples, normalized to a common
    spatial resolution.

    ViT attention-rollout maps are produced at patch resolution (14x14 for a
    224x224 input), while CNN Grad-CAM maps are at full spatial resolution.
    To make quantitative metrics (IoU/Dice/CoM/EMD) comparable across
    architectures, all maps are bilinearly upsampled to `target_size` before
    being returned.
    """
    model.to(device).eval()
    ds = loader.dataset
    chunk = loader.batch_size
    out = []
    for s in range(0, len(indices), chunk):
        idxs = indices[s : s + chunk]
        batch = torch.stack([ds[i][0] for i in idxs]).to(device)
        if model.is_transformer:
            maps = batch_attention_rollout_heatmaps(model, batch)
        else:
            maps = compute_batch_gradcam_heatmaps(model, arch, batch)
        out.append(np.asarray(maps))
        del batch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    maps = np.concatenate(out, axis=0)  # (N, H, W)
    if tuple(maps.shape[1:]) != tuple(target_size):
        t = torch.from_numpy(maps).float().unsqueeze(1)  # (N, 1, H, W)
        t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
        maps = t.squeeze(1).numpy()  # (N, 224, 224)
    return maps


def parse_csv(value: Optional[str], cast=None) -> Optional[List[Any]]:
    if not value:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [cast(v) for v in items] if cast else items


def main(config_path, ratios, strategies, architectures, max_test_samples, compute_emd, heatmap_threshold):
    cfg = load_config(config_path)
    set_global_seed(cfg["project"]["seed"])
    device = cfg["project"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("cuda unavailable; falling back to cpu.")
        device = "cpu"

    output_dir = Path(cfg["project"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    xai_dir = output_dir / "xai"
    heatmap_dir = xai_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    all_cells = deduplicate_baseline_cells(build_factorial_grid(cfg))
    cells = [
        c for c in all_cells
        if (ratios is None or c.ratio in ratios)
        and (strategies is None or c.distribution_strategy in strategies)
        and (architectures is None or c.architecture in architectures)
    ]
    if not cells:
        raise ValueError("No cells match the requested filters.")
    logger.info(f"Processing {len(cells)} cell(s) for H4.")

    loader = build_fixed_test_loader(cfg, batch_size=cfg["training"]["batch_size"])
    indices = select_test_indices(loader, max_test_samples)
    logger.info(f"Test set {len(loader.dataset)}; explaining {len(indices)} images.")

    # Independent anatomical reference (outer-retina ROI) -- computed once,
    # identical for every cell. No model is ever compared against itself.
    anatomical_ref = build_anatomical_reference(loader, indices)

    results_csv = output_dir / "master_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"missing {results_csv}")
    results_df = pd.read_csv(results_csv)
    alpha = cfg.get("statistics", {}).get("alpha", 0.05)

    # Search for the optimal ratio AMONG AUGMENTED CELLS ONLY (ratio greater
    # than 0). Including the ratio=0.0 baseline in the search would make H4's
    # preservation test circular (the baseline compared against itself).
    min_augmented_ratio = min(c.ratio for c in all_cells if c.ratio > 0.0)
    aug_results = results_df[results_df["ratio"] >= min_augmented_ratio - 1e-9].copy()
    opt = find_optimal_ratio(
        aug_results, ["f1_macro", "accuracy", "roc_auc_ovr"], alpha=alpha
    )
    optimal_ratio = float(opt.get("f1_macro", {}).get("optimal_ratio", min_augmented_ratio))
    logger.info(
        f"H1-optimal augmented ratio (f1_macro, ratio=0.0 excluded): {optimal_ratio}"
    )

    num_classes = len(cfg["data"]["classes"])
    rows: List[Dict[str, Any]] = []
    for arch in sorted({c.architecture for c in cells}):
        for cell in [c for c in cells if c.architecture == arch]:
            ckpt = checkpoint_dir / f"{cell_run_id(cell.ratio, cell.distribution_strategy, cell.architecture, cell.seed)}.pt"
            if not ckpt.exists():
                logger.warning(f"missing {ckpt}; skipping {cell.run_id}")
                continue
            model = build_classifier(arch, num_classes=num_classes, pretrained=False).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            hmap = heatmaps_for_samples(model, arch, loader, indices, device)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            # Score against the independent anatomical reference, NOT another
            # model's attention.
            per_img = [
                evaluate_heatmap_pair(hmap[i], anatomical_ref[i], heatmap_threshold, compute_emd)
                for i in range(hmap.shape[0])
            ]
            agg = pd.DataFrame(per_img).mean(numeric_only=True).to_dict()
            agg.update({
                "ratio": cell.ratio,
                "distribution_strategy": cell.distribution_strategy,
                "architecture": cell.architecture,
                "seed": cell.seed,
                "run_id": cell.run_id,
                "n_images": len(indices),
            })
            rows.append(agg)
            logger.info(
                f"[{arch}] {cell.run_id}: IoU={agg['iou']:.3f} Dice={agg['dice']:.3f} "
                f"CoM={agg['center_of_mass_distance']:.2f}"
                + (f" EMD={agg['earth_movers_distance']:.2f}" if compute_emd else "")
            )
            heatmap_name = cell_run_id(cell.ratio, cell.distribution_strategy, cell.architecture, cell.seed)
            np.savez_compressed(
                heatmap_dir / f"{heatmap_name}.npz",
                model_heatmaps=hmap, anatomical_reference=anatomical_ref, indices=np.asarray(indices),
            )

    return rows, xai_dir, results_df, optimal_ratio, alpha


def write_reports(rows, xai_dir, results_df, optimal_ratio, alpha):
    if not rows:
        raise RuntimeError("No XAI metrics computed.")
    xai_df = pd.DataFrame(rows)
    xai_df.to_csv(xai_dir / "xai_metrics.csv", index=False)
    logger.info(f"Saved {len(xai_df)} rows -> {xai_dir / 'xai_metrics.csv'}")

    xai_df["optimal_ratio"] = optimal_ratio
    try:
        findings = run_all_hypothesis_tests(
            results_df, xai_df, alpha=alpha, optimal_cell={"ratio": optimal_ratio}
        )
        findings_out = {
            hid: {
                "statement": res.statement,
                "supported": bool(res.supported) if res.supported is not None else None,
                "evidence_keys": list(res.evidence.keys()),
                "decision": res.evidence.get("decision"),
                "p_ratio_by_metric": res.evidence.get("p_ratio_by_metric"),
                "preservation": res.evidence.get("preservation"),
            }
            for hid, res in findings.items()
        }
        h4 = findings["H4"]
        h4_decision = h4.evidence.get("decision")
        h4_preservation = h4.evidence.get("preservation")
        h4_pvalues = h4.evidence.get("p_ratio_by_metric")
    except Exception as exc:
        logger.warning(f"H4 statistical test failed ({exc}); recording untestable finding.")
        h4_decision = {
            "supported": None,
            "testable": False,
            "error": str(exc),
            "note": (
                "Not enough independent cell replications for the 3-way ANOVA; "
                "statistical inference skipped. XAI metrics were still computed."
            ),
        }
        findings_out = {"H4": {"statement": None, "supported": None, "decision": h4_decision}}
    h4_decision["reference_method"] = {
        "type": "independent_anatomical_roi",
        "detail": (
            "Every cell (including ratio=0.0) is scored against the SAME "
            "per-image outer-retina (ONL+RPE) binary ROI derived from the "
            "model-free intensity-profile segmenter. No model is ever scored "
            "against its own attention, so the ratio=0.0 'baseline' row is a "
            "genuine measure of baseline attention quality, not a "
            "self-comparison perfect score."
        ),
    }
    h4_decision["optimal_ratio_search"] = {
        "method": (
            "Optimal augmented ratio searched only among cells with ratio greater "
            "than 0.0, so the preservation test is never a self-comparison against "
            "the baseline. Baseline heatmaps (ratio=0.0) are also scored against "
            "the independent anatomical reference."
        ),
    }
    with open(xai_dir / "h4_findings.json", "w") as f:
        json.dump(findings_out, f, indent=2, default=str)
    logger.info(f"H4 supported={h4_decision.get('supported')} -> {xai_dir / 'h4_findings.json'}")

    metric_cols = ["iou", "dice", "center_of_mass_distance"]
    if "earth_movers_distance" in xai_df.columns:
        metric_cols.append("earth_movers_distance")
    summary = xai_df.groupby(
        ["ratio", "distribution_strategy", "architecture"], as_index=False
    )[metric_cols].mean()
    summary.sort_values(["architecture", "ratio", "distribution_strategy"], inplace=True)
    summary.to_csv(xai_dir / "xai_metrics_summary.csv", index=False)
    logger.info("H4 explainability run complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="H4 explainability driver.")
    p.add_argument("--config", type=str, default="configs/config_full_factorial.yaml")
    p.add_argument("--ratios", type=str, default=None)
    p.add_argument("--strategies", type=str, default=None)
    p.add_argument("--architectures", type=str, default=None)
    p.add_argument("--max-test-samples", type=int, default=120)
    p.add_argument("--no-emd", action="store_true")
    p.add_argument("--heatmap-threshold", type=float, default=0.5)
    a = p.parse_args()

    rows, xai_dir, results_df, optimal_ratio, alpha = main(
        a.config,
        parse_csv(a.ratios, cast=float),
        parse_csv(a.strategies),
        parse_csv(a.architectures),
        a.max_test_samples,
        not a.no_emd,
        a.heatmap_threshold,
    )
    write_reports(rows, xai_dir, results_df, optimal_ratio, alpha)