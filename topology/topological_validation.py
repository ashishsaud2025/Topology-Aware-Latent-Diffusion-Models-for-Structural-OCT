"""End-to-end topology validation (Stage 2B).

Runs retinal layer segmentation on real and synthetic OCT images, computes
per-boundary topological invariants (Betti numbers), and compares the two
distributions. This is the data-quality gate before synthetic images enter
the classification pipeline: synthetic images must preserve the same
layer-continuity topology as real images.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from topology.layer_segmentation import BOUNDARY_NAMES, RetinalLayerSegmenter
from topology.persistent_homology import compute_boundary_topology
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TopologyValidator:
    """Compare topological signatures of real vs synthetic OCT images.

    Args:
        segmenter: RetinalLayerSegmenter instance.
        n_batches: Number of images processed per window.
        max_workers: For future parallelization (reserved).
        use_persistence: Whether to use persistent homology when available.
    """
    segmenter: RetinalLayerSegmenter
    n_batches: int = 1
    max_workers: int = 1
    use_persistence: bool = True

    def _load_image(self, path: str | Path) -> np.ndarray:
        """Load a preprocessed grayscale OCT image.

        Preprocessed images (real + synthetic) are z-score normalized float32
        TIFFs. OpenCV's IMREAD_GRAYSCALE cannot decode 32-bit float samples
        ("TIFFRGBAImageOK: Sorry, can not handle images with 32-bit samples"),
        so use IMREAD_UNCHANGED and cast to float32 -- mirroring the loader in
        data/dataset.py::OCTImageDataset._load_image.
        """
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        # cv2 reads 16-bit or 8-bit depending on dtype; normalize to float32.
        return img.astype(np.float32)

    def _image_signature(self, image: np.ndarray) -> Dict[str, Dict[str, int]]:
        """Segment an image and compute its per-boundary Betti signature."""
        result = self.segmenter.segment(image)
        return compute_boundary_topology(result.boundaries, use_persistence=self.use_persistence)

    def _topology_features(
        self, image_paths: List[str | Path], label: str
    ) -> pd.DataFrame:
        """Compute topological signatures for a list of images.

        Returns a DataFrame with one row per image:
            filepath, label, B1_beta_0, B1_beta_1, ..., B7_beta_1
        """
        rows = []
        for path in image_paths:
            try:
                img = self._load_image(path)
                sig = self._image_signature(img)
                row = {"filepath": str(path), "label": label}
                for bname in BOUNDARY_NAMES:
                    row[f"{bname}_beta_0"] = sig[bname]["beta_0"]
                    row[f"{bname}_beta_1"] = sig[bname]["beta_1"]
                rows.append(row)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Skipping {path}: {exc}")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _mannwhitney(real: pd.Series, synth: pd.Series) -> float:
        """Two-sided Mann-Whitney U p-value (or NaN if degenerate)."""
        r = real.dropna().values
        s = synth.dropna().values
        if len(r) == 0 or len(s) == 0 or np.all(r == r[0]) and np.all(s == s[0]):
            return float("nan")
        try:
            return float(stats.mannwhitneyu(r, s, alternative="two-sided").pvalue)
        except ValueError:
            return float("nan")

    def compare_distributions(
        self, real_paths: List[str | Path], synth_paths: List[str | Path]
    ) -> pd.DataFrame:
        """Compare real vs synthetic topology across all 7 boundaries.

        Returns a summary DataFrame with one row per boundary:
            boundary, mean_beta0_real, mean_beta0_synth, mean_beta1_real,
            mean_beta1_synth, p_value_beta0, p_value_beta1, topology_passed
        """
        real_df = self._topology_features(real_paths, "real")
        synth_df = self._topology_features(synth_paths, "synthetic")

        rows = []
        for bname in BOUNDARY_NAMES:
            b0_real = real_df[f"{bname}_beta_0"]
            b0_synth = synth_df[f"{bname}_beta_0"]
            b1_real = real_df[f"{bname}_beta_1"]
            b1_synth = synth_df[f"{bname}_beta_1"]

            p0 = self._mannwhitney(b0_real, b0_synth)
            p1 = self._mannwhitney(b1_real, b1_synth)

            # Pass if means are close AND no significant distribution shift
            topology_passed = bool(
                abs(b0_real.mean() - b0_synth.mean()) <= 0.5
                and abs(b1_real.mean() - b1_synth.mean()) <= 0.5
                and (np.isnan(p0) or p0 >= 0.05)
                and (np.isnan(p1) or p1 >= 0.05)
            )
            rows.append({
                "boundary": bname,
                "mean_beta0_real": round(float(b0_real.mean()), 4),
                "mean_beta0_synth": round(float(b0_synth.mean()), 4),
                "mean_beta1_real": round(float(b1_real.mean()), 4),
                "mean_beta1_synth": round(float(b1_synth.mean()), 4),
                "p_value_beta0": round(p0, 6) if not np.isnan(p0) else None,
                "p_value_beta1": round(p1, 6) if not np.isnan(p1) else None,
                "topology_passed": topology_passed,
            })
        return pd.DataFrame(rows)


def run_topological_validation(
    cfg: Dict[str, Any],
    real_index: pd.DataFrame,
    synthetic_index: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Run the full Stage 2B topology validation.

    Args:
        cfg: Full pipeline config.
        real_index: DataFrame of real images with [filepath, label].
        synthetic_index: DataFrame of synthetic images with [filepath, label].
        output_dir: Where to write topology_report.csv and per-class results.

    Returns:
        Combined per-image topology DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_cfg = cfg.get("topology", {}).get("segmentation", {})
    segmenter = RetinalLayerSegmenter(
        backend=seg_cfg.get("backend", "profile"),
        checkpoint_path=seg_cfg.get("checkpoint_path"),
        device=cfg["project"].get("device", "cuda"),
    )
    validator = TopologyValidator(
        segmenter=segmenter,
        use_persistence=cfg.get("topology", {}).get("use_persistence", True),
    )

    # Per-class comparison: sample real + synthetic per class
    class_summaries = []
    all_image_rows = []

    for cls in cfg["data"]["classes"]:
        real_paths = real_index.loc[real_index["label"] == cls, "filepath"].tolist()
        synth_paths = synthetic_index.loc[synthetic_index["label"] == cls, "filepath"].tolist()
        if not real_paths or not synth_paths:
            logger.warning(f"Class '{cls}' missing real or synthetic images; skipping.")
            continue

        # Cap the number of images per class for computational feasibility
        max_samples = cfg.get("topology", {}).get("max_samples_per_class", 200)
        real_sample = real_paths[:max_samples]
        synth_sample = synth_paths[:max_samples]

        per_image_real = validator._topology_features(real_sample, "real")
        per_image_synth = validator._topology_features(synth_sample, "synthetic")
        per_image_real["topology_class"] = cls
        per_image_synth["topology_class"] = cls
        all_image_rows.extend([per_image_real, per_image_synth])

        summary = validator.compare_distributions(real_sample, synth_sample)
        summary.insert(0, "class", cls)
        class_summaries.append(summary)

    # Combine and persist
    all_images_df = pd.concat(all_image_rows, ignore_index=True)
    all_images_df.to_csv(output_dir / "topology_per_image.csv", index=False)

    summary_df = (
        pd.concat(class_summaries, ignore_index=True)
        if class_summaries
        else pd.DataFrame()
    )
    summary_df.to_csv(output_dir / "topology_report.csv", index=False)

    overall_pass = bool(
        summary_df["topology_passed"].all() if not summary_df.empty else False
    )
    logger.info(
        f"Topology validation complete: {len(all_images_df)} images, "
        f"{len(summary_df)} boundary-class comparisons, overall_pass={overall_pass}"
    )
    logger.info(f"Topology report written to {output_dir / 'topology_report.csv'}")
    return summary_df