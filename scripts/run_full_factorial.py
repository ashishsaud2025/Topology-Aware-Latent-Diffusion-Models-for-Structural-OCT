"""Full-factorial orchestration runner.

Runs the COMPLETE 5 x 3 x 3 factorial design described in
configs/config_full_factorial.yaml (5 ratios, 3 distribution strategies
including fully_balanced, 3 architectures including ViT). Both the fast-track
config (configs/config.yaml) and the full config write to the SAME outputs/
root; checkpoints, experimental indices, results, XAI and topology artifacts
all live under that single directory.

Cell-count note: 5 ratios x 3 strategies x 3 architectures = 45 raw cells.
deduplicate_baseline_cells() collapses ratio=0.0 across strategies (the
distribution strategy is a no-op when there is no synthetic data), giving
3 architectures x (1 baseline + 4 non-zero ratios x 3 strategies) = 39
unique training runs.

The runner is designed around INCREMENTAL grid completion:

  * Checkpoint reuse: cells whose checkpoints already exist in
    outputs/checkpoints/ are skipped instead of retrained. At ratio=0.0 any
    strategy's baseline is reused because the training set is identical.
    Optionally pass --existing-checkpoints DIR to copy matching checkpoints
    from an EXTERNAL directory (e.g. a renamed output root) into
    outputs/checkpoints/ instead of retraining them.
  * Idempotent resume: cells whose checkpoints already exist in the target
    checkpoint dir are skipped.
  * Filtering: --ratios / --strategies / --architectures limit the grid to
    the requested factor levels (e.g. just the missing 0.5/0.75 cells, or
    just fully_balanced, or just ViT).

Usage:
    # Complete the whole grid, reusing already-trained fast-track checkpoints:
    python scripts/run_full_factorial.py --config configs/config_full_factorial.yaml \
        --existing-checkpoints outputs/checkpoints

    # Do ONLY the missing ratios 0.5 + 0.75 for the two CNNs (fast-track
    # strategies proportional + minority_only):
    python scripts/run_full_factorial.py --config configs/config_full_factorial.yaml \
        --ratios 0.5,0.75 --strategies proportional,minority_only \
        --architectures resnet50,efficientnet_b0

    # Do ONLY fully_balanced across ALL ratios and architectures:
    python scripts/run_full_factorial.py --config configs/config_full_factorial.yaml \
        --strategies fully_balanced --existing-checkpoints outputs/checkpoints

    # Do ONLY ViT across all ratios + strategies (reusing the CNN baselines is
    # unnecessary but harmless; run with --architectures vit_base):
    python scripts/run_full_factorial.py --config configs/config_full_factorial.yaml \
        --architectures vit_base

    # Train only one cell (cluster/Slurm array dispatch; index into the
    # filtered grid when filters are given):
    python scripts/run_full_factorial.py --config configs/config_full_factorial.yaml \
        --cell-index 7 --train-only

    # Re-run evaluation + analysis over ALL completed checkpoints:
    python scripts/run_full_factorial.py --config configs/config_full_factorial.yaml --skip-training

    Note: --skip-training with filters evaluates only the filtered subset.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, List, Optional

# Ensure the project root is importable when this script is run directly
# (e.g. `python scripts/run_full_factorial.py`, which puts scripts/ on sys.path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from experiment.dataset_builder import materialize_all_cells
from experiment.factorial_design import (
    ExperimentalCell,
    build_factorial_grid,
    deduplicate_baseline_cells,
    grid_summary,
)
from evaluation.evaluate import evaluate_all_cells
from hypotheses.hypothesis_tests import run_all_hypothesis_tests
from stats.statistical_analysis import summarize_main_and_interaction_effects
from utils.logging_utils import cell_run_id, get_logger
from utils.seed import load_config, set_global_seed

logger = get_logger("run_full_factorial")


# Helpers
def _require(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}. {what}")


def _parse_csv(value: Optional[str], cast=None) -> Optional[List[Any]]:
    """Parse a comma-separated CLI value into a list (optionally cast)."""
    if not value:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    if cast is not None:
        items = [cast(v) for v in items]
    return items


def filter_cells(
    cells: List[ExperimentalCell],
    ratios: Optional[List[float]] = None,
    strategies: Optional[List[str]] = None,
    architectures: Optional[List[str]] = None,
) -> List[ExperimentalCell]:
    """Restrict the grid to the requested factor levels. Unspecified factors
    are left unrestricted."""
    if not any([ratios, strategies, architectures]):
        return cells

    def _keep(c: ExperimentalCell) -> bool:
        if ratios is not None and c.ratio not in ratios:
            return False
        if strategies is not None and c.distribution_strategy not in strategies:
            return False
        if architectures is not None and c.architecture not in architectures:
            return False
        return True

    return [c for c in cells if _keep(c)]


def _expected_checkpoint_path(cell: ExperimentalCell, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"{cell_run_id(cell.ratio, cell.distribution_strategy, cell.architecture, cell.seed)}.pt"


def find_reusable_checkpoint(
    cell: ExperimentalCell, existing_dir: Path
) -> Optional[Path]:
    """Locate an already-trained checkpoint that can be reused for `cell`.

    - ratio > 0: exact run-id match required (same ratio/strategy/arch/seed).
    - ratio == 0: distribution strategy is a no-op (no synthetic data), so any
      strategy's baseline checkpoint for the same architecture+seed is valid.
    """
    if cell.ratio == 0.0:
        matches = sorted(
            existing_dir.glob(f"ratio0.00_*_{cell.architecture}_seed{cell.seed}.pt")
        )
        return matches[0] if matches else None
    exact = _expected_checkpoint_path(cell, existing_dir)
    return exact if exact.exists() else None


def build_fixed_test_loader(cfg: dict) -> torch.utils.data.DataLoader:
    """Build the single fixed real test set DataLoader (identical across all
    cells) from the persisted test split CSV. Shared with the fast-track run
    via the common data_processed/ directory."""
    from data.dataset import OCTImageDataset, get_default_transforms

    split_dir = Path(cfg["data"]["processed_dir"]) / "splits"
    _require(split_dir / "test.csv", "Run the preprocessing stage first.")
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


def load_real_train_df(cfg: dict) -> pd.DataFrame:
    """Load the real training split CSV (shared with the fast-track run)."""
    split_dir = Path(cfg["data"]["processed_dir"]) / "splits"
    _require(split_dir / "train.csv", "Run the preprocessing stage first.")
    return pd.read_csv(split_dir / "train.csv")


def load_synthetic_pool(cfg: dict) -> pd.DataFrame:
    """Load the pre-generated synthetic pool index produced by
    scripts/run_generation.py (shared via data_processed/)."""
    synthetic_index = (
        Path(cfg["data"]["processed_dir"]) / "synthetic" / "synthetic_index.csv"
    )
    _require(
        synthetic_index,
        "Run scripts/run_generation.py (or the fast-track run) to generate "
        "the synthetic pool first.",
    )
    return pd.read_csv(synthetic_index)


# Main

def main(
    config_path: str,
    cell_index: Optional[int],
    train_only: bool,
    skip_training: bool,
    ratios: Optional[List[float]],
    strategies: Optional[List[str]],
    architectures: Optional[List[str]],
    existing_checkpoints: Optional[str],
) -> None:
    if train_only and skip_training:
        raise ValueError("--train-only and --skip-training are mutually exclusive.")

    cfg = load_config(config_path)
    set_global_seed(cfg["project"]["seed"])

    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    index_dir = output_dir / "experimental_indices"
    existing_ckpt_dir = Path(existing_checkpoints) if existing_checkpoints else None
    if existing_ckpt_dir is not None and not existing_ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"--existing-checkpoints directory not found: {existing_ckpt_dir}"
        )
    logger.info(f"Full-factorial output directory: {output_dir.resolve()}")

    # Full factorial grid 
    cells = deduplicate_baseline_cells(build_factorial_grid(cfg))
    full_summary = grid_summary(cells)
    logger.info(f"Full factorial grid: {full_summary}")
    logger.info(f"Unique training runs in full grid: {len(cells)}")

    targets = filter_cells(cells, ratios, strategies, architectures)
    if ratios or strategies or architectures:
        logger.info(
            f"Filtered to {len(targets)} cell(s) "
            f"(ratios={ratios or 'all'}, strategies={strategies or 'all'}, "
            f"architectures={architectures or 'all'})"
        )
    if cell_index is not None:
        if not 0 <= cell_index < len(targets):
            raise IndexError(
                f"--cell-index {cell_index} out of range for {len(targets)} "
                f"target cell(s)."
            )
        targets = [targets[cell_index]]
        logger.info(f"Selected cell-index {cell_index}: {targets[0].run_id}")

    if not targets:
        raise ValueError("No cells match the requested filters.")

    # Stage 5: Materialize per-cell training datasets 
    if not skip_training:
        logger.info("Stage 5: materializing per-cell training indices...")
        real_train_df = load_real_train_df(cfg)
        synthetic_pool_df = load_synthetic_pool(cfg)
        materialize_all_cells(real_train_df, synthetic_pool_df, targets, index_dir)
        logger.info(f"Materialized indices for {len(targets)} cell(s) -> {index_dir}")
    else:
        logger.info("--skip-training: assuming indices are already materialized.")

    # Stage 6: Train classification models 
    if not skip_training:
        from data.dataset import OCTImageDataset, get_default_transforms
        from training.train_classifier import train_experimental_cell

        # Load the shared validation split (never receives synthetic data)
        split_dir = Path(cfg["data"]["processed_dir"]) / "splits"
        _require(split_dir / "val.csv", "Run the preprocessing stage first.")
        val_df = pd.read_csv(split_dir / "val.csv")

        class_to_idx = {c: i for i, c in enumerate(cfg["data"]["classes"])}
        val_transform = get_default_transforms(cfg["data"]["image_size"], train=False)
        val_dataset = OCTImageDataset(
            val_df, class_to_idx, val_transform, image_size=cfg["data"]["image_size"]
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=True,
            drop_last=False,
        )

        n_skipped = 0
        n_reused = 0
        n_trained = 0

        for cell in targets:
            ckpt_path = _expected_checkpoint_path(cell, checkpoint_dir)

            # 1) Already present in the target dir -> skip
            if ckpt_path.exists():
                logger.info(f"Checkpoint already exists: {ckpt_path.name}. Skipping.")
                n_skipped += 1
                continue

            # 2) Present in a previous run's checkpoint dir -> copy, don't retrain
            if existing_ckpt_dir is not None:
                src = find_reusable_checkpoint(cell, existing_ckpt_dir)
                if src is not None:
                    shutil.copy2(src, ckpt_path)
                    logger.info(
                        f"Reused existing checkpoint {src.name} -> {ckpt_path.name}"
                    )
                    n_reused += 1
                    continue

            # 3) Else train this cell
            logger.info(f"Training cell: {cell.run_id}")
            safe_name = cell.run_id.replace("|", "_").replace("=", "")
            train_index_csv = index_dir / f"{safe_name}.csv"
            if not train_index_csv.exists():
                raise FileNotFoundError(
                    f"Materialized training index not found: {train_index_csv}."
                )
            train_df = pd.read_csv(train_index_csv)
            train_transform = get_default_transforms(cfg["data"]["image_size"], train=True)
            train_dataset = OCTImageDataset(
                train_df, class_to_idx, train_transform, image_size=cfg["data"]["image_size"]
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg["training"]["batch_size"],
                shuffle=True,
                num_workers=cfg["training"]["num_workers"],
                pin_memory=True,
                drop_last=True,
            )
            train_experimental_cell(cell, train_loader, val_loader, cfg, checkpoint_dir)
            logger.info(f"Completed cell: {cell.run_id}")
            n_trained += 1

        logger.info(
            f"Cell disposition: {n_trained} trained, {n_reused} reused, "
            f"{n_skipped} already present (of {len(targets)} target cells)."
        )
    else:
        logger.info("--skip-training: skipping training.")

    if train_only:
        logger.info("--train-only: done.")
        return

    # Stage 7: Evaluate on the fixed real test set 
    # If filters were applied, evaluate only the filtered subset (checkpoints
    # for the rest may legitimately not exist yet). If no filters, evaluate the
    # full 39-cell grid (all checkpoints must be present).
    eval_cells = targets if (ratios or strategies or architectures) else cells
    logger.info(
        f"Stage 7: evaluating {len(eval_cells)} cell(s) on the fixed real test set..."
    )
    test_loader = build_fixed_test_loader(cfg)
    results_path = output_dir / "master_results.csv"

    results_df = None
    if results_path.exists():
        logger.info(f"Loading existing rows from {results_path}")
        results_df = pd.read_csv(results_path)

    # Evaluate only the current batch of cells (skip any whose run_id already
    # has a row in the accumulated results).
    if eval_cells:
        existing_run_ids = (
            set(results_df["run_id"]) if results_df is not None and "run_id" in results_df else set()
        )
        batch_cells = [c for c in eval_cells if c.run_id not in existing_run_ids]
        if batch_cells:
            tmp_results = output_dir / "master_results_tmp.csv"
            batch_df = evaluate_all_cells(batch_cells, checkpoint_dir, test_loader, cfg, tmp_results)
            results_df = batch_df if results_df is None else pd.concat(
                [results_df, batch_df], ignore_index=True
            )
            results_df = results_df.drop_duplicates(subset="run_id", keep="last")
            results_df.to_csv(results_path, index=False)
            logger.info(f"Appended {len(batch_df)} new cell(s) -> {results_path}")
        else:
            logger.info("All target cells already evaluated; no new evaluation needed.")
    else:
        logger.info("No cells to evaluate in this batch.")

    if results_df is None or results_df.empty:
        raise ValueError("No results available to analyze.")

    # grid_complete == True only when this batch (or the accumulated results)
    # covers the entire deduplicated grid.
    grid_complete = len(results_df) >= len(cells)
    logger.info(
        f"Results accumulated: {len(results_df)} / {len(cells)} grid cells "
        f"({'COMPLETE' if grid_complete else 'partial'})."
    )

    # Stage 10: Statistical analysis (3-way factorial ANOVA) 
    if grid_complete:
        logger.info("Stage 10: statistical analysis (3-way factorial ANOVA)...")
        metric_vars = ["accuracy", "f1_macro", "roc_auc_ovr", "balanced_accuracy"]
        anova_summaries = summarize_main_and_interaction_effects(
            results_df, metric_vars, cfg
        )
        for metric, table in anova_summaries.items():
            out = output_dir / f"anova_{metric}.csv"
            table.to_csv(out)
            logger.info(f"  ANOVA table for '{metric}' saved to {out}")

        # Stage 11: Experimental Findings (H1-H4) 
        logger.info("Stage 11: hypothesis testing (H1-H4)...")
        alpha = cfg.get("statistics", {}).get("alpha", 0.05)
        findings = run_all_hypothesis_tests(results_df, None, alpha=alpha)

        findings_out = output_dir / "hypothesis_findings.json"
        with open(findings_out, "w") as f:
            json.dump(
                {
                    hid: {
                        "statement": res.statement,
                        "supported": bool(res.supported) if res.supported is not None else None,
                        "evidence_keys": list(res.evidence.keys()),
                        "decision": res.evidence.get("decision"),
                    }
                    for hid, res in findings.items()
                },
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Hypothesis findings saved to {findings_out}")
    else:
        logger.info(
            "Grid not yet complete -- skipping ANOVA + hypothesis tests "
            "(run again after remaining cells finish)."
        )

    logger.info("Full-factorial run complete (incremental batch).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full 5x3x3 factorial grid incrementally "
                    "(non-destructive; writes to outputs/)."
    )
    parser.add_argument("--config", type=str, default="configs/config_full_factorial.yaml")
    parser.add_argument(
        "--cell-index",
        type=int,
        default=None,
        help="Run only the Nth cell of the (possibly filtered) grid "
             "(for cluster array-job dispatch).",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Materialize indices + train only; skip evaluation/statistics.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip materialization + training; only evaluate + run statistics "
             "using existing checkpoints.",
    )
    parser.add_argument(
        "--existing-checkpoints",
        type=str,
        default=None,
        help="Directory of already-trained checkpoints from an EXTERNAL previous "
             "run (optional). Matching checkpoints are copied into "
             "outputs/checkpoints/ instead of retrained. With the consolidated "
             "outputs/ root, existing checkpoints are simply resumed. At "
             "ratio=0.0 any strategy's baseline is reused for the same "
             "architecture.",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default=None,
        help="Comma-separated synthetic ratios to include, e.g. '0.5,0.75'.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated distribution strategies, e.g. 'fully_balanced'.",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        default=None,
        help="Comma-separated architectures, e.g. 'vit_base'.",
    )
    args = parser.parse_args()

    main(
        args.config,
        args.cell_index,
        args.train_only,
        args.skip_training,
        _parse_csv(args.ratios, cast=float),
        _parse_csv(args.strategies),
        _parse_csv(args.architectures),
        args.existing_checkpoints,
    )