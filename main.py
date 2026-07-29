"""Top-level CLI entrypoint.

    python main.py preprocess  --config configs/config.yaml
    python main.py generate    --config configs/config.yaml
    python main.py run-grid    --config configs/config.yaml [--cell-index N]
    python main.py analyze     --config configs/config.yaml
    python main.py full        --config configs/config.yaml   # orchestrates all stages inline
"""
from __future__ import annotations

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCT topology-aware LDM augmentation study -- pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("preprocess", "generate", "run-grid", "analyze", "full"):
        sp = subparsers.add_parser(name)
        sp.add_argument("--config", type=str, default="configs/config.yaml")
        if name == "run-grid":
            sp.add_argument("--cell-index", type=int, default=None)

    args = parser.parse_args()

    if args.command == "preprocess":
        from data.preprocessing import (
            load_raw_dataset_index,
            preprocess_images,
            run_class_analysis_report,
            save_splits_to_csv,
            stratified_patient_level_split,
        )
        from utils.seed import load_config, set_global_seed
        from pathlib import Path

        cfg = load_config(args.config)
        set_global_seed(cfg["project"]["seed"])
        raw_index = load_raw_dataset_index(cfg["data"]["raw_dir"], cfg["data"]["classes"])
        processed_index = preprocess_images(
            raw_index, cfg["data"]["processed_dir"], cfg["data"]["image_size"]
        )
        splits = stratified_patient_level_split(
            processed_index,
            train_frac=cfg["data"]["train_split"],
            val_frac=cfg["data"]["val_split"],
            test_frac=cfg["data"]["test_split"],
            seed=cfg["project"]["seed"],
        )
        # Persist split CSVs so downstream stages can load them independently
        split_out_dir = Path(cfg["data"]["processed_dir"]) / "splits"
        save_splits_to_csv(splits, str(split_out_dir))
        run_class_analysis_report(splits, str(split_out_dir / "class_analysis_report.csv"))

    elif args.command == "generate":
        from scripts.run_generation import main as run_generation_main

        run_generation_main(args.config)

    elif args.command == "run-grid":
        from scripts.run_experiment_grid import main as run_grid_main

        run_grid_main(args.config, args.cell_index)

    elif args.command == "analyze":
        from scripts.run_analysis import main as run_analysis_main

        run_analysis_main(args.config)

    elif args.command == "full":
        from scripts.run_full_pipeline import main as run_full_main

        run_full_main(args.config)


if __name__ == "__main__":
    main()
