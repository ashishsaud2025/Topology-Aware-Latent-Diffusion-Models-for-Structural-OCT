"""
=============================================================================
Quick Demo — End-to-End OCT Pipeline Walkthrough
=============================================================================
Run this from the project root:

    python _quick_demo.py

It will:
  1. Generate a small synthetic test dataset (3 classes, 10 images each)
  2. Run the full preprocessing pipeline (split, class analysis)
  3. Train a tiny AutoencoderKL + DiffusionModelUNet (2 epochs each)
  4. Generate a small pool of synthetic images
  5. Build the 3×3×2 factorial grid (3 ratios × 3 distributions × 2 architectures)
  6. Train a classifier for each cell (1 epoch each, on CPU)
  7. Evaluate all cells on the fixed real test set
  8. Print comparative results

Expected runtime: ~5–10 minutes on CPU.
=============================================================================
"""
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Step 0: Create test data + output directories
# ---------------------------------------------------------------------------
print("=" * 70)
print("QUICK DEMO — OCT Topology Diffusion Pipeline")
print("=" * 70)

demo_dir = PROJECT_ROOT / "_demo_output"
test_raw_dir = PROJECT_ROOT / "data" / "test_raw"

# Ensure test raw data exists
if not (test_raw_dir / "NORMAL").exists():
    print("\nGenerating test dataset...")
    from data.init_test_data import main as init_data
    # run directly
    import cv2
    rng = np.random.RandomState(42)
    classes = ["NORMAL", "CNV", "DME"]
    image_size = 64
    for cls in classes:
        cls_dir = test_raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            img = rng.randn(image_size, image_size).astype(np.float32) * 0.3
            gradient = np.linspace(-0.5, 0.5, image_size).reshape(1, image_size)
            img += gradient
            if cls == "CNV":
                cx, cy = rng.randint(16, 48, size=2)
                img[cy-4:cy+4, cx-4:cx+4] += 1.0
            elif cls == "DME":
                img += 0.3 * np.sin(np.linspace(0, 4*np.pi, image_size)).reshape(1, image_size)
            img -= img.min()
            img /= (img.max() + 1e-8)
            img_uint8 = (img * 255).astype(np.uint8)
            cv2.imwrite(str(cls_dir / f"{cls}_{i:04d}.jpeg"), img_uint8)
    print("  ✓ Test dataset generated")

# Clean demo output
if demo_dir.exists():
    shutil.rmtree(demo_dir)
demo_dir.mkdir(parents=True)
processed_dir = demo_dir / "processed"

# ---------------------------------------------------------------------------
# Step 1: Config
# ---------------------------------------------------------------------------
cfg = {
    "project": {
        "name": "oct-quick-demo",
        "seed": 42,
        "output_dir": str(demo_dir),
        "device": "cpu",
    },
    "data": {
        "raw_dir": str(test_raw_dir),
        "processed_dir": str(processed_dir),
        "classes": ["NORMAL", "CNV", "DME"],
        "image_size": 64,
        "test_split": 0.2,
        "val_split": 0.2,
        "train_split": 0.6,
    },
    "generative": {
        "model_type": "monai_ldm",
        "autoencoder_checkpoint": str(demo_dir / "models" / "autoencoder.pt"),
        "diffusion_checkpoint": str(demo_dir / "models" / "diffusion_unet.pt"),
        "latent_channels": 1,
        "num_train_epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_inference_steps": 5,
        "scheduler": "ddpm",
        "conditioning": "class_label",
    },
    "experiment": {
        "factor_a_synthetic_ratio": [0.0, 0.25, 0.5],
        "factor_b_distribution_strategy": ["proportional", "minority_only", "fully_balanced"],
        "factor_c_architecture": ["resnet50", "efficientnet_b0"],
        "n_seeds_per_cell": 1,
    },
    "training": {
        "batch_size": 4,
        "num_epochs": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "optimizer": "adamw",
        "lr_scheduler": "cosine",
        "early_stopping_patience": 3,
        "num_workers": 0,
    },
    "evaluation": {
        "metrics": ["accuracy", "f1"],
    },
}

config_path = demo_dir / "config.yaml"
with open(config_path, "w") as f:
    yaml.dump(cfg, f)
print(f"\n[1] Config saved to {config_path}")

from utils.seed import set_global_seed
set_global_seed(cfg["project"]["seed"])

# ---------------------------------------------------------------------------
# Stage 1: Preprocessing
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[2] PREPROCESSING")
print(f"{'='*70}")

from data.preprocessing import (
    load_raw_dataset_index,
    preprocess_images,
    stratified_patient_level_split,
    save_splits_to_csv,
    run_class_analysis_report,
)

raw_index = load_raw_dataset_index(cfg["data"]["raw_dir"], cfg["data"]["classes"])
print(f"  Raw images: {len(raw_index)}")

processed_index = preprocess_images(raw_index, cfg["data"]["processed_dir"], cfg["data"]["image_size"])
print(f"  Processed images: {len(processed_index)}")

splits = stratified_patient_level_split(
    processed_index,
    train_frac=cfg["data"]["train_split"],
    val_frac=cfg["data"]["val_split"],
    test_frac=cfg["data"]["test_split"],
    seed=cfg["project"]["seed"],
)
split_dir = processed_dir / "splits"
save_splits_to_csv(splits, str(split_dir))
run_class_analysis_report(splits, str(split_dir / "class_analysis_report.csv"))
print(f"  Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

# ---------------------------------------------------------------------------
# Stage 2: Dataset + DataLoader
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[3] DATASET + DATALOADER")
print(f"{'='*70}")

import pandas as pd
import torch

from data.dataset import OCTImageDataset, build_data_loaders, get_default_transforms

train_df = pd.read_csv(split_dir / "train.csv")
val_df = pd.read_csv(split_dir / "val.csv")
test_df = pd.read_csv(split_dir / "test.csv")
class_to_idx = {c: i for i, c in enumerate(cfg["data"]["classes"])}

train_transform = get_default_transforms(cfg["data"]["image_size"], train=True)
val_transform = get_default_transforms(cfg["data"]["image_size"], train=False)

train_dataset = OCTImageDataset(train_df, class_to_idx, train_transform, image_size=cfg["data"]["image_size"])
val_dataset = OCTImageDataset(val_df, class_to_idx, val_transform, image_size=cfg["data"]["image_size"])
test_dataset = OCTImageDataset(test_df, class_to_idx, val_transform, image_size=cfg["data"]["image_size"])

loaders = build_data_loaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=cfg["training"]["batch_size"],
    num_workers=cfg["training"]["num_workers"],
)

batch = next(iter(loaders["train"]))
img, label = batch[0], batch[1]
print(f"  Train batch: img={img.shape}, label={label.shape}")
print(f"  ✓ DataLoaders ready")

# ---------------------------------------------------------------------------
# Stage 3: Build grayscale loaders for generative training
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[4] BUILD GRAYSCALE LOADERS FOR GENERATIVE TRAINING")
print(f"{'='*70}")

# AutoencoderKL (via build_autoencoder) expects 1-channel input. The standard
# OCTImageDataset transforms convert grayscale->RGB (3ch).  For the generative
# stage we use grayscale-only transforms.
from torchvision import transforms as tv_transforms
from data.dataset import OCTImageDataset, build_data_loaders

gen_transform = tv_transforms.Lambda(lambda x: x)  # identity — _load_image returns (1,H,W) normalized tensor

gen_train = OCTImageDataset(train_df, class_to_idx, gen_transform, image_size=cfg["data"]["image_size"])
gen_val   = OCTImageDataset(val_df,   class_to_idx, gen_transform, image_size=cfg["data"]["image_size"])
gen_test  = OCTImageDataset(test_df,  class_to_idx, gen_transform, image_size=cfg["data"]["image_size"])

gen_loaders = build_data_loaders(gen_train, gen_val, gen_test, batch_size=4, num_workers=0)
print(f"  ✓ Grayscale loaders ready (img.shape from next batch should be [N,1,H,W])")
batch = next(iter(gen_loaders["train"]))
print(f"    Batch img shape: {batch[0].shape}")

# ---------------------------------------------------------------------------
# Stage 4: Train Autoencoder (via standard builders)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[5] TRAIN AUTOENCODERKL (2 epochs, via config-driven builder)")
print(f"{'='*70}")

from generative.train_ldm import (
    build_autoencoder, build_diffusion_unet,
    train_autoencoder_stage, train_diffusion_stage,
)
from monai.networks.schedulers import DDPMScheduler
from torch.utils.data import DataLoader

cfg["generative"]["autoencoder_checkpoint"] = str(Path(cfg["generative"]["autoencoder_checkpoint"]))
ae = build_autoencoder(cfg)
t0 = time.time()
ae = train_autoencoder_stage(ae, gen_loaders["train"], gen_loaders["val"], cfg)
print(f"  ✓ Autoencoder trained ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Stage 5: Train Diffusion UNet (via config-driven builder)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[6] TRAIN DIFFUSION UNET (2 epochs, via config-driven builder)")
print(f"{'='*70}")

cfg["generative"]["diffusion_checkpoint"] = str(Path(cfg["generative"]["diffusion_checkpoint"]))
unet = build_diffusion_unet(cfg)
scheduler = DDPMScheduler(num_train_timesteps=cfg["generative"]["num_inference_steps"])
t0 = time.time()
unet = train_diffusion_stage(ae, unet, scheduler, gen_loaders["train"], gen_loaders["val"], cfg)
print(f"  ✓ Diffusion UNet trained ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Stage 6: Generate Synthetic Pool
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[7] GENERATE SYNTHETIC IMAGE POOL (using trained AE + UNet)")
print(f"{'='*70}")

from generative.generate_synthetic import (
    compute_max_synthetic_requirement,
    generate_synthetic_pool,
)
from data.preprocessing import compute_class_distribution

real_train_counts = compute_class_distribution(splits["train"]).counts
print(f"  Real train counts: {real_train_counts}")

max_req = compute_max_synthetic_requirement(
    real_train_counts,
    cfg["experiment"]["factor_a_synthetic_ratio"],
    cfg["experiment"]["factor_b_distribution_strategy"],
)
print(f"  Max synthetic required: {max_req}")

# No override needed -- standard architecture was used.

synthetic_pool_df = generate_synthetic_pool(
    cfg, max_req, processed_dir / "synthetic"
)
print(f"  Generated {len(synthetic_pool_df)} synthetic images")
print(f"  Synthetic pool classes: {synthetic_pool_df['label'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# Stage 7: Build factorial design + materialize experimental datasets
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[8] FACTORIAL DESIGN + EXPERIMENTAL DATASETS")
print(f"{'='*70}")

from experiment.factorial_design import build_factorial_grid, deduplicate_baseline_cells, grid_summary
from experiment.dataset_builder import materialize_all_cells

cells = build_factorial_grid(cfg)
cells = deduplicate_baseline_cells(cells)
print(f"  Grid: {grid_summary(cells)}")

index_dir = demo_dir / "experimental_indices"
materialize_all_cells(splits["train"], synthetic_pool_df, cells, str(index_dir))
print(f"  ✓ Materialized {len(cells)} experimental datasets")

# ---------------------------------------------------------------------------
# Stage 8: Train classifiers for each cell (1 epoch each for demo)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[9] TRAIN CLASSIFIERS (1 epoch per cell)")
print(f"{'='*70}")

from training.train_classifier import train_experimental_cell
from utils.logging_utils import cell_run_id

checkpoint_dir = demo_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

val_dataset_for_training = OCTImageDataset(
    pd.read_csv(split_dir / "val.csv"),
    class_to_idx,
    val_transform,
    image_size=cfg["data"]["image_size"],
)
val_loader_for_training = DataLoader(
    val_dataset_for_training,
    batch_size=cfg["training"]["batch_size"],
    shuffle=False,
    num_workers=cfg["training"]["num_workers"],
    pin_memory=True,
    drop_last=False,
)

for cell in cells:
    safe_name = cell.run_id.replace("|", "_").replace("=", "")
    train_index_csv = index_dir / f"{safe_name}.csv"
    cell_train_df = pd.read_csv(train_index_csv)
    cell_train_dataset = OCTImageDataset(
        cell_train_df, class_to_idx, train_transform, image_size=cfg["data"]["image_size"]
    )
    cell_train_loader = DataLoader(
        cell_train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    ckpt = train_experimental_cell(cell, cell_train_loader, val_loader_for_training, cfg, str(checkpoint_dir))
    print(f"  ✓ {cell.run_id}: checkpoint saved to {ckpt.name}")

# ---------------------------------------------------------------------------
# Stage 9: Evaluate all cells
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[10] EVALUATE ON FIXED REAL TEST SET")
print(f"{'='*70}")

from evaluation.evaluate import evaluate_all_cells

results_df = evaluate_all_cells(
    cells, checkpoint_dir, loaders["test"], cfg, str(demo_dir / "master_results.csv")
)
print(f"  ✓ Evaluated {len(results_df)} cells")
print(f"\n  Results summary:")
for _, row in results_df.iterrows():
    print(f"    {row['run_id']:45s} acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}")

# ---------------------------------------------------------------------------
# Stage 10: Statistical analysis (ANOVA)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("[11] STATISTICAL ANALYSIS (3-way ANOVA)")
print(f"{'='*70}")

from stats.statistical_analysis import summarize_main_and_interaction_effects

try:
    anova_tables = summarize_main_and_interaction_effects(
        results_df, ["accuracy", "f1_macro"], cfg
    )
    for metric, table in anova_tables.items():
        print(f"\n  ANOVA for {metric}:")
        print(f"  {table.to_string()}")
except (ValueError, RuntimeError) as e:
    print(f"  ANOVA skipped (insufficient variance among cells): {e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("QUICK DEMO COMPLETE")
print(f"{'='*70}")
print(f"\nResults:")
print(f"  Test accuracy range:  [{results_df['accuracy'].min():.4f}, {results_df['accuracy'].max():.4f}]")
print(f"  Test F1-macro range:  [{results_df['f1_macro'].min():.4f}, {results_df['f1_macro'].max():.4f}]")
print(f"  Best cell:            {results_df.loc[results_df['accuracy'].idxmax(), 'run_id']}")
print(f"\nAll artifacts in: {demo_dir}")
print(f"  {demo_dir / 'master_results.csv'}")
print(f"  {demo_dir / 'checkpoints' / '*'}")
print(f"  {processed_dir / 'synthetic' / '*'}")
print(f"  {demo_dir / 'experimental_indices' / '*'}")
print(f"\nTo clean up: shutil.rmtree('{demo_dir}')")