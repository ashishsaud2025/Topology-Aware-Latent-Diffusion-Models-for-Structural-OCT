# Topology-Aware Latent Diffusion Models for Structural OCT Image Augmentation

A systematic factorial study of synthetic OCT data augmentation using MONAI Latent
Diffusion Models. The pipeline covers preprocessing, generative model fine-tuning
(AutoencoderKL + Diffusion UNet), synthetic image generation, classifier training
(ResNet50, EfficientNet-B0) across a 3×3×2 factorial grid, evaluation, and
statistical analysis.

## Pipeline

```
Real OCT Dataset
  → Data Preprocessing & Class Analysis (CLAHE, z-score, stratified split)
  → Train AutoencoderKL (perceptual + KL loss)
  → Train Diffusion UNet in latent space (class-conditioned DDPM)
  → Generate Synthetic Images per Class
  → Experimental Design (Factor A: synthetic ratio, Factor B: distribution strategy, Factor C: architecture)
  → Materialize Experimental Datasets (mixing real + synthetic)
  → Train Classification Models (ResNet50 / EfficientNet-B0)
  → Evaluate on Fixed Real Test Set (accuracy, F1-macro)
  → Statistical Analysis (3-way ANOVA, main & interaction effects)
```

## Repository layout

```
oct-topology-diffusion/
├── configs/
│   └── config.yaml                    # single source of truth for all experiment parameters
├── data/
│   ├── preprocessing.py               # loading, CLAHE, z-score, stratified patient-level split
│   ├── dataset.py                     # PyTorch Dataset/DataLoader (real + synthetic mixing)
│   └── init_test_data.py              # generates a synthetic test dataset for development
├── generative/
│   ├── train_ldm.py                   # MONAI AutoencoderKL + DiffusionModelUNet training
│   └── generate_synthetic.py          # per-class conditional sampling via DDPM
├── experiment/
│   ├── factorial_design.py            # builds the 3×3×2 factorial cell grid (Factors A/B/C)
│   └── dataset_builder.py             # allocates synthetic counts per cell
├── models/
│   └── classifiers.py                 # ResNet50 / EfficientNet-B0 factory
├── training/
│   └── train_classifier.py            # generic training loop over an experimental cell
├── evaluation/
│   └── evaluate.py                    # accuracy + F1-macro on a test DataLoader
├── explainability/                    # (stubs — Grad-CAM, Attention Rollout, quantitative XAI)
│   ├── gradcam.py
│   ├── attention_rollout.py
│   └── quantitative_xai.py
├── stats/
│   └── statistical_analysis.py        # 3-way factorial ANOVA
├── hypotheses/
│   └── hypothesis_tests.py            # (stub — H1–H4 test runners)
├── utils/
│   ├── seed.py                        # deterministic seeding
│   └── logging_utils.py               # cell_run_id helper
├── scripts/
│   ├── run_full_pipeline.py           # (stub — orchestrator)
│   ├── run_generation.py              # (stub)
│   ├── run_experiment_grid.py         # (stub)
│   └── run_analysis.py                # (stub)
├── tests/
│   └── test_factorial_design.py       # unit tests for the factorial grid
├── _quick_demo.py                     # end-to-end pipeline demo (see Quickstart)
├── _smoke_test.py                     # smoke tests for all modules
├── requirements.txt
└── .gitignore
```

## Quickstart

### 1. Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # Linux/Mac
pip install -r requirements.txt
```

### 2. End-to-end demo (~5–10 min on CPU)

```bash
python _quick_demo.py
```

This runs the entire pipeline on a synthetic 30-image OCT dataset:

| Stage | What happens |
|-------|-------------|
| 1–2   | Generate test data → Preprocess (CLAHE, z-score, stratified 60/20/20 split) |
| 3     | Build DataLoaders (RGB 3‑channel for classifiers, grayscale 1‑channel for generator) |
| 4     | Train AutoencoderKL (2 epochs, 54.8M params) |
| 5     | Train Diffusion UNet latent-space model (2 epochs, 26.6M params) |
| 6     | Generate 15 synthetic images (5 per class) from the trained generator |
| 7     | Build 3×3×2 factorial grid (3 ratios × 3 strategies × 2 architectures → 14 unique cells) |
| 8     | Train a classifier per cell (1 epoch) |
| 9     | Evaluate all cells on the fixed real test set |
| 10    | Statistical analysis (3‑way ANOVA) |

All artifacts land in `_demo_output/`:
- `config.yaml` — saved config
- `processed/` — preprocessed images + split CSVs
- `models/` — trained autoencoder & diffusion checkpoints (`.pt`)
- `checkpoints/` — classifier checkpoints per cell
- `experimental_indices/` — train index CSVs per cell
- `processed/synthetic/` — generated synthetic images + index
- `master_results.csv` — aggregated accuracy & F1 for all cells

### 3. Smoke tests

```bash
python _smoke_test.py
```

Validates all modules import correctly and core functions run without errors (8 subtests).

### 4. Full pipeline via CLI entrypoint

```bash
# Preprocess + class analysis
python main.py preprocess --config configs/config.yaml

# Fine-tune MONAI LDM and generate synthetic images per class
python main.py generate --config configs/config.yaml

# Build the factorial grid and run all experimental cells
python main.py run-grid --config configs/config.yaml

# Aggregate results, run XAI + statistical analysis, test H1–H4
python main.py analyze --config configs/config.yaml
```

## Implementation status

| Module | Status | Notes |
|--------|--------|-------|
| `data/preprocessing.py` | ✅ Complete | CLAHE, z‑score, stratified patient‑level split, class analysis report |
| `data/dataset.py` | ✅ Complete | OCTImageDataset, mix of real + synthetic, DataLoader builder |
| `data/init_test_data.py` | ✅ Complete | Generates 3‑class synthetic OCT data for development |
| `generative/train_ldm.py` | ✅ Complete | AE + UNet builders, training loops, checkpoint save/load |
| `generative/generate_synthetic.py` | ✅ Complete | Class‑conditional DDPM sampling, synthetic pool generation |
| `models/classifiers.py` | ✅ Complete | ResNet50, EfficientNet‑B0 with pretrained‑weight management |
| `experiment/factorial_design.py` | ✅ Complete | 3‑factor grid, deduplication, metadata summaries |
| `experiment/dataset_builder.py` | ✅ Complete | `allocate_synthetic_counts` with 3 strategies |
| `training/train_classifier.py` | ✅ Complete | Training loop, gradient scaling, early stopping, checkpointing |
| `evaluation/evaluate.py` | ✅ Complete | Accuracy + F1‑macro on any DataLoader |
| `stats/statistical_analysis.py` | ✅ Complete | 3‑way factorial ANOVA via statsmodels |
| `explainability/` | 🚧 Stubs | Grad‑CAM, Attention Rollout, quantitative XAI ready for integration |
| `hypotheses/` | 🚧 Stub | H1–H4 hypothesis test skeleton |
| `scripts/run_*.py` | 🚧 Stubs | Orchestration scripts (pipeline currently runs via `_quick_demo.py`) |

## Project structure notes

- **Model checkpoint compatibility**: `build_autoencoder()` / `build_diffusion_unet()` are the
  single source of truth for architecture definitions. The demo uses these same builders so
  checkpoints are always compatible with `generate_synthetic_pool()`.
- **Channel handling**: The AutoencoderKL expects 1‑channel (grayscale) input. The classifier
  DataLoaders use 3‑channel (RGB). Separate transforms handle each.
- **Config-driven**: All experiment parameters (factors, training hyperparameters, paths) live in
  the config dict, serialised to `config.yaml` at the start of each run.