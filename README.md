# Topology-Aware Latent Diffusion Models for Structural OCT Image Augmentation

A systematic factorial study of synthetic OCT data augmentation using MONAI Latent
Diffusion Models. The pipeline covers preprocessing, generative model fine-tuning
(AutoencoderKL + Diffusion UNet), synthetic image generation, topological validation,
classifier training (ResNet50, EfficientNet-B0, ViT-Base/16) across a configurable
factorial grid (synthetic ratio x distribution strategy x architecture), evaluation,
quantitative explainability (Grad-CAM / Attention Rollout), and statistical testing
of four pre-registered hypotheses (H1-H4).

The full pipeline, including the statistics and hypothesis-testing stages, is
implemented and runnable end-to-end on a real dataset via the CLI or the
individual stage scripts (see Quickstart below). Results from a full training
run will be added to this README once a larger-dataset, multi-seed evaluation
is complete.

## Pipeline

```
Real OCT Dataset
  -> Data Preprocessing & Class Analysis (CLAHE, z-score, stratified split)
  -> Train AutoencoderKL (perceptual + KL loss)
  -> Train Diffusion UNet in latent space (class-conditioned DDPM)
  -> Generate Synthetic Images per Class
  -> Topological Validation (Stage 2B: layer segmentation + persistent homology)
  -> Experimental Design (Factor A: synthetic ratio, Factor B: distribution strategy, Factor C: architecture)
  -> Materialize Experimental Datasets (mixing real + synthetic)
  -> Train Classification Models (ResNet50 / EfficientNet-B0 / ViT-Base/16)
  -> Evaluate on Fixed Real Test Set (accuracy, F1-macro, balanced accuracy, ROC-AUC)
  -> Explainability (Grad-CAM for CNNs, Attention Rollout for ViT) + Quantitative XAI (IoU, Dice, CoM, EMD vs. anatomical reference)
  -> Statistical Analysis (3-way ANOVA, Tukey HSD post-hoc, main & interaction effects)
  -> Hypothesis Testing (H1-H4)
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
├── topology/
│   ├── layer_segmentation.py          # 8-layer retinal segmentation (profile + UNet backends)
│   ├── persistent_homology.py         # Betti numbers (beta_0, beta_1) + persistence diagrams
│   └── topological_validation.py      # real-vs-synthetic topology gate (Stage 2B)
├── experiment/
│   ├── factorial_design.py            # builds the config-driven factorial cell grid (Factors A/B/C)
│   └── dataset_builder.py             # allocates synthetic counts per cell
├── models/
│   └── classifiers.py                 # ResNet50 / EfficientNet-B0 / ViT-Base/16 factory
├── training/
│   └── train_classifier.py            # generic training loop over an experimental cell
├── evaluation/
│   └── evaluate.py                    # accuracy, F1, balanced accuracy, ROC-AUC, per-class report
├── explainability/
│   ├── gradcam.py                     # Grad-CAM for ResNet50 / EfficientNet-B0
│   ├── attention_rollout.py           # Attention Rollout for ViT-Base/16
│   └── quantitative_xai.py            # IoU, Dice, center-of-mass distance, earth mover's distance
├── stats/
│   └── statistical_analysis.py        # 3-way factorial ANOVA + Tukey HSD post-hoc
├── hypotheses/
│   └── hypothesis_tests.py            # H1-H4 test runners (ratio effect, distribution effect,
│                                       # architecture interaction, explainability preservation)
├── utils/
│   ├── seed.py                        # deterministic seeding
│   └── logging_utils.py               # cell_run_id helper
├── scripts/
│   ├── run_full_pipeline.py           # end-to-end orchestrator (mirrors the pipeline diagram)
│   ├── run_generation.py              # Stage: LDM fine-tuning + synthetic pool generation
│   ├── run_experiment_grid.py         # Stage: build grid + train one/all classifier cells
│   ├── run_full_factorial.py          # runs every cell of the configured factorial grid end-to-end
│   ├── run_analysis.py                # aggregates results, runs ANOVA + hypothesis tests
│   ├── run_h4_explainability.py       # quantitative-XAI pass (Grad-CAM/rollout vs. anatomical ref) for H4
│   ├── visualize_xai.py               # renders heatmap-comparison / ratio x architecture XAI figures
│   ├── train_duke_segmentation.py     # trains the Duke DME layer-segmentation UNet (Stage 2B backend)
│   ├── _probe_attn.py / _smoke_xai.py # small dev-time debugging/smoke scripts
├── tests/
│   ├── test_factorial_design.py       # unit tests for the factorial grid
│   └── test_topology.py               # unit tests for persistent-homology / topology validation
├── ui/                                 # local Gradio app for exploring trained checkpoints
│   ├── app.py                          # Gradio UI: encode/decode, generate synthetic samples, classify
│   ├── model_core.py                   # loads project checkpoints (assumes ui/ sits at project_root/ui)
│   ├── summary.py                      # renders master_results/ANOVA/hypothesis summaries in the UI
│   └── README.md                       # setup + file-layout notes for the UI app
├── _quick_demo.py                     # end-to-end pipeline demo on synthetic data (see Quickstart)
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

### 2. End-to-end demo (~5-10 min on CPU)

```bash
python _quick_demo.py
```

This runs the entire pipeline on a synthetic 30-image OCT dataset:

| Stage | What happens |
|-------|-------------|
| 1-2   | Generate test data -> Preprocess (CLAHE, z-score, stratified 60/20/20 split) |
| 3     | Build DataLoaders (RGB 3-channel for classifiers, grayscale 1-channel for generator) |
| 4     | Train AutoencoderKL (2 epochs, 54.8M params) |
| 5     | Train Diffusion UNet latent-space model (2 epochs, 26.6M params) |
| 6     | Generate 15 synthetic images (5 per class) from the trained generator |
| 7     | Build 3x3x2 factorial grid (3 ratios x 3 strategies x 2 architectures -> 14 unique cells) |
| 8     | Train a classifier per cell (1 epoch) |
| 9     | Evaluate all cells on the fixed real test set |
| 10    | Statistical analysis (3-way ANOVA) |

All artifacts land in `_demo_output/`:
- `config.yaml` - saved config
- `processed/` - preprocessed images + split CSVs
- `models/` - trained autoencoder & diffusion checkpoints (`.pt`)
- `checkpoints/` - classifier checkpoints per cell
- `experimental_indices/` - train index CSVs per cell
- `processed/synthetic/` - generated synthetic images + index
- `master_results.csv` - aggregated accuracy & F1 for all cells

### 3. Full pipeline via CLI entrypoint

```bash
# Preprocess + class analysis
python main.py preprocess --config configs/config.yaml

# Fine-tune MONAI LDM and generate synthetic images per class
python main.py generate --config configs/config.yaml

# Build the factorial grid and run all experimental cells
python main.py run-grid --config configs/config.yaml [--cell-index N]

# Aggregate results, run XAI + statistical analysis, test H1-H4
python main.py analyze --config configs/config.yaml

# Or run every stage above in one inline process (see scripts/run_full_pipeline.py)
python main.py full --config configs/config.yaml
```

For a large factorial run, `scripts/run_full_factorial.py` and
`scripts/run_h4_explainability.py` can be used directly so long-running stages
(LDM fine-tuning, the full training sweep) can be checkpointed/resumed
independently rather than run inline.

### 4. Explore results locally (optional)

A small Gradio app under `ui/` loads your trained checkpoints (autoencoder,
diffusion UNet, classifiers) and the aggregated result CSVs to let you
interactively generate synthetic samples, classify, and browse the
`master_results.csv` / ANOVA / hypothesis summaries.

```bash
cd ui
pip install -r requirements.txt
python app.py
```

See `ui/README.md` for the expected file layout (`ui/` must sit directly under
the project root, alongside `models/` and `outputs/`).

## Implementation status

| Module | Status | Notes |
|--------|--------|-------|
| `data/preprocessing.py` | Complete | CLAHE, z-score, stratified patient-level split, class analysis report |
| `data/dataset.py` | Complete | OCTImageDataset, mix of real + synthetic, DataLoader builder |
| `data/init_test_data.py` | Complete | Generates synthetic test data for development |
| `generative/train_ldm.py` | Complete | AE + UNet builders, training loops, checkpoint save/load |
| `generative/generate_synthetic.py` | Complete | Class-conditional DDPM sampling, synthetic pool generation |
| `topology/layer_segmentation.py` | Complete | 8-layer retinal segmentation (intensity-profile + MONAI UNet backends) |
| `topology/persistent_homology.py` | Complete | Betti numbers + persistence diagrams (ripser/gudhi with exact fallback) |
| `topology/topological_validation.py` | Complete | Real-vs-synthetic topology comparison, Mann-Whitney gate |
| `models/classifiers.py` | Complete | ResNet50, EfficientNet-B0, ViT-Base/16 (timm) with pretrained-weight management |
| `experiment/factorial_design.py` | Complete | Config-driven factor grid, baseline deduplication, metadata summaries |
| `experiment/dataset_builder.py` | Complete | `allocate_synthetic_counts` with 3 strategies |
| `training/train_classifier.py` | Complete | Training loop, gradient scaling, early stopping, checkpointing |
| `evaluation/evaluate.py` | Complete | Accuracy, precision/recall/F1 (macro+weighted), balanced accuracy, ROC-AUC (OVR), per-class report |
| `explainability/` | Complete | Grad-CAM (CNNs), Attention Rollout (ViT), quantitative XAI (IoU/Dice/CoM/EMD vs. independent anatomical reference) |
| `stats/statistical_analysis.py` | Complete | 3-way factorial ANOVA (statsmodels) + Tukey HSD post-hoc |
| `hypotheses/hypothesis_tests.py` | Complete | H1-H4 test runners |
| `scripts/run_full_factorial.py`, `run_h4_explainability.py`, `run_analysis.py` | Complete | Orchestration for a full factorial run + aggregation |
| `scripts/visualize_xai.py` | Complete | Grad-CAM/reference heatmap comparisons + ratio x architecture XAI grids |
| `ui/` | Complete | Local Gradio app for interactive checkpoint exploration and results browsing |
| `scripts/run_full_pipeline.py`, `run_generation.py`, `run_experiment_grid.py` | Complete | Stage orchestration entrypoints used by `main.py` |

## Results

Pending: a full training run on a larger dataset with multiple seeds per cell
is in progress. This section will be filled in with topology-gate pass rates,
H1-H4 hypothesis outcomes, and headline classifier/XAI metrics once that run
completes.

## Project structure notes

- **Model checkpoint compatibility**: `build_autoencoder()` / `build_diffusion_unet()` are the
  single source of truth for architecture definitions. The demo uses these same builders so
  checkpoints are always compatible with `generate_synthetic_pool()`.
- **Channel handling**: The AutoencoderKL expects 1-channel (grayscale) input. The classifier
  DataLoaders use 3-channel (RGB). Separate transforms handle each.
- **Config-driven**: All experiment parameters (factors, training hyperparameters, paths) live in
  the config dict, serialised to `config.yaml` at the start of each run.
