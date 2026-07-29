# Topology-Aware Latent Diffusion Models for Structural OCT Image Augmentation and Explainable Visual Auditing

Research codebase skeleton for the systematic 3×5×3 factorial study of synthetic OCT data
augmentation, covering generative fine-tuning (MONAI LDM), classifier training (ResNet50,
EfficientNet-B0, ViT), quantitative evaluation, and XAI-based structural auditing
(Grad-CAM / Attention Rollout).

## Pipeline

```
Real OCT Dataset
      -> Data Preprocessing & Class Analysis
      -> Fine-tune MONAI Generative Model
      -> Generate Synthetic Images per Class
      -> Experimental Design (Factor A: ratio, Factor B: distribution, Factor C: architecture)
      -> Create Experimental Datasets
      -> Train Classification Models (ResNet50 / EfficientNet-B0 / ViT)
      -> Evaluate on Fixed Real Test Set (Accuracy, Precision, Recall, F1, ROC-AUC, per-class)
      -> Explainability Analysis (Grad-CAM, Attention Rollout)
      -> Quantitative Explainability Analysis (IoU, Dice, CoM distance, EMD)
      -> Statistical Analysis (main effects, interaction effects, significance tests)
      -> Experimental Findings (H1-H4)
```

## Repository layout

```
oct-topology-diffusion/
├── configs/
│   └── config.yaml              # single source of truth for all experiment parameters
├── data/
│   ├── preprocessing.py         # loading, cleaning, class-balance analysis
│   └── dataset.py               # PyTorch Dataset/DataLoader classes (real + synthetic)
├── generative/
│   ├── train_ldm.py             # fine-tune MONAI Latent Diffusion Model
│   └── generate_synthetic.py    # per-class conditional sampling
├── experiment/
│   ├── factorial_design.py      # builds the 3x5x3 factorial cell grid (Factors A/B/C)
│   └── dataset_builder.py       # materializes each experimental cell's train set
├── models/
│   └── classifiers.py           # ResNet50 / EfficientNet-B0 / ViT factory + wrappers
├── training/
│   └── train_classifier.py      # generic training loop over an experimental cell
├── evaluation/
│   └── evaluate.py              # Accuracy/Precision/Recall/F1/ROC-AUC + per-class report
├── explainability/
│   ├── gradcam.py                # Grad-CAM for CNNs (ResNet50, EfficientNet-B0)
│   ├── attention_rollout.py      # Attention Rollout for ViT
│   └── quantitative_xai.py       # IoU, Dice, Center-of-Mass distance, EMD vs. lesion masks
├── stats/
│   └── statistical_analysis.py  # factorial ANOVA, interaction effects, post-hoc tests
├── hypotheses/
│   └── hypothesis_tests.py      # H1-H4 test runners built on top of stats + evaluation
├── utils/
│   ├── seed.py
│   └── logging_utils.py
├── scripts/
│   ├── run_full_pipeline.py     # end-to-end orchestrator (mirrors the pipeline diagram)
│   ├── run_generation.py
│   ├── run_experiment_grid.py
│   └── run_analysis.py
├── tests/
│   └── test_factorial_design.py
├── requirements.txt
└── main.py                      # CLI entrypoint
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Preprocess + class analysis
python main.py preprocess --config configs/config.yaml

# 2. Fine-tune the MONAI LDM and generate synthetic images per class
python main.py generate --config configs/config.yaml

# 3. Build the 3x5x3 factorial grid and run all experimental cells
python main.py run-grid --config configs/config.yaml

# 4. Aggregate results, run XAI + statistical analysis, test H1-H4
python main.py analyze --config configs/config.yaml
```

## Status

This is a **skeleton**: function/class signatures, docstrings, config wiring, and TODOs are in
place; model-specific implementation details (MONAI generative network internals, dataset paths,
lesion-mask sourcing for quantitative XAI, etc.) are left as `TODO` for you to fill in against
your actual dataset and compute environment.
