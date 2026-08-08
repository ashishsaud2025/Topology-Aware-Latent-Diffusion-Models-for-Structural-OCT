# OCT Model Explorer

A local Gradio app for interactively loading your project's checkpoints
(autoencoder, diffusion UNet, classifiers) and generating report-ready
output images: encode/decode, generate synthetic samples, classify.

# File Layout

```
oct-topology-diffusion/          <- project root (has models/, outputs/, etc.)
├── models/
│   └── classifiers.py
├── outputs/
│   └── ... (master_results.csv, anova_*.csv, etc.)
└── ui/                          <- put app.py, model_core.py, summary.py HERE
    ├── app.py
    ├── model_core.py
    └── summary.py
```

`model_core.py` adds the project root to `sys.path` automatically based on
its own location (`parent.parent`), so this exact folder depth matters --
don't move it directly into the project root or it'll look one level too
high.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if you're using a virtual environment,
which is recommended if you don't already have one for this project.)

## Run

```bash
cd ui
python app.py
```

Then open the printed `http://127.0.0.1:7860` URL in your browser.

## Usage

1. **Tab 1 (Load Models)**: paste the full local path to each checkpoint
   and click Load. Loading is **strict** i.e. if a checkpoint's architecture
   doesn't match what's built in `model_core.py`, you get a clear error
   immediately, not a silently broken model.
2. **Tab 2 (Autoencoder)**: upload a real OCT image, click "Encode -> Decode"
   to see input / latent / reconstruction side by side.
3. **Tab 3 (Diffusion)**: pick a class, timesteps, and seed, click Generate
   for one image, or "Generate all classes" for a 4-panel report figure.
   Requires BOTH the autoencoder and diffusion UNet loaded (diffusion
   generates in latent space, then decodes through the autoencoder).
4. **Tab 4 (Summary: Results Dashboard)**: point it at your results folder
   (the one with `master_results.csv`, the `anova_*.csv` files,
   `xai_metrics_summary.csv`, `topology_report.csv`, and `h4_findings.json`)
   and click "Load Summary" to render:
   - **Hypothesis findings** (H1-H4) as color-coded supported/not-supported
     cards with key p-values
   - **All 4 ANOVA tables** (accuracy, balanced accuracy, F1-macro, ROC-AUC)
     in collapsible accordions
   - **Marginal-means plot**: performance by ratio / architecture /
     distribution strategy, side by side
   - **XAI metrics heatmap grid**: IoU/Dice/CoM/EMD by (ratio, architecture)
   - **Topology validation table**: real-vs-synthetic boundary comparison,
     with a pass/fail count

   Any individual file that's missing from the folder is skipped rather
   than crashing the whole tab so you'll just see that section come back
   empty.

5. **Tab 5 (Classifier)**: pick an architecture (must be loaded in Tab 1
   first), upload an image, get class probabilities.

Every generated/reconstructed image is auto-saved to `./report_outputs/`
with a timestamped filename, in addition to the download button Gradio
puts on every image in the UI -- so you won't lose anything even if you
forget to click download.