"""
OCT Model Explorer -- local Gradio app for interactively loading your
autoencoder / diffusion UNet / classifier checkpoints and producing
report-ready output images (encode/decode, generate, classify).

Run locally with:
    pip install gradio monai torch torchvision einops --break-system-packages
    python app.py
Then open the printed http://127.0.0.1:7860 URL in a browser.

Every generated image is auto-saved to ./report_outputs/ with a
timestamped filename, in addition to being shown in the UI (which also has
its own per-image download button) -- so nothing you generate is lost even
if you forget to click download before closing the tab.
"""

from __future__ import annotations

import datetime
import glob
import os

import gradio as gr
import numpy as np
from PIL import Image

import model_core as mc
import summary as sm

OUTPUT_DIR = "report_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Holds whatever is currently loaded. Kept as a plain dict (not gr.State)
# since these are large models we want loaded ONCE and reused across many
# UI interactions, not re-created per request.
STATE = {"ae": None, "unet": None, "classifiers": {}}  # classifiers: arch -> model


def _save(arr: np.ndarray, prefix: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(OUTPUT_DIR, f"{prefix}_{ts}.png")
    Image.fromarray(arr).save(path)
    return path


# loaders

def ui_load_autoencoder(path):
    if not path or not os.path.exists(path):
        return f"File not found: {path}"
    try:
        model, status = mc.load_autoencoder(path)
        STATE["ae"] = model
        return status
    except Exception as e:
        return f"ERROR: {e}"


def ui_load_diffusion(path):
    if not path or not os.path.exists(path):
        return f"File not found: {path}"
    try:
        model, status = mc.load_diffusion_unet(path)
        STATE["unet"] = model
        return status
    except Exception as e:
        return f"ERROR: {e}"


def ui_load_classifier(arch, path):
    if not path or not os.path.exists(path):
        return f"File not found: {path}"
    try:
        model, status = mc.load_classifier(arch, path)
        STATE["classifiers"][arch] = model
        return status
    except Exception as e:
        return f"ERROR: {e}"


# tab actions

def ui_encode_decode(img):
    if STATE["ae"] is None:
        raise gr.Error("Load the Autoencoder checkpoint first (Load Models tab).")
    if img is None:
        raise gr.Error("Upload an image first.")
    inp, latent, recon, mse = mc.ae_encode_decode(STATE["ae"], img)
    _save(inp, "ae_input")
    _save(latent, "ae_latent")
    p = _save(recon, "ae_reconstruction")
    return inp, latent, recon, f"MSE(input, reconstruction) = {mse:.6f}\nSaved: {p}"


def ui_generate(class_name, num_train_timesteps, num_inference_steps, seed):
    if STATE["unet"] is None:
        raise gr.Error("Load the Diffusion UNet checkpoint first (Load Models tab).")
    if STATE["ae"] is None:
        raise gr.Error("Load the Autoencoder checkpoint first -- generation decodes through it.")
    class_idx = mc.CLASS_NAMES.index(class_name)
    img, latent = mc.diffusion_generate(
        STATE["unet"], STATE["ae"], class_idx,
        int(num_train_timesteps), int(num_inference_steps), int(seed),
    )
    p = _save(img, f"synthetic_{class_name}")
    return img, latent, f"Saved: {p}"


def ui_generate_grid(num_train_timesteps, num_inference_steps, seed):
    if STATE["unet"] is None or STATE["ae"] is None:
        raise gr.Error("Load both the Autoencoder and Diffusion UNet checkpoints first.")
    imgs = []
    for i, name in enumerate(mc.CLASS_NAMES):
        img, _ = mc.diffusion_generate(
            STATE["unet"], STATE["ae"], i,
            int(num_train_timesteps), int(num_inference_steps), int(seed),
        )
        _save(img, f"grid_{name}")
        imgs.append((img, name))
    return imgs


def ui_classify(arch, img):
    if arch not in STATE["classifiers"]:
        raise gr.Error(f"Load the {arch} checkpoint first (Load Models tab).")
    if img is None:
        raise gr.Error("Upload an image first.")
    return mc.classify_image(STATE["classifiers"][arch], img)


def ui_load_summary(results_folder):
    if not results_folder or not os.path.isdir(results_folder):
        raise gr.Error(f"Not a valid folder: {results_folder}")

    findings = sm.load_hypothesis_findings(results_folder)
    hyp_html = sm.render_hypothesis_html(findings)

    anova_tables = sm.load_anova_tables(results_folder)
    anova_dfs = [anova_tables.get(label) for label in sm.ANOVA_FILES.keys()]

    marginal_fig = sm.plot_marginal_means(results_folder)
    xai_fig = sm.plot_xai_heatmap_grid(results_folder)
    topo_df, topo_note = sm.load_topology_report(results_folder)

    found = glob.glob(os.path.join(results_folder, "**", "*.csv"), recursive=True) + \
            glob.glob(os.path.join(results_folder, "**", "*.json"), recursive=True)
    status = f"Loaded from `{results_folder}`. Found {len(found)} csv/json files."

    return (status, hyp_html, *anova_dfs, marginal_fig, xai_fig, topo_df,
            topo_note if topo_df is not None else "topology_report.csv not found in this folder.")


# UI

with gr.Blocks(title="OCT Model Explorer") as demo:
    gr.Markdown("# OCT Model Explorer\nLoad checkpoints, run inference, get report-ready images. "
                f"All outputs are also auto-saved to `./{OUTPUT_DIR}/`.")

    with gr.Tab("1. Load Models"):
        gr.Markdown("Paste the full local path to each checkpoint (`.pt`). "
                     "Loading uses a **strict** state_dict match -- if the architecture "
                     "doesn't match the file, you'll get a clear error instead of a silently "
                     "broken model.")
        with gr.Row():
            ae_path = gr.Textbox(label="Autoencoder checkpoint path", placeholder="/path/to/autoencoder.pt")
            ae_btn = gr.Button("Load Autoencoder")
        ae_status = gr.Textbox(label="Status", interactive=False)
        ae_btn.click(ui_load_autoencoder, inputs=ae_path, outputs=ae_status)

        with gr.Row():
            unet_path = gr.Textbox(label="Diffusion UNet checkpoint path", placeholder="/path/to/diffusion_unet.pt")
            unet_btn = gr.Button("Load Diffusion UNet")
        unet_status = gr.Textbox(label="Status", interactive=False)
        unet_btn.click(ui_load_diffusion, inputs=unet_path, outputs=unet_status)

        gr.Markdown("### Classifiers")
        for arch in ["resnet50", "efficientnet_b0", "vit_base"]:
            with gr.Row():
                cls_path = gr.Textbox(label=f"{arch} checkpoint path", placeholder=f"/path/to/{arch}.pt")
                cls_btn = gr.Button(f"Load {arch}")
            cls_status = gr.Textbox(label="Status", interactive=False)
            cls_btn.click(ui_load_classifier, inputs=[gr.State(arch), cls_path], outputs=cls_status)

    with gr.Tab("2. Autoencoder: Encode / Decode"):
        gr.Markdown("Upload a real OCT scan to see how the autoencoder compresses and reconstructs it.")
        with gr.Row():
            ae_input_img = gr.Image(type="pil", label="Input image")
            ae_run_btn = gr.Button("Encode -> Decode", variant="primary")
        with gr.Row():
            ae_out_input = gr.Image(label="Input (as fed to model)")
            ae_out_latent = gr.Image(label="Latent (3ch as RGB)")
            ae_out_recon = gr.Image(label="Reconstruction")
        ae_out_info = gr.Textbox(label="Info", interactive=False)
        ae_run_btn.click(ui_encode_decode, inputs=ae_input_img,
                          outputs=[ae_out_input, ae_out_latent, ae_out_recon, ae_out_info])

    with gr.Tab("3. Diffusion: Generate"):
        gr.Markdown("Generate a synthetic OCT image via DDIM sampling through the diffusion UNet, "
                     "then decode through the autoencoder. Requires BOTH models loaded.")
        with gr.Row():
            gen_class = gr.Dropdown(mc.CLASS_NAMES, value="NORMAL", label="Class")
            gen_T = gr.Number(value=250, precision=0, label="num_train_timesteps (must match training)")
            gen_steps = gr.Slider(10, 250, value=50, step=5, label="num_inference_steps (DDIM)")
            gen_seed = gr.Number(value=42, precision=0, label="Seed")
        gen_btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            gen_out_img = gr.Image(label="Generated image (decoded)")
            gen_out_latent = gr.Image(label="Final latent (3ch as RGB)")
        gen_out_info = gr.Textbox(label="Info", interactive=False)
        gen_btn.click(ui_generate, inputs=[gen_class, gen_T, gen_steps, gen_seed],
                      outputs=[gen_out_img, gen_out_latent, gen_out_info])

        gr.Markdown("---\n**Generate all 4 classes at once** (same seed/steps as above) -- handy for a report figure.")
        grid_btn = gr.Button("Generate all classes")
        grid_gallery = gr.Gallery(label="All classes", columns=4)
        grid_btn.click(ui_generate_grid, inputs=[gen_T, gen_steps, gen_seed], outputs=grid_gallery)

    with gr.Tab("4. Summary: Results Dashboard"):
        gr.Markdown(
            "Point this at your results folder (the one containing "
            "`master_results.csv`, the `anova_*.csv` files, `xai_metrics_summary.csv`, "
            "`topology_report.csv`, and `h4_findings.json`) to render everything together. "
            "Any individual file that's missing is skipped, not a hard failure."
        )
        with gr.Row():
            results_folder = gr.Textbox(label="Results folder path",
                                          placeholder="/path/to/outputs", scale=4)
            load_summary_btn = gr.Button("Load Summary", variant="primary", scale=1)
        summary_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Hypothesis findings")
        hyp_html_out = gr.HTML()

        gr.Markdown("### 3-way ANOVA tables")
        anova_outputs = []
        with gr.Row():
            for label in sm.ANOVA_FILES.keys():
                with gr.Accordion(label, open=False):
                    df_out = gr.Dataframe(label=label, wrap=True)
                    anova_outputs.append(df_out)

        gr.Markdown("### Classifier performance -- marginal means")
        marginal_plot = gr.Plot()

        gr.Markdown(
            "### Explainability (XAI) metrics -- ratio x architecture\n"
            "IoU/Dice measure overlap with an independent anatomical (outer-retina ONL+RPE) "
            "reference; CoM/EMD measure spatial distance -- lower is better for those two. "
            "*Note: `xai_confusion_matrix.csv`, despite its filename, contains this same kind "
            "of IoU/Dice/CoM/EMD data (a proportional-only slice), not an actual per-class "
            "prediction confusion matrix -- no true/predicted-label file was available to "
            "build a real one.*"
        )
        xai_plot = gr.Plot()

        gr.Markdown("### Topology validation (real vs. synthetic layer structure)")
        topo_note_out = gr.Markdown()
        topo_df_out = gr.Dataframe(wrap=True)

        load_summary_btn.click(
            ui_load_summary,
            inputs=results_folder,
            outputs=[summary_status, hyp_html_out, *anova_outputs,
                     marginal_plot, xai_plot, topo_df_out, topo_note_out],
        )

    with gr.Tab("5. Classifier: Predict"):
        gr.Markdown("Upload an image (real or generated) and classify it with a loaded model.")
        with gr.Row():
            cls_arch_select = gr.Dropdown(["resnet50", "efficientnet_b0", "vit_base"],
                                            value="resnet50", label="Architecture (must be loaded in Tab 1)")
            cls_input_img = gr.Image(type="pil", label="Input image")
        cls_btn = gr.Button("Classify", variant="primary")
        cls_out = gr.Label(label="Predicted class probabilities", num_top_classes=4)
        cls_btn.click(ui_classify, inputs=[cls_arch_select, cls_input_img], outputs=cls_out)


if __name__ == "__main__":
    demo.launch()
