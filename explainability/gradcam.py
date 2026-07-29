"""Explainability Analysis (CNN branch): Grad-CAM for ResNet50 / EfficientNet-B0.

Produces class-discriminative localization heatmaps highlighting which
image regions drove the classifier's prediction, used both qualitatively
(visual auditing) and quantitatively (explainability/quantitative_xai.py).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

# TODO: pip package `grad-cam` (pytorch-grad-cam) is listed in requirements.txt
# and provides a battle-tested implementation; wire it in below rather than
# hand-rolling hooks, unless custom behavior is required.
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image


def get_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Return the convolutional layer Grad-CAM should hook into, per
    configs/config.yaml['explainability']['target_layers'].

    TODO: for resnet50 -> model.backbone.layer4[-1]
          for efficientnet_b0 -> model.backbone.features[-1] (confirm exact
          submodule name against the torchvision EfficientNet-B0 definition)
    """
    raise NotImplementedError("TODO: resolve target layer per architecture")


def compute_gradcam_heatmap(
    model: nn.Module,
    architecture: str,
    input_tensor: torch.Tensor,
    target_class: int | None,
) -> np.ndarray:
    """Compute a single Grad-CAM heatmap (H, W), normalized to [0, 1], for
    `input_tensor` (shape 1 x C x H x W).

    If `target_class` is None, Grad-CAM targets the model's own top-1
    prediction (standard "explain what the model actually decided" mode).

    TODO: instantiate pytorch_grad_cam.GradCAM with
    target_layers=[get_target_layer(model, architecture)], run
    cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)]),
    and return the resulting heatmap.
    """
    raise NotImplementedError("TODO: implement Grad-CAM heatmap computation")


def batch_gradcam_heatmaps(
    model: nn.Module,
    architecture: str,
    inputs: torch.Tensor,
    target_classes: torch.Tensor | None = None,
) -> np.ndarray:
    """Vectorized convenience wrapper computing Grad-CAM for a batch,
    returning an array of shape (N, H, W).
    """
    heatmaps = []
    for i in range(inputs.shape[0]):
        tgt = None if target_classes is None else int(target_classes[i].item())
        heatmaps.append(
            compute_gradcam_heatmap(model, architecture, inputs[i : i + 1], tgt)
        )
    return np.stack(heatmaps, axis=0)
