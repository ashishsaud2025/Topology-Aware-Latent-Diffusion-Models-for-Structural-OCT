"""Explainability Analysis (CNN branch): Grad-CAM for ResNet50 / EfficientNet-B0.

Produces class-discriminative localization heatmaps highlighting which
image regions drove the classifier's prediction, used both qualitatively
(visual auditing) and quantitatively (explainability/quantitative_xai.py).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Return the convolutional layer Grad-CAM should hook into.

    The ClassifierWrapper records the architecture-appropriate target layer
    (backbone.layer4 for ResNet50, backbone.features[-1] for EfficientNet-B0)
    at build time in models/classifiers.py; when that attribute is present we
    prefer it over the string config so the factory's validation is reused.
    """
    recorded = getattr(model, "gradcam_target_layer", None)
    if recorded is not None:
        return recorded

    backbone = model.backbone if hasattr(model, "backbone") else model
    if architecture == "resnet50":
        return backbone.layer4
    if architecture == "efficientnet_b0":
        # torchvision EfficientNet-B0: last feature block ends with the
        # final Conv2d ("conv_head") before global average pooling.
        return backbone.features[-1][-1]
    raise ValueError(f"No Grad-CAM target layer defined for architecture '{architecture}'")


def compute_batch_gradcam_heatmaps(
    model: nn.Module,
    architecture: str,
    inputs: torch.Tensor,
    target_classes: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmaps for a batch (N x C x H x W) using a single
    GradCAM instance (batch-parallel forward/backward), returning an array of
    shape (N, H, W), each heatmap clipped to [0, 1].

    `target_classes` (iterable of ints, length N) optionally forces the CAM
    target per image; when None, each image is explained against the model's
    own top-1 prediction (standard "explain what the model decided").
    """
    if inputs.dim() != 4:
        raise ValueError(
            f"inputs must be 4D (N x C x H x W), got shape {tuple(inputs.shape)}"
        )

    target_layer = get_target_layer(model, architecture)
    was_training = model.training
    model.eval()

    try:
        cam = GradCAM(model=model, target_layers=[target_layer])
        targets = None
        if target_classes is not None:
            targets = [ClassifierOutputTarget(int(t)) for t in target_classes]
        grayscale_cam = cam(input_tensor=inputs, targets=targets)
        return np.clip(grayscale_cam, 0.0, 1.0)
    finally:
        if was_training:
            model.train()


def compute_gradcam_heatmap(
    model: nn.Module,
    architecture: str,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Compute a single Grad-CAM heatmap (H, W), normalized to [0, 1], for
    `input_tensor` (shape 1 x C x H x W). If `target_class` is None, the
    model's own top-1 prediction is explained.
    """
    if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
        raise ValueError(
            f"input_tensor must be 4D with batch size 1, "
            f"got shape {tuple(input_tensor.shape)}"
        )
    targets = None if target_class is None else torch.tensor([int(target_class)])
    return compute_batch_gradcam_heatmaps(model, architecture, input_tensor, targets)[0]


def batch_gradcam_heatmaps(
    model: nn.Module,
    architecture: str,
    inputs: torch.Tensor,
    target_classes: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Convenience alias of :func:`compute_batch_gradcam_heatmaps`."""
    return compute_batch_gradcam_heatmaps(model, architecture, inputs, target_classes)