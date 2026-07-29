"""Model factory for Factor C: Classification Architecture.

Provides a unified interface over ResNet50, EfficientNet-B0, and Vision
Transformer (ViT), so training/evaluation/explainability code can treat all
three architectures interchangeably where possible, while exposing the
architecture-specific hooks needed for Grad-CAM (CNNs) vs. Attention
Rollout (ViT).
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class ClassifierWrapper(nn.Module):
    """Wraps a backbone + classification head and records which layer(s)
    XAI methods should hook into.

    Attributes:
        backbone: the underlying torchvision/timm model.
        architecture_name: one of {"resnet50", "efficientnet_b0", "vit_base"}.
        gradcam_target_layer: module reference for Grad-CAM (CNNs only, else None).
        is_transformer: True for ViT, enabling attention-rollout hooks.
    """

    def __init__(self, backbone: nn.Module, architecture_name: str, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.architecture_name = architecture_name
        self.num_classes = num_classes
        self.is_transformer = architecture_name.startswith("vit")
        self.gradcam_target_layer: nn.Module | None = None  # set in build_classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_classifier(architecture: str, num_classes: int, pretrained: bool = True) -> ClassifierWrapper:
    """Factory returning a ClassifierWrapper for the requested architecture.

    architecture in {"resnet50", "efficientnet_b0", "vit_base"}
    """
    if architecture == "resnet50":
        # TODO: from torchvision.models import resnet50, ResNet50_Weights
        # backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        # backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        raise NotImplementedError("TODO: build torchvision ResNet50 + replace fc head")

    elif architecture == "efficientnet_b0":
        # TODO: from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        # backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        # backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, num_classes)
        raise NotImplementedError("TODO: build torchvision EfficientNet-B0 + replace classifier head")

    elif architecture == "vit_base":
        # TODO: via timm: backbone = timm.create_model("vit_base_patch16_224",
        #   pretrained=pretrained, num_classes=num_classes)
        # Ensure timm's attention modules expose attention weights (e.g. via
        # forward hooks or timm's `return_attention`/custom monkeypatch) for
        # explainability/attention_rollout.py.
        raise NotImplementedError("TODO: build timm ViT-Base + confirm attention weight access")

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_input_transform_stats(architecture: str) -> Dict[str, Any]:
    """Return the ImageNet (or architecture-specific) mean/std normalization
    stats expected by each pretrained backbone, so data/dataset.py transforms
    stay consistent with the pretrained weights being fine-tuned.

    TODO: confirm whether OCT grayscale images should be replicated to 3
    channels to match ImageNet-pretrained backbones, or whether a
    single-channel variant of each architecture should be used instead.
    """
    raise NotImplementedError("TODO: return per-architecture normalization stats")
