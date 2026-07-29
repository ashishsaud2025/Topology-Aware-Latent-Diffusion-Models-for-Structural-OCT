"""Model factory for Factor C: Classification Architecture.

Provides a unified interface over ResNet50, EfficientNet-B0, and Vision
Transformer (ViT), so training/evaluation/explainability code can treat all
three architectures interchangeably where possible, while exposing the
architecture-specific hooks needed for Grad-CAM (CNNs) vs. Attention
Rollout (ViT).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

from utils.logging_utils import get_logger

logger = get_logger(__name__)


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


def _build_resnet50(num_classes: int, pretrained: bool) -> ClassifierWrapper:
    """Build torchvision ResNet50 with the final fully-connected layer replaced
    to match `num_classes`.

    Grad-CAM target: backbone.layer4 (the last convolutional block before the
    global average pooling and FC layer).
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    backbone = resnet50(weights=weights)

    # Replace the final fully-connected layer
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    wrapper = ClassifierWrapper(backbone, "resnet50", num_classes)
    wrapper.gradcam_target_layer = backbone.layer4
    logger.info(
        f"Built ResNet50 (pretrained={pretrained}, "
        f"in_features={in_features} -> num_classes={num_classes})"
    )
    return wrapper


def _build_efficientnet_b0(num_classes: int, pretrained: bool) -> ClassifierWrapper:
    """Build torchvision EfficientNet-B0 with the final classifier head replaced
    to match `num_classes`.

    Grad-CAM target: backbone.features[-1] (the last convolutional feature
    block / "conv_head" in torchvision's implementation, which is the
    features[-1][-1] module — the final MBConv block's convolutional layer).
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    backbone = efficientnet_b0(weights=weights)

    # Replace the final classifier head
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    wrapper = ClassifierWrapper(backbone, "efficientnet_b0", num_classes)
    # The last feature layer in torchvision's EfficientNet is features[-1][0]
    # (the first operation within the final block). For Grad-CAM we want the
    # last convolutional layer, which is features[-1][-1] if the block ends
    # with a conv, or the block itself. We use the last block.
    wrapper.gradcam_target_layer = backbone.features[-1]
    logger.info(
        f"Built EfficientNet-B0 (pretrained={pretrained}, "
        f"in_features={in_features} -> num_classes={num_classes})"
    )
    return wrapper


def _build_vit_base(num_classes: int, pretrained: bool) -> ClassifierWrapper:
    """Build ViT-Base via timm (vit_base_patch16_224) with classification head
    replaced to match `num_classes`.

    ViT does NOT use Grad-CAM; instead, attention-rollout is applied by
    explainability/attention_rollout.py. The attention weight tensors are
    captured via forward hooks on `backbone.blocks[i].attn`.

    The `gradcam_target_layer` is set to None for this architecture.
    """
    backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Ensure the model returns attention weights during forward pass.
    # timm ViT models have an `attn` method but we need the raw attention
    # matrices from each block. We enable attention extraction by setting
    # `output_attn=True` on each attention module. The hooks in
    # explainability/attention_rollout.py will capture these.
    # For now, we monkey-patch a flag that the rollout code can check.
    for block in backbone.blocks:
        if hasattr(block, "attn"):
            # timm's Attention doesn't return attention by default, but we
            # can set need_weights if available. We'll rely on forward hooks
            # in attention_rollout.py instead.
            pass

    wrapper = ClassifierWrapper(backbone, "vit_base", num_classes)
    wrapper.gradcam_target_layer = None
    logger.info(
        f"Built ViT-Base/16 (pretrained={pretrained}, num_classes={num_classes}, "
        f"is_transformer={wrapper.is_transformer})"
    )
    return wrapper


def build_classifier(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
) -> ClassifierWrapper:
    """Factory returning a ClassifierWrapper for the requested architecture.

    Supported architectures in {"resnet50", "efficientnet_b0", "vit_base"}.

    Args:
        architecture: Name of the architecture to build.
        num_classes: Number of output classes for the classification head.
        pretrained: Whether to load ImageNet-pretrained weights.

    Returns:
        A ClassifierWrapper instance ready for training or evaluation.

    Raises:
        ValueError: If `architecture` is not recognized.
    """
    architecture_map = {
        "resnet50": _build_resnet50,
        "efficientnet_b0": _build_efficientnet_b0,
        "vit_base": _build_vit_base,
    }

    builder = architecture_map.get(architecture)
    if builder is None:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Supported: {list(architecture_map.keys())}"
        )

    return builder(num_classes, pretrained)


def get_input_transform_stats(architecture: str) -> Dict[str, Any]:
    """Return the ImageNet normalization stats expected by each pretrained
    backbone, so data transforms stay consistent with the pretrained weights.

    All three architectures (ResNet50, EfficientNet-B0, ViT) were pretrained
    on ImageNet and expect the standard ImageNet normalization:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    OCT images are grayscale, replicated to 3 channels in dataset.py, so
    these same normalization stats apply.

    Returns:
        Dict with keys "mean" (list[float]) and "std" (list[float]).
    """
    imagenet_stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    if architecture not in ("resnet50", "efficientnet_b0", "vit_base"):
        logger.warning(
            f"Unknown architecture '{architecture}' for transform stats. "
            "Falling back to ImageNet normalization."
        )

    return imagenet_stats