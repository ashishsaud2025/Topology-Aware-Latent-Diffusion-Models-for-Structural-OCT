"""Explainability Analysis (ViT branch): Attention Rollout.

Implements the Abnar & Zuidema (2020) attention-rollout method: recursively
multiplies attention matrices across transformer layers (with a residual
"identity" term to account for skip connections) to produce a single
image-level attention map showing which input patches the ViT ultimately
attended to for its prediction.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def register_attention_hooks(
    model: nn.Module,
) -> List[torch.Tensor]:
    """Attach forward hooks to every transformer block's attention module to
    capture per-layer post-softmax attention weight matrices (shape:
    batch x num_heads x num_patches+1 x num_patches+1, including CLS token).

    timm's ViT uses fused SDPA attention by default (`fused_attn=True`),
    which does NOT materialize the attention weight matrix. To capture the
    weights we temporarily disable fused attention and hook the `attn_drop`
    module input, which holds the post-softmax weights.

    Returns a list of tensors captured during the next forward pass. The
    hooks are removed (and fused attention restored) when the returned
    context is closed.
    """
    captured: List[torch.Tensor] = []
    handles = []
    restored_models = []

    if not hasattr(model, "blocks"):
        # ClassifierWrapper wraps the timm Vit backbone at model.backbone
        backbone = getattr(model, "backbone", model)
        blocks = getattr(backbone, "blocks", None)
        stem = getattr(backbone, "patch_embed", None)
    else:
        blocks = model.blocks
        stem = getattr(model, "patch_embed", None)

    if blocks is None:
        raise ValueError("Model has no `blocks` iteration — not a ViT.")

    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        # Disable fused SDPA so softmax attention weights are materialized
        if getattr(attn, "fused_attn", False):
            restored_models.append(attn)
            attn.fused_attn = False

        attn_drop = getattr(attn, "attn_drop", None)
        if attn_drop is None:
            continue

        def _hook(mod, ins, out, _store=captured):
            _store.append(ins[0].detach().cpu())

        handles.append(attn_drop.register_forward_hook(_hook))

    class _RolloutContext:
        def __init__(self, h, restored):
            self._handles = h
            self._restored = restored

        def __enter__(self):
            return captured

        def __exit__(self, *exc):
            for h in self._handles:
                h.remove()
            for a in self._restored:
                a.fused_attn = True
            return False

    return _RolloutContext(handles, restored_models)


def compute_rollout(
    attentions: List[torch.Tensor],
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    residual_mixing: float = 0.5,
) -> np.ndarray:
    """Compute the attention rollout map from a list of per-layer attention
    tensors (each: batch x heads x tokens x tokens).

    Steps (Abnar & Zuidema, 2020):
      1. Fuse attention heads per layer (mean, max, or min across heads).
      2. Add identity matrix I to account for residual/skip connections:
         A_hat = residual_mixing*A + (1-residual_mixing)*I
      3. Re-normalize rows to sum to 1.
      4. Recursively multiply across layers: rollout = A_hat_L @ ... @ A_hat_1.
      5. Extract the CLS token's row (attention paid to each patch),
         excluding the CLS-to-CLS entry, and reshape to the patch grid.

    `discard_ratio` optionally zeroes out the lowest-attention fraction of
    patches per layer before renormalizing, a common noise-reduction trick.

    Returns a (grid_h, grid_w) attention map, normalized to [0, 1].
    """
    if not attentions:
        raise ValueError("No attention tensors captured.")

    batch = attentions[0].shape[0]
    num_tokens = attentions[0].shape[-1]
    # 14x14 for vit_base_patch16_224: num_tokens = 197 = 1 CLS + 196 patches
    num_patches = num_tokens - 1
    grid_h = grid_w = int(round(num_patches ** 0.5))
    if grid_h * grid_w != num_patches:
        raise ValueError(f"num_patches {num_patches} is not a perfect square.")

    rollout = None
    for attn in attentions:
        # Fuse heads
        if head_fusion == "mean":
            fused = attn.mean(dim=1)
        elif head_fusion == "max":
            fused = attn.max(dim=1).values
        elif head_fusion == "min":
            fused = attn.min(dim=1).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        a = fused
        if discard_ratio > 0:
            # Zero out the lowest-attention fraction per row (noise reduction).
            k = int(discard_ratio * num_tokens)
            if k > 0 and k < num_tokens:
                row_sorted, _ = a.sort(dim=-1)  # ascending, per row
                thresh = row_sorted[:, :, k : k + 1]  # k-th smallest per row
                a = a * (a > thresh).float()

        # Add residual identity and renormalize rows
        a = residual_mixing * a + (1.0 - residual_mixing) * torch.eye(
            num_tokens, device=a.device
        ).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        rollout = a if rollout is None else torch.matmul(rollout, a)

    # Extract CLS row (index 0), exclude CLS-to-CLS column, reshape to grid
    cls_row = rollout[0, 0, 1:]
    heatmap = cls_row.reshape(grid_h, grid_w).numpy()

    # Normalize to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-12:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap


def batch_attention_rollout_heatmaps(
    model: nn.Module,
    inputs: torch.Tensor,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> np.ndarray:
    """Run forward passes with attention hooks attached, then compute rollout
    maps for every image in the batch.

    Returns an array of shape (N, grid_h, grid_w).
    """
    was_training = model.training
    model.eval()

    try:
        ctx = register_attention_hooks(model)
        with ctx as attentions:
            with torch.no_grad():
                _ = model(inputs)
        heatmaps = [
            compute_rollout(
                [a[i : i + 1] for a in attentions],
                discard_ratio=discard_ratio,
                head_fusion=head_fusion,
            )
            for i in range(inputs.shape[0])
        ]
    finally:
        if was_training:
            model.train()

    return np.stack(heatmaps, axis=0)