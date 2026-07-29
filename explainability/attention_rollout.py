"""Explainability Analysis (ViT branch): Attention Rollout.

Implements the Abnar & Zuidema (2020) attention-rollout method: recursively
multiplies attention matrices across transformer layers (with a residual
"identity" term to account for skip connections) to produce a single
image-level attention map showing which input patches the ViT ultimately
attended to for its prediction.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


def register_attention_hooks(model: nn.Module) -> List[torch.Tensor]:
    """Attach forward hooks to every transformer block's attention module to
    capture per-layer attention weight matrices (shape:
    batch x num_heads x num_patches+1 x num_patches+1, including the CLS token).

    Returns a list that will be populated with attention tensors during the
    next forward pass (caller must run a forward pass, then read this list).

    TODO: implementation depends on the exact timm ViT internals -- typically
    requires monkeypatching `Attention.forward` to store `attn` before the
    softmax@V step, since timm does not expose attention weights by default.
    """
    raise NotImplementedError("TODO: implement attention-capturing forward hooks for timm ViT")


def compute_rollout(attentions: List[torch.Tensor], discard_ratio: float = 0.0, head_fusion: str = "mean") -> np.ndarray:
    """Compute the attention rollout map from a list of per-layer attention
    tensors (each: batch x heads x tokens x tokens).

    Steps (Abnar & Zuidema, 2020):
      1. Fuse attention heads per layer (mean, max, or min across heads).
      2. Add identity matrix I to account for residual/skip connections:
         A_hat = 0.5*A + 0.5*I  (or another configured residual mixing ratio).
      3. Re-normalize rows to sum to 1.
      4. Recursively multiply across layers: rollout = A_hat_L @ ... @ A_hat_1.
      5. Extract the CLS token's row (attention paid to each patch),
         excluding the CLS-to-CLS entry, and reshape to the patch grid.

    `discard_ratio` optionally zeroes out the lowest-attention fraction of
    patches per layer before renormalizing, a common noise-reduction trick.

    Returns a (grid_h, grid_w) attention map, normalized to [0, 1].
    """
    raise NotImplementedError("TODO: implement attention rollout matrix multiplication chain")


def batch_attention_rollout_heatmaps(
    model: nn.Module,
    inputs: torch.Tensor,
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Convenience wrapper: run forward passes with attention hooks attached,
    then compute rollout maps for every image in the batch.

    Returns an array of shape (N, grid_h, grid_w), to be upsampled to the
    original image resolution before quantitative comparison in
    explainability/quantitative_xai.py.
    """
    raise NotImplementedError("TODO: implement batched attention rollout")
