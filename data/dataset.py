"""PyTorch Dataset / DataLoader definitions for real and synthetic OCT images.

Used both by the generative fine-tuning stage (real images only) and by the
classification stage (mixed real + synthetic, per experimental cell).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class OCTImageDataset(Dataset):
    """Generic OCT image dataset backed by an index DataFrame with columns
    [filepath, label, is_synthetic].

    `is_synthetic` lets downstream code (e.g. quantitative XAI, ablations)
    trace back whether a given sample was real or generated.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        class_to_idx: dict,
        transform: Optional[Callable] = None,
        filepath_col: str = "filepath",
        label_col: str = "label",
        is_synthetic_col: str = "is_synthetic",
    ):
        self.df = index_df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.filepath_col = filepath_col
        self.label_col = label_col
        self.is_synthetic_col = is_synthetic_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # TODO: implement actual image loading, e.g. via PIL.Image.open
        # image = Image.open(row[self.filepath_col]).convert("L")  # OCT is grayscale
        raise NotImplementedError("TODO: implement image loading + transform application")

    @classmethod
    def from_experimental_cell(
        cls,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        ratio: float,
        distribution_strategy: str,
        class_to_idx: dict,
        transform: Optional[Callable] = None,
        seed: int = 0,
    ) -> "OCTImageDataset":
        """Build the training dataset for one experimental cell by mixing
        `real_df` with a sampled subset of `synthetic_df` according to
        (`ratio`, `distribution_strategy`).

        This is the dataset-construction hook consumed by
        experiment/dataset_builder.py — kept here so the mixing logic lives
        next to the Dataset class it feeds.
        """
        # TODO: delegate actual sampling logic to
        # experiment.dataset_builder.build_mixed_index(...) and wrap result here.
        raise NotImplementedError("TODO: implement real+synthetic mixing per (ratio, strategy)")


def get_default_transforms(image_size: int, train: bool):
    """Return the torchvision/albumentations transform pipeline.

    TODO: define real vs. synthetic-aware augmentation. Keep test-set
    transforms identical across all experimental cells (only resize +
    normalize, no augmentation) to guarantee a fair, fixed evaluation set.
    """
    raise NotImplementedError("TODO: implement transform pipelines")
