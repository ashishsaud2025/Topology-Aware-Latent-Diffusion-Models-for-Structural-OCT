"""PyTorch Dataset / DataLoader definitions for real and synthetic OCT images.

Used both by the generative fine-tuning stage (real images only) and by the
classification stage (mixed real + synthetic, per experimental cell).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supported image file extensions
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def _to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert single-channel (1, H, W) image to 3-channel (3, H, W) by
    replicating the single channel, since all pretrained models expect
    3-channel RGB input."""
    if img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img


def get_default_transforms(image_size: int, train: bool) -> T.Compose:
    """Return the torchvision transform pipeline.

    For training:
      - Resize to image_size
      - Random horizontal flip
      - Convert to tensor (scales from [0,1] or z-score normalized float32)
      - Convert grayscale to 3-channel RGB

    For validation/test:
      - Resize to image_size
      - Convert to tensor
      - Convert grayscale to 3-channel RGB

    Important: The test-set transforms must be IDENTICAL across all
    experimental cells (only resize + to_tensor + rgb, no augmentation)
    to guarantee a fair, fixed evaluation set.

    The transforms assume input images are already preprocessed by
    `preprocessing.preprocess_images()` which outputs z-score normalized
    float32 images. If loading raw images, they should be normalized
    inside the Dataset's __getitem__.
    """
    if train:
        return T.Compose([
            T.Resize((image_size, image_size), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(_to_rgb),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size), antialias=True),
            T.Lambda(_to_rgb),
        ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

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
        image_size: int = 224,
    ):
        self.df = index_df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.filepath_col = filepath_col
        self.label_col = label_col
        self.is_synthetic_col = is_synthetic_col
        self.image_size = image_size

        # Ensure is_synthetic column exists (default False for real data)
        if self.is_synthetic_col not in self.df.columns:
            self.df[self.is_synthetic_col] = False

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        """Return (image_tensor, class_index, is_synthetic) for the given index.

        Images loaded from disk are expected to be:
          - Float32 grayscale (1, H, W) or (H, W) arrays, z-score normalized
          - Or uint8 [0,255] grayscale images (will be normalized to [0,1])

        The transform pipeline handles grayscale->RGB conversion.
        """
        row = self.df.iloc[idx]

        # Load image
        img = self._load_image(row[self.filepath_col])

        # Get label index
        label_str = row[self.label_col]
        label_idx = self.class_to_idx.get(label_str, -1)
        if label_idx == -1:
            raise KeyError(
                f"Label '{label_str}' not found in class_to_idx mapping. "
                f"Available classes: {list(self.class_to_idx.keys())}"
            )

        # Get synthetic flag
        is_synthetic = bool(row.get(self.is_synthetic_col, False))

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, label_idx, is_synthetic

    def _load_image(self, filepath: str) -> torch.Tensor:
        """Load an image from disk and return as a normalized tensor.

        Handles:
          - Float32 PNGs saved by preprocessing.py (z-score normalized)
          - Standard uint8 JPEG/PNG images
          - Single-channel (grayscale) images
        """
        path = Path(filepath)
        ext = path.suffix.lower()

        if ext not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {ext}")

        # Read image
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        # Handle different depths and channels
        if img.dtype == np.uint16:
            img = (img / 65535.0).astype(np.float32)
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        # else assume float32 already (from preprocessing pipeline)

        # Ensure single-channel grayscale
        if img.ndim == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        # Add channel dimension: (H, W) -> (1, H, W)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        # Resize if needed (should already be done by preprocessing, but
        # this handles edge cases where raw images are loaded directly)
        if img.shape[1] != self.image_size or img.shape[2] != self.image_size:
            img_resized = np.zeros((1, self.image_size, self.image_size), dtype=img.dtype)
            resized = cv2.resize(
                img[0], (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
            img_resized[0] = resized
            img = img_resized

        return torch.from_numpy(img)

    @classmethod
    def from_experimental_cell(
        cls,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        ratio: float,
        distribution_strategy: str,
        class_to_idx: dict,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        seed: int = 0,
    ) -> "OCTImageDataset":
        """Build the training dataset for one experimental cell by mixing
        `real_df` with a sampled subset of `synthetic_df` according to
        (`ratio`, `distribution_strategy`).

        This is the dataset-construction hook consumed by
        experiment/dataset_builder.py — the mixing logic is delegated
        to `allocate_synthetic_counts` from `experiment.dataset_builder`.
        """
        # Import here to avoid circular dependency at import time
        from experiment.dataset_builder import allocate_synthetic_counts

        real_counts = real_df["label"].value_counts().to_dict()
        synth_counts = allocate_synthetic_counts(
            real_counts, ratio, distribution_strategy
        )

        # Sample synthetic images per class
        sampled_synth_frames = []
        for class_name, n in synth_counts.items():
            if n <= 0:
                continue
            class_pool = synthetic_df[synthetic_df["label"] == class_name]
            if len(class_pool) == 0:
                logger.warning(
                    f"No synthetic images available for class '{class_name}'. Skipping."
                )
                continue
            if n > len(class_pool):
                logger.warning(
                    f"Requested {n} synthetic images for '{class_name}' but only "
                    f"{len(class_pool)} available. Using all available."
                )
                n = len(class_pool)
            sampled = class_pool.sample(n=n, random_state=seed)
            sampled_synth_frames.append(sampled)

        # Combine real + synthetic
        if sampled_synth_frames:
            mixed_df = pd.concat([real_df] + sampled_synth_frames, ignore_index=True)
        else:
            mixed_df = real_df.copy()

        return cls(
            index_df=mixed_df,
            class_to_idx=class_to_idx,
            transform=transform,
            image_size=image_size,
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def build_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int = 8,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Build standard DataLoaders for train/val/test.

    Args:
        train_dataset: Training dataset (may include synthetic images).
        val_dataset: Validation dataset (real images only).
        test_dataset: Fixed real test set (real images only, never augmented).
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader worker processes.

    Returns:
        Dict with keys "train", "val", "test" mapping to DataLoaders.
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}