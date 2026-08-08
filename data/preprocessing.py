"""Stage 1 of the pipeline: Data Preprocessing & Class Analysis.

Responsibilities:
  - Load the raw real OCT dataset (e.g. Kermany et al. OCT2017, OCT-C8, or a
    private clinical dataset).
  - Standardize image format, resolution, and intensity normalization.
  - Perform a fixed, stratified train/val/test split. The TEST split must
    remain 100% real and untouched by synthetic data for the entire study
    (this is the "Fixed Real Test Set" referenced later in evaluation).
  - Quantify class imbalance to inform Factor B (distribution strategy) and
    to decide which classes are "minority" for the minority_only strategy.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClassDistribution:
    """Per-class sample counts for a given data split."""
    counts: Dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def minority_classes(self) -> List[str]:
        """Classes below the mean class frequency; used to target Factor B
        `minority_only` synthetic allocation."""
        if not self.counts:
            return []
        mean_count = self.total / len(self.counts)
        return [c for c, n in self.counts.items() if n < mean_count]

    def imbalance_ratio(self) -> float:
        """Majority:minority ratio, a standard imbalance severity metric."""
        if not self.counts:
            return 1.0
        return max(self.counts.values()) / max(min(self.counts.values()), 1)


def _extract_patient_id(filename: str) -> Optional[str]:
    """Try to extract a patient/subject identifier from an image filename.

    Supports common OCT dataset naming conventions:
      - OCT2017:  <patient_id>-<image_id>.jpeg  (e.g. "10234-1.jpeg")
      - OCT2017 (class-prefixed): "CNV-1016042-1.jpeg" -> patient=1016042
      - Custom:   <patient_id>_<something>.png, PAT_<id>_<image>.png
      - Fallback: use the filename stem itself as a unique patient id.

    Returns a string patient identifier, or None if the filename is malformed.
    """
    stem = Path(filename).stem

    # Pattern 0: Kermany OCT2017 with class prefix "CNV-1016042-1.jpeg"
    # -> patient=1016042  (class name prefix [A-Za-z]+, then patient-image)
    m = re.match(r"^[A-Za-z]+-(\d+)-(\d+)$", stem)
    if m:
        return m.group(1)

    # Pattern 1: OCT2017-style "12345-2.jpeg" -> patient=12345
    m = re.match(r"^(\d+)-(\d+)$", stem)
    if m:
        return m.group(1)

    # Pattern 2: "PAT_<id>_<image>" or "SUBJECT_<id>_<image>"
    m = re.match(r"^(?:PAT|SUBJECT|SUBJ|patient)_(\d+)_", stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # Pattern 3: "ID-<id>_<image>" or "ID<id>_<image>"
    m = re.match(r"^ID[_-]?(\d+)_", stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # Fallback: treat the whole stem as patient id (each file unique)
    return stem


def subset_patient_level(
    index_df: pd.DataFrame,
    max_samples_per_class: int,
    seed: int = 42,
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> pd.DataFrame:
    """Select a deterministic per-class subset while keeping ALL images from
    each selected patient (patient-level grouping, so the downstream
    patient-level split never leaks a patient across train/val/test).

    Used to limit the full Kermany OCT2017 dataset (~83k images) to a
    PC-friendly subset (Option A), e.g. ~1,000 images per class.

    Selection algorithm (per class):
      1. Group images by patient, count images per patient.
      2. Shuffle patients deterministically using `seed`.
      3. Greedily pick patients (in shuffled order) until the accumulated
         image count reaches `max_samples_per_class` or patients run out.

    Returns:
        A filtered DataFrame with the same columns as `index_df`.
    """
    rng = np.random.RandomState(seed)
    selected_frames = []

    for class_name in sorted(index_df[label_col].unique()):
        class_df = index_df[index_df[label_col] == class_name]
        target = min(max_samples_per_class, len(class_df))

        # Group by patient (insertion-order preserved) and count images
        patient_order = list(class_df.groupby(patient_col, sort=False).groups.keys())
        patient_order = [pid for pid in patient_order if pid is not None]
        rng.shuffle(patient_order)

        picked_frames = []
        n_picked_images = 0
        for pid in patient_order:
            if n_picked_images >= target:
                break
            patient_df = class_df[class_df[patient_col] == pid]
            n_picked_images += len(patient_df)
            picked_frames.append(patient_df)

        if picked_frames:
            selected_frames.append(pd.concat(picked_frames))
            n_patients = len(
                pd.concat(picked_frames)[patient_col].unique()
            )
            logger.info(
                f"  Subset '{class_name}': selected {n_picked_images} images "
                f"({n_patients} patients) targeting max {target}."
            )

    if not selected_frames:
        return index_df

    subset_df = pd.concat(selected_frames, ignore_index=True)
    logger.info(
        f"Subset selected {len(subset_df)} images across "
        f"{subset_df[patient_col].nunique()} patients "
        f"(max_samples_per_class={max_samples_per_class})."
    )
    return subset_df


def load_raw_dataset_index(
    raw_dir: str | Path,
    classes: List[str],
    max_samples_per_class: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Walk `raw_dir` and build an index DataFrame with columns
    [filepath, label, patient_id].

    Expects the dataset folder structure:
        raw_dir/
            NORMAL/
                image1.jpeg
                image2.jpeg
                ...
            CNV/
                image1.jpeg
                ...
            DME/
                ...
            DRUSEN/
                ...

    Patient IDs are extracted from filenames using `_extract_patient_id`.
    If a file cannot be assigned a patient_id, it gets a unique one.

    If `max_samples_per_class` is set, the index is reduced to a
    deterministic per-class subset via `subset_patient_level` (patient-level
    grouping to avoid leakage) before returning. This is the Option A
    mechanism for running the full Kermany dataset on modest hardware.

    Returns a DataFrame with columns:
        filepath : str   (absolute path to the image file)
        label    : str   (class name, e.g. "NORMAL", "CNV", "DME", "DRUSEN")
        patient_id : str (anonymized patient identifier for group-level splits)
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    records = []
    for class_name in classes:
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            logger.warning(f"Class directory not found: {class_dir}, skipping.")
            continue

        for fpath in sorted(class_dir.iterdir()):
            if fpath.suffix.lower() not in (".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".tif"):
                continue

            patient_id = _extract_patient_id(fpath.name)
            if patient_id is None:
                patient_id = f"unknown_{class_name}_{len(records)}"

            records.append({
                "filepath": str(fpath.resolve()),
                "label": class_name,
                "patient_id": patient_id,
            })

    if not records:
        raise FileNotFoundError(
            f"No images found under {raw_dir} for classes {classes}. "
            "Check your config's data.raw_dir and data.classes."
        )

    df = pd.DataFrame(records)
    logger.info(
        f"Loaded {len(df)} images across {df['label'].nunique()} classes "
        f"({df['patient_id'].nunique()} unique patient IDs)."
    )

    # Option A: reduce the full dataset to a PC-friendly subset
    if max_samples_per_class and max_samples_per_class > 0:
        df = subset_patient_level(df, max_samples_per_class, seed=seed)

    return df


def compute_class_distribution(index_df: pd.DataFrame, label_col: str = "label") -> ClassDistribution:
    """Compute per-class counts from an indexed dataset DataFrame."""
    counts = index_df[label_col].value_counts().to_dict()
    return ClassDistribution(counts=counts)


def stratified_patient_level_split(
    index_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> Dict[str, pd.DataFrame]:
    """Split at the PATIENT level (not image level) to avoid leakage, while
    preserving class stratification.

    Strategy:
      1. Aggregate patients: for each patient, assign the majority class
         among their images (or the class if they have only one).
      2. Stratified split of patients into train/val/test.
      3. Map the patient-level split back to the image-level DataFrame.

    The "test" split is the FIXED REAL TEST SET used throughout evaluation;
    it must never receive synthetic images and must remain identical across
    every experimental cell for fair comparison.

    Returns a dict: {"train": df, "val": df, "test": df}
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "splits must sum to 1.0"

    # Deduplicate patients and assign each a majority class label
    patient_class = (
        index_df.groupby(patient_col)[label_col]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    patient_class.columns = [patient_col, "patient_label"]

    patients = patient_class[patient_col].values
    patient_labels = patient_class["patient_label"].values

    # First split: separate test set
    test_size = test_frac / (train_frac + val_frac + test_frac)
    trainval_patients, test_patients, trainval_labels, _ = train_test_split(
        patients,
        patient_labels,
        test_size=test_size,
        stratify=patient_labels,
        random_state=seed,
    )

    # Second split: separate train and val from the remaining
    val_size = val_frac / (train_frac + val_frac)
    train_patients, val_patients = train_test_split(
        trainval_patients,
        test_size=val_size,
        stratify=trainval_labels,
        random_state=seed + 1,  # different seed to avoid correlation
    )

    # Map patient-level splits back to image-level DataFrames
    train_mask = index_df[patient_col].isin(set(train_patients))
    val_mask = index_df[patient_col].isin(set(val_patients))
    test_mask = index_df[patient_col].isin(set(test_patients))

    # Handle edge case: some patients may have been dropped if they only had
    # images in a minority class that got collapsed — fall back to label split
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        logger.warning(
            "Patient-level split produced empty split(s); falling back to image-level split."
        )
        # Image-level stratified train/val/test split as fallback
        train_idx, temp_idx, _, temp_labels = train_test_split(
            np.arange(len(index_df)),
            index_df[label_col].values,
            test_size=(val_frac + test_frac),
            stratify=index_df[label_col].values,
            random_state=seed,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_frac / (val_frac + test_frac),
            stratify=temp_labels,
            random_state=seed + 1,
        )
        splits = {
            "train": index_df.iloc[train_idx].reset_index(drop=True),
            "val": index_df.iloc[val_idx].reset_index(drop=True),
            "test": index_df.iloc[test_idx].reset_index(drop=True),
        }
    else:
        splits = {
            "train": index_df[train_mask].reset_index(drop=True),
            "val": index_df[val_mask].reset_index(drop=True),
            "test": index_df[test_mask].reset_index(drop=True),
        }

    for name, df in splits.items():
        dist = compute_class_distribution(df)
        logger.info(
            f"  {name}: {len(df)} images, {df[patient_col].nunique()} patients, "
            f"class counts: {dist.counts}"
        )

    return splits


def preprocess_images(
    index_df: pd.DataFrame,
    output_dir: str | Path,
    image_size: int,
    apply_clahe: bool = True,
    normalize: bool = True,
) -> pd.DataFrame:
    """Resize, normalize (z-score), and optionally apply CLAHE to raw images,
    saving preprocessed copies into `output_dir/<label>/`.

    For OCT images, the standard preprocessing pipeline is:
      1. Convert to grayscale if needed
      2. Resize to `image_size x image_size`
      3. Optional: CLAHE for local contrast enhancement (common in OCT)
      4. Z-score normalization (mean=0, std=1 per image)
      5. Save as PNG for lossless storage

    Note: We keep all images as single-channel (grayscale). Conversion to
    3-channel RGB is done in `get_default_transforms()` in dataset.py to
    match the pretrained model input expectations.

    Returns an updated index DataFrame with `filepath` pointing to the
    preprocessed files.
    """
    output_dir = Path(output_dir)
    records = []

    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Preprocessing images"):
        src_path = Path(row["filepath"])
        label = row["label"]

        # Read image
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Could not read image: {src_path}, skipping.")
            continue

        # Resize
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # CLAHE (local contrast enhancement, standard for OCT)
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

        # Z-score normalization
        if normalize:
            img = img.astype(np.float32)
            mean = img.mean()
            std = img.std()
            if std > 0:
                img = (img - mean) / std
            else:
                img = np.zeros_like(img)
        else:
            img = img.astype(np.float32) / 255.0

        # Save preprocessed image.
        # NOTE: must use .tiff (not .png) for float32 — OpenCV's PNG encoder
        # silently falls back to uint8 and clamps z-score values to [0,255],
        # corrupting the data. TIFF supports float32 losslessly.
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{src_path.stem}_processed.tiff"
        out_path = label_dir / out_name
        cv2.imwrite(str(out_path), img.astype(np.float32))

        records.append({
            "filepath": str(out_path.resolve()),
            "label": label,
            "patient_id": row.get("patient_id", f"unknown_{label}_{len(records)}"),
            "is_synthetic": False,
        })

    if not records:
        raise RuntimeError("No images were successfully preprocessed. Check your dataset.")

    processed_df = pd.DataFrame(records)
    logger.info(
        f"Preprocessed {len(processed_df)} images to {output_dir} "
        f"(size={image_size}x{image_size}, clahe={apply_clahe}, z-score={normalize})."
    )
    return processed_df


def run_class_analysis_report(splits: Dict[str, pd.DataFrame], output_path: str | Path) -> None:
    """Generate and persist a class-balance report (counts, imbalance ratio,
    minority classes) per split, to be consulted when configuring
    Factor B (distribution strategy) in configs/config.yaml.
    """
    rows = []
    for split_name, df in splits.items():
        dist = compute_class_distribution(df)
        row = {
            "split": split_name,
            "total": dist.total,
            "imbalance_ratio": round(dist.imbalance_ratio(), 4),
            "minority_classes": ",".join(dist.minority_classes),
        }
        row.update(dist.counts)
        rows.append(row)

    report_df = pd.DataFrame(rows)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(str(out_path), index=False)
    logger.info(f"Class analysis report written to {output_path}")
    print("\n--- Class Analysis Report ---")
    print(report_df.to_string(index=False))


def save_splits_to_csv(
    splits: Dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Persist each split in `splits` to `output_dir/<split_name>.csv` so that
    downstream stages (dataset_builder, training) can load them independently.

    Args:
        splits: Dict of split_name -> DataFrame with columns [filepath, label, ...]
        output_dir: Directory to write the CSV files into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in splits.items():
        path = output_dir / f"{split_name}.csv"
        df.to_csv(str(path), index=False)
        logger.debug(f"Saved split {split_name} ({len(df)} rows) to {path}")
    logger.info(f"Saved {len(splits)} split CSV(s) to {output_dir}")
