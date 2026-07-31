"""Train the Duke DME retinal layer segmentation UNet (Stage 2B production backend).

Usage:
    python scripts/train_duke_segmentation.py \
        --data-dir data_raw/duke_dme \
        --out models/segmentation_unet.pt \
        --epochs 100

The Duke DME dataset provides 110 annotated OCT B-scans with 8 manually
labeled retinal layer boundaries (Chiu et al.). Each B-scan should be paired
with an 8-bit label map (0=background/choroid, 1..8 = ILM, NFL, GCL, IPL,
INL, OPL, ONL, RPE) stored as .npy or .png with a matching basename.

Expected directory layout:
    data_raw/duke_dme/
        images/         # OCT B-scans (grayscale .png/.tif)
        labels/         # label maps (.npy preferred) with matching basenames
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.logging_utils import get_logger
from utils.seed import set_global_seed

logger = get_logger("train_duke_segmentation")


class DukeSegmentationDataset(Dataset):
    """Pairs of (OCT B-scan, 8-layer label map) for the Duke DME dataset."""

    def __init__(self, data_dir: str | Path, image_ext: str = ".png") -> None:
        data_dir = Path(data_dir)
        self.image_dir = data_dir / "images"
        self.label_dir = data_dir / "labels"
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected Duke data at {data_dir} with images/ and labels/ subdirs"
            )

        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(self.image_dir.glob(f"*{image_ext}")):
            label_path = self.label_dir / f"{img_path.stem}.npy"
            if not label_path.exists():
                label_path = self.label_dir / f"{img_path.stem}{image_ext}"
            if label_path.exists():
                self.samples.append((img_path, label_path))
        if not self.samples:
            raise FileNotFoundError(
                f"No paired images/labels found in {data_dir} "
                "(labels can be .npy or same-extension .png)"
            )
        logger.info(f"Loaded {len(self.samples)} Duke DME segmentation samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label_path = self.samples[idx]
        img = np.load(img_path) if img_path.suffix == ".npy" else _load_image(img_path)
        label = (
            np.load(label_path)
            if label_path.suffix == ".npy"
            else _load_image(label_path, as_label=True)
        )
        if img.ndim == 3:
            img = img.squeeze()
        if label.ndim == 3:
            label = label.squeeze()
        # Normalize image to [0,1]; labels are LongTensor class ids 0..8
        img = (img - img.min()) / (np.ptp(img) + 1e-8)
        return (
            torch.from_numpy(img).float().unsqueeze(0),  # (1, H, W)
            torch.from_numpy(label.astype(np.int64)),     # (H, W)
        )


def _load_image(path: Path, as_label: bool = False) -> np.ndarray:
    import cv2

    if as_label:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # Map unique pixel values to 0..8 by value rank (background=0 first).
        unique = np.unique(arr)
        mapping = {v: i for i, v in enumerate(sorted(unique))}  # background 0
        mapped = np.vectorize(mapping.get)(arr).astype(np.int64)
        return np.clip(mapped, 0, 8)
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return arr.astype(np.float32)


def build_unet(device: torch.device):
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=9,  # background + 8 layers
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Duke UNet on {device}")

    dataset = DukeSegmentationDataset(args.data_dir, image_ext=args.image_ext)
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_unet(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=Path(args.data_dir) / "runs")
        except ImportError:
            logger.warning("tensorboard not installed; skipping SummaryWriter")
    best_dice = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation: mean Dice over classes 1..8
        model.eval()
        all_dice = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = torch.argmax(model(x), dim=1)
                for c in range(1, 9):
                    p, t = pred == c, y == c
                    inter = (p & t).sum().float()
                    denom = p.sum() + t.sum() + 1e-8
                    all_dice.append((2 * inter / denom).item())
        mean_dice = float(np.mean(all_dice))
        avg_loss = total_loss / max(n_batches, 1)
        logger.info(
            f"Epoch {epoch+1}: loss={avg_loss:.4f} val_dice={mean_dice:.4f}"
        )
        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("val/dice", mean_dice, epoch)

        if mean_dice > best_dice:
            best_dice = mean_dice
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_dice": mean_dice},
                out_path,
            )
            logger.info(f"Saved best checkpoint ({best_dice:.4f}) -> {out_path}")
        scheduler.step()

    logger.info(f"Training complete. Best validation Dice = {best_dice:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Duke DME layer segmentation UNet")
    parser.add_argument("--data-dir", required=True, help="Duke DME dataset root")
    parser.add_argument("--out", default="models/segmentation_unet.pt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-ext", default=".png", help="B-scan file extension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()