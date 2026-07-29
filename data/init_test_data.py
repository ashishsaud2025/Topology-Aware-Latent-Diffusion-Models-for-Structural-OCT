"""Generate a small synthetic test dataset under data/test_raw/ for quick
end-to-end verification of the pipeline.

Run:
    python data/init_test_data.py

Creates:
    data/test_raw/
        NORMAL/   (10 synthetic 64x64 PNGs)
        CNV/      (10 synthetic 64x64 PNGs)
        DME/      (10 synthetic 64x64 PNGs)
"""
import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "test_raw"
CLASSES = ["NORMAL", "CNV", "DME"]
N_PER_CLASS = 10
IMAGE_SIZE = 64
SEED = 42

rng = np.random.RandomState(SEED)

print(f"Generating test dataset in: {OUTPUT_DIR}")
for cls in CLASSES:
    cls_dir = OUTPUT_DIR / cls
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N_PER_CLASS):
        # Random noise with gradient structure mimicking OCT slices
        img = rng.randn(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32) * 0.3
        gradient = np.linspace(-0.5, 0.5, IMAGE_SIZE).reshape(1, IMAGE_SIZE)
        img += gradient
        # Add some class-specific variation
        if cls == "CNV":
            # Bright spot (neovascularization)
            cx, cy = rng.randint(16, 48, size=2)
            img[cy-4:cy+4, cx-4:cx+4] += 1.0
        elif cls == "DME":
            # Diffuse thickening (band of higher intensity)
            img += 0.3 * np.sin(np.linspace(0, 4*np.pi, IMAGE_SIZE)).reshape(1, IMAGE_SIZE)
        # Normalize to [0, 255] uint8
        img -= img.min()
        img /= (img.max() + 1e-8)
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(str(cls_dir / f"{cls}_{i:04d}.jpeg"), img_uint8)
    print(f"  Created {N_PER_CLASS} images in {cls}/")

print("Done.")