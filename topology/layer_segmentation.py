"""Retinal layer segmentation for OCT B-scans (Stage 2B component).

Maps an OCT B-scan (grayscale, z-score normalized) to a segmentation of 8
retinal layers, and produces the 7 binary boundary maps between consecutive
layers. This is the structural input for the persistent-homology based
topological validation.

The approach follows the literature (Yamauchi, Wu, & Okada, 2025) that
evaluates OCT synthesis quality *through* layer segmentation fidelity. Two
segmenter backends are provided:

  1. ``profile``  -- an A-scan intensity
     profile segmentation. OCT B-scans have a characteristic layered
     intensity structure (RNFL bright, IPL dark, RPE bright, etc.). This
     backend detects transitions in the column-wise (A-scan) intensity
     profile and is used as a default when no annotated segmentation model
     has been trained yet.

  2. ``unet``     -- a MONAI UNet segmenter that can be trained on the Duke
     DME dataset (110 annotated B-scans, 8 layers) and loaded from a
     checkpoint. This backend requires a Duke-trained checkpoint.

Both backends produce the same output contract: a label map of shape
(H, W) with integer values 0..8 (0=background/choroid, 1..8 = the eight
layers) plus the derived binary boundary maps.

Layer ordering (superior to inferior, i.e. top to bottom of the B-scan):
    1  ILM  Inner Limiting Membrane
    2  NFL  Nerve Fiber Layer
    3  GCL  Ganglion Cell Layer
    4  IPL  Inner Plexiform Layer
    5  INL  Inner Nuclear Layer
    6  OPL  Outer Plexiform Layer
    7  ONL  Outer Nuclear Layer
    8  RPE  Retinal Pigment Epithelium

The 7 boundaries are interfaces between consecutive layers:
    B1 = ILM/NFL   B2 = NFL/GCL   B3 = GCL/IPL   B4 = IPL/INL
    B5 = INL/OPL   B6 = OPL/ONL   B7 = ONL/RPE
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Layer names, superior -> inferior (= top -> bottom of B-scan)
LAYER_NAMES: List[str] = [
    "ILM_Inner_Limiting_Membrane",
    "NFL_Nerve_Fiber_Layer",
    "GCL_Ganglion_Cell_Layer",
    "IPL_Inner_Plexiform_Layer",
    "INL_Inner_Nuclear_Layer",
    "OPL_Outer_Plexiform_Layer",
    "ONL_Outer_Nuclear_Layer",
    "RPE_Retinal_Pigment_Epithelium",
]

# 7 boundaries between consecutive layers
BOUNDARY_NAMES: List[str] = [
    "B1_ILM_NFL",
    "B2_NFL_GCL",
    "B3_GCL_IPL",
    "B4_IPL_INL",
    "B5_INL_OPL",
    "B6_OPL_ONL",
    "B7_ONL_RPE",
]

# Segmentation result container

@dataclass
class SegmentationResult:
    """Output of a retinal layer segmentation for a single B-scan.

    Attributes:
        label_map: Integer label map of shape (H, W) with values in 0..8.
            Background/choroid = 0, layers 1..8 per LAYER_NAMES.
        boundaries: Dict mapping boundary name -> binary boundary map
            (H, W) bool array, True where the boundary between two layers is
            present. The boundary map is a 1-px thick edge between
            consecutive layer regions.
        source_image: The grayscale image that was segmented (original
            resolution, z-score normalized) when available.
    """
    label_map: np.ndarray
    boundaries: Dict[str, np.ndarray] = field(default_factory=dict)
    source_image: Optional[np.ndarray] = None

    def compute_boundaries(self) -> "SegmentationResult":
        """Derive the 7 binary boundary maps from `label_map`."""
        self.boundaries = {}
        for i, bname in enumerate(BOUNDARY_NAMES):
            # Boundary i is between layer (i) and layer (i+1)
            lower = self.label_map == (i + 1)
            upper = self.label_map == (i + 2)
            # A boundary pixel is a pixel of the lower layer that has a
            # 4-connected neighbor belonging to the upper layer.
            shifted_up = np.zeros_like(lower)
            shifted_up[:-1, :] = upper[1:, :]
            shifted_down = np.zeros_like(lower)
            shifted_down[1:, :] = upper[:-1, :]
            shifted_left = np.zeros_like(lower)
            shifted_left[:, :-1] = upper[:, 1:]
            shifted_right = np.zeros_like(lower)
            shifted_right[:, 1:] = upper[:, :-1]

            boundary = (
                lower
                & (shifted_up | shifted_down | shifted_left | shifted_right)
            )
            self.boundaries[bname] = boundary
        return self


# Intensity-profile segmenter (dependency-free default)
class IntensityProfileSegmenter:
    """Segments OCT B-scans using A-scan intensity profile analysis.

    OCT B-scans exhibit a characteristic vertically layered intensity
    structure. This segmenter works column-wise (per A-scan):

      1. Denoise each A-scan with a median filter.
      2. Locate the RPE (brightest sustained band); its top edge is the
         ONL/RPE boundary (B7).
      3. Detect the retina surface (dark vitreous -> bright RNFL) as the
         strongest dark-to-bright transition near the top of the A-scan
         (the ILM/NFL boundary).
      4. Greedily pick the 5 strongest gradient extrema between B1 and B7
         (with a minimum separation) as the internal boundaries B2..B6.
      5. Fit the detected boundary rows with a smooth polynomial across
         columns so the layer boundaries are continuous curves.

    This provides a layer-structure-aware topological signature
    even before a deep-learning segmentation model has been trained on
    Duke annotations. For production runs, prefer the ``UNetSegmenter``
    backend.
    """

    def __init__(
        self,
        n_layers: int = 8,
        median_kernel: int = 11,
        poly_order: int = 4,
    ) -> None:
        self.n_layers = n_layers
        self.median_kernel = median_kernel
        self.poly_order = poly_order

    # helpers 

    @staticmethod
    def _denoise_profile(profile: np.ndarray, kernel: int) -> np.ndarray:
        """Median filter a 1-D A-scan intensity profile."""
        k = kernel if kernel % 2 == 1 else kernel + 1
        p_u8 = np.clip(profile, 0, 255).astype(np.uint8).reshape(1, -1)
        return cv2.medianBlur(p_u8, k).reshape(-1).astype(np.float32)

    def _find_rpe_row(self, profile: np.ndarray) -> int:
        """Locate the RPE as the brightest sustained band in the lower half."""
        h = len(profile)
        lower_half = profile[h // 2:]
        # RPE is the strongest sustained (thick) bright band -> maximize the
        # product of intensity and local thickness (sum over a window).
        win = max(5, h // 50)
        scores = np.convolve(lower_half, np.ones(win) / win, mode="same")
        return int(np.argmax(scores) + h // 2)

    @staticmethod
    def _fit_smooth_boundary(
        rows: np.ndarray,
        width: int,
        poly_order: int,
    ) -> np.ndarray:
        """Fit a smooth polynomial to per-column boundary row estimates."""
        cols = np.arange(width)
        xs = np.arange(width)

        # Remove NaN estimates (columns where the boundary was not found)
        valid = ~np.isnan(rows)
        if valid.sum() < max(poly_order + 1, 4):
            # Fallback: flat boundary at the median row
            return np.full(width, int(np.nanmedian(rows)))

        coefs = np.polyfit(xs[valid], rows[valid], deg=poly_order)
        poly = np.poly1d(coefs)
        smooth = poly(xs)
        return np.clip(
            np.round(smooth), 0, rows.shape[0] - 1
        ).astype(int)

    def _detect_boundaries_in_profile(
        self, profile: np.ndarray, rpe_row: int
    ) -> List[int]:
        """Detect the 7 layer-boundary rows in one A-scan intensity profile.

        Strategy (per column):
          1. B7 = ONL/RPE: walk upward from the RPE center to its top edge.
          2. B1 = ILM/NFL: strongest dark-to-bright gradient near the top.
          3. B2..B6: 5 strongest gradient extrema between B1 and B7,
             greedily picked with a minimum separation.

        Returns exactly 7 absolute rows, or [] if detection fails so the
        caller can fall back to evenly-spaced boundaries.
        """
        H = len(profile)
        n_transitions = self.n_layers - 1  # 7

        # B7: top edge of the RPE band
        rpe_center = max(0, min(H - 1, int(rpe_row)))
        rpe_lo, rpe_hi = max(0, rpe_center - 4), min(H, rpe_center + 4)
        rpe_intensity = float(np.mean(profile[rpe_lo:rpe_hi]))
        threshold = rpe_intensity * 0.55
        b7 = rpe_center
        for r in range(rpe_center, max(0, rpe_center - int(H * 0.06)), -1):
            if profile[r] < threshold:
                b7 = r
                break

        # Search region between the retina surface and the RPE
        search_top = max(1, int(b7 - H * 0.70))
        if b7 - search_top < 14:
            return []
        sub = profile[search_top : b7 + 1]
        grad = np.gradient(sub)
        abs_grad = np.abs(grad)

        # B1: strongest dark-to-bright transition near the top
        top_zone = max(3, len(sub) // 10)
        b1_rel = int(np.argmax(grad[:top_zone]))
        b1 = search_top + b1_rel

        # B2..B6: strongest gradient extrema between B1 and B7
        min_sep = max(3, len(sub) // 40)
        lo = min(b1_rel + min_sep, len(sub) - 1)
        hi = max(lo + 1, len(sub) - 1)
        region_grad = abs_grad[lo:hi]
        if len(region_grad) < 5:
            return []

        from scipy.ndimage import maximum_filter1d

        local_max = region_grad == maximum_filter1d(
            region_grad, size=2 * min_sep + 1, mode="nearest"
        )
        local_max &= region_grad > 1e-6
        candidate_rel = np.where(local_max)[0] + lo
        if len(candidate_rel) == 0:
            return []

        strengths = abs_grad[candidate_rel]
        order = np.argsort(-strengths)
        picked: List[int] = []
        for idx in order:
            r = int(candidate_rel[idx])
            if all(abs(r - p) >= min_sep for p in picked):
                picked.append(r)
                if len(picked) == 5:  # B2..B6
                    break
        picked = [p for p in picked if abs((search_top + p) - b7) >= min_sep]
        if len(picked) < 5:
            return []

        boundaries = sorted([b1] + [search_top + p for p in picked] + [b7])
        return boundaries if len(boundaries) == n_transitions else []

    # main entry 

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment an OCT B-scan into 8 retinal layers.

        Args:
            image: Grayscale OCT B-scan as float32 (z-score normalized),
                shape (H, W).

        Returns:
            SegmentationResult with label_map (values 0..8) and derived
            binary boundary maps.
        """
        if image.ndim == 3:
            image = image.squeeze()
        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2-D grayscale image, got shape {image.shape}"
            )

        H, W = image.shape
        if np.max(image) <= 1.0 and np.min(image) >= -1.0:
            # Assume the image is normalized [0,1] -> convert to [0,255]
            img = (image - np.min(image)) / (np.ptp(image) + 1e-8) * 255.0
        else:
            # z-score normalized -> rescale to [0,255] for processing
            std = image.std()
            if std > 0:
                img = (image - image.mean()) / (std + 1e-8)
                img = (img - img.min()) / (np.ptp(img) + 1e-8) * 255.0
            else:
                img = np.full_like(image, 128.0)

        img = img.astype(np.float32)

        # Per-column boundary row estimates: boundaries[i][col] = row index
        # where boundary i occurs on column `col`.
        boundary_rows = np.full((self.n_layers - 1, W), np.nan, dtype=np.float64)

        for col in range(W):
            profile = self._denoise_profile(img[:, col], self.median_kernel)
            rpe_row = self._find_rpe_row(profile)

            # Primary: gradient-extrema based boundary detection
            detected = self._detect_boundaries_in_profile(profile, rpe_row)
            if len(detected) == self.n_layers - 1:
                for i, row in enumerate(detected):
                    boundary_rows[i, col] = row
                continue

            # Fallback: evenly-spaced boundaries.
            search_top = max(1, int(rpe_row - H * 0.62))  # NFL starts ~60% up
            sub_len = rpe_row - search_top + 1
            if sub_len < 8:
                continue
            fracs = np.linspace(0.05, 0.95, self.n_layers - 1)
            for i, frac in enumerate(fracs):
                rel = int(round(frac * (sub_len - 1)))
                boundary_rows[i, col] = search_top + rel

        # Fit smooth polynomial curves for each boundary
        label_map = np.zeros((H, W), dtype=np.uint8)
        for i in range(self.n_layers - 1, -1, -1):
            # Assign layer i+1 from boundary i (above) to boundary i+1 (below)
            if i == 0:
                top_rows = np.zeros(W, dtype=int)  # top of image
            else:
                top_rows = self._fit_smooth_boundary(
                    boundary_rows[i - 1], W, self.poly_order
                )
            if i == self.n_layers - 1:
                bottom_rows = np.full(W, H - 1, dtype=int)  # bottom of image
            else:
                bottom_rows = self._fit_smooth_boundary(
                    boundary_rows[i], W, self.poly_order
                )

            for col in range(W):
                r0 = max(0, int(top_rows[col]))
                r1 = min(H - 1, int(bottom_rows[col]))
                if r1 > r0:
                    label_map[r0 : r1 + 1, col] = i + 1

        result = SegmentationResult(label_map=label_map, source_image=image)
        return result.compute_boundaries()


# MONAI UNet segmenter backend (production, trained on Duke annotations)
class UNetSegmenter:
    """Retinal layer segmentation using a MONAI UNet trained on annotated OCT.

    Loads a checkpoint produced by training a UNet on the Duke DME dataset
    (110 B-scans with 8 manually annotated layer boundaries). Expects the
    checkpoint to contain a state_dict for a MONAI ``UNet`` with
    ``spatial_dims=2``, ``in_channels=1``, ``out_channels=9`` (background +
    8 layers).

    If no checkpoint is available, this backend raises an error so the
    pipeline can fall back to ``IntensityProfileSegmenter``.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._load_model()

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"UNet segmentation checkpoint not found: {self.checkpoint_path}"
            )
        try:
            from monai.networks.nets import UNet
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MONAI is required for the UNet segmenter backend."
            ) from exc

        import torch

        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=9,  # background + 8 layers
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
        ).to(self.device)

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Loaded UNet segmenter from {self.checkpoint_path}")

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment a single B-scan with the UNet.

        Args:
            image: Grayscale float32 image, shape (H, W).

        Returns:
            SegmentationResult.
        """
        import torch

        if image.ndim == 3:
            image = image.squeeze()
        H, W = image.shape

        x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)
        logits = self.model(x)
        label_map_np = (
            torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        )
        result = SegmentationResult(label_map=label_map_np, source_image=image)
        return result.compute_boundaries()


# Unified factory
@dataclass
class RetinalLayerSegmenter:
    """Unified interface for retinal layer segmentation.

    Args:
        backend: "profile" (dependency-free intensity profile heuristic) or
            "unet" (MONAI UNet trained on Duke annotations).
        checkpoint_path: Required when backend="unet".
        device: Torch device for the UNet backend.
    """

    backend: str = "profile"
    checkpoint_path: Optional[str | Path] = None
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.backend not in ("profile", "unet"):
            raise ValueError(
                f"Unknown segmenter backend '{self.backend}'. "
                "Use 'profile' or 'unet'."
            )
        self._impl = (
            IntensityProfileSegmenter()
            if self.backend == "profile"
            else UNetSegmenter(self.checkpoint_path, self.device)
        )

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment an OCT B-scan into 8 retinal layers.

        Args:
            image: Grayscale OCT B-scan (H, W), z-score normalized.

        Returns:
            SegmentationResult with label_map and boundary maps.
        """
        return self._impl.segment(image)