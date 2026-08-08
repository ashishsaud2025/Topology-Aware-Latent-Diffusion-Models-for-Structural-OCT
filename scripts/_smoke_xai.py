"""Temporary smoke test for Grad-CAM + Attention Rollout on real checkpoints."""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainability.attention_rollout import batch_attention_rollout_heatmaps
from explainability.gradcam import compute_batch_gradcam_heatmaps
from models.classifiers import build_classifier

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_dir = Path("outputs/checkpoints")
x = torch.randn(4, 3, 224, 224, device=device)

# Grad-CAM on resnet50 (batch of 4 with single GradCAM instance)
m = build_classifier("resnet50", 4, pretrained=False).to(device)
m.load_state_dict(torch.load(ckpt_dir / "ratio0.00_proportional_resnet50_seed0.pt", map_location=device))
hms = compute_batch_gradcam_heatmaps(m, "resnet50", x)
print("gradcam resnet50 batch:", hms.shape, float(hms.min()), float(hms.max()))

# Grad-CAM on efficientnet_b0 (targeted)
m = build_classifier("efficientnet_b0", 4, pretrained=False).to(device)
m.load_state_dict(torch.load(ckpt_dir / "ratio0.00_proportional_efficientnet_b0_seed0.pt", map_location=device))
hms = compute_batch_gradcam_heatmaps(m, "efficientnet_b0", x, target_classes=torch.tensor([1, 2, 0, 3]))
print("gradcam efficientnet batch:", hms.shape, float(hms.min()), float(hms.max()))

# Attention rollout on vit_base (batch of 4)
m = build_classifier("vit_base", 4, pretrained=False).to(device)
m.load_state_dict(torch.load(ckpt_dir / "ratio0.00_proportional_vit_base_seed0.pt", map_location=device))
maps = batch_attention_rollout_heatmaps(m, x)
print("attention rollout vit batch:", maps.shape, float(maps.min()), float(maps.max()))