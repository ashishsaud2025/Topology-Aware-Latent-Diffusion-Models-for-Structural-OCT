"""Temporary probe: verify timm ViT attention capture mechanics."""
import torch
import timm

m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=4).eval()
a = m.blocks[0].attn
print("fused_attn before:", a.fused_attn)
a.fused_attn = False
captured = []
h = a.attn_drop.register_forward_hook(
    lambda mod, ins, out: captured.append(ins[0].detach())
)
x = torch.randn(1, 197, 768)
with torch.no_grad():
    y = a(x)
print("captured:", len(captured))
if captured:
    print("shape:", tuple(captured[0].shape))
    print("rowsums close to 1:", torch.allclose(captured[0].sum(-1), torch.ones_like(captured[0].sum(-1)), atol=1e-3))
h.remove()