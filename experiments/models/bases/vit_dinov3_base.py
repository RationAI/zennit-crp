"""``vit_base_patch16_dinov3.lvd1689m`` — DINOv3 ViT-B/16 (Siméoni et
al. 2025) via timm. 86 M params; 12 blocks × 12 heads × head_dim 64,
embed dim 768; 5 prefix tokens (1 cls + 4 register); RoPE; default
input 256².

Backbone for the register-token study at base scale. A public
ImageNet-1k linear head exists for this backbone (canvit
``dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe``; see
research/registers/dinov3_checkpoints.md)."""
from .base import Base


class DinoV3Base(Base):
    timm_name = "vit_base_patch16_dinov3.lvd1689m"
