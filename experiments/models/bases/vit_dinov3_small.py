"""``vit_small_patch16_dinov3.lvd1689m`` — DINOv3 ViT-S/16 (Siméoni et
al. 2025) via timm. 21 M params; 12 blocks × 6 heads × head_dim 64,
embed dim 384; 5 prefix tokens (1 cls + 4 register); RoPE, no
learned pos-embed interpolation constraint; default input 256².

Backbone for the register-token study at small scale — same frozen
self-supervised weights as the large variant (LVD-1689M pretrain),
classifier heads trained separately."""
from .base import Base


class DinoV3Small(Base):
    timm_name = "vit_small_patch16_dinov3.lvd1689m"
