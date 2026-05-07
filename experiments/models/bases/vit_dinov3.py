"""``vit_large_patch16_dinov3`` — DINOv3 ViT-L/16 (Darcet et al. 2024)
via timm. 304 M params; 24 blocks × 16 heads × head_dim 64; 5 prefix
tokens (1 cls + 4 register).

This is the backbone the unfolded-attention CRP walkthrough uses. The
unfolded-attention substitution is applied at composite-context-entry
time, not here — the backbone is the standard timm model."""
from .base import Base


class DinoV3(Base):
    timm_name = "vit_large_patch16_dinov3"
