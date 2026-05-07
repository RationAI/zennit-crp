"""``vit_base_patch16_224`` — the standard timm ViT-Base/16 trained on
ImageNet-1k. Smaller / faster than DINOv3 ViT-L; useful for quick
explainability experiments."""
from .base import Base


class ViTBase(Base):
    timm_name = "vit_base_patch16_224"
