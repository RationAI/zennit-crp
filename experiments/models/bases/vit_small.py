"""``vit_small_patch16_224.augreg_in21k_ft_in1k`` — 22 M-param ViT-S/16
pretrained on ImageNet-21k and fine-tuned on ImageNet-1k. The smallest
plain-ViT in our base registry; default backbone for finetune runs on
FunnyBirds, dSprites, and ColoredMNIST."""
from .base import Base


class ViTSmall(Base):
    timm_name = "vit_small_patch16_224.augreg_in21k_ft_in1k"
