"""Concrete ``ModelDataset`` pairs — each composes a zoo model with a dataset.

One class per journal ``(model, dataset)`` pair. ``_setup_model`` just wraps the
matching :mod:`experiments.models.zoo` class (source of truth for architecture +
weight provenance); the dataset mixin supplies loading + class labels.
"""
from __future__ import annotations

from experiments.models.zoo import (
    ColoredMnistViTSmall,
    DspritesViTSmall,
    FunnyBirdsDinoV3Small,
    FunnyBirdsViTSmall,
    ImagenetDinoV3Base,
    ImagenetViTBase,
    ImagenetViTBaseTorchvision,
)

from .model_dataset import (
    ColoredMnistMixin,
    DspritesMixin,
    FunnyBirdsMixin,
    ImagenetMixin,
    ModelDataset,
)
from .names_paths import (
    M_VIT_BASE,
    M_VIT_BASE_TORCHVISION,
    M_VIT_DINOV3_BASE,
    M_VIT_DINOV3_SMALL,
    M_VIT_SMALL,
)


class VitSmallFunnyBirds(FunnyBirdsMixin, ModelDataset):
    """Finetuned ViT-S/16 + linear head on clean FunnyBirds."""
    MODEL = M_VIT_SMALL

    def _setup_model(self) -> None:
        self.model = FunnyBirdsViTSmall(device=self.device, **self._ckpt_kw())
        self.backbone = self.model.backbone
        self.num_classes = self.model.num_classes
        self.label = "ViT-S/16 · FunnyBirds"


class VitDinoV3SmallFunnyBirds(FunnyBirdsMixin, ModelDataset):
    """Finetuned DINOv3-S/16 (+reg) + linear head on clean FunnyBirds."""
    MODEL = M_VIT_DINOV3_SMALL

    def _setup_model(self) -> None:
        self.model = FunnyBirdsDinoV3Small(device=self.device, **self._ckpt_kw())
        self.backbone = self.model.backbone
        self.num_classes = self.model.num_classes
        self.label = "DINOv3-S/16 (+reg, finetuned) · FunnyBirds"


class VitSmallDsprites(DspritesMixin, ModelDataset):
    """Finetuned ViT-S/16 + linear head on dSprites (shape target)."""
    MODEL = M_VIT_SMALL

    def _setup_model(self) -> None:
        self.model = DspritesViTSmall(device=self.device, **self._ckpt_kw())
        self.backbone = self.model.backbone
        self.num_classes = self.model.num_classes
        self.label = "ViT-S/16 · dSprites"


class VitSmallColoredMnist(ColoredMnistMixin, ModelDataset):
    """Finetuned ViT-S/16 + linear head on colored-MNIST."""
    MODEL = M_VIT_SMALL

    def _setup_model(self) -> None:
        self.model = ColoredMnistViTSmall(device=self.device, **self._ckpt_kw())
        self.backbone = self.model.backbone
        self.num_classes = self.model.num_classes
        self.label = "ViT-S/16 · ColoredMNIST"


class VitBaseImagenet(ImagenetMixin, ModelDataset):
    """Off-the-shelf timm ViT-B/16, ImageNet-1k."""
    MODEL = M_VIT_BASE

    def _setup_model(self) -> None:
        self.model = ImagenetViTBase(device=self.device)
        self.backbone = self.model.backbone
        self.num_classes = self.model.num_classes
        self.label = "ViT-B/16 · ImageNet"


class VitBaseTorchvisionImagenet(ImagenetMixin, ModelDataset):
    """timm ViT-B/16 skeleton carrying torchvision IMAGENET1K_V1 weights."""
    MODEL = M_VIT_BASE_TORCHVISION

    def _setup_model(self) -> None:
        self.model = ImagenetViTBaseTorchvision(device=self.device)
        self.backbone = self.model.backbone
        self.num_classes = self.model.num_classes
        self.label = "ViT-B/16 (torchvision V1) · ImageNet"


class VitDinoV3BaseImagenet(ImagenetMixin, ModelDataset):
    """DINOv3-B/16 (+reg) backbone + canvit ImageNet-1k linear head."""
    MODEL = M_VIT_DINOV3_BASE

    def _setup_model(self) -> None:
        self.model = ImagenetDinoV3Base(device=self.device)
        self.backbone = self.model.backbone
        self.num_classes = self.model.num_classes
        self.label = "DINOv3-B/16 (+reg, canvit head) · ImageNet"
