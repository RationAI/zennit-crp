"""Loader for the FunnyBirds authors' pretrained ViT-B/16 classifier.

Provenance
----------
Checkpoint ``vit_base_patch16_224_final_1_checkpoint_best.pth.tar`` published by
the FunnyBirds authors (Hesse, Schaub-Meyer, Roth; ICCV 2023) on the TU Darmstadt
visinf mirror, via the official ``visinf/funnybirds-framework`` repo:

    https://download.visinf.tu-darmstadt.de/data/funnybirds/models/vit_base_patch16_224_final_1_checkpoint_best.pth.tar

Their training entry point (``train.py``) fine-tunes an ImageNet-pretrained
ViT-B/16 (rwightman "jx" weights) with the classification head swapped to 50
outputs (``model.head = nn.Linear(768, 50)``). Architecture is a vendored copy of
the Chefer ViT (``models/ViT/ViT_new.py``) whose ``state_dict`` layout is
**byte-for-byte identical** to timm's ``vit_base_patch16_224`` (verified: zero
missing / extra keys against ``timm.create_model('vit_base_patch16_224',
num_classes=50)``), so we load it straight into a stock timm ViT-B/16 — which also
lets the AttnLRP composite (type-based, timm-module-path aware) operate on it
unchanged, exactly like the ``imagenet`` full-timm model in
:mod:`experiments.model_io`.

Preprocessing (their pipeline, reproduced faithfully)
-----------------------------------------------------
* PNG → RGB (alpha dropped) → resize to **256×256** → ``ToTensor`` (``[0, 1]``).
* **No** ImageNet mean/std normalization anywhere.
* Their ``ViTModel`` wrapper bilinearly interpolates the input to **224×224**
  before the ViT. We replicate that inside :meth:`FunnyBirdsViTB.forward`.

So the dataset-side transform yields un-normalized ``[0, 1]`` 256×256 tensors
(consistent with the repo's "no-normalize dataset, model owns the forward
boundary" convention) and the wrapper handles the 256→224 downscale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor

TIMM_NAME = "vit_base_patch16_224"
NUM_CLASSES = 50

# Default on-disk location of the downloaded checkpoint.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = (
    REPO_ROOT / "data" / "funnybirds_models"
    / "vit_base_patch16_224_final_1_checkpoint_best.pth.tar"
)


class FunnyBirdsViTB(nn.Module):
    """FunnyBirds authors' ViT-B/16, wrapped to match the experiments' surface.

    Exposes the full timm classifier under ``self.backbone`` (so
    ``backbone.blocks.{i}`` attribution layer-names and SAE site modules resolve,
    identical to :class:`experiments.model_io._TimmFullProbe`), and a ``forward``
    that returns 50-way class logits directly after downscaling the input to
    224×224 — reproducing the authors' ``ViTModel`` wrapper.
    """

    def __init__(self, timm_model: nn.Module) -> None:
        super().__init__()
        self.backbone = timm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.backbone(x)


def funnybirds_vitb_transforms() -> Tuple[Callable, Callable]:
    """Return ``(transform, normalize)`` for the authors' ViT-B.

    * ``transform`` — resize to 256×256 + ``ToTensor`` (``[0, 1]``), **no
      normalize** — matches ``datasets/funny_birds.py`` in their repo. The
      256→224 downscale is done in :meth:`FunnyBirdsViTB.forward`.
    * ``normalize`` — identity closure (they train/eval without mean/std
      normalization), kept for API parity with
      :func:`experiments.models.transforms.backbone_transforms`.
    """
    transform = Compose([Resize((256, 256)), ToTensor()])

    def normalize(t: torch.Tensor) -> torch.Tensor:
        return t

    return transform, normalize


def load_funnybirds_vitb(
    ckpt_path: Path | str = DEFAULT_CKPT, device: str = "cpu",
) -> FunnyBirdsViTB:
    """Build a timm ViT-B/16 (num_classes=50), load the authors' fine-tuned
    weights, and return a frozen eval :class:`FunnyBirdsViTB`.

    The ``.pth.tar`` stores the weights under the ``['state_dict']`` key with
    raw ViT keys (no prefix); they map 1:1 onto timm's ``vit_base_patch16_224``.
    """
    ckpt_path = Path(ckpt_path)
    tm = timm.create_model(TIMM_NAME, pretrained=False, num_classes=NUM_CLASSES)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    missing, unexpected = tm.load_state_dict(state, strict=True)
    model = FunnyBirdsViTB(tm).eval().to(device)
    model.requires_grad_(False)
    return model
