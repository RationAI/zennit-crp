"""The model zoo — one class per model recorded in the experiment journal.

Each class encapsulates a complete frozen eval model: the architecture is
encoded in the class, the only parameters are *where the weights come from*
(an optional explicit ``checkpoint`` for the finetuned probes) and ``device``.
Every model exposes the same surface:

* ``backbone``    — the ViT whose ``blocks.{i}`` paths the composites/concepts
  resolve against,
* ``forward(x)``  — class logits,
* ``num_classes`` / ``head_name`` / ``source`` — metadata (``source`` is the
  provenance string: checkpoint path, ``timm:<name>``, or an HF repo id).

:data:`MODELS` maps the canonical model tag (``<base>_<dataset>``, the string
that keys FV caches and the gallery figure tree — keep stable) to its class.
:data:`DEFAULT_MODELS` maps an eval-dataset key to the journal model used for
it by default (concept flipping, gallery ``checkpoint`` source).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Type

import timm
import torch
import torch.nn as nn

from .probe import Probe

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Finetuned probes (train_probe checkpoints) ───────────────────────────────

class FinetunedProbe(Probe):
    """Eval probe rebuilt from a ``train_probe`` checkpoint.

    The payload self-describes the architecture (``base`` / ``head`` /
    ``num_classes`` / ``head_kwargs``) — ``base``/``run_tag`` only locate the
    newest ``data/runs/finetune_<base>_<run_tag>/<ts>/best.pt`` when no
    explicit ``checkpoint`` is given. Frozen-head-only payloads (no
    ``backbone_state_dict``) keep the timm-pretrained backbone.
    """

    def __init__(self, base: Optional[str] = None, run_tag: Optional[str] = None,
                 *, checkpoint: Optional[Path | str] = None, device: str = "cpu"):
        from experiments.models import build_base, build_head

        if checkpoint is None:
            if base is None or run_tag is None:
                raise ValueError("pass either checkpoint= or (base, run_tag)")
            runs = sorted((REPO_ROOT / "data" / "runs"
                           / f"finetune_{base}_{run_tag}").glob("*/best.pt"))
            if not runs:
                raise FileNotFoundError(
                    f"no probe under data/runs/finetune_{base}_{run_tag}/")
            checkpoint = runs[-1]
        checkpoint = Path(checkpoint)
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        base_obj = build_base(ck["base"])
        head_obj = build_head(ck["head"], embed_dim=base_obj.embed_dim,
                              num_classes=ck["num_classes"],
                              head_kwargs=ck.get("head_kwargs", {}))
        super().__init__(base_obj, head_obj)
        if "backbone_state_dict" in ck:
            self.backbone.load_state_dict(ck["backbone_state_dict"])
        self.head.load_state_dict(ck["head_state_dict"])
        self.eval().to(device)
        self.requires_grad_(False)
        self.num_classes = int(ck["num_classes"])
        self.head_name = str(ck["head"])
        self.source = checkpoint
        # payload metadata without the weight tensors (val_acc, dataset, …)
        self.meta = {k: v for k, v in ck.items() if not k.endswith("state_dict")}


class FunnyBirdsViTSmall(FinetunedProbe):
    """OneCycle-finetuned ViT-S/16 + linear head on clean FunnyBirds train."""

    def __init__(self, *, checkpoint: Optional[Path | str] = None, device: str = "cpu"):
        super().__init__("vit_small", "funny-birds-train-clean",
                         checkpoint=checkpoint, device=device)


class FunnyBirdsDinoV3Small(FinetunedProbe):
    """OneCycle-finetuned DINOv3-S/16 + linear head on clean FunnyBirds train."""

    def __init__(self, *, checkpoint: Optional[Path | str] = None, device: str = "cpu"):
        super().__init__("vit_dinov3_small", "funny-birds-train-clean",
                         checkpoint=checkpoint, device=device)


class DspritesViTSmall(FinetunedProbe):
    """OneCycle-finetuned ViT-S/16 + linear head on dSprites (shape target)."""

    def __init__(self, *, checkpoint: Optional[Path | str] = None, device: str = "cpu"):
        super().__init__("vit_small", "dsprites", checkpoint=checkpoint, device=device)


class ColoredMnistViTSmall(FinetunedProbe):
    """OneCycle-finetuned ViT-S/16 + linear head on colored-MNIST train."""

    def __init__(self, *, checkpoint: Optional[Path | str] = None, device: str = "cpu"):
        super().__init__("vit_small", "colored-mnist-train",
                         checkpoint=checkpoint, device=device)


# ── Off-the-shelf ImageNet classifiers ───────────────────────────────────────

class ImagenetViTBase(nn.Module):
    """timm ViT-B/16, ImageNet-1k-pretrained, full classifier.

    The classification head lives *inside* ``backbone`` (timm's ``norm`` +
    ``head``); the LRP composites are type-based, so they handle those modules
    like any other linear/norm.
    """

    TIMM_NAME = "vit_base_patch16_224"

    def __init__(self, *, device: str = "cpu"):
        super().__init__()
        self.backbone = timm.create_model(self.TIMM_NAME, pretrained=True,
                                          num_classes=1000)
        self.eval().to(device)
        self.requires_grad_(False)
        self.num_classes = 1000
        self.head_name = "timm_builtin"
        self.source = f"timm:{self.TIMM_NAME}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# The canvit probes (https://github.com/m2b3/dinov3-in1k-probes) are linear
# classifiers on the frozen DINOv3 cls token, trained on ImageNet-1k at
# 512×512, published per backbone size:
DINOV3_IN1K_HEAD_REPOS = {
    "vit_base_patch16_dinov3.lvd1689m":
        "canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe",
    "vit_small_patch16_dinov3.lvd1689m":
        "canvit/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe",
    "vit_large_patch16_dinov3.lvd1689m":
        "canvit/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe",
}


def load_dinov3_in1k_head(timm_name: str, device: str = "cpu") -> nn.Linear:
    """Download the matching canvit ImageNet-1k linear head (cls token → 1000
    logits) and return it as a frozen ``nn.Linear``."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download(DINOV3_IN1K_HEAD_REPOS[timm_name],
                                   "model.safetensors"))
    head = nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0])
    head.load_state_dict(sd)
    head.requires_grad_(False)
    return head.eval().to(device)


class ImagenetDinoV3Base(nn.Module):
    """DINOv3-B/16 backbone (timm, lvd1689m weights, 256 px) + the public
    canvit ImageNet-1k linear head on the final-norm cls token, wrapped as one
    classifier module so attribution sees a single model."""

    TIMM_NAME = "vit_base_patch16_dinov3.lvd1689m"
    IMG_SIZE = 256

    def __init__(self, *, device: str = "cpu"):
        super().__init__()
        self.backbone = timm.create_model(self.TIMM_NAME, pretrained=True,
                                          num_classes=0, img_size=self.IMG_SIZE)
        self.head = load_dinov3_in1k_head(self.TIMM_NAME, device)
        self.eval().to(device)
        self.requires_grad_(False)
        self.num_classes = 1000
        self.head_name = "canvit_in1k_linear_cls"
        self.source = DINOV3_IN1K_HEAD_REPOS[self.TIMM_NAME]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.forward_features(x)[:, 0])


# ── Registries ────────────────────────────────────────────────────────────────

# Keyed by the canonical model tag ``<base>_<dataset>`` — the same string that
# names FV cache dirs and the gallery figure tree. Keep keys stable.
MODELS: Dict[str, Type[nn.Module]] = {
    "vit_small_funny_birds":        FunnyBirdsViTSmall,
    "vit_dinov3_small_funny_birds": FunnyBirdsDinoV3Small,
    "vit_small_dsprites":           DspritesViTSmall,
    "vit_small_colored_mnist":      ColoredMnistViTSmall,
    "vit_base_imagenet":            ImagenetViTBase,
    "vit_dinov3_base_imagenet":     ImagenetDinoV3Base,
}

# Journal-default model per eval-dataset key (concept flipping, gallery
# ``checkpoint`` model source).
DEFAULT_MODELS: Dict[str, str] = {
    "funny_birds":   "vit_small_funny_birds",
    "dsprites":      "vit_small_dsprites",
    "colored_mnist": "vit_small_colored_mnist",
    "imagenet":      "vit_base_imagenet",
}

__all__ = [
    "MODELS", "DEFAULT_MODELS",
    "FinetunedProbe", "FunnyBirdsViTSmall", "FunnyBirdsDinoV3Small",
    "DspritesViTSmall", "ColoredMnistViTSmall",
    "ImagenetViTBase", "ImagenetDinoV3Base",
    "DINOV3_IN1K_HEAD_REPOS", "load_dinov3_in1k_head",
]
