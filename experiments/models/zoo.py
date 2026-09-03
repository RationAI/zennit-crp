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

Pairing a model with a dataset — and the flat ``<model>_<dataset>`` tag that
keys FV caches and the gallery figure tree — lives in
:mod:`experiments.model_datasets` (``find`` / ``find_by_tag``); this module is
just the class definitions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

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


def _timm_key_from_torchvision(key: str) -> str:
    """torchvision ViT state-dict key -> timm state-dict key (1:1 bijection,
    verified logit-exact both directions on ViT-B/16)."""
    import re
    key = key.replace("class_token", "cls_token").replace("conv_proj", "patch_embed.proj")
    key = key.replace("encoder.pos_embedding", "pos_embed")
    m = re.match(r"encoder\.layers\.encoder_layer_(\d+)\.(.*)", key)
    if m:
        i, rest = m.groups()
        rest = (rest.replace("ln_1", "norm1").replace("ln_2", "norm2")
                    .replace("self_attention.in_proj_weight", "attn.qkv.weight")
                    .replace("self_attention.in_proj_bias", "attn.qkv.bias")
                    .replace("self_attention.out_proj", "attn.proj")
                    .replace("mlp.linear_1", "mlp.fc1").replace("mlp.linear_2", "mlp.fc2")
                    .replace("mlp.0", "mlp.fc1").replace("mlp.3", "mlp.fc2"))
        return f"blocks.{i}.{rest}"
    return key.replace("encoder.ln", "norm").replace("heads.head", "head")


class ImagenetViTBaseTorchvision(ImagenetViTBase):
    """timm ViT-B/16 skeleton carrying the torchvision ``ViT_B_16`` /
    ``IMAGENET1K_V1`` checkpoint, transplanted via the key bijection
    (weights only — architecture and all tooling stay the timm layout).
    ``pretrained_cfg`` is overridden to the V1 preset's preprocessing (256
    bilinear resize → 224 crop → ImageNet mean/std), so ``backbone_transforms``
    yields the pipeline these weights were trained with, not the augreg2 one.
    Journal model record M7."""

    def __init__(self, *, device: str = "cpu"):
        super().__init__(device=device)
        from torchvision.models import ViT_B_16_Weights
        tv_state = ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        self.backbone.load_state_dict(
            {_timm_key_from_torchvision(k): v for k, v in tv_state.items()}, strict=True)
        # preprocessing follows the weights: torchvision V1 preset
        self.backbone.pretrained_cfg = dict(
            self.backbone.pretrained_cfg,
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
            interpolation="bilinear", crop_pct=224 / 256,
            input_size=(3, 224, 224),
        )
        self.eval().to(device)
        self.source = "torchvision:ViT_B_16/IMAGENET1K_V1 (timm-skeleton transplant)"


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


# Model↔dataset selection now lives in :mod:`experiments.model_datasets`
# (``find`` / ``find_by_tag`` over the ``(model, dataset)`` registry). The zoo
# keeps only the model classes; the flat ``<model>_<dataset>`` tag — still the
# name of the FV cache dirs and gallery figure tree — is ``ModelDataset.tag``.

__all__ = [
    "FinetunedProbe", "FunnyBirdsViTSmall", "FunnyBirdsDinoV3Small",
    "DspritesViTSmall", "ColoredMnistViTSmall",
    "ImagenetViTBase", "ImagenetViTBaseTorchvision", "ImagenetDinoV3Base",
    "DINOV3_IN1K_HEAD_REPOS", "load_dinov3_in1k_head",
]
