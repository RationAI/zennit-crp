"""Abstract :class:`Base` — a frozen ViT backbone with two feature
extractors. Subclasses set the timm model name; everything else is
shared."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm
from timm.data import create_transform, resolve_data_config


class Base(nn.Module):
    """Frozen ViT backbone.

    Subclasses set :attr:`timm_name`. The backbone is created with
    ``num_classes=0`` (timm's classification head stripped to identity).
    All parameters are frozen on construction; the trainable bits live
    in the :class:`~models.heads.base.Head` we compose on top.

    Two feature extractors:

    * :meth:`extract_cls` → ``(B, D)`` cls-token pre-logits (fc_norm
      applied via timm's ``forward_head(pre_logits=True)``). Cheap.
    * :meth:`extract_tokens` → ``(B, T, D)`` post-block-norm full token
      sequence (cls + register + patch). Expensive but needed by the
      attentive probe.
    """

    timm_name: str

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            self.timm_name, pretrained=True, num_classes=0,
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.embed_dim: int = getattr(
            self.backbone, "embed_dim", self.backbone.num_features,
        )

    @torch.no_grad()
    def extract_cls(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone.forward_features(x)
        return self.backbone.forward_head(out, pre_logits=True)

    @torch.no_grad()
    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def get_transform(self):
        """Return the timm input transform (resize + normalize) the
        backbone was trained with."""
        cfg = resolve_data_config({}, model=self.backbone)
        return create_transform(**cfg, is_training=False)
