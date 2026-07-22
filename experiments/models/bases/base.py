"""Abstract :class:`Base` — a frozen ViT backbone with two feature
extractors. Subclasses set the timm model name; everything else is
shared."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

from ..transforms import backbone_transforms


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

    Neither extractor wraps the forward in ``torch.no_grad()`` — frozen
    params (set in ``__init__``) are sufficient to keep weight updates
    off during head training, and the AttnLRP composite *needs* the
    autograd graph at attribution time so its custom backward rules can
    fire. Wrap explicitly at call sites that don't need the graph
    (e.g. one-shot feature caching), as :func:`train_probe.cache_cmd`
    does.
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

    def extract_cls(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone.forward_features(x)
        return self.backbone.forward_head(out, pre_logits=True)

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def get_transform(self):
        """Return the dataset-side input transform: resize + ``ToTensor``
        only, NO normalize. Output is plain ``[0, 1]`` tensors —
        display-ready, and uniform across all consumers (DataLoader,
        FeatureVisualization, Lightning, raw ``model(x)``).

        Normalize is the model's responsibility, applied at the forward
        boundary via :meth:`get_normalize`. This split keeps the dataset
        decoupled from the backbone's training stats: a dataset built
        once can feed both an ImageNet-normalized timm ViT and a
        no-normalize visinf checkpoint.

        Canonical implementation lives in :func:`models.transforms.backbone_transforms`.
        """
        return backbone_transforms(self.backbone)[0]

    def get_normalize(self):
        """Return the per-batch normalize callable ``(x - mean) / std``
        with the backbone's canonical mean/std from ``pretrained_cfg``.

        Apply this to a batch immediately before any backbone forward.
        For models trained without normalize (e.g. the visinf vit_base
        checkpoint, where the notebook's TRANSFORM_SPEC dispatcher sets
        mean/std to identity), this is a no-op closure.

        Canonical implementation lives in :func:`models.transforms.backbone_transforms`.
        """
        return backbone_transforms(self.backbone)[1]
