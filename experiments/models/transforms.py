"""Canonical dataset/forward transform split for a timm backbone.

Single source of truth for the *un-normalized dataset + normalize-at-forward*
convention used across the experiments and by :class:`models.bases.base.Base`
(`get_transform`/`get_normalize` delegate here). Keeping it in one leaf module
(no intra-package imports) lets every consumer — the Base, the zoo, the
CRP gallery — reach the same implementation without duplicating it or risking
an import cycle.
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
from timm.data import create_transform, resolve_data_config


def backbone_transforms(backbone) -> Tuple[Callable, Callable]:
    """Return ``(transform, normalize)`` for ``backbone``:

    * ``transform`` — resize/crop/ToTensor with mean=0/std=1, i.e. **no
      normalize**; yields display-ready ``[0, 1]`` tensors (what a DataLoader /
      FeatureVisualization should store).
    * ``normalize`` — ``(x - mean) / std`` closure with the backbone's canonical
      stats, applied at the forward boundary (FV ``preprocess_fn`` / direct
      ``model(x)``). For a model trained without normalize this is a no-op.
    """
    cfg = resolve_data_config({}, model=backbone)
    transform = create_transform(
        **{**cfg, "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)}, is_training=False)
    mean = torch.tensor(cfg["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, -1, 1, 1)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - mean.to(x.device, x.dtype)) / std.to(x.device, x.dtype)

    return transform, normalize
