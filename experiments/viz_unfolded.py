"""Tiny display helper for the unfolded-attention CRP walkthrough.

Just one function: :func:`to_display`, which converts a model-input
tensor (or batch of them) to a display-ready ``(H, W, 3)`` ndarray in
``[0, 1]``. No model dependency — dataset transforms emit unnormalized
``[0, 1]`` tensors so display is just clamp + permute.

For everything else (concept atlases, reference samples, etc.) use the
upstream CRP utilities directly:

* :func:`crp.image.plot_grid` — grid layouts of reference images.
* :func:`crp.image.imgify` — tensor → PIL.
* :class:`crp.visualization.FeatureVisualization` — the main pipeline.

The walkthrough notebook composes these inline; no helper module
abstractions wrap them.
"""
from __future__ import annotations

import numpy as np
import torch


def to_display(image: torch.Tensor) -> np.ndarray:
    """Convert a model-input tensor to a display-ready ``(H, W, 3)``
    ndarray in ``[0, 1]``. Accepts ``(1, 3, H, W)`` or ``(3, H, W)``.
    """
    img = image.detach().cpu()
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img.clamp(0, 1)[0].permute(1, 2, 0).numpy()


__all__ = ["to_display"]
