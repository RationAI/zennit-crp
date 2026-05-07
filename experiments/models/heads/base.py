"""Abstract :class:`Head` — a trainable classification module that
consumes features from a :class:`~models.bases.base.Base` backbone."""
from __future__ import annotations

import torch.nn as nn


class Head(nn.Module):
    """Subclasses set :attr:`input_kind` to declare the feature shape
    they consume:

    * ``"cls"``    → ``(B, D)`` — pre-logits cls token (cheap cache).
    * ``"tokens"`` → ``(B, T, D)`` — full token sequence (expensive
      cache, but lets the head see patch-level evidence).

    The training pipeline reads ``input_kind`` to pick the right cache.
    """

    input_kind: str  # "cls" | "tokens"
