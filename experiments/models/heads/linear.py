"""Linear classification head on the cls-token pre-logits.

Cheap (~50 K params for 50 classes on ViT-L). Walkthrough-compatible —
this is what timm's ``model.head`` slot expects."""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Head


class LinearHead(Head):
    """``nn.Linear(embed_dim, num_classes)`` on the pre-logits cls feature.

    Sees only the global cls token — ignores patch + register tokens.
    Plateaus on tasks where classes share most spatial parts (e.g.
    FunnyBirds), since the global feature can't disambiguate.
    """

    input_kind = "cls"

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D), already pre-logits-normed by the base.
        return self.linear(x)
