"""Attentive classification head — a learned query attends over the full
token sequence (cls + register + patch) via :class:`nn.MultiheadAttention`,
then ``LayerNorm`` + ``Linear``.

This is the canonical 'attentive probe' from the DINOv2 / DINOv3 eval
protocols (Oquab et al. 2024; Darcet et al. 2024). Sees patch-level
evidence — important for tasks where classes are defined by combinations
of spatially-local parts (FunnyBirds, segmentation-like benchmarks).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Head


class AttentiveHead(Head):
    """Learned-query attention pooling + ``Linear`` classifier.

    Trainable params (ViT-L, num_heads=8): ~4.2 M (4 × D² MHA
    projections + LN + linear). Backbone stays frozen.

    Parameters
    ----------
    embed_dim
        Token embedding dimension (matches the base's ``embed_dim``).
    num_classes
        Number of output classes.
    num_heads
        MultiheadAttention heads in the pooling layer. Must divide
        ``embed_dim``.
    """

    input_kind = "tokens"

    def __init__(
        self, embed_dim: int, num_classes: int, num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D). Cached features may be fp16 — cast for
        # numerically stable attention.
        tokens = tokens.float()
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)              # (B, 1, D)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))         # (B, D)
        return self.linear(pooled)
