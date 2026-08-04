"""Block classification head — one extra transformer block (self-attention +
MLP) on top of the frozen backbone, then ``LayerNorm`` + ``Linear`` on the
cls position.

Sits between :class:`LinearHead` (cls only, ~50 K params) and
:class:`AttentiveHead` (learned-query pooling, ~4 M params): a real ViT
block (~12.6 M params on ViT-L) that lets the head re-mix tokens before
reading out cls. Useful when the frozen backbone's cls is close but not
quite enough — e.g. fine-grained classification where the missing signal
is in patches the cls aggregator under-weights.

**Vanilla forward.** Atomic submodules
(:class:`~zennit_ext.BilinearMatmul`,
:class:`~zennit_ext.SoftmaxAlongLastDim`,
:class:`~zennit_ext.ScaleByConstant`) and the MLP/LNs are all
plain PyTorch. The head trains with autograd's standard chain-rule
backward. AttnLRP rules are applied at attribution time by the
composite's per-rule canonizers (see :mod:`zennit_ext`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zennit_extentions import (
    BilinearMatmul, SoftmaxAlongLastDim, ScaleByConstant,
)

from .base import Head


class _UnfoldedSelfAttention(nn.Module):
    """Multi-head self-attention with vanilla unfolded primitives.

    Same module-family as :class:`AttentiveHead` but full self-attention
    over T tokens instead of pooling over them with a single learned
    query. All forwards are vanilla PyTorch; LRP rules are applied at
    attribution time by the composite.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self._scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale_q = ScaleByConstant(self._scale)
        self.qk_scores = BilinearMatmul()
        self.softmax = SoftmaxAlongLastDim()
        self.context = BilinearMatmul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, hd).transpose(1, 2)

        q = self.scale_q(q)
        scores = self.qk_scores(q, k.transpose(-2, -1))
        weights = self.softmax(scores)
        ctx = self.context(weights, v)

        ctx = ctx.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(ctx)


class _Mlp(nn.Module):
    """ViT-style two-layer MLP with GELU."""

    def __init__(self, embed_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(round(embed_dim * mlp_ratio))
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class BlockHead(Head):
    """Pre-norm transformer block + cls-token classifier.

    Trainable params (ViT-L, num_heads=16, mlp_ratio=4.0): ~12.6 M
    (4 × D² self-attention + 2 × D × 4D MLP + LNs + classifier).
    Backbone stays frozen.

    Parameters
    ----------
    embed_dim
        Token embedding dimension. Must be divisible by ``num_heads``.
    num_classes
        Number of output classes.
    num_heads
        Self-attention heads.
    mlp_ratio
        MLP hidden-dim multiplier (ViT default: 4.0).
    """

    input_kind = "tokens"

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 16,
        *,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _UnfoldedSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _Mlp(embed_dim, mlp_ratio)

        self.norm_out = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D). Cached features may be fp16 → cast for stable LN.
        x = tokens.float()
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        cls = x[:, 0]
        return self.linear(self.norm_out(cls))
