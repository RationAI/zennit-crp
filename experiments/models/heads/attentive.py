"""Attentive classification head — a learned query attends over the full
token sequence (cls + register + patch), then ``LayerNorm`` + ``Linear``.

Canonical 'attentive probe' from the DINOv2 / DINOv3 eval protocols
(Oquab et al. 2024; Darcet et al. 2024). Sees patch-level evidence —
important for tasks where classes are defined by combinations of
spatially-local parts (FunnyBirds, segmentation-like benchmarks).

**Vanilla forward.** All atomic submodules
(:class:`~zennit_ext.BilinearMatmul`,
:class:`~zennit_ext.SoftmaxAlongLastDim`,
:class:`~zennit_ext.ScaleByConstant`) have plain PyTorch
forwards; autograd's standard backward applies during ``loss.backward()``,
so this head trains with correct chain-rule gradients.

For attribution, a composite (e.g. :class:`~zennit_extensions.AttnLRPBaselineComposite`)
assigns the AttnLRP rules to these submodule types via its ``layer_map`` —
zennit ``Hook``s (``AlphaBetaMatmul`` on ``BilinearMatmul``, ``Pass`` on
``SoftmaxAlongLastDim`` / ``ScaleByConstant``) that fire during the
attribution backward and detach on exit, leaving the forwards vanilla.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from zennit_extensions import (
    BilinearMatmul, SoftmaxAlongLastDim, ScaleByConstant,
)

from .base import Head


class AttentiveHead(Head):
    """Learned-query attention pooling + ``Linear`` classifier.

    Trainable params (ViT-L, num_heads=8): ~4.2 M
    (4 × D² Q/K/V/out projections + LN + linear). Backbone stays frozen.

    Parameters
    ----------
    embed_dim
        Token embedding dimension (matches the base's ``embed_dim``).
        Must be divisible by ``num_heads``.
    num_classes
        Number of output classes.
    num_heads
        MultiheadAttention heads in the pooling layer.
    """

    input_kind = "tokens"

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self._scale = self.head_dim ** -0.5

        # Learned query — broadcast over the batch at forward time.
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        # Q/K/V/output projections — plain nn.Linear receive the standard
        # ε-LRP (or γ-LRP) rule from the composite's layer_map at attribution.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Vanilla unfolded primitives — exposed as named submodules so the
        # composite's per-rule canonizers can rebind their forwards at
        # attribution time. At training time their forwards are bare
        # ``a @ b`` / ``F.softmax`` / ``x * scalar`` and autograd's
        # standard backward applies.
        self.scale_q = ScaleByConstant(self._scale)
        self.qk_scores = BilinearMatmul()
        self.softmax = SoftmaxAlongLastDim()
        self.context = BilinearMatmul()

        # Final classifier on the pooled, normed feature.
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D). Cached features may be fp16 — cast for
        # numerically stable attention.
        tokens = tokens.float()
        B, T, D = tokens.shape
        H, hd = self.num_heads, self.head_dim

        # Project query (broadcast to batch), key, value.
        q = self.q_proj(self.query.expand(B, -1, -1))   # (B, 1, D)
        k = self.k_proj(tokens)                          # (B, T, D)
        v = self.v_proj(tokens)                          # (B, T, D)

        # Per-head reshape: (B, *, D) → (B, H, *, hd). View ops, no rule.
        q = q.reshape(B, 1, H, hd).transpose(1, 2)       # (B, H, 1, hd)
        k = k.reshape(B, T, H, hd).transpose(1, 2)       # (B, H, T, hd)
        v = v.reshape(B, T, H, hd).transpose(1, 2)       # (B, H, T, hd)

        # Scale Q (vanilla; LRP identity rule applied by composite at attr-time).
        q = self.scale_q(q)

        # Bilinear matmul q @ kᵀ. k.transpose is a view, autograd-trivial.
        scores = self.qk_scores(q, k.transpose(-2, -1))   # (B, H, 1, T)
        weights = self.softmax(scores)

        # Bilinear matmul weights @ v.
        ctx = self.context(weights, v)                    # (B, H, 1, hd)

        # Reshape back: (B, H, 1, hd) → (B, 1, D) → (B, D).
        ctx = ctx.transpose(1, 2).reshape(B, 1, D).squeeze(1)

        # Output projection + classify (each Linear → ε-LRP via composite).
        pooled = self.out_proj(ctx)
        pooled = self.norm(pooled)
        return self.linear(pooled)
