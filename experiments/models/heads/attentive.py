"""Attentive classification head — a learned query attends over the full
token sequence (cls + register + patch), then ``LayerNorm`` + ``Linear``.

Canonical 'attentive probe' from the DINOv2 / DINOv3 eval protocols
(Oquab et al. 2024; Darcet et al. 2024). Sees patch-level evidence —
important for tasks where classes are defined by combinations of
spatially-local parts (FunnyBirds, segmentation-like benchmarks).

**AttnLRP-aware unfold.** Attention is implemented from primitives
(``BilinearMatmul``, ``SoftmaxAlongLastDim``, ``ScaleByConstant`` from
``crp.attention_unfolded``) so the AttnLRP composite's bilinear /
softmax-identity / scalar-identity rules apply to the head exactly the
same way they apply to backbone attention. A vanilla
``nn.MultiheadAttention`` would be a black box to the composite — its
backward would run the standard PyTorch chain rule, which is not LRP.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from crp.attention_unfolded import (
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
    matmul_rule, alpha, beta, epsilon
        Bilinear-matmul LRP-rule hyperparameters used by both the
        ``q @ kᵀ`` and ``weights @ v`` ops. The defaults match the
        AttnLRP-paper recipe + this repo's working composite
        (``alpha=0.5, beta=0.5``); you only need to touch them if the
        AttnLRP composite is run with non-default α/β so the head's
        backward stays consistent with the backbone's.
    """

    input_kind = "tokens"

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 8,
        *,
        matmul_rule: str = "alpha_beta",
        alpha: float = 0.5,
        beta: float = 0.5,
        epsilon: float = 1e-6,
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
        # ε-LRP (or γ-LRP) rule from the composite's layer_map.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Unfolded attention primitives (same module classes the backbone
        # substitution uses), so the AttnLRP rules they bake in propagate
        # relevance correctly through this head too.
        # ``identity`` rule on softmax/scale ↔ AttnLRP identity (Eq. 7).
        # ``alpha_beta`` rule on the bilinears ↔ Bach-2015 generalised to
        # bilinear (RESEARCH_NOTES.md Entry 6).
        self.scale_q = ScaleByConstant(self._scale, rule="identity")
        self.qk_scores = BilinearMatmul(
            rule=matmul_rule, epsilon=epsilon, alpha=alpha, beta=beta,
        )
        self.softmax = SoftmaxAlongLastDim(rule="identity")
        self.context = BilinearMatmul(
            rule=matmul_rule, epsilon=epsilon, alpha=alpha, beta=beta,
        )

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

        # Scale Q (separate Module → identity LRP rule).
        q = self.scale_q(q)

        # Bilinear matmul q @ kᵀ → AlphaBeta rule baked into the module.
        # k.transpose(-2, -1) is a view, autograd-trivial.
        scores = self.qk_scores(q, k.transpose(-2, -1))   # (B, H, 1, T)
        weights = self.softmax(scores)                    # identity rule

        # Bilinear matmul weights @ v → AlphaBeta rule baked in.
        ctx = self.context(weights, v)                    # (B, H, 1, hd)

        # Reshape back: (B, H, 1, hd) → (B, 1, D) → (B, D).
        ctx = ctx.transpose(1, 2).reshape(B, 1, D).squeeze(1)

        # Output projection + classify (each Linear → ε-LRP via composite).
        pooled = self.out_proj(ctx)
        pooled = self.norm(pooled)
        return self.linear(pooled)
