"""``attnlrp_alpha2beta1`` — reference with Bach's classical α=2, β=−1 attention.

Isolates the **AlphaBeta emphasis on the attention bilinear**. The reference
uses balanced α=β=0.5 (tightest magnitude control). The classical
``alpha2beta1`` rule (Bach 2015) amplifies positive evidence and subtracts
negative, which tends to sharpen but can break exact conservation on the
bilinear. Compare to gauge how much the attention path's pos/neg weighting
matters. Linears stay γ=0.25, residual stays ratio.
"""
from __future__ import annotations

from zennit_ext import AttnLRPCombinedComposite
from ._base import LRPConfig, register

CONFIG = register(LRPConfig(
    name="attnlrp_alpha2beta1",
    description="Reference but attention AlphaBeta α=2, β=−1 (Bach classical). "
                "Isolates pos/neg emphasis on the attention bilinear.",
    build=lambda: AttnLRPCombinedComposite(
        alpha=2.0, beta=-1.0, linear_gamma=0.25, residual_lrp="ratio"),
    site="proj_drop",
    isolates="attention αβ emphasis",
))
