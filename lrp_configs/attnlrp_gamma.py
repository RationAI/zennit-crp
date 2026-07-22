"""``attnlrp_gamma`` — the REFERENCE configuration.

Full AttnLRP: attention is unfolded and the Q·Kᵀ / weights·V bilinear products
get the AlphaBeta-on-bilinear rule, so relevance flows through **both** the
value path and the query/key (attention-formation) path. Linears use γ-LRP
(γ=0.25, AttnLRP §3.2.1); the block skip connection uses the ratio split.

Every other variant in :mod:`lrp_configs` is this recipe with exactly one
building block changed, so the variant-ranking table reads as single-knob
ablations against this row.
"""
from __future__ import annotations

from zennit_ext import AttnLRPCombinedComposite
from ._base import LRPConfig, register

CONFIG = register(LRPConfig(
    name="attnlrp_gamma",
    description="Full bilinear attention (AlphaBeta α=β=0.5) + γ=0.25 linears + "
                "ratio residual. Reference recipe; all variants vary one knob from it.",
    build=lambda: AttnLRPCombinedComposite(
        alpha=0.5, beta=0.5, linear_gamma=0.25, residual_lrp="ratio"),
    site="proj_drop",
    isolates="",  # the reference
))
