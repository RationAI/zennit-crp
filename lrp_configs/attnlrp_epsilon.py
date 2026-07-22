"""``attnlrp_epsilon`` — reference with ε-LRP linears instead of γ-LRP.

Isolates the **linear rule**: ε-LRP (plain, conservative, signed) vs the
reference's γ-LRP (biases relevance toward positive contributions, suppresses
gradient-shattering noise on deep ViTs). Everything else matches
:mod:`lrp_configs.attnlrp_gamma` — full bilinear attention, ratio residual.
"""
from __future__ import annotations

from zennit_ext import AttnLRPCombinedComposite
from ._base import LRPConfig, register

CONFIG = register(LRPConfig(
    name="attnlrp_epsilon",
    description="Full bilinear attention + ε-LRP linears (linear_gamma=None) + "
                "ratio residual. Isolates ε vs γ on the linear/MLP path.",
    build=lambda: AttnLRPCombinedComposite(
        alpha=0.5, beta=0.5, linear_gamma=None, residual_lrp="ratio"),
    site="proj_drop",
    isolates="linear rule (ε vs γ)",
))
