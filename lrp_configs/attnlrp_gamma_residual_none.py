"""``attnlrp_gamma_residual_none`` — reference with NO special residual rule.

Isolates the **skip-connection split**. With ``residual_lrp=None`` the block
addition ``x + sublayer(x)`` is left to the default gradient-proportional
split (relevance flows in proportion to the local gradient, which on a
near-identity residual sends almost everything down the identity branch).
Compare against the reference (``ratio`` split) and
:mod:`lrp_configs.attnlrp_gamma_residual_symmetric` to rank how the skip
connection should share relevance.
"""
from __future__ import annotations

from zennit_ext import AttnLRPCombinedComposite
from ._base import LRPConfig, register

CONFIG = register(LRPConfig(
    name="attnlrp_gamma_residual_none",
    description="Reference but residual_lrp=None (default gradient split on the "
                "skip connection). Isolates residual relevance splitting.",
    build=lambda: AttnLRPCombinedComposite(
        alpha=0.5, beta=0.5, linear_gamma=0.25, residual_lrp=None),
    site="proj_drop",
    isolates="residual split",
))
