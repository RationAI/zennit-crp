"""``attnlrp_gamma_residual_symmetric`` — reference with a symmetric skip split.

Isolates the **skip-connection split**, paired with
:mod:`lrp_configs.attnlrp_gamma_residual_none` and the reference (``ratio``).
``residual_lrp='symmetric'`` splits relevance 50/50 between the identity and
the sublayer branch regardless of their magnitudes — the simplest conservative
choice. Compare its faithfulness against ratio (magnitude-proportional) and
none (gradient default).
"""
from __future__ import annotations

from zennit_ext import AttnLRPCombinedComposite
from ._base import LRPConfig, register

CONFIG = register(LRPConfig(
    name="attnlrp_gamma_residual_symmetric",
    description="Reference but residual_lrp='symmetric' (50/50 skip split). "
                "Isolates residual relevance splitting.",
    build=lambda: AttnLRPCombinedComposite(
        alpha=0.5, beta=0.5, linear_gamma=0.25, residual_lrp="symmetric"),
    site="proj_drop",
    isolates="residual split",
))
