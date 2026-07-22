"""``attnlrp_gamma_palrp`` — reference + PA-LRP on the positional embedding.

Isolates whether the **positional embedding should consume relevance**. PA-LRP
(Bakish et al. 2025) applies a conservative rule to the additive
``x + pos_embed`` step so the absolute positional embedding absorbs its share
of relevance instead of passing it all through to the patch tokens. No-op on
RoPE-only models. Otherwise identical to :mod:`lrp_configs.attnlrp_gamma`.
"""
from __future__ import annotations

from zennit_ext import AttnLRPCombinedComposite
from ._base import LRPConfig, register

CONFIG = register(LRPConfig(
    name="attnlrp_gamma_palrp",
    description="Reference + PA-LRP on x+pos_embed (palrp=True). Isolates "
                "letting the positional embedding consume relevance.",
    build=lambda: AttnLRPCombinedComposite(
        alpha=0.5, beta=0.5, linear_gamma=0.25, residual_lrp="ratio", palrp=True),
    site="proj_drop",
    isolates="positional-embedding PA-LRP",
))
