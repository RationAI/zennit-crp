"""``attnlrp_baseline`` — AttnLRP exactly as published (Achtibat et al., ICML
2024), the faithful reference recipe we try to surpass with
:mod:`lrp_configs.cp_lrp_baseline`.

One rule per operation, straight from the paper: the ``q@kᵀ`` / ``attn@v``
bilinears get the matmul rule (Eq. 15), softmax gets the softmax rule
(Prop. 3.1), residual adds the standard ε add, GELU the identity rule (Eq. 9),
LayerNorm the Prop. 3.4 pass-through; FFN linears use γ-LRP and the attention
projections ε-LRP (Table 4). Unlike ``cp_lrp_baseline`` the query/key path is
NOT stop-gradiented, so relevance flows through attention formation too.

See :class:`zennit_ext.attnlrp_composites.AttnLRPBaselineComposite`.
"""
from __future__ import annotations

from zennit_extensions.attnlrp_composites import AttnLRPBaselineComposite
from ._base import LRPConfig, register

CONFIG = register(LRPConfig(
    name="attnlrp_baseline",
    description="AttnLRP as published: matmul rule (Eq. 15) on both bilinears, "
                "softmax rule (Prop. 3.1), ε add residual, γ=0.05 FFN / ε "
                "attention projections, γ=0.25 patch conv.",
    build=lambda: AttnLRPBaselineComposite(),
    site="proj_drop",
    isolates="",  # published reference
))
