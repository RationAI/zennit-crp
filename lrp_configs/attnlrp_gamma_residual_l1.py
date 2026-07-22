"""``attnlrp_gamma_residual_l1`` — sign-preserving, L1-conserving skip split.

Isolates the **skip-connection split**, like the other ``residual_*`` configs,
using the L1 residual rule instead of the Otsuki magnitude ratio.

Each operand keeps its OWN sign, normalised by ``|x|+|branch|``:

    R_x = R_y * x / (|x|+|branch|),   R_branch = R_y * branch / (|x|+|branch|)

So an opposing branch (sign opposite to the running stream) gets opposite-sign
relevance — the sign-awareness the magnitude ``ratio`` rule lacks. The price:
conservation holds in the **L1 sense** (``|R_x|+|R_branch|=|R_y|``), NOT the
signed sum (``R_x+R_branch = R_y·(x+branch)/(|x|+|branch|)``), so this variant
forgoes LRP's global "relevance sums to the logit" property (its global mass
collapses — see ``research/residual_lrp_notes.md``). Kept for **manual
inspection / research only**; production residual rule is Otsuki ``ratio``.
Per-block ranking is scale-invariant, so its concept-flipping AOPC may still be
sensible — untested. See :class:`zennit_ext.attnlrp_rules.ResidualL1`.
"""
from __future__ import annotations

from zennit_ext import AttnLRPCombinedComposite
from ._base import LRPConfig, register

CONFIG = register(LRPConfig(
    name="attnlrp_gamma_residual_l1",
    description="Reference but residual_lrp='l1' (sign-preserving, L1-conserving "
                "skip split; signed numerator, |.| denominator). Isolates "
                "residual relevance splitting; trades signed-sum for L1 mass.",
    build=lambda: AttnLRPCombinedComposite(
        alpha=0.5, beta=0.5, linear_gamma=0.25, residual_lrp="l1"),
    site="proj_drop",
    isolates="residual split (sign-preserving, L1-conserving)",
))
