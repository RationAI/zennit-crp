"""``chefer_lrp`` — the LRP relevance stage of Chefer et al. (CVPR 2021),
"Transformer Interpretability Beyond Attention Visualization", reproduced exactly
as defined in that paper. This is the engine that feeds the attention-map
relevance ``R_A`` to the Chefer row of the insertion-deletion benchmark; the
gradient×relevance rollout (their Eq. 13-14) sits in the attribution method, not
here.

Not one of the two LRP recipes under comparison (that is ``cp_lrp_baseline`` vs
``attnlrp_baseline``) — it exists solely to run the Chefer baseline faithfully.

See :class:`zennit_ext.attnlrp_composites.CheferLRPComposite`.
"""
from __future__ import annotations

from zennit_extensions.attnlrp_composites import CheferLRPComposite
from ._base import LRPConfig, register

CONFIG = register(LRPConfig(
    name="chefer_lrp",
    description="Chefer et al. (CVPR 2021) LRP: z⁺ linears (Eq. 4), z-rule + Eq. 9 "
                "normalization on attention matmuls and skip-adds, softmax not "
                "propagated, z^B box rule on the patch conv. Engine for the Chefer "
                "benchmark row; also usable standalone for input-pixel maps.",
    build=lambda: CheferLRPComposite(),
    site="proj_drop",
    isolates="",  # baseline for a competing method, not an ablation of ours
))
