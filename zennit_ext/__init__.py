"""zennit_ext — LRP / AttnLRP primitives for transformers.

CRP-independent extensions to zennit: unfolded-attention module
primitives + substitution canonizers (:mod:`attention_unfolded`), AttnLRP
rule/canonizer kernels (:mod:`attnlrp_rules`), and composite recipes
(:mod:`attnlrp_composites`).
"""
from zennit_ext.attention_unfolded import *  # noqa: F401,F403
from zennit_ext.attnlrp_rules import *  # noqa: F401,F403
from zennit_ext.attnlrp_composites import *  # noqa: F401,F403
