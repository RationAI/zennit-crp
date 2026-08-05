"""zennit_extensions — LRP / AttnLRP primitives for transformers.

CRP-independent extensions to zennit: unfolded-attention module primitives
(:mod:`attention_unfolded`), LRP rule kernels (:mod:`rules`), forward-graph
canonizers (:mod:`canonisation.canonizers`), and one composite per source paper
(:mod:`lrp_composites`). Rules and canonizers are imported from their own
modules; only attention primitives and composites are re-exported here.
"""
from zennit_extensions.attention_unfolded import *  # noqa: F401,F403
from zennit_extensions.lrp_composites import *  # noqa: F401,F403
