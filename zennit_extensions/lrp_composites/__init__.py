"""LRP composites — one module per source paper.

Only the math lives here. Experiments track the composites they use in their
own explicit name→class dicts; setup provenance is recorded in the experiment
journal, not in source code.
"""
from zennit_extensions.lrp_composites.attnlrp import AttnLRPBaselineComposite
from zennit_extensions.lrp_composites.chefer2021 import CheferLRPComposite
from zennit_extensions.lrp_composites.cp_lrp import CPLRPComposite

__all__ = ["AttnLRPBaselineComposite", "CheferLRPComposite", "CPLRPComposite"]
