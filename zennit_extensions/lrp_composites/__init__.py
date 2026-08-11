"""LRP composites — one module per source paper.

Only the math lives here; setup provenance is recorded in the experiment
journal, not in source code. :data:`COMPOSITES` is the single name→class
registry — gathered data, web manifests and CLIs reference these name strings,
so keep them stable.
"""
from zennit_extensions.lrp_composites.attnlrp import AttnLRPBaselineComposite
from zennit_extensions.lrp_composites.chefer2021 import CheferLRPComposite
from zennit_extensions.lrp_composites.cp_lrp import CPLRPComposite

COMPOSITES = {
    "cp_lrp_baseline": CPLRPComposite,
    "attnlrp_baseline": AttnLRPBaselineComposite,
    "chefer_lrp": CheferLRPComposite,
}

__all__ = [
    "AttnLRPBaselineComposite", "CheferLRPComposite", "CPLRPComposite",
    "COMPOSITES",
]
