"""LRP composites — one module per source paper.

Two reference-verified composites, both grad×input (g-convention). Only the math
lives here; setup provenance is in the experiment journal. :data:`COMPOSITES` is
the single name→class registry — keep the name strings stable (gathered data /
web manifests reference them).
"""
from zennit_extensions.lrp_composites.chefer2021 import CheferLRPComposite
from zennit_extensions.lrp_composites.cp_lrp import CPLRPComposite

#: Both composites use the grad×input (g-convention) backward: uniform read-out
#: heatmap = ``x.grad·x``, per-layer relevance = ``g × activation``
#: (:mod:`experiments.gradinput`).
#: * ``cp_lrp_baseline`` — CP-LRP, the LXT-certified recipe that reproduces
#:   AttnLRP for ViTs (``tutorials/vit_crp/lxt_reference.ipynb``).
#: * ``chefer_lrp`` — Chefer CVPR'21, verified vs the reference NPZs.
COMPOSITES = {
    "cp_lrp_baseline": CPLRPComposite,
    "chefer_lrp": CheferLRPComposite,
}

__all__ = ["CheferLRPComposite", "CPLRPComposite", "COMPOSITES"]
