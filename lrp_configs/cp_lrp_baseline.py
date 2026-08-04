"""``cp_lrp_baseline`` — the LXT-analogue value-path-only recipe (the default).

Attention is unfolded, but the Q and K probes carry a ``StopGradient`` so
relevance flows **only through the value path** (CP-LRP): the attention weights
are treated as constants and the softmax / Q·Kᵀ bilinear conduct no relevance.
It produces the cleanest, sharpest heatmaps (hence its use in the walkthrough),
at the cost of attributing nothing to *why* a token was attended to.

Relative to :mod:`lrp_configs.attnlrp_baseline` (AttnLRP as published), the only
changed building block is the attention path: StopGradient on Q/K here vs the
matmul rule (Eq. 15) + softmax rule (Prop. 3.1) there. That single difference is
the headline comparison — does attributing the query/key (attention-formation)
path help or just add noise?
"""
from __future__ import annotations

import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma, Pass

from zennit_extentions import (
    QInspectionLayer, KInspectionLayer, StopGradient,
    SoftmaxAlongLastDim, ScaleByConstant, ResidualAdd, LayerScaleMul,
    ResidualRatio, Uniform,
    LayerNormForwardCanonizer, DropoutPassthroughCanonizer,
    TimmBlockResidualCanonizer, EvaBlockResidualCanonizer,
    EvaAttentionSubstitutionCanonizer, TimmAttentionSubstitutionCanonizer,
)
from ._base import LRPConfig, register


def _build() -> LayerMapComposite:
    layer_map = [
        (nn.Linear, Gamma(gamma=0.10)), (nn.Conv2d, Gamma(gamma=0.25)),
        (nn.GELU, Pass()), (nn.LayerNorm, Pass()), (nn.Dropout, Pass()),
        (SoftmaxAlongLastDim, Pass()), (ScaleByConstant, Pass()),
        (ResidualAdd, ResidualRatio(epsilon=1e-6)),
        (LayerScaleMul, Uniform(factor=2)),
        (QInspectionLayer, StopGradient()), (KInspectionLayer, StopGradient()),
        (nn.Identity, Pass()),
    ]
    canonizers = [
        LayerNormForwardCanonizer(), DropoutPassthroughCanonizer(),
        TimmBlockResidualCanonizer(),
        EvaBlockResidualCanonizer(layerscale_uniform=True),
        EvaAttentionSubstitutionCanonizer(block_indices=None),
        TimmAttentionSubstitutionCanonizer(block_indices=None),
    ]
    return LayerMapComposite(layer_map=layer_map, canonizers=canonizers)


CONFIG = register(LRPConfig(
    name="cp_lrp_baseline",
    description="Value-path only (StopGradient on Q/K), γ=0.10 linears, ratio "
                "residual split. The LXT-analogue baseline — cleanest heatmaps.",
    build=_build,
    site="proj_drop",
    isolates="",  # the baseline itself
))
