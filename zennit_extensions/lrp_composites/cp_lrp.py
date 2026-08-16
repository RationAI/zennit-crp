"""CP-LRP composite — value-path-only conservative propagation.

Attention is unfolded, but the Q and K probes carry a ``StopGradient`` so
relevance flows only through the value path: the attention weights are treated
as constants and the softmax / Q·Kᵀ bilinear conduct no relevance. Residual adds
use the Otsuki ratio split; linears γ-LRP.

Sourced from 'XAI for Transformers: Better Explanations through Conservative
Propagation', https://proceedings.mlr.press/v162/ali22a.html
"""
from __future__ import annotations

import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma, Pass

from zennit_extensions.attention_unfolded import (
    KInspectionLayer,
    LayerNormDetachedStd,
    LayerScaleMul,
    QInspectionLayer,
    ResidualAdd,
    ScaleByConstant,
    SoftmaxAlongLastDim,
)
from zennit_extensions.canonisation.canonizers import (
    EvaAttentionSubstitutionCanonizer,
    EvaBlockResidualCanonizer,
    LayerNormSubstitutionCanonizer,
    VanillaViTAttentionSubstitutionCanonizer,
    VanillaViTBlockResidualCanonizer,
    VanillaViTPosEmbedCanonizer,
)
from zennit_extensions.cp_lrp import StopGradient
from zennit_extensions.rules.attnlrp import LayerNormEpsilon
from zennit_extensions.rules.residuals_otsuki2024 import ResidualRatio


class CPLRPComposite(LayerMapComposite):
    """CP-LRP (Ali et al., 2022): StopGradient on the Q/K probes, so the softmax
    is a graph constant and relevance flows via ``context = attn @ v`` only.
    γ=0.10 linears / γ=0.25 patch conv; Otsuki ratio residual split; LayerScale
    → ``Pass`` (bias-free elementwise linear γ-multiply, ε-attribution ≈
    identity). LayerNorm is handled the Ali-et-al. way (the origin of the
    σ-detach heuristic): :class:`LayerNormSubstitutionCanonizer` +
    :class:`LayerNormEpsilon`, with ``layernorm_bias_mode`` selecting the β
    handling; unsubstituted LayerNorm subclasses fall back to ``Pass``.
    """

    def __init__(self, *, linear_gamma: float = 0.10, conv_gamma: float = 0.25,
                 epsilon: float = 1e-6, layernorm_bias_mode: str = "absorb",
                 canonizers=None):
        canonizers = list(canonizers or []) + [
            VanillaViTBlockResidualCanonizer(),
            EvaBlockResidualCanonizer(layerscale_uniform=True),
            VanillaViTPosEmbedCanonizer(),
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
            LayerNormSubstitutionCanonizer(),
        ]
        layer_map = [
            (nn.Linear, Gamma(gamma=linear_gamma)),
            (nn.Conv2d, Gamma(gamma=conv_gamma)),
            (nn.GELU, Pass()),
            (LayerNormDetachedStd, LayerNormEpsilon(epsilon=epsilon, bias_mode=layernorm_bias_mode)),
            (nn.LayerNorm, Pass()),
            (nn.Dropout, Pass()),
            (SoftmaxAlongLastDim, Pass()),
            (ScaleByConstant, Pass()),
            (ResidualAdd, ResidualRatio(epsilon=epsilon)),
            (LayerScaleMul, Pass()),
            (QInspectionLayer, StopGradient()),
            (KInspectionLayer, StopGradient()),
            (nn.Identity, Pass()),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)
