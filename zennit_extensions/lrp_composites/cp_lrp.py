"""CP-LRP composite — grad×input (g-convention), LXT-parity certified.

The backward stream carries ``g`` with relevance ``R = g ⊙ x`` at every tensor;
the input heatmap is ``x.grad * x`` (one line, no adapter). For CRP the per-layer
relevance is ``g × activation`` (the read-outs in :mod:`experiments.gradinput`).

This is the recipe of ``tutorials/vit_crp/lxt_reference.ipynb`` (Path A / A2):
|Δ| ≈ 1e-8 vs LXT on the fused torchvision model; cross-skeleton r = 1.0 under the
unified canonization.

* patch-embed conv → :class:`GammaGradInput` (γ=0.25)
* FFN linears (:class:`FFNLinear` marker) + classifier ``head`` →
  :class:`GammaGradInput` (γ=0.10)
* bare ``nn.Linear`` (qkv / proj) → no hook = ε in the g-convention
* GELU → :class:`IdentityGradTimesInput` (``y/(x+ε)``)
* Q/K probes → :class:`StopGradient` (CP-LRP AH-rule: softmax a graph constant)
* LayerNorm → σ-detach substitution, rule-free (autograd in g = ε-rule)
* residual adds → rule-free (autograd in g = conserving ε-split)

Sourced from 'XAI for Transformers: Better Explanations through Conservative
Propagation', https://proceedings.mlr.press/v162/ali22a.html (AH-rule via
StopGradient + Gradient×Input extraction).
"""
from __future__ import annotations

import torch.nn as nn
from zennit.core import Composite

from zennit_extensions.attention_unfolded import (
    FFNLinear,
    KInspectionLayer,
    QInspectionLayer,
)
from zennit_extensions.canonisation.canonizers import (
    FFNLinearSubstitutionCanonizer,
    LayerNormSubstitutionCanonizer,
    VanillaViTAttentionSubstitutionCanonizer,
)
from zennit_extensions.cp_lrp import StopGradient
from zennit_extensions.rules.attnlrp import GammaGradInput, IdentityGradTimesInput


class CPLRPComposite(Composite):
    """CP-LRP in the grad×input convention (timm/torchvision ViT skeletons)."""

    def __init__(self, *, conv_gamma: float = 0.25, linear_gamma: float = 0.10,
                 gelu_epsilon: float = 1e-10, canonizers=None):
        self._gamma_conv = GammaGradInput(gamma=conv_gamma)
        self._gamma_lin = GammaGradInput(gamma=linear_gamma)
        self._gelu = IdentityGradTimesInput(epsilon=gelu_epsilon)
        self._stop = StopGradient()
        canonizers = list(canonizers or []) + [
            LayerNormSubstitutionCanonizer(),
            FFNLinearSubstitutionCanonizer(),
            VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
        ]
        super().__init__(module_map=self._module_map, canonizers=canonizers)

    def _module_map(self, ctx, name, module):
        if isinstance(module, nn.Conv2d):
            return self._gamma_conv
        if isinstance(module, FFNLinear):
            return self._gamma_lin
        if isinstance(module, nn.Linear) and name.split(".")[-1] == "head":
            return self._gamma_lin
        if isinstance(module, nn.GELU):
            return self._gelu
        if isinstance(module, (QInspectionLayer, KInspectionLayer)):
            return self._stop
        return None
