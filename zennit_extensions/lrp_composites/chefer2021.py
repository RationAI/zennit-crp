"""Chefer et al. (CVPR 2021) composite — the LRP relevance stage of
'Transformer Interpretability Beyond Attention Visualization'.

Sourced from https://doi.org/10.1109/CVPR46437.2021.00084
"""
from __future__ import annotations

import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import Pass, ZBox, ZPlus

from zennit_extensions.attention_unfolded import (
    BilinearMatmul,
    LayerScaleMul,
    ResidualAdd,
    ScaleByConstant,
    SoftmaxAlongLastDim,
)
from zennit_extensions.canonisation.canonizers import (
    EvaAttentionSubstitutionCanonizer,
    EvaBlockResidualCanonizer,
    VanillaViTAttentionSubstitutionCanonizer,
    VanillaViTBlockResidualCanonizer,
    VanillaViTPosEmbedCanonizer,
)
from zennit_extensions.rules.chefer2021 import CheferAdd, CheferMatmul


class CheferLRPComposite(LayerMapComposite):
    """Chefer et al. (CVPR 2021) 'Transformer Interpretability Beyond Attention
    Visualization' — the LRP relevance stage exactly as defined in that paper,
    the engine for the Chefer attention row of the insertion-deletion benchmark:

    * hidden linears → z⁺ rule (α=1, β=0, positive contributions only; Eq. 4);
    * bilinears ``q@kᵀ`` / ``attn@v`` → :class:`CheferMatmul` (z-rule + Eq. 9
      conservation normalisation — the paper normalises attention matmuls too);
    * softmax → ``Pass``: the paper does NOT propagate LRP through the softmax;
      attention is aggregated by the gradient×relevance rollout (Eq. 13-14),
      which lives in the attribution method, not this composite;
    * residual adds → :class:`CheferAdd` (z-rule + Eq. 9 normalisation, their
      ``Add`` layer);
    * GELU / LayerNorm / Dropout / LayerScale → ``Pass`` (LayerScale is a
      bias-free elementwise linear γ-multiply; its ε-attribution ≈ identity).

    Standalone input-pixel attribution: the patch-embed conv (the first,
    pixel-space layer) uses the z^B box rule (:class:`zennit.rules.ZBox`),
    Chefer's first-layer ``Conv2d`` branch, with pixel bounds ``low`` / ``high``
    — set these to the min / max of the *normalised* input for exact conservation
    (defaults ``-3`` / ``3`` cover a standard mean/std normalisation). For the
    benchmark's Chefer row (which reads ``R_A`` at the softmax, never pixels) the
    conv rule is irrelevant.

    Sourced from 'Transformer Interpretability Beyond Attention Visualization',
    https://doi.org/10.1109/CVPR46437.2021.00084
    """

    def __init__(self, *, epsilon: float = 1e-6, low: float = -3.0,
                 high: float = 3.0, canonizers=None):
        self._zbox = ZBox(low=low, high=high)   # first (pixel-space) conv
        canonizers = list(canonizers or []) + [
            VanillaViTBlockResidualCanonizer(),
            EvaBlockResidualCanonizer(layerscale_uniform=True),
            VanillaViTPosEmbedCanonizer(),
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
        ]
        layer_map = [
            (BilinearMatmul, CheferMatmul(epsilon=epsilon)),
            (SoftmaxAlongLastDim, Pass()),
            (ScaleByConstant, Pass()),
            (ResidualAdd, CheferAdd(epsilon=epsilon)),
            (LayerScaleMul, Pass()),
            (nn.GELU, Pass()),
            (nn.LayerNorm, Pass()),
            (nn.Dropout, Pass()),
            (nn.Linear, ZPlus()),
            (nn.Conv2d, ZPlus()),
            (nn.Identity, Pass()),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)

    def mapping(self, ctx, name, module):
        # Chefer's first-layer (pixel-space) conv uses the z^B box rule; the
        # patch-embed conv is the only Conv2d in a ViT, so match it by type.
        if isinstance(module, nn.Conv2d):
            return self._zbox
        return super().mapping(ctx, name, module)
