"""Chefer et al. (CVPR 2021) composite — the LRP relevance stage of
'Transformer Interpretability Beyond Attention Visualization', **code-exact**.

Sourced from https://doi.org/10.1109/CVPR46437.2021.00084 — released code at
https://github.com/hila-chefer/Transformer-Explainability (commit c3e578f).

This composite reproduces the authors' *released code* (not the paper):
bilinears ``q@kᵀ`` / ``attn@v`` use the plain z-rule with ÷2 (their
``einsum`` RelPropSimple + ``cam /= 2``); residual adds use z-rule + global
absolute-mass renormalization (their ``Add`` layer); softmax / LayerNorm /
GELU / Dropout are identity (``Pass``); hidden linears use z⁺ (zennit
``ZPlus`` with ``zero_params=['bias']`` — their ``F.linear(x, w)`` excludes
bias from the denominator).

For ``transformer_attribution`` the attention-relevance rollout reads ``R_A``
at the softmax (above every block), so nothing below the attention softmax is
read — the patch-embed conv is left to zennit's default handling (pixel-space
Chefer, ``method="full"``, is out of scope and has no reference ground truth).
"""
from __future__ import annotations

import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import Pass, ZPlus

from zennit_extensions.attention_unfolded import (
    BilinearMatmul,
    LayerScaleMul,
    PosEmbedAdd,
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
from zennit_extensions.rules.chefer2021 import (
    CheferAdd,
    CheferMatmul,
)


class CheferLRPComposite(LayerMapComposite):
    """Chefer et al. (CVPR 2021) LRP composite — **code-exact** (released code).

    Rule set (mirrors ``ViT_LRP.py`` relprop chain):

    * ``BilinearMatmul`` (qk_scores, context) → :class:`CheferMatmul`
      (z-rule + ÷2; ``ViT_LRP.py:160-173``).
    * ``SoftmaxAlongLastDim`` → ``Pass`` (their ``Softmax`` relprop = identity).
    * ``ResidualAdd`` / ``PosEmbedAdd`` → :class:`CheferAdd`
      (z-rule + global abs-mass renorm; their ``Add`` layer).
    * ``ScaleByConstant`` / ``LayerScaleMul`` / ``GELU`` / ``LayerNorm`` /
      ``Dropout`` / ``Identity`` → ``Pass`` (all identity in their relprop).
    * ``nn.Linear`` → ``ZPlus(stabilizer=1e-9, zero_params=['bias'])`` — their
      ``Linear.relprop(α=1)`` is exactly z⁺ with bias excluded from the
      denominator (``F.linear(x, w)`` without bias).

    The patch-embed ``nn.Conv2d`` is intentionally unmapped: the reference
    ``transformer_attribution`` never propagates below the attention softmax, so
    no conv rule is exercised (pixel-space ``method="full"`` is out of scope).

    Sourced from 'Transformer Interpretability Beyond Attention Visualization',
    https://doi.org/10.1109/CVPR46437.2021.00084
    """

    def __init__(self, *, stabilizer: float = 1e-9, canonizers=None):
        canonizers = list(canonizers or []) + [
            VanillaViTBlockResidualCanonizer(),
            EvaBlockResidualCanonizer(layerscale_uniform=True),
            VanillaViTPosEmbedCanonizer(),
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
        ]
        layer_map = [
            (BilinearMatmul, CheferMatmul()),
            (SoftmaxAlongLastDim, Pass()),
            (ScaleByConstant, Pass()),
            (ResidualAdd, CheferAdd()),
            (PosEmbedAdd, CheferAdd()),
            (LayerScaleMul, Pass()),
            (nn.GELU, Pass()),
            (nn.LayerNorm, Pass()),
            (nn.Dropout, Pass()),
            (nn.Linear, ZPlus(stabilizer=stabilizer, zero_params=["bias"])),
            (nn.Identity, Pass()),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)
