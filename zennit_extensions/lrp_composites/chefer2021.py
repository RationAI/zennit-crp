"""Chefer et al. (CVPR 2021) composite — the LRP relevance stage of
'Transformer Interpretability Beyond Attention Visualization', **code-exact**,
in the grad×input (g-convention) convention.

Sourced from https://doi.org/10.1109/CVPR46437.2021.00084 — released code at
https://github.com/hila-chefer/Transformer-Explainability (commit c3e578f).

Every rule runs in the g-convention (backward stream carries ``g`` with
``R = g⊙x``; heatmap = ``x.grad·x``; per-layer relevance = ``g×activation``). The
Chefer relevance splits are unchanged — the g-wrappers only convert at the rule
boundaries — so the reference reproduction (``tutorials/vit_crp/chefer_reference.ipynb``,
pearson r ≥ 0.999998) is preserved.

Rule set (mirrors ``ViT_LRP.py`` relprop chain):

* ``BilinearMatmul`` (qk_scores, context) → :class:`CheferMatmulGradInput`
  (z-rule + ÷2).
* ``ResidualAdd`` / ``PosEmbedAdd`` → :class:`CheferAddGradInput`
  (z-rule + global abs-mass renorm).
* ``nn.Linear`` → :class:`ZPlusGradInput` (z⁺, bias excluded).
* softmax / LayerNorm / GELU / Dropout / LayerScale / scale / Identity →
  :class:`IdentityGradTimesInput` (their relprop is identity-relevance).
* patch-embed conv → unmapped = ε in the g-convention (off the reproduced
  ``transformer_attribution`` path — R_A is read at the softmax).
"""
from __future__ import annotations

import torch.nn as nn
from zennit.core import Composite

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
from zennit_extensions.rules.attnlrp import IdentityGradTimesInput
from zennit_extensions.rules.chefer2021 import (
    CheferAddGradInput,
    CheferMatmulGradInput,
    ZPlusGradInput,
)

_IDENTITY_RELEVANCE = (SoftmaxAlongLastDim, ScaleByConstant, LayerScaleMul,
                       nn.GELU, nn.LayerNorm, nn.Dropout, nn.Identity)


class CheferLRPComposite(Composite):
    """Chefer et al. (CVPR 2021) LRP composite — grad×input, code-exact.

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
        self._matmul = CheferMatmulGradInput()
        self._add = CheferAddGradInput()
        self._linear = ZPlusGradInput(stabilizer=stabilizer, zero_params=["bias"])
        self._identity = IdentityGradTimesInput(epsilon=1e-10)
        super().__init__(module_map=self._module_map, canonizers=canonizers)

    def _module_map(self, ctx, name, module):
        if isinstance(module, BilinearMatmul):
            return self._matmul
        if isinstance(module, (ResidualAdd, PosEmbedAdd)):
            return self._add
        if isinstance(module, nn.Linear):
            return self._linear
        if isinstance(module, _IDENTITY_RELEVANCE):
            return self._identity
        return None
