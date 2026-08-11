"""AttnLRP composite — the recipe exactly as published.

Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
Transformers', https://proceedings.mlr.press/v235/achtibat24a.html
"""
from __future__ import annotations

import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Gamma, Pass

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
)
from zennit_extensions.rules.attnlrp import EpsilonAdd, MatmulAttnLRP, SoftmaxAttnLRP, Uniform


class AttnLRPBaselineComposite(LayerMapComposite):
    """AttnLRP exactly as published (Achtibat et al., ICML 2024) — the faithful
    reference recipe, one rule per operation from the paper:

    * bilinears ``q@kᵀ`` / ``attn@v`` → :class:`MatmulAttnLRP` (Eq. 15);
    * softmax → :class:`SoftmaxAttnLRP` (Prop. 3.1);
    * residual adds → :class:`EpsilonAdd` (standard ε add, LXT ``add2``);
    * GELU → ``Pass`` (Eq. 9 elementwise identity = pure pass-through); LayerNorm → ``Pass`` (Prop. 3.4
      pass-through, no bias absorbs relevance);
    * FFN linears (``mlp.*``) → γ-LRP (Table 4, γ=0.05); attention projections
      ``W_q/W_k/W_v`` (``attn.qkv``) and ``W_o`` (``attn.proj``) and the classifier
      head → ε-LRP (Table 4); patch-embed conv → γ-LRP (γ=0.25);
    * LayerScale γ-multiply → :class:`Uniform` (Eq. 14).

    The FFN-γ / projection-ε split is by module name (the paper's Table 4
    distinguishes them; ``LayerMapComposite`` matches by type alone), so
    :meth:`mapping` special-cases ``nn.Linear`` before the type map.

    Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
    Transformers', https://proceedings.mlr.press/v235/achtibat24a.html
    """

    def __init__(
        self, *, ffn_gamma: float = 0.05, conv_gamma: float = 0.25,
        epsilon: float = 1e-6, canonizers=None,
    ):
        self._ffn_gamma = ffn_gamma
        self._epsilon = epsilon
        canonizers = list(canonizers or []) + [
            VanillaViTBlockResidualCanonizer(),
            EvaBlockResidualCanonizer(layerscale_uniform=True),
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
        ]
        layer_map = [
            (BilinearMatmul, MatmulAttnLRP(epsilon=epsilon)),
            (SoftmaxAlongLastDim, SoftmaxAttnLRP()),
            (ScaleByConstant, Pass()),
            (ResidualAdd, EpsilonAdd(epsilon=epsilon)),   # PosEmbedAdd inherits this
            (LayerScaleMul, Uniform(factor=2)),
            (nn.GELU, Pass()),   # AttnLRP Eq. 9 identity = pure pass-through
            (nn.LayerNorm, Pass()),
            (nn.Dropout, Pass()),
            (nn.Conv2d, Gamma(gamma=conv_gamma)),
            (nn.Identity, Pass()),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)

    def mapping(self, ctx, name, module):
        # FFN linears → γ; attention projections (qkv/proj) + head → ε (Table 4).
        if isinstance(module, nn.Linear):
            return (Gamma(gamma=self._ffn_gamma) if ".mlp." in name
                    else Epsilon(epsilon=self._epsilon))
        return super().mapping(ctx, name, module)
