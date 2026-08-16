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
    FFNLinear,
    LayerNormDetachedStd,
    LayerScaleMul,
    ResidualAdd,
    ScaleByConstant,
    SoftmaxAlongLastDim,
)
from zennit_extensions.canonisation.canonizers import (
    EvaAttentionSubstitutionCanonizer,
    EvaBlockResidualCanonizer,
    FFNLinearSubstitutionCanonizer,
    LayerNormSubstitutionCanonizer,
    VanillaViTAttentionSubstitutionCanonizer,
    VanillaViTBlockResidualCanonizer,
    VanillaViTPosEmbedCanonizer,
)
from zennit_extensions.rules.attnlrp import (
    EpsilonAdd,
    LayerNormEpsilon,
    MatmulAttnLRP,
    SoftmaxAttnLRP,
)


class AttnLRPBaselineComposite(LayerMapComposite):
    """AttnLRP exactly as published (Achtibat et al., ICML 2024) — the faithful
    reference recipe, one rule per operation from the paper:

    * bilinears ``q@kᵀ`` / ``attn@v`` → :class:`MatmulAttnLRP` (Eq. 15);
    * softmax → :class:`SoftmaxAttnLRP` (Prop. 3.1);
    * residual adds → :class:`EpsilonAdd` (standard ε add, LXT ``add2``);
    * GELU → ``Pass`` (Eq. 9 elementwise identity = pure pass-through);
    * LayerNorm → substituted by :class:`LayerNormSubstitutionCanonizer` with
      :class:`LayerNormDetachedStd` (σ detached, Prop. 3.4 identity on ``x/σ``)
      and attributed by :class:`LayerNormEpsilon` (ε on the remaining affine
      map, §3.3.3 — the LXT ``layer_norm_grad_fn`` treatment). β handling is
      selectable via ``layernorm_bias_mode`` (A.2.1: ``'absorb'`` recommended
      default / ``'omit'`` / ``'distribute'``); any LayerNorm subclass the
      canonizer does not substitute falls back to ``Pass``;
    * FFN linears → γ-LRP (Table B.5, γ=0.05) via the :class:`FFNLinear` marker
      installed by :class:`FFNLinearSubstitutionCanonizer`; every unmarked
      ``nn.Linear`` — attention projections ``W_q/W_k/W_v`` (``attn.qkv``),
      ``W_o`` (``attn.proj``), classifier head — → ε-LRP (Table B.5);
      patch-embed conv → γ-LRP (γ=0.25);
    * LayerScale γ-multiply → ``Pass``: a bias-free elementwise linear op, whose
      ε-attribution ``R·|y|/(|y|+ε) ≈ R`` collapses to the identity (the paper
      attributes affine weighting via the ε-rule, §3.3.3; Eq. 14 does not apply —
      the uniform rule presupposes N differentiable input factors, γ is a
      parameter).

    The FFN-γ / projection-ε split (Table B.5) is purely type-based:
    :class:`FFNLinearSubstitutionCanonizer` retypes FFN linears as
    :class:`FFNLinear`, whose ``layer_map`` entry precedes ``nn.Linear``
    (isinstance would match both), and every unmarked linear falls to ε.

    Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
    Transformers', https://proceedings.mlr.press/v235/achtibat24a.html
    """

    def __init__(
        self, *, ffn_gamma: float = 0.05, conv_gamma: float = 0.25,
        epsilon: float = 1e-6, softmax_bias_mode: str = "absorb",
        layernorm_bias_mode: str = "absorb", canonizers=None,
    ):
        canonizers = list(canonizers or []) + [
            VanillaViTBlockResidualCanonizer(),
            EvaBlockResidualCanonizer(layerscale_uniform=True),
            VanillaViTPosEmbedCanonizer(),
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
            LayerNormSubstitutionCanonizer(),
            FFNLinearSubstitutionCanonizer(),
        ]
        layer_map = [
            (BilinearMatmul, MatmulAttnLRP(epsilon=epsilon)),
            (SoftmaxAlongLastDim, SoftmaxAttnLRP(bias_mode=softmax_bias_mode, epsilon=epsilon)),
            (LayerNormDetachedStd, LayerNormEpsilon(epsilon=epsilon, bias_mode=layernorm_bias_mode)),
            (ScaleByConstant, Pass()),
            (ResidualAdd, EpsilonAdd(epsilon=epsilon)),   # PosEmbedAdd is a standalone type: unmapped here ⇒ plain add (default unchanged); PA-LRP opt-in maps it to PosEmbedSink
            (LayerScaleMul, Pass()),
            (nn.GELU, Pass()),   # AttnLRP Eq. 9 identity = pure pass-through
            (nn.LayerNorm, Pass()),
            (nn.Dropout, Pass()),
            (FFNLinear, Gamma(gamma=ffn_gamma)),   # Table B.5 FFN-γ; MUST precede nn.Linear
            (nn.Linear, Epsilon(epsilon=epsilon)),  # qkv / proj / head → ε
            (nn.Conv2d, Gamma(gamma=conv_gamma)),
            (nn.Identity, Pass()),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)
