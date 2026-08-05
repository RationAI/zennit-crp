"""AttnLRP composites — ε / γ / combined / baseline / Chefer recipes.

Each composite is a ``LayerMapComposite`` pre-bundled with the ViT canonizers
from :mod:`zennit_extensions.canonisation.canonizers`; LRP rules live in
:mod:`zennit_extensions.rules`, the hookable attention modules in
:mod:`zennit_extensions.attention_unfolded`.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Gamma, Pass, ZBox, ZPlus

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
    TimmAttentionSubstitutionCanonizer,
    TimmViTCanonizer,
)
from zennit_extensions.rules.attnlrp import EpsilonAdd, MatmulAttnLRP, SoftmaxAttnLRP, Uniform
from zennit_extensions.rules.bajger_contrib import AlphaBetaMatmul, ResidualL1
from zennit_extensions.rules.chefer2021 import CheferAdd, CheferMatmul
from zennit_extensions.rules.residuals_otsuki2024 import ResidualRatio


def _epsilon_layer_map(epsilon: float):
    return [
        (nn.Linear, Epsilon(epsilon=epsilon)),
        (nn.Conv2d, Epsilon(epsilon=epsilon)),
        (nn.GELU, Pass()),
        (nn.LayerNorm, Pass()),
        (nn.Dropout, Pass()),
        (nn.Identity, Pass()),
    ]


def _gamma_layer_map(gamma: float):
    return [
        (nn.Linear, Gamma(gamma=gamma)),
        (nn.Conv2d, Gamma(gamma=gamma)),
        (nn.GELU, Pass()),
        (nn.LayerNorm, Pass()),
        (nn.Dropout, Pass()),
        (nn.Identity, Pass()),
    ]


class AttnLRPEpsilonComposite(LayerMapComposite):
    """Baseline AttnLRP (ε-LRP variant) for ViTs (Achtibat et al. 2024,
    arXiv:2402.05602). Drop-in replacement for ``EpsilonPlusFlat`` when
    attributing through a timm ViT.

    Layer map:

    * ``nn.Linear`` / ``nn.Conv2d`` → :class:`zennit.rules.Epsilon`.
    * ``nn.GELU`` / ``nn.LayerNorm`` / ``nn.Dropout`` / ``nn.Identity``
      → :class:`zennit.rules.Pass`.

    :class:`TimmViTCanonizer` is pre-bundled. Pass extra canonizers via
    the ``canonizers`` kwarg if you also need (for example) a
    ``SequentialMergeBatchNorm`` for a hybrid model.

    Parameters
    ----------
    palrp : bool
        Deprecated no-op (the PA-LRP canonizer was removed); accepted for
        back-compat.
    residual_lrp : {None, 'symmetric', 'ratio'}
        Non-None installs the block-level :class:`TimmBlockResidualCanonizer`
        so residual adds become hookable ``ResidualAdd`` modules; note this
        composite's layer map assigns them no rule (plain add remains).
        See :class:`AttnLRPCombinedComposite` for actual residual rules.
    """

    def __init__(
        self, epsilon: float = 1e-6, canonizers=None, *,
        palrp: bool = False, residual_lrp: Optional[str] = None,
    ):
        canonizers = list(canonizers or []) + [
            TimmViTCanonizer(palrp=palrp, residual=residual_lrp is not None, epsilon=epsilon),
        ]
        super().__init__(layer_map=_epsilon_layer_map(epsilon), canonizers=canonizers)


class AttnLRPGammaComposite(LayerMapComposite):
    """AttnLRP with γ-LRP on ViT linears (Achtibat §3.2.1).

    Same structure as :class:`AttnLRPEpsilonComposite` but maps
    ``nn.Linear`` and ``nn.Conv2d`` to :class:`zennit.rules.Gamma` instead of
    :class:`zennit.rules.Epsilon`. Recommended over the ε-only variant when
    attributing through a deep ViT — γ biases relevance toward positive
    contributions and reduces the gradient-shattering noise that shows
    up as an insertion/deletion-AUC anomaly on finer concept granularities
    under the bare ε-LRP composite.

    Parameters
    ----------
    gamma : float
        γ scaling. Default 0.25 per AttnLRP §3.2.1.
    epsilon : float
        Unused (γ-LRP needs no stabilizer); kept for signature parity with
        :class:`AttnLRPEpsilonComposite`.
    canonizers : list[Canonizer] | None
        Extra canonizers to apply alongside :class:`TimmViTCanonizer`.
    palrp, residual_lrp :
        As on :class:`AttnLRPEpsilonComposite` (see its docs).
    """

    def __init__(
        self, gamma: float = 0.25, epsilon: float = 1e-6, canonizers=None, *,
        palrp: bool = False, residual_lrp: Optional[str] = None,
    ):
        canonizers = list(canonizers or []) + [
            TimmViTCanonizer(palrp=palrp, residual=residual_lrp is not None, epsilon=epsilon),
        ]
        super().__init__(layer_map=_gamma_layer_map(gamma), canonizers=canonizers)


class AttnLRPCombinedComposite(LayerMapComposite):
    """Combination composite — the canonical recipe-builder for AttnLRP.

    Always substitutes the model's attention blocks with their unfolded
    forms (both Eva and standard timm Attention are covered — the
    substitution canonizers each filter on their own target class). The
    AlphaBeta-on-bilinear rule is applied unconditionally on the
    unfolded ``BilinearMatmul`` modules; ``SoftmaxAlongLastDim`` gets
    the AttnLRP identity rule; ``ScaleByConstant`` gets the
    graph-constant identity rule.

    Parameters
    ----------
    alpha, beta : float
        AlphaBeta-on-bilinear hyperparameters (Bach 2015 generalised to
        bilinear; see ``RESEARCH_NOTES.md`` Entry 6). ``α + β = 1`` for
        exact conservation. Common choices:

        * ``α=0.5, β=0.5`` — balanced; tightest magnitude control (default).
        * ``α=1, β=0`` — z+ rule (positive only); slightly looser.
        * ``α=2, β=-1`` — Bach's classical "alpha2beta1"; amplifies
          positive evidence, suppresses negative.
    layerscale_uniform : bool
        Uniform allocation rule on the LayerScale γ multiplication
        (CaiT / Eva blocks).
    linear_gamma : float | None
        If non-None, use γ-LRP on Linears with this γ value instead of
        ε-LRP. Recommended ≤0.25 per AttnLRP §3.2.1.
    epsilon : float
        ε for ε-stabilised rules. Default 1e-6.
    palrp : bool
        When True, map :class:`~zennit_extensions.attention_unfolded.PosEmbedAdd`
        to ``Uniform(factor=2)`` (PA-LRP on ``x + pos_embed``). Currently a
        no-op: no canonizer installs ``PosEmbedAdd`` (the PA-LRP canonizer
        was removed), so the entry only applies if you add the module
        yourself.
    residual_lrp : {None, 'symmetric', 'ratio', 'l1'}
        Block-level residual rule on ``ResidualAdd``. ``'ratio'`` (Otsuki
        |x|/|branch| split) is the production default. ``'l1'`` conserves
        absolute mass (``|R_x|+|R_branch|=|R_y|``) instead of the signed sum —
        research only. ``None`` leaves the add unmapped (non-conservative).
        See :class:`~zennit_extensions.rules.bajger_contrib.ResidualL1`.
    """

    def __init__(
        self, *,
        alpha: float = 0.5,
        beta: float = 0.5,
        layerscale_uniform: bool = False,
        linear_gamma: Optional[float] = None,
        epsilon: float = 1e-6,
        palrp: bool = False,
        residual_lrp: Optional[str] = None,
        canonizers=None,
    ):
        # The residual rule is a LAYER_MAP choice on the single ``ResidualAdd``
        # module — NOT a separate module per rule. ``residual_lrp`` just picks
        # which hook this composite maps ``ResidualAdd`` to.
        res_rules = {
            "ratio": lambda: ResidualRatio(epsilon=epsilon),   # Otsuki |x|/|branch|
            "symmetric": lambda: Uniform(factor=2),            # ½ each
            "l1": lambda: ResidualL1(epsilon=epsilon),         # sign-preserving (research)
        }
        if residual_lrp is not None and residual_lrp not in res_rules:
            raise ValueError(
                f"residual_lrp must be None or one of {sorted(res_rules)}; "
                f"got {residual_lrp!r}")
        canonizers = list(canonizers or []) + [
            TimmViTCanonizer(
                palrp=palrp, residual=residual_lrp is not None,
                layerscale_uniform=layerscale_uniform,
                epsilon=epsilon,
            ),
            # Substitute attention to its unfolded form. Both canonizers'
            # isinstance filters no-op on the other's target class; the LRP
            # rules below are Hooks assigned via the layer_map.
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            TimmAttentionSubstitutionCanonizer(block_indices=None),
        ]
        base_map = (
            _gamma_layer_map(linear_gamma)
            if linear_gamma is not None
            else _epsilon_layer_map(epsilon)
        )
        # Residual / pos-embed / LayerScale rules. ``PosEmbedAdd`` is a subclass
        # of ``ResidualAdd`` and MUST precede it (zennit matches by isinstance,
        # first hit wins). ``residual_lrp=None`` leaves ``ResidualAdd`` unmapped
        # → plain (non-conservative) add.
        residual_entries = []
        if palrp:
            residual_entries.append((PosEmbedAdd, Uniform(factor=2)))
        if residual_lrp is not None:
            residual_entries.append((ResidualAdd, res_rules[residual_lrp]()))
        residual_entries.append((LayerScaleMul, Uniform(factor=2)))
        # AttnLRP rules on the unfolded attention modules, assigned the
        # idiomatic zennit way — a Hook per module type via the layer_map:
        #  * BilinearMatmul (q@kᵀ, weights@v) → AlphaBeta-on-bilinear rule.
        #  * SoftmaxAlongLastDim → identity rule (Eq. 9) = Pass.
        #  * ScaleByConstant → graph-constant identity = Pass.
        layer_map = [
            (BilinearMatmul, AlphaBetaMatmul(alpha=alpha, beta=beta, epsilon=epsilon)),
            (SoftmaxAlongLastDim, Pass()),
            (ScaleByConstant, Pass()),
            *residual_entries,
        ] + base_map
        super().__init__(layer_map=layer_map, canonizers=canonizers)


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
            TimmViTCanonizer(
                palrp=False, residual=True, layerscale_uniform=True, epsilon=epsilon),
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            TimmAttentionSubstitutionCanonizer(block_indices=None),
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
    * GELU / LayerNorm / Dropout → ``Pass``; LayerScale → :class:`Uniform`.

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
            TimmViTCanonizer(
                palrp=False, residual=True, layerscale_uniform=True, epsilon=epsilon),
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            TimmAttentionSubstitutionCanonizer(block_indices=None),
        ]
        layer_map = [
            (BilinearMatmul, CheferMatmul(epsilon=epsilon)),
            (SoftmaxAlongLastDim, Pass()),
            (ScaleByConstant, Pass()),
            (ResidualAdd, CheferAdd(epsilon=epsilon)),
            (LayerScaleMul, Uniform(factor=2)),
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


__all__ = [
    "AttnLRPEpsilonComposite", "AttnLRPGammaComposite", "AttnLRPCombinedComposite",
    "AttnLRPBaselineComposite", "CheferLRPComposite",
]
