"""AttnLRP composites — ε / γ / combined recipes.

Split out of the former ``transformer_patches.py``. Each composite is a
``LayerMapComposite`` pre-bundled with the ViT canonizers (from
:mod:`zennit_ext.attnlrp_rules`) and, for the combined recipe, the
attention-substitution canonizers from :mod:`zennit_ext.attention_unfolded`.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Gamma, Pass

from zennit_ext.attnlrp_rules import (
    TimmViTCanonizer, AlphaBetaMatmul, ResidualRatio, ResidualL1, Uniform,
)

# ─── 6. Composites — one per recipe (3 total after cleanup) ──────────────────


def _epsilon_layer_map(epsilon: float):
    return [
        (nn.Linear, Epsilon(epsilon=epsilon)),
        (nn.Conv2d, Epsilon(epsilon=epsilon)),
        (nn.GELU, Pass()),
        (nn.LayerNorm, Pass()),
        (nn.Dropout, Pass()),
        (nn.Identity, Pass()),
    ]


def _gamma_layer_map(gamma: float, epsilon: float):
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

    * ``nn.Linear`` → :class:`zennit.rules.Epsilon` (ε-LRP).
    * ``nn.Conv2d`` → :class:`zennit.rules.Epsilon` (covers the patch-embed Conv2d).
    * ``nn.GELU`` / ``nn.LayerNorm`` / ``nn.Dropout`` / ``nn.Identity``
      → :class:`zennit.rules.Pass` (LayerNorm's std stop-gradient is a
      forward rewrite installed by :class:`TimmViTCanonizer`).

    :class:`TimmViTCanonizer` is pre-bundled. Pass extra canonizers via
    the ``canonizers`` kwarg if you also need (for example) a
    ``SequentialMergeBatchNorm`` for a hybrid model.

    Parameters
    ----------
    palrp : bool
        Pass ``True`` to enable PA-LRP on the additive ``x + pos_embed``
        step (Bakish et al. 2025). Default False.
    residual_lrp : {None, 'symmetric', 'ratio'}
        Pass to install a conservative residual rule on Block additions.
        See :class:`TimmBlockResidualCanonizer`.
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
        ε-stabilizer in the LRP denominator. Default 1e-6.
    canonizers : list[Canonizer] | None
        Extra canonizers to apply alongside :class:`TimmViTCanonizer`.
    """

    def __init__(
        self, gamma: float = 0.25, epsilon: float = 1e-6, canonizers=None, *,
        palrp: bool = False, residual_lrp: Optional[str] = None,
    ):
        canonizers = list(canonizers or []) + [
            TimmViTCanonizer(palrp=palrp, residual=residual_lrp is not None, epsilon=epsilon),
        ]
        super().__init__(
            layer_map=_gamma_layer_map(gamma, epsilon), canonizers=canonizers,
        )


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
        PA-LRP on the ``x + pos_embed`` step (Bakish et al. 2025).
        Only relevant for ViTs with absolute pos_embed; no-op on DINOv3
        (RoPE only).
    residual_lrp : {None, 'symmetric', 'ratio', 'l1'}
        Block-level residual rule. ``'ratio'`` recommended (Otsuki |x|/|branch|
        split). ``'l1'`` is the sign-preserving split (``R=R_y·z/(|x|+|branch|)``):
        bounded, continuous, sign follows the operand, conserves ABSOLUTE mass
        locally (``|R_x|+|R_branch|=|R_y|``) instead of the signed sum — so it
        drops LRP's global "sums to the logit" guarantee. Kept for manual
        inspection / research only (see RESEARCH note); ``'ratio'`` is the
        production default. See :class:`~zennit_ext.attnlrp_rules.ResidualL1`.
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
        # Lazy import to avoid the import cycle
        # (attnlrp_composites → attention_unfolded → attnlrp_rules).
        from zennit_ext.attention_unfolded import (
            EvaAttentionSubstitutionCanonizer,
            TimmAttentionSubstitutionCanonizer,
            BilinearMatmul,
            SoftmaxAlongLastDim,
            ScaleByConstant,
            ResidualAdd,
            PosEmbedAdd,
            LayerScaleMul,
        )
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
            # Substitute attention to its unfolded form (forward rewrite).
            # Both canonizers are always registered — each one's isinstance
            # filter no-ops on the other's target class. The per-rule LRP
            # rules below are assigned as zennit Hooks via the layer_map.
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            TimmAttentionSubstitutionCanonizer(block_indices=None),
        ]
        base_map = (
            _gamma_layer_map(linear_gamma, epsilon)
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

__all__ = ["AttnLRPEpsilonComposite", "AttnLRPGammaComposite", "AttnLRPCombinedComposite"]
