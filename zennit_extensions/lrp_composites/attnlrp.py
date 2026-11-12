"""AttnLRP composite — grad×input (g-convention), for the concept-detector bench.

The AttnLRP rules have lived in :mod:`zennit_extensions.rules.attnlrp` with unit
tests since the rules were ported, but nothing ever wired them into a composite,
so the method the benchmark is positioned against has never actually been run on
it. This is that composite.

Convention. The benchmark reads relevance as ``g ⊙ activation`` through
:class:`experiments.gradinput.GradTimesInputAttribution`, so every rule here is
the grad×input form. That matters for one rule in particular:

**The softmax needs no hook.** AttnLRP's softmax rule (Prop. 3.1) is
``R_i = x_i (R_i - s_i Σ_j R_j)`` in the relevance convention. Substituting
``R = g ⊙ y`` with ``y = s`` gives

    x_i g_i^in = x_i (g_i s_i - s_i Σ_j g_j s_j)
         g_i^in = s_i (g_i - Σ_j g_j s_j)

which is exactly the Jacobian-vector product of the softmax. In the g-convention
AttnLRP's softmax rule *is* ordinary autograd, so attaching a hook would at best
duplicate it. :func:`softmax_rule_is_autograd` checks this numerically rather
than leaving it as an argument on paper.

Rules, by module:

* bilinear matmuls (scores ``q·kᵀ`` and context ``w·v``) →
  :class:`MatmulAttnLRPGradTimesInput`, the Eq. 15 rule that splits relevance
  between two input-dependent factors instead of sending it all to one;
* patch-embed conv and FFN linears → :class:`GammaGradInput`, matching CP-LRP's
  treatment so the two differ *only* inside attention, which is the comparison
  the paper wants;
* GELU → :class:`IdentityGradTimesInput`;
* softmax, LayerNorm, residual adds, qkv/proj linears → rule-free: autograd in
  the g-convention is already the intended rule (ε for linears, the identity
  argument above for softmax, a conserving split for the adds).

Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
Transformers', Achtibat et al., ICML 2024 (PMLR 235).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from zennit.core import Composite

from zennit_extensions.attention_unfolded import BilinearMatmul, FFNLinear
from zennit_extensions.canonisation.canonizers import (
    EvaAttentionSubstitutionCanonizer,
    FFNLinearSubstitutionCanonizer,
    LayerNormSubstitutionCanonizer,
    VanillaViTAttentionSubstitutionCanonizer,
)
from zennit_extensions.rules.attnlrp import (
    GammaGradInput,
    IdentityGradTimesInput,
    MatmulAttnLRPGradTimesInput,
)


class AttnLRPComposite(Composite):
    """AttnLRP in the grad×input convention (timm ViT + Eva/DINOv3 skeletons)."""

    def __init__(self, *, conv_gamma: float = 0.25, linear_gamma: float = 0.10,
                 gelu_epsilon: float = 1e-10, matmul_epsilon: float = 1e-6,
                 canonizers=None):
        self._gamma_conv = GammaGradInput(gamma=conv_gamma)
        self._gamma_lin = GammaGradInput(gamma=linear_gamma)
        self._gelu = IdentityGradTimesInput(epsilon=gelu_epsilon)
        self._matmul = MatmulAttnLRPGradTimesInput(epsilon=matmul_epsilon)
        canonizers = list(canonizers or []) + [
            LayerNormSubstitutionCanonizer(),
            FFNLinearSubstitutionCanonizer(),
            VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
            EvaAttentionSubstitutionCanonizer(block_indices=None),
        ]
        super().__init__(module_map=self._module_map, canonizers=canonizers)

    def _module_map(self, ctx, name, module):
        if isinstance(module, BilinearMatmul):
            return self._matmul
        if isinstance(module, nn.Conv2d):
            return self._gamma_conv
        if isinstance(module, FFNLinear):
            return self._gamma_lin
        if isinstance(module, nn.Linear) and name.split(".")[-1] == "head":
            return self._gamma_lin
        if isinstance(module, nn.GELU):
            return self._gelu
        return None


def softmax_rule_is_autograd(n: int = 64, seed: int = 0, atol: float = 1e-10) -> float:
    """Check the claim in the module docstring, numerically.

    Returns the largest absolute difference between AttnLRP's softmax rule
    written in the g-convention and plain autograd through the softmax.
    """
    torch.manual_seed(seed)
    x = torch.randn(4, n, dtype=torch.float64, requires_grad=True)
    s = torch.softmax(x, dim=-1)
    g_out = torch.randn(4, n, dtype=torch.float64)
    # plain autograd
    (auto,) = torch.autograd.grad(s, x, grad_outputs=g_out)
    # AttnLRP Prop. 3.1 in relevance form, converted back to g
    rel_out = g_out * s
    rel_in = x * (rel_out - s * rel_out.sum(dim=-1, keepdim=True))
    g_in = rel_in / x
    return float((auto - g_in).detach().abs().max())
