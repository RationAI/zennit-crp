"""AttnLRP for vision transformers — Rule / Canonizer / Hook / Composite stack.

The module exposes the building blocks of AttnLRP (Achtibat et al., ICML 2024,
arXiv:2402.05602) as small, single-responsibility classes that compose:

* **Rules** — zennit ``Hook`` subclasses (PascalCase, no suffix, like
  zennit's own ``Epsilon`` / ``Gamma`` / ``Pass``): :class:`AlphaBetaMatmul`
  (bilinear ``q@kᵀ`` / ``weights@v``), :class:`ResidualRatio` (Otsuki
  residual split), :class:`Uniform` (Eq. 7 uniform allocation — LayerScale,
  ``x+pos_embed``), :class:`Identity` (ε-gated elementwise identity). Each is
  assigned to its target module type via a composite ``layer_map``; softmax /
  scale-by-constant identity reuse zennit's stock ``Pass``. ``nn.Linear`` /
  ``nn.Conv2d`` use zennit's ``Epsilon`` / ``Gamma``.

* **Canonizers** (forward-graph rewrites only — the idiomatic job of a
  zennit ``Canonizer``): :class:`LayerNormForwardCanonizer` (stop-gradient on
  the std), :class:`DropoutPassthroughCanonizer`,
  :class:`TimmBlockResidualCanonizer` / :class:`EvaBlockResidualCanonizer`
  (route residual adds + LayerScale muls through ``ResidualAdd`` /
  ``UniformAdd`` / ``LayerScaleMul`` modules so the rules above attach),
  :class:`VitPosEmbedPALRPCanonizer` (PA-LRP ``x+pos_embed``). Attention
  itself is substituted to its unfolded form by the canonizers in
  :mod:`zennit_ext.attention_unfolded`. :class:`TimmViTCanonizer` bundles the
  forward rewrites; each reverts on ``composite.context()`` exit.

* **Composites** (in :mod:`zennit_ext.attnlrp_composites`):
  :class:`AttnLRPEpsilonComposite`, :class:`AttnLRPGammaComposite`, and
  :class:`AttnLRPCombinedComposite` (the canonical recipe — attention
  substitution + the rule ``layer_map``).

The three ViT concept classes in :mod:`crp.concepts`
(``HeadConcept`` / ``EmbeddingDimConcept`` / ``TokenConcept``) read at the
``q_lrp_probe`` / ``k_lrp_probe`` / ``v_lrp_probe`` / ``proj_drop`` sites
installed by the attention-substitution canonizers in
:mod:`zennit_ext.attention_unfolded`.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.autograd import Function

from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.eva import EvaBlock
from timm.models.vision_transformer import Block as TimmBlock, VisionTransformer

from zennit.canonizers import AttributeCanonizer, Canonizer, CompositeCanonizer
from zennit.composites import LayerMapComposite
from zennit.core import BasicHook, Hook, stabilize
from zennit.rules import Epsilon, Gamma, Pass


# ─── 1. Stabilizers ──────────────────────────────────────────────────────────


# ─── 2. Rule autograd Function kernels ───────────────────────────────────────


def stop_gradient(input):
    """Detach ``input`` from the autograd graph (CP-LRP variant on normalisations)."""
    return input.detach()


# ─── 2b. LRP rules as zennit Hook subclasses ─────────────────────────────────
#
# Idiomatic zennit: a rule is a ``Hook``/``BasicHook`` subclass (PascalCase,
# no suffix, cf. ``Epsilon`` / ``Gamma`` / ``Pass``) assigned to a module via a
# composite's ``layer_map``. ``forward`` stashes the tensors the backward needs;
# ``backward(module, grad_input, grad_output)`` returns the modified grad_input
# tuple (the propagated relevance). Softmax / scale-by-constant identity is just
# zennit's stock :class:`~zennit.rules.Pass`.


class AlphaBetaMatmul(Hook):
    """AlphaBeta-on-bilinear LRP rule for a 2-input matmul ``y = a @ b``
    (AttnLRP bilinear; Bach 2015 generalised — see ``RESEARCH_NOTES.md``
    Entry 6). Attach to :class:`~zennit_ext.attention_unfolded.BilinearMatmul`
    via a composite ``layer_map``.

    ``α + β = 1`` ⇒ exact conservation (modulo ε).
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, epsilon: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["a"] = args[0]
        self.stored_tensors["b"] = args[1]

    def backward(self, module, grad_input, grad_output):
        a, b = self.stored_tensors["a"], self.stored_tensors["b"]
        rel = grad_output[0]
        eps, al, be = self.epsilon, self.alpha, self.beta
        a_pos, a_neg = a.clamp(min=0), a.clamp(max=0)
        b_pos, b_neg = b.clamp(min=0), b.clamp(max=0)
        y_pos = a_pos @ b_pos + a_neg @ b_neg
        y_neg = a_pos @ b_neg + a_neg @ b_pos
        f = rel / (y_pos + eps)
        g = rel / (y_neg - eps)
        bpT, bnT = b_pos.transpose(-1, -2), b_neg.transpose(-1, -2)
        grad_a = 0.5 * (
            a_pos * (al * (f @ bpT) + be * (g @ bnT))
            + a_neg * (al * (f @ bnT) + be * (g @ bpT))
        )
        apT, anT = a_pos.transpose(-1, -2), a_neg.transpose(-1, -2)
        grad_b = 0.5 * (
            b_pos * (al * (apT @ f) + be * (anT @ g))
            + b_neg * (al * (anT @ f) + be * (apT @ g))
        )
        return (grad_a, grad_b)

    def copy(self):
        return AlphaBetaMatmul(self.alpha, self.beta, self.epsilon)


class ResidualRatio(Hook):
    """Otsuki ratio-split residual rule for a 2-input add ``y = x + branch``:
    distribute ``R_y`` ∝ ``|x|`` vs ``|branch|``. Attach to
    :class:`~zennit_ext.attention_unfolded.ResidualAdd`.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["x"] = args[0]
        self.stored_tensors["branch"] = args[1]

    def backward(self, module, grad_input, grad_output):
        x, branch = self.stored_tensors["x"], self.stored_tensors["branch"]
        rel = grad_output[0]
        denom = x.abs() + branch.abs() + self.epsilon
        return (rel * x.abs() / denom, rel * branch.abs() / denom)

    def copy(self):
        return ResidualRatio(self.epsilon)


class Uniform(Hook):
    """Uniform allocation rule (AttnLRP Eq. 7): divide the incoming relevance
    equally, ``grad_input / factor``. ``factor=2`` is the per-bilinear default
    (e.g. the LayerScale γ multiply or the ``x + pos_embed`` add).
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def backward(self, module, grad_input, grad_output):
        return tuple(
            (g / self.factor if g is not None else None) for g in grad_input
        )

    def copy(self):
        return Uniform(self.factor)


class Identity(Hook):
    """AttnLRP identity rule for elementwise activations (Eq. 9): relevance
    passes through, ε-gated. Attach to ``nn.GELU``. (Pure pass-through is
    zennit's stock ``Pass``.)

    ``formula``:

    * ``'relevance'`` (default) — ``grad_in = grad_out · y/(y+ε)``. Correct in
      zennit's stock relevance-space convention.
    * ``'grad_times_input'`` — ``grad_in = grad_out · y/(x+ε)`` (the per-element
      ``f(x)/x``). Correct under ``lxt.efficient.monkey_patch_zennit`` (zennit's
      ``BasicHook`` reformulated in gradient×input space); matches LXT's
      ``identity_rule_implicit_fn`` for bit-parity with the LXT vit recipe.
    """

    def __init__(self, epsilon: float = 1e-6, formula: str = "relevance"):
        super().__init__()
        self.epsilon = epsilon
        self.formula = formula

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["input"] = args[0]
        self.stored_tensors["output"] = output

    def backward(self, module, grad_input, grad_output):
        y = self.stored_tensors["output"]
        if self.formula == "grad_times_input":
            x = self.stored_tensors["input"]
            return (grad_output[0] * (y / (x + self.epsilon)),)
        return (grad_output[0] * (y / (y + self.epsilon)),)

    def copy(self):
        return Identity(self.epsilon, self.formula)


# ─── 3. Forward-method replacements (installed per-instance by Canonizers) ──


def layer_norm_forward(self, x):
    """LayerNorm with stop-gradient on the std. Identity rule on the
    normalised output (AttnLRP §3.2.2)."""
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.eps).sqrt()
    y = (x - mean) / stop_gradient(std)
    if self.weight is not None:
        y = y * self.weight
    if self.bias is not None:
        y = y + self.bias
    return y


def dropout_passthrough_forward(self, x):
    """Disable dropout during attribution (model may be in train mode)."""
    return x


def _make_mha_cplrp_forward(original_forward):
    """Build an ``nn.MultiheadAttention.forward`` replacement that
    implements CP-LRP by detaching ``query`` and ``key`` before
    delegating to ``original_forward`` (which is the BOUND method of the
    target instance, captured at canonizer-apply time).

    Mirror of :func:`lxt.efficient.patches.cp_multi_head_attention_forward`:
    once Q and K carry no gradient, the softmax weights downstream are a
    graph constant, and ``out = softmax(QKᵀ/√d) @ V`` routes
    ``R_V = weightsᵀ @ R_out`` via standard autograd → CP-LRP value path.

    `torchvision.models.vit_b_16` calls MHA with ``query is key is value``
    (self-attention). After ``query.detach()`` and ``key.detach()``, the
    three are distinct tensor objects, so MHA's ``_in_projection_packed``
    fast-path is skipped and the chunked ``_in_projection`` path is used:
    the Q and K row-blocks of ``in_proj_weight`` get no gradient, only
    the V block does. The Q/K linear projections are LRP-dead exactly as
    LXT's recipe intends.
    """
    def patched(self, query, key, value, *args, **kwargs):
        return original_forward(query.detach(), key.detach(), value, *args, **kwargs)
    return patched


# NOTE: ``_eva_attention_forward`` was removed in the unfolding refactor.
# Eva attention is now handled by :class:`zennit_ext.attention_unfolded.EvaAttentionUnfolded`
# + :class:`zennit_ext.attention_unfolded.EvaAttentionSubstitutionCanonizer`. The
# substitution path replaces the entire ``EvaAttention`` module with a
# subgraph of named ``nn.Module`` kernels (``BilinearMatmul``,
# ``SoftmaxAlongLastDim``, ``RotaryEmbedding``, etc.), each owning one
# LRP rule. See ``UNFOLDING_ATTENTION_REFACTOR.md`` and
# ``RESEARCH_NOTES.md`` Entries 4-6.


# NOTE: ``_timm_attention_forward`` was removed in the always-unfold cleanup.
# Standard timm ``Attention`` is now substituted with
# :class:`zennit_ext.attention_unfolded.TimmAttentionUnfolded` (analogous to the
# Eva path). Both attention paths are unified under the unfolded form,
# so concept attribution works on every supported backbone.


def _eva_block_forward(
    self, x, rope=None, attn_mask=None, is_causal=False,
    *, residual_rule: str = "ratio", layerscale_uniform: bool = False,
):
    """``EvaBlock.forward`` replacement applying one of the residual-LRP
    rules and (optionally) a uniform-rule allocation on the LayerScale γ
    multiplications.

    Parameters bound by the canonizer:

    * ``residual_rule`` — ``'symmetric'`` (factor-2 uniform allocation)
      or ``'ratio'`` (Otsuki ``|x|``/``|branch|`` proportional split).
    * ``layerscale_uniform`` — see :func:`_apply_layerscale`.
    """
    # Residual adds + LayerScale muls route through ``nn.Module`` instances
    # (``self._lrp_res{1,2}`` = ResidualAdd/UniformAdd; ``self._lrp_ls{1,2}``
    # = LayerScaleMul) attached by :class:`EvaBlockResidualCanonizer`, so the
    # composite ``layer_map`` assigns each its LRP rule as a zennit Hook.
    attn_branch = self.attn(
        self.norm1(x), rope=rope, attn_mask=attn_mask, is_causal=is_causal,
    )
    if self.gamma_1 is not None:
        attn_branch = (
            self._lrp_ls1(attn_branch) if hasattr(self, "_lrp_ls1")
            else self.gamma_1 * attn_branch
        )
    branch1 = self.drop_path1(attn_branch)
    x = self._lrp_res1(x, branch1)

    mlp_branch = self.mlp(self.norm2(x))
    if self.gamma_2 is not None:
        mlp_branch = (
            self._lrp_ls2(mlp_branch) if hasattr(self, "_lrp_ls2")
            else self.gamma_2 * mlp_branch
        )
    branch2 = self.drop_path2(mlp_branch)
    x = self._lrp_res2(x, branch2)
    return x


def _timm_block_forward(
    self, x, attn_mask=None, is_causal=False,
    *, residual_rule: str = "ratio",
):
    """timm ``Block.forward`` replacement with one of the residual-LRP rules.
    LayerScale on standard timm Blocks is an ``nn.Module`` (``ls1``/``ls2``),
    not a parameter — handled transparently by its own forward; no
    layerscale-uniform branch needed here.
    """
    branch1 = self.drop_path1(
        self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal))
    )
    x = self._lrp_res1(x, branch1)
    branch2 = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    x = self._lrp_res2(x, branch2)
    return x


def vit_pos_embed_palrp(self, x):
    """timm ``VisionTransformer._pos_embed`` replacement implementing PA-LRP.

    PA-LRP (Bakish et al., NeurIPS 2025; arXiv 2506.02138) treats the
    additive ``x = x + pos_embed`` step as a bilinear-style operation under
    the LRP uniform rule (Eq. 7 of AttnLRP) and allocates relevance equally
    between the two operands. ``pos_embed`` is a learned parameter with no
    upstream input, so its half is "absorbed"; the remaining half flows
    back to ``x``. Without this, the additive step is transparent to
    backward and ``x`` receives the full upstream relevance, double-counting
    it against ``pos_embed``.

    Implementation: identical to upstream ``_pos_embed`` (timm 1.0.x —
    handles ``cls_token``/``reg_token``, ``no_embed_class`` deit-3 variant,
    ``dynamic_img_size``) except the additive step is routed through a
    :class:`~zennit_ext.attention_unfolded.UniformAdd` module, which the
    composite ``layer_map`` gives the :class:`Uniform` (factor-2) rule.
    """
    to_cat = []
    if self.cls_token is not None:
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
    if self.reg_token is not None:
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

    if self.pos_embed is None:
        return torch.cat(to_cat + [x.view(x.shape[0], -1, x.shape[-1])], dim=1)

    if self.dynamic_img_size:
        B, H, W, C = x.shape
        prev_grid_size = self.patch_embed.grid_size
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(H, W),
            old_size=prev_grid_size,
            num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
        )
        x = x.view(B, -1, C)
    else:
        pos_embed = self.pos_embed

    if self.no_embed_class:
        x = self._lrp_posadd(x, pos_embed)
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = self._lrp_posadd(x, pos_embed)

    return self.pos_drop(x)


# ─── 4. Canonizers — one class per kind of model graph mutation ─────────────


# NOTE: ``AttentionTapsCanonizer`` was removed in the unfolding refactor.
# The qkv_tap / attn_out_tap Identity submodules it injected are no
# longer needed — concepts target the named submodules of
# :class:`zennit_ext.attention_unfolded.EvaAttentionUnfolded` directly. Concept
# work on standard timm ViTs (without unfolding) is unsupported; only
# attribution still works there.


def _bind_forward(module: nn.Module, fn: Callable, attr: str = "forward") -> dict:
    """Bind ``fn`` as ``attr`` on ``module``'s class — return dict for
    AttributeCanonizer."""
    return {attr: fn.__get__(module, type(module))}


class LayerNormForwardCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``nn.LayerNorm.forward`` for the AttnLRP
    stop-gradient-on-std variant (:func:`layer_norm_forward`).

    AttnLRP §3.2.2 — treats LayerNorm's normalisation as element-wise so
    relevance flows through the affine output unchanged.
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, nn.LayerNorm):
            return None
        return _bind_forward(module, layer_norm_forward)

    def copy(self):
        return type(self)()


class DropoutPassthroughCanonizer(AttributeCanonizer):
    """Canonizer that disables ``nn.Dropout`` during attribution (model may
    be in train mode)."""

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, nn.Dropout):
            return None
        return _bind_forward(module, dropout_passthrough_forward)

    def copy(self):
        return type(self)()


class TimmBlockResidualCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``forward`` on timm ``vision_transformer.Block``
    to apply a conservative LRP rule at each residual addition.

    Parameters
    ----------
    residual_rule : {'symmetric', 'ratio'}
        ``'symmetric'`` halves gradient at every additive node (uniform
        allocation, matches ResNet symmetric rule). ``'ratio'``
        distributes upstream relevance proportionally to ``|x|`` and
        ``|branch|`` (Otsuki-style ratio split, the closer LRP-ε
        analogue and reported as superior for ResNets).
    """

    def __init__(self, *, residual_rule: str = "ratio"):
        if residual_rule not in ("symmetric", "ratio"):
            raise ValueError(
                f"residual_rule must be 'symmetric' or 'ratio'; got {residual_rule!r}"
            )
        self.residual_rule = residual_rule
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, TimmBlock):
            return None
        needed = ("ls1", "ls2", "drop_path1", "drop_path2", "norm1",
                  "norm2", "attn", "mlp")
        if not all(hasattr(module, a) for a in needed):
            return None
        from zennit_ext.attention_unfolded import ResidualAdd, UniformAdd
        ResMod = ResidualAdd if self.residual_rule == "ratio" else UniformAdd

        def fwd(self, x, attn_mask=None, is_causal=False):
            return _timm_block_forward(
                self, x, attn_mask=attn_mask, is_causal=is_causal,
            )

        attrs = _bind_forward(module, fwd)
        attrs["_lrp_res1"] = ResMod()
        attrs["_lrp_res2"] = ResMod()
        return attrs

    def copy(self):
        return type(self)(residual_rule=self.residual_rule)


class EvaBlockResidualCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``forward`` on timm ``eva.EvaBlock`` (DINOv3 etc.)
    for the residual-LRP variant. Handles the optional
    ``gamma_1``/``gamma_2`` LayerScale parameters and (optionally) wraps
    them under the AttnLRP uniform rule.

    Parameters
    ----------
    residual_rule : {'symmetric', 'ratio'}
        Same as :class:`TimmBlockResidualCanonizer`.
    layerscale_uniform : bool
        When True, route the LayerScale γ multiplications through
        :class:`~zennit_ext.attention_unfolded.LayerScaleMul` modules, which
        the composite ``layer_map`` gives the :class:`Uniform` (factor-2)
        rule (AttnLRP Eq. 7, treating γ as a constant operand). Default False.
    """

    def __init__(
        self, *, residual_rule: str = "ratio", layerscale_uniform: bool = False,
    ):
        if residual_rule not in ("symmetric", "ratio"):
            raise ValueError(
                f"residual_rule must be 'symmetric' or 'ratio'; got {residual_rule!r}"
            )
        self.residual_rule = residual_rule
        self.layerscale_uniform = layerscale_uniform
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, EvaBlock):
            return None
        needed = ("drop_path1", "drop_path2", "norm1", "norm2", "attn", "mlp",
                  "gamma_1", "gamma_2")
        if not all(hasattr(module, a) for a in needed):
            return None
        from zennit_ext.attention_unfolded import (
            ResidualAdd, UniformAdd, LayerScaleMul,
        )
        ResMod = ResidualAdd if self.residual_rule == "ratio" else UniformAdd

        def fwd(self, x, rope=None, attn_mask=None, is_causal=False):
            return _eva_block_forward(
                self, x, rope=rope, attn_mask=attn_mask, is_causal=is_causal,
            )

        attrs = _bind_forward(module, fwd)
        attrs["_lrp_res1"] = ResMod()
        attrs["_lrp_res2"] = ResMod()
        if self.layerscale_uniform:
            # LayerScale γ·branch → uniform rule (γ a param, absorbs half).
            if module.gamma_1 is not None:
                attrs["_lrp_ls1"] = LayerScaleMul(module.gamma_1)
            if module.gamma_2 is not None:
                attrs["_lrp_ls2"] = LayerScaleMul(module.gamma_2)
        return attrs

    def copy(self):
        return type(self)(
            residual_rule=self.residual_rule,
            layerscale_uniform=self.layerscale_uniform,
        )


class TorchvisionMHACPLRPCanonizer(AttributeCanonizer):
    """Canonizer that installs CP-LRP on ``torch.nn.MultiheadAttention``.

    Drop-in zennit replacement for LXT's instance-level patch in
    :func:`lxt.efficient.patches.cp_multi_head_attention_forward`. Each
    matched MHA instance gets its forward rebound to a wrapper that
    calls ``query.detach()`` and ``key.detach()`` before delegating to
    the original forward — so Q,K paths carry no gradient and only the
    value path receives relevance.

    Targets ``nn.MultiheadAttention`` (any instance). The intended use
    is on `torchvision.models.vision_transformer` ViTs (vit_b_16,
    vit_l_16, etc.); other models using ``nn.MultiheadAttention`` are
    canonized identically.

    Combine with :class:`LayerNormForwardCanonizer` and
    :class:`DropoutPassthroughCanonizer` (and ``nn.GELU`` → ``Pass`` in the
    composite ``layer_map``) to obtain the zennit-side equivalent of
    LXT-efficient's published vision-transformer recipe
    (``lxt.efficient.models.vit_torch.cp_LRP``).
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, nn.MultiheadAttention):
            return None
        # Capture the BOUND original forward so the closure has self pre-baked.
        original_forward = module.forward
        patched = _make_mha_cplrp_forward(original_forward)
        return _bind_forward(module, patched)

    def copy(self):
        return type(self)()


class VitPosEmbedPALRPCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``_pos_embed`` on timm ``VisionTransformer``
    instances to apply the PA-LRP uniform rule (Bakish et al. 2025;
    arXiv:2506.02138). See :func:`vit_pos_embed_palrp`."""

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, VisionTransformer) or not hasattr(module, "_pos_embed"):
            return None
        from zennit_ext.attention_unfolded import UniformAdd
        attrs = _bind_forward(module, vit_pos_embed_palrp, attr="_pos_embed")
        attrs["_lrp_posadd"] = UniformAdd()  # x + pos_embed → uniform rule via layer_map
        return attrs

    def copy(self):
        return type(self)()


class TimmViTCanonizer(CompositeCanonizer):
    """Aggregator: bundles the per-module canonizers needed for AttnLRP on
    a timm ViT (standard or Eva-stack).

    Combines:

    * :class:`LayerNormForwardCanonizer` — LayerNorm with stop-gradient(std).
    * :class:`DropoutPassthroughCanonizer` — disable Dropout for backward.
    * :class:`VitPosEmbedPALRPCanonizer` (when ``palrp=True``) — PA-LRP on
      the ``x + pos_embed`` step (Bakish et al. 2025).
    * :class:`TimmBlockResidualCanonizer` and
      :class:`EvaBlockResidualCanonizer` (when ``residual_lrp`` is set) —
      ratio or symmetric residual rule.

    Attention itself is NOT handled here. It is substituted to its
    unfolded form (:class:`TimmAttentionUnfolded` /
    :class:`EvaAttentionUnfolded`) by their respective substitution
    canonizers, applied automatically by
    :class:`AttnLRPCombinedComposite`.

    All mutations are instance-level and reversible (revert on
    ``composite.context()`` exit).

    Parameters
    ----------
    palrp : bool
        Enable PA-LRP on the absolute pos_embed addition. Only relevant
        for ViTs with ``self.pos_embed`` (vit_base etc.); no-op for
        DINOv3 (RoPE only).
    residual_lrp : {None, 'symmetric', 'ratio'}
        Block-level residual rule. ``'ratio'`` is the recommended
        default for transformers; ``'symmetric'`` matches the ResNet
        AttnLRP paper baseline.
    layerscale_uniform : bool
        Apply the uniform allocation rule to LayerScale γ
        multiplications (CaiT / Eva blocks only).
    epsilon : float
        ε for ε-stabilised rules.
    """

    def __init__(
        self,
        *,
        palrp: bool = False,
        residual_lrp: Optional[str] = None,
        layerscale_uniform: bool = False,
        epsilon: float = 1e-6,
    ):
        canonizers: List[Canonizer] = [
            LayerNormForwardCanonizer(),
            DropoutPassthroughCanonizer(),
        ]
        if palrp:
            canonizers.append(VitPosEmbedPALRPCanonizer())
        if residual_lrp is not None:
            canonizers.append(TimmBlockResidualCanonizer(residual_rule=residual_lrp))
            canonizers.append(EvaBlockResidualCanonizer(
                residual_rule=residual_lrp,
                layerscale_uniform=layerscale_uniform,
            ))
        elif layerscale_uniform:
            # User asked for layerscale_uniform but didn't pick a residual
            # rule; default to ratio so the EvaBlock forward gets installed
            # (the layerscale_uniform wrapper lives inside that forward).
            canonizers.append(EvaBlockResidualCanonizer(
                residual_rule="ratio", layerscale_uniform=True,
            ))
        super().__init__(canonizers)


# ─── 5. Hooks — LRP backward for Linear / Conv2d ─────────────────────────────
#
# We use zennit's stock :class:`zennit.rules.Epsilon` and
# :class:`zennit.rules.Gamma` directly. The previous version shipped a
# ``GradientTimesInputBasicHook`` subclass under names ``GTIEpsilon`` /
# ``GTIGamma`` that violated conservation by 100-200% on ordinary inputs
# (audited in ``experiments/audit_gti_hook.py``); those names were
# removed in the unfolding-refactor cleanup. Use ``zennit.rules.Epsilon``
# and ``zennit.rules.Gamma`` directly.


__all__ = [
    # LRP rules as zennit Hook subclasses
    "AlphaBetaMatmul", "ResidualRatio", "Uniform", "Identity",
    "stop_gradient",
    "layer_norm_forward", "dropout_passthrough_forward", "vit_pos_embed_palrp",
    "LayerNormForwardCanonizer", "DropoutPassthroughCanonizer",
    "TimmBlockResidualCanonizer", "EvaBlockResidualCanonizer", "TorchvisionMHACPLRPCanonizer",
    "VitPosEmbedPALRPCanonizer", "TimmViTCanonizer",
]
