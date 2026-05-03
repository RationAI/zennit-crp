"""AttnLRP for vision transformers — Rule / Canonizer / Hook / Composite stack.

The module exposes the building blocks of AttnLRP (Achtibat et al., ICML 2024,
arXiv:2402.05602) as small, single-responsibility classes that compose:

* **Rule kernels** — `_IdentityRuleFn`, `_DivideGradientFn`,
  `_ResidualRatioFn`, `_MatmulFactor2Fn`. Each is one autograd
  ``Function`` implementing one LRP backward semantic. Inlined into the
  forward pass via canonizer-installed forward methods.

* **Canonizers** — one class per *kind of model graph mutation*:
  :class:`AttentionTapsCanonizer`, :class:`LayerNormForwardCanonizer`,
  :class:`GELUIdentityRuleCanonizer`, :class:`DropoutPassthroughCanonizer`,
  :class:`TimmAttentionForwardCanonizer`,
  :class:`EvaAttentionForwardCanonizer`,
  :class:`TimmBlockResidualCanonizer`, :class:`EvaBlockResidualCanonizer`,
  :class:`VitPosEmbedPALRPCanonizer`. Each touches one module type and
  reverts on ``composite.context()`` exit.
  The aggregator :class:`TimmViTCanonizer` bundles them into one
  config-driven object.

* **Hooks** — :class:`GradientTimesInputBasicHook` and the two ``BasicHook``
  subclasses :class:`GTIEpsilon`, :class:`GTIGamma` running the LRP backward
  for ``nn.Linear``/``nn.Conv2d`` in the gradient×input form
  ``R = grad·output / input``.

* **Composites** — one *named* composite per remedy / recipe, so calls
  read like ``AttnLRPMatmulFactor2Composite()`` rather than a magic
  toggle dict. Available:

  * :class:`AttnLRPEpsilonComposite` — baseline ε-LRP recipe.
  * :class:`AttnLRPGammaComposite` — γ-LRP on Linears (Achtibat §3.2.1).
  * :class:`AttnLRPMatmulFactor2Composite` — Achtibat Prop. 3.3 ``2y+ε``
    stabiliser on bilinear matmuls.
  * :class:`AttnLRPSignedEpsilonComposite` — sign-aware ε in the identity
    rule (Achtibat Eq. 16).
  * :class:`AttnLRPRopeDetachComposite` — RoPE cos/sin treated as graph
    constants (RoFormer, Su et al. 2021, arXiv:2104.09864).
  * :class:`AttnLRPLayerScaleUniformComposite` — uniform allocation rule
    on the LayerScale γ multiplication (CaiT, Touvron et al. 2021,
    arXiv:2103.17239).
  * :class:`AttnLRPLinearGammaComposite` — γ=0.05 on Linears (a
    conservative variant of :class:`AttnLRPGammaComposite`).

  Combinations are deferred to :class:`AttnLRPCombinedComposite` whose
  *one* responsibility is "combine remedies the user has already
  individually validated".

The four ViT concept classes in :mod:`crp.attention_concepts` consume the
named taps installed by :class:`AttentionTapsCanonizer`.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.autograd import Function

from zennit.canonizers import AttributeCanonizer, Canonizer, CompositeCanonizer
from zennit.composites import LayerMapComposite
from zennit.core import BasicHook, ParamMod, Stabilizer, stabilize
from zennit.rules import Epsilon, Gamma, GammaMod, NoMod, Pass


# ─── 1. Stabilizers ──────────────────────────────────────────────────────────


def _signed_eps_add(input: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Add ``ε·sign(input)`` in-place of plain ``+ ε``.

    AttnLRP Eq. 16 (Achtibat et al. 2024, arXiv:2402.05602): the
    sign-aware stabiliser preserves the rule's behaviour around zero
    crossings, where ``input + ε`` would shift small negatives toward
    zero (cancellation) and small positives further from zero
    (asymmetry). The convention ``sign(0) = +1`` matches zennit's
    :func:`zennit.core.stabilize`.
    """
    sign = (input == 0).to(input.dtype) + input.sign()
    return input + sign * epsilon


# ─── 2. Rule autograd Function kernels ───────────────────────────────────────


class _IdentityRuleFn(Function):
    """LRP-0 identity rule for element-wise activations (Bach et al. 2015;
    AttnLRP §3.2.2, Eq. 9): backward returns ``(output/stab(output)) · R_y``.

    For elementwise ``y = f(x)`` the LRP-0 derivation treats the activation
    as a 1×1 "linear" with weight ``f(x)/x`` per element, giving::

        R_x_i = x_i · (f(x_i)/x_i) / y_i · R_y_i = R_y_i   (when y_i ≠ 0)

    so the rule reduces to ``R_x = R_y`` whenever the element contributed
    to the output, and to 0 when it did not. The ε-stabilised form is::

        R_x = R_y · y / (y + ε·sign(y))   ≈ R_y for |y| ≫ ε, 0 for |y| ≪ ε.

    Implementation: save ``output / stab(output)`` on forward (≈ ±1
    everywhere except where output is near zero), multiply by incoming
    relevance on backward.

    *Earlier (buggy) version used ``output/stab(input)`` instead* — that
    over-dampened relevance whenever the input was near ε regardless of
    whether the activation was active, breaking conservation by up to
    100× per layer (see ``experiments/audit_identity_rule.py``).

    Parameters (via :func:`identity_rule_implicit`):

    * ``epsilon`` — stabiliser magnitude. Default ``1e-6``.
    * ``signed`` — when True, use ``y + ε·sign(y)`` instead of ``y + ε``
      (AttnLRP Eq. 16). zennit's :func:`zennit.core.stabilize` is
      sign-aware by default; this flag exists for parity with the
      paper's notation.
    """

    @staticmethod
    def forward(ctx, fn, input, epsilon=1e-6, signed=False):
        output = fn(input)
        if input.requires_grad:
            denom = _signed_eps_add(output, epsilon) if signed else output + epsilon
            ctx.save_for_backward(output / denom)
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        gradient = ctx.saved_tensors[0] * out_relevance[0]
        return None, gradient, None, None


class _DivideGradientFn(Function):
    """Uniform rule (AttnLRP §3.2, Eq. 7): backward divides incoming grad by
    ``factor``.

    Used after bilinear ops (matmul, ⊙) to allocate relevance equally among
    operands. ``factor=2`` per bilinear; in attention without
    :class:`_MatmulFactor2Fn`, Q×4, K×4 (each enters two bilinears via
    softmax(QKᵀ)) and V×2 (one bilinear via attn·V).
    """

    @staticmethod
    def forward(ctx, input, factor=2):
        ctx.factor = factor
        return input

    @staticmethod
    def backward(ctx, *out_relevance):
        return out_relevance[0] / ctx.factor, None


class _ResidualRatioFn(Function):
    """Ratio rule for residual addition ``y = x + branch``.

    Distributes the upstream relevance ``R_y`` proportionally to ``|x|`` and
    ``|branch|`` — Otsuki-style ratio split. Conservation:
    ``R_x + R_branch = R_y · (|x| + |branch|) / (|x| + |branch| + ε) ≈ R_y``.
    """

    @staticmethod
    def forward(ctx, x, branch, epsilon=1e-6):
        ctx.save_for_backward(x, branch)
        ctx.epsilon = epsilon
        return x + branch

    @staticmethod
    def backward(ctx, grad_output):
        x, branch = ctx.saved_tensors
        abs_x = x.abs()
        abs_b = branch.abs()
        denom = abs_x + abs_b + ctx.epsilon
        return grad_output * (abs_x / denom), grad_output * (abs_b / denom), None


class _MatmulFactor2Fn(Function):
    """AttnLRP bilinear matmul rule (Achtibat et al. 2024, Prop. 3.3,
    Eq. 14, arXiv:2402.05602).

    For ``Y = A @ B`` the rule returns relevance attributed to each
    operand in **pure R form** (no further stabilisation needed
    upstream)::

        scaled = R_Y / (2 · Y + ε[·sign(Y)])
        R_A    = A · (scaled @ B^T)
        R_B    = B · (A^T @ scaled)

    Conservation: ``sum(R_A) + sum(R_B) ≈ sum(R_Y)`` (the ``2·Y``
    denominator splits each upstream relevance evenly across the two
    bilinear chains).

    The ``A ·`` / ``B ·`` operand multiplication is **mandatory** for
    the chain to compose correctly: the upstream of these matmul
    operands is softmax (Pass) or a non-Linear chain that does not own
    an operand-multiplication step. Including it here makes the rule
    self-contained — the result is the bilinear's R contribution to
    each operand, ready to flow back through the rest of the LRP graph.

    Earlier (buggy) version returned only ``scaled @ B^T`` etc. without
    operand multiplication, which the diagnostic showed over-divides
    on the broken downstream chain (no Linear hook upstream of softmax
    to re-introduce the operand factor).

    When this rule is used, drop the operand-side
    ``_DivideGradientFn`` factor-2/4 calls — this rule's ``2·Y``
    denominator already enforces the conservation factor.
    """

    @staticmethod
    def forward(ctx, a, b, epsilon=1e-6, signed=False):
        out = a @ b
        ctx.save_for_backward(a, b, out)
        ctx.epsilon = epsilon
        ctx.signed = signed
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b, out = ctx.saved_tensors
        if ctx.signed:
            denom = _signed_eps_add(2 * out, ctx.epsilon)
        else:
            denom = 2 * out + ctx.epsilon
        scaled = grad_out / denom
        grad_a = a * (scaled @ b.transpose(-1, -2))
        grad_b = b * (a.transpose(-1, -2) @ scaled)
        return grad_a, grad_b, None, None


# Convenience callables (used inside the forward replacements below).


def identity_rule_implicit(fn, input, *, epsilon: float = 1e-6, signed: bool = False):
    """Apply ``fn(input)`` with the AttnLRP identity rule inlined into backward."""
    return _IdentityRuleFn.apply(fn, input, epsilon, signed)


def divide_gradient(input, factor: int = 2):
    """Identity in forward; divide incoming relevance by ``factor`` on backward."""
    return _DivideGradientFn.apply(input, factor)


def stop_gradient(input):
    """Detach ``input`` from the autograd graph (CP-LRP variant on normalisations)."""
    return input.detach()


def residual_ratio(x, branch, epsilon: float = 1e-6):
    """Apply ratio-split residual rule. Forward: ``x + branch``. Backward
    distributes upstream relevance ∝ ``|x|`` vs ``|branch|`` (Otsuki).
    """
    return _ResidualRatioFn.apply(x, branch, epsilon)


def matmul_factor_2(a, b, *, epsilon: float = 1e-6, signed: bool = False):
    """Apply :class:`_MatmulFactor2Fn` to ``a @ b`` (AttnLRP Prop. 3.3)."""
    return _MatmulFactor2Fn.apply(a, b, epsilon, signed)


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


def _eva_attention_forward(
    self, x, rope=None, attn_mask=None, is_causal=False,
    *, matmul_factor_2_rule: bool = False, rope_detach: bool = False,
    signed_epsilon: bool = False, epsilon: float = 1e-6,
):
    """timm ``EvaAttention.forward`` replacement for AttnLRP + concept hooking.

    Mirrors :func:`_timm_attention_forward` but for the ``Eva``-stack
    Attention used by DINOv3 ViTs (and EVA*, BEiT-style models). Differences
    vs. the standard timm Attention path:

    * Forward signature takes ``rope`` (rotary positional embedding tensor)
      passed in by the parent ``EvaBlock.forward``.
    * RoPE is applied to ``q`` and ``k`` after q_norm/k_norm and before the
      attention bilinear. We **insert** :func:`divide_gradient` factors on
      q/k/v *before* RoPE so the AttnLRP uniform rule still applies to the
      pre-rotation operands (RoPE is a per-token rotation: a unitary
      transform that doesn't change ``|q|``/``|k|`` magnitudes).
    * Bypasses the fused ``F.scaled_dot_product_attention`` path so the
      autograd graph contains the explicit softmax + matmul ops.
    * Routes through the named taps installed by
      :class:`AttentionTapsCanonizer`.

    Strategy parameters (bound by the canonizer at canonize time):

    * ``matmul_factor_2_rule`` — when True, replace ``q @ kᵀ`` and
      ``attn @ v`` with :func:`matmul_factor_2` (AttnLRP Prop. 3.3,
      ``2y+ε`` stabiliser) and drop the redundant
      ``divide_gradient(q,4)/(k,4)/(v,2)`` calls.
    * ``rope_detach`` — when True, ``rope.detach()`` before the rotary
      embedding op. RoPE has no learnable parameters (Su et al. 2021,
      arXiv:2104.09864) so cos/sin can be treated as graph constants;
      relevance routes purely through q/k.
    * ``signed_epsilon`` — when True with ``matmul_factor_2_rule``, use
      sign-aware ε in the bilinear stabiliser (AttnLRP Eq. 16).

    The DINOv3 ViT-L variant has ``q_bias=k_bias=v_bias=None`` and
    ``qkv.bias=False``; we only support that simple path here. Variants
    with separate Q/K/V biases (``vit_*_dinov3_qkvb``) would need an
    additional branch (lift verbatim from the upstream forward).
    """
    from timm.layers import apply_rot_embed_cat

    B, N, C = x.shape
    qkv_flat = self.qkv(x)              # (B, N, 3*num_heads*head_dim)
    qkv_flat = self.qkv_tap(qkv_flat)   # ← K/Q/V-side hook tap
    qkv = qkv_flat.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    # When matmul_factor_2_rule is on, the ``2·Y + ε`` denominator inside
    # :class:`_MatmulFactor2Fn` already gives the bilinear conservation —
    # the operand-side divide_gradient calls are redundant and would
    # over-shrink relevance. When off, fall back to the operand-side
    # uniform-rule allocation (4×4×2 over the two bilinears).
    if not matmul_factor_2_rule:
        q = divide_gradient(q, 4)
        k = divide_gradient(k, 4)
        v = divide_gradient(v, 2)

    # RoPE — first ``num_prefix_tokens`` (cls + reg) are NOT rotated.
    if rope is not None:
        npt = self.num_prefix_tokens
        half = getattr(self, "rotate_half", False)
        rope_used = rope.detach() if rope_detach else rope
        q = torch.cat(
            [q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope_used, half=half)],
            dim=2,
        ).type_as(v)
        k = torch.cat(
            [k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope_used, half=half)],
            dim=2,
        ).type_as(v)

    # Manual softmax + matmul (bypass fused_attn so the autograd graph
    # exposes the bilinear ops to the LRP rules).
    q = q * self.scale
    if matmul_factor_2_rule:
        attn = matmul_factor_2(q, k.transpose(-2, -1), epsilon=epsilon, signed=signed_epsilon)
    else:
        attn = q @ k.transpose(-2, -1)

    try:
        from timm.models.eva import resolve_self_attn_mask, maybe_add_mask
        attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
        attn = maybe_add_mask(attn, attn_bias)
    except (ImportError, AttributeError):
        if attn_mask is not None:
            attn = attn + attn_mask
        if is_causal:
            mask = torch.triu(
                torch.full(
                    (N, N), float("-inf"), device=attn.device, dtype=attn.dtype
                ),
                diagonal=1,
            )
            attn = attn + mask

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    if matmul_factor_2_rule:
        x = matmul_factor_2(attn, v, epsilon=epsilon, signed=signed_epsilon)
    else:
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    if hasattr(self, "norm"):
        x = self.norm(x)
    x = self.attn_out_tap(x)            # ← output-side hook tap
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _timm_attention_forward(
    self, x, attn_mask=None, is_causal=False,
    *, matmul_factor_2_rule: bool = False, signed_epsilon: bool = False,
    epsilon: float = 1e-6,
):
    """timm ``Attention.forward`` replacement for AttnLRP + concept hooking.

    Tracks the upstream timm signature ``(self, x, attn_mask=None,
    is_causal=False)`` so it remains a drop-in across timm ≥ 1.0.

    Differences from upstream timm forward:

    * Routes ``self.qkv(x)`` through ``self.qkv_tap`` and the per-head
      attention output through ``self.attn_out_tap`` — both ``nn.Identity``
      submodules installed by :class:`AttentionTapsCanonizer`.
    * Q, K, V each pass through :func:`divide_gradient` (factors 4, 4, 2) —
      AttnLRP uniform rule on the ``QKᵀ`` and ``attn·V`` bilinears
      (Eq. 14–15 of arXiv:2402.05602). Skipped when
      ``matmul_factor_2_rule`` is enabled — see :func:`_eva_attention_forward`.
    * Bypasses the fused ``F.scaled_dot_product_attention`` path so the
      autograd graph contains the explicit softmax + matmul ops where the
      rules apply.
    """
    B, N, _ = x.shape
    qkv_flat = self.qkv(x)
    qkv_flat = self.qkv_tap(qkv_flat)
    qkv = qkv_flat.reshape(B, N, 3, self.num_heads, self.head_dim).permute(
        2, 0, 3, 1, 4
    )
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if not matmul_factor_2_rule:
        q = divide_gradient(q, 4)
        k = divide_gradient(k, 4)
        v = divide_gradient(v, 2)

    q = q * self.scale
    if matmul_factor_2_rule:
        attn = matmul_factor_2(q, k.transpose(-2, -1), epsilon=epsilon, signed=signed_epsilon)
    else:
        attn = q @ k.transpose(-2, -1)

    try:
        from timm.models.vision_transformer import (
            resolve_self_attn_mask, maybe_add_mask,
        )
        attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
        attn = maybe_add_mask(attn, attn_bias)
    except ImportError:
        if attn_mask is not None:
            attn = attn + attn_mask
        if is_causal:
            mask = torch.triu(
                torch.full(
                    (N, N), float("-inf"), device=attn.device, dtype=attn.dtype
                ),
                diagonal=1,
            )
            attn = attn + mask

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    if matmul_factor_2_rule:
        x = matmul_factor_2(attn, v, epsilon=epsilon, signed=signed_epsilon)
    else:
        x = attn @ v

    out_dim = getattr(self, "attn_dim", self.num_heads * self.head_dim)
    x = x.transpose(1, 2).reshape(B, N, out_dim)
    if hasattr(self, "norm"):
        x = self.norm(x)
    x = self.attn_out_tap(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _apply_layerscale(scale, branch, *, layerscale_uniform: bool):
    """Apply ``scale * branch`` (LayerScale) with optional uniform-rule
    allocation.

    When ``layerscale_uniform`` is True, the multiplication is wrapped in
    :func:`divide_gradient(., 2)` per AttnLRP Eq. 7 (uniform rule on
    bilinears). LayerScale γ is a learned scalar with no upstream input
    (Touvron et al. 2021 CaiT, arXiv:2103.17239), so its allocated half
    is "absorbed" — relevance flowing back into ``branch`` is multiplied
    by ``γ/2`` instead of ``γ``.
    """
    out = scale * branch
    if layerscale_uniform:
        out = divide_gradient(out, 2)
    return out


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
    attn_branch = self.attn(
        self.norm1(x), rope=rope, attn_mask=attn_mask, is_causal=is_causal,
    )
    if self.gamma_1 is not None:
        attn_branch = _apply_layerscale(
            self.gamma_1, attn_branch, layerscale_uniform=layerscale_uniform,
        )
    branch1 = self.drop_path1(attn_branch)
    if residual_rule == "ratio":
        x = residual_ratio(x, branch1)
    else:
        x = divide_gradient(x + branch1, 2)

    mlp_branch = self.mlp(self.norm2(x))
    if self.gamma_2 is not None:
        mlp_branch = _apply_layerscale(
            self.gamma_2, mlp_branch, layerscale_uniform=layerscale_uniform,
        )
    branch2 = self.drop_path2(mlp_branch)
    if residual_rule == "ratio":
        x = residual_ratio(x, branch2)
    else:
        x = divide_gradient(x + branch2, 2)
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
    if residual_rule == "ratio":
        x = residual_ratio(x, branch1)
    else:
        x = divide_gradient(x + branch1, 2)
    branch2 = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    if residual_rule == "ratio":
        x = residual_ratio(x, branch2)
    else:
        x = divide_gradient(x + branch2, 2)
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
    ``dynamic_img_size``) except the result of the additive step is wrapped
    in :func:`divide_gradient` with ``factor=2``.
    """
    to_cat = []
    if self.cls_token is not None:
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
    if self.reg_token is not None:
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

    if self.pos_embed is None:
        return torch.cat(to_cat + [x.view(x.shape[0], -1, x.shape[-1])], dim=1)

    if self.dynamic_img_size:
        from timm.layers.pos_embed import resample_abs_pos_embed
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
        x = divide_gradient(x + pos_embed, 2)
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = divide_gradient(x + pos_embed, 2)

    return self.pos_drop(x)


# ─── 4. Canonizers — one class per kind of model graph mutation ─────────────


_TAPS = ("qkv_tap", "attn_out_tap")


class AttentionTapsCanonizer(Canonizer):
    """Inject ``qkv_tap`` and ``attn_out_tap`` (both ``nn.Identity``) into
    every timm-style ``Attention`` submodule.

    Detection: presence of ``qkv`` (a ``nn.Linear``), ``num_heads``,
    ``head_dim``.

    Both taps are named hook points consumed by
    :mod:`crp.attention_concepts`:

    * ``qkv_tap`` lives between ``Linear(D, 3D)`` and the K/Q/V split,
      shape ``(B, N, 3D)``.
    * ``attn_out_tap`` lives between the per-head attention output (after
      ``attn @ v``, optional post-attention norm) and ``proj``, shape
      ``(B, N, D)``.

    Registers on ``apply``, removes on :meth:`remove` (so the model is
    reverted when ``composite.context()`` exits). User-pre-injected taps
    are respected and not auto-removed.
    """

    def __init__(self):
        self.module: Optional[nn.Module] = None
        # Tracks which of the named taps THIS instance added (so remove()
        # only deletes the ones it created, not any pre-injected by the user).
        self._added: List[str] = []

    def apply(self, root_module):
        instances = []
        for _name, module in root_module.named_modules():
            if (
                hasattr(module, "qkv")
                and isinstance(getattr(module, "qkv"), nn.Linear)
                and hasattr(module, "num_heads")
                and hasattr(module, "head_dim")
            ):
                inst = self.copy()
                inst.register(module)
                instances.append(inst)
        return instances

    def register(self, module):
        self.module = module
        self._added = []
        for tap_name in _TAPS:
            existing = getattr(module, tap_name, None)
            if isinstance(existing, nn.Identity):
                continue
            module.add_module(tap_name, nn.Identity())
            self._added.append(tap_name)

    def remove(self):
        if self.module is None:
            return
        for tap_name in self._added:
            if tap_name in self.module._modules:
                del self.module._modules[tap_name]
        self._added = []

    def copy(self):
        return type(self)()


# Back-compat alias: pre-iter-10 code referenced ``QKVTapCanonizer``.
QKVTapCanonizer = AttentionTapsCanonizer


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


class GELUIdentityRuleCanonizer(AttributeCanonizer):
    """Canonizer that routes ``nn.GELU`` through :class:`_IdentityRuleFn`.

    AttnLRP §3.2.2 — element-wise non-linearities use the identity rule
    (relevance flows back through the input/output ratio).

    Parameters
    ----------
    signed_epsilon : bool
        When True, use sign-aware ε (``input + ε·sign(input)``) in the
        identity rule's stabiliser instead of plain ``input + ε``.
        AttnLRP Eq. 16. Default False (back-compat with the existing
        baseline composite).
    epsilon : float
        Stabiliser magnitude. Default ``1e-6``.
    """

    def __init__(self, *, signed_epsilon: bool = False, epsilon: float = 1e-6):
        self.signed_epsilon = signed_epsilon
        self.epsilon = epsilon
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, nn.GELU):
            return None
        original_forward = type(module).forward
        signed = self.signed_epsilon
        eps = self.epsilon

        def patched(self, x):
            return identity_rule_implicit(
                lambda inp: original_forward(self, inp), x,
                epsilon=eps, signed=signed,
            )

        return _bind_forward(module, patched)

    def copy(self):
        return type(self)(signed_epsilon=self.signed_epsilon, epsilon=self.epsilon)


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


class TimmAttentionForwardCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``forward`` on timm ``vision_transformer.Attention``
    for the AttnLRP-compatible :func:`_timm_attention_forward`.

    Parameters
    ----------
    matmul_factor_2_rule : bool
        When True, use :class:`_MatmulFactor2Fn` on the ``QKᵀ`` and
        ``attn·V`` bilinears (AttnLRP Prop. 3.3) and skip the
        operand-side ``divide_gradient(q,4)/(k,4)/(v,2)`` calls.
    signed_epsilon : bool
        When True with ``matmul_factor_2_rule``, use sign-aware ε in the
        bilinear stabiliser (AttnLRP Eq. 16).
    epsilon : float
        ε magnitude when ``matmul_factor_2_rule`` is enabled.
    """

    def __init__(
        self, *, matmul_factor_2_rule: bool = False,
        signed_epsilon: bool = False, epsilon: float = 1e-6,
    ):
        self.matmul_factor_2_rule = matmul_factor_2_rule
        self.signed_epsilon = signed_epsilon
        self.epsilon = epsilon
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        try:
            from timm.models.vision_transformer import Attention as TimmAttention
        except ImportError:
            return None
        if not isinstance(module, TimmAttention):
            return None
        m2 = self.matmul_factor_2_rule
        sgn = self.signed_epsilon
        eps = self.epsilon

        def fwd(self, x, attn_mask=None, is_causal=False):
            return _timm_attention_forward(
                self, x, attn_mask=attn_mask, is_causal=is_causal,
                matmul_factor_2_rule=m2, signed_epsilon=sgn, epsilon=eps,
            )

        return _bind_forward(module, fwd)

    def copy(self):
        return type(self)(
            matmul_factor_2_rule=self.matmul_factor_2_rule,
            signed_epsilon=self.signed_epsilon,
            epsilon=self.epsilon,
        )


class EvaAttentionForwardCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``forward`` on timm ``eva.EvaAttention`` (used by
    DINOv3 / EVA / BEiT-style models) for :func:`_eva_attention_forward`.

    Parameters
    ----------
    matmul_factor_2_rule : bool
        See :class:`TimmAttentionForwardCanonizer`.
    rope_detach : bool
        When True, ``rope.detach()`` before applying the rotary
        embedding. RoPE has no learnable parameters (Su et al. 2021,
        arXiv:2104.09864) so cos/sin can be treated as graph constants.
    signed_epsilon : bool
        See :class:`TimmAttentionForwardCanonizer`.
    epsilon : float
        ε magnitude when ``matmul_factor_2_rule`` is enabled.
    """

    def __init__(
        self, *, matmul_factor_2_rule: bool = False, rope_detach: bool = False,
        signed_epsilon: bool = False, epsilon: float = 1e-6,
    ):
        self.matmul_factor_2_rule = matmul_factor_2_rule
        self.rope_detach = rope_detach
        self.signed_epsilon = signed_epsilon
        self.epsilon = epsilon
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        try:
            from timm.models.eva import EvaAttention
        except ImportError:
            return None
        if not isinstance(module, EvaAttention):
            return None
        m2 = self.matmul_factor_2_rule
        rd = self.rope_detach
        sgn = self.signed_epsilon
        eps = self.epsilon

        def fwd(self, x, rope=None, attn_mask=None, is_causal=False):
            return _eva_attention_forward(
                self, x, rope=rope, attn_mask=attn_mask, is_causal=is_causal,
                matmul_factor_2_rule=m2, rope_detach=rd,
                signed_epsilon=sgn, epsilon=eps,
            )

        return _bind_forward(module, fwd)

    def copy(self):
        return type(self)(
            matmul_factor_2_rule=self.matmul_factor_2_rule,
            rope_detach=self.rope_detach,
            signed_epsilon=self.signed_epsilon,
            epsilon=self.epsilon,
        )


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
        try:
            from timm.models.vision_transformer import Block as TimmBlock
        except ImportError:
            return None
        if not isinstance(module, TimmBlock):
            return None
        needed = ("ls1", "ls2", "drop_path1", "drop_path2", "norm1",
                  "norm2", "attn", "mlp")
        if not all(hasattr(module, a) for a in needed):
            return None
        rule = self.residual_rule

        def fwd(self, x, attn_mask=None, is_causal=False):
            return _timm_block_forward(
                self, x, attn_mask=attn_mask, is_causal=is_causal,
                residual_rule=rule,
            )

        return _bind_forward(module, fwd)

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
        When True, apply :func:`divide_gradient(γ·x, 2)` to the LayerScale
        multiplications (AttnLRP Eq. 7 uniform rule on bilinears, treating
        γ as a constant operand). Default False.
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
        try:
            from timm.models.eva import EvaBlock
        except ImportError:
            return None
        if not isinstance(module, EvaBlock):
            return None
        needed = ("drop_path1", "drop_path2", "norm1", "norm2", "attn", "mlp",
                  "gamma_1", "gamma_2")
        if not all(hasattr(module, a) for a in needed):
            return None
        rule = self.residual_rule
        ls_uniform = self.layerscale_uniform

        def fwd(self, x, rope=None, attn_mask=None, is_causal=False):
            return _eva_block_forward(
                self, x, rope=rope, attn_mask=attn_mask, is_causal=is_causal,
                residual_rule=rule, layerscale_uniform=ls_uniform,
            )

        return _bind_forward(module, fwd)

    def copy(self):
        return type(self)(
            residual_rule=self.residual_rule,
            layerscale_uniform=self.layerscale_uniform,
        )


class VitPosEmbedPALRPCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``_pos_embed`` on timm ``VisionTransformer``
    instances to apply the PA-LRP uniform rule (Bakish et al. 2025;
    arXiv:2506.02138). See :func:`vit_pos_embed_palrp`."""

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        try:
            from timm.models.vision_transformer import VisionTransformer
        except ImportError:
            return None
        if not isinstance(module, VisionTransformer) or not hasattr(module, "_pos_embed"):
            return None
        return _bind_forward(module, vit_pos_embed_palrp, attr="_pos_embed")

    def copy(self):
        return type(self)()


class TimmViTCanonizer(CompositeCanonizer):
    """Aggregator: bundles all single-responsibility canonizers needed for
    AttnLRP on a timm ViT (standard or Eva-stack).

    Combines:

    * :class:`AttentionTapsCanonizer`
    * :class:`LayerNormForwardCanonizer`
    * :class:`GELUIdentityRuleCanonizer`
    * :class:`DropoutPassthroughCanonizer`
    * :class:`TimmAttentionForwardCanonizer` and
      :class:`EvaAttentionForwardCanonizer` (each only fires on its own
      module type, so safe to install both)
    * :class:`VitPosEmbedPALRPCanonizer` (when ``palrp=True``)
    * :class:`TimmBlockResidualCanonizer` and
      :class:`EvaBlockResidualCanonizer` (when ``residual_lrp`` is set)

    All mutations are instance-level and reversible. Bundled into the
    named composites below; pass it explicitly to other composites if you
    want a custom rule map.

    The remedy-flag parameters (``matmul_factor_2_rule`` etc.) propagate
    to the relevant per-module canonizer. Each is also exposed as a
    standalone composite — use those instead of toggling here unless
    deliberately combining several remedies.
    """

    def __init__(
        self,
        *,
        palrp: bool = False,
        residual_lrp: Optional[str] = None,
        matmul_factor_2_rule: bool = False,
        rope_detach: bool = False,
        signed_epsilon: bool = False,
        layerscale_uniform: bool = False,
        epsilon: float = 1e-6,
    ):
        canonizers: List[Canonizer] = [
            AttentionTapsCanonizer(),
            LayerNormForwardCanonizer(),
            GELUIdentityRuleCanonizer(signed_epsilon=signed_epsilon, epsilon=epsilon),
            DropoutPassthroughCanonizer(),
            TimmAttentionForwardCanonizer(
                matmul_factor_2_rule=matmul_factor_2_rule,
                signed_epsilon=signed_epsilon,
                epsilon=epsilon,
            ),
            EvaAttentionForwardCanonizer(
                matmul_factor_2_rule=matmul_factor_2_rule,
                rope_detach=rope_detach,
                signed_epsilon=signed_epsilon,
                epsilon=epsilon,
            ),
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
            # User asked for layerscale_uniform but didn't pick a residual rule;
            # default to ratio so the EvaBlock forward gets installed (the
            # layerscale_uniform wrapper lives inside that forward).
            canonizers.append(EvaBlockResidualCanonizer(
                residual_rule="ratio", layerscale_uniform=True,
            ))
        super().__init__(canonizers)


# ─── 5. Hooks — LRP backward for Linear / Conv2d ─────────────────────────────
#
# We use zennit's stock :class:`zennit.rules.Epsilon` and
# :class:`zennit.rules.Gamma` directly. The previous version of this module
# shipped a ``GradientTimesInputBasicHook`` subclass that pre-multiplied
# ``grad_output`` by the layer's output and post-divided the returned
# relevance by ``stabilize(input)`` — neither of those factors is part of the
# LRP-ε rule (Bach et al. 2015; Montavon et al. 2019,
# iphome.hhi.de/samek/pdf/MonXAI19.pdf). The conservation audit in
# ``experiments/audit_gti_hook.py`` confirmed the discrepancy: standard
# ``Epsilon`` gives sum(R_in)/sum(R_out) ≈ 1 + O(ε), the GTI hook gave
# −2.1 (i.e. 210 % deviation) on ordinary inputs, blowing up further
# whenever input components were near ε. The thin aliases below preserve
# the public symbol names (back-compat with existing imports / tests).


class GTIEpsilon(Epsilon):
    """**Deprecated alias** for :class:`zennit.rules.Epsilon`.

    Earlier versions of this module shipped a custom
    ``GradientTimesInputBasicHook`` subclass under this name with a
    rule that did *not* match LRP-ε and violated conservation by 100×
    on ordinary inputs (audited in ``experiments/audit_gti_hook.py``).
    Now a thin alias for zennit's stock ε-LRP rule. Kept for back-compat;
    new code should import :class:`zennit.rules.Epsilon` directly.
    """


class GTIGamma(Gamma):
    """**Deprecated alias** for :class:`zennit.rules.Gamma`.

    Same story as :class:`GTIEpsilon`: previously a buggy GTI subclass,
    now a thin alias for zennit's stock γ-LRP rule. AttnLRP §3.2.1
    recommends γ ≈ 0.25 on ViT linears.
    """


# ─── 6. Composites — one per remedy / recipe ─────────────────────────────────


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

    * ``nn.Linear`` → :class:`GTIEpsilon` (ε-LRP, gradient×input form).
    * ``nn.Conv2d`` → :class:`GTIEpsilon` (covers the patch-embed Conv2d).
    * ``nn.GELU`` / ``nn.LayerNorm`` / ``nn.Dropout`` / ``nn.Identity``
      → :class:`zennit.rules.Pass` — handled in-graph by the embedded
      autograd functions installed by :class:`TimmViTCanonizer`.

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
            TimmViTCanonizer(palrp=palrp, residual_lrp=residual_lrp, epsilon=epsilon),
        ]
        super().__init__(layer_map=_epsilon_layer_map(epsilon), canonizers=canonizers)


class AttnLRPGammaComposite(LayerMapComposite):
    """AttnLRP with γ-LRP on ViT linears (Achtibat §3.2.1).

    Same structure as :class:`AttnLRPEpsilonComposite` but maps
    ``nn.Linear`` and ``nn.Conv2d`` to :class:`GTIGamma` instead of
    :class:`GTIEpsilon`. Recommended over the ε-only variant when
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
            TimmViTCanonizer(palrp=palrp, residual_lrp=residual_lrp, epsilon=epsilon),
        ]
        super().__init__(
            layer_map=_gamma_layer_map(gamma, epsilon), canonizers=canonizers,
        )


class AttnLRPMatmulFactor2Composite(AttnLRPEpsilonComposite):
    """AttnLRP-ε with the bilinear matmul rule from Achtibat et al. Prop. 3.3
    enabled (the ``2·Y + ε`` stabiliser on ``QKᵀ`` and ``attn·V``).

    Drops the operand-side ``divide_gradient(q,4)/(k,4)/(v,2)`` calls in
    favour of the AttnLRP rule's output-side stabilisation. See
    :class:`_MatmulFactor2Fn`. Identical to :class:`AttnLRPEpsilonComposite`
    in every other respect.

    Reference: Achtibat et al. ICML 2024, arXiv:2402.05602, Proposition 3.3
    and Eq. 14.
    """

    def __init__(
        self, epsilon: float = 1e-6, canonizers=None, *,
        palrp: bool = False, residual_lrp: Optional[str] = None,
    ):
        canonizers = list(canonizers or []) + [
            TimmViTCanonizer(
                palrp=palrp, residual_lrp=residual_lrp,
                matmul_factor_2_rule=True, epsilon=epsilon,
            ),
        ]
        # Skip the parent constructor's canonizer wiring; install our own.
        LayerMapComposite.__init__(
            self, layer_map=_epsilon_layer_map(epsilon), canonizers=canonizers,
        )


class AttnLRPSignedEpsilonComposite(AttnLRPEpsilonComposite):
    """AttnLRP-ε with sign-aware ε in the identity rule (and the matmul rule
    if a future composite combines them).

    The identity rule's stabiliser becomes ``input + ε·sign(input)``
    instead of ``input + ε``, preserving sign behaviour around zero
    crossings (AttnLRP Eq. 16, arXiv:2402.05602).
    """

    def __init__(
        self, epsilon: float = 1e-6, canonizers=None, *,
        palrp: bool = False, residual_lrp: Optional[str] = None,
    ):
        canonizers = list(canonizers or []) + [
            TimmViTCanonizer(
                palrp=palrp, residual_lrp=residual_lrp,
                signed_epsilon=True, epsilon=epsilon,
            ),
        ]
        LayerMapComposite.__init__(
            self, layer_map=_epsilon_layer_map(epsilon), canonizers=canonizers,
        )


class AttnLRPRopeDetachComposite(AttnLRPEpsilonComposite):
    """AttnLRP-ε with the RoPE rotary embedding detached from the autograd
    graph (RoFormer; Su et al. 2021, arXiv:2104.09864).

    No effect on models without RoPE — the canonizer only fires on
    ``timm.models.eva.EvaAttention`` and the detach happens inside the
    rope-application branch (no rope, no detach).
    """

    def __init__(
        self, epsilon: float = 1e-6, canonizers=None, *,
        palrp: bool = False, residual_lrp: Optional[str] = None,
    ):
        canonizers = list(canonizers or []) + [
            TimmViTCanonizer(
                palrp=palrp, residual_lrp=residual_lrp,
                rope_detach=True, epsilon=epsilon,
            ),
        ]
        LayerMapComposite.__init__(
            self, layer_map=_epsilon_layer_map(epsilon), canonizers=canonizers,
        )


class AttnLRPLayerScaleUniformComposite(AttnLRPEpsilonComposite):
    """AttnLRP-ε with the uniform allocation rule on the LayerScale γ
    multiplications inside ``EvaBlock``.

    The ``γ·branch`` step is wrapped in :func:`divide_gradient(., 2)` so
    γ (a learned scalar with no upstream input) absorbs half the relevance
    by the AttnLRP uniform rule (Eq. 7 of arXiv:2402.05602). LayerScale
    introduced by Touvron et al. (CaiT, arXiv:2103.17239); without
    explicit handling it acts as a transparent multiplier and shrinks the
    branch's relevance by the small initial value of γ (1e-4 to 1e-5).

    This composite installs the residual rule too (defaults to ``'ratio'``)
    because the LayerScale wrapper lives inside the block-forward
    replacement.
    """

    def __init__(
        self, epsilon: float = 1e-6, canonizers=None, *,
        palrp: bool = False, residual_lrp: Optional[str] = "ratio",
    ):
        canonizers = list(canonizers or []) + [
            TimmViTCanonizer(
                palrp=palrp, residual_lrp=residual_lrp,
                layerscale_uniform=True, epsilon=epsilon,
            ),
        ]
        LayerMapComposite.__init__(
            self, layer_map=_epsilon_layer_map(epsilon), canonizers=canonizers,
        )


class AttnLRPLinearGammaComposite(AttnLRPGammaComposite):
    """Conservative variant of :class:`AttnLRPGammaComposite` with γ=0.05
    on ``nn.Linear`` instead of the AttnLRP-paper default γ=0.25.

    Smaller γ shifts the rule toward ε-LRP — useful when the standard γ
    inflates relevance on deep stacks (we have observed ~10¹⁶ inflation
    on vit_base under γ=0.25 with the milestone-A audit pipeline).
    """

    def __init__(
        self, gamma: float = 0.05, epsilon: float = 1e-6, canonizers=None, *,
        palrp: bool = False, residual_lrp: Optional[str] = None,
    ):
        super().__init__(
            gamma=gamma, epsilon=epsilon, canonizers=canonizers,
            palrp=palrp, residual_lrp=residual_lrp,
        )


class AttnLRPCombinedComposite(LayerMapComposite):
    """Combine multiple already-validated remedies into one composite.

    Use this **after** each remedy has been individually evaluated with its
    dedicated composite class — it's the single composite whose
    responsibility is "express a deliberate combination". Toggles here are
    therefore appropriate (the responsibility of this class is precisely
    to compose).

    Parameters
    ----------
    matmul_factor_2 : bool
        Enable :class:`_MatmulFactor2Fn` (see
        :class:`AttnLRPMatmulFactor2Composite`).
    signed_epsilon : bool
        Enable sign-aware ε in the identity rule and the matmul rule (see
        :class:`AttnLRPSignedEpsilonComposite`).
    rope_detach : bool
        Detach RoPE (see :class:`AttnLRPRopeDetachComposite`).
    layerscale_uniform : bool
        Uniform rule on LayerScale (see
        :class:`AttnLRPLayerScaleUniformComposite`).
    linear_gamma : float | None
        If non-None, use γ-LRP on Linears with this γ instead of ε-LRP.
        Recommended ≤0.25 per AttnLRP §3.2.1.
    epsilon : float
        ε for the LRP rule denominators. Default 1e-6.
    palrp, residual_lrp : as elsewhere.
    """

    def __init__(
        self, *,
        matmul_factor_2: bool = False,
        signed_epsilon: bool = False,
        rope_detach: bool = False,
        layerscale_uniform: bool = False,
        linear_gamma: Optional[float] = None,
        epsilon: float = 1e-6,
        palrp: bool = False,
        residual_lrp: Optional[str] = None,
        canonizers=None,
    ):
        if layerscale_uniform and residual_lrp is None:
            residual_lrp = "ratio"
        canonizers = list(canonizers or []) + [
            TimmViTCanonizer(
                palrp=palrp, residual_lrp=residual_lrp,
                matmul_factor_2_rule=matmul_factor_2,
                rope_detach=rope_detach,
                signed_epsilon=signed_epsilon,
                layerscale_uniform=layerscale_uniform,
                epsilon=epsilon,
            ),
        ]
        if linear_gamma is not None:
            layer_map = _gamma_layer_map(linear_gamma, epsilon)
        else:
            layer_map = _epsilon_layer_map(epsilon)
        super().__init__(layer_map=layer_map, canonizers=canonizers)


# ─── 7. Public-name aliases for the underscored helper forwards ──────────────
#
# Pre-restructure (when there was no remedies system) these were the public
# entry points for advanced users wanting to install a custom forward. We keep
# the names exported but mark them as advanced — the canonizer classes are the
# preferred API.

timm_attention_forward = _timm_attention_forward
eva_attention_forward = _eva_attention_forward
eva_block_forward = _eva_block_forward
timm_block_forward = _timm_block_forward


__all__ = [
    # autograd Function rule kernels
    "_IdentityRuleFn",
    "_DivideGradientFn",
    "_ResidualRatioFn",
    "_MatmulFactor2Fn",
    # rule convenience callables
    "identity_rule_implicit",
    "divide_gradient",
    "residual_ratio",
    "matmul_factor_2",
    "stop_gradient",
    # forward replacements (advanced; canonizers install them automatically)
    "layer_norm_forward",
    "dropout_passthrough_forward",
    "timm_attention_forward",
    "eva_attention_forward",
    "timm_block_forward",
    "eva_block_forward",
    "vit_pos_embed_palrp",
    # canonizers (one per kind of mutation)
    "AttentionTapsCanonizer",
    "QKVTapCanonizer",  # back-compat alias
    "LayerNormForwardCanonizer",
    "GELUIdentityRuleCanonizer",
    "DropoutPassthroughCanonizer",
    "TimmAttentionForwardCanonizer",
    "EvaAttentionForwardCanonizer",
    "TimmBlockResidualCanonizer",
    "EvaBlockResidualCanonizer",
    "VitPosEmbedPALRPCanonizer",
    "TimmViTCanonizer",
    # hooks (deprecated aliases — prefer zennit.rules.Epsilon / .Gamma)
    "GTIEpsilon",
    "GTIGamma",
    # composites
    "AttnLRPEpsilonComposite",
    "AttnLRPGammaComposite",
    "AttnLRPMatmulFactor2Composite",
    "AttnLRPSignedEpsilonComposite",
    "AttnLRPRopeDetachComposite",
    "AttnLRPLayerScaleUniformComposite",
    "AttnLRPLinearGammaComposite",
    "AttnLRPCombinedComposite",
]
