"""Atomic ``nn.Module`` kernels for AttnLRP-aware attention (Phase 1 of the
attention-unfolding refactor).

Each module here owns exactly one LRP rule, embedded in its forward via the
autograd ``Function`` kernels already living in
:mod:`crp.transformer_patches`. The outer container
:class:`EvaAttentionUnfolded` wires them together with the same semantics as
:func:`crp.transformer_patches._eva_attention_forward`, but exposes every
intermediate tensor as the output of a *named submodule* so:

* zennit hooks can attach by submodule name or type (no tap injection),
* concept-conditioning can target Q/K/V or post-softmax weights without
  hard-coded slice arithmetic on a fused ``qkv`` tap,
* ``register_full_backward_hook`` reports per-rule relevance directly.

Phase 1 substitutes ONE attention block with this unfolded variant and
verifies forward + backward parity with the existing canonizer-installed
``_eva_attention_forward``. Phase 2 expands to all blocks and migrates the
concept classes; Phase 3 retires ``_eva_attention_forward``.

See ``UNFOLDING_ATTENTION_REFACTOR.md`` for the full plan.

Module catalogue
----------------

Bilinear / multi-input modules (own custom autograd through their forward):

* :class:`BilinearMatmul` — ``a @ b`` with the AttnLRP Prop. 3.3 ``2y+ε``
  rule (or bare matmul when ``rule='passthrough'``).
* :class:`AddBias` — ``x + bias`` where ``bias`` is a leaf constant
  (typically the resolved attention mask). Identity-on-x backward.
* :class:`ResidualAdd` — ``x + branch`` with the Otsuki ratio split rule
  (or symmetric uniform rule).

Single-input modules:

* :class:`SoftmaxAlongLastDim` — ``F.softmax(dim=-1)`` with the AttnLRP
  identity rule for normalisations (Eq. 9; ``R_in = R_out``).
* :class:`RotaryEmbedding` — wraps ``apply_rot_embed_cat`` with optional
  ``num_prefix_tokens`` skip and optional ``rope.detach()``.
* :class:`ScaleByConstant` — ``x * scalar``; identity rule (constants
  absorb no relevance).
* :class:`ChunkAlongLastDim` — splits a tensor into ``n`` chunks along the
  last dim. Backward is concatenation, which is identity in LRP terms.
* :class:`ReshapeMergeHeads` — ``x.transpose(1,2).reshape(B,N,C)``. Pure
  reshape, identity in LRP terms.
* :class:`LayerScaleMul` — wraps the existing ``divide_gradient(γ·x, 2)``
  uniform-rule for CaiT-style LayerScale (Touvron et al. 2021).

Container modules:

* :class:`EvaAttentionUnfolded` — the unfolded analogue of
  :class:`timm.models.eva.EvaAttention`. Constructor takes a real
  ``EvaAttention`` instance and references its parameters/submodules
  directly (no copy), so weight loading and grad parity are automatic.

Canonizers:

* :class:`EvaAttentionSubstitutionCanonizer` — replaces an
  ``EvaAttention`` instance with an :class:`EvaAttentionUnfolded` wrapper
  on ``apply()``, restores the original on ``remove()``. Phase 1 default
  applies to a single block (``block_indices=(0,)``) so the rest of the
  model continues to use the existing canonizer-installed forward.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from zennit.canonizers import Canonizer

# Re-use the autograd Function kernels already validated in transformer_patches.
# Importing them keeps the LRP semantics consistent between the unfolded path
# and the legacy single-forward path.
from crp.transformer_patches import (
    _DivideGradientFn,
    _MatmulFactor2Fn,
    _ResidualRatioFn,
)


# ─── 1. New autograd Function kernels (only Softmax is genuinely new) ───────


class _SoftmaxIdentityRuleFn(Function):
    """Softmax with the AttnLRP identity rule (R_in = R_out).

    AttnLRP §3.2.2 / Eq. 9 (Achtibat et al. 2024, arXiv:2402.05602): for
    normalisations like softmax, the LRP-0 derivation reduces to identity
    because the softmax outputs are 1×1-linear-equivalent in y/x. The
    paper's recipe is to *pass relevance straight through softmax*: the
    upstream relevance ``R_y`` is also the relevance on the pre-softmax
    scores ``R_x``.

    Why we need a custom Function: bare ``F.softmax`` would route relevance
    through the actual Jacobian (which couples positions via the softmax
    normalisation, producing non-conserving relevance flows). The identity
    rule short-circuits that.

    Forward: standard ``F.softmax(x, dim=-1)``.
    Backward: ``grad_out`` (i.e. ``R_in = R_out``).
    """

    @staticmethod
    def forward(ctx, x):
        return F.softmax(x, dim=-1)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


class _ScaleIdentityRuleFn(Function):
    """``y = x * scalar`` with R_in = R_out (constant absorbs no relevance).

    AttnLRP treats multiplication by a graph-constant operand (no upstream
    input, e.g. ``self.scale = head_dim**-0.5``) as identity — the constant
    has no relevance to receive, so the bilinear's R_x is the full
    upstream R_y. Bare autograd would multiply ``grad_out`` by the scalar,
    which is wrong for an LRP allocation step (the scaling cancels out
    when we compute ``R = grad·input`` — but because zennit's Linear hook
    already accounts for the scale as part of ``W``, double-counting at
    this node is what we avoid).

    This module is most relevant when a downstream consumer (the matmul
    rule, ε-LRP on a Linear) computes ``R = some_factor·grad``. Pulling
    the scale out as identity keeps the backward magnitude correct.
    """

    @staticmethod
    def forward(ctx, x, scalar: float):
        ctx.scalar = scalar
        return x * scalar

    @staticmethod
    def backward(ctx, grad_out):
        # Identity on the relevance: pass grad_out through unchanged.
        # The scalar is a constant; no second-input grad to return.
        return grad_out, None


# ─── 2. Bilinear / 2-input nn.Modules ────────────────────────────────────────


class BilinearMatmul(nn.Module):
    """``y = a @ b`` with one of the AttnLRP bilinear rules.

    Parameters
    ----------
    rule : {'matmul_factor_2', 'passthrough'}
        ``'matmul_factor_2'`` (default) routes through
        :class:`crp.transformer_patches._MatmulFactor2Fn` — the AttnLRP
        Prop. 3.3 ``2y+ε`` stabiliser. Conservation:
        ``sum(R_a) + sum(R_b) ≈ sum(R_y)``.
        ``'passthrough'`` uses bare ``torch.matmul`` so autograd flows
        the natural Jacobian. Used in tests for forward parity without
        any LRP overlay.
    epsilon : float
        ε for the ``2y+ε`` denominator. Default 1e-6.
    signed : bool
        When True, use sign-aware ε (AttnLRP Eq. 16). Default False.

    Notes on hook framework integration (plan §P1)
    ----------------------------------------------
    zennit's :class:`zennit.core.BasicHook` only attributes the *first*
    positional input — it stores ``self.stored_tensors['input'] = args``
    but its ``backward`` only re-runs the forward through the
    ``input_modifiers`` on ``original_input = args[0]``, leaving any
    further positional args as constants. That's a poor fit for true
    bilinear ops where both ``a`` and ``b`` should receive relevance.

    Rather than write a custom 2-input ``Hook`` subclass, we bake the
    AttnLRP rule directly into the module's ``forward`` via the
    autograd ``Function`` (``_MatmulFactor2Fn``). zennit then attaches
    no rule to ``BilinearMatmul`` itself (use ``Pass`` in the layer
    map). The Function's backward already returns the right ``(R_a,
    R_b)`` pair. This keeps the design simple and inherits the
    already-audited rule kernel.
    """

    def __init__(
        self,
        *,
        rule: str = "matmul_factor_2",
        epsilon: float = 1e-6,
        signed: bool = False,
    ):
        super().__init__()
        if rule not in ("matmul_factor_2", "passthrough"):
            raise ValueError(
                f"rule must be 'matmul_factor_2' or 'passthrough'; got {rule!r}"
            )
        self.rule = rule
        self.epsilon = epsilon
        self.signed = signed

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.rule == "matmul_factor_2":
            return _MatmulFactor2Fn.apply(a, b, self.epsilon, self.signed)
        return a @ b

    def extra_repr(self) -> str:
        return f"rule={self.rule!r}, epsilon={self.epsilon}, signed={self.signed}"


class AddBias(nn.Module):
    """``y = x + bias`` where ``bias`` is a leaf constant (e.g. attention mask).

    When ``bias is None`` the forward is identity (no add). The mask
    tensor has no upstream input — it's a graph constant — so it
    absorbs no relevance under any LRP rule. Bare autograd does the
    right thing: ``grad_x = grad_y`` and ``grad_bias`` is computed too
    but discarded by the consumer.
    """

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if bias is None:
            return x
        return x + bias


class ResidualAdd(nn.Module):
    """``y = x + branch`` with one of the LRP residual rules.

    Parameters
    ----------
    rule : {'ratio', 'symmetric'}
        ``'ratio'`` (default): Otsuki ratio split through
        :class:`_ResidualRatioFn` — distributes ``R_y`` ∝ ``|x|`` /
        ``|branch|``.
        ``'symmetric'``: factor-2 uniform allocation through
        :class:`_DivideGradientFn` — both ``x`` and ``branch`` receive
        ``R_y / 2``.
    epsilon : float
        Stabiliser for the ratio rule. Default 1e-6.
    """

    def __init__(self, *, rule: str = "ratio", epsilon: float = 1e-6):
        super().__init__()
        if rule not in ("ratio", "symmetric"):
            raise ValueError(
                f"rule must be 'ratio' or 'symmetric'; got {rule!r}"
            )
        self.rule = rule
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        if self.rule == "ratio":
            return _ResidualRatioFn.apply(x, branch, self.epsilon)
        # symmetric: bare addition followed by /2 grad split.
        return _DivideGradientFn.apply(x + branch, 2)

    def extra_repr(self) -> str:
        return f"rule={self.rule!r}"


# ─── 3. Single-input nn.Modules ─────────────────────────────────────────────


class SoftmaxAlongLastDim(nn.Module):
    """``F.softmax(x, dim=-1)`` with the AttnLRP identity rule on backward.

    See :class:`_SoftmaxIdentityRuleFn` for the rule derivation. Used
    inside :class:`EvaAttentionUnfolded` between ``q@kᵀ`` and
    ``attn@v``.

    Parameters
    ----------
    rule : {'identity', 'passthrough'}
        ``'identity'`` (default): apply the AttnLRP identity rule on
        backward — ``R_in = R_out`` per Eq. 9 of arXiv:2402.05602.
        ``'passthrough'``: bare ``F.softmax``, autograd's natural
        Jacobian. Used in tests to verify forward parity / autograd
        backward parity against stock attention.
    """

    def __init__(self, *, rule: str = "identity"):
        super().__init__()
        if rule not in ("identity", "passthrough"):
            raise ValueError(
                f"rule must be 'identity' or 'passthrough'; got {rule!r}"
            )
        self.rule = rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rule == "identity":
            return _SoftmaxIdentityRuleFn.apply(x)
        return F.softmax(x, dim=-1)

    def extra_repr(self) -> str:
        return f"rule={self.rule!r}"


class RotaryEmbedding(nn.Module):
    """Apply RoPE (rotary positional embedding) to a Q or K tensor.

    Wraps ``timm.layers.apply_rot_embed_cat`` and handles the
    ``num_prefix_tokens`` skip used by DINOv3 / EVA-style models (the
    cls + register tokens at the front of the sequence are NOT
    rotated).

    Parameters
    ----------
    num_prefix_tokens : int
        Number of leading positions left un-rotated.
    rotate_half : bool
        Layout flag passed to ``apply_rot_embed_cat`` — ``False`` for the
        interleaved layout, ``True`` for the half-split layout. Mirrors
        ``EvaAttention.rotate_half``.
    detach_rope : bool
        When True, ``rope.detach()`` before the rotary op. RoPE has no
        learnable parameters (Su et al. 2021, arXiv:2104.09864), so
        cos/sin can be treated as graph constants and relevance routes
        purely through the rotated tensor. Equivalent to the
        :class:`crp.transformer_patches.AttnLRPRopeDetachComposite`
        remedy for this op.

    Forward signature: ``(q, rope)``. ``rope`` may be ``None`` (no
    rotation; identity).
    """

    def __init__(
        self,
        num_prefix_tokens: int = 0,
        *,
        rotate_half: bool = False,
        detach_rope: bool = False,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.rotate_half = rotate_half
        self.detach_rope = detach_rope

    def forward(
        self, q: torch.Tensor, rope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rope is None:
            return q
        from timm.layers import apply_rot_embed_cat

        rope_used = rope.detach() if self.detach_rope else rope
        npt = self.num_prefix_tokens
        if npt == 0:
            return apply_rot_embed_cat(q, rope_used, half=self.rotate_half)
        prefix = q[:, :, :npt, :]
        rotated = apply_rot_embed_cat(q[:, :, npt:, :], rope_used, half=self.rotate_half)
        return torch.cat([prefix, rotated], dim=2)

    def extra_repr(self) -> str:
        return (
            f"num_prefix_tokens={self.num_prefix_tokens}, "
            f"rotate_half={self.rotate_half}, detach_rope={self.detach_rope}"
        )


class ScaleByConstant(nn.Module):
    """``y = x * value`` with the identity LRP rule (constant absorbs no R).

    See :class:`_ScaleIdentityRuleFn` for the rule. With ``rule='passthrough'``
    the multiplication is bare and autograd's natural backward (which
    multiplies by the scalar) is used — useful for verifying forward
    parity / autograd backward parity against stock attention.
    """

    def __init__(self, value: float, *, rule: str = "identity"):
        super().__init__()
        if rule not in ("identity", "passthrough"):
            raise ValueError(
                f"rule must be 'identity' or 'passthrough'; got {rule!r}"
            )
        self.value = float(value)
        self.rule = rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rule == "identity":
            return _ScaleIdentityRuleFn.apply(x, self.value)
        return x * self.value

    def extra_repr(self) -> str:
        return f"value={self.value}, rule={self.rule!r}"


class ChunkAlongLastDim(nn.Module):
    """Split a tensor into ``n`` equal chunks along the last dim.

    Backward is ``torch.cat`` along the same dim, which is identity in
    LRP terms (each chunk's relevance returns to its original slice in
    the cat'd tensor; total ``sum(R)`` is preserved). Bare autograd
    handles this correctly — no custom Function needed.
    """

    def __init__(self, n: int):
        super().__init__()
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}")
        self.n = n

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(x.chunk(self.n, dim=-1))

    def extra_repr(self) -> str:
        return f"n={self.n}"


class ReshapeMergeHeads(nn.Module):
    """Merge per-head context back into ``(B, N, C)``.

    Forward: ``x.transpose(1, 2).reshape(B, N, out_dim)``. ``out_dim``
    defaults to the product ``num_heads * head_dim`` inferred from the
    input shape; pass explicitly when the head dim differs from the
    output dim (the rare ``attn_dim`` variant in some timm Attention
    classes).

    Identity in LRP terms — pure reshape preserves ``sum(R)`` and
    bare autograd routes relevance correctly.
    """

    def __init__(self, out_dim: Optional[int] = None):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_heads, N, head_dim)
        B, _, N, _ = x.shape
        out_dim = self.out_dim
        if out_dim is None:
            out_dim = x.shape[1] * x.shape[3]
        return x.transpose(1, 2).reshape(B, N, out_dim)

    def extra_repr(self) -> str:
        return f"out_dim={self.out_dim}"


class LayerScaleMul(nn.Module):
    """``y = γ * x`` with the AttnLRP uniform rule (CaiT-style LayerScale).

    γ is a learnable per-channel parameter (Touvron et al. 2021,
    arXiv:2103.17239). With no upstream input it's a graph leaf, so the
    AttnLRP uniform rule (Eq. 7, arXiv:2402.05602) allocates half the
    relevance to γ (which is "absorbed") and half back to ``x``.

    Implementation: forward computes ``γ·x`` then routes through
    :class:`_DivideGradientFn` with ``factor=2``. With
    ``layerscale_uniform=False`` the multiplication is bare and the
    branch loses ``γ × …`` of its relevance — usually the wrong
    behaviour for LayerScale's small initial γ (1e-4..1e-5).

    Parameters
    ----------
    gamma : nn.Parameter
        The LayerScale parameter from the parent ``EvaBlock``. Shared
        by reference; not copied.
    layerscale_uniform : bool
        When True (default for this Module — the whole point of using
        it), divide the upstream gradient by 2.
    """

    def __init__(self, gamma: nn.Parameter, *, layerscale_uniform: bool = True):
        super().__init__()
        # NB: ``gamma`` is a Parameter on the parent module; we reference
        # it without re-registering, so weight loading still flows through
        # the parent.
        self.gamma = gamma
        self.layerscale_uniform = layerscale_uniform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gamma * x
        if self.layerscale_uniform:
            out = _DivideGradientFn.apply(out, 2)
        return out

    def extra_repr(self) -> str:
        return f"layerscale_uniform={self.layerscale_uniform}"


# ─── 4. Container: EvaAttentionUnfolded ─────────────────────────────────────


class EvaAttentionUnfolded(nn.Module):
    """Unfolded ``timm.models.eva.EvaAttention`` analogue.

    Constructor takes a *real* ``EvaAttention`` instance and stores
    references to its parameters / sub-modules. The container does NOT
    copy weights — the original module's parameters live on, accessible
    via ``self.qkv``, ``self.proj`` etc. So when this container is
    swapped in by :class:`EvaAttentionSubstitutionCanonizer`, weight
    loading from a checkpoint Just Works (the checkpoint targets
    ``blocks.{i}.attn.qkv.weight`` and we still have an ``attn.qkv``
    submodule).

    Forward signature matches ``EvaAttention.forward`` exactly:
    ``(self, x, rope=None, attn_mask=None, is_causal=False)``. The
    semantics mirror :func:`crp.transformer_patches._eva_attention_forward`
    with the ``matmul_factor_2_rule=True, layerscale_uniform=True,
    residual_lrp='ratio'`` recipe — the working_combo for DINOv3.

    Parameters
    ----------
    orig : timm.models.eva.EvaAttention
        Source module to wrap. Its parameters become this container's
        parameters by reference.
    matmul_rule : {'matmul_factor_2', 'passthrough'}
        Controls the rule on both bilinear matmuls. Default
        ``'matmul_factor_2'`` (AttnLRP Prop. 3.3).
    epsilon : float
        ε for the matmul rule. Default 1e-6.
    signed_epsilon : bool
        Sign-aware ε in the matmul rule. Default False.
    rope_detach : bool
        Detach RoPE in the rotary op. Default False.
    """

    def __init__(
        self,
        orig,  # timm.models.eva.EvaAttention; not type-annotated to allow import-light tests
        *,
        matmul_rule: str = "matmul_factor_2",
        epsilon: float = 1e-6,
        signed_epsilon: bool = False,
        rope_detach: bool = False,
    ):
        super().__init__()
        if not (
            hasattr(orig, "qkv")
            and isinstance(orig.qkv, nn.Linear)
            and hasattr(orig, "num_heads")
            and hasattr(orig, "proj")
        ):
            raise TypeError(
                "EvaAttentionUnfolded expects a timm EvaAttention-like instance"
            )
        # The DINOv3 ViT-L variant: q_bias=k_bias=v_bias=None, qkv.bias=False.
        # Variants with separate per-Q/K/V biases (vit_*_dinov3_qkvb) need
        # a different qkv path; not handled in Phase 1.
        if getattr(orig, "q_bias", None) is not None:
            raise NotImplementedError(
                "Phase 1 EvaAttentionUnfolded does not support per-Q/K/V bias "
                "EvaAttention variants (q_bias is set). Lift the bias-cat "
                "branch from the upstream forward when adding support."
            )

        # Cache shape constants from the original module.
        self.num_heads = int(orig.num_heads)
        self.head_dim = int(orig.head_dim)
        self.num_prefix_tokens = int(getattr(orig, "num_prefix_tokens", 0))
        self.scale = float(orig.scale)
        rotate_half = bool(getattr(orig, "rotate_half", False))

        # Reference (do not copy) the parameter-bearing submodules.
        self.qkv = orig.qkv
        self.q_norm = orig.q_norm
        self.k_norm = orig.k_norm
        self.attn_drop = orig.attn_drop
        # Some Eva variants have a post-attention norm; use Identity if not.
        self.norm = getattr(orig, "norm", nn.Identity())
        self.proj = orig.proj
        self.proj_drop = orig.proj_drop

        # When the matmul kernels are in passthrough mode, also fall back
        # the softmax + scale kernels to bare-autograd ('passthrough')
        # so the unfolded backward bit-matches the stock EvaAttention
        # autograd backward — this is what the Phase 1 backward-parity
        # test asserts before we stack any LRP rule on top.
        scalar_rule = "passthrough" if matmul_rule == "passthrough" else "identity"
        softmax_rule = "passthrough" if matmul_rule == "passthrough" else "identity"

        # Atomic LRP-aware kernels.
        self.split = ChunkAlongLastDim(3)
        self.rope_q = RotaryEmbedding(
            self.num_prefix_tokens, rotate_half=rotate_half, detach_rope=rope_detach,
        )
        self.rope_k = RotaryEmbedding(
            self.num_prefix_tokens, rotate_half=rotate_half, detach_rope=rope_detach,
        )
        self.scale_q = ScaleByConstant(self.scale, rule=scalar_rule)
        self.qk_scores = BilinearMatmul(
            rule=matmul_rule, epsilon=epsilon, signed=signed_epsilon,
        )
        self.add_mask = AddBias()
        self.softmax = SoftmaxAlongLastDim(rule=softmax_rule)
        self.context = BilinearMatmul(
            rule=matmul_rule, epsilon=epsilon, signed=signed_epsilon,
        )
        self.reshape = ReshapeMergeHeads()
        # NB: we deliberately do NOT keep ``orig`` as an attribute here —
        # PyTorch's ``__setattr__`` would re-register it as a submodule
        # (even with a leading underscore, because it's an ``nn.Module``),
        # and the dual-registration bloats ``named_modules`` and confuses
        # any tooling that walks the module tree. The substitution
        # canonizer keeps its own reference for ``remove()``.

    # ─── forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape

        # Linear projection → (B, N, 3·D).
        qkv_flat = self.qkv(x)

        # Split last dim into Q/K/V chunks of (B, N, D) each. This is
        # equivalent to the upstream layout ``qkv.reshape(B, N, 3, H, hd)
        # .permute(2, 0, 3, 1, 4).unbind(0)`` because both isolate
        # ``qkv_flat[..., 0:D] / [D:2D] / [2D:3D]`` as Q / K / V before
        # the per-head reshape.
        q_flat, k_flat, v_flat = self.split(qkv_flat)

        # Per-chunk reshape to (B, num_heads, N, head_dim). This is a
        # pure view operation — identity in LRP terms — so we don't
        # bother with a Module wrapper.
        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = _to_heads(q_flat)
        k = _to_heads(k_flat)
        v = _to_heads(v_flat)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope_q(q, rope)
        k = self.rope_k(k, rope)
        # Match the upstream's ``.type_as(v)`` cast (a no-op when q/k/v
        # share dtype, the standard case).
        q = q.type_as(v)
        k = k.type_as(v)

        q = self.scale_q(q)

        # q @ k.transpose — k.transpose is a view, autograd-trivial.
        scores = self.qk_scores(q, k.transpose(-2, -1))

        # Resolve the additive mask (timm helper if available, else fall back).
        scores = self._resolve_and_add_mask(scores, attn_mask, is_causal, N)

        weights = self.softmax(scores)
        weights = self.attn_drop(weights)

        ctx = self.context(weights, v)
        out = self.reshape(ctx)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def _resolve_and_add_mask(self, scores, attn_mask, is_causal, N):
        # Mirrors the mask resolution in _eva_attention_forward. We pass
        # the resolved bias through ``AddBias`` so the LRP graph contains
        # an explicit add_mask submodule.
        try:
            from timm.models.eva import resolve_self_attn_mask
            attn_bias = resolve_self_attn_mask(N, scores, attn_mask, is_causal)
        except (ImportError, AttributeError):
            attn_bias = None
            if attn_mask is not None:
                attn_bias = attn_mask
            if is_causal:
                causal = torch.triu(
                    torch.full(
                        (N, N), float("-inf"), device=scores.device, dtype=scores.dtype,
                    ),
                    diagonal=1,
                )
                attn_bias = causal if attn_bias is None else attn_bias + causal
        return self.add_mask(scores, attn_bias)

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"num_prefix_tokens={self.num_prefix_tokens}, scale={self.scale}"
        )


# ─── 5. Substitution canonizer ──────────────────────────────────────────────


class EvaAttentionSubstitutionCanonizer(Canonizer):
    """Replace one or more ``EvaAttention`` instances with
    :class:`EvaAttentionUnfolded` wrappers (Phase 1).

    Lifecycle
    ---------
    * ``apply(root)`` walks ``root`` for ``EvaAttention`` modules whose
      block index is in ``block_indices`` (or all, if ``block_indices``
      is None). For each one it constructs an :class:`EvaAttentionUnfolded`
      around the original and re-binds the parent ``EvaBlock``'s
      ``.attn`` attribute to the unfolded version. The original module
      is stashed on the canonizer instance.
    * ``remove()`` re-binds the parent's ``.attn`` to the original
      module reference. Weight-sharing means no parameter state is
      lost; the original module is intact (its parameters were
      referenced, not moved, by the unfolded container).

    The Phase 1 default (``block_indices=(0,)``) substitutes only the
    first attention block — the rest of the model continues to use the
    existing canonizer-installed forward (or no canonizer at all). This
    lets us validate forward + backward parity on a single block before
    expanding to all blocks in Phase 2.

    Parameters
    ----------
    block_indices : tuple[int, ...] | None
        Indices of blocks to substitute. Default ``(0,)`` for Phase 1.
        Pass ``None`` to substitute all attention blocks; pass an
        explicit tuple to target a subset.
    matmul_rule, epsilon, signed_epsilon, rope_detach
        Forwarded to :class:`EvaAttentionUnfolded`.

    Notes
    -----
    Round-trip safety (plan §P6): we verify reversibility in
    ``tests/test_attention_unfolded.py`` — apply → forward → remove →
    re-apply with a different composite → forward — and assert tensor
    equality.

    The unfolded container references the original module's parameter
    submodules (``qkv``, ``proj``, etc.) by attribute, NOT by deep
    copy, so:

    * Optimizer state attached to the original parameters still
      works after substitution.
    * Restoring on ``remove()`` re-points ``.attn`` to the original
      instance, which still owns the parameters.
    * Re-applying picks up any in-place parameter updates between
      apply / remove cycles.
    """

    def __init__(
        self,
        *,
        block_indices: Optional[Sequence[int]] = (0,),
        matmul_rule: str = "matmul_factor_2",
        epsilon: float = 1e-6,
        signed_epsilon: bool = False,
        rope_detach: bool = False,
    ):
        self.block_indices = (
            None if block_indices is None else tuple(int(i) for i in block_indices)
        )
        self.matmul_rule = matmul_rule
        self.epsilon = epsilon
        self.signed_epsilon = signed_epsilon
        self.rope_detach = rope_detach

        # State filled by ``register``.
        self.parent: Optional[nn.Module] = None
        self.attr_name: Optional[str] = None
        self.original_module: Optional[nn.Module] = None
        self.unfolded_module: Optional[EvaAttentionUnfolded] = None

    # ─── Canonizer interface ────────────────────────────────────────────────

    def apply(self, root_module: nn.Module) -> List["EvaAttentionSubstitutionCanonizer"]:
        try:
            from timm.models.eva import EvaAttention
        except ImportError:
            return []

        # Find candidate (parent, attr_name, attn_module, block_index) tuples.
        instances: List[EvaAttentionSubstitutionCanonizer] = []
        for parent_name, parent in root_module.named_modules():
            for attr_name, child in parent.named_children():
                if not isinstance(child, EvaAttention):
                    continue
                if self.block_indices is not None:
                    block_idx = _extract_block_index(parent_name)
                    if block_idx is None or block_idx not in self.block_indices:
                        continue
                inst = self.copy()
                inst.register(parent, attr_name, child)
                instances.append(inst)
        return instances

    def register(
        self,
        parent: nn.Module,
        attr_name: str,
        original: nn.Module,
    ) -> None:
        self.parent = parent
        self.attr_name = attr_name
        self.original_module = original
        unfolded = EvaAttentionUnfolded(
            original,
            matmul_rule=self.matmul_rule,
            epsilon=self.epsilon,
            signed_epsilon=self.signed_epsilon,
            rope_detach=self.rope_detach,
        )
        # Re-bind via setattr so the parent's module dict is updated AND
        # nn.Module's __setattr__ moves the params/buffers to the right
        # place. Because the unfolded module references the original's
        # submodules, those submodules end up nested under the unfolded
        # container — which is exactly what we want (same parameter
        # names from the parent's POV: blocks.i.attn.qkv.weight etc.).
        setattr(parent, attr_name, unfolded)
        self.unfolded_module = unfolded

    def remove(self) -> None:
        if self.parent is None or self.attr_name is None or self.original_module is None:
            return
        setattr(self.parent, self.attr_name, self.original_module)
        # Don't null out the references — re-apply may want to use them.
        self.unfolded_module = None

    def copy(self) -> "EvaAttentionSubstitutionCanonizer":
        return type(self)(
            block_indices=self.block_indices,
            matmul_rule=self.matmul_rule,
            epsilon=self.epsilon,
            signed_epsilon=self.signed_epsilon,
            rope_detach=self.rope_detach,
        )


def _extract_block_index(parent_name: str) -> Optional[int]:
    """Return ``i`` if ``parent_name`` ends in ``...blocks.i`` else None.

    Matches ``"blocks.0"`` (top-level blocks) and any
    ``"<sub>.blocks.0"`` nested-stack variant.
    """
    parts = parent_name.split(".")
    for j in range(len(parts) - 1):
        if parts[j] == "blocks":
            try:
                return int(parts[j + 1])
            except (ValueError, IndexError):
                continue
    return None


__all__ = [
    # autograd Function kernels new to this module
    "_SoftmaxIdentityRuleFn",
    "_ScaleIdentityRuleFn",
    # 2-input kernels
    "BilinearMatmul",
    "AddBias",
    "ResidualAdd",
    # single-input kernels
    "SoftmaxAlongLastDim",
    "RotaryEmbedding",
    "ScaleByConstant",
    "ChunkAlongLastDim",
    "ReshapeMergeHeads",
    "LayerScaleMul",
    # container + canonizer
    "EvaAttentionUnfolded",
    "EvaAttentionSubstitutionCanonizer",
]
