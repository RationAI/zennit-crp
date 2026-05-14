"""Atomic ``nn.Module`` kernels for AttnLRP-aware attention — VANILLA forwards.

Every module here is a plain PyTorch op with autograd's standard backward.
LRP behaviour (custom backward = relevance flow, not Jacobian-vector
product) is injected EXCLUSIVELY at attribution time by the per-module
canonizers in this file (e.g. :class:`BilinearMatmulAlphaBetaCanonizer`),
which rebind the module's ``forward`` to route through an autograd
``Function`` whose backward implements the LRP rule. The canonizers are
applied by an outer composite (``composite.context(model) → enter`` →
canonizers ``apply`` → forward+backward inside the context fires LRP
rules → ``exit`` → canonizers ``remove`` → modules return to vanilla).

This separation matters because these modules are also used inside
trainable classification heads (``AttentiveHead``, ``BlockHead``). At
training time no composite is active, so the modules behave like normal
PyTorch ops and the optimizer receives correct chain-rule gradients.

Module catalogue (all vanilla forwards)
---------------------------------------

Bilinear / multi-input modules:

* :class:`BilinearMatmul` — ``a @ b``.
* :class:`AddBias` — ``x + bias`` (bias may be None → identity).
* :class:`ResidualAdd` — ``x + branch``.

Single-input modules:

* :class:`SoftmaxAlongLastDim` — ``F.softmax(x, dim=-1)``.
* :class:`RotaryEmbedding` — RoPE; optional prefix-tokens skip and
  optional ``rope.detach()`` (structural choice — RoPE has no learnable
  parameters, not an LRP rule).
* :class:`ScaleByConstant` — ``x * value``.
* :class:`ChunkAlongLastDim` — ``x.chunk(n, dim=-1)``.
* :class:`ReshapeMergeHeads` — ``x.transpose(1, 2).reshape(B, N, out_dim)``.
* :class:`LayerScaleMul` — ``γ * x`` (γ is a learnable per-channel
  parameter from the parent ``EvaBlock``).

Container modules:

* :class:`EvaAttentionUnfolded` — composes the vanilla atomics into a
  forward-parity replacement for ``timm.models.eva.EvaAttention``.
* :class:`TimmAttentionUnfolded` — same idea for the standard
  ``timm.models.vision_transformer.Attention`` (no RoPE, no LayerScale,
  ``num_prefix_tokens=1``).

Substitution canonizers:

* :class:`EvaAttentionSubstitutionCanonizer` — swaps ``EvaAttention`` for
  :class:`EvaAttentionUnfolded` on ``apply()``, restores on ``remove()``.
* :class:`TimmAttentionSubstitutionCanonizer` — same for standard timm
  ``Attention``. Both canonizers coexist cleanly; each one's
  ``isinstance`` filter skips the other's target class.

LRP-rule canonizers (one per rule, registered at attribution time):

* :class:`BilinearMatmulAlphaBetaCanonizer` — Bach 2015 generalised to
  bilinear (RESEARCH_NOTES.md Entry 6) on :class:`BilinearMatmul`.
* :class:`SoftmaxIdentityCanonizer` — AttnLRP Eq. 9 identity rule on
  :class:`SoftmaxAlongLastDim`.
* :class:`ScaleByConstantIdentityCanonizer` — graph-constant identity
  rule on :class:`ScaleByConstant`.
* :class:`ResidualRatioCanonizer` — Otsuki ratio split on
  :class:`ResidualAdd`.
* :class:`LayerScaleUniformCanonizer` — uniform allocation rule
  (AttnLRP Eq. 7) on :class:`LayerScaleMul`.

The autograd ``Function`` kernels (``_AlphaBetaMatmulFn``,
``_SoftmaxIdentityRuleFn``, ``_ScaleIdentityRuleFn``) live near the
canonizers that use them. Kernels reused from
:mod:`crp.transformer_patches` (``_DivideGradientFn``,
``_ResidualRatioFn``) are imported on demand by the canonizers.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from timm.layers import apply_rot_embed_cat
from timm.models.eva import EvaAttention
from timm.models.vision_transformer import Attention as TimmAttention

from zennit.canonizers import AttributeCanonizer, Canonizer


# ─── 1. Vanilla atomic modules ──────────────────────────────────────────────


class BilinearMatmul(nn.Module):
    """``y = a @ b``. Vanilla bilinear matmul; autograd's standard backward.

    Used inside :class:`EvaAttentionUnfolded` and :class:`TimmAttentionUnfolded`
    for ``q @ kᵀ`` and ``weights @ v``, and inside classification heads
    that want these matmuls exposed as named submodules. To enable the
    AttnLRP AlphaBeta rule on backward at attribution time, register
    :class:`BilinearMatmulAlphaBetaCanonizer` via the composite.
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b


class AddBias(nn.Module):
    """``y = x + bias`` where ``bias`` is a leaf constant or ``None``.

    When ``bias is None`` forward is identity (no add). Vanilla; autograd
    handles the ``+`` correctly under any LRP composite's layer_map for
    standard adds. (No dedicated rule canonizer because the typical use
    site is the resolved attention mask, which is a constant — autograd
    routes ``grad_x = grad_y`` and discards the unused ``grad_bias``.)
    """

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if bias is None:
            return x
        return x + bias


class ResidualAdd(nn.Module):
    """``y = x + branch``. Vanilla addition.

    For the LRP residual-ratio rule (Otsuki split: distribute ``R_y``
    proportionally to ``|x|`` and ``|branch|``), register
    :class:`ResidualRatioCanonizer` via the composite at attribution time.
    """

    def forward(self, x: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return x + branch


class SoftmaxAlongLastDim(nn.Module):
    """``y = F.softmax(x, dim=-1)``. Vanilla.

    For the AttnLRP identity rule (``R_in = R_out``, AttnLRP Eq. 9),
    register :class:`SoftmaxIdentityCanonizer` via the composite at
    attribution time.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)


class RotaryEmbedding(nn.Module):
    """Apply RoPE to a Q or K tensor with optional prefix-tokens skip.

    Wraps ``timm.layers.apply_rot_embed_cat`` and handles the
    ``num_prefix_tokens`` skip used by DINOv3 / EVA-style models (cls +
    register tokens at the front of the sequence are NOT rotated).

    Parameters
    ----------
    num_prefix_tokens : int
        Number of leading positions left un-rotated.
    rotate_half : bool
        Layout flag passed to ``apply_rot_embed_cat``. Mirrors
        ``EvaAttention.rotate_half``.
    detach_rope : bool
        When True, ``rope.detach()`` before the rotary op. RoPE has no
        learnable parameters (Su et al. 2021, arXiv:2104.09864), so
        cos/sin can be treated as graph constants and gradients route
        purely through the rotated tensor. This is a structural choice
        about whether RoPE owns parameters in the autograd graph — it
        is NOT an LRP rule and is safe at training time. Default False
        for forward parity with stock ``EvaAttention``.

    Forward signature: ``(q, rope)``. ``rope`` may be ``None`` (identity).
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
    """``y = x * value`` where ``value`` is a graph constant.

    For the AttnLRP graph-constant identity rule on backward (constant
    absorbs no relevance), register :class:`ScaleByConstantIdentityCanonizer`
    via the composite at attribution time.
    """

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.value

    def extra_repr(self) -> str:
        return f"value={self.value}"


class ChunkAlongLastDim(nn.Module):
    """Split a tensor into ``n`` equal chunks along the last dim.

    Backward is ``torch.cat`` along the same dim — identity in LRP terms
    (each chunk's relevance returns to its original slice). No canonizer
    needed; bare autograd routes relevance correctly.
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

    Forward: ``x.transpose(1, 2).reshape(B, N, out_dim)``. Identity in
    LRP terms — pure reshape preserves ``sum(R)``; no canonizer needed.
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
    """``y = γ * x`` (CaiT-style LayerScale; Touvron et al. 2021).

    γ is a learnable per-channel parameter. For the AttnLRP uniform rule
    on backward (Eq. 7 — γ absorbs half the relevance, branch gets the
    other half), register :class:`LayerScaleUniformCanonizer` via the
    composite at attribution time.

    Parameters
    ----------
    gamma : nn.Parameter
        The LayerScale parameter from the parent ``EvaBlock``. Shared by
        reference; not copied.
    """

    def __init__(self, gamma: nn.Parameter):
        super().__init__()
        # NB: ``gamma`` is a Parameter on the parent module; we reference
        # it without re-registering, so weight loading still flows through
        # the parent.
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


# ─── 2. Container: EvaAttentionUnfolded (vanilla forward) ───────────────────


class EvaAttentionUnfolded(nn.Module):
    """Unfolded ``timm.models.eva.EvaAttention`` analogue — vanilla forward.

    Constructor takes a *real* ``EvaAttention`` instance and stores
    references to its parameters / sub-modules. The container does NOT
    copy weights — the original module's parameters live on, accessible
    via ``self.qkv``, ``self.proj`` etc. So when this container is
    swapped in by :class:`EvaAttentionSubstitutionCanonizer`, weight
    loading from a checkpoint Just Works (the checkpoint targets
    ``blocks.{i}.attn.qkv.weight`` and we still have an ``attn.qkv``
    submodule).

    Forward signature matches ``EvaAttention.forward`` exactly:
    ``(self, x, rope=None, attn_mask=None, is_causal=False)``. All atomic
    submodules are vanilla (see module docstrings); LRP behaviour is
    opt-in via composite-time canonizers.

    Parameters
    ----------
    orig : timm.models.eva.EvaAttention
        Source module to wrap. Its parameters become this container's
        parameters by reference.
    rope_detach : bool
        Forwarded to :class:`RotaryEmbedding`. Structural choice (RoPE
        has no parameters), not an LRP rule. Default False.
    """

    def __init__(
        self,
        orig,  # timm.models.eva.EvaAttention
        *,
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
        # a different qkv path; not handled here.
        if getattr(orig, "q_bias", None) is not None:
            raise NotImplementedError(
                "EvaAttentionUnfolded does not support per-Q/K/V bias "
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

        # Atomic vanilla submodules. LRP rules are layered in by the
        # per-module canonizers at composite-context-entry time.
        self.split = ChunkAlongLastDim(3)
        self.rope_q = RotaryEmbedding(
            self.num_prefix_tokens, rotate_half=rotate_half, detach_rope=rope_detach,
        )
        self.rope_k = RotaryEmbedding(
            self.num_prefix_tokens, rotate_half=rotate_half, detach_rope=rope_detach,
        )
        self.scale_q = ScaleByConstant(self.scale)
        self.qk_scores = BilinearMatmul()
        self.add_mask = AddBias()
        self.softmax = SoftmaxAlongLastDim()
        self.context = BilinearMatmul()
        self.reshape = ReshapeMergeHeads()
        # Identity hook points so concept-conditioning can target Q, K, V
        # individually (post per-head reshape, before they enter the
        # bilinear). q_norm / k_norm already exist as nn.Modules (LayerNorm
        # or Identity) so they're hookable directly; V has no analog
        # transformation, so we add an explicit Identity for symmetry.
        self.v_id = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv_flat = self.qkv(x)
        q_flat, k_flat, v_flat = self.split(qkv_flat)

        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = _to_heads(q_flat)
        k = _to_heads(k_flat)
        v = _to_heads(v_flat)
        v = self.v_id(v)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope_q(q, rope)
        k = self.rope_k(k, rope)
        q = q.type_as(v)
        k = k.type_as(v)

        q = self.scale_q(q)
        scores = self.qk_scores(q, k.transpose(-2, -1))
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


# ─── 3. Substitution canonizer (no LRP knobs) ───────────────────────────────


class EvaAttentionSubstitutionCanonizer(Canonizer):
    """Replace ``EvaAttention`` instances with :class:`EvaAttentionUnfolded`.

    Lifecycle
    ---------
    * ``apply(root)`` walks ``root`` for ``EvaAttention`` modules whose
      block index is in ``block_indices`` (or all, if ``block_indices``
      is None). For each one it constructs a vanilla
      :class:`EvaAttentionUnfolded` around the original and re-binds the
      parent ``EvaBlock``'s ``.attn`` attribute to the unfolded version.
    * ``remove()`` re-binds the parent's ``.attn`` to the original
      module reference. Weight-sharing means no parameter state is lost.

    This canonizer makes NO LRP-rule decisions — it just exposes the
    attention as named submodules so other canonizers can register
    forward-rebinds (e.g. :class:`BilinearMatmulAlphaBetaCanonizer`) on
    the new ``BilinearMatmul`` / ``SoftmaxAlongLastDim`` / etc. instances.

    Parameters
    ----------
    block_indices : tuple[int, ...] | None
        Indices of blocks to substitute. ``None`` (default) means
        substitute all attention blocks.
    rope_detach : bool
        Forwarded to :class:`EvaAttentionUnfolded` → :class:`RotaryEmbedding`.
        Structural (whether RoPE owns parameters in the autograd graph),
        not an LRP rule. Default False.
    """

    def __init__(
        self,
        *,
        block_indices: Optional[Sequence[int]] = None,
        rope_detach: bool = False,
    ):
        self.block_indices = (
            None if block_indices is None else tuple(int(i) for i in block_indices)
        )
        self.rope_detach = rope_detach

        # State filled by ``register``.
        self.parent: Optional[nn.Module] = None
        self.attr_name: Optional[str] = None
        self.original_module: Optional[nn.Module] = None
        self.unfolded_module: Optional[EvaAttentionUnfolded] = None

    def apply(self, root_module: nn.Module) -> List["EvaAttentionSubstitutionCanonizer"]:
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

    def register(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        parent: nn.Module,
        attr_name: str,
        original: nn.Module,
    ) -> None:
        # Signature intentionally differs from ``Canonizer.register(self)``;
        # zennit's own canonizers (e.g. ``MergeBatchNorm.register(linears,
        # batch_norm)``) do the same — ABC enforces method NAME, not
        # signature, and zennit's lifecycle calls ``apply()`` which then
        # dispatches to this overload.
        self.parent = parent
        self.attr_name = attr_name
        self.original_module = original
        unfolded = EvaAttentionUnfolded(original, rope_detach=self.rope_detach)
        setattr(parent, attr_name, unfolded)
        self.unfolded_module = unfolded

    def remove(self) -> None:
        if self.parent is None or self.attr_name is None or self.original_module is None:
            return
        setattr(self.parent, self.attr_name, self.original_module)
        self.unfolded_module = None

    def copy(self) -> "EvaAttentionSubstitutionCanonizer":
        return type(self)(
            block_indices=self.block_indices,
            rope_detach=self.rope_detach,
        )


# ─── 3b. Timm-standard unfolded attention + substitution canonizer ──────────


class TimmAttentionUnfolded(nn.Module):
    """Unfolded ``timm.models.vision_transformer.Attention`` analogue.

    Same role as :class:`EvaAttentionUnfolded` but for the standard timm
    ViT attention (no RoPE, no LayerScale, ``num_prefix_tokens=1``: cls
    only). Constructor takes a real ``Attention`` instance, references
    its parameter-bearing submodules (``qkv``, ``q_norm``, ``k_norm``,
    ``proj``, ``attn_drop``, ``proj_drop``, optional ``norm``) by
    attribute, and exposes hookable atomic ops so the AttnLRP composite
    can canonize them and the concept classes can target ``q_id`` /
    ``k_id`` / ``v_id`` / ``context`` / ``proj_drop``.

    Forward signature: ``(x, attn_mask=None, is_causal=False)`` — matches
    the stock timm `Attention.forward` exactly. Numerical output is
    bit-identical to stock when the source instance has
    ``fused_attn=False`` (the explicit-math path); when ``fused_attn=True``
    on the source, stock uses ``F.scaled_dot_product_attention`` which
    can differ at fp32 noise level, but the substituted unfolded
    instance takes the explicit path either way.

    Parameters
    ----------
    orig : timm.models.vision_transformer.Attention
        Source module to wrap.
    num_prefix_tokens : int
        Number of leading (non-spatial) tokens in the model's sequence —
        ``1`` for the standard ``cls``-only setup, more if the model was
        built with ``reg_tokens > 0``. Stock timm ``Attention`` does not
        carry this attribute (only the top-level ``VisionTransformer``
        does), so the substitution canonizer is responsible for passing
        it in at construction time. Default ``1`` matches the most common
        case but the canonizer always overrides it.
    """

    def __init__(self, orig, *, num_prefix_tokens: int = 1) -> None:
        super().__init__()
        if not (
            hasattr(orig, "qkv")
            and isinstance(orig.qkv, nn.Linear)
            and hasattr(orig, "num_heads")
            and hasattr(orig, "proj")
        ):
            raise TypeError(
                "TimmAttentionUnfolded expects a timm vision_transformer.Attention-like instance"
            )

        # Cache shape constants from the original module.
        self.num_heads = int(orig.num_heads)
        # Stock timm sets head_dim. Older variants compute it from dim/num_heads.
        if hasattr(orig, "head_dim"):
            self.head_dim = int(orig.head_dim)
        else:
            self.head_dim = int(orig.qkv.weight.shape[0] // 3 // self.num_heads)
        self.num_prefix_tokens = int(num_prefix_tokens)
        # Scale: stock timm sets self.scale = head_dim ** -0.5
        self.scale = float(getattr(orig, "scale", self.head_dim ** -0.5))

        # Reference (do not copy) the parameter-bearing / stateful submodules.
        self.qkv = orig.qkv
        # q_norm / k_norm may be nn.LayerNorm or nn.Identity depending on
        # the qk_norm flag at construction time.
        self.q_norm = getattr(orig, "q_norm", nn.Identity())
        self.k_norm = getattr(orig, "k_norm", nn.Identity())
        self.attn_drop = orig.attn_drop
        # Newer timm variants add a post-attention norm; older variants don't.
        self.norm = getattr(orig, "norm", nn.Identity())
        self.proj = orig.proj
        self.proj_drop = orig.proj_drop

        # Atomic vanilla submodules. LRP rules are layered in by the
        # per-module canonizers at composite-context-entry time.
        self.split = ChunkAlongLastDim(3)
        self.scale_q = ScaleByConstant(self.scale)
        self.qk_scores = BilinearMatmul()
        self.add_mask = AddBias()
        self.softmax = SoftmaxAlongLastDim()
        self.context = BilinearMatmul()
        self.reshape = ReshapeMergeHeads()
        # Identity hook points for Q / K / V concepts. q_id is placed
        # AFTER q_norm so the recorded tensor reflects all per-tensor
        # transformations the concept "should see"; symmetric for k_id.
        # v_id has no norm in stock timm, so it sits right after the
        # per-head reshape (same as in EvaAttentionUnfolded).
        self.q_id = nn.Identity()
        self.k_id = nn.Identity()
        self.v_id = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv_flat = self.qkv(x)
        q_flat, k_flat, v_flat = self.split(qkv_flat)

        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = _to_heads(q_flat)
        k = _to_heads(k_flat)
        v = _to_heads(v_flat)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.q_id(q)
        k = self.k_id(k)
        v = self.v_id(v)

        q = self.scale_q(q)
        scores = self.qk_scores(q, k.transpose(-2, -1))
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
        # Standard timm Attention's explicit-math path does
        #   if attn_mask is not None: attn = attn + attn_mask
        # and ignores is_causal on this path (it only applies to the
        # F.scaled_dot_product_attention dispatch). We still honour
        # is_causal=True by building a triangular mask for consistency
        # with the Eva unfolded variant.
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


class TimmAttentionSubstitutionCanonizer(Canonizer):
    """Replace standard timm ``Attention`` instances with :class:`TimmAttentionUnfolded`.

    Mirror of :class:`EvaAttentionSubstitutionCanonizer` for the
    standard timm `vision_transformer.Attention` class. Both canonizers
    are typically bundled into the same composite — each one's
    ``isinstance`` filter skips the other's target so there's no
    coupling.

    Parameters
    ----------
    block_indices : tuple[int, ...] | None
        Indices of blocks to substitute. ``None`` (default) means
        substitute all attention blocks.
    """

    def __init__(
        self,
        *,
        block_indices: Optional[Sequence[int]] = None,
    ):
        self.block_indices = (
            None if block_indices is None else tuple(int(i) for i in block_indices)
        )

        # State filled by ``register``.
        self.parent: Optional[nn.Module] = None
        self.attr_name: Optional[str] = None
        self.original_module: Optional[nn.Module] = None
        self.unfolded_module: Optional[TimmAttentionUnfolded] = None

    def apply(self, root_module: nn.Module) -> List["TimmAttentionSubstitutionCanonizer"]:
        # Stock timm `Attention` does not carry `num_prefix_tokens`. The
        # value lives on the top-level `VisionTransformer` instance and the
        # canonizer is the right place to read it once and mediate it down
        # to each unfolded replacement. ``getattr`` with a fallback to 1
        # handles bare attentions used in tests.
        num_prefix_tokens = int(getattr(root_module, "num_prefix_tokens", 1))

        instances: List[TimmAttentionSubstitutionCanonizer] = []
        for parent_name, parent in root_module.named_modules():
            for attr_name, child in parent.named_children():
                if not isinstance(child, TimmAttention):
                    continue
                if self.block_indices is not None:
                    block_idx = _extract_block_index(parent_name)
                    if block_idx is None or block_idx not in self.block_indices:
                        continue
                inst = self.copy()
                inst.register(parent, attr_name, child, num_prefix_tokens=num_prefix_tokens)
                instances.append(inst)
        return instances

    def register(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        parent: nn.Module,
        attr_name: str,
        original: nn.Module,
        *,
        num_prefix_tokens: int = 1,
    ) -> None:
        # See note on ``EvaAttentionSubstitutionCanonizer.register`` —
        # zennit's lifecycle dispatches through ``apply()``; the ABC
        # only requires the method name.
        self.parent = parent
        self.attr_name = attr_name
        self.original_module = original
        unfolded = TimmAttentionUnfolded(original, num_prefix_tokens=num_prefix_tokens)
        setattr(parent, attr_name, unfolded)
        self.unfolded_module = unfolded

    def remove(self) -> None:
        if self.parent is None or self.attr_name is None or self.original_module is None:
            return
        setattr(self.parent, self.attr_name, self.original_module)
        self.unfolded_module = None

    def copy(self) -> "TimmAttentionSubstitutionCanonizer":
        return type(self)(block_indices=self.block_indices)


def _extract_block_index(parent_name: str) -> Optional[int]:
    """Return ``i`` if ``parent_name`` ends in ``...blocks.i`` else None."""
    parts = parent_name.split(".")
    for j in range(len(parts) - 1):
        if parts[j] == "blocks":
            try:
                return int(parts[j + 1])
            except (ValueError, IndexError):
                continue
    return None


# ─── 4. Autograd Function kernels for the LRP rules ─────────────────────────
# These are used ONLY by the canonizers below — never by the modules' own
# forwards. They live here (not in transformer_patches) because their sole
# consumers are the per-module canonizers right below them.


class _SoftmaxIdentityRuleFn(Function):
    """Softmax with the AttnLRP identity rule (R_in = R_out).

    AttnLRP §3.2.2 / Eq. 9 (Achtibat et al. 2024, arXiv:2402.05602): for
    normalisations like softmax, the LRP-0 derivation reduces to identity
    because the softmax outputs are 1×1-linear-equivalent in y/x. The
    paper's recipe is to *pass relevance straight through softmax*: the
    upstream relevance ``R_y`` is also the relevance on the pre-softmax
    scores ``R_x``.

    Forward: standard ``F.softmax(x, dim=-1)``.
    Backward: ``grad_out`` (i.e. ``R_in = R_out``).
    """

    @staticmethod
    def forward(ctx, x):
        return F.softmax(x, dim=-1)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_out,) = grad_outputs
        return grad_out


class _ScaleIdentityRuleFn(Function):
    """``y = x * scalar`` with R_in = R_out (constant absorbs no relevance).

    AttnLRP treats multiplication by a graph-constant operand as identity
    — the constant has no relevance to receive, so R_x = R_y. Bare
    autograd would multiply ``grad_out`` by the scalar; this Function
    short-circuits that.
    """

    @staticmethod
    def forward(ctx, x, scalar: float):
        return x * scalar

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_out,) = grad_outputs
        return grad_out, None


class _AlphaBetaMatmulFn(Function):
    """AlphaBeta-on-bilinear LRP rule (Bach 2015 generalised to bilinear).

    See ``RESEARCH_NOTES.md`` Entry 6. Replaces the standard AttnLRP
    Prop. 3.3 ``2y+ε`` denominator with separate ``Y^+`` and ``|Y^-|``
    denominators built from sign-decomposed operand contributions,
    structurally avoiding the LRP-0 Case-B cancellation amplification.

    Forward: standard ``Y = A @ B``.

    Backward (per element of A):

    .. math::

        R_A = \\tfrac{1}{2} \\big\\{ A^+ \\odot [\\alpha (F B^{+T})
              + \\beta (G B^{-T})] + A^- \\odot [\\alpha (F B^{-T})
              + \\beta (G B^{+T})] \\big\\}

    where ``A^± = max/min(A, 0)``, ``B^± = max/min(B, 0)``,
    ``Y^+ = A^+ B^+ + A^- B^-``, ``Y^- = A^+ B^- + A^- B^+``,
    ``F = R_Y / (Y^+ + ε)``, ``G = R_Y / (Y^- − ε)``. Symmetric for
    ``R_B``.

    Conservation: ``Σ R_A + Σ R_B = (α+β)·Σ R_Y``; with ``α + β = 1``
    we get exact conservation modulo ε.
    """

    @staticmethod
    def forward(ctx, a, b, alpha=1.0, beta=0.0, epsilon=1e-6):
        out = a @ b
        a_pos = a.clamp(min=0)
        a_neg = a.clamp(max=0)
        b_pos = b.clamp(min=0)
        b_neg = b.clamp(max=0)
        with torch.no_grad():
            y_pos = a_pos @ b_pos + a_neg @ b_neg
            y_neg = a_pos @ b_neg + a_neg @ b_pos
        ctx.save_for_backward(a_pos, a_neg, b_pos, b_neg, y_pos, y_neg)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.epsilon = epsilon
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_out,) = grad_outputs
        a_pos, a_neg, b_pos, b_neg, y_pos, y_neg = ctx.saved_tensors
        eps = ctx.epsilon
        alpha = ctx.alpha
        beta = ctx.beta
        f = grad_out / (y_pos + eps)
        g = grad_out / (y_neg - eps)
        bpT = b_pos.transpose(-1, -2)
        bnT = b_neg.transpose(-1, -2)
        grad_a = 0.5 * (
            a_pos * (alpha * (f @ bpT) + beta * (g @ bnT))
            + a_neg * (alpha * (f @ bnT) + beta * (g @ bpT))
        )
        apT = a_pos.transpose(-1, -2)
        anT = a_neg.transpose(-1, -2)
        grad_b = 0.5 * (
            b_pos * (alpha * (apT @ f) + beta * (anT @ g))
            + b_neg * (alpha * (anT @ f) + beta * (apT @ g))
        )
        return grad_a, grad_b, None, None, None


# ─── 5. LRP-rule canonizers (one per rule, vanilla ↔ rule-aware swap) ───────
#
# Each canonizer rebinds the forward of a specific module class with a
# wrapper that routes through an autograd Function whose backward
# implements the LRP rule. On ``apply()`` the wrapper is bound; on
# ``remove()`` the original ``forward`` is restored. Forward output is
# numerically identical to vanilla in every case (Functions do
# ``a @ b`` / ``F.softmax`` / ``x * scalar`` etc. — only backward differs).


class BilinearMatmulAlphaBetaCanonizer(AttributeCanonizer):
    """Apply AlphaBeta-on-bilinear rule to :class:`BilinearMatmul`.

    See :class:`_AlphaBetaMatmulFn` for the derivation. Default
    ``α=0.5, β=0.5`` matches RESEARCH_NOTES.md Entry 6's working recipe.

    Parameters
    ----------
    alpha, beta : float
        Bach 2015 mixture coefficients. ``α + β = 1`` for exact
        conservation. Common choices:
        ``α=0.5, β=0.5`` (balanced; tightest magnitude control),
        ``α=1, β=0`` (z+, positive only),
        ``α=2, β=-1`` (Bach's classical "alpha2beta1").
    epsilon : float
        Stabiliser. Default ``1e-6``.
    """

    def __init__(self, *, alpha: float = 0.5, beta: float = 0.5, epsilon: float = 1e-6):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, BilinearMatmul):
            return None
        alpha = self.alpha
        beta = self.beta
        eps = self.epsilon

        def patched(self, a, b):
            return _AlphaBetaMatmulFn.apply(a, b, alpha, beta, eps)

        return _bind_forward(module, patched)

    def copy(self):
        return type(self)(alpha=self.alpha, beta=self.beta, epsilon=self.epsilon)


class SoftmaxIdentityCanonizer(AttributeCanonizer):
    """Apply AttnLRP identity rule (R_in = R_out) to :class:`SoftmaxAlongLastDim`.

    AttnLRP Eq. 9. Forward output is bit-identical to vanilla
    ``F.softmax``; only backward differs (relevance passes through).
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, SoftmaxAlongLastDim):
            return None

        def patched(self, x):
            return _SoftmaxIdentityRuleFn.apply(x)

        return _bind_forward(module, patched)

    def copy(self):
        return type(self)()


class ScaleByConstantIdentityCanonizer(AttributeCanonizer):
    """Apply graph-constant identity rule to :class:`ScaleByConstant`.

    Constant absorbs no relevance, so ``R_in = R_out`` (the scalar gets
    ``None``). Forward output is bit-identical to vanilla ``x * value``.
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, ScaleByConstant):
            return None

        def patched(self, x):
            return _ScaleIdentityRuleFn.apply(x, self.value)

        return _bind_forward(module, patched)

    def copy(self):
        return type(self)()


class ResidualRatioCanonizer(AttributeCanonizer):
    """Apply Otsuki ratio split (``R ∝ |x| / |branch|``) to :class:`ResidualAdd`.

    Conservation:
    ``R_x + R_branch = R_y · (|x| + |branch|) / (|x| + |branch| + ε) ≈ R_y``.

    Parameters
    ----------
    epsilon : float
        Stabiliser. Default ``1e-6``.
    """

    def __init__(self, *, epsilon: float = 1e-6):
        self.epsilon = epsilon
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, ResidualAdd):
            return None
        from crp.transformer_patches import _ResidualRatioFn
        eps = self.epsilon

        def patched(self, x, branch):
            return _ResidualRatioFn.apply(x, branch, eps)

        return _bind_forward(module, patched)

    def copy(self):
        return type(self)(epsilon=self.epsilon)


class LayerScaleUniformCanonizer(AttributeCanonizer):
    """Apply uniform allocation rule (AttnLRP Eq. 7) to :class:`LayerScaleMul`.

    Backward divides the upstream gradient by 2: γ absorbs half the
    relevance, the branch gets the other half. Important for LayerScale's
    small initial γ (1e-4..1e-5) where bare autograd would near-zero the
    branch's relevance.
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, LayerScaleMul):
            return None
        from crp.transformer_patches import _DivideGradientFn

        def patched(self, x):
            out = self.gamma * x
            return _DivideGradientFn.apply(out, 2)

        return _bind_forward(module, patched)

    def copy(self):
        return type(self)()


# ─── 6. Internal helpers ────────────────────────────────────────────────────


def _bind_forward(module: nn.Module, fn: Callable, attr: str = "forward") -> dict:
    """Bind ``fn`` as ``attr`` on ``module``'s class — return dict for
    AttributeCanonizer."""
    return {attr: fn.__get__(module, type(module))}


__all__ = [
    # Vanilla atomic modules
    "BilinearMatmul",
    "AddBias",
    "ResidualAdd",
    "SoftmaxAlongLastDim",
    "RotaryEmbedding",
    "ScaleByConstant",
    "ChunkAlongLastDim",
    "ReshapeMergeHeads",
    "LayerScaleMul",
    # Containers + substitution canonizers
    "EvaAttentionUnfolded",
    "EvaAttentionSubstitutionCanonizer",
    "TimmAttentionUnfolded",
    "TimmAttentionSubstitutionCanonizer",
    # Per-rule canonizers
    "BilinearMatmulAlphaBetaCanonizer",
    "SoftmaxIdentityCanonizer",
    "ScaleByConstantIdentityCanonizer",
    "ResidualRatioCanonizer",
    "LayerScaleUniformCanonizer",
    # Autograd Function kernels (used internally by canonizers)
    "_SoftmaxIdentityRuleFn",
    "_ScaleIdentityRuleFn",
    "_AlphaBetaMatmulFn",
]
