"""Atomic ``nn.Module`` kernels for AttnLRP-aware attention — VANILLA forwards.

Every module here is a plain PyTorch op with autograd's standard backward.
LRP behaviour (custom backward = relevance flow, not Jacobian-vector
product) is injected EXCLUSIVELY at attribution time by the zennit ``Hook``
rules in :mod:`zennit_ext.attnlrp_rules` (``AlphaBetaMatmul`` on
:class:`BilinearMatmul`, a selectable residual rule on :class:`ResidualAdd`
(and its :class:`PosEmbedAdd` alias), ``Uniform`` on :class:`LayerScaleMul`,
``Pass`` on :class:`SoftmaxAlongLastDim` / :class:`ScaleByConstant`), assigned
by module type through the composite's ``layer_map``. zennit hooks fire in the single
backward pass and detach on ``composite.context()`` exit, so these modules
stay vanilla outside attribution.

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
* :class:`LRPInspectionLayer` (base) + role-specific subclasses
  :class:`QInspectionLayer` / :class:`KInspectionLayer` /
  :class:`VInspectionLayer` — ``nn.Identity`` placeholders marking
  hookable LRP probe sites. Subclassing lets layer_map composites target
  Q vs K vs V independently.

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

LRP rules are NOT canonizers here — they are zennit ``Hook`` subclasses in
:mod:`zennit_ext.attnlrp_rules`, assigned by module type via a composite
``layer_map`` (see :mod:`zennit_ext.attnlrp_composites`). The only ``Hook``
defined in this file is :class:`StopGradient` (zeros backward relevance;
layer-map onto :class:`QInspectionLayer` / :class:`KInspectionLayer` for the
CP-LRP attention rule — relevance flows through the value path only).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from timm.layers import apply_rot_embed_cat

from zennit.canonizers import AttributeCanonizer




# ─── 1. Vanilla atomic modules ──────────────────────────────────────────────


class BilinearMatmul(nn.Module):
    """``y = a @ b``. Vanilla bilinear matmul; autograd's standard backward.

    Used inside :class:`EvaAttentionUnfolded` and :class:`TimmAttentionUnfolded`
    for ``q @ kᵀ`` and ``weights @ v``, and inside classification heads
    that want these matmuls exposed as named submodules. To enable the
    AttnLRP AlphaBeta rule on backward at attribution time, assign the
    :class:`~zennit_ext.attnlrp_rules.AlphaBetaMatmul` hook to this type via a
    composite ``layer_map``.
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
    """``y = x + branch``. Vanilla addition — the **single** module type for a
    residual skip merge.

    The LRP behaviour is **not** encoded in the module: pick the residual rule
    in the composite ``layer_map`` by mapping this type to the desired hook —
    :class:`~zennit_ext.attnlrp_rules.ResidualRatio` (Otsuki ``|x|``/``|branch|``
    split), :class:`~zennit_ext.attnlrp_rules.Uniform` (½ each, symmetric), or
    :class:`~zennit_ext.attnlrp_rules.ResidualL1` (sign-preserving, L1-conserving).
    Mapping it to nothing leaves the plain (non-conservative) add. Do NOT create
    a separate module type per rule — one module, many selectable rules.
    """

    def forward(self, x: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return x + branch


class PosEmbedAdd(ResidualAdd):
    """Alias of :class:`ResidualAdd` (same ``x + branch`` op) kept as a distinct
    dispatch type *only* so the ``x + pos_embed`` PA-LRP merge can take its own
    ``layer_map`` rule (the uniform ½ split) independently of the residual skips.
    Different graph role, not a different rule for the same module. Order it
    BEFORE ``ResidualAdd`` in a ``layer_map`` (zennit matches by ``isinstance``,
    first hit wins)."""


class SoftmaxAlongLastDim(nn.Module):
    """``y = F.softmax(x, dim=-1)``. Vanilla.

    For the AttnLRP identity rule (``R_in = R_out``, AttnLRP Eq. 9), assign
    zennit's stock ``Pass`` rule to this type via a composite ``layer_map``.
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
    absorbs no relevance), assign zennit's stock ``Pass`` rule to this type
    via a composite ``layer_map``.
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
    other half), assign the :class:`~zennit_ext.attnlrp_rules.Uniform` rule
    to this type via a composite ``layer_map``.

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


class LRPInspectionLayer(nn.Identity):
    """Identity placeholder marking a hookable site for LRP relevance
    inspection.

    Mathematically a no-op (just :class:`nn.Identity`). The reason it has
    its own class name is so graphviz / torchview renders ``LRPInspectionLayer``
    instead of an anonymous ``Identity`` node, and so
    ``get_layer_names(model, [LRPInspectionLayer])`` can enumerate the
    sites that concept classes are designed to hook.

    Three role-specific subclasses (:class:`QInspectionLayer`,
    :class:`KInspectionLayer`, :class:`VInspectionLayer`) are used on the
    unfolded attention containers. Subclassing lets ``LayerMapComposite``
    target Q vs K vs V independently (e.g. mapping
    ``QInspectionLayer → StopGradient()`` for CP-LRP-style attention)
    while ``isinstance(m, LRPInspectionLayer)`` enumeration still catches
    all three for concept hooking.
    """
    pass


class QInspectionLayer(LRPInspectionLayer):
    """Identity probe at the Q slot of an unfolded attention container.

    Layer-map target for Q-specific LRP rules — e.g. :class:`StopGradient`
    for the CP-LRP attention rule (block relevance flow through the Q
    projection).
    """
    pass


class KInspectionLayer(LRPInspectionLayer):
    """Identity probe at the K slot of an unfolded attention container.

    See :class:`QInspectionLayer`. Pair with :class:`StopGradient` in the
    composite layer_map for CP-LRP attention.
    """
    pass


class VInspectionLayer(LRPInspectionLayer):
    """Identity probe at the V slot of an unfolded attention container.

    See :class:`QInspectionLayer`. Under CP-LRP, the V probe stays as a
    pure identity so relevance flows through the value path unchanged.
    """
    pass


# ─── 2. Shared reshape helper ───────────────────────────────────────────────


def _to_heads(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """``(B, N, num_heads * head_dim) → (B, num_heads, N, head_dim)``.

    Per-head reshape shared by :class:`EvaAttentionUnfolded` and
    :class:`TimmAttentionUnfolded` forwards.
    """
    B, N, _ = t.shape
    return t.reshape(B, N, num_heads, head_dim).transpose(1, 2)


class TimmAttentionUnfolded(nn.Module):
    """Unfolded ``timm.models.vision_transformer.Attention`` analogue.

    Constructor takes a real ``Attention`` instance, references
    its parameter-bearing submodules (``qkv``, ``q_norm``, ``k_norm``,
    ``proj``, ``attn_drop``, ``proj_drop``, optional ``norm``) by
    attribute, and exposes hookable atomic ops for zennit canonizers.

    Forward signature mirrors
    stock ``timm.models.vision_transformer.Attention.forward``
    The block keeps invoking ``self.attn(x, attn_mask=…, is_causal=…)`` with no signature break.
    For ViT image classification ``is_causal`` is always ``False``; the
    kwarg is accepted but trivially unused on that path.

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
        # LRP inspection sites. Placed AFTER ``split``, BEFORE
        # ``_to_heads`` — tensor shape is ``(B, N, embed_dim)``, identical
        # to ``proj_drop`` output. HeadConcept slices embed_dim per-head
        # internally; q_norm / k_norm and the per-head reshape are
        # downstream and not inspected through these probes. Distinct
        # subclasses (Q/K/V) so a LayerMapComposite can target them
        # independently with different LRP rules.
        self.q_lrp_probe = QInspectionLayer()
        self.k_lrp_probe = KInspectionLayer()
        self.v_lrp_probe = VInspectionLayer()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv_flat = self.qkv(x)
        q_flat, k_flat, v_flat = self.split(qkv_flat)
        q_flat = self.q_lrp_probe(q_flat)
        k_flat = self.k_lrp_probe(k_flat)
        v_flat = self.v_lrp_probe(v_flat)

        q = _to_heads(q_flat, self.num_heads, self.head_dim)
        k = _to_heads(k_flat, self.num_heads, self.head_dim)
        v = _to_heads(v_flat, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

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


# ─── 3. Container: EvaAttentionUnfolded (vanilla forward) ───────────────────


# ─── 3. Substitution canonizer (no LRP knobs) ───────────────────────────────


# ─── 3b. Timm-standard unfolded attention + substitution canonizer ──────────


# ─── 4. Autograd Function kernels for the LRP rules ─────────────────────────
# These are used ONLY by the canonizers below — never by the modules' own
# forwards. They live here (not in transformer_patches) because their sole
# consumers are the per-module canonizers right below them.


# ─── 5. LRP-rule canonizers (one per rule, vanilla ↔ rule-aware swap) ───────
#
# Each canonizer rebinds the forward of a specific module class with a
# wrapper that routes through an autograd Function whose backward
# implements the LRP rule. On ``apply()`` the wrapper is bound; on
# ``remove()`` the original ``forward`` is restored. Forward output is
# numerically identical to vanilla in every case (Functions do
# ``a @ b`` / ``F.softmax`` / ``x * scalar`` etc. — only backward differs).


# ─── 6. Internal helpers ────────────────────────────────────────────────────



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
    submodules are vanilla (see module docstrings)..

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
        # LRP inspection sites for HeadConcept / EmbeddingDimConcept hooking.
        # Placed AFTER ``split``, BEFORE ``_to_heads`` — tensor shape is
        # ``(B, N, embed_dim)``, identical to ``proj_drop`` output. One
        # shape contract everywhere: HeadConcept slices embed_dim per-head
        # internally; the per-head reshape is downstream and irrelevant
        # to attribution. Distinct subclasses (Q/K/V) so a LayerMapComposite
        # can target them independently with different LRP rules.
        self.q_lrp_probe = QInspectionLayer()
        self.k_lrp_probe = KInspectionLayer()
        self.v_lrp_probe = VInspectionLayer()

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
        q_flat = self.q_lrp_probe(q_flat)
        k_flat = self.k_lrp_probe(k_flat)
        v_flat = self.v_lrp_probe(v_flat)

        q = _to_heads(q_flat, self.num_heads, self.head_dim)
        k = _to_heads(k_flat, self.num_heads, self.head_dim)
        v = _to_heads(v_flat, self.num_heads, self.head_dim)

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
