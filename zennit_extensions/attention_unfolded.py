"""Vanilla (standard-autograd) attention atomics + unfolded timm-attention containers.

Everything here is an ``nn.Module`` with a plain PyTorch forward. LRP
behaviour is injected only at attribution time by zennit ``Hook`` rules
(:mod:`zennit_extensions.rules`), assigned by module type through a
composite ``layer_map`` and detached on ``composite.context()`` exit — so
these modules also work inside trainable heads, receiving correct
chain-rule gradients.

:class:`EvaAttentionUnfolded` / :class:`TimmAttentionUnfolded` compose the
atomics into forward-parity replacements for timm attention modules,
referencing (not copying) the original parameters so checkpoint loading
still works. They are installed by the substitution canonizers in
:mod:`zennit_extensions.canonisation.canonizers`.

:class:`LRPInspectionLayer` (and its Q/K/V subclasses) = identity probe
sites on the ``(B, N, embed_dim)`` tensors for concept hooking, targetable
per-role by a ``layer_map`` — e.g. Q/K →
:class:`~zennit_extensions.cp_lrp.StopGradient` for value-path-only CP-LRP.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import apply_rot_embed_cat


__all__ = [
    "AddBias", "BilinearMatmul", "ChunkAlongLastDim", "EvaAttentionUnfolded",
    "KInspectionLayer", "LRPInspectionLayer", "LayerScaleMul", "PosEmbedAdd",
    "QInspectionLayer", "ReshapeMergeHeads", "ResidualAdd", "RotaryEmbedding",
    "ScaleByConstant", "SoftmaxAlongLastDim", "TimmAttentionUnfolded",
    "VInspectionLayer",
]


# ─── 1. Vanilla atomic modules ──────────────────────────────────────────────


class BilinearMatmul(nn.Module):
    """``y = a @ b``. Layer-map target for a bilinear LRP rule — e.g.
    :class:`~zennit_extensions.rules.bajger_contrib.AlphaBetaMatmul` or
    :class:`~zennit_extensions.rules.attnlrp.MatmulAttnLRP` (both the
    ``q@kᵀ`` and ``attn@v`` products)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b


class AddBias(nn.Module):
    """``y = x + bias`` (identity when ``bias is None``). No dedicated rule:
    the typical bias is a constant mask, so autograd already routes
    relevance correctly (``grad_x = grad_y``; ``grad_bias`` discarded)."""

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if bias is None:
            return x
        return x + bias


class ResidualAdd(nn.Module):
    """``y = x + branch`` — the single module type for a residual skip merge.

    The residual *rule* is a ``layer_map`` choice on this one type:
    :class:`~zennit_extensions.rules.residuals_otsuki2024.ResidualRatio`
    (``|x|``/``|branch|`` split), :class:`~zennit_extensions.rules.attnlrp.Uniform`
    (½ each), or :class:`~zennit_extensions.rules.bajger_contrib.ResidualL1`
    (sign-preserving). Unmapped = plain (non-conservative) add. Do NOT create
    a module type per rule — one module, many selectable rules.
    """

    def forward(self, x: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return x + branch


class PosEmbedAdd(ResidualAdd):
    """Alias of :class:`ResidualAdd` kept as a distinct dispatch type only so
    the ``x + pos_embed`` merge can take its own ``layer_map`` rule (the PA-LRP
    uniform ½ split). Order it BEFORE ``ResidualAdd`` in a map (zennit matches
    by ``isinstance``, first hit wins)."""


class SoftmaxAlongLastDim(nn.Module):
    """``y = softmax(x, dim=-1)``. For the AttnLRP identity rule (Eq. 9),
    assign zennit's stock ``Pass`` rule; the full cross-term rule is
    :class:`~zennit_extensions.rules.attnlrp.SoftmaxAttnLRP`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE via ``timm.layers.apply_rot_embed_cat`` with optional
    prefix-token skip (DINOv3/EVA cls+register tokens are not rotated).

    ``rope=None`` → identity. ``detach_rope=True`` detaches cos/sin so
    gradients route purely through the rotated tensor; structural choice
    (RoPE has no learnable parameters), not an LRP rule, safe at training
    time. Default False for forward parity with stock ``EvaAttention``.
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
    """``y = x * value`` with ``value`` a graph constant. AttnLRP treats it
    as absorbing no relevance → stock zennit ``Pass`` via the ``layer_map``."""

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.value

    def extra_repr(self) -> str:
        return f"value={self.value}"


class ChunkAlongLastDim(nn.Module):
    """Split into ``n`` chunks along the last dim. Backward is ``torch.cat``
    — LRP-identity; no rule needed."""

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
    """``x.transpose(1, 2).reshape(B, N, out_dim)`` — merge per-head context.
    Pure reshape preserves ``sum(R)``; no rule needed."""

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
    """``y = γ * x`` (CaiT LayerScale). For the AttnLRP uniform rule on
    backward (γ absorbs half the relevance), assign
    :class:`~zennit_extensions.rules.attnlrp.Uniform` via the ``layer_map``.

    Parameters
    ----------
    gamma : nn.Parameter
        The parent ``EvaBlock``'s LayerScale parameter, referenced without
        re-registering so weight loading still flows through the parent.
    """

    def __init__(self, gamma: nn.Parameter):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class LRPInspectionLayer(nn.Identity):
    """Named ``nn.Identity`` marking a hookable relevance-inspection site —
    named so graphviz renders it and ``get_layer_names(model,
    [LRPInspectionLayer])`` can enumerate probe sites. The Q/K/V subclasses
    let a ``layer_map`` target roles independently while
    ``isinstance(m, LRPInspectionLayer)`` still enumerates all three."""
    pass


class QInspectionLayer(LRPInspectionLayer):
    """Q-slot probe. Pair with :class:`~zennit_extensions.cp_lrp.StopGradient`
    for CP-LRP (block relevance flow through the Q projection)."""
    pass


class KInspectionLayer(LRPInspectionLayer):
    """K-slot probe. See :class:`QInspectionLayer`."""
    pass


class VInspectionLayer(LRPInspectionLayer):
    """V-slot probe. Under CP-LRP stays a pure identity so relevance flows
    through the value path unchanged."""
    pass


# ─── 2. Unfolded attention containers (vanilla forwards) ────────────────────


def _to_heads(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """``(B, N, num_heads * head_dim) → (B, num_heads, N, head_dim)``."""
    B, N, _ = t.shape
    return t.reshape(B, N, num_heads, head_dim).transpose(1, 2)


class TimmAttentionUnfolded(nn.Module):
    """Unfolded ``timm.models.vision_transformer.Attention`` — forward-parity
    replacement composed of the vanilla atomics. References (not copies) the
    original's parameter-bearing submodules, so checkpoint loading is
    unaffected.

    Stock timm ``Attention`` does not carry ``num_prefix_tokens`` (it lives
    only on the top-level ``VisionTransformer``), so the substitution
    canonizer passes it in at construction. ``is_causal`` is accepted for
    signature parity; always False on the ViT classification path.
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

        # Atomic vanilla submodules; a composite layer_map assigns LRP rules.
        self.split = ChunkAlongLastDim(3)
        self.scale_q = ScaleByConstant(self.scale)
        self.qk_scores = BilinearMatmul()
        self.add_mask = AddBias()
        self.softmax = SoftmaxAlongLastDim()
        self.context = BilinearMatmul()
        self.reshape = ReshapeMergeHeads()
        # Probe sites on the (B, N, embed_dim) tensors (same shape as
        # ``proj_drop`` output): HeadConcept slices embed_dim per-head
        # internally, so the downstream per-head reshape is irrelevant to
        # attribution. Distinct Q/K/V subclasses → independent layer_map rules.
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
        # Stock timm adds attn_mask on this explicit-math path and handles
        # is_causal only on the F.scaled_dot_product_attention dispatch; we
        # build the triangular mask explicitly for parity with the Eva variant.
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


class EvaAttentionUnfolded(nn.Module):
    """Unfolded ``timm.models.eva.EvaAttention`` — forward-parity replacement
    composed of the vanilla atomics, with RoPE and a prefix-token skip.
    References the original's parameters, so checkpoint keys
    (``blocks.{i}.attn.qkv.weight``, …) keep resolving.

    Variants with separate per-Q/K/V biases (``vit_*_dinov3_qkvb``) are not
    supported; ``rope_detach`` is forwarded to :class:`RotaryEmbedding`.
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

        # Atomic vanilla submodules; a composite layer_map assigns LRP rules.
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
        # Probe sites — same placement and rationale as TimmAttentionUnfolded.
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
