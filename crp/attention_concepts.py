"""Vision-transformer concept-detector classes for CRP.

There are **two orthogonal granularity dimensions** for an attention block:

1. **Where** the relevance is read.
   * **Output side (default)** — at the per-head pre-projection output tokens
     ``attn @ v`` (shape ``(B, N, num_heads*head_dim)``). Hooked at the named
     ``attn.attn_out_tap`` Identity submodule injected by
     :class:`crp.transformer_patches.AttentionTapsCanonizer`. This treats the
     attention head as a single concept detector (the standard CRP analogue
     to a CNN filter).
   * **K/Q/V side** — at the post-``qkv``-Linear pre-attention pack
     (shape ``(B, N, 3*num_heads*head_dim)``). Hooked at ``attn.qkv_tap``.
     Splits each head into three concept detectors, one per
     query / key / value projection.

2. **Whether the per-head ``head_dim`` axis is split.** Either treat all
   ``head_dim`` indices within a head as one concept (``head_dim`` summed
   out), or treat each individual index as its own concept (kept).

Crossing these two dimensions gives the four supported concept classes:

| class               | tap            | concept id                | ``attribute()`` shape                |
|---------------------|----------------|---------------------------|--------------------------------------|
| :class:`HeadConcept`         | ``attn_out_tap`` | ``head``               | ``(B, num_heads)``                  |
| :class:`HeadDimConcept`      | ``attn_out_tap`` | ``(head, dim)``         | ``(B, num_heads, head_dim)``        |
| :class:`KQVHeadConcept`      | ``qkv_tap``      | ``(part, head)``        | ``(B, 3, num_heads)``               |
| :class:`KQVHeadDimConcept`   | ``qkv_tap``      | ``(part, head, dim)``    | ``(B, 3, num_heads, head_dim)``     |

``part`` is one of ``'q'``, ``'k'``, ``'v'`` (or its int index 0/1/2). All
classes also accept a flat row-major ``int`` concept id in the layout of
``attribute()``'s output (after the batch dim), which is what
:class:`crp.maximization.Maximization` / :class:`crp.visualization.FeatureVisualization`
pass through when iterating ``argsort``ed indices.

All four classes share one base (:class:`_AttentionConcept`); the four
subclasses differ only in the two boolean flags ``KQV_SPLIT`` and
``DIM_SPLIT``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from crp.concepts import ChannelConcept


PART_OFFSETS: Dict[str, int] = {"q": 0, "k": 1, "v": 2}
PARTS: Tuple[str, ...] = ("q", "k", "v")


def _coerce_part(part: Union[str, int]) -> int:
    """Normalise a ``part`` argument to its 0/1/2 int index."""
    if isinstance(part, str):
        if part not in PART_OFFSETS:
            raise ValueError(f"part must be one of {tuple(PART_OFFSETS)}; got {part!r}")
        return PART_OFFSETS[part]
    if isinstance(part, (int, np.integer)):
        if not 0 <= int(part) < 3:
            raise ValueError(f"part-as-int must be in [0,2]; got {part}")
        return int(part)
    raise TypeError(f"part must be str or int; got {type(part).__name__}")


class _AttentionConcept(ChannelConcept):
    """Shared base for the four ViT attention-concept classes.

    Subclasses set two class-level booleans:

    * ``KQV_SPLIT`` — if True, the concept lives on the ``qkv_tap`` (last dim
      ``3*D``) and the concept id has a leading ``part`` axis. If False,
      lives on ``attn_out_tap`` (last dim ``D``).
    * ``DIM_SPLIT`` — if True, the per-head ``head_dim`` axis is split into
      individual concepts; if False, summed out per head.

    See the module docstring for the resulting ``(class, tap, shape)``
    matrix.
    """

    KQV_SPLIT: bool = False
    DIM_SPLIT: bool = False

    def __init__(self, model: torch.nn.Module = None) -> None:
        # layer_name -> (num_heads, head_dim).
        self._layer_dims: Dict[str, Tuple[int, int]] = {}
        if model is not None:
            self.register_from_model(model)

    # ── tap name (subclass-driven) ────────────────────────────────────────────

    @property
    def tap_name(self) -> str:
        """Suffix the concept's hook tap appears under on each Attention module."""
        return "qkv_tap" if self.KQV_SPLIT else "attn_out_tap"

    def _expected_last_dim(self, num_heads: int, head_dim: int) -> int:
        return (3 if self.KQV_SPLIT else 1) * num_heads * head_dim

    # ── registration ──────────────────────────────────────────────────────────

    def register_layer(self, layer_name: str, num_heads: int, head_dim: int) -> None:
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError("head_dim must be a positive integer")
        self._layer_dims[layer_name] = (num_heads, head_dim)

    def register_from_model(self, model: torch.nn.Module) -> None:
        """Discover every attention-like submodule in ``model`` and record
        its ``(num_heads, head_dim)`` under the module's name and under both
        tap names ``<name>.qkv_tap`` / ``<name>.attn_out_tap``.

        ``head_dim`` is inferred from a per-module attribute (``head_dim``)
        or computed from ``embed_dim / num_heads`` if only the embedding
        size is exposed. Recognises the common ViT/Transformer attribute
        spellings on timm, HuggingFace, and torchvision attention modules.
        """
        head_attrs = ("num_heads", "n_heads", "num_attention_heads")
        dim_attrs = ("head_dim",)
        embed_attrs = ("embed_dim", "hidden_dim", "hidden_size", "d_model")

        for name, module in model.named_modules():
            num_heads = next(
                (getattr(module, a) for a in head_attrs if hasattr(module, a)),
                None,
            )
            head_dim = next(
                (getattr(module, a) for a in dim_attrs if hasattr(module, a)),
                None,
            )
            if num_heads is None:
                continue
            if head_dim is None:
                embed = next(
                    (getattr(module, a) for a in embed_attrs if hasattr(module, a)),
                    None,
                )
                if embed is None or num_heads == 0 or embed % num_heads != 0:
                    continue
                head_dim = embed // num_heads
            if not isinstance(num_heads, int) or not isinstance(head_dim, int):
                continue
            self._layer_dims[name] = (num_heads, head_dim)
            self._layer_dims[f"{name}.qkv_tap"] = (num_heads, head_dim)
            self._layer_dims[f"{name}.attn_out_tap"] = (num_heads, head_dim)

    def _resolve_dims(self, layer_name: str) -> Tuple[int, int]:
        """Resolve ``(num_heads, head_dim)`` for ``layer_name`` with a
        parent-name fallback (so ``blocks.6.attn.attn_out_tap`` resolves
        from a registration on ``blocks.6.attn``)."""
        if not layer_name:
            raise ValueError("layer_name must be provided to resolve attention dims")
        if layer_name in self._layer_dims:
            return self._layer_dims[layer_name]
        parts = layer_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:i])
            if parent in self._layer_dims:
                return self._layer_dims[parent]
        raise ValueError(
            f"No attention dims registered for layer {layer_name!r}. "
            "Pass the model to the concept constructor, call "
            "concept.register_from_model(model), or use concept.register_layer(...)."
        )

    # ── concept-id decode ─────────────────────────────────────────────────────

    def _axis_sizes(self, num_heads: int, head_dim: int) -> List[int]:
        """The active axes of this concept's id, row-major. Used both for
        flat-int decode and for tuple-arity validation."""
        sizes: List[int] = []
        if self.KQV_SPLIT:
            sizes.append(3)
        sizes.append(num_heads)
        if self.DIM_SPLIT:
            sizes.append(head_dim)
        return sizes

    def _decode_concept_id(
        self, concept_id, num_heads: int, head_dim: int
    ) -> Tuple[int, int, int]:
        """Decode a concept id into ``(part_idx_or_0, head_id, dim_id_or_0)``.

        Accepted forms:

        * Bare ``int`` — flat row-major index across the active axes.
        * Tuple of int/str matching the active axes' arity. ``part`` may be
          ``'q'|'k'|'v'`` or its int alias.
        """
        sizes = self._axis_sizes(num_heads, head_dim)
        total = math.prod(sizes)

        if isinstance(concept_id, (int, np.integer)):
            flat = int(concept_id)
            if not 0 <= flat < total:
                raise IndexError(
                    f"flat concept index {flat} out of range [0, {total}) for "
                    f"{type(self).__name__} (axes={sizes})"
                )
            coords: List[int] = []
            for s in reversed(sizes):
                coords.append(flat % s)
                flat //= s
            coords = list(reversed(coords))
        elif isinstance(concept_id, (tuple, list)):
            if len(concept_id) != len(sizes):
                raise ValueError(
                    f"{type(self).__name__} expects a {len(sizes)}-tuple "
                    f"(axes={sizes}); got {concept_id!r}"
                )
            coords = list(concept_id)
        else:
            raise ValueError(
                f"concept id must be int or tuple; got {type(concept_id).__name__} "
                f"({concept_id!r})"
            )

        # Normalise + range-check axis-by-axis.
        it = iter(coords)
        part = _coerce_part(next(it)) if self.KQV_SPLIT else 0
        head = int(next(it))
        if not 0 <= head < num_heads:
            raise IndexError(f"head index {head} out of range [0, {num_heads})")
        if self.DIM_SPLIT:
            dim = int(next(it))
            if not 0 <= dim < head_dim:
                raise IndexError(f"dim index {dim} out of range [0, {head_dim})")
        else:
            dim = 0
        return part, head, dim

    def _concept_to_slices(
        self, concept_id, num_heads: int, head_dim: int
    ) -> List[slice]:
        """One slice on the tap tensor's last dim for one concept id.

        Layout of the last dim:

        * ``qkv_tap``: ``[Q[h0..d_h-1], Q[h1...], ..., K[...], V[...]]``
          — three contiguous ``D``-blocks, each with heads contiguous.
        * ``attn_out_tap``: ``[head0[d0..d_h-1], head1[...], ...]`` —
          one ``D``-block, heads contiguous.
        """
        part, head, dim = self._decode_concept_id(concept_id, num_heads, head_dim)
        D = num_heads * head_dim
        prefix = part * D if self.KQV_SPLIT else 0

        if self.DIM_SPLIT:
            offset = prefix + head * head_dim + dim
            return [slice(offset, offset + 1)]
        offset = prefix + head * head_dim
        return [slice(offset, offset + head_dim)]

    def _per_token_relevance(
        self, relevance: torch.Tensor, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        """Reshape ``(B, N, last_dim)`` relevance to per-concept-id shape,
        keeping the token axis ``N`` for downstream sum / argmax."""
        B, N, _ = relevance.shape
        if self.KQV_SPLIT:
            rel = relevance.view(B, N, 3, num_heads, head_dim)
        else:
            rel = relevance.view(B, N, num_heads, head_dim)
        if not self.DIM_SPLIT:
            rel = rel.sum(dim=-1)
        return rel

    # ── mask ──────────────────────────────────────────────────────────────────

    def mask(self, batch_id: int, concept_ids: List, layer_name: str = None):
        num_heads, head_dim = self._resolve_dims(layer_name)
        d_total = self._expected_last_dim(num_heads, head_dim)

        slice_list: List[slice] = []
        for cid in concept_ids:
            slice_list.extend(self._concept_to_slices(cid, num_heads, head_dim))

        def mask_fct(grad: torch.Tensor) -> torch.Tensor:
            if grad[batch_id].shape[-1] != d_total:
                raise ValueError(
                    f"{type(self).__name__} expects last-dim {d_total} on layer "
                    f"{layer_name!r}; got {grad[batch_id].shape[-1]}. "
                    f"Hook is not on the {self.tap_name} tensor."
                )
            mask = torch.zeros_like(grad[batch_id])
            for sl in slice_list:
                mask[..., sl] = 1
            grad[batch_id] = grad[batch_id] * mask
            return grad

        return mask_fct

    # ── attribute (per-concept relevance from masked tensor) ──────────────────

    def attribute(
        self,
        relevance: torch.Tensor,
        mask: torch.Tensor = None,
        layer_name: str = None,
        abs_norm: bool = True,
    ) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        num_heads, head_dim = self._resolve_dims(layer_name)
        rel_l = self._per_token_relevance(relevance, num_heads, head_dim).sum(dim=1)

        if abs_norm:
            shape = rel_l.shape
            flat = rel_l.reshape(shape[0], -1)
            denom = flat.abs().sum(dim=-1, keepdim=True) + 1e-10
            rel_l = (flat / denom).reshape(shape)

        return rel_l

    # ── FeatureVisualization compatibility ────────────────────────────────────

    def reference_sampling(
        self,
        relevance: torch.Tensor,
        layer_name: str = None,
        max_target: str = "sum",
        abs_norm: bool = True,
    ):
        """Sort the batch dim by per-concept relevance — required by
        :class:`crp.maximization.Maximization` (and therefore
        :class:`crp.visualization.FeatureVisualization`).

        Returns ``(d_c_sorted, rel_c_sorted, rf_c_sorted)`` each of shape
        ``(batch, num_concepts)`` where ``num_concepts`` is the row-major
        flatten of the concept axes (matching :meth:`attribute`).
        ``rf_c_sorted`` holds the **token (sequence) index** at which the
        concept's relevance peaks — the closest analogue of the
        "receptive field neuron" used by :class:`~crp.concepts.ChannelConcept`
        for 2D conv layers.
        """
        num_heads, head_dim = self._resolve_dims(layer_name)
        B = relevance.shape[0]
        rel_pt = self._per_token_relevance(relevance, num_heads, head_dim)
        N = rel_pt.shape[1]
        flat = rel_pt.reshape(B, N, -1)  # (B, N, num_concepts)

        rf_neuron = torch.argmax(flat, dim=1)  # (B, num_concepts)

        if max_target == "sum":
            rel_c = flat.sum(dim=1)
        elif max_target == "max":
            rel_c = torch.gather(flat, 1, rf_neuron.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")

        if abs_norm:
            rel_c = rel_c / (rel_c.abs().sum(-1, keepdim=True) + 1e-10)

        d_c_sorted = torch.argsort(rel_c, dim=0, descending=True)
        rel_c_sorted = torch.gather(rel_c, 0, d_c_sorted)
        rf_c_sorted = torch.gather(rf_neuron, 0, d_c_sorted)
        return d_c_sorted, rel_c_sorted, rf_c_sorted


# ── concrete classes (flag combinations) ─────────────────────────────────────


class HeadConcept(_AttentionConcept):
    """One concept per attention head, measured at the **output tokens**.

    Tap: ``attn_out_tap``.
    Concept id: ``head_id`` (int) or ``(head_id,)``.
    ``attribute()`` shape: ``(B, num_heads)``.
    """

    KQV_SPLIT = False
    DIM_SPLIT = False


class HeadDimConcept(_AttentionConcept):
    """One concept per ``(head, dim)`` of the attention output tokens.

    Tap: ``attn_out_tap``.
    Concept id: ``(head_id, dim_id)`` or flat ``int`` row-major over
    ``(num_heads, head_dim)``.
    ``attribute()`` shape: ``(B, num_heads, head_dim)``.
    """

    KQV_SPLIT = False
    DIM_SPLIT = True


class KQVHeadConcept(_AttentionConcept):
    """One concept per ``(part, head)`` on the K/Q/V projections.

    Tap: ``qkv_tap``.
    Concept id: ``(part, head_id)`` with ``part ∈ {'q','k','v'}`` (or int
    0/1/2), or flat ``int`` row-major over ``(3, num_heads)``.
    ``attribute()`` shape: ``(B, 3, num_heads)``.
    """

    KQV_SPLIT = True
    DIM_SPLIT = False


class KQVHeadDimConcept(_AttentionConcept):
    """One concept per ``(part, head, dim)`` — a single column of W_Q / W_K
    / W_V per head.

    Tap: ``qkv_tap``.
    Concept id: ``(part, head_id, dim_id)`` or flat ``int`` row-major over
    ``(3, num_heads, head_dim)``.
    ``attribute()`` shape: ``(B, 3, num_heads, head_dim)``.

    Row vs column resolution: in ``Q = X · W_Q`` with ``W_Q ∈ R^{D×D}``, each
    column of W_Q maps to one output feature of Q (i.e. one of the
    ``head_dim`` indices within a head). CRP's CNN convention treats one
    output filter as one concept; the analogue here is one **column**.
    """

    KQV_SPLIT = True
    DIM_SPLIT = True


__all__ = [
    "PARTS",
    "PART_OFFSETS",
    "HeadConcept",
    "HeadDimConcept",
    "KQVHeadConcept",
    "KQVHeadDimConcept",
]
