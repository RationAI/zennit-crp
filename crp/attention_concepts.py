"""Vision-transformer concept-detector classes for CRP.

These hook the post-qkv pre-attention tap (an `nn.Identity` named ``qkv_tap``
inserted by ``crp.transformer_patches.inject_qkv_taps``). The tap output has
shape ``(B, N, 3*D)`` with layout ``[Q | K | V]`` along the last axis, where
``D = num_heads * head_dim``. Within each part, heads are contiguous:
``part_offset + head*head_dim : part_offset + (head+1)*head_dim``.

Four concept granularities are provided, all sharing the same hook tap:

* :class:`HeadConcept` — one concept per attention head; mask = ``Q[h] ∪ K[h] ∪ V[h]``.
* :class:`KQVConcept` — three concepts per attention block, one for each whole
  Q/K/V projection (across all heads).
* :class:`KQVHeadConcept` — ``3 × num_heads`` concepts; one per ``(part, head)``.
* :class:`HeadDimConcept` — ``3 × num_heads × head_dim`` concepts; one per
  ``(part, head, dim)`` (i.e. one column of W_Q / W_K / W_V).

A previous POC class :class:`AttentionHeadConcept` (in :mod:`crp.concepts`)
hooked at the post-``proj`` attention output, which mixes all heads via the
output Linear so a head-stripe mask there does **not** isolate head ``h``.
That class has been removed; the four classes here replace it. See
``CURRENT_STATE.md`` for the audit finding.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from crp.concepts import ChannelConcept


PART_OFFSETS: Dict[str, int] = {"q": 0, "k": 1, "v": 2}
PARTS: Tuple[str, ...] = ("q", "k", "v")


def _coerce_part(part: Union[str, int]) -> str:
    if isinstance(part, str):
        if part not in PART_OFFSETS:
            raise ValueError(f"part must be one of {tuple(PART_OFFSETS)}; got {part!r}")
        return part
    if isinstance(part, (int, np.integer)):
        if not 0 <= int(part) < 3:
            raise ValueError(f"part-as-int must be in [0,2]; got {part}")
        return PARTS[int(part)]
    raise TypeError(f"part must be str or int; got {type(part).__name__}")


class _BaseAttentionConcept(ChannelConcept):
    """Abstract base: parameterised mask + aggregation on the qkv_tap tensor.

    Subclasses implement :meth:`_concept_to_slices` (mask shape) and
    :meth:`_aggregate` (sum-reduction axes for ``attribute``).
    """

    def __init__(self) -> None:
        # layer_name -> (num_heads, head_dim)
        self._layer_dims: Dict[str, Tuple[int, int]] = {}

    # ── registration ──────────────────────────────────────────────────────────

    def register_layer(self, layer_name: str, num_heads: int, head_dim: int) -> None:
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError("head_dim must be a positive integer")
        self._layer_dims[layer_name] = (num_heads, head_dim)

    def register_from_model(self, model) -> None:
        """Register every Attention-like submodule that exposes both
        ``num_heads`` (or alias) and ``head_dim`` (or computable from
        ``embed_dim`` / ``hidden_dim``).

        Registers under both the attention module's name AND its
        ``<name>.qkv_tap`` child name, so resolution succeeds regardless of
        which is supplied as the layer key.
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

    def _resolve_dims(self, layer_name: str) -> Tuple[int, int]:
        """Resolve (num_heads, head_dim) for ``layer_name`` with parent fallback."""
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
            "Call concept.register_from_model(model) or concept.register_layer(...)."
        )

    # ── subclass hooks ────────────────────────────────────────────────────────

    def _concept_to_slices(
        self, concept_id, num_heads: int, head_dim: int
    ) -> List[slice]:
        raise NotImplementedError

    def _per_token_relevance(
        self, relevance: torch.Tensor, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        """Reshape ``(B, N, 3*D)`` to ``(B, N, *concept_axes)`` — per-token,
        per-concept-id relevance. Used by both :meth:`_aggregate` (which sums
        out the sequence dim) and :meth:`reference_sampling` (which keeps it
        for receptive-field neuron lookup)."""
        raise NotImplementedError

    def _aggregate(
        self, relevance: torch.Tensor, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        return self._per_token_relevance(relevance, num_heads, head_dim).sum(dim=1)

    # ── mask ──────────────────────────────────────────────────────────────────

    def mask(self, batch_id: int, concept_ids: List, layer_name: str = None):
        num_heads, head_dim = self._resolve_dims(layer_name)
        d_total = 3 * num_heads * head_dim

        slice_list: List[slice] = []
        for cid in concept_ids:
            slice_list.extend(self._concept_to_slices(cid, num_heads, head_dim))

        def mask_fct(grad: torch.Tensor) -> torch.Tensor:
            if grad[batch_id].shape[-1] != d_total:
                raise ValueError(
                    f"qkv_tap last-dim mismatch for layer {layer_name!r}: "
                    f"expected {d_total}, got {grad[batch_id].shape[-1]}. "
                    "Hook is not on the qkv_tap tensor."
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
        rel_l = self._aggregate(relevance, num_heads, head_dim)

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
        ``(batch, num_concepts)``, where ``num_concepts`` is the row-major
        flatten of the concept axes (matching the layout of
        :meth:`attribute`). ``rf_c_sorted`` holds, per (sample-rank, concept),
        the **token (sequence) index** at which the concept's relevance peaks
        — the closest analogue of the "receptive field neuron" used by
        :class:`~crp.concepts.ChannelConcept` for 2D conv layers.
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


# ── (A) head as concept ──────────────────────────────────────────────────────


class HeadConcept(_BaseAttentionConcept):
    """One concept per attention head.

    Concept id: bare ``int`` head index, or ``(head_id,)`` tuple. The mask
    selects all three head-stripes ``Q[h] ∪ K[h] ∪ V[h]`` per the spec
    *"the full triple of weight matrices considered as a single unit"*.

    ``attribute`` output shape: ``(B, num_heads)``.
    """

    def _concept_to_slices(self, concept_id, num_heads, head_dim):
        if isinstance(concept_id, (tuple, list)):
            if len(concept_id) != 1:
                raise ValueError(
                    f"HeadConcept expects a single head index; got {concept_id!r}"
                )
            (head_id,) = concept_id
        else:
            head_id = concept_id
        head_id = int(head_id)
        if not 0 <= head_id < num_heads:
            raise IndexError(f"head index {head_id} out of range [0, {num_heads})")
        D = num_heads * head_dim
        s, e = head_id * head_dim, (head_id + 1) * head_dim
        return [
            slice(s, e),
            slice(D + s, D + e),
            slice(2 * D + s, 2 * D + e),
        ]

    def _per_token_relevance(self, relevance, num_heads, head_dim):
        B, N, _ = relevance.shape
        rel = relevance.view(B, N, 3, num_heads, head_dim)
        return rel.sum(dim=(2, 4))  # -> (B, N, num_heads)


# ── (B1) whole Q / K / V as a concept ────────────────────────────────────────


class KQVConcept(_BaseAttentionConcept):
    """Three concepts per attention block: whole Q, whole K, whole V projections.

    Concept id: ``'q'``, ``'k'``, ``'v'``, or an ``int`` in ``[0,2]`` (Q=0,K=1,V=2).

    ``attribute`` output shape: ``(B, 3)``, last axis ordered (Q, K, V).
    """

    def _concept_to_slices(self, concept_id, num_heads, head_dim):
        if isinstance(concept_id, (tuple, list)):
            if len(concept_id) != 1:
                raise ValueError(
                    f"KQVConcept expects (part,) or scalar; got {concept_id!r}"
                )
            (raw,) = concept_id
        else:
            raw = concept_id
        part = _coerce_part(raw)
        D = num_heads * head_dim
        offset = PART_OFFSETS[part] * D
        return [slice(offset, offset + D)]

    def _per_token_relevance(self, relevance, num_heads, head_dim):
        B, N, _ = relevance.shape
        D = num_heads * head_dim
        rel = relevance.view(B, N, 3, D)
        return rel.sum(dim=3)  # -> (B, N, 3)


# ── (B2) per-head Q / K / V — a (part, head) concept ─────────────────────────


class KQVHeadConcept(_BaseAttentionConcept):
    """``3 × num_heads`` concepts per attention block: one per ``(part, head)``.

    Concept id: ``(part, head_id)`` where ``part`` is ``'q'|'k'|'v'`` (or
    int 0..2) and ``head_id`` is in ``[0, num_heads)``.

    ``attribute`` output shape: ``(B, 3, num_heads)``, axis 1 ordered (Q, K, V).
    """

    def _concept_to_slices(self, concept_id, num_heads, head_dim):
        if isinstance(concept_id, (int, np.integer)):
            # Flat index in row-major (3, num_heads): part = i // H, head = i % H
            flat = int(concept_id)
            if not 0 <= flat < 3 * num_heads:
                raise IndexError(
                    f"flat concept index {flat} out of range [0, {3*num_heads})"
                )
            raw_part, head_id = divmod(flat, num_heads)
        elif isinstance(concept_id, (tuple, list)) and len(concept_id) == 2:
            raw_part, head_id = concept_id
        else:
            raise ValueError(
                f"KQVHeadConcept expects (part, head_id) or flat int; got {concept_id!r}"
            )
        part = _coerce_part(raw_part)
        head_id = int(head_id)
        if not 0 <= head_id < num_heads:
            raise IndexError(f"head index {head_id} out of range [0, {num_heads})")
        D = num_heads * head_dim
        offset = PART_OFFSETS[part] * D + head_id * head_dim
        return [slice(offset, offset + head_dim)]

    def _per_token_relevance(self, relevance, num_heads, head_dim):
        B, N, _ = relevance.shape
        rel = relevance.view(B, N, 3, num_heads, head_dim)
        return rel.sum(dim=4)  # -> (B, N, 3, num_heads)


# ── (C) per-(part, head, dim) — a single column of W_Q/W_K/W_V ───────────────


class HeadDimConcept(_BaseAttentionConcept):
    """``3 × num_heads × head_dim`` concepts per attention block: one column of
    W_Q / W_K / W_V per head, per output-feature index.

    Row vs column resolution: in ``Q = X · W_Q`` with ``W_Q ∈ R^{D×D}``, each
    column of W_Q maps to one output feature of Q (i.e. one of the ``head_dim``
    indices within a head). CRP's CNN convention treats one output filter as
    one concept; the analogue here is one **column**. (See
    ``IMPLEMENTATION_PLAN.md`` Phase 3.)

    Concept id: ``(part, head_id, dim_id)``.

    ``attribute`` output shape: ``(B, 3, num_heads, head_dim)``.
    """

    def _concept_to_slices(self, concept_id, num_heads, head_dim):
        if isinstance(concept_id, (int, np.integer)):
            # Flat index in row-major (3, num_heads, head_dim).
            flat = int(concept_id)
            if not 0 <= flat < 3 * num_heads * head_dim:
                raise IndexError(
                    f"flat concept index {flat} out of range "
                    f"[0, {3*num_heads*head_dim})"
                )
            raw_part, rem = divmod(flat, num_heads * head_dim)
            head_id, dim_id = divmod(rem, head_dim)
        elif isinstance(concept_id, (tuple, list)) and len(concept_id) == 3:
            raw_part, head_id, dim_id = concept_id
        else:
            raise ValueError(
                f"HeadDimConcept expects (part, head_id, dim_id) or flat int; "
                f"got {concept_id!r}"
            )
        part = _coerce_part(raw_part)
        head_id = int(head_id)
        dim_id = int(dim_id)
        if not 0 <= head_id < num_heads:
            raise IndexError(f"head index {head_id} out of range [0, {num_heads})")
        if not 0 <= dim_id < head_dim:
            raise IndexError(f"dim index {dim_id} out of range [0, {head_dim})")
        D = num_heads * head_dim
        offset = PART_OFFSETS[part] * D + head_id * head_dim + dim_id
        return [slice(offset, offset + 1)]

    def _per_token_relevance(self, relevance, num_heads, head_dim):
        B, N, _ = relevance.shape
        return relevance.view(B, N, 3, num_heads, head_dim)
        # -> (B, N, 3, num_heads, head_dim)


__all__ = [
    "PARTS",
    "PART_OFFSETS",
    "HeadConcept",
    "KQVConcept",
    "KQVHeadConcept",
    "HeadDimConcept",
]
