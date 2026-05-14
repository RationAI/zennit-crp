"""CRP Concept classes for the unfolded attention API.

Three concept classes total. They operate on 3D ``(B, N, embed_dim)``
relevance tensors recorded at:

* :class:`crp.attention_unfolded.LRPInspectionLayer` sites (the
  ``q_lrp_probe`` / ``k_lrp_probe`` / ``v_lrp_probe`` attributes on the
  unfolded attention container) — Q/K/V token sequences just after the
  ``qkv`` projection split, before per-head reshape.
* ``proj_drop`` — output of the attention block, post-projection.

All four sites carry the same shape ``(B, N, embed_dim)``, so the same
concept instance can be hooked at any of them. The interpretation of the
heatmap depends on the site:

* hooked at ``q_lrp_probe`` → "what input pixels populated this head's
  query subspace?"
* hooked at ``v_lrp_probe`` → "what input pixels populated this head's
  value subspace?"
* hooked at ``proj_drop`` → "what input pixels populated this head's
  contribution to the residual stream?"
* etc.

API mirrors :class:`crp.concepts.ChannelConcept` — ``mask(batch_id,
concept_ids, layer_name)`` returns a closure that modifies the gradient
in-place. Concept ids are integers (heads / dims / token positions
depending on the class).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from crp.concepts import Concept


class HeadConcept(Concept):
    """Per-head attribution on a 3D ``(B, N, embed_dim)`` relevance tensor.

    Slices ``embed_dim`` into ``num_heads`` contiguous segments of size
    ``head_dim = embed_dim / num_heads``. One concept = one head; the
    head's relevance is the sum over its embed_dim slice and over the
    (filtered) token axis.

    Hookable at any 3D site of the unfolded attention block:
    ``q_lrp_probe``, ``k_lrp_probe``, ``v_lrp_probe``, ``proj_drop``.

    Parameters
    ----------
    num_heads : int
        Number of attention heads. The constructor argument is required
        because ``embed_dim`` alone doesn't determine the per-head split.
        The substitution canonizer reads this from the model when
        building the unfolded attention; the user passes the same value
        here. For DINOv3 ViT-L: 16. For ``vit_base_patch16_224``: 12.
    token_filter : slice, optional
        Restrict the token axis to a subset before aggregating. Default:
        ``slice(None)`` (= include all tokens: cls + register + spatial).

        Examples for DINOv3 (``num_prefix_tokens = 5``: 1 cls + 4 reg):

        * ``slice(None)`` — all 261 tokens (default)
        * ``slice(5, None)`` — spatial patch tokens only (256)
        * ``slice(0, 1)`` — cls only
        * ``slice(1, 5)`` — register tokens only
        * ``slice(0, 5)`` — all prefix tokens (cls + register)

        The slice is applied to the ``N`` axis at both ``mask`` time
        (zeroes excluded positions) and ``attribute`` time (excludes
        them from the per-head sum). User must know the model's prefix
        layout when picking a slice.

    Concept ids : ``List[int]``. Each id is a head index in ``[0, num_heads)``.
    """

    def __init__(self, num_heads: int, token_filter: slice = slice(None)):
        self.num_heads = int(num_heads)
        self.token_filter = token_filter

    def mask(self, batch_id: int, concept_ids: List, layer_name: Optional[str] = None):
        num_heads = self.num_heads
        token_filter = self.token_filter
        head_ids = [int(h) for h in concept_ids]
        for h in head_ids:
            if not 0 <= h < num_heads:
                raise IndexError(f"head {h} out of [0, {num_heads})")

        def mask_fct(grad: torch.Tensor) -> torch.Tensor:
            # grad shape: (B, N, embed_dim).
            t = grad[batch_id]
            embed_dim = t.shape[-1]
            head_dim = embed_dim // num_heads
            mask = torch.zeros_like(t)
            for h in head_ids:
                mask[token_filter, h * head_dim:(h + 1) * head_dim] = 1
            grad[batch_id] = t * mask
            return grad

        return mask_fct

    def attribute(
        self,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        abs_norm: bool = True,
    ) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask
        # (B, N, embed_dim) → filter tokens → reshape into (B, N_f, num_heads, head_dim)
        rel = relevance[:, self.token_filter, :]
        B = rel.shape[0]
        embed_dim = rel.shape[-1]
        head_dim = embed_dim // self.num_heads
        rel = rel.reshape(B, rel.shape[1], self.num_heads, head_dim)
        # Sum over filtered tokens AND head_dim → (B, num_heads).
        rel = rel.sum(dim=(1, 3))
        if abs_norm:
            rel = rel / (rel.abs().sum(-1, keepdim=True) + 1e-10)
        return rel

    def reference_sampling(
        self,
        relevance: torch.Tensor,
        layer_name: Optional[str] = None,
        max_target: str = "sum",
        abs_norm: bool = True,
    ):
        # relevance: (B, N, embed_dim). Filter tokens, reshape into
        # (B, N_f, num_heads, head_dim), then aggregate per head.
        rel = relevance[:, self.token_filter, :]
        B = rel.shape[0]
        embed_dim = rel.shape[-1]
        head_dim = embed_dim // self.num_heads
        rel = rel.reshape(B, rel.shape[1], self.num_heads, head_dim)
        # Per-head, per-token: (B, N_f, num_heads). Sum over head_dim.
        rel_per_token = rel.sum(dim=-1)
        # Argmax over filtered tokens to get the receptive-field token id
        # per (batch, head). Shape (B, num_heads).
        rf_token_filtered = torch.argmax(rel_per_token, dim=1)
        # Map back to absolute token id using the slice's start.
        offset = self.token_filter.start or 0
        rf_neuron = rf_token_filtered + offset
        if max_target == "sum":
            rel_c = rel_per_token.sum(dim=1)
        elif max_target == "max":
            rel_c = torch.gather(rel_per_token, 1, rf_token_filtered.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")
        if abs_norm:
            rel_c = rel_c / (rel_c.abs().sum(-1, keepdim=True) + 1e-10)
        d_c_sorted = torch.argsort(rel_c, dim=0, descending=True)
        rel_c_sorted = torch.gather(rel_c, 0, d_c_sorted)
        rf_c_sorted = torch.gather(rf_neuron, 0, d_c_sorted)
        return d_c_sorted, rel_c_sorted, rf_c_sorted


class EmbeddingDimConcept(Concept):
    """Per-embedding-dimension attribution on a 3D ``(B, N, embed_dim)``
    relevance tensor. Same hook sites as :class:`HeadConcept`.

    One concept = one ``embed_dim`` index. The dim's relevance is the
    sum over the (filtered) token axis at that dim. Finer granularity
    than HeadConcept (which sums ``head_dim`` adjacent indices into one
    head): ``num_heads * head_dim = embed_dim`` distinct concept ids.

    Parameters
    ----------
    num_heads : int
        Number of heads. Used only for the convenience decoder
        ``head_id = dim // head_dim`` so callers can label which head a
        dim belongs to. Not used for indexing.
    token_filter : slice, optional
        See :class:`HeadConcept`.

    Concept ids : ``List[int]``. Each id is an absolute dim in
    ``[0, embed_dim)``.
    """

    def __init__(self, num_heads: int, token_filter: slice = slice(None)):
        self.num_heads = int(num_heads)
        self.token_filter = token_filter

    def head_of(self, dim_id: int, embed_dim: int) -> int:
        """Convenience: which head does this dim belong to?
        ``head_id = dim_id // head_dim`` where ``head_dim = embed_dim / num_heads``.
        """
        return int(dim_id) // (embed_dim // self.num_heads)

    def mask(self, batch_id: int, concept_ids: List, layer_name: Optional[str] = None):
        token_filter = self.token_filter
        dim_ids = [int(d) for d in concept_ids]

        def mask_fct(grad: torch.Tensor) -> torch.Tensor:
            t = grad[batch_id]  # (N, embed_dim)
            embed_dim = t.shape[-1]
            for d in dim_ids:
                if not 0 <= d < embed_dim:
                    raise IndexError(f"dim {d} out of [0, {embed_dim})")
            mask = torch.zeros_like(t)
            for d in dim_ids:
                mask[token_filter, d] = 1
            grad[batch_id] = t * mask
            return grad

        return mask_fct

    def attribute(
        self,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        abs_norm: bool = True,
    ) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask
        # (B, N, embed_dim) → filter tokens → sum over tokens → (B, embed_dim).
        rel = relevance[:, self.token_filter, :].sum(dim=1)
        if abs_norm:
            rel = rel / (rel.abs().sum(-1, keepdim=True) + 1e-10)
        return rel

    def reference_sampling(
        self,
        relevance: torch.Tensor,
        layer_name: Optional[str] = None,
        max_target: str = "sum",
        abs_norm: bool = True,
    ):
        # relevance: (B, N, embed_dim). Per-dim, per-token: (B, N_f, embed_dim).
        rel = relevance[:, self.token_filter, :]
        # Argmax over filtered tokens → (B, embed_dim).
        rf_token_filtered = torch.argmax(rel, dim=1)
        offset = self.token_filter.start or 0
        rf_neuron = rf_token_filtered + offset
        if max_target == "sum":
            rel_c = rel.sum(dim=1)
        elif max_target == "max":
            rel_c = torch.gather(rel, 1, rf_token_filtered.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")
        if abs_norm:
            rel_c = rel_c / (rel_c.abs().sum(-1, keepdim=True) + 1e-10)
        d_c_sorted = torch.argsort(rel_c, dim=0, descending=True)
        rel_c_sorted = torch.gather(rel_c, 0, d_c_sorted)
        rf_c_sorted = torch.gather(rf_neuron, 0, d_c_sorted)
        return d_c_sorted, rel_c_sorted, rf_c_sorted


class TokenConcept(Concept):
    """Per-token-position attribution on a 3D ``(B, N, embed_dim)``
    relevance tensor. Hooked at ``proj_drop`` (or any 3D site).

    One concept = one token position. The position's relevance is the
    sum over all ``embed_dim`` at that token. Different granularity from
    :class:`HeadConcept` and :class:`EmbeddingDimConcept`: those select
    on ``embed_dim`` (subspaces), this selects on ``N`` (positions).

    Useful for attributing what each cls / register token contributed,
    or what each spatial patch position contributed.

    Parameters
    ----------
    token_filter : slice, optional
        Universe of token positions to consider. Default ``slice(None)``
        (all tokens). Concept ids index into the positions remaining
        after this filter has been applied — i.e., concept id 0 maps to
        the first position passing the filter.

    Concept ids : ``List[int]``. Each id is a token position index in
    the post-filter axis ``[0, N_filtered)``.
    """

    def __init__(self, token_filter: slice = slice(None)):
        self.token_filter = token_filter

    def mask(self, batch_id: int, concept_ids: List, layer_name: Optional[str] = None):
        token_filter = self.token_filter
        position_ids = [int(p) for p in concept_ids]

        def mask_fct(grad: torch.Tensor) -> torch.Tensor:
            t = grad[batch_id]  # (N, embed_dim)
            mask = torch.zeros_like(t)
            # Apply filter once to get the universe of allowed positions.
            allowed = mask[token_filter]
            for p in position_ids:
                if not 0 <= p < allowed.shape[0]:
                    raise IndexError(
                        f"token position {p} out of [0, {allowed.shape[0]}) "
                        f"under token_filter={token_filter!r}"
                    )
                allowed[p] = 1
            mask[token_filter] = allowed
            grad[batch_id] = t * mask
            return grad

        return mask_fct

    def attribute(
        self,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        abs_norm: bool = True,
    ) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask
        # (B, N, embed_dim) → filter token universe → sum over embed_dim
        # → (B, N_filtered).
        rel = relevance[:, self.token_filter, :].sum(dim=-1)
        if abs_norm:
            rel = rel / (rel.abs().sum(-1, keepdim=True) + 1e-10)
        return rel

    def reference_sampling(
        self,
        relevance: torch.Tensor,
        layer_name: Optional[str] = None,
        max_target: str = "sum",
        abs_norm: bool = True,
    ):
        # relevance: (B, N, embed_dim). Sum embed_dim per (batch, position)
        # and treat positions in the filtered universe as the "concepts".
        rel_per_pos = relevance[:, self.token_filter, :].sum(dim=-1)  # (B, N_f)
        # No "receptive-field" sub-axis — each concept IS one token id.
        # rf_neuron mirrors the absolute token id for the caller's bookkeeping.
        offset = self.token_filter.start or 0
        N_f = rel_per_pos.shape[1]
        rf_neuron = (torch.arange(N_f, device=rel_per_pos.device) + offset).expand_as(rel_per_pos)
        rel_c = rel_per_pos
        if abs_norm:
            rel_c = rel_c / (rel_c.abs().sum(-1, keepdim=True) + 1e-10)
        d_c_sorted = torch.argsort(rel_c, dim=0, descending=True)
        rel_c_sorted = torch.gather(rel_c, 0, d_c_sorted)
        rf_c_sorted = torch.gather(rf_neuron, 0, d_c_sorted)
        return d_c_sorted, rel_c_sorted, rf_c_sorted


__all__ = ["HeadConcept", "EmbeddingDimConcept", "TokenConcept"]
