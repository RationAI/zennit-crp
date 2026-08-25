from __future__ import annotations

import torch
import numpy as np

from typing import List, Dict, Optional


def _filter_positions(token_filter: slice, n_tokens: int, device) -> torch.Tensor:
    """Absolute indices of the tokens selected by ``token_filter``.

    Handles arbitrary steps and negative bounds, unlike ``start + offset``
    arithmetic, which silently assumes contiguous step-1 slices.
    """
    return torch.arange(n_tokens, device=device)[token_filter]


class Concept:
    """Abstract base for concept attribution."""

    def mask(self, batch_id, concept_ids, layer_name):

        raise NotImplementedError("'Concept'class must be implemented!")

    def mask_rf(self, neuron_ids, layer_name):

        raise NotImplementedError("'Concept'class must be implemented!")

    def reference_sampling(self, relevance, layer_name: str = None, max_target: str = "sum", abs_norm=True):

        raise NotImplementedError("'Concept'class must be implemented!")

    def get_rf_indices(self, output_shape, layer_name):

        raise NotImplementedError("'Concept'class must be implemented!")

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        raise NotImplementedError("'Concept'class must be implemented!")


class ChannelConcept(Concept):
    """Channel-wise concept for ``Conv2d`` / ``Linear`` layers."""

    @staticmethod
    def mask(batch_id: int, concept_ids: List, layer_name=None):
        """Build a gradient hook that zeros every channel except ``concept_ids``.

        Parameters
        ----------
        batch_id : int
            Index of the sample in the batch to mask.
        concept_ids : list of int
            Channel indices to keep.
        """

        def mask_fct(grad):

            mask = torch.zeros_like(grad[batch_id])
            mask[concept_ids] = 1
            grad[batch_id] = grad[batch_id] * mask

            return grad

        return mask_fct

    @staticmethod
    def mask_rf(batch_id: int, c_n_map: Dict[int, List], layer_name=None):
        """Build a gradient hook that keeps selected neurons within selected channels.

        Parameters
        ----------
        batch_id : int
            Index of the sample in the batch to mask.
        c_n_map : dict[int, list[int]]
            ``{channel: [neuron_indices]}``. Neuron indices address a channel's
            spatial grid flattened to 1D (e.g. a ``[3, 20, 20]`` channel becomes
            ``[3, 400]`` with indices ``0..399``).
        """

        def mask_fct(grad):

            grad_shape = grad.shape
            grad = grad.view(*grad_shape[:2], -1)

            mask = torch.zeros_like(grad[batch_id])

            for channel in c_n_map:
            
                mask[channel, c_n_map[channel]] = 1

            grad[batch_id] = grad[batch_id] * mask
            return grad.view(grad_shape)

        return mask_fct

    def get_rf_indices(self, output_shape, layer_name=None):

        if len(output_shape) == 1:
            return [0]
        else:
            end = np.prod(output_shape[1:])
            return np.arange(0, end)

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        return rel_l

    def reference_sampling(self, relevance, layer_name: str = None, max_target: str = "sum", abs_norm=True):
        """
        Parameters:
            max_target: str. Either 'sum' or 'max'.
        """

        # position of receptive field neuron
        rel_l = relevance.view(*relevance.shape[:2], -1)
        rf_neuron = torch.argmax(rel_l, dim=-1)

        # channel maximization target
        if max_target == "sum":
            rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        elif max_target == "max":
            rel_l = torch.gather(rel_l, -1, rf_neuron.unsqueeze(-1)).squeeze(-1)

        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)
        
        d_ch_sorted = torch.argsort(rel_l, dim=0, descending=True)
        rel_ch_sorted = torch.gather(rel_l, 0, d_ch_sorted)
        rf_ch_sorted = torch.gather(rf_neuron, 0, d_ch_sorted)

        return d_ch_sorted, rel_ch_sorted, rf_ch_sorted


class HeadConcept(Concept):
    """Per-head attribution on a 3D ``(B, N, embed_dim)`` relevance tensor.

    Slices ``embed_dim`` into ``num_heads`` contiguous segments of size
    ``head_dim = embed_dim / num_heads``. One concept = one head; its
    relevance is the sum over the head's ``embed_dim`` slice and the
    filtered token axis.

    Parameters
    ----------
    num_heads : int
        Number of attention heads. Required because ``embed_dim`` alone does
        not fix the per-head split.
    token_filter : slice, optional
        Restrict the token axis before aggregating. Default ``slice(None)``
        (all tokens). Applied at both ``mask`` time (zeroes excluded
        positions) and ``attribute`` time (excludes them from the per-head
        sum). Callers must know the model's prefix layout to pick a
        meaningful slice (e.g. ``slice(0, 1)`` for cls,
        ``slice(num_prefix, None)`` for patches).

    Concept ids : ``List[int]``, each a head index in ``[0, num_heads)``.
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
        # Map back to absolute token ids via the slice's selected positions.
        positions = _filter_positions(self.token_filter, relevance.shape[1], rel.device)
        rf_neuron = positions[rf_token_filtered]
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

    One concept = one ``embed_dim`` index; its relevance is the sum over
    the filtered token axis. Finer-grained than :class:`HeadConcept`, which
    sums ``head_dim`` adjacent dims per head.

    Parameters
    ----------
    num_heads : int
        Used only by :meth:`head_of` to label which head a dim belongs to;
        not used for indexing.
    token_filter : slice, optional
        See :class:`HeadConcept`.

    Concept ids : ``List[int]``, each an absolute dim in ``[0, embed_dim)``.
    """

    def __init__(self, num_heads: int, token_filter: slice = slice(None)):
        self.num_heads = int(num_heads)
        self.token_filter = token_filter

    def head_of(self, dim_id: int, embed_dim: int) -> int:
        """Which head owns ``dim_id``? Returns ``dim_id // head_dim``
        (``head_dim = embed_dim / num_heads``)."""
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
        # Map back to absolute token ids via the slice's selected positions.
        positions = _filter_positions(self.token_filter, relevance.shape[1], rel.device)
        rf_neuron = positions[rf_token_filtered]
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

    One concept = one token position; its relevance is the sum over
    ``embed_dim`` at that token. Selects on ``N`` (positions), unlike
    :class:`HeadConcept` / :class:`EmbeddingDimConcept` which select on
    ``embed_dim``.

    Parameters
    ----------
    token_filter : slice, optional
        Universe of token positions to consider. Default ``slice(None)``
        (all tokens). Concept ids index the positions remaining after the
        filter (id 0 = first surviving position).

    Concept ids : ``List[int]``, each a position in ``[0, N_filtered)``.
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
        positions = _filter_positions(self.token_filter, relevance.shape[1], rel_per_pos.device)
        rf_neuron = positions.expand_as(rel_per_pos)
        rel_c = rel_per_pos
        if abs_norm:
            rel_c = rel_c / (rel_c.abs().sum(-1, keepdim=True) + 1e-10)
        d_c_sorted = torch.argsort(rel_c, dim=0, descending=True)
        rel_c_sorted = torch.gather(rel_c, 0, d_c_sorted)
        rf_c_sorted = torch.gather(rf_neuron, 0, d_c_sorted)
        return d_c_sorted, rel_c_sorted, rf_c_sorted


__all__ = ["Concept", "ChannelConcept", "HeadConcept", "EmbeddingDimConcept", "TokenConcept"]
