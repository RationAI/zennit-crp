"""CRP Concept classes for the unfolded Eva attention block.

A Concept here is a CRP-style partition of a hidden tensor into named
"parts" (heads, channels, or prefix-token positions). Each class:

* knows what part-shape it expects on its target tensor
  (``(B, num_heads, N, head_dim)`` for the per-head ones;
  ``(B, N, embed_dim)`` for the post-projection ones),
* takes a ``model`` reference at construction so it can look up the
  parent :class:`crp.attention_unfolded.EvaAttentionUnfolded` of any
  hookable submodule path you pass in,
* exposes ``mask(...)`` / ``attribute(...)`` / ``reference_sampling(...)``
  with an explicit ``layer_name`` argument every call.

**Submodule mapping (read from concept docstrings — not from a class
attribute).** Each concept is the natural attribution lens at one
specific submodule of the unfolded Eva attention:

* :class:`HeadConcept`              → ``...attn.context``
* :class:`QConcept`                 → ``...attn.rope_q``
* :class:`KConcept`                 → ``...attn.rope_k``
* :class:`VConcept`                 → ``...attn.v_id``
* :class:`AttnOutputDimConcept`     → ``...attn.proj_drop`` (spatial tokens only)
* :class:`RegisterTokenConcept`     → ``...attn.proj_drop`` (prefix tokens only)

**No auto-discovery, no path-resolution heuristics.** The user passes
``layer_name`` as a fully qualified path that matches
``model.named_modules()``. For a wrapped model (e.g.
``Probe(backbone=ViT, head=...)``), that means including the wrapper
prefix: ``"backbone.blocks.6.attn.context"``. The walkthrough notebook's
discovery cell prints the actual paths so the user copies them.

If the parent of the path you pass is not an
:class:`EvaAttentionUnfolded` (because the substitution canonizer hasn't
been applied), every method fails loudly with a clear error.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from crp.concepts import ChannelConcept


def _layer_attn(model: nn.Module, layer_name: str) -> nn.Module:
    """Return the parent attention module of ``layer_name``.

    The hookable submodules (``context``, ``rope_q``, ``rope_k``,
    ``v_id``, ``proj_drop``) all live one level under the attention
    block. We trim the last path segment and look up the result via
    :meth:`torch.nn.Module.get_submodule`.

    Returns whichever attention class is currently bound at that path:
    stock ``timm.models.eva.EvaAttention`` when the composite context
    is NOT active, or :class:`crp.attention_unfolded.EvaAttentionUnfolded`
    when it IS active. Both expose ``num_heads``, ``head_dim`` and
    (for Eva-style) ``num_prefix_tokens`` — and that is all the concept
    methods need to read.

    The leaf submodule itself (``context`` etc.) only exists during the
    composite context. That is fine: zennit needs the leaf for
    ``record_layer`` hook registration *during* attribution; the concept
    methods called *after* attribution work off the recorded relevance
    tensor and only need the parent's dim attributes.
    """
    parent_path = layer_name.rsplit(".", 1)[0]
    return model.get_submodule(parent_path)


# ─── 1. Per-head concepts (Q, K, V, Head) ────────────────────────────────────


class _PerHeadAttentionConcept(ChannelConcept):
    """Shared base for concepts on per-head tensors of shape
    ``(B, num_heads, N, head_dim)``.

    Spatial-tokens-only by default: the first ``num_prefix_tokens`` of
    the token axis (cls + register tokens) are zeroed in masks and
    excluded from the spatial sum in :meth:`attribute`. They are
    addressable separately via :class:`RegisterTokenConcept`.

    Construction-time flag:

    * ``dim_split`` — if ``False`` (default), one concept per attention
      head; if ``True``, one concept per ``(head, dim)`` pair.

    Concept id encoding:
    * with ``dim_split=False``: ``int head_id`` or ``(head_id,)``;
      ``attribute()`` returns shape ``(B, num_heads)``.
    * with ``dim_split=True``: ``(head_id, dim_id)`` tuple or flat
      row-major ``int`` over ``(num_heads, head_dim)``;
      ``attribute()`` returns shape ``(B, num_heads, head_dim)``.
    """

    def __init__(self, model: nn.Module, *, dim_split: bool = False):
        self.model = model
        self.dim_split = bool(dim_split)

    # ── concept id decode ───────────────────────────────────────────────────

    def _decode(
        self, concept_id, num_heads: int, head_dim: int,
    ) -> Tuple[int, Optional[int]]:
        """Decode a concept id into ``(head, dim_or_None)``."""
        if isinstance(concept_id, (int, np.integer)):
            flat = int(concept_id)
            if self.dim_split:
                total = num_heads * head_dim
                if not 0 <= flat < total:
                    raise IndexError(
                        f"flat concept index {flat} out of range "
                        f"[0, {total}) for {type(self).__name__}(dim_split=True)"
                    )
                head, dim = divmod(flat, head_dim)
                return head, dim
            if not 0 <= flat < num_heads:
                raise IndexError(
                    f"head index {flat} out of range [0, {num_heads}) for "
                    f"{type(self).__name__}"
                )
            return flat, None
        if isinstance(concept_id, (tuple, list)):
            if self.dim_split:
                if len(concept_id) != 2:
                    raise ValueError(
                        f"{type(self).__name__}(dim_split=True) expects "
                        f"a 2-tuple (head, dim); got {concept_id!r}"
                    )
                head, dim = int(concept_id[0]), int(concept_id[1])
                if not 0 <= head < num_heads:
                    raise IndexError(f"head {head} out of [0, {num_heads})")
                if not 0 <= dim < head_dim:
                    raise IndexError(f"dim {dim} out of [0, {head_dim})")
                return head, dim
            if len(concept_id) != 1:
                raise ValueError(
                    f"{type(self).__name__} expects a 1-tuple (head,); got {concept_id!r}"
                )
            head = int(concept_id[0])
            if not 0 <= head < num_heads:
                raise IndexError(f"head {head} out of [0, {num_heads})")
            return head, None
        raise TypeError(
            f"concept id must be int or tuple; got {type(concept_id).__name__}"
        )

    # ── mask (zero out non-selected concepts AND prefix tokens) ─────────────

    def mask(self, batch_id: int, concept_ids: List, layer_name: Optional[str] = None):
        attn = _layer_attn(self.model, layer_name)
        num_heads, head_dim, npt = attn.num_heads, attn.head_dim, attn.num_prefix_tokens
        decoded = [self._decode(cid, num_heads, head_dim) for cid in concept_ids]

        def mask_fct(grad: torch.Tensor) -> torch.Tensor:
            # grad shape: (B, num_heads, N, head_dim).
            t = grad[batch_id]
            if t.dim() != 3 or t.shape[0] != num_heads or t.shape[2] != head_dim:
                raise ValueError(
                    f"{type(self).__name__} expects per-head tensor of shape "
                    f"(B, {num_heads}, N, {head_dim}) on layer {layer_name!r}; "
                    f"got {tuple(grad.shape)} (per-batch slice {tuple(t.shape)})"
                )
            if t.shape[1] <= npt:
                raise ValueError(
                    f"{type(self).__name__}: token axis has {t.shape[1]} positions "
                    f"but num_prefix_tokens = {npt}; no spatial tokens left"
                )
            mask = torch.zeros_like(t)
            for head, dim in decoded:
                # Spatial tokens only — slice [npt:] on the N axis.
                if dim is None:
                    mask[head, npt:, :] = 1
                else:
                    mask[head, npt:, dim] = 1
            grad[batch_id] = t * mask
            return grad

        return mask_fct

    # ── attribute (sum spatial tokens, per-concept reduction) ───────────────

    def attribute(
        self,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        abs_norm: bool = True,
    ) -> torch.Tensor:
        npt = _layer_attn(self.model, layer_name).num_prefix_tokens
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask
        # (B, num_heads, N, head_dim) → keep only spatial tokens [npt:].
        rel_spatial = relevance[:, :, npt:, :]
        if self.dim_split:
            # Sum over the spatial token axis → (B, num_heads, head_dim).
            rel = rel_spatial.sum(dim=2)
        else:
            # Sum over spatial tokens AND head_dim → (B, num_heads).
            rel = rel_spatial.sum(dim=(2, 3))
        if abs_norm:
            shape = rel.shape
            flat = rel.reshape(shape[0], -1)
            denom = flat.abs().sum(dim=-1, keepdim=True) + 1e-10
            rel = (flat / denom).reshape(shape)
        return rel

    # ── reference_sampling for FeatureVisualization compat ──────────────────

    def reference_sampling(
        self,
        relevance: torch.Tensor,
        layer_name: Optional[str] = None,
        max_target: str = "sum",
        abs_norm: bool = True,
    ):
        # Argmax over the spatial token axis only — receptive-field token id
        # is reported relative to the absolute (B, num_heads, N, head_dim)
        # token axis (so the caller can map it back to the original token
        # coordinates), but the argmax search excludes the prefix tokens.
        npt = _layer_attn(self.model, layer_name).num_prefix_tokens
        B = relevance.shape[0]
        rel_spatial = relevance[:, :, npt:, :]
        if self.dim_split:
            # (B, num_heads, N_spatial, head_dim) → (B, num_heads*head_dim, N_spatial)
            rel = rel_spatial.permute(0, 1, 3, 2).reshape(B, -1, rel_spatial.shape[2])
        else:
            # Sum head_dim → (B, num_heads, N_spatial)
            rel = rel_spatial.sum(dim=-1)
        # rel shape: (B, num_concepts, N_spatial). Argmax on the spatial axis.
        rf_neuron_spatial = torch.argmax(rel, dim=-1)  # (B, num_concepts)
        # Map back to absolute token id (offset by the prefix count).
        rf_neuron = rf_neuron_spatial + npt
        if max_target == "sum":
            rel_c = rel.sum(dim=-1)
        elif max_target == "max":
            rel_c = torch.gather(
                rel, -1, rf_neuron_spatial.unsqueeze(-1),
            ).squeeze(-1)
        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")
        if abs_norm:
            rel_c = rel_c / (rel_c.abs().sum(-1, keepdim=True) + 1e-10)
        d_c_sorted = torch.argsort(rel_c, dim=0, descending=True)
        rel_c_sorted = torch.gather(rel_c, 0, d_c_sorted)
        rf_c_sorted = torch.gather(rf_neuron, 0, d_c_sorted)
        return d_c_sorted, rel_c_sorted, rf_c_sorted


class HeadConcept(_PerHeadAttentionConcept):
    """One concept per attention head; conditioned at the per-head context
    output (= ``attn @ V``, before head-merging reshape).

    **Pass** ``layer_name`` **=** the path to an
    :class:`EvaAttentionUnfolded.context` submodule, e.g.
    ``"blocks.6.attn.context"`` (bare ViT) or
    ``"backbone.blocks.6.attn.context"`` (Probe-wrapped).

    Tensor at the layer: ``(B, num_heads, N, head_dim)``.
    With ``dim_split=True`` the granularity becomes ``(head, dim)``.

    This is the "per-head contribution to the attention sub-output" view
    — the natural CRP analogue of a CNN filter. To condition on a
    specific head's effect on the *residual stream* (post-projection),
    use :class:`AttnOutputDimConcept` instead.
    """


class QConcept(_PerHeadAttentionConcept):
    """One concept per attention head's Q vector (post q_norm + RoPE).

    **Pass** ``layer_name`` **=** the path to ``...attn.rope_q``.
    Tensor: ``(B, num_heads, N, head_dim)``. ``dim_split=True`` selects
    one concept per query feature.
    """


class KConcept(_PerHeadAttentionConcept):
    """One concept per attention head's K vector (post k_norm + RoPE).

    **Pass** ``layer_name`` **=** the path to ``...attn.rope_k``.
    Tensor: ``(B, num_heads, N, head_dim)``. ``dim_split=True`` selects
    one concept per key feature.
    """


class VConcept(_PerHeadAttentionConcept):
    """One concept per attention head's V vector (post per-head reshape).

    **Pass** ``layer_name`` **=** the path to ``...attn.v_id`` (an
    ``nn.Identity`` placed after the per-head reshape so V is hookable;
    V is otherwise reshape-only between the qkv split and the bilinear).
    Tensor: ``(B, num_heads, N, head_dim)``.
    """


# NOTE: ``AttnWeightConcept`` (would target ``attn.softmax``) was
# removed by design. Softmax weights have no fixed semantic per neuron:
# the same (head, query, key) cell combines different concepts for
# different inputs. Useful for inspecting K/Q relations directly via
# tensor recording (``record_layer=['blocks.6.attn.softmax']``) but not
# for identifying concepts via reference-sample retrieval.


# ─── 2. Post-projection concept (residual-stream contribution) ──────────────


class AttnOutputDimConcept(ChannelConcept):
    """One concept per output channel (embed-dim feature) of the post-
    projection attention output (= attention block's contribution to the
    residual stream).

    **Pass** ``layer_name`` **=** the path to ``...attn.proj_drop``.
    Tensor: ``(B, N, embed_dim)``. Concept id forms:

    * ``int channel_id`` — one output channel of ``proj``.
    * ``(channel_id,)`` tuple — same.

    **Token aggregation (not per-token conditioning).** The mask zeros
    all NON-selected channels but keeps every (token, channel) entry of
    the selected channels for the **spatial** token range
    ``[num_prefix_tokens:]``. ``attribute()`` then sums over the spatial
    token axis. Prefix tokens (cls + register) are zeroed; address them
    via :class:`RegisterTokenConcept`.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def mask(self, batch_id: int, concept_ids: List, layer_name: Optional[str] = None):
        attn = _layer_attn(self.model, layer_name)
        embed_dim = int(attn.num_heads) * int(attn.head_dim)
        npt = int(attn.num_prefix_tokens)

        channels = []
        for cid in concept_ids:
            if isinstance(cid, (int, np.integer)):
                ch = int(cid)
            elif isinstance(cid, (tuple, list)) and len(cid) == 1:
                ch = int(cid[0])
            else:
                raise ValueError(
                    f"AttnOutputDimConcept expects int or 1-tuple; got {cid!r}"
                )
            if not 0 <= ch < embed_dim:
                raise IndexError(f"channel {ch} out of [0, {embed_dim})")
            channels.append(ch)

        def mask_fct(grad: torch.Tensor) -> torch.Tensor:
            # grad shape: (B, N, embed_dim). Channel-only conditioning AND
            # spatial-only token slice — register / cls tokens (the first
            # ``npt``) are zeroed out; they're addressable separately via
            # :class:`RegisterTokenConcept`.
            t = grad[batch_id]
            if t.dim() != 2 or t.shape[1] != embed_dim:
                raise ValueError(
                    f"AttnOutputDimConcept expects (B, N, {embed_dim}) on "
                    f"layer {layer_name!r}; got per-batch shape {tuple(t.shape)}"
                )
            if t.shape[0] <= npt:
                raise ValueError(
                    f"AttnOutputDimConcept: token axis has {t.shape[0]} "
                    f"positions but num_prefix_tokens = {npt}; no spatial "
                    f"tokens left"
                )
            mask = torch.zeros_like(t)
            mask[npt:, channels] = 1
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
        npt = int(_layer_attn(self.model, layer_name).num_prefix_tokens)
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask
        # (B, N, embed_dim) → keep spatial tokens only → sum over tokens → (B, embed_dim).
        rel = relevance[:, npt:, :].sum(dim=1)
        if abs_norm:
            denom = rel.abs().sum(dim=-1, keepdim=True) + 1e-10
            rel = rel / denom
        return rel


# ─── 3. Register / cls token concept (prefix-token residual contribution) ───


class RegisterTokenConcept(ChannelConcept):
    """One concept per prefix (cls + register) token of the post-projection
    attention output.

    **Pass** ``layer_name`` **=** the path to ``...attn.proj_drop``.
    Tensor: ``(B, N, embed_dim)``. Concepts live on the first
    ``num_prefix_tokens`` of the token axis.

    DINOv3 ViT-L has ``num_prefix_tokens = 5`` (1 cls + 4 register).
    Each token carries a global, non-spatial signal — register tokens
    absorb high-norm artifacts per Darcet et al. ICLR 2024
    (arXiv:2309.16588), and the cls token is the model's classification
    aggregator.

    Construction-time flag:

    * ``dim_split`` — if ``False`` (default), one concept per prefix
      token (5 concepts on DINOv3); if ``True``, one concept per
      ``(prefix_token_id, channel_id)`` pair.

    Concept id encoding:
    * ``dim_split=False``: ``int token_id`` or ``(token_id,)``;
      ``attribute()`` returns shape ``(B, num_prefix_tokens)``.
    * ``dim_split=True``: ``(token_id, channel_id)`` tuple or flat
      row-major ``int`` over ``(num_prefix_tokens, embed_dim)``;
      ``attribute()`` returns shape ``(B, num_prefix_tokens, embed_dim)``.
    """

    def __init__(self, model: nn.Module, *, dim_split: bool = False):
        self.model = model
        self.dim_split = bool(dim_split)

    def _decode(
        self, concept_id, num_prefix_tokens: int, embed_dim: int,
    ) -> Tuple[int, Optional[int]]:
        """Decode a concept id into ``(token_id, channel_or_None)``."""
        if isinstance(concept_id, (int, np.integer)):
            flat = int(concept_id)
            if self.dim_split:
                total = num_prefix_tokens * embed_dim
                if not 0 <= flat < total:
                    raise IndexError(
                        f"flat concept {flat} out of [0, {total}) for "
                        f"RegisterTokenConcept(dim_split=True)"
                    )
                tok, ch = divmod(flat, embed_dim)
                return tok, ch
            if not 0 <= flat < num_prefix_tokens:
                raise IndexError(
                    f"prefix token {flat} out of [0, {num_prefix_tokens})"
                )
            return flat, None
        if isinstance(concept_id, (tuple, list)):
            if self.dim_split:
                if len(concept_id) != 2:
                    raise ValueError(
                        "RegisterTokenConcept(dim_split=True) expects (token, channel)"
                    )
                tok, ch = int(concept_id[0]), int(concept_id[1])
                if not 0 <= tok < num_prefix_tokens:
                    raise IndexError(f"token {tok} out of [0, {num_prefix_tokens})")
                if not 0 <= ch < embed_dim:
                    raise IndexError(f"channel {ch} out of [0, {embed_dim})")
                return tok, ch
            if len(concept_id) != 1:
                raise ValueError("RegisterTokenConcept expects (token,) or int")
            tok = int(concept_id[0])
            if not 0 <= tok < num_prefix_tokens:
                raise IndexError(f"token {tok} out of [0, {num_prefix_tokens})")
            return tok, None
        raise TypeError(
            f"concept id must be int or tuple; got {type(concept_id).__name__}"
        )

    def mask(self, batch_id: int, concept_ids: List, layer_name: Optional[str] = None):
        attn = _layer_attn(self.model, layer_name)
        embed_dim = int(attn.num_heads) * int(attn.head_dim)
        npt = int(attn.num_prefix_tokens)
        if npt <= 0:
            raise ValueError(
                f"RegisterTokenConcept needs num_prefix_tokens > 0; the "
                f"attention at {layer_name!r} has num_prefix_tokens={npt}."
            )
        decoded = [self._decode(cid, npt, embed_dim) for cid in concept_ids]

        def mask_fct(grad: torch.Tensor) -> torch.Tensor:
            # grad shape: (B, N, embed_dim). Keep selected prefix tokens;
            # zero everything else (other prefix tokens AND all spatial
            # tokens — those are addressable via AttnOutputDimConcept).
            t = grad[batch_id]
            if t.dim() != 2 or t.shape[1] != embed_dim:
                raise ValueError(
                    f"RegisterTokenConcept expects (B, N, {embed_dim}) on "
                    f"layer {layer_name!r}; got per-batch shape {tuple(t.shape)}"
                )
            if t.shape[0] < npt:
                raise ValueError(
                    f"RegisterTokenConcept: token axis has {t.shape[0]} positions "
                    f"but num_prefix_tokens = {npt}"
                )
            mask = torch.zeros_like(t)
            for tok, ch in decoded:
                if ch is None:
                    mask[tok, :] = 1
                else:
                    mask[tok, ch] = 1
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
        npt = int(_layer_attn(self.model, layer_name).num_prefix_tokens)
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask
        # (B, N, embed_dim) → keep prefix tokens only → (B, num_prefix_tokens, embed_dim).
        rel_prefix = relevance[:, :npt, :]
        if not self.dim_split:
            # Sum over channels → (B, num_prefix_tokens).
            rel = rel_prefix.sum(dim=-1)
        else:
            rel = rel_prefix  # (B, num_prefix_tokens, embed_dim)
        if abs_norm:
            shape = rel.shape
            flat = rel.reshape(shape[0], -1)
            denom = flat.abs().sum(dim=-1, keepdim=True) + 1e-10
            rel = (flat / denom).reshape(shape)
        return rel


__all__ = [
    "HeadConcept",
    "QConcept",
    "KConcept",
    "VConcept",
    "AttnOutputDimConcept",
    "RegisterTokenConcept",
]
