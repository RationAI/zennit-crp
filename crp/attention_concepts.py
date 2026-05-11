"""Concept classes for conditional attribution on unfolded ViT attention.

This module defines the *axes of conditioning* exposed by the unfolded
attention refactor. Each concept class is a thin wrapper around a CRP
``ChannelConcept`` that:

* knows which **named submodule** of :class:`crp.attention_unfolded.EvaAttentionUnfolded`
  to target (its ``LAYER_SUFFIX``),
* knows the **shape convention** of that submodule's tensor,
* implements ``mask`` (zero-out non-selected concepts during attribution
  backward) and ``attribute`` (per-concept relevance reduction).

**Strict separation between spatial and prefix (cls + register) tokens.**
Spatial patch tokens are translation-equivariant: the same channel
responds similarly to the same feature regardless of position, so
per-token conditioning of spatial tokens would not generalise across
reference samples. Register / cls tokens (the first ``num_prefix_tokens``
positions of the token axis) carry global non-spatial meaning per
Darcet et al. ICLR 2024 (arXiv:2309.16588) and ARE meaningfully
addressable by token id. The two are exposed via separate concept
classes; **no concept class mixes them**.

Conditioning points inside one attention block:

1. **Q / K / V inputs to the bilinears (spatial-only)** —
   :class:`QConcept`, :class:`KConcept`, :class:`VConcept`. Hook the
   per-head Q (post q_norm + RoPE), K (post k_norm + RoPE), or V (post
   per-head reshape) tensors of shape ``(B, num_heads, N, head_dim)``.
   Per-head conditioning by default; per-(head, dim) via
   ``dim_split=True``. Excludes the first ``num_prefix_tokens`` of the
   token axis.

2. **Per-head context output (spatial-only)** — :class:`HeadConcept`.
   Hooks the ``context`` output (= ``attn @ V``, before the reshape
   that merges heads back) of shape ``(B, num_heads, N, head_dim)``.
   "Per-head contribution to the attention output" — natural CRP
   analogue to a CNN filter. Spatial tokens only.

3. **Post-projection residual contribution, spatial channels** —
   :class:`AttnOutputDimConcept`. Hooks the ``proj_drop`` output of
   shape ``(B, N, embed_dim)``. **Per-channel conditioning only** with
   relevance aggregated over spatial tokens. The
   "OV-circuit-output" view used by the Anthropic mathematical-framework
   / induction-heads line of work. Excludes prefix tokens.

4. **Register / cls token contribution** — :class:`RegisterTokenConcept`.
   Same hook point (``proj_drop``) but addresses **only the first
   ``num_prefix_tokens``** of the token axis. Per-token conditioning by
   default; per-(token, channel) via ``dim_split=True``. Each prefix
   token carries a distinct global signal (cls = classification
   aggregator; register tokens absorb high-norm artifacts) and is
   meaningfully retrievable via reference-sample search.

**Removed concept classes** (per design review):

* ``AttnWeightConcept`` (would have hooked ``attn.softmax``) — softmax
  weights have no fixed semantic per neuron: the same (head, query, key)
  cell combines different concepts for different inputs. Useful for
  inspecting K/Q relations directly (via tensor recording) but not for
  identifying concepts via reference-sample retrieval. Not exposed as a
  concept; the named submodule remains hookable for inspection.

All concepts auto-discover their target submodules from the model on
construction (``concept = HeadConcept(model)``), or layers can be
registered manually with ``concept.register_layer(name, num_heads,
head_dim)``.

Naming convention for layer paths: a model with attention modules at
``blocks.0.attn``, …, ``blocks.23.attn`` (substituted by
:class:`crp.attention_unfolded.EvaAttentionSubstitutionCanonizer`) will
expose hookable submodules like ``blocks.6.attn.context``,
``blocks.6.attn.rope_q``, ``blocks.6.attn.proj_drop``, etc. The concept
classes below build the full path as ``f"{attn_name}.{LAYER_SUFFIX}"``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from crp.concepts import ChannelConcept


def _find_unfolded_attentions(model: nn.Module):
    """Yield ``(name, EvaAttentionUnfolded_instance)`` for every unfolded
    attention submodule in ``model``. Skips modules that haven't been
    substituted (so concepts are silently no-op on a non-substituted
    model — call ``register_layer`` explicitly in that case)."""
    # Lazy import to avoid a hard cycle: attention_concepts → attention_unfolded
    # → transformer_patches → attention_concepts (via crp/__init__.py).
    from crp.attention_unfolded import EvaAttentionUnfolded
    for name, module in model.named_modules():
        if isinstance(module, EvaAttentionUnfolded):
            yield name, module


def _resolve_dims(layer_name: str, dims: Dict[str, tuple]) -> tuple:
    """Return registered dims for ``layer_name``, with two fallbacks:

    1. **Parent-prefix**: if ``blocks.6.attn.context.subthing`` is asked
       for, fall back to a registration on ``blocks.6.attn.context``.
    2. **Wrapper-prefix**: if ``blocks.6.attn.context`` is asked for and
       the registration was made under a wrapped model (e.g. ``Probe``)
       so the actual key is ``backbone.blocks.6.attn.context``, find the
       unique key that ends with ``.layer_name`` and return its dims.
       Raises if multiple keys match (ambiguous).

    The wrapper-prefix fallback is what makes notebook code that
    constructs paths as ``blocks.{i}.attn.{suffix}`` work even when the
    model has been wrapped in a ``Probe``-style container that prefixes
    every module path with ``backbone.``.
    """
    if layer_name in dims:
        return dims[layer_name]
    # 1. Parent-prefix
    parts = layer_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:i])
        if parent in dims:
            return dims[parent]
    # 2. Wrapper-prefix (suffix match)
    needle = "." + layer_name
    matches = [k for k in dims if k.endswith(needle)]
    if len(matches) == 1:
        return dims[matches[0]]
    if len(matches) > 1:
        raise ValueError(
            f"Layer name {layer_name!r} is ambiguous — multiple keys end "
            f"with it: {matches}. Pass the full path."
        )
    raise ValueError(
        f"No attention dims registered for {layer_name!r}. Pass the model "
        "to the concept constructor, call register_layer(...), or check "
        "that the model's attention modules have been substituted by "
        "EvaAttentionSubstitutionCanonizer."
    )


# ─── 1. Per-head concepts (Q, K, V, Head) ────────────────────────────────────


class _PerHeadAttentionConcept(ChannelConcept):
    """Shared base for concepts on per-head tensors of shape
    ``(B, num_heads, N, head_dim)``.

    **Spatial-only conditioning.** All operations exclude the first
    ``num_prefix_tokens`` of the token axis (cls + register tokens in
    DINOv3 etc.) — those carry global non-spatial meaning that must
    not be mixed with patch-token conditioning. Register / cls tokens
    are addressable separately via :class:`RegisterTokenConcept`.

    Subclasses set:

    * ``LAYER_SUFFIX`` — the named submodule of
      :class:`EvaAttentionUnfolded` to target (e.g. ``"context"``,
      ``"rope_q"``, ``"v_id"``).

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

    LAYER_SUFFIX: str = ""  # subclass override

    def __init__(self, model: Optional[nn.Module] = None, *, dim_split: bool = False):
        self.dim_split = bool(dim_split)
        # layer_name -> (num_heads, head_dim, num_prefix_tokens)
        self._dims: Dict[str, Tuple[int, int, int]] = {}
        if model is not None:
            for attn_name, attn in _find_unfolded_attentions(model):
                self._dims[f"{attn_name}.{self.LAYER_SUFFIX}"] = (
                    int(attn.num_heads),
                    int(attn.head_dim),
                    int(getattr(attn, "num_prefix_tokens", 0)),
                )

    def register_layer(
        self, layer_name: str, num_heads: int, head_dim: int,
        num_prefix_tokens: int = 0,
    ) -> None:
        self._dims[layer_name] = (
            int(num_heads), int(head_dim), int(num_prefix_tokens),
        )

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
        num_heads, head_dim, npt = _resolve_dims(layer_name, self._dims)
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
        _, _, npt = _resolve_dims(layer_name, self._dims)
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
        _, _, npt = _resolve_dims(layer_name, self._dims)
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
    """One concept per attention head, conditioned at the per-head context
    output (= ``attn @ V``, before head-merging reshape).

    Targets ``EvaAttentionUnfolded.context``. Tensor shape:
    ``(B, num_heads, N, head_dim)``.

    With ``dim_split=True`` the granularity becomes ``(head, dim)``.

    This is the "per-head contribution to the attention sub-output"
    view — the natural CRP analogue of a CNN filter. To condition on a
    specific head's effect on the *residual stream* (post-projection),
    use :class:`AttnOutputConcept` instead.
    """

    LAYER_SUFFIX = "context"


class QConcept(_PerHeadAttentionConcept):
    """One concept per attention head's Q vector (post q_norm + RoPE).

    Targets ``EvaAttentionUnfolded.rope_q``. Tensor shape:
    ``(B, num_heads, N, head_dim)``.

    With ``dim_split=True`` the granularity becomes ``(head, dim)`` —
    one concept per query feature.
    """

    LAYER_SUFFIX = "rope_q"


class KConcept(_PerHeadAttentionConcept):
    """One concept per attention head's K vector (post k_norm + RoPE).

    Targets ``EvaAttentionUnfolded.rope_k``. Tensor shape:
    ``(B, num_heads, N, head_dim)``. ``dim_split=True`` selects per
    key feature.
    """

    LAYER_SUFFIX = "rope_k"


class VConcept(_PerHeadAttentionConcept):
    """One concept per attention head's V vector (post per-head reshape).

    Targets ``EvaAttentionUnfolded.v_id`` (an ``nn.Identity`` placed
    after the per-head reshape so V is hookable; V is otherwise
    reshape-only between the qkv split and the bilinear). Tensor
    shape: ``(B, num_heads, N, head_dim)``.
    """

    LAYER_SUFFIX = "v_id"


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

    Targets ``EvaAttentionUnfolded.proj_drop``. Tensor shape:
    ``(B, N, embed_dim)``. Concept id forms:

    * ``int channel_id`` — one output channel of ``proj``.
    * ``(channel_id,)`` tuple — same.

    This is the "OV-circuit-output" view used in the Anthropic
    mathematical-framework / induction-heads line of work
    (Elhage et al., Olsson et al.). Conditioning on a single channel
    asks: "which feature in the attention block's residual contribution
    is most relevant to the prediction?". Different from
    :class:`HeadConcept` (pre-proj per-head) because ``proj`` mixes head
    outputs into the residual-stream basis — channels here are in the
    *embed_dim* coordinates, not per-head coordinates.

    **Token aggregation (not per-token conditioning).** The mask zeros
    all NON-selected channels but keeps every (token, channel) entry of
    the selected channels. ``attribute()`` then sums over the token
    axis. This is intentional: ViT spatial token positions carry no
    fixed semantic — the same channel responds similarly to the same
    feature regardless of where it appears in the image. Per-token
    conditioning would treat the same concept differently based on
    accidental spatial location and would not generalise across
    reference samples; channel-only conditioning + spatial aggregation
    is the meaningful axis.

    *Register / cls tokens (DINOv3's first 5 prefix tokens).* Currently
    lumped with patch tokens in the spatial sum. They DO carry global
    non-spatial meaning per Darcet et al. ICLR 2024 (arXiv:2309.16588)
    and could in principle be conditioned on by token-id, but the right
    interpretive framing is open research. The channel-only baseline
    here applies uniformly to all tokens; refining the register-token
    treatment is tracked as future work.
    """

    LAYER_SUFFIX = "proj_drop"

    def __init__(self, model: Optional[nn.Module] = None):
        # layer_name -> (embed_dim, num_prefix_tokens)
        self._dims: Dict[str, Tuple[int, int]] = {}
        if model is not None:
            for attn_name, attn in _find_unfolded_attentions(model):
                embed_dim = int(attn.num_heads) * int(attn.head_dim)
                self._dims[f"{attn_name}.{self.LAYER_SUFFIX}"] = (
                    embed_dim, int(getattr(attn, "num_prefix_tokens", 0)),
                )

    def register_layer(
        self, layer_name: str, embed_dim: int, num_prefix_tokens: int = 0,
    ) -> None:
        self._dims[layer_name] = (int(embed_dim), int(num_prefix_tokens))

    def mask(self, batch_id: int, concept_ids: List, layer_name: Optional[str] = None):
        embed_dim, npt = _resolve_dims(layer_name, self._dims)

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
        _, npt = _resolve_dims(layer_name, self._dims)
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

    Targets ``EvaAttentionUnfolded.proj_drop``. Tensor shape:
    ``(B, N, embed_dim)``. Concepts live on the first ``num_prefix_tokens``
    of the token axis.

    DINOv3 ViT-L has ``num_prefix_tokens = 5`` (1 cls + 4 register).
    Each token carries a global, non-spatial signal — register tokens
    absorb high-norm artifacts per Darcet et al. ICLR 2024
    (arXiv:2309.16588), and the cls token is the model's classification
    aggregator. Per-token addressing makes sense at this granularity:
    each prefix token can encode different global features and is
    meaningfully retrievable via reference-sample search (a different
    contract from spatial patch tokens, which are translation-equivariant).

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

    Use cases:

    * Find images where the cls token (``token_id=0``) carries the most
      relevance — the natural classification-aggregator-attention view.
    * Find images that activate a specific register token most — useful
      for the artifact-storage hypothesis of Darcet et al.
    """

    LAYER_SUFFIX = "proj_drop"

    def __init__(self, model: Optional[nn.Module] = None, *, dim_split: bool = False):
        self.dim_split = bool(dim_split)
        # layer_name -> (embed_dim, num_prefix_tokens)
        self._dims: Dict[str, Tuple[int, int]] = {}
        if model is not None:
            for attn_name, attn in _find_unfolded_attentions(model):
                embed_dim = int(attn.num_heads) * int(attn.head_dim)
                self._dims[f"{attn_name}.{self.LAYER_SUFFIX}"] = (
                    embed_dim, int(getattr(attn, "num_prefix_tokens", 0)),
                )

    def register_layer(
        self, layer_name: str, embed_dim: int, num_prefix_tokens: int,
    ) -> None:
        if num_prefix_tokens <= 0:
            raise ValueError(
                f"RegisterTokenConcept needs num_prefix_tokens > 0; got "
                f"{num_prefix_tokens}"
            )
        self._dims[layer_name] = (int(embed_dim), int(num_prefix_tokens))

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
        embed_dim, npt = _resolve_dims(layer_name, self._dims)
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
        _, npt = _resolve_dims(layer_name, self._dims)
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
