"""AttnLRP for vision transformers — Rule / Canonizer / Hook / Composite stack.

The module exposes the building blocks of AttnLRP (Achtibat et al., ICML 2024,
arXiv:2402.05602) as small, single-responsibility classes that compose:

* **Rules** — zennit ``Hook`` subclasses (PascalCase, no suffix, like
  zennit's own ``Epsilon`` / ``Gamma`` / ``Pass``): :class:`AlphaBetaMatmul`
  (bilinear ``q@kᵀ`` / ``weights@v``), :class:`ResidualRatio` (Otsuki
  residual split), :class:`Uniform` (Eq. 7 uniform allocation — LayerScale,
  ``x+pos_embed``), :class:`Identity` (ε-gated elementwise identity). Each is
  assigned to its target module type via a composite ``layer_map``; softmax /
  scale-by-constant identity reuse zennit's stock ``Pass``. ``nn.Linear`` /
  ``nn.Conv2d`` use zennit's ``Epsilon`` / ``Gamma``.

* **Canonizers** (forward-graph rewrites only — the idiomatic job of a
  zennit ``Canonizer``): :class:`LayerNormForwardCanonizer` (stop-gradient on
  the std), :class:`DropoutPassthroughCanonizer`,
  :class:`TimmBlockResidualCanonizer` / :class:`EvaBlockResidualCanonizer`
  (route residual adds + LayerScale muls through ``ResidualAdd`` /
  ``LayerScaleMul`` modules so the rules above attach),
  :class:`VitPosEmbedPALRPCanonizer` (PA-LRP ``x+pos_embed``). Attention
  itself is substituted to its unfolded form by the canonizers in
  :mod:`zennit_ext.attention_unfolded`. :class:`TimmViTCanonizer` bundles the
  forward rewrites; each reverts on ``composite.context()`` exit.

* **Composites** (in :mod:`zennit_ext.attnlrp_composites`):
  :class:`AttnLRPEpsilonComposite`, :class:`AttnLRPGammaComposite`, and
  :class:`AttnLRPCombinedComposite` (the canonical recipe — attention
  substitution + the rule ``layer_map``).

The three ViT concept classes in :mod:`crp.concepts`
(``HeadConcept`` / ``EmbeddingDimConcept`` / ``TokenConcept``) read at the
``q_lrp_probe`` / ``k_lrp_probe`` / ``v_lrp_probe`` / ``proj_drop`` sites
installed by the attention-substitution canonizers in
:mod:`zennit_ext.attention_unfolded`.
"""
# from __future__ import annotations

# from typing import Callable, Optional

# import torch
# import torch.nn as nn
# from torch.autograd import Function

# from timm.layers.pos_embed import resample_abs_pos_embed

# from zennit.canonizers import AttributeCanonizer
# from zennit.composites import LayerMapComposite
# from zennit.core import BasicHook
# from zennit.rules import Epsilon, Gamma, Pass



# ─── 1. Stabilizers ──────────────────────────────────────────────────────────


# ─── 2. Rule autograd Function kernels ───────────────────────────────────────


# def stop_gradient(input):
#     """Detach ``input`` from the autograd graph (CP-LRP variant on normalisations)."""
#     return input.detach()


# ─── 2b. LRP rules as zennit Hook subclasses ─────────────────────────────────
#
# Idiomatic zennit: a rule is a ``Hook``/``BasicHook`` subclass (PascalCase,
# no suffix, cf. ``Epsilon`` / ``Gamma`` / ``Pass``) assigned to a module via a
# composite's ``layer_map``. ``forward`` stashes the tensors the backward needs;
# ``backward(module, grad_input, grad_output)`` returns the modified grad_input
# tuple (the propagated relevance). Softmax / scale-by-constant identity is just
# zennit's stock :class:`~zennit.rules.Pass`.


# ─── 3. Forward-method replacements (installed per-instance by Canonizers) ──


# def layer_norm_forward(self, x):
#     """LayerNorm with stop-gradient on the std. Identity rule on the
#     normalised output (AttnLRP §3.2.2)."""
#     mean = x.mean(dim=-1, keepdim=True)
#     var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
#     std = (var + self.eps).sqrt()
#     y = (x - mean) / stop_gradient(std)
#     if self.weight is not None:
#         y = y * self.weight
#     if self.bias is not None:
#         y = y + self.bias
#     return y


# def dropout_passthrough_forward(self, x):
#     """Disable dropout during attribution (model may be in train mode)."""
#     return x


# def _make_mha_cplrp_forward(original_forward):
#     """Build an ``nn.MultiheadAttention.forward`` replacement that
#     implements CP-LRP by detaching ``query`` and ``key`` before
#     delegating to ``original_forward`` (which is the BOUND method of the
#     target instance, captured at canonizer-apply time).

#     Mirror of :func:`lxt.efficient.patches.cp_multi_head_attention_forward`:
#     once Q and K carry no gradient, the softmax weights downstream are a
#     graph constant, and ``out = softmax(QKᵀ/√d) @ V`` routes
#     ``R_V = weightsᵀ @ R_out`` via standard autograd → CP-LRP value path.

#     `torchvision.models.vit_b_16` calls MHA with ``query is key is value``
#     (self-attention). After ``query.detach()`` and ``key.detach()``, the
#     three are distinct tensor objects, so MHA's ``_in_projection_packed``
#     fast-path is skipped and the chunked ``_in_projection`` path is used:
#     the Q and K row-blocks of ``in_proj_weight`` get no gradient, only
#     the V block does. The Q/K linear projections are LRP-dead exactly as
#     LXT's recipe intends.
#     """
#     def patched(self, query, key, value, *args, **kwargs):
#         return original_forward(query.detach(), key.detach(), value, *args, **kwargs)
#     return patched


# NOTE: ``_eva_attention_forward`` was removed in the unfolding refactor.
# Eva attention is now handled by :class:`zennit_ext.attention_unfolded.EvaAttentionUnfolded`
# + :class:`zennit_ext.attention_unfolded.EvaAttentionSubstitutionCanonizer`. The
# substitution path replaces the entire ``EvaAttention`` module with a
# subgraph of named ``nn.Module`` kernels (``BilinearMatmul``,
# ``SoftmaxAlongLastDim``, ``RotaryEmbedding``, etc.), each owning one
# LRP rule. See ``UNFOLDING_ATTENTION_REFACTOR.md`` and
# ``RESEARCH_NOTES.md`` Entries 4-6.


# NOTE: ``_timm_attention_forward`` was removed in the always-unfold cleanup.
# Standard timm ``Attention`` is now substituted with
# :class:`zennit_ext.attention_unfolded.TimmAttentionUnfolded` (analogous to the
# Eva path). Both attention paths are unified under the unfolded form,
# so concept attribution works on every supported backbone.


# def vit_pos_embed_palrp(self, x):
#     """timm ``VisionTransformer._pos_embed`` replacement implementing PA-LRP.

#     PA-LRP (Bakish et al., NeurIPS 2025; arXiv 2506.02138) treats the
#     additive ``x = x + pos_embed`` step as a bilinear-style operation under
#     the LRP uniform rule (Eq. 7 of AttnLRP) and allocates relevance equally
#     between the two operands. ``pos_embed`` is a learned parameter with no
#     upstream input, so its half is "absorbed"; the remaining half flows
#     back to ``x``. Without this, the additive step is transparent to
#     backward and ``x`` receives the full upstream relevance, double-counting
#     it against ``pos_embed``.

#     Implementation: identical to upstream ``_pos_embed`` (timm 1.0.x —
#     handles ``cls_token``/``reg_token``, ``no_embed_class`` deit-3 variant,
#     ``dynamic_img_size``) except the additive step is routed through a
#     :class:`~zennit_ext.attention_unfolded.PosEmbedAdd` module, which the
#     composite ``layer_map`` gives the :class:`Uniform` (factor-2) rule.
#     """
#     to_cat = []
#     if self.cls_token is not None:
#         to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
#     if self.reg_token is not None:
#         to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

#     if self.pos_embed is None:
#         return torch.cat(to_cat + [x.view(x.shape[0], -1, x.shape[-1])], dim=1)

#     if self.dynamic_img_size:
#         B, H, W, C = x.shape
#         prev_grid_size = self.patch_embed.grid_size
#         pos_embed = resample_abs_pos_embed(
#             self.pos_embed,
#             new_size=(H, W),
#             old_size=prev_grid_size,
#             num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
#         )
#         x = x.view(B, -1, C)
#     else:
#         pos_embed = self.pos_embed

#     if self.no_embed_class:
#         x = self._lrp_posadd(x, pos_embed)
#         if to_cat:
#             x = torch.cat(to_cat + [x], dim=1)
#     else:
#         if to_cat:
#             x = torch.cat(to_cat + [x], dim=1)
#         x = self._lrp_posadd(x, pos_embed)

#     return self.pos_drop(x)


# # ─── 4. Canonizers — one class per kind of model graph mutation ─────────────



# class LayerNormForwardCanonizer(AttributeCanonizer):
#     """Canonizer that swaps ``nn.LayerNorm.forward`` for the AttnLRP
#     stop-gradient-on-std variant (:func:`layer_norm_forward`).

#     AttnLRP §3.2.2 — treats LayerNorm's normalisation as element-wise so
#     relevance flows through the affine output unchanged.

#     Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
#     Transformers', https://proceedings.mlr.press/v235/achtibat24a.html
#     """

#     def __init__(self):
#         super().__init__(self._attribute_map)

#     def _attribute_map(self, _name, module):
#         if not isinstance(module, nn.LayerNorm):
#             return None
#         return _bind_forward(module, layer_norm_forward)

#     def copy(self):
#         return type(self)()


# ─── 5. Hooks — LRP backward for Linear / Conv2d ─────────────────────────────
#
# We use zennit's stock :class:`zennit.rules.Epsilon` and
# :class:`zennit.rules.Gamma` directly. The previous version shipped a
# ``GradientTimesInputBasicHook`` subclass under names ``GTIEpsilon`` /
# ``GTIGamma`` that violated conservation by 100-200% on ordinary inputs
# (audited in ``experiments/audit_gti_hook.py``); those names were
# removed in the unfolding-refactor cleanup. Use ``zennit.rules.Epsilon``
# and ``zennit.rules.Gamma`` directly.

