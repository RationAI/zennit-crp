from timm.models.eva import EvaAttention, EvaBlock
from timm.models.vision_transformer import Attention as TimmAttention, Block as TimmBlock, VisionTransformer
from zennit.canonizers import AttributeCanonizer, Canonizer, CompositeCanonizer

from typing import Callable, List, Sequence

from zennit_extensions.attention_unfolded import EvaAttentionUnfolded, TimmAttentionUnfolded


import torch.nn as nn

from zennit_extensions.attnlrp_rules import LayerNormForwardCanonizer, _bind_forward, _make_mha_cplrp_forward, dropout_passthrough_forward, vit_pos_embed_palrp


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
    attention as named submodules so a composite ``layer_map`` can assign
    rule hooks (e.g. :class:`~zennit_ext.attnlrp_rules.AlphaBetaMatmul`) to
    the new ``BilinearMatmul`` / ``SoftmaxAlongLastDim`` / etc. instances.

    No source paper — infrastructure for unfolded attention (Eva/DINOv3).

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


class TimmAttentionSubstitutionCanonizer(Canonizer):
    """Replace standard timm ``Attention`` instances with :class:`TimmAttentionUnfolded`.

    Mirror of :class:`EvaAttentionSubstitutionCanonizer` for the
    standard timm `vision_transformer.Attention` class. Both canonizers
    are typically bundled into the same composite — each one's
    ``isinstance`` filter skips the other's target so there's no
    coupling.

    No source paper — infrastructure for unfolded attention (standard timm ViT).

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


def _bind_forward(module: nn.Module, fn: Callable, attr: str = "forward") -> dict:
    """Bind ``fn`` as ``attr`` on ``module``'s class — return dict for
    AttributeCanonizer."""
    return {attr: fn.__get__(module, type(module))}



class VitPosEmbedPALRPCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``_pos_embed`` on timm ``VisionTransformer``
    instances to apply the PA-LRP uniform rule (Bakish et al. 2025;
    arXiv:2506.02138). See :func:`vit_pos_embed_palrp`.

    Sourced from PA-LRP (Bakish et al., 2025),
    https://openreview.net/forum?id=bZ0MXXoldX
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, VisionTransformer) or not hasattr(module, "_pos_embed"):
            return None
        from zennit_extensions.attention_unfolded import PosEmbedAdd
        attrs = _bind_forward(module, vit_pos_embed_palrp, attr="_pos_embed")
        attrs["_lrp_posadd"] = PosEmbedAdd()  # x + pos_embed → its own rule via layer_map
        return attrs

    def copy(self):
        return type(self)()


def _eva_block_forward(self, x, rope=None, attn_mask=None, is_causal=False):
    """``EvaBlock.forward`` replacement that routes the residual adds (and,
    optionally, the LayerScale γ multiplications) through ``nn.Module`` instances
    so a composite ``layer_map`` can attach an LRP rule to them.

    The residual *rule* is NOT chosen here — it is selected by mapping
    :class:`~zennit_ext.attention_unfolded.ResidualAdd` to a hook in the
    composite ``layer_map``. The canonizer always installs the same
    ``ResidualAdd`` type.
    """
    # ``self._lrp_res{1,2}`` = ResidualAdd, ``self._lrp_ls{1,2}`` = LayerScaleMul,
    # attached by :class:`EvaBlockResidualCanonizer`; the composite ``layer_map``
    # assigns each its LRP rule as a zennit Hook.
    attn_branch = self.attn(
        self.norm1(x), rope=rope, attn_mask=attn_mask, is_causal=is_causal,
    )
    if self.gamma_1 is not None:
        attn_branch = (
            self._lrp_ls1(attn_branch) if hasattr(self, "_lrp_ls1")
            else self.gamma_1 * attn_branch
        )
    branch1 = self.drop_path1(attn_branch)
    x = self._lrp_res1(x, branch1)

    mlp_branch = self.mlp(self.norm2(x))
    if self.gamma_2 is not None:
        mlp_branch = (
            self._lrp_ls2(mlp_branch) if hasattr(self, "_lrp_ls2")
            else self.gamma_2 * mlp_branch
        )
    branch2 = self.drop_path2(mlp_branch)
    x = self._lrp_res2(x, branch2)
    return x


class EvaBlockResidualCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``forward`` on timm ``eva.EvaBlock`` (DINOv3 etc.)
    for the residual-LRP variant. Handles the optional
    ``gamma_1``/``gamma_2`` LayerScale parameters and (optionally) wraps
    them under the AttnLRP uniform rule.

    Parameters
    ----------
    layerscale_uniform : bool
        When True, route the LayerScale γ multiplications through
        :class:`~zennit_ext.attention_unfolded.LayerScaleMul` modules, which
        the composite ``layer_map`` gives the :class:`Uniform` (factor-2)
        rule (AttnLRP Eq. 7, treating γ as a constant operand). Default False.

    The residual *rule* is chosen in the composite ``layer_map`` (map
    ``ResidualAdd`` to a hook), not here — this canonizer always installs the
    single ``ResidualAdd`` type.

    No source paper — infrastructure for hookable residual adds (Eva/DINOv3).
    """

    def __init__(self, *, layerscale_uniform: bool = False):
        self.layerscale_uniform = layerscale_uniform
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, EvaBlock):
            return None
        needed = ("drop_path1", "drop_path2", "norm1", "norm2", "attn", "mlp",
                  "gamma_1", "gamma_2")
        if not all(hasattr(module, a) for a in needed):
            return None
        from zennit_extensions.attention_unfolded import ResidualAdd, LayerScaleMul

        def fwd(self, x, rope=None, attn_mask=None, is_causal=False):
            return _eva_block_forward(
                self, x, rope=rope, attn_mask=attn_mask, is_causal=is_causal,
            )

        attrs = _bind_forward(module, fwd)
        attrs["_lrp_res1"] = ResidualAdd()
        attrs["_lrp_res2"] = ResidualAdd()
        if self.layerscale_uniform:
            # LayerScale γ·branch → uniform rule (γ a param, absorbs half).
            if module.gamma_1 is not None:
                attrs["_lrp_ls1"] = LayerScaleMul(module.gamma_1)
            if module.gamma_2 is not None:
                attrs["_lrp_ls2"] = LayerScaleMul(module.gamma_2)
        return attrs

    def copy(self):
        return type(self)(layerscale_uniform=self.layerscale_uniform)


def _timm_block_forward(self, x, attn_mask=None, is_causal=False):
    """timm ``Block.forward`` replacement that routes the two residual adds
    through ``self._lrp_res{1,2}`` (:class:`~zennit_ext.attention_unfolded.ResidualAdd`)
    so a composite ``layer_map`` can attach the chosen residual rule. The rule is
    a ``layer_map`` decision, not a property of this module. LayerScale on standard
    timm Blocks is its own ``nn.Module`` (``ls1``/``ls2``), handled transparently.
    """
    branch1 = self.drop_path1(
        self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal))
    )
    x = self._lrp_res1(x, branch1)
    branch2 = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    x = self._lrp_res2(x, branch2)
    return x


class TimmBlockResidualCanonizer(AttributeCanonizer):
    """Canonizer that swaps ``forward`` on timm ``vision_transformer.Block`` so
    each residual addition routes through a
    :class:`~zennit_ext.attention_unfolded.ResidualAdd` module — making the add
    hookable. The residual *rule* (ratio / symmetric / L1 / none) is then chosen
    in the composite ``layer_map`` by mapping ``ResidualAdd`` to a hook; this
    canonizer installs only the (single) module type and takes no rule argument.

    No source paper — infrastructure for hookable residual adds.
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, TimmBlock):
            return None
        needed = ("ls1", "ls2", "drop_path1", "drop_path2", "norm1",
                  "norm2", "attn", "mlp")
        if not all(hasattr(module, a) for a in needed):
            return None
        from zennit_extensions.attention_unfolded import ResidualAdd

        def fwd(self, x, attn_mask=None, is_causal=False):
            return _timm_block_forward(
                self, x, attn_mask=attn_mask, is_causal=is_causal,
            )

        attrs = _bind_forward(module, fwd)
        attrs["_lrp_res1"] = ResidualAdd()
        attrs["_lrp_res2"] = ResidualAdd()
        return attrs

    def copy(self):
        return type(self)()


class DropoutPassthroughCanonizer(AttributeCanonizer):
    """Canonizer that disables ``nn.Dropout`` during attribution (model may
    be in train mode).

    No source paper — infrastructure for disabling dropout during attribution.
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, nn.Dropout):
            return None
        return _bind_forward(module, dropout_passthrough_forward)

    def copy(self):
        return type(self)()


class TimmViTCanonizer2(CompositeCanonizer):
    """Aggregator: bundles the per-module canonizers needed for AttnLRP on
    a timm ViT (standard or Eva-stack).

    Combines:

    * :class:`LayerNormForwardCanonizer` — LayerNorm with stop-gradient(std).
    * :class:`DropoutPassthroughCanonizer` — disable Dropout for backward.
    * :class:`VitPosEmbedPALRPCanonizer` (when ``palrp=True``) — PA-LRP on
      the ``x + pos_embed`` step (Bakish et al. 2025).
    * :class:`TimmBlockResidualCanonizer` and
      :class:`EvaBlockResidualCanonizer` (when ``residual`` is True) — route the
      residual adds through ``ResidualAdd`` so the composite ``layer_map`` can
      apply a residual rule. The rule itself is a ``layer_map`` choice.

    Attention itself is NOT handled here. It is substituted to its
    unfolded form (:class:`TimmAttentionUnfolded` /
    :class:`EvaAttentionUnfolded`) by their respective substitution
    canonizers, applied automatically by
    :class:`AttnLRPCombinedComposite`.

    All mutations are instance-level and reversible (revert on
    ``composite.context()`` exit).

    No source paper — infrastructure aggregator bundling the per-module
    canonizers.

    Parameters
    ----------
    palrp : bool
        Enable PA-LRP on the absolute pos_embed addition. Only relevant
        for ViTs with ``self.pos_embed`` (vit_base etc.); no-op for
        DINOv3 (RoPE only).
    residual : bool
        Route the block residual adds through ``ResidualAdd`` modules so the
        composite ``layer_map`` can apply a residual rule. The rule itself
        (ratio / symmetric / L1 / none) is selected in the ``layer_map``, not
        here.
    layerscale_uniform : bool
        Apply the uniform allocation rule to LayerScale γ
        multiplications (CaiT / Eva blocks only).
    epsilon : float
        ε for ε-stabilised rules.
    """

    def __init__(
        self,
        *,
        palrp: bool = False,
        residual: bool = False,
        layerscale_uniform: bool = False,
        epsilon: float = 1e-6,
    ):
        canonizers: List[Canonizer] = [
            LayerNormForwardCanonizer(),
            DropoutPassthroughCanonizer(),
        ]
        if palrp:
            canonizers.append(VitPosEmbedPALRPCanonizer())
        if residual:
            canonizers.append(TimmBlockResidualCanonizer())
            canonizers.append(EvaBlockResidualCanonizer(
                layerscale_uniform=layerscale_uniform,
            ))
        elif layerscale_uniform:
            # LayerScale-uniform wants the EvaBlock forward installed (the
            # wrapper lives inside that forward) even without a residual rule.
            canonizers.append(EvaBlockResidualCanonizer(layerscale_uniform=True))
        super().__init__(canonizers)


class TorchvisionMHACPLRPCanonizer(AttributeCanonizer):
    """Canonizer that installs CP-LRP on ``torch.nn.MultiheadAttention``.

    Drop-in zennit replacement for LXT's instance-level patch in
    :func:`lxt.efficient.patches.cp_multi_head_attention_forward`. Each
    matched MHA instance gets its forward rebound to a wrapper that
    calls ``query.detach()`` and ``key.detach()`` before delegating to
    the original forward — so Q,K paths carry no gradient and only the
    value path receives relevance.

    Targets ``nn.MultiheadAttention`` (any instance). The intended use
    is on `torchvision.models.vision_transformer` ViTs (vit_b_16,
    vit_l_16, etc.); other models using ``nn.MultiheadAttention`` are
    canonized identically.

    Combine with :class:`LayerNormForwardCanonizer` and
    :class:`DropoutPassthroughCanonizer` (and ``nn.GELU`` → ``Pass`` in the
    composite ``layer_map``) to obtain the zennit-side equivalent of
    LXT-efficient's published vision-transformer recipe
    (``lxt.efficient.models.vit_torch.cp_LRP``).

    Sourced from 'XAI for Transformers: Better Explanations through Conservative
    Propagation', https://proceedings.mlr.press/v162/ali22a.html
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, nn.MultiheadAttention):
            return None
        # Capture the BOUND original forward so the closure has self pre-baked.
        original_forward = module.forward
        patched = _make_mha_cplrp_forward(original_forward)
        return _bind_forward(module, patched)

    def copy(self):
        return type(self)()


