from typing import Callable, List, Optional, Sequence

import torch.nn as nn
from timm.models.eva import EvaAttention, EvaBlock
from timm.models.vision_transformer import Attention as TimmAttention, Block as TimmBlock
from zennit.canonizers import AttributeCanonizer, Canonizer

from zennit_extensions.attention_unfolded import EvaAttentionUnfolded, TimmAttentionUnfolded


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

    ``apply(root)`` substitutes every ``EvaAttention`` whose block index is
    in ``block_indices`` (all, if None); ``remove()`` re-binds the original.
    Weight-sharing means no parameter state is lost. No LRP-rule decisions —
    it only exposes named submodules so a composite ``layer_map`` can assign
    rule hooks to them.

    Parameters
    ----------
    block_indices : tuple[int, ...] | None
        Blocks to substitute; ``None`` (default) substitutes all.
    rope_detach : bool
        Forwarded to :class:`EvaAttentionUnfolded` → :class:`RotaryEmbedding`.
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
        # Signature intentionally differs from ``Canonizer.register(self)`` —
        # zennit's lifecycle dispatches through ``apply()`` (the ABC enforces
        # the method name, not its signature), and zennit's own canonizers
        # do the same.
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


class VanillaViTAttentionSubstitutionCanonizer(Canonizer):
    """Replace standard timm ``Attention`` instances with :class:`TimmAttentionUnfolded`.
    Mirror of :class:`EvaAttentionSubstitutionCanonizer`; each canonizer's
    ``isinstance`` filter skips the other's target, so both can be bundled.

    Parameters
    ----------
    block_indices : tuple[int, ...] | None
        Blocks to substitute; ``None`` (default) substitutes all.
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

    def apply(self, root_module: nn.Module) -> List["VanillaViTAttentionSubstitutionCanonizer"]:
        # Stock timm ``Attention`` does not carry ``num_prefix_tokens``; it
        # lives on the top-level VisionTransformer, read once here and passed
        # down. Fallback 1 covers bare attentions in tests.
        num_prefix_tokens = int(getattr(root_module, "num_prefix_tokens", 1))

        instances: List[VanillaViTAttentionSubstitutionCanonizer] = []
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
        # See note on ``EvaAttentionSubstitutionCanonizer.register``.
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

    def copy(self) -> "VanillaViTAttentionSubstitutionCanonizer":
        return type(self)(block_indices=self.block_indices)


def _bind_forward(module: nn.Module, fn: Callable, attr: str = "forward") -> dict:
    """Bind ``fn`` as ``attr`` on ``module``'s class — return dict for
    AttributeCanonizer."""
    return {attr: fn.__get__(module, type(module))}



def _eva_block_forward(self, x, rope=None, attn_mask=None, is_causal=False):
    """``EvaBlock.forward`` replacement that routes the residual adds (and,
    optionally, the LayerScale γ multiplications) through ``nn.Module`` instances
    so a composite ``layer_map`` can attach an LRP rule to them. The rule is a
    ``layer_map`` decision, not a property of this module.
    """
    # ``self._lrp_res{1,2}`` = ResidualAdd, ``self._lrp_ls{1,2}`` = LayerScaleMul,
    # attached by :class:`EvaBlockResidualCanonizer`.
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
    """Swap ``forward`` on timm ``eva.EvaBlock`` so the two residual adds
    route through :class:`~zennit_extensions.attention_unfolded.ResidualAdd`
    modules — making them hookable. The residual *rule* is a ``layer_map``
    decision; this canonizer installs only the (single) module type.

    Parameters
    ----------
    layerscale_uniform : bool
        Also route the LayerScale γ multiplications through
        :class:`~zennit_extensions.attention_unfolded.LayerScaleMul` modules
        (typically mapped to ``Uniform(factor=2)``, AttnLRP Eq. 7).
        Default False.
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
    through ``self._lrp_res{1,2}`` (:class:`~zennit_extensions.attention_unfolded.ResidualAdd`)
    so a composite ``layer_map`` can attach the chosen residual rule. LayerScale
    on standard timm Blocks is already an ``nn.Module`` (``ls1``/``ls2``).
    """
    branch1 = self.drop_path1(
        self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal))
    )
    x = self._lrp_res1(x, branch1)
    branch2 = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    x = self._lrp_res2(x, branch2)
    return x


class VanillaViTBlockResidualCanonizer(AttributeCanonizer):
    """Swap ``forward`` on timm ``vision_transformer.Block`` so each residual
    add routes through a :class:`~zennit_extensions.attention_unfolded.ResidualAdd`
    module — making the add hookable. The residual *rule* is a ``layer_map``
    decision; this canonizer installs only the (single) module type.
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

