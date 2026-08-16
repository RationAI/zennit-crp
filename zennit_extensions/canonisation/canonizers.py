from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
from timm.layers import GluMlp, Mlp, SwiGLU
from timm.models.eva import EvaAttention, EvaBlock
from timm.models.vision_transformer import (
    Attention as TimmAttention, Block as TimmBlock, VisionTransformer,
)
from zennit.canonizers import AttributeCanonizer, Canonizer

from zennit_extensions.attention_unfolded import (
    EvaAttentionUnfolded,
    FFNLinear,
    LayerNormDetachedStd,
    TimmAttentionUnfolded,
)

#: timm MLP/FFN container types whose ``nn.Linear`` children are FFN linears
#: (paper Table B.5 gives these γ-LRP, everything else ε-LRP).
FFN_CONTAINER_TYPES = (Mlp, GluMlp, SwiGLU)


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


class LayerNormSubstitutionCanonizer(Canonizer):
    """Replace every plain ``nn.LayerNorm`` with
    :class:`~zennit_extensions.attention_unfolded.LayerNormDetachedStd`
    (σ detached — the LXT LayerNorm treatment). Matches exact type only, not
    subclasses (e.g. timm's channels-first ``LayerNorm2d`` has a different
    forward and must not be swapped). No LRP-rule decisions — the composite
    ``layer_map`` assigns the rule to the substituted type.
    """

    def __init__(self):
        self.parent: Optional[nn.Module] = None
        self.attr_name: Optional[str] = None
        self.original_module: Optional[nn.Module] = None
        self.detached_module: Optional[LayerNormDetachedStd] = None

    def apply(self, root_module: nn.Module) -> List["LayerNormSubstitutionCanonizer"]:
        instances: List[LayerNormSubstitutionCanonizer] = []
        for _parent_name, parent in root_module.named_modules():
            for attr_name, child in parent.named_children():
                if type(child) is not nn.LayerNorm:
                    continue
                inst = self.copy()
                inst.register(parent, attr_name, child)
                instances.append(inst)
        return instances

    def register(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        parent: nn.Module,
        attr_name: str,
        original: nn.LayerNorm,
    ) -> None:
        # Same signature deviation as the attention substitution canonizers
        # above — zennit dispatches through ``apply()``.
        self.parent = parent
        self.attr_name = attr_name
        self.original_module = original
        detached = LayerNormDetachedStd(original)
        setattr(parent, attr_name, detached)
        self.detached_module = detached

    def remove(self) -> None:
        if self.parent is None or self.attr_name is None or self.original_module is None:
            return
        setattr(self.parent, self.attr_name, self.original_module)
        self.detached_module = None

    def copy(self) -> "LayerNormSubstitutionCanonizer":
        return type(self)()


class FFNLinearSubstitutionCanonizer(Canonizer):
    """Retype every plain ``nn.Linear`` child of a timm MLP container
    (:data:`FFN_CONTAINER_TYPES`) as the marker alias
    :class:`~zennit_extensions.attention_unfolded.FFNLinear` — same
    parameters by reference, identical forward. This lifts the paper's
    Table B.5 FFN-γ / projection-ε split from a name-substring match into the
    type system: composites map ``(FFNLinear, Gamma)`` before
    ``(nn.Linear, Epsilon)`` and every unmarked linear (qkv/proj/head)
    correctly falls to ε. Exact-type match, so an already-substituted
    ``FFNLinear`` is never re-wrapped. No LRP-rule decisions here.
    """

    def __init__(self):
        self.parent: Optional[nn.Module] = None
        self.attr_name: Optional[str] = None
        self.original_module: Optional[nn.Module] = None
        self.alias_module: Optional[FFNLinear] = None

    def apply(self, root_module: nn.Module) -> List["FFNLinearSubstitutionCanonizer"]:
        instances: List[FFNLinearSubstitutionCanonizer] = []
        for _name, container in root_module.named_modules():
            if not isinstance(container, FFN_CONTAINER_TYPES):
                continue
            for attr_name, child in container.named_children():
                if type(child) is not nn.Linear:
                    continue
                inst = self.copy()
                inst.register(container, attr_name, child)
                instances.append(inst)
        return instances

    def register(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        parent: nn.Module,
        attr_name: str,
        original: nn.Linear,
    ) -> None:
        # Same signature deviation as the substitution canonizers above —
        # zennit dispatches through ``apply()``.
        self.parent = parent
        self.attr_name = attr_name
        self.original_module = original
        alias = FFNLinear.from_linear(original)
        setattr(parent, attr_name, alias)
        self.alias_module = alias

    def remove(self) -> None:
        if self.parent is None or self.attr_name is None or self.original_module is None:
            return
        setattr(self.parent, self.attr_name, self.original_module)
        self.alias_module = None

    def copy(self) -> "FFNLinearSubstitutionCanonizer":
        return type(self)()


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
        (typically mapped to ``Pass``: bias-free elementwise linear γ-multiply,
        ε-attribution ≈ identity).
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


def _vit_pos_embed_unfolded(self, x):
    """``VisionTransformer._pos_embed`` replacement that routes the
    ``x + pos_embed`` merge through ``self.pos_embed_add``
    (:class:`~zennit_extensions.attention_unfolded.PosEmbedAdd`), making it
    hookable for a PA-LRP positional-sink rule. Faithful transcription of the
    stock method (handles ``cls_token``/``reg_token``, ``no_embed_class``
    deit-3 variant, ``dynamic_img_size``) — only the additive step differs.
    """
    to_cat = []
    if self.cls_token is not None:
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
    if self.reg_token is not None:
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

    if self.pos_embed is None:
        return torch.cat(to_cat + [x.view(x.shape[0], -1, x.shape[-1])], dim=1)

    if self.dynamic_img_size:
        from timm.layers.pos_embed import resample_abs_pos_embed
        B, H, W, C = x.shape
        prev_grid_size = self.patch_embed.grid_size
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(H, W),
            old_size=prev_grid_size,
            num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
        )
        x = x.view(B, -1, C)
    else:
        pos_embed = self.pos_embed

    if self.no_embed_class:
        # deit-3 / big-vision: positional embedding does not overlap with the
        # class token — add first, then concat.
        x = self.pos_embed_add(x, pos_embed)
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        # original timm / deit-vit: pos_embed has an entry for the class token
        # — concat first, then add.
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = self.pos_embed_add(x, pos_embed)

    return self.pos_drop(x)


class VanillaViTPosEmbedCanonizer(AttributeCanonizer):
    """Install a :class:`~zennit_extensions.attention_unfolded.PosEmbedAdd`
    module on every ``vision_transformer.VisionTransformer`` and swap its
    ``_pos_embed`` so ``x + pos_embed`` routes through it — making the
    positional merge hookable (PA-LRP, paper §3.2). Installs structure only;
    which rule (if any) acts on ``PosEmbedAdd`` is a ``layer_map`` decision
    (PA-LRP opt-in via :class:`~zennit_extensions.rules.palrp.PosEmbedSink`).
    Skips positional-free variants (``pos_embed is None``). Unmapped, the
    module computes plain ``x + pos_embed`` so default recipes are unchanged.
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        if not isinstance(module, VisionTransformer):
            return None
        needed = ("_pos_embed", "pos_embed", "cls_token", "reg_token",
                  "pos_drop", "patch_embed", "no_embed_class",
                  "num_prefix_tokens", "dynamic_img_size")
        if not all(hasattr(module, a) for a in needed) or module.pos_embed is None:
            return None
        from zennit_extensions.attention_unfolded import PosEmbedAdd

        def fwd(self, x):
            return _vit_pos_embed_unfolded(self, x)

        attrs = _bind_forward(module, fwd, attr="_pos_embed")
        attrs["pos_embed_add"] = PosEmbedAdd()
        return attrs

    def copy(self):
        return type(self)()

