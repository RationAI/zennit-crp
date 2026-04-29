"""AttnLRP for vision transformers — idiomatic zennit Canonizer + Hook + Composite.

This module implements the AttnLRP rules of Achtibat et al. (ICML 2024,
arXiv:2402.05602) on top of zennit, exposing them as the standard zennit
abstractions:

* :class:`QKVTapCanonizer` / :class:`TimmViTCanonizer` — model-graph and
  forward-method changes needed for AttnLRP, registered on
  ``composite.context()`` enter and reverted on exit.
* :class:`GradientTimesInputBasicHook` — a :class:`zennit.core.BasicHook`
  subclass that runs the LRP backward in the gradient×input formulation
  ``R = grad·output / input`` (the framework into which AttnLRP's
  identity / uniform rules are embedded). Pure subclass; no global side
  effects.
* :class:`AttnLRPEpsilonComposite` — drop-in composite that combines the
  canonizer with the LRP rule hooks. Mapped via
  :class:`zennit.composites.LayerMapComposite`.

The four ViT concept classes in :mod:`crp.attention_concepts` hook the
named tap ``…attn.qkv_tap`` that :class:`QKVTapCanonizer` adds.

Usage::

    from zennit.composites import EpsilonPlusFlat
    from crp.transformer_patches import AttnLRPEpsilonComposite
    from crp.attention_concepts import HeadConcept
    from crp.attribution import CondAttribution

    composite = AttnLRPEpsilonComposite()      # canonizer pre-bundled
    attribution = CondAttribution(model)
    concept = HeadConcept()
    concept.register_from_model(model)
    result = attribution(
        data, [{"blocks.6.attn.qkv_tap": [0], "y": [281]}],
        composite, mask_map=concept.mask,
    )
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Function

from zennit.canonizers import AttributeCanonizer, Canonizer, CompositeCanonizer
from zennit.composites import LayerMapComposite
from zennit.core import BasicHook, ParamMod, Stabilizer, stabilize
from zennit.rules import GammaMod, NoMod, Pass


# ─── 1. Autograd Functions encoding AttnLRP rules ────────────────────────────


class _IdentityRuleFn(Function):
    """Identity rule (AttnLRP §3.1, Eq. 9): backward returns ``(out / in) * grad_out``.

    Used on element-wise non-linearities (GELU, ReLU, …) where LRP relevance
    flows through unchanged up to the input/output ratio. Inlined into the
    forward pass via a Canonizer-installed forward method, so the autograd
    graph carries it without any global mutation.
    """

    @staticmethod
    def forward(ctx, fn, input, epsilon=1e-10):
        output = fn(input)
        if input.requires_grad:
            ctx.save_for_backward(output / (input + epsilon))
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        gradient = ctx.saved_tensors[0] * out_relevance[0]
        return None, gradient, None


class _DivideGradientFn(Function):
    """Uniform rule (AttnLRP §3.1, Eq. 7): backward divides incoming grad by ``factor``.

    Used after bilinear ops (matmul, ⊙) to allocate relevance equally among
    operands. ``factor=2`` per bilinear; in attention Q×4, K×4 (each enters
    two bilinears via softmax(QKᵀ)) and V×2 (one bilinear via attn·V).
    """

    @staticmethod
    def forward(ctx, input, factor=2):
        ctx.factor = factor
        return input

    @staticmethod
    def backward(ctx, *out_relevance):
        return out_relevance[0] / ctx.factor, None


def identity_rule_implicit(fn, input):
    """Apply ``fn(input)`` with the AttnLRP identity rule inlined into backward."""
    return _IdentityRuleFn.apply(fn, input)


def divide_gradient(input, factor=2):
    """Identity in forward; divide incoming relevance by ``factor`` on backward."""
    return _DivideGradientFn.apply(input, factor)


def stop_gradient(input):
    """Detach ``input`` from the autograd graph (CP-LRP variant on normalisations)."""
    return input.detach()


# ─── 2. Replacement forward methods (installed per-instance by Canonizer) ────


def layer_norm_forward(self, x):
    """LayerNorm with stop-gradient on the std. Identity rule on the
    normalised output (AttnLRP §3.2.2)."""
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.eps).sqrt()
    y = (x - mean) / stop_gradient(std)
    if self.weight is not None:
        y = y * self.weight
    if self.bias is not None:
        y = y + self.bias
    return y


def dropout_passthrough_forward(self, x):
    """Disable dropout during attribution (model may be in train mode)."""
    return x


def vit_pos_embed_palrp(self, x):
    """timm ``VisionTransformer._pos_embed`` replacement implementing PA-LRP.

    PA-LRP (Bakish et al., NeurIPS 2025; arXiv 2506.02138) treats the
    additive ``x = x + pos_embed`` step as a bilinear-style operation under
    the LRP uniform rule (Eq. 7 of AttnLRP) and allocates relevance equally
    between the two operands. Because ``pos_embed`` is a learned parameter
    with no upstream input, its half is "absorbed"; the remaining half flows
    back to ``x``. Without this, the additive step is transparent to
    backward and ``x`` receives the full upstream relevance, double-counting
    it against ``pos_embed``.

    Implementation: identical to the upstream ``_pos_embed`` (timm 1.0.x —
    handles ``cls_token``/``reg_token``, ``no_embed_class`` deit-3 variant,
    ``dynamic_img_size``) except the result of the additive step is wrapped
    in :func:`divide_gradient` with ``factor=2``.
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
        x = divide_gradient(x + pos_embed, 2)
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = divide_gradient(x + pos_embed, 2)

    return self.pos_drop(x)


def timm_attention_forward(self, x, attn_mask=None, is_causal=False):
    """timm ``Attention.forward`` replacement for AttnLRP + concept hooking.

    Tracks the upstream timm signature ``(self, x, attn_mask=None,
    is_causal=False)`` so it remains a drop-in across timm ≥ 1.0.

    Differences from upstream timm forward:

    * Routes ``self.qkv(x)`` through ``self.qkv_tap`` (an ``nn.Identity``
      submodule installed by :class:`QKVTapCanonizer`) — the named hook
      point used by :mod:`crp.attention_concepts`.
    * Q, K, V each pass through :func:`divide_gradient` (factors 4, 4, 2) —
      AttnLRP uniform rule on the ``QKᵀ`` and ``attn·V`` bilinears
      (Eq. 14–15 of arXiv:2402.05602).
    * Bypasses the fused ``F.scaled_dot_product_attention`` path so the
      autograd graph contains the explicit softmax + matmul ops where the
      rules apply.
    """
    B, N, _ = x.shape
    qkv_flat = self.qkv(x)              # (B, N, 3*num_heads*head_dim)
    qkv_flat = self.qkv_tap(qkv_flat)   # ← named hook tap
    qkv = qkv_flat.reshape(B, N, 3, self.num_heads, self.head_dim).permute(
        2, 0, 3, 1, 4
    )
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = divide_gradient(q, 4)
    k = divide_gradient(k, 4)
    v = divide_gradient(v, 2)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)

    try:
        from timm.models.vision_transformer import (
            resolve_self_attn_mask, maybe_add_mask,
        )
        attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
        attn = maybe_add_mask(attn, attn_bias)
    except ImportError:
        if attn_mask is not None:
            attn = attn + attn_mask
        if is_causal:
            mask = torch.triu(
                torch.full(
                    (N, N), float("-inf"), device=attn.device, dtype=attn.dtype
                ),
                diagonal=1,
            )
            attn = attn + mask

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ v

    out_dim = getattr(self, "attn_dim", self.num_heads * self.head_dim)
    x = x.transpose(1, 2).reshape(B, N, out_dim)
    if hasattr(self, "norm"):
        x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# ─── 3. Canonizers — register on composite.context() enter, revert on exit ──


class QKVTapCanonizer(Canonizer):
    """Inject ``qkv_tap = nn.Identity()`` into every timm-style ``Attention``
    submodule. Detection: presence of ``qkv`` (a ``nn.Linear``), ``num_heads``,
    ``head_dim``.

    The tap is the named hook point used by :mod:`crp.attention_concepts`.
    Registers on apply, removes on :meth:`remove` (so the model is reverted
    when ``composite.context()`` exits).
    """

    def __init__(self):
        self.module: Optional[nn.Module] = None

    def apply(self, root_module):
        instances = []
        for _name, module in root_module.named_modules():
            if (
                hasattr(module, "qkv")
                and isinstance(getattr(module, "qkv"), nn.Linear)
                and hasattr(module, "num_heads")
                and hasattr(module, "head_dim")
            ):
                inst = self.copy()
                inst.register(module)
                instances.append(inst)
        return instances

    def register(self, module):
        existing = getattr(module, "qkv_tap", None)
        if isinstance(existing, nn.Identity):
            # User pre-injected; respect that, do not auto-remove.
            self.module = None
            return
        self.module = module
        module.add_module("qkv_tap", nn.Identity())

    def remove(self):
        if self.module is None:
            return
        del self.module._modules["qkv_tap"]

    def copy(self):
        return type(self)()


def _attn_attribute_map(_name, module):
    """AttributeCanonizer attribute_map: swap ``forward`` on timm Attention."""
    try:
        from timm.models.vision_transformer import Attention as TimmAttention
    except ImportError:
        return None
    if isinstance(module, TimmAttention):
        bound = timm_attention_forward.__get__(module, type(module))
        return {"forward": bound}
    return None


def _layer_norm_attribute_map(_name, module):
    """AttributeCanonizer attribute_map: swap ``forward`` on LayerNorm."""
    if isinstance(module, nn.LayerNorm):
        bound = layer_norm_forward.__get__(module, type(module))
        return {"forward": bound}
    return None


def _gelu_attribute_map(_name, module):
    """AttributeCanonizer attribute_map: route GELU through identity-rule autograd."""
    if isinstance(module, nn.GELU):
        # Capture the unmodified class-level forward for the inner call.
        original_forward = type(module).forward

        def patched(self, x):
            return identity_rule_implicit(lambda inp: original_forward(self, inp), x)

        return {"forward": patched.__get__(module, type(module))}
    return None


def _dropout_attribute_map(_name, module):
    """AttributeCanonizer attribute_map: pass-through on Dropout."""
    if isinstance(module, nn.Dropout):
        bound = dropout_passthrough_forward.__get__(module, type(module))
        return {"forward": bound}
    return None


def _vit_pos_embed_attribute_map(_name, module):
    """AttributeCanonizer attribute_map: swap ``_pos_embed`` on timm
    ``VisionTransformer`` instances to apply PA-LRP."""
    try:
        from timm.models.vision_transformer import VisionTransformer
    except ImportError:
        return None
    if isinstance(module, VisionTransformer) and hasattr(module, "_pos_embed"):
        bound = vit_pos_embed_palrp.__get__(module, type(module))
        return {"_pos_embed": bound}
    return None


class TimmViTCanonizer(CompositeCanonizer):
    """All graph + forward changes needed for AttnLRP on a timm ViT, in one
    canonizer. Composes:

    * :class:`QKVTapCanonizer` — install ``qkv_tap`` Identity submodule.
    * ``AttributeCanonizer`` — swap ``forward`` on ``Attention``,
      ``LayerNorm``, ``GELU``, ``Dropout`` instances; optionally swap
      ``_pos_embed`` on ``VisionTransformer`` for the PA-LRP rule
      (Bakish et al.).

    All mutations are instance-level and reversible. Bundled into
    :class:`AttnLRPEpsilonComposite`; pass it explicitly to other
    composites if you want a custom rule map.

    Parameters
    ----------
    palrp : bool
        If True, also swap the ViT's ``_pos_embed`` to wrap the additive
        ``x + pos_embed`` step in a uniform-rule ``divide_gradient(., 2)``.
        Defaults to False because (a) AttnLRP §3 / Achtibat et al. don't use
        it and the milestone-A audit baseline is without; (b) the
        ``vit_pos_embed_palrp`` reimplements timm's ``_pos_embed`` and is
        sensitive to upstream timm changes. Enable per AttnLRP-paper-Eq.-1
        conservation if needed (see ``FUTURE_STATE.md`` Milestone D).
    """

    def __init__(self, palrp: bool = False):
        canonizers = [
            QKVTapCanonizer(),
            AttributeCanonizer(_attn_attribute_map),
            AttributeCanonizer(_layer_norm_attribute_map),
            AttributeCanonizer(_gelu_attribute_map),
            AttributeCanonizer(_dropout_attribute_map),
        ]
        if palrp:
            canonizers.append(AttributeCanonizer(_vit_pos_embed_attribute_map))
        super().__init__(canonizers)


# ─── 4. Custom Hook — gradient×input LRP backward ────────────────────────────


class GradientTimesInputBasicHook(BasicHook):
    """Subclass of :class:`zennit.core.BasicHook` that runs LRP backward in
    the gradient×input framework: ``R = grad·output / input``.

    This is the formulation into which AttnLRP's identity / uniform rules
    (encoded as autograd Functions in the forward) are designed to fit.
    Standard zennit hooks return ``R = inputs * gradients`` directly; this
    subclass instead

    1. multiplies ``grad_output`` by the saved forward output before the
       backward,
    2. runs the standard zennit backward through the modified module,
    3. divides the resulting relevance by the saved input.

    Pure subclass — no monkey-patching of zennit. Plug into any
    :class:`zennit.core.Composite` via the standard mapping API.
    """

    def forward(self, module, input, output):
        # Match zennit's "no kwargs" forward signature so register detects 3 params.
        self.stored_tensors["input"] = input
        self.stored_tensors["output"] = output

    def backward(self, module, grad_input, grad_output):
        assert len(grad_output) == 1, "single-output module required"

        original_input = self.stored_tensors["input"][0].clone()
        gti_grad = grad_output[0] * self.stored_tensors["output"]
        gti_grad = gti_grad.requires_grad_(True)

        inputs, outputs = [], []
        for in_mod, param_mod, out_mod in zip(
            self.input_modifiers, self.param_modifiers, self.output_modifiers
        ):
            inp = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                out = modified.forward(inp)
                out = out_mod(out)
            inputs.append(inp)
            outputs.append(out)

        grad_outputs = self.gradient_mapper(gti_grad, outputs)
        gradients = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            create_graph=gti_grad.requires_grad,
        )
        relevance = self.reducer(inputs, gradients)
        relevance = relevance / stabilize(original_input, epsilon=1e-10)

        return tuple(
            relevance if original.shape == relevance.shape else None
            for original in grad_input
        )


# ─── 5. Convenience rule + composite ─────────────────────────────────────────


class GTIEpsilon(GradientTimesInputBasicHook):
    """ε-LRP in the gradient×input formulation. Drop-in for
    :class:`zennit.rules.Epsilon` inside an
    :class:`AttnLRPEpsilonComposite`."""

    def __init__(self, epsilon=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(zero_params=zero_params)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(
                lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])
            ),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class GTIGamma(GradientTimesInputBasicHook):
    """γ-LRP in the gradient×input formulation, single-branch (positive
    weight clamp).

    AttnLRP §3.2.1 recommends γ ≈ 0.25 on ViT linear layers to mitigate
    gradient shattering. The rule modifies the weights as
    ``W' = W + γ·max(W, 0)`` per backward pass and runs the LRP backward
    through the modified module — wrapped in the GTI framework
    (multiply grad_output by output, divide relevance by input) so the
    AttnLRP autograd Functions further down the graph compose correctly.

    Parameters
    ----------
    gamma : float
        Scaling for the positive-weight branch. Default 0.25.
    epsilon : float
        ε-stabilizer in the denominator. Default 1e-6.
    zero_params : list[str] | None
        Names of parameters to zero in the modified module (e.g. ``["bias"]``).
    """

    def __init__(self, gamma=0.25, epsilon=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[GammaMod(gamma, min=0, zero_params=zero_params)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(
                lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])
            ),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class AttnLRPEpsilonComposite(LayerMapComposite):
    """AttnLRP (ε-LRP variant) for ViTs. Drop-in replacement for
    ``EpsilonPlusFlat`` when attributing through a timm ViT.

    Maps:

    * ``nn.Linear`` → :class:`GTIEpsilon` (ε-LRP, gradient×input form).
    * ``nn.Conv2d`` → :class:`GTIEpsilon` (covers the patch-embed Conv2d).
    * ``nn.GELU`` / ``nn.LayerNorm`` / ``nn.Dropout`` / ``nn.Identity``
      → :class:`zennit.rules.Pass` — handled in-graph by the embedded
      autograd functions installed by :class:`TimmViTCanonizer`.

    :class:`TimmViTCanonizer` is pre-bundled. Pass extra canonizers via
    the ``canonizers`` kwarg if you also need (for example) a
    ``SequentialMergeBatchNorm`` for a hybrid model.
    """

    def __init__(self, epsilon=1e-6, canonizers=None, palrp: bool = False):
        canonizers = list(canonizers or []) + [TimmViTCanonizer(palrp=palrp)]
        layer_map = [
            (nn.Linear, GTIEpsilon(epsilon)),
            (nn.Conv2d, GTIEpsilon(epsilon)),
            (nn.GELU, Pass()),
            (nn.LayerNorm, Pass()),
            (nn.Dropout, Pass()),
            (nn.Identity, Pass()),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)


class AttnLRPGammaComposite(LayerMapComposite):
    """AttnLRP with γ-LRP on ViT linears (AttnLRP §3.2.1).

    Same structure as :class:`AttnLRPEpsilonComposite` but maps
    ``nn.Linear`` and ``nn.Conv2d`` to :class:`GTIGamma` instead of
    :class:`GTIEpsilon`. Recommended over the ε-only variant when
    attributing through a deep ViT — the γ rule biases relevance toward
    positive contributions and reduces gradient-shattering noise that
    shows up as a deletion/insertion-AUC anomaly on finer concept
    granularities under the bare ε-LRP composite.

    Parameters
    ----------
    gamma : float
        γ scaling. Default 0.25 per AttnLRP §3.2.1.
    epsilon : float
        ε-stabilizer in the LRP denominator. Default 1e-6.
    canonizers : list[Canonizer] | None
        Extra canonizers to apply alongside :class:`TimmViTCanonizer`.
    """

    def __init__(self, gamma=0.25, epsilon=1e-6, canonizers=None, palrp: bool = False):
        canonizers = list(canonizers or []) + [TimmViTCanonizer(palrp=palrp)]
        layer_map = [
            (nn.Linear, GTIGamma(gamma=gamma, epsilon=epsilon)),
            (nn.Conv2d, GTIGamma(gamma=gamma, epsilon=epsilon)),
            (nn.GELU, Pass()),
            (nn.LayerNorm, Pass()),
            (nn.Dropout, Pass()),
            (nn.Identity, Pass()),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)


__all__ = [
    # autograd Functions (AttnLRP rule kernels)
    "identity_rule_implicit",
    "divide_gradient",
    "stop_gradient",
    # forward replacements (advanced use; Canonizer installs them automatically)
    "layer_norm_forward",
    "dropout_passthrough_forward",
    "timm_attention_forward",
    "vit_pos_embed_palrp",
    # Canonizers
    "QKVTapCanonizer",
    "TimmViTCanonizer",
    # Hooks
    "GradientTimesInputBasicHook",
    "GTIEpsilon",
    "GTIGamma",
    # Composites
    "AttnLRPEpsilonComposite",
    "AttnLRPGammaComposite",
]
