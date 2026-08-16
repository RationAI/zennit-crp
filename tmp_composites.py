"""PA-LRP example composites — instances of zennit base classes (no wrappers).

Inspect-only temporary file. Each composite below is a plain instance of a
zennit base composite class (``LayerMapComposite`` / ``NameMapComposite`` /
``MixedComposite``) with canonizers and rules given directly to the
constructor — PA-LRP is opted into *only* by adding the positional-sink rules
(``PosEmbedSink`` Eq. 5, ``RotaryRopeSink`` Eq. 10) to the layer map. Default
recipes (``AttnLRPBaselineComposite`` etc.) are unchanged: they install the
positional structure (``VanillaViTPosEmbedCanonizer``, the existing
``RotaryEmbedding`` modules) but map no rule to it.

Paper: Bakish, Zimerman, Chefer & Wolf, "Revisiting LRP: Positional
Attribution as the Missing Ingredient for Transformer Explainability",
arXiv:2506.02138.

Run the demo (offline, no weight download)::

    uv run python tmp_composites.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from zennit.composites import Composite, LayerMapComposite, MixedComposite, NameMapComposite
from zennit.rules import Epsilon, Gamma, Pass

from zennit_extensions.attention_unfolded import (
    BilinearMatmul,
    FFNLinear,
    LayerNormDetachedStd,
    LayerScaleMul,
    PosEmbedAdd,
    ResidualAdd,
    RotaryEmbedding,
    ScaleByConstant,
    SoftmaxAlongLastDim,
)
from zennit_extensions.canonisation.canonizers import (
    EvaAttentionSubstitutionCanonizer,
    EvaBlockResidualCanonizer,
    FFNLinearSubstitutionCanonizer,
    LayerNormSubstitutionCanonizer,
    VanillaViTAttentionSubstitutionCanonizer,
    VanillaViTBlockResidualCanonizer,
    VanillaViTPosEmbedCanonizer,
)
from zennit_extensions.rules.attnlrp import (
    EpsilonAdd,
    LayerNormEpsilon,
    MatmulAttnLRP,
    SoftmaxAttnLRP,
)
from zennit_extensions.rules.palrp import PosEmbedSink, RotaryRopeSink


# ── the default AttnLRP composite as one flat, ad-hoc instance ──────────────
# Everything AttnLRPBaselineComposite does, spelled out with no wrapper class
# and no callables: copy this block (plus the imports above) into a notebook
# and edit lines directly. The paper's Table B.5 FFN-γ/projection-ε split is
# type-based — FFNLinearSubstitutionCanonizer retypes MLP linears as
# FFNLinear, so the map stays a pure literal. ORDER MATTERS on the two Linear
# entries: FFNLinear is an nn.Linear subclass, first isinstance match wins.

attnlrp_default = LayerMapComposite(
    layer_map=[
        (BilinearMatmul, MatmulAttnLRP(epsilon=1e-6)),                        # Eq. 15
        (SoftmaxAlongLastDim, SoftmaxAttnLRP(bias_mode="absorb")),            # Prop. 3.1
        (LayerNormDetachedStd, LayerNormEpsilon(epsilon=1e-6, bias_mode="absorb")),  # LXT LN
        (ResidualAdd, EpsilonAdd(epsilon=1e-6)),                              # LXT add2
        (LayerScaleMul, Pass()),          # bias-free elementwise linear ⇒ identity
        (ScaleByConstant, Pass()),
        (nn.GELU, Pass()),                # Eq. 9 elementwise identity
        (nn.LayerNorm, Pass()),           # fallback: subclasses the canonizer skips
        (nn.Dropout, Pass()),
        (FFNLinear, Gamma(gamma=0.05)),   # Table B.5 FFN-γ — MUST precede nn.Linear
        (nn.Linear, Epsilon(epsilon=1e-6)),  # qkv / proj / head → ε
        (nn.Conv2d, Gamma(gamma=0.25)),   # patch embed
        (nn.Identity, Pass()),
    ],
    canonizers=[
        VanillaViTBlockResidualCanonizer(),
        EvaBlockResidualCanonizer(layerscale_uniform=True),
        VanillaViTPosEmbedCanonizer(),
        EvaAttentionSubstitutionCanonizer(block_indices=None),
        VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
        LayerNormSubstitutionCanonizer(),     # nn.LayerNorm → LayerNormDetachedStd
        FFNLinearSubstitutionCanonizer(),     # MLP nn.Linear → FFNLinear marker
    ],
)


# ── canonizers: structure for both stacks (pos-embed + RoPE unfolded) ───────
# Each fires only on its own backbone's module type; the others no-op, so one
# list serves vanilla ViTs (learnable pos_embed) and DINOv3/Eva (RoPE) alike.
PALRP_CANONIZERS = [
    VanillaViTBlockResidualCanonizer(),
    EvaBlockResidualCanonizer(layerscale_uniform=True),
    VanillaViTPosEmbedCanonizer(),
    EvaAttentionSubstitutionCanonizer(block_indices=None),
    VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
    LayerNormSubstitutionCanonizer(),
    FFNLinearSubstitutionCanonizer(),
]


def _attnlrp_layer_map(epsilon: float = 1e-6, ffn_gamma: float = 0.05,
                       conv_gamma: float = 0.25):
    """The AttnLRP type-keyed layer_map (mirrors ``AttnLRPBaselineComposite``).
    Fully type-based incl. the Table B.5 Linear split (FFNLinear marker via
    ``FFNLinearSubstitutionCanonizer``). Returned as a list so callers can
    prepend positional-sink entries."""
    return [
        (BilinearMatmul, MatmulAttnLRP(epsilon=epsilon)),
        (SoftmaxAlongLastDim, SoftmaxAttnLRP()),
        (LayerNormDetachedStd, LayerNormEpsilon(epsilon=epsilon)),
        (ScaleByConstant, Pass()),
        (ResidualAdd, EpsilonAdd(epsilon=epsilon)),
        (LayerScaleMul, Pass()),
        (nn.GELU, Pass()),
        (nn.LayerNorm, Pass()),
        (nn.Dropout, Pass()),
        (FFNLinear, Gamma(gamma=ffn_gamma)),    # MUST precede nn.Linear
        (nn.Linear, Epsilon(epsilon=epsilon)),
        (nn.Conv2d, Gamma(gamma=conv_gamma)),
        (nn.Identity, Pass()),
    ]


# ── example 1: LayerMapComposite — one PA-LRP instance, both stacks ─────────
# Prepend the two positional-sink rules so they win over the type map. Each
# fires only where its module exists (PosEmbedAdd on vanilla; RotaryEmbedding
# on DINOv3/Eva).
palrp_attnlrp = LayerMapComposite(
    layer_map=[
        (PosEmbedAdd, PosEmbedSink(epsilon=1e-6)),      # Eq. 5
        (RotaryEmbedding, RotaryRopeSink()),            # Eq. 10
        *_attnlrp_layer_map(epsilon=1e-6),
    ],
    canonizers=PALRP_CANONIZERS,
)


# ── example 2: MixedComposite — PA-LRP overlay on the full AttnLRP recipe ──
# The positional overlay (composites[0]) wins first; the full AttnLRP recipe
# (composites[1]) handles everything else. Now that the Linear split is
# type-based there is no callable module_map anywhere — pure literal maps.
_palrp_overlay = LayerMapComposite(
    layer_map=[
        (PosEmbedAdd, PosEmbedSink(epsilon=1e-6)),
        (RotaryEmbedding, RotaryRopeSink()),
    ],
    canonizers=[],   # canonizers come from composites[1] via MixedComposite
)
_full_attnlrp = LayerMapComposite(
    layer_map=_attnlrp_layer_map(epsilon=1e-6),
    canonizers=PALRP_CANONIZERS,
)
palrp_mixed = MixedComposite(
    composites=[_palrp_overlay, _full_attnlrp],
    canonizers=[],   # already merged from the sub-composites
)


# ── example 3: NameMapComposite — PA-LRP at one site by exact layer name ────
# Repo convention: explicit full layer names. ``backbone.pos_embed_add`` is the
# name VanillaViTPosEmbedCanonizer installs on a vanilla ViT (the zoo wrappers
# name the timm root ``backbone``). Demonstrates opting into PA-LRP at a single
# named layer without a type-wide rule.
palrp_named = MixedComposite(
    composites=[
        NameMapComposite(name_map=[
            (("backbone.pos_embed_add",), PosEmbedSink(epsilon=1e-6)),
        ]),
        _full_attnlrp,
    ],
    canonizers=[],
)


# ── sink collection + Eq. 11 aggregation ─────────────────────────────────────


def collect_positional_sinks(model) -> dict[str, torch.Tensor]:
    """Gather every ``_palrp_sink`` stash on ``PosEmbedAdd`` / ``RotaryEmbedding``
    modules. CONTRACT: call right after the attribution backward, while the
    composite context is active (sinks are per-backward scratch; the next
    backward or context exit invalidates them)."""
    sinks: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (PosEmbedAdd, RotaryEmbedding)):
            sink = getattr(mod, "_palrp_sink", None)
            if sink is not None:
                sinks[name] = sink
    return sinks


def positional_relevance(sinks: dict[str, torch.Tensor], grid_hw: tuple[int, int],
                         n_prefix: int = 1) -> dict[str, torch.Tensor]:
    """Eq. 11 aggregation: ``R_i = Σ_d E_i[d]⁺ + Σ_k Σ_{d'} P_{i,k}[d']⁺``.

    Here the positional half: sum only positive entries over the feature
    dims, sum over layers (``RotaryEmbedding`` sinks per block), drop prefix
    tokens, reshape patch tokens to ``grid_hw``. Returns per-sample spatial
    grids ``{'positional': (B, H, W), 'rope': (B, H, W), 'input': (B, H, W)}``.
    """
    pos_grid = None
    rope_grid = None
    for name, sink in sinks.items():
        pos = torch.relu(sink)                      # (·)+, Eq. 11
        is_rope = "rope_" in name                   # RotaryEmbedding submodule
        # token axis is -2; sum over all feature/head dims (everything except
        # batch=0 and token=-2) → per-token, per-sample relevance.
        dims = tuple(d for d in range(pos.dim()) if d not in (0, pos.dim() - 2))
        per_token = pos.sum(dim=dims) if dims else pos
        # drop prefix tokens (cls/register, unrotated for RoPE anyway)
        n_tokens = per_token.shape[-1]
        n_patches = grid_hw[0] * grid_hw[1]
        if n_tokens > n_patches:
            per_token = per_token[..., n_tokens - n_patches:]   # keep last n_patches
        B = per_token.shape[0]
        grid = per_token.reshape(B, grid_hw[0], grid_hw[1])
        if is_rope:
            rope_grid = grid if rope_grid is None else rope_grid + grid   # Σ_k
        else:
            pos_grid = grid
    combined = None
    for g in (pos_grid, rope_grid):
        if g is not None:
            combined = g if combined is None else combined + g
    return {
        "input": pos_grid,        # learnable pos_embed sink (vanilla)
        "rope": rope_grid,        # Σ over layers (DINOv3/Eva)
        "positional": combined,   # Eq. 11 total positional relevance
    }


# ── offline demo ────────────────────────────────────────────────────────────


def _demo() -> None:
    import timm

    print("PA-LRP example composites (instances of zennit base classes):")
    for n in ("palrp_attnlrp", "palrp_mixed", "palrp_named"):
        c = globals()[n]
        print(f"  {n:14s} {type(c).__name__}(canonizers={[type(x).__name__ for x in c.canonizers]})")

    # Sinks are stashed on the probe-site modules (PosEmbedAdd / RotaryEmbedding)
    # during backward and vanish when the composite context exits (the wrapper
    # modules are removed). So: forward + backward INSIDE the context, collect
    # sinks BEFORE exiting. (CondAttribution manages its own composite context
    # internally and reverts on return — to capture sinks, run the
    # forward/backward yourself under the composite context, or register a
    # side-channel collector alongside it.)
    def _run(model, x, comp, grid_hw, n_prefix):
        with comp.context(model):
            out = model(x)
            target = out[:, 1]
            target.sum().backward()
            sinks = collect_positional_sinks(model)
        return sinks

    # Vanilla stack (learnable pos_embed → input-level PA-LRP sink). B=2.
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).eval()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    sinks = _run(m, x, palrp_attnlrp, (14, 14), n_prefix=1)
    print(f"\nvanilla vit_tiny: {len(sinks)} sinks (B=2)")
    for name, sink in sinks.items():
        print(f"  {name:24s} shape={tuple(sink.shape)}  sum={sink.sum().item():.4g}")
    maps = positional_relevance(sinks, (14, 14), n_prefix=1)
    if maps["positional"] is not None and not isinstance(maps["positional"], float):
        print(f"  positional grid: {tuple(maps['positional'].shape)}  "
              f"sum={maps['positional'].sum().item():.4g}")

    # Eva stack (RoPE → attention-level PA-LRP sinks, one rope_q + rope_k per block).
    m_eva = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=False, num_classes=10).eval()
    x_eva = torch.randn(2, 3, 256, 256, requires_grad=True)
    sinks_eva = _run(m_eva, x_eva, palrp_attnlrp, (16, 16), n_prefix=4)
    rope_sinks = [n for n in sinks_eva if "rope_" in n]
    print(f"\nDINOv3-small: {len(sinks_eva)} sinks total, {len(rope_sinks)} RoPE "
          f"(expect 24 = 12 blocks × {{rope_q, rope_k}})")
    for name in rope_sinks[:4]:
        s = sinks_eva[name]
        print(f"  {name:40s} shape={tuple(s.shape)}  sum={s.sum().item():.4g}  "
              f"prefix-row-zero={torch.all(s[..., :4, :] == 0).item()}")
    maps_eva = positional_relevance(sinks_eva, (16, 16), n_prefix=4)
    if maps_eva["rope"] is not None and not isinstance(maps_eva["rope"], float):
        print(f"  rope positional grid: {tuple(maps_eva['rope'].shape)}  "
              f"sum={maps_eva['rope'].sum().item():.4g}")
    print("\nDemo OK. Sinks are per-sample (B=2); read under active context.")


if __name__ == "__main__":
    _demo()
