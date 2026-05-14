"""ViT integration tests for the AttnLRP Canonizer + Composite stack.

Covers:

* Standard timm ViT path (vit_tiny) under
  :class:`AttnLRPEpsilonComposite`, :class:`AttnLRPGammaComposite`,
  :class:`AttnLRPCombinedComposite` — attention is always unfolded
  (:class:`TimmAttentionUnfolded` swaps in via the substitution
  canonizer).
* Eva-stack path (untrained, fast) under
  :class:`AttnLRPCombinedComposite` — substitutes EvaAttention with
  EvaAttentionUnfolded and runs concept-conditioned attribution via
  :class:`HeadConcept`, :class:`QConcept`, :class:`AttnOutputDimConcept`.

Run::

    uv run pytest tests/test_vit_integration.py -v
"""

import pytest
import torch

timm = pytest.importorskip("timm")
zennit = pytest.importorskip("zennit")


from crp.attention_concepts import (
    HeadConcept,
    EmbeddingDimConcept,
    TokenConcept,
)
from crp.attention_unfolded import (
    EvaAttentionUnfolded,
    EvaAttentionSubstitutionCanonizer,
)
from crp.attribution import CondAttribution
from crp.transformer_patches import (
    AttnLRPEpsilonComposite,
    AttnLRPGammaComposite,
    AttnLRPCombinedComposite,
    TimmViTCanonizer,
)


# ── shared fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vit_tiny():
    """Smallest readily-available standard timm ViT (12 blocks, 3 heads,
    head_dim=64). Random init avoids weight download."""
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def eva_tiny():
    """A small Eva-stack ViT for the unfolded-substitution integration
    tests. Random init.

    The unfolded refactor's Phase 1 only supports Eva variants with
    ``q_bias = None`` (i.e. all Q/K/V biases packed into the qkv Linear,
    no separate bias tensors). Per-Q/K/V-bias variants (most ``eva*``
    models, including ``eva02_tiny_*``) are Phase 2 work — we skip
    those and search for a compatible alternative."""
    candidates = [
        "vit_small_patch16_dinov3",
        "vit_base_patch16_dinov3",
        "vit_tiny_patch14_dinov2",
    ]
    last_exc = None
    for name in candidates:
        try:
            m = timm.create_model(name, pretrained=False)
            m.eval()
            # Probe: does the first attention have q_bias? If so, skip
            # this model — Phase 1 doesn't support it.
            attn0 = m.blocks[0].attn
            if getattr(attn0, "q_bias", None) is not None:
                continue
            return m
        except Exception as e:
            last_exc = e
    pytest.skip(
        f"No q_bias=None Eva-stack model available; Phase 2 of "
        f"EvaAttentionUnfolded needs to add per-Q/K/V-bias support. "
        f"Last error: {last_exc}"
    )


@pytest.fixture
def img_batch():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224, requires_grad=True)


# ── TimmViTCanonizer round-trip (standard timm path) ────────────────────────


class TestTimmViTCanonizer:
    def test_forward_swap_is_reversible(self, vit_tiny):
        """TimmViTCanonizer installs forward closures on LayerNorm /
        GELU / Dropout (attention is now handled by the separate
        substitution canonizer). After context exit, the original
        forwards must be restored.
        """
        ln = vit_tiny.blocks[0].norm1
        original_class_forward = type(ln).forward
        assert "forward" not in ln.__dict__, "test pre-condition: clean state"
        canonizer = TimmViTCanonizer()
        instances = canonizer.apply(vit_tiny)
        try:
            assert "forward" in ln.__dict__, "canonizer did not swap LayerNorm forward"
            assert ln.forward.__func__ is not original_class_forward
        finally:
            for inst in instances:
                inst.remove()
        assert "forward" not in ln.__dict__
        assert type(ln).forward is original_class_forward


# ── Standard timm path: composites instantiate + run ────────────────────────


def test_epsilon_composite_runs(vit_tiny, img_batch):
    composite = AttnLRPEpsilonComposite()
    with composite.context(vit_tiny) as modified:
        out = modified(img_batch)
    assert out.shape == (1, 1000)


def test_gamma_composite_runs(vit_tiny, img_batch):
    composite = AttnLRPGammaComposite()
    with composite.context(vit_tiny) as modified:
        out = modified(img_batch)
    assert out.shape == (1, 1000)


def test_gamma_composite_attribution_end_to_end(vit_tiny, img_batch):
    composite = AttnLRPGammaComposite(gamma=0.25)
    attribution = CondAttribution(vit_tiny)
    res = attribution(img_batch, [{"y": [42]}], composite)
    assert res.heatmap.shape == (1, 224, 224)


def test_combined_composite_runs_on_standard_timm(vit_tiny, img_batch):
    """Combined composite must attribute standard timm ViTs without
    crashing — attention is substituted to TimmAttentionUnfolded; the
    Eva substitution canonizer no-ops on stock timm Attention."""
    composite = AttnLRPCombinedComposite(
        alpha=0.5, beta=0.5,
        residual_lrp="ratio",
    )
    attribution = CondAttribution(vit_tiny)
    res = attribution(img_batch, [{"y": [42]}], composite)
    assert res.heatmap.shape == (1, 224, 224)
    assert torch.isfinite(res.heatmap).all(), "combined recipe NaN'd on vit_tiny"


# ── Conservation diagnostic (standard timm path) ────────────────────────────


def test_conservation_combined_recipe(vit_tiny):
    """sum(R_input) / target_logit should be O(1)–O(100) under the
    AttnLRP-correct combined recipe — not blow up to NaN. Loose bound;
    diagnostic, not gating."""
    torch.manual_seed(0)
    data = torch.randn(1, 3, 224, 224, requires_grad=True)
    target = 42
    with torch.no_grad():
        logit_val = vit_tiny(data)[0, target].item()
    composite = AttnLRPCombinedComposite(
        alpha=0.5, beta=0.5, residual_lrp="ratio",
    )
    attribution = CondAttribution(vit_tiny)
    attribution(data, [{"y": [target]}], composite)
    sum_R = data.grad.sum().item()
    assert torch.isfinite(torch.tensor(sum_R)), f"sum(R)={sum_R} not finite"
    ratio = sum_R / logit_val if abs(logit_val) > 1e-8 else float("nan")
    assert abs(ratio) < 100, (
        f"conservation ratio {ratio} > 100; sum(R)={sum_R}, logit={logit_val}"
    )


# ── Eva-stack (unfolded) path: substitution + concept conditioning ──────────


class TestEvaUnfoldedIntegration:
    """End-to-end tests of the unfolded substitution + new concept classes
    on a small Eva-stack ViT. Random weights — we test plumbing, shape,
    and finiteness, not numerical accuracy."""

    @pytest.fixture
    def img224(self):
        torch.manual_seed(0)
        return torch.randn(1, 3, 224, 224, requires_grad=True)

    def test_substitution_canonizer_replaces_attention(self, eva_tiny):
        """After EvaAttentionSubstitutionCanonizer applies, every block's
        .attn must be an EvaAttentionUnfolded instance, then restored on
        remove()."""
        sub = EvaAttentionSubstitutionCanonizer(block_indices=None)
        instances = sub.apply(eva_tiny)
        try:
            for i, block in enumerate(eva_tiny.blocks):
                assert isinstance(block.attn, EvaAttentionUnfolded), (
                    f"block {i}.attn not substituted: {type(block.attn).__name__}"
                )
        finally:
            for inst in instances:
                inst.remove()
        for block in eva_tiny.blocks:
            assert not isinstance(block.attn, EvaAttentionUnfolded), (
                "substitution canonizer did not restore on remove()"
            )

    def test_combined_composite_with_unfolded_runs_on_eva(self, eva_tiny, img224):
        composite = AttnLRPCombinedComposite(
            alpha=0.5, beta=0.5,
            layerscale_uniform=True,
            residual_lrp="ratio",
        )
        # Resize input if needed.
        H = W = eva_tiny.default_cfg.get("input_size", (3, 224, 224))[-1]
        if H != 224:
            img224 = torch.randn(1, 3, H, W, requires_grad=True)
        attribution = CondAttribution(eva_tiny)
        res = attribution(img224, [{"y": [42]}], composite)
        assert res.heatmap.shape[-1] == H

    def test_head_concept_attribution_on_eva(self, eva_tiny, img224):
        composite = AttnLRPCombinedComposite(
            alpha=0.5, beta=0.5, layerscale_uniform=True, residual_lrp="ratio",
        )
        # HeadConcept now operates on 3D `(B, N, embed_dim)` tensors
        # at any LRP inspection site. Hookable at q_lrp_probe / k_lrp_probe
        # / v_lrp_probe (post-qkv-split, pre-reshape) or at proj_drop
        # (attention output). Construction is model-free; num_heads passed
        # explicitly.
        num_heads = int(eva_tiny.blocks[0].attn.num_heads)
        concept = HeadConcept(num_heads=num_heads)
        n_blocks = len(eva_tiny.blocks)
        target_block = n_blocks // 2
        layer = f"blocks.{target_block}.attn.proj_drop"
        H = eva_tiny.default_cfg.get("input_size", (3, 224, 224))[-1]
        if img224.shape[-1] != H:
            img224 = torch.randn(1, 3, H, H, requires_grad=True)
        attribution = CondAttribution(eva_tiny)
        res = attribution(
            img224, [{layer: [1], "y": [42]}], composite, mask_map=concept.mask,
        )
        assert res.heatmap.shape[-1] == H

    def test_head_concept_at_q_lrp_probe(self, eva_tiny, img224):
        """HeadConcept at the q_lrp_probe site — same shape contract as
        proj_drop, different semantic interpretation (which heads' query
        subspace was populated by which input pixels)."""
        composite = AttnLRPCombinedComposite(
            alpha=0.5, beta=0.5, layerscale_uniform=True, residual_lrp="ratio",
        )
        num_heads = int(eva_tiny.blocks[0].attn.num_heads)
        concept = HeadConcept(num_heads=num_heads)
        n_blocks = len(eva_tiny.blocks)
        target_block = n_blocks // 2
        layer = f"blocks.{target_block}.attn.q_lrp_probe"
        H = eva_tiny.default_cfg.get("input_size", (3, 224, 224))[-1]
        if img224.shape[-1] != H:
            img224 = torch.randn(1, 3, H, H, requires_grad=True)
        attribution = CondAttribution(eva_tiny)
        res = attribution(
            img224, [{layer: [0], "y": [42]}], composite, mask_map=concept.mask,
        )
        assert res.heatmap.shape[-1] == H

    def test_embedding_dim_concept_at_proj_drop(self, eva_tiny, img224):
        composite = AttnLRPCombinedComposite(
            alpha=0.5, beta=0.5, layerscale_uniform=True, residual_lrp="ratio",
        )
        num_heads = int(eva_tiny.blocks[0].attn.num_heads)
        concept = EmbeddingDimConcept(num_heads=num_heads)
        n_blocks = len(eva_tiny.blocks)
        target_block = n_blocks // 2
        layer = f"blocks.{target_block}.attn.proj_drop"
        H = eva_tiny.default_cfg.get("input_size", (3, 224, 224))[-1]
        if img224.shape[-1] != H:
            img224 = torch.randn(1, 3, H, H, requires_grad=True)
        attribution = CondAttribution(eva_tiny)
        res = attribution(
            img224, [{layer: [0, 5, 10], "y": [42]}], composite,
            mask_map=concept.mask,
        )
        assert res.heatmap.shape[-1] == H

    def test_token_concept_at_proj_drop(self, eva_tiny, img224):
        """TokenConcept addresses individual token positions at proj_drop.
        Model-free constructor; concept ids index positions in the
        post-filter universe (default = all tokens)."""
        composite = AttnLRPCombinedComposite(
            alpha=0.5, beta=0.5, layerscale_uniform=True, residual_lrp="ratio",
        )
        concept = TokenConcept()
        n_blocks = len(eva_tiny.blocks)
        target_block = n_blocks // 2
        layer = f"blocks.{target_block}.attn.proj_drop"
        H = eva_tiny.default_cfg.get("input_size", (3, 224, 224))[-1]
        if img224.shape[-1] != H:
            img224 = torch.randn(1, 3, H, H, requires_grad=True)
        attribution = CondAttribution(eva_tiny)
        # Condition on token position 0 (cls token by convention).
        res = attribution(
            img224, [{layer: [0], "y": [42]}], composite,
            mask_map=concept.mask,
        )
        assert res.heatmap.shape[-1] == H
