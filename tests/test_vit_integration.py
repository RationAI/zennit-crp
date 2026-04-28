"""ViT integration tests for the AttnLRP Canonizer + Hook + Composite stack.

Exercise the full pipeline: model load → composite (canonizer applies) →
CondAttribution → mask hook on ``qkv_tap`` → shape checks. The composite is
context-managed so each test starts and ends with a clean (uncanonised) model.

Skipped if ``timm`` / ``zennit`` are missing.

Run::

    uv run pytest tests/test_vit_integration.py -v
"""

import pytest
import torch

timm = pytest.importorskip("timm")
zennit = pytest.importorskip("zennit")


from crp.attention_concepts import (
    HeadConcept,
    KQVConcept,
    KQVHeadConcept,
    HeadDimConcept,
)
from crp.attribution import CondAttribution
from crp.transformer_patches import (
    AttnLRPEpsilonComposite,
    AttnLRPGammaComposite,
    QKVTapCanonizer,
    TimmViTCanonizer,
    timm_attention_forward,
)


# ── module-scope fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vit_tiny():
    """Smallest readily-available ViT for fast tests. ``vit_tiny_patch16_224``:
    12 blocks, num_heads=3, head_dim=64. Random init avoids weight download."""
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.eval()
    return model


@pytest.fixture
def img_batch():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224, requires_grad=True)


# ── canonizer mechanics: register / remove cycle ──────────────────────────────


class TestQKVTapCanonizer:
    def test_register_adds_qkv_tap_to_every_attention(self, vit_tiny):
        canonizer = QKVTapCanonizer()
        instances = canonizer.apply(vit_tiny)
        try:
            # vit_tiny has 12 attention blocks
            assert len(instances) == 12
            attn_modules = [
                m for m in vit_tiny.modules()
                if hasattr(m, "qkv") and isinstance(m.qkv, torch.nn.Linear)
                and hasattr(m, "num_heads") and hasattr(m, "head_dim")
            ]
            for m in attn_modules:
                assert isinstance(m.qkv_tap, torch.nn.Identity)
        finally:
            for inst in instances:
                inst.remove()

    def test_remove_reverts_qkv_tap(self, vit_tiny):
        canonizer = QKVTapCanonizer()
        instances = canonizer.apply(vit_tiny)
        for inst in instances:
            inst.remove()
        attn = vit_tiny.blocks[0].attn
        assert not hasattr(attn, "qkv_tap")

    def test_idempotent_with_pre_injected_tap(self, vit_tiny):
        # User pre-injects a tap manually; canonizer must respect it and not
        # delete it on remove.
        attn = vit_tiny.blocks[0].attn
        attn.add_module("qkv_tap", torch.nn.Identity())
        canonizer = QKVTapCanonizer()
        instances = canonizer.apply(vit_tiny)
        for inst in instances:
            inst.remove()
        # block 0 had pre-existing tap → still present.
        assert isinstance(attn.qkv_tap, torch.nn.Identity)
        # other blocks had canonizer-added taps → removed.
        attn_other = vit_tiny.blocks[1].attn
        assert not hasattr(attn_other, "qkv_tap")
        # Cleanup for downstream tests.
        del attn._modules["qkv_tap"]


class TestTimmViTCanonizer:
    def test_forward_swap_is_reversible(self, vit_tiny):
        attn = vit_tiny.blocks[0].attn
        original_class_forward = type(attn).forward
        canonizer = TimmViTCanonizer()
        instances = canonizer.apply(vit_tiny)
        try:
            # forward attribute on instance points at our patched function
            assert attn.forward.__func__ is timm_attention_forward
        finally:
            for inst in instances:
                inst.remove()
        # After remove: instance attribute deleted, class-level forward returns.
        assert "forward" not in attn.__dict__
        assert type(attn).forward is original_class_forward


# ── forward parity (model still callable inside composite context) ────────────


def test_forward_runs_under_composite(vit_tiny, img_batch):
    composite = AttnLRPEpsilonComposite()
    with composite.context(vit_tiny) as modified:
        out = modified(img_batch)
    assert out.shape == (1, 1000)


def test_forward_runs_under_gamma_composite(vit_tiny, img_batch):
    composite = AttnLRPGammaComposite()
    with composite.context(vit_tiny) as modified:
        out = modified(img_batch)
    assert out.shape == (1, 1000)


def test_gamma_composite_attribution_end_to_end(vit_tiny, img_batch):
    """γ-LRP composite must produce a pixel-space heatmap of the right shape
    when paired with a HeadConcept mask."""
    c = HeadConcept()
    c.register_from_model(vit_tiny)
    attribution = CondAttribution(vit_tiny)
    composite = AttnLRPGammaComposite()
    conditions = [{LAYER_NAME: [0], "y": [42]}]
    result = attribution(img_batch, conditions, composite, mask_map=c.mask)
    B, _, H, W = img_batch.shape
    assert result.heatmap.shape == (B, H, W)


def test_gamma_differs_from_epsilon(vit_tiny, img_batch):
    """Numerical sanity: γ-LRP and ε-LRP should produce different heatmaps
    on the same input + concept (γ biases toward positive contributions)."""
    c = HeadConcept()
    c.register_from_model(vit_tiny)
    attribution = CondAttribution(vit_tiny)
    conditions = [{LAYER_NAME: [0], "y": [42]}]

    img_eps = img_batch.detach().clone().requires_grad_(True)
    eps_result = attribution(
        img_eps, conditions, AttnLRPEpsilonComposite(), mask_map=c.mask
    )
    img_gam = img_batch.detach().clone().requires_grad_(True)
    gam_result = attribution(
        img_gam, conditions, AttnLRPGammaComposite(gamma=0.25), mask_map=c.mask
    )
    assert not torch.allclose(eps_result.heatmap, gam_result.heatmap, atol=1e-6)


# ── concept attribution end-to-end ────────────────────────────────────────────


LAYER_NAME = "blocks.6.attn.qkv_tap"


def _attribute(model, concept, conditions, data):
    attribution = CondAttribution(model)
    composite = AttnLRPEpsilonComposite()
    return attribution(data, conditions, composite, mask_map=concept.mask)


class TestEndToEndShapes:
    """One conditional pass per concept granularity. Verifies the pipeline
    runs end-to-end and returns a pixel-space heatmap of the right shape."""

    def test_head_concept(self, vit_tiny, img_batch):
        c = HeadConcept()
        c.register_from_model(vit_tiny)
        # Note: register_from_model walks named_modules, but qkv_tap doesn't
        # exist yet (canonizer hasn't applied). The concept registers under
        # both "blocks.X.attn" and "blocks.X.attn.qkv_tap" based on parent
        # match — see _BaseAttentionConcept.register_from_model.
        conditions = [{LAYER_NAME: [0], "y": [42]}]
        result = _attribute(vit_tiny, c, conditions, img_batch)
        B, _, H, W = img_batch.shape
        assert result.heatmap.shape == (B, H, W)

    def test_kqv_concept(self, vit_tiny, img_batch):
        c = KQVConcept()
        c.register_from_model(vit_tiny)
        conditions = [{LAYER_NAME: ["q"], "y": [42]}]
        result = _attribute(vit_tiny, c, conditions, img_batch)
        B, _, H, W = img_batch.shape
        assert result.heatmap.shape == (B, H, W)

    def test_kqv_head_concept(self, vit_tiny, img_batch):
        c = KQVHeadConcept()
        c.register_from_model(vit_tiny)
        conditions = [{LAYER_NAME: [("k", 1)], "y": [42]}]
        result = _attribute(vit_tiny, c, conditions, img_batch)
        B, _, H, W = img_batch.shape
        assert result.heatmap.shape == (B, H, W)

    def test_head_dim_concept(self, vit_tiny, img_batch):
        c = HeadDimConcept()
        c.register_from_model(vit_tiny)
        conditions = [{LAYER_NAME: [("v", 0, 0)], "y": [42]}]
        result = _attribute(vit_tiny, c, conditions, img_batch)
        B, _, H, W = img_batch.shape
        assert result.heatmap.shape == (B, H, W)


# ── per-concept relevance shapes ──────────────────────────────────────────────


class TestRelevanceShapes:
    """Per-concept relevance from ``concept.attribute()`` after a real backward."""

    def _record_relevance(self, model, concept, data, layer_name, raw_concept_id):
        attribution = CondAttribution(model)
        composite = AttnLRPEpsilonComposite()
        conditions = [{layer_name: [raw_concept_id], "y": [42]}]
        result = attribution(
            data,
            conditions,
            composite,
            mask_map=concept.mask,
            record_layer=[layer_name],
        )
        rel = result.relevances[layer_name]
        return concept.attribute(rel, layer_name=layer_name, abs_norm=False)

    def test_head_concept_shape(self, vit_tiny, img_batch):
        c = HeadConcept()
        c.register_from_model(vit_tiny)
        scores = self._record_relevance(vit_tiny, c, img_batch, LAYER_NAME, 0)
        # vit_tiny has 3 heads
        assert scores.shape == (1, 3)

    def test_head_dim_shape(self, vit_tiny, img_batch):
        c = HeadDimConcept()
        c.register_from_model(vit_tiny)
        scores = self._record_relevance(
            vit_tiny, c, img_batch, LAYER_NAME, ("q", 0, 0)
        )
        assert scores.shape == (1, 3, 3, 64)
