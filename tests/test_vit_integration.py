"""ViT integration tests for the new attention concept classes.

These exercise the full pipeline: model load → tap injection → monkey-patch →
CRP attribution → shape and basic-conservation checks.

Skipped if ``timm`` and/or ``zennit`` are not installed (e.g. on dev hosts
without those optional dependencies).

Run with::

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
from crp.transformer_patches import inject_qkv_taps, prepare_timm_vit


# ── module-scope fixtures (one model load per session) ────────────────────────


@pytest.fixture(scope="module")
def vit_tiny():
    """Smallest readily-available ViT variant for fast tests.

    `vit_tiny_patch16_224` has 12 blocks, num_heads=3, head_dim=64. Avoids the
    pretrained-weight download by initialising randomly.
    """
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def patched_vit(vit_tiny):
    prepare_timm_vit(vit_tiny)
    return vit_tiny


@pytest.fixture
def img_batch():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224, requires_grad=True)


# ── tap injection ─────────────────────────────────────────────────────────────


class TestTapInjection:
    def test_taps_added_to_every_attention(self, vit_tiny):
        # Use a fresh copy so other tests aren't affected
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        tapped = inject_qkv_taps(model)
        # vit_tiny has 12 blocks
        assert len(tapped) == 12
        for name, _ in tapped:
            module = dict(model.named_modules())[name]
            assert isinstance(module.qkv_tap, torch.nn.Identity)

    def test_idempotent(self, vit_tiny):
        first = inject_qkv_taps(vit_tiny)
        second = inject_qkv_taps(vit_tiny)
        assert len(first) == len(second)


# ── forward parity ────────────────────────────────────────────────────────────


def test_patched_forward_runs(patched_vit, img_batch):
    """Patched forward must complete without error and yield correct shape."""
    out = patched_vit(img_batch)
    assert out.shape == (1, 1000)  # ImageNet head


# ── concept attribution end-to-end ────────────────────────────────────────────


def _attribute(model, concept, conditions, data):
    attribution = CondAttribution(model)
    composite = None  # default zennit composite from the user; not strictly needed for this smoke
    return attribution(data, conditions, composite, mask_map=concept.mask)


class TestEndToEndShapes:
    """One conditional pass per concept def. Verifies the pipeline doesn't blow
    up and returns a heatmap of input shape. Numerical correctness is a
    follow-on task (see ``IMPLEMENTATION_PLAN.md`` Phase 5)."""

    LAYER_NAME = "blocks.6.attn.qkv_tap"

    def test_head_concept(self, patched_vit, img_batch):
        c = HeadConcept()
        c.register_from_model(patched_vit)
        conditions = [{self.LAYER_NAME: [0], "y": [42]}]  # head 0, target class 42
        result = _attribute(patched_vit, c, conditions, img_batch)
        assert result.heatmap.shape == img_batch.shape[1:]  # (3, 224, 224)

    def test_kqv_concept(self, patched_vit, img_batch):
        c = KQVConcept()
        c.register_from_model(patched_vit)
        conditions = [{self.LAYER_NAME: ["q"], "y": [42]}]
        result = _attribute(patched_vit, c, conditions, img_batch)
        assert result.heatmap.shape == img_batch.shape[1:]

    def test_kqv_head_concept(self, patched_vit, img_batch):
        c = KQVHeadConcept()
        c.register_from_model(patched_vit)
        conditions = [{self.LAYER_NAME: [("k", 1)], "y": [42]}]
        result = _attribute(patched_vit, c, conditions, img_batch)
        assert result.heatmap.shape == img_batch.shape[1:]

    def test_head_dim_concept(self, patched_vit, img_batch):
        c = HeadDimConcept()
        c.register_from_model(patched_vit)
        conditions = [{self.LAYER_NAME: [("v", 0, 0)], "y": [42]}]
        result = _attribute(patched_vit, c, conditions, img_batch)
        assert result.heatmap.shape == img_batch.shape[1:]


# ── per-concept relevance shapes ──────────────────────────────────────────────


class TestRelevanceShapes:
    """Verify the per-concept relevance scores from each concept's attribute()
    have the right shape after a real backward pass."""

    LAYER_NAME = "blocks.6.attn.qkv_tap"

    def _record_relevance(self, model, concept, data, layer_name, raw_concept_id):
        attribution = CondAttribution(model)
        conditions = [{layer_name: [raw_concept_id], "y": [42]}]
        result = attribution(
            data,
            conditions,
            composite=None,
            mask_map=concept.mask,
            record_layer=[layer_name],
        )
        rel = result.relevances[layer_name]
        return concept.attribute(rel, layer_name=layer_name, abs_norm=False)

    def test_head_concept_shape(self, patched_vit, img_batch):
        c = HeadConcept()
        c.register_from_model(patched_vit)
        scores = self._record_relevance(
            patched_vit, c, img_batch, self.LAYER_NAME, 0
        )
        # vit_tiny has 3 heads
        assert scores.shape == (1, 3)

    def test_head_dim_shape(self, patched_vit, img_batch):
        c = HeadDimConcept()
        c.register_from_model(patched_vit)
        scores = self._record_relevance(
            patched_vit, c, img_batch, self.LAYER_NAME, ("q", 0, 0)
        )
        assert scores.shape == (1, 3, 3, 64)
