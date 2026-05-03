"""Unit tests for crp.attention_concepts on the unfolded attention API.

Pure tensor-level — no model load, no real attention forward — so they
run anywhere torch is installed. The tests construct synthetic relevance
tensors of the shape each concept expects and exercise mask + attribute
+ shape semantics.

Concepts under test:
* :class:`HeadConcept` (per-head, optional dim_split) on `attn.context`
* :class:`QConcept`, :class:`KConcept`, :class:`VConcept`
  (Q/K/V per-head, optional dim_split) on `attn.rope_q`/`attn.rope_k`/`attn.v_id`
* :class:`AttnWeightConcept` on `attn.softmax`
* :class:`AttnOutputConcept` on `attn.proj_drop`
"""
from __future__ import annotations

import pytest
import torch

from crp.attention_concepts import (
    HeadConcept,
    QConcept,
    KConcept,
    VConcept,
    AttnWeightConcept,
    AttnOutputConcept,
)


# ── shared fixtures ──────────────────────────────────────────────────────────

B, NUM_HEADS, N, HEAD_DIM = 2, 4, 7, 6
EMBED_DIM = NUM_HEADS * HEAD_DIM


@pytest.fixture
def per_head_relevance():
    """(B, num_heads, N, head_dim) — the shape used by HeadConcept and Q/K/V."""
    torch.manual_seed(0)
    return torch.randn(B, NUM_HEADS, N, HEAD_DIM)


@pytest.fixture
def attn_weight_relevance():
    """(B, num_heads, N, N) — the shape after softmax."""
    torch.manual_seed(1)
    return torch.randn(B, NUM_HEADS, N, N)


@pytest.fixture
def proj_output_relevance():
    """(B, N, embed_dim) — the shape at attention's residual contribution."""
    torch.manual_seed(2)
    return torch.randn(B, N, EMBED_DIM)


# ── HeadConcept ──────────────────────────────────────────────────────────────


class TestHeadConcept:
    LAYER = "blocks.0.attn.context"

    def _make(self, dim_split=False):
        c = HeadConcept(dim_split=dim_split)
        c.register_layer(self.LAYER, NUM_HEADS, HEAD_DIM)
        return c

    def test_per_head_mask_zeros_other_heads(self, per_head_relevance):
        c = self._make()
        m = c.mask(batch_id=0, concept_ids=[1], layer_name=self.LAYER)
        masked = m(per_head_relevance.clone())
        # Batch 0, head 1 should match original; other heads zero.
        assert torch.equal(masked[0, 1], per_head_relevance[0, 1])
        for h in (0, 2, 3):
            assert torch.all(masked[0, h] == 0)
        # Batch 1 untouched.
        assert torch.equal(masked[1], per_head_relevance[1])

    def test_per_head_attribute_shape(self, per_head_relevance):
        c = self._make()
        rel = c.attribute(per_head_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, NUM_HEADS)
        # Equals sum over (N, head_dim).
        expected = per_head_relevance.sum(dim=(2, 3))
        assert torch.allclose(rel, expected)

    def test_dim_split_mask_zeros_other_dims(self, per_head_relevance):
        c = self._make(dim_split=True)
        m = c.mask(batch_id=0, concept_ids=[(2, 3)], layer_name=self.LAYER)
        masked = m(per_head_relevance.clone())
        # Only (head=2, dim=3) survives in batch 0.
        for h in range(NUM_HEADS):
            for d in range(HEAD_DIM):
                if h == 2 and d == 3:
                    assert torch.equal(masked[0, h, :, d], per_head_relevance[0, h, :, d])
                else:
                    assert torch.all(masked[0, h, :, d] == 0)

    def test_dim_split_flat_int_id(self, per_head_relevance):
        c = self._make(dim_split=True)
        # flat id 14 = head 2, dim 2 (with HEAD_DIM=6)
        m = c.mask(batch_id=0, concept_ids=[14], layer_name=self.LAYER)
        masked = m(per_head_relevance.clone())
        assert torch.equal(masked[0, 2, :, 2], per_head_relevance[0, 2, :, 2])
        assert torch.all(masked[0, 2, :, 0] == 0)

    def test_dim_split_attribute_shape(self, per_head_relevance):
        c = self._make(dim_split=True)
        rel = c.attribute(per_head_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, NUM_HEADS, HEAD_DIM)

    def test_abs_norm_sums_to_one(self, per_head_relevance):
        c = self._make()
        rel = c.attribute(per_head_relevance, layer_name=self.LAYER, abs_norm=True)
        per_batch_sum = rel.abs().sum(dim=-1)
        assert torch.allclose(per_batch_sum, torch.ones_like(per_batch_sum), atol=1e-5)

    def test_invalid_head_index_raises(self):
        c = self._make()
        with pytest.raises(IndexError):
            c.mask(batch_id=0, concept_ids=[NUM_HEADS], layer_name=self.LAYER)

    def test_unregistered_layer_raises(self):
        c = HeadConcept()
        with pytest.raises(ValueError):
            c.mask(batch_id=0, concept_ids=[0], layer_name="nope.attn.context")

    def test_layer_suffix(self):
        assert HeadConcept.LAYER_SUFFIX == "context"


# ── Q / K / V concepts ───────────────────────────────────────────────────────


class TestQKVConcepts:
    """Three concepts share base, only LAYER_SUFFIX differs. Smoke-test
    each + that they target the right submodule name."""

    @pytest.mark.parametrize("cls,suffix", [
        (QConcept, "rope_q"),
        (KConcept, "rope_k"),
        (VConcept, "v_id"),
    ])
    def test_layer_suffix(self, cls, suffix):
        assert cls.LAYER_SUFFIX == suffix

    @pytest.mark.parametrize("cls", [QConcept, KConcept, VConcept])
    def test_per_head_mask(self, cls, per_head_relevance):
        c = cls()
        layer = f"blocks.0.attn.{cls.LAYER_SUFFIX}"
        c.register_layer(layer, NUM_HEADS, HEAD_DIM)
        m = c.mask(batch_id=1, concept_ids=[2], layer_name=layer)
        masked = m(per_head_relevance.clone())
        assert torch.equal(masked[1, 2], per_head_relevance[1, 2])
        for h in (0, 1, 3):
            assert torch.all(masked[1, h] == 0)
        assert torch.equal(masked[0], per_head_relevance[0])

    @pytest.mark.parametrize("cls", [QConcept, KConcept, VConcept])
    def test_attribute_shape(self, cls, per_head_relevance):
        c = cls()
        layer = f"blocks.0.attn.{cls.LAYER_SUFFIX}"
        c.register_layer(layer, NUM_HEADS, HEAD_DIM)
        rel = c.attribute(per_head_relevance, layer_name=layer, abs_norm=False)
        assert rel.shape == (B, NUM_HEADS)

    @pytest.mark.parametrize("cls", [QConcept, KConcept, VConcept])
    def test_dim_split(self, cls, per_head_relevance):
        c = cls(dim_split=True)
        layer = f"blocks.0.attn.{cls.LAYER_SUFFIX}"
        c.register_layer(layer, NUM_HEADS, HEAD_DIM)
        rel = c.attribute(per_head_relevance, layer_name=layer, abs_norm=False)
        assert rel.shape == (B, NUM_HEADS, HEAD_DIM)


# ── AttnWeightConcept ───────────────────────────────────────────────────────


class TestAttnWeightConcept:
    LAYER = "blocks.0.attn.softmax"

    def _make(self, granularity="head"):
        c = AttnWeightConcept(granularity=granularity)
        c.register_layer(self.LAYER, NUM_HEADS)
        return c

    def test_head_granularity_mask_zeros_other_heads(self, attn_weight_relevance):
        c = self._make("head")
        m = c.mask(batch_id=0, concept_ids=[2], layer_name=self.LAYER)
        masked = m(attn_weight_relevance.clone())
        assert torch.equal(masked[0, 2], attn_weight_relevance[0, 2])
        for h in (0, 1, 3):
            assert torch.all(masked[0, h] == 0)

    def test_head_granularity_attribute_shape(self, attn_weight_relevance):
        c = self._make("head")
        rel = c.attribute(attn_weight_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, NUM_HEADS)

    def test_head_query_granularity(self, attn_weight_relevance):
        c = self._make("head_query")
        m = c.mask(batch_id=0, concept_ids=[(1, 3)], layer_name=self.LAYER)
        masked = m(attn_weight_relevance.clone())
        # head=1, query=3 row preserved; rest zero.
        assert torch.equal(masked[0, 1, 3], attn_weight_relevance[0, 1, 3])
        for q in (0, 2, 4):
            assert torch.all(masked[0, 1, q] == 0)

    def test_head_query_key_granularity(self, attn_weight_relevance):
        c = self._make("head_query_key")
        m = c.mask(batch_id=0, concept_ids=[(1, 3, 5)], layer_name=self.LAYER)
        masked = m(attn_weight_relevance.clone())
        assert masked[0, 1, 3, 5].item() == attn_weight_relevance[0, 1, 3, 5].item()
        # One cell preserved out of (NUM_HEADS * N * N).
        nonzero = (masked[0] != 0).sum().item()
        assert nonzero == 1

    def test_invalid_granularity(self):
        with pytest.raises(ValueError):
            AttnWeightConcept(granularity="nope")


# ── AttnOutputConcept ───────────────────────────────────────────────────────


class TestAttnOutputConcept:
    LAYER = "blocks.0.attn.proj_drop"

    def _make(self):
        c = AttnOutputConcept()
        c.register_layer(self.LAYER, EMBED_DIM)
        return c

    def test_mask_zeros_other_channels(self, proj_output_relevance):
        c = self._make()
        m = c.mask(batch_id=0, concept_ids=[5, 7], layer_name=self.LAYER)
        masked = m(proj_output_relevance.clone())
        for ch in range(EMBED_DIM):
            if ch in (5, 7):
                assert torch.equal(masked[0, :, ch], proj_output_relevance[0, :, ch])
            else:
                assert torch.all(masked[0, :, ch] == 0)

    def test_attribute_shape(self, proj_output_relevance):
        c = self._make()
        rel = c.attribute(proj_output_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, EMBED_DIM)
        # Equals sum over tokens.
        expected = proj_output_relevance.sum(dim=1)
        assert torch.allclose(rel, expected)

    def test_invalid_channel_raises(self):
        c = self._make()
        with pytest.raises(IndexError):
            c.mask(batch_id=0, concept_ids=[EMBED_DIM], layer_name=self.LAYER)

    def test_layer_suffix(self):
        assert AttnOutputConcept.LAYER_SUFFIX == "proj_drop"
