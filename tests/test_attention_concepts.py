"""Unit tests for crp.attention_concepts on the unfolded attention API.

Pure tensor-level — no real ViT forward — so they run anywhere torch is
installed. We build a minimal stub model that exposes the same attribute
contract the concepts read (``num_heads``, ``head_dim``,
``num_prefix_tokens``) at the parent path the concept's ``layer_name``
points at.

Concepts under test:
* :class:`HeadConcept` / :class:`QConcept` / :class:`KConcept` /
  :class:`VConcept` — per-head, optional ``dim_split``
* :class:`AttnOutputDimConcept` — per-channel, spatial aggregation
* :class:`RegisterTokenConcept` — per-prefix-token at proj_drop
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from crp.attention_concepts import (
    HeadConcept,
    QConcept,
    KConcept,
    VConcept,
    AttnOutputDimConcept,
    RegisterTokenConcept,
)


# ── shared fixtures ──────────────────────────────────────────────────────────

B, NUM_HEADS, N, HEAD_DIM = 2, 4, 9, 6
NUM_PREFIX = 2
EMBED_DIM = NUM_HEADS * HEAD_DIM


class _StubAttn(nn.Module):
    """Bare attention stub: exposes the dim attributes the concepts read.

    Standing in for either timm's stock ``EvaAttention`` or
    :class:`crp.attention_unfolded.EvaAttentionUnfolded` — both expose
    these attributes; the concept methods don't care which class is bound.
    """
    def __init__(self, num_heads: int, head_dim: int, num_prefix_tokens: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_prefix_tokens = num_prefix_tokens


class _StubBlock(nn.Module):
    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn


class _StubModel(nn.Module):
    """One-block ``Probe``-shaped stub so layer paths like
    ``blocks.0.attn.context`` resolve via ``get_submodule('blocks.0.attn')``.
    """
    def __init__(self, num_prefix_tokens: int = 0):
        super().__init__()
        self.blocks = nn.ModuleList([
            _StubBlock(_StubAttn(NUM_HEADS, HEAD_DIM, num_prefix_tokens)),
        ])


@pytest.fixture
def model_no_prefix():
    return _StubModel(num_prefix_tokens=0)


@pytest.fixture
def model_prefix():
    return _StubModel(num_prefix_tokens=NUM_PREFIX)


@pytest.fixture
def per_head_relevance():
    """(B, num_heads, N, head_dim) — shape for HeadConcept and Q/K/V."""
    torch.manual_seed(0)
    return torch.randn(B, NUM_HEADS, N, HEAD_DIM)


@pytest.fixture
def proj_output_relevance():
    """(B, N, embed_dim) — shape at proj_drop."""
    torch.manual_seed(2)
    return torch.randn(B, N, EMBED_DIM)


# ── HeadConcept ──────────────────────────────────────────────────────────────


class TestHeadConcept:
    LAYER = "blocks.0.attn.context"

    def test_per_head_mask_zeros_other_heads(self, model_no_prefix, per_head_relevance):
        c = HeadConcept(model_no_prefix)
        m = c.mask(batch_id=0, concept_ids=[1], layer_name=self.LAYER)
        masked = m(per_head_relevance.clone())
        assert torch.equal(masked[0, 1], per_head_relevance[0, 1])
        for h in (0, 2, 3):
            assert torch.all(masked[0, h] == 0)
        assert torch.equal(masked[1], per_head_relevance[1])

    def test_per_head_attribute_shape(self, model_no_prefix, per_head_relevance):
        c = HeadConcept(model_no_prefix)
        rel = c.attribute(per_head_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, NUM_HEADS)
        expected = per_head_relevance.sum(dim=(2, 3))
        assert torch.allclose(rel, expected)

    def test_dim_split_mask_zeros_other_dims(self, model_no_prefix, per_head_relevance):
        c = HeadConcept(model_no_prefix, dim_split=True)
        m = c.mask(batch_id=0, concept_ids=[(2, 3)], layer_name=self.LAYER)
        masked = m(per_head_relevance.clone())
        for h in range(NUM_HEADS):
            for d in range(HEAD_DIM):
                if h == 2 and d == 3:
                    assert torch.equal(masked[0, h, :, d], per_head_relevance[0, h, :, d])
                else:
                    assert torch.all(masked[0, h, :, d] == 0)

    def test_dim_split_flat_int_id(self, model_no_prefix, per_head_relevance):
        c = HeadConcept(model_no_prefix, dim_split=True)
        # flat 14 = head 2, dim 2 (HEAD_DIM=6)
        m = c.mask(batch_id=0, concept_ids=[14], layer_name=self.LAYER)
        masked = m(per_head_relevance.clone())
        assert torch.equal(masked[0, 2, :, 2], per_head_relevance[0, 2, :, 2])
        assert torch.all(masked[0, 2, :, 0] == 0)

    def test_dim_split_attribute_shape(self, model_no_prefix, per_head_relevance):
        c = HeadConcept(model_no_prefix, dim_split=True)
        rel = c.attribute(per_head_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, NUM_HEADS, HEAD_DIM)

    def test_abs_norm_sums_to_one(self, model_no_prefix, per_head_relevance):
        c = HeadConcept(model_no_prefix)
        rel = c.attribute(per_head_relevance, layer_name=self.LAYER, abs_norm=True)
        per_batch_sum = rel.abs().sum(dim=-1)
        assert torch.allclose(per_batch_sum, torch.ones_like(per_batch_sum), atol=1e-5)

    def test_invalid_head_index_raises(self, model_no_prefix):
        c = HeadConcept(model_no_prefix)
        with pytest.raises(IndexError):
            c.mask(batch_id=0, concept_ids=[NUM_HEADS], layer_name=self.LAYER)

    def test_bad_parent_path_raises(self, model_no_prefix):
        """layer_name pointing at a non-existent submodule must fail loudly."""
        c = HeadConcept(model_no_prefix)
        with pytest.raises((AttributeError, KeyError)):
            c.mask(batch_id=0, concept_ids=[0], layer_name="nope.attn.context")


# ── Q / K / V concepts ───────────────────────────────────────────────────────


class TestQKVConcepts:
    """Three concepts share base. Smoke-test each."""

    @pytest.mark.parametrize("cls,leaf", [
        (QConcept, "rope_q"),
        (KConcept, "rope_k"),
        (VConcept, "v_id"),
    ])
    def test_per_head_mask(self, model_no_prefix, per_head_relevance, cls, leaf):
        c = cls(model_no_prefix)
        layer = f"blocks.0.attn.{leaf}"
        m = c.mask(batch_id=1, concept_ids=[2], layer_name=layer)
        masked = m(per_head_relevance.clone())
        assert torch.equal(masked[1, 2], per_head_relevance[1, 2])
        for h in (0, 1, 3):
            assert torch.all(masked[1, h] == 0)
        assert torch.equal(masked[0], per_head_relevance[0])

    @pytest.mark.parametrize("cls,leaf", [
        (QConcept, "rope_q"), (KConcept, "rope_k"), (VConcept, "v_id"),
    ])
    def test_attribute_shape(self, model_no_prefix, per_head_relevance, cls, leaf):
        c = cls(model_no_prefix)
        rel = c.attribute(per_head_relevance, layer_name=f"blocks.0.attn.{leaf}", abs_norm=False)
        assert rel.shape == (B, NUM_HEADS)

    @pytest.mark.parametrize("cls,leaf", [
        (QConcept, "rope_q"), (KConcept, "rope_k"), (VConcept, "v_id"),
    ])
    def test_dim_split(self, model_no_prefix, per_head_relevance, cls, leaf):
        c = cls(model_no_prefix, dim_split=True)
        rel = c.attribute(per_head_relevance, layer_name=f"blocks.0.attn.{leaf}", abs_norm=False)
        assert rel.shape == (B, NUM_HEADS, HEAD_DIM)


# ── AttnOutputDimConcept ─────────────────────────────────────────────────────


class TestAttnOutputDimConcept:
    """Per-channel conditioning at the post-projection residual stream.
    Spatial token axis aggregated in attribute (no per-token conditioning).
    """
    LAYER = "blocks.0.attn.proj_drop"

    def test_mask_zeros_other_channels(self, model_no_prefix, proj_output_relevance):
        c = AttnOutputDimConcept(model_no_prefix)
        m = c.mask(batch_id=0, concept_ids=[5, 7], layer_name=self.LAYER)
        masked = m(proj_output_relevance.clone())
        for ch in range(EMBED_DIM):
            if ch in (5, 7):
                assert torch.equal(masked[0, :, ch], proj_output_relevance[0, :, ch])
            else:
                assert torch.all(masked[0, :, ch] == 0)

    def test_attribute_shape_sums_tokens(self, model_no_prefix, proj_output_relevance):
        c = AttnOutputDimConcept(model_no_prefix)
        rel = c.attribute(proj_output_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, EMBED_DIM)
        expected = proj_output_relevance.sum(dim=1)
        assert torch.allclose(rel, expected)

    def test_invalid_channel_raises(self, model_no_prefix):
        c = AttnOutputDimConcept(model_no_prefix)
        with pytest.raises(IndexError):
            c.mask(batch_id=0, concept_ids=[EMBED_DIM], layer_name=self.LAYER)


# ── Prefix-token isolation in spatial concepts ──────────────────────────────


class TestPrefixIsolation:
    """When num_prefix_tokens > 0, the spatial concepts must:
    1. zero out the prefix tokens in the mask (no relevance flows through them);
    2. exclude prefix tokens from the attribute() spatial sum.
    """
    LAYER_HEAD = "blocks.0.attn.context"
    LAYER_OUT = "blocks.0.attn.proj_drop"

    def test_head_concept_mask_zeros_prefix(self, model_prefix, per_head_relevance):
        c = HeadConcept(model_prefix)
        m = c.mask(batch_id=0, concept_ids=[1], layer_name=self.LAYER_HEAD)
        masked = m(per_head_relevance.clone())
        assert torch.all(masked[0, 1, :NUM_PREFIX, :] == 0), (
            "HeadConcept did not zero prefix tokens of the selected head"
        )
        assert torch.equal(
            masked[0, 1, NUM_PREFIX:, :], per_head_relevance[0, 1, NUM_PREFIX:, :],
        )
        for h in (0, 2, 3):
            assert torch.all(masked[0, h] == 0)

    def test_head_concept_attribute_excludes_prefix(self, model_prefix, model_no_prefix, per_head_relevance):
        c = HeadConcept(model_prefix)
        rel = c.attribute(per_head_relevance, layer_name=self.LAYER_HEAD, abs_norm=False)
        expected = per_head_relevance[:, :, NUM_PREFIX:, :].sum(dim=(2, 3))
        assert torch.allclose(rel, expected)
        # Sanity: differs from the no-prefix version.
        c0 = HeadConcept(model_no_prefix)
        rel0 = c0.attribute(per_head_relevance, layer_name=self.LAYER_HEAD, abs_norm=False)
        assert not torch.allclose(rel, rel0)

    def test_attn_output_dim_mask_zeros_prefix(self, model_prefix, proj_output_relevance):
        c = AttnOutputDimConcept(model_prefix)
        m = c.mask(batch_id=0, concept_ids=[3], layer_name=self.LAYER_OUT)
        masked = m(proj_output_relevance.clone())
        assert torch.all(masked[0, :NUM_PREFIX, 3] == 0)
        assert torch.equal(
            masked[0, NUM_PREFIX:, 3], proj_output_relevance[0, NUM_PREFIX:, 3],
        )

    def test_attn_output_dim_attribute_excludes_prefix(self, model_prefix, proj_output_relevance):
        c = AttnOutputDimConcept(model_prefix)
        rel = c.attribute(proj_output_relevance, layer_name=self.LAYER_OUT, abs_norm=False)
        expected = proj_output_relevance[:, NUM_PREFIX:, :].sum(dim=1)
        assert torch.allclose(rel, expected)


# ── RegisterTokenConcept ─────────────────────────────────────────────────────


class TestRegisterTokenConcept:
    """Per-prefix-token conditioning at proj_drop."""
    LAYER = "blocks.0.attn.proj_drop"

    def test_per_token_mask_zeros_other_tokens(self, model_prefix, proj_output_relevance):
        c = RegisterTokenConcept(model_prefix)
        m = c.mask(batch_id=0, concept_ids=[1], layer_name=self.LAYER)
        masked = m(proj_output_relevance.clone())
        assert torch.equal(masked[0, 1, :], proj_output_relevance[0, 1, :])
        assert torch.all(masked[0, 0, :] == 0)
        # All spatial tokens — zeroed (out of scope for register concept).
        assert torch.all(masked[0, NUM_PREFIX:, :] == 0)

    def test_per_token_attribute_shape_sums_channels(self, model_prefix, proj_output_relevance):
        c = RegisterTokenConcept(model_prefix)
        rel = c.attribute(proj_output_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, NUM_PREFIX)
        expected = proj_output_relevance[:, :NUM_PREFIX, :].sum(dim=-1)
        assert torch.allclose(rel, expected)

    def test_dim_split_mask(self, model_prefix, proj_output_relevance):
        c = RegisterTokenConcept(model_prefix, dim_split=True)
        m = c.mask(batch_id=0, concept_ids=[(1, 7)], layer_name=self.LAYER)
        masked = m(proj_output_relevance.clone())
        assert masked[0, 1, 7].item() == proj_output_relevance[0, 1, 7].item()
        nonzero = (masked[0] != 0).sum().item()
        assert nonzero == 1

    def test_dim_split_attribute_shape(self, model_prefix, proj_output_relevance):
        c = RegisterTokenConcept(model_prefix, dim_split=True)
        rel = c.attribute(proj_output_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, NUM_PREFIX, EMBED_DIM)

    def test_dim_split_flat_int_id(self, model_prefix, proj_output_relevance):
        c = RegisterTokenConcept(model_prefix, dim_split=True)
        flat = 1 * EMBED_DIM + 3
        m = c.mask(batch_id=0, concept_ids=[flat], layer_name=self.LAYER)
        masked = m(proj_output_relevance.clone())
        assert masked[0, 1, 3].item() == proj_output_relevance[0, 1, 3].item()
        assert (masked[0] != 0).sum().item() == 1

    def test_invalid_token_raises(self, model_prefix):
        c = RegisterTokenConcept(model_prefix)
        with pytest.raises(IndexError):
            c.mask(batch_id=0, concept_ids=[NUM_PREFIX], layer_name=self.LAYER)

    def test_zero_prefix_attention_rejected(self, model_no_prefix):
        """If the attention has num_prefix_tokens=0, RegisterTokenConcept's
        mask must fail loudly — there are no prefix tokens to address."""
        c = RegisterTokenConcept(model_no_prefix)
        with pytest.raises(ValueError):
            c.mask(batch_id=0, concept_ids=[0], layer_name=self.LAYER)
