"""Unit tests for crp.concepts (HeadConcept / EmbeddingDimConcept / TokenConcept).

All three concepts operate on 3D ``(B, N, embed_dim)`` relevance tensors.
The mask/attribute/reference_sampling APIs mirror upstream
:class:`crp.concepts.ChannelConcept` — concepts are constructed with no
model dependency; ``num_heads`` is the only structural arg
(:class:`HeadConcept` and :class:`EmbeddingDimConcept`); :class:`TokenConcept`
is fully model-free.
"""
from __future__ import annotations

import pytest
import torch

from crp.concepts import HeadConcept, EmbeddingDimConcept, TokenConcept


# ── shared fixtures ──────────────────────────────────────────────────────────

B, N, NUM_HEADS, HEAD_DIM = 2, 9, 4, 6
EMBED_DIM = NUM_HEADS * HEAD_DIM  # 24
NUM_PREFIX = 2  # cls + 1 register, for example


@pytest.fixture
def relevance_3d():
    """Standard 3D relevance tensor at a probe / proj_drop site."""
    torch.manual_seed(0)
    return torch.randn(B, N, EMBED_DIM)


# ── HeadConcept ──────────────────────────────────────────────────────────────


class TestHeadConcept:
    def test_mask_zeros_other_heads(self, relevance_3d):
        c = HeadConcept(num_heads=NUM_HEADS)
        m = c.mask(batch_id=0, concept_ids=[1])
        masked = m(relevance_3d.clone())
        # Head 1 occupies dims [HEAD_DIM, 2*HEAD_DIM).
        assert torch.equal(
            masked[0, :, HEAD_DIM:2 * HEAD_DIM],
            relevance_3d[0, :, HEAD_DIM:2 * HEAD_DIM],
        )
        # Head 0 dims zeroed.
        assert torch.all(masked[0, :, :HEAD_DIM] == 0)
        # Head 2, 3 dims zeroed.
        assert torch.all(masked[0, :, 2 * HEAD_DIM:] == 0)
        # Other batch untouched.
        assert torch.equal(masked[1], relevance_3d[1])

    def test_attribute_shape_and_sum(self, relevance_3d):
        c = HeadConcept(num_heads=NUM_HEADS)
        rel = c.attribute(relevance_3d, abs_norm=False)
        assert rel.shape == (B, NUM_HEADS)
        # Reproduce manually: sum over N AND head_dim per head.
        manual = relevance_3d.reshape(B, N, NUM_HEADS, HEAD_DIM).sum(dim=(1, 3))
        assert torch.allclose(rel, manual)

    def test_token_filter_excludes_prefix(self, relevance_3d):
        c = HeadConcept(num_heads=NUM_HEADS, token_filter=slice(NUM_PREFIX, None))
        rel = c.attribute(relevance_3d, abs_norm=False)
        # Sum only over spatial (N - NUM_PREFIX) tokens.
        manual = (
            relevance_3d[:, NUM_PREFIX:, :]
            .reshape(B, N - NUM_PREFIX, NUM_HEADS, HEAD_DIM)
            .sum(dim=(1, 3))
        )
        assert torch.allclose(rel, manual)

    def test_token_filter_in_mask(self, relevance_3d):
        c = HeadConcept(num_heads=NUM_HEADS, token_filter=slice(NUM_PREFIX, None))
        m = c.mask(batch_id=0, concept_ids=[2])
        masked = m(relevance_3d.clone())
        # Prefix tokens at all dims: zeroed.
        assert torch.all(masked[0, :NUM_PREFIX, :] == 0)
        # Spatial tokens at head 2's slice: kept.
        assert torch.equal(
            masked[0, NUM_PREFIX:, 2 * HEAD_DIM:3 * HEAD_DIM],
            relevance_3d[0, NUM_PREFIX:, 2 * HEAD_DIM:3 * HEAD_DIM],
        )
        # Spatial tokens at other heads' slices: zeroed.
        assert torch.all(masked[0, NUM_PREFIX:, :2 * HEAD_DIM] == 0)
        assert torch.all(masked[0, NUM_PREFIX:, 3 * HEAD_DIM:] == 0)

    def test_invalid_head_index_raises(self):
        c = HeadConcept(num_heads=NUM_HEADS)
        with pytest.raises(IndexError):
            c.mask(batch_id=0, concept_ids=[NUM_HEADS])

    def test_abs_norm_sums_to_one(self, relevance_3d):
        c = HeadConcept(num_heads=NUM_HEADS)
        rel = c.attribute(relevance_3d, abs_norm=True)
        per_batch = rel.abs().sum(dim=-1)
        assert torch.allclose(per_batch, torch.ones_like(per_batch), atol=1e-5)


# ── EmbeddingDimConcept ──────────────────────────────────────────────────────


class TestEmbeddingDimConcept:
    def test_mask_zeros_other_dims(self, relevance_3d):
        c = EmbeddingDimConcept(num_heads=NUM_HEADS)
        m = c.mask(batch_id=0, concept_ids=[7])
        masked = m(relevance_3d.clone())
        assert torch.equal(masked[0, :, 7], relevance_3d[0, :, 7])
        for d in range(EMBED_DIM):
            if d == 7:
                continue
            assert torch.all(masked[0, :, d] == 0)

    def test_attribute_shape(self, relevance_3d):
        c = EmbeddingDimConcept(num_heads=NUM_HEADS)
        rel = c.attribute(relevance_3d, abs_norm=False)
        assert rel.shape == (B, EMBED_DIM)
        # Sum over tokens.
        manual = relevance_3d.sum(dim=1)
        assert torch.allclose(rel, manual)

    def test_head_of_decoder(self):
        c = EmbeddingDimConcept(num_heads=NUM_HEADS)
        # head_dim = EMBED_DIM / NUM_HEADS = 6.
        assert c.head_of(0, EMBED_DIM) == 0
        assert c.head_of(5, EMBED_DIM) == 0
        assert c.head_of(6, EMBED_DIM) == 1
        assert c.head_of(EMBED_DIM - 1, EMBED_DIM) == NUM_HEADS - 1

    def test_invalid_dim_raises(self, relevance_3d):
        c = EmbeddingDimConcept(num_heads=NUM_HEADS)
        m = c.mask(batch_id=0, concept_ids=[EMBED_DIM])
        with pytest.raises(IndexError):
            m(relevance_3d.clone())


# ── TokenConcept ─────────────────────────────────────────────────────────────


class TestTokenConcept:
    def test_no_args_construction(self):
        c = TokenConcept()
        assert c.token_filter == slice(None)

    def test_mask_zeros_other_positions(self, relevance_3d):
        c = TokenConcept()
        m = c.mask(batch_id=0, concept_ids=[3])
        masked = m(relevance_3d.clone())
        assert torch.equal(masked[0, 3, :], relevance_3d[0, 3, :])
        for p in range(N):
            if p == 3:
                continue
            assert torch.all(masked[0, p, :] == 0)

    def test_attribute_shape_and_sum(self, relevance_3d):
        c = TokenConcept()
        rel = c.attribute(relevance_3d, abs_norm=False)
        assert rel.shape == (B, N)
        # Sum over embed_dim per token.
        manual = relevance_3d.sum(dim=-1)
        assert torch.allclose(rel, manual)

    def test_token_filter_prefix_only(self, relevance_3d):
        c = TokenConcept(token_filter=slice(0, NUM_PREFIX))
        rel = c.attribute(relevance_3d, abs_norm=False)
        # Output is only over the filter universe.
        assert rel.shape == (B, NUM_PREFIX)
        manual = relevance_3d[:, :NUM_PREFIX, :].sum(dim=-1)
        assert torch.allclose(rel, manual)

    def test_concept_id_indexes_into_filtered_universe(self, relevance_3d):
        # token_filter=slice(NUM_PREFIX, None) → concept id 0 maps to absolute pos NUM_PREFIX.
        c = TokenConcept(token_filter=slice(NUM_PREFIX, None))
        m = c.mask(batch_id=0, concept_ids=[0])
        masked = m(relevance_3d.clone())
        # Absolute position NUM_PREFIX kept; everything else (including
        # all prefix tokens) zeroed.
        assert torch.equal(masked[0, NUM_PREFIX, :], relevance_3d[0, NUM_PREFIX, :])
        assert torch.all(masked[0, :NUM_PREFIX, :] == 0)
        assert torch.all(masked[0, NUM_PREFIX + 1:, :] == 0)

    def test_invalid_position_in_filtered_universe_raises(self, relevance_3d):
        c = TokenConcept(token_filter=slice(NUM_PREFIX, None))
        # Filtered universe has N - NUM_PREFIX positions; id == that = out of range.
        m = c.mask(batch_id=0, concept_ids=[N - NUM_PREFIX])
        with pytest.raises(IndexError):
            m(relevance_3d.clone())
