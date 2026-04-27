"""Unit tests for crp.attention_concepts.

These exercise mask shapes, aggregation axes, and conservation between the
four concept granularities. They use torch tensors directly (no model) so
they run anywhere torch is installed.

Run with::

    uv run pytest tests/test_attention_concepts.py -v
"""

import pytest
import torch

from crp.attention_concepts import (
    HeadConcept,
    KQVConcept,
    KQVHeadConcept,
    HeadDimConcept,
    PART_OFFSETS,
    PARTS,
)


# ── shared fixtures ───────────────────────────────────────────────────────────

LAYER = "blocks.0.attn.qkv_tap"
B, N, NUM_HEADS, HEAD_DIM = 2, 5, 4, 6
D = NUM_HEADS * HEAD_DIM


@pytest.fixture
def relevance():
    """Reproducible relevance tensor of shape (B, N, 3*D)."""
    torch.manual_seed(0)
    return torch.randn(B, N, 3 * D)


def _register(concept):
    concept.register_layer(LAYER, NUM_HEADS, HEAD_DIM)
    return concept


def _apply_mask(concept, batch_id, concept_ids, grad):
    """Run the mask function to completion and return the masked grad."""
    mask_fct = concept.mask(batch_id, concept_ids, layer_name=LAYER)
    return mask_fct(grad.clone())


# ── HeadConcept ───────────────────────────────────────────────────────────────


class TestHeadConcept:
    def test_mask_covers_three_stripes(self, relevance):
        c = _register(HeadConcept())
        head_id = 2
        masked = _apply_mask(c, batch_id=0, concept_ids=[head_id], grad=relevance)
        m = masked[0]  # (N, 3*D)

        kept_indices = []
        for part_idx in range(3):
            offset = part_idx * D + head_id * HEAD_DIM
            kept_indices.extend(range(offset, offset + HEAD_DIM))
        kept = torch.tensor(kept_indices)

        assert (m[:, kept] != 0).all() or (m[:, kept] == relevance[0, :, kept]).all()
        rest = torch.tensor(
            [i for i in range(3 * D) if i not in set(kept_indices)]
        )
        assert (m[:, rest] == 0).all()

    def test_mask_other_batch_untouched(self, relevance):
        c = _register(HeadConcept())
        masked = _apply_mask(c, batch_id=0, concept_ids=[1], grad=relevance)
        assert torch.equal(masked[1], relevance[1])

    def test_mask_supports_int_and_tuple_ids(self, relevance):
        c = _register(HeadConcept())
        m_int = _apply_mask(c, 0, [0], relevance)
        m_tuple = _apply_mask(c, 0, [(0,)], relevance)
        assert torch.equal(m_int, m_tuple)

    def test_attribute_shape(self, relevance):
        c = _register(HeadConcept())
        out = c.attribute(relevance, layer_name=LAYER, abs_norm=False)
        assert out.shape == (B, NUM_HEADS)

    def test_attribute_abs_norm_sums_to_one(self, relevance):
        c = _register(HeadConcept())
        out = c.attribute(relevance, layer_name=LAYER, abs_norm=True)
        per_batch_sum = out.abs().sum(dim=-1)
        assert torch.allclose(per_batch_sum, torch.ones_like(per_batch_sum), atol=1e-5)

    def test_aggregation_matches_manual(self, relevance):
        c = _register(HeadConcept())
        out = c.attribute(relevance, layer_name=LAYER, abs_norm=False)
        # manual: for each head h, sum over (N, q+k+v slices)
        manual = torch.zeros(B, NUM_HEADS)
        for h in range(NUM_HEADS):
            for part_idx in range(3):
                s = part_idx * D + h * HEAD_DIM
                e = s + HEAD_DIM
                manual[:, h] += relevance[:, :, s:e].sum(dim=(1, 2))
        assert torch.allclose(out, manual, atol=1e-5)

    def test_out_of_range_head_raises(self):
        c = _register(HeadConcept())
        with pytest.raises(IndexError):
            c.mask(0, [NUM_HEADS], layer_name=LAYER)


# ── KQVConcept ────────────────────────────────────────────────────────────────


class TestKQVConcept:
    def test_mask_covers_whole_part(self, relevance):
        c = _register(KQVConcept())
        for part_str in PARTS:
            masked = _apply_mask(c, 0, [part_str], relevance)
            offset = PART_OFFSETS[part_str] * D
            kept = masked[0, :, offset : offset + D]
            other = torch.cat(
                [masked[0, :, : offset], masked[0, :, offset + D :]], dim=-1
            )
            assert torch.equal(kept, relevance[0, :, offset : offset + D])
            assert (other == 0).all()

    def test_part_int_alias(self, relevance):
        c = _register(KQVConcept())
        m_str = _apply_mask(c, 0, ["q"], relevance)
        m_int = _apply_mask(c, 0, [0], relevance)
        assert torch.equal(m_str, m_int)

    def test_attribute_shape(self, relevance):
        c = _register(KQVConcept())
        out = c.attribute(relevance, layer_name=LAYER, abs_norm=False)
        assert out.shape == (B, 3)

    def test_aggregation_matches_manual(self, relevance):
        c = _register(KQVConcept())
        out = c.attribute(relevance, layer_name=LAYER, abs_norm=False)
        manual = torch.stack(
            [relevance[:, :, p * D : (p + 1) * D].sum(dim=(1, 2)) for p in range(3)],
            dim=-1,
        )
        assert torch.allclose(out, manual, atol=1e-5)

    def test_invalid_part_raises(self):
        c = _register(KQVConcept())
        with pytest.raises(ValueError):
            c.mask(0, ["x"], layer_name=LAYER)


# ── KQVHeadConcept ────────────────────────────────────────────────────────────


class TestKQVHeadConcept:
    def test_mask_covers_part_head_stripe(self, relevance):
        c = _register(KQVHeadConcept())
        masked = _apply_mask(c, 0, [("k", 1)], relevance)
        offset = PART_OFFSETS["k"] * D + 1 * HEAD_DIM
        kept_slice = slice(offset, offset + HEAD_DIM)
        assert torch.equal(masked[0, :, kept_slice], relevance[0, :, kept_slice])

        # everything else zero
        m = masked[0]
        m[:, kept_slice] = 0
        assert (m == 0).all()

    def test_attribute_shape(self, relevance):
        c = _register(KQVHeadConcept())
        out = c.attribute(relevance, layer_name=LAYER, abs_norm=False)
        assert out.shape == (B, 3, NUM_HEADS)

    def test_part_axis_ordering(self, relevance):
        """Axis 1 should be ordered (Q, K, V)."""
        c = _register(KQVHeadConcept())
        out = c.attribute(relevance, layer_name=LAYER, abs_norm=False)
        for p_idx, p_str in enumerate(PARTS):
            for h in range(NUM_HEADS):
                s = PART_OFFSETS[p_str] * D + h * HEAD_DIM
                manual = relevance[:, :, s : s + HEAD_DIM].sum(dim=(1, 2))
                assert torch.allclose(out[:, p_idx, h], manual, atol=1e-5)

    def test_invalid_arity_raises(self):
        c = _register(KQVHeadConcept())
        with pytest.raises(ValueError):
            c.mask(0, ["q"], layer_name=LAYER)
        with pytest.raises(ValueError):
            c.mask(0, [(0,)], layer_name=LAYER)


# ── HeadDimConcept ────────────────────────────────────────────────────────────


class TestHeadDimConcept:
    def test_mask_covers_single_column(self, relevance):
        c = _register(HeadDimConcept())
        masked = _apply_mask(c, 0, [("v", 3, 5)], relevance)
        col = PART_OFFSETS["v"] * D + 3 * HEAD_DIM + 5
        assert torch.equal(masked[0, :, col], relevance[0, :, col])
        m = masked[0].clone()
        m[:, col] = 0
        assert (m == 0).all()

    def test_attribute_shape(self, relevance):
        c = _register(HeadDimConcept())
        out = c.attribute(relevance, layer_name=LAYER, abs_norm=False)
        assert out.shape == (B, 3, NUM_HEADS, HEAD_DIM)

    def test_aggregation_matches_manual(self, relevance):
        c = _register(HeadDimConcept())
        out = c.attribute(relevance, layer_name=LAYER, abs_norm=False)
        rel_5d = relevance.view(B, N, 3, NUM_HEADS, HEAD_DIM)
        manual = rel_5d.sum(dim=1)
        assert torch.allclose(out, manual, atol=1e-5)

    def test_dim_out_of_range_raises(self):
        c = _register(HeadDimConcept())
        with pytest.raises(IndexError):
            c.mask(0, [("q", 0, HEAD_DIM)], layer_name=LAYER)


# ── conservation across granularities ─────────────────────────────────────────


class TestConservation:
    """Coarser concepts must equal sums of finer concepts (same masks composed).

    Since ``attribute`` is a deterministic axis-reduction with no abs_norm,
    these identities hold up to floating-point error.
    """

    def test_headdim_sum_equals_kqvhead(self, relevance):
        fine = _register(HeadDimConcept()).attribute(
            relevance, layer_name=LAYER, abs_norm=False
        )  # (B, 3, H, d_h)
        coarse = _register(KQVHeadConcept()).attribute(
            relevance, layer_name=LAYER, abs_norm=False
        )  # (B, 3, H)
        assert torch.allclose(fine.sum(dim=-1), coarse, atol=1e-5)

    def test_kqvhead_sum_equals_kqv(self, relevance):
        fine = _register(KQVHeadConcept()).attribute(
            relevance, layer_name=LAYER, abs_norm=False
        )  # (B, 3, H)
        coarse = _register(KQVConcept()).attribute(
            relevance, layer_name=LAYER, abs_norm=False
        )  # (B, 3)
        assert torch.allclose(fine.sum(dim=-1), coarse, atol=1e-5)

    def test_kqvhead_sum_over_parts_equals_head(self, relevance):
        kqv_per_head = _register(KQVHeadConcept()).attribute(
            relevance, layer_name=LAYER, abs_norm=False
        )  # (B, 3, H)
        head = _register(HeadConcept()).attribute(
            relevance, layer_name=LAYER, abs_norm=False
        )  # (B, H)
        assert torch.allclose(kqv_per_head.sum(dim=1), head, atol=1e-5)


# ── registration ──────────────────────────────────────────────────────────────


class TestRegistration:
    def test_parent_fallback(self, relevance):
        c = HeadConcept()
        c.register_layer("blocks.7.attn", NUM_HEADS, HEAD_DIM)
        # qkv_tap child resolves via parent
        out = c.attribute(
            relevance, layer_name="blocks.7.attn.qkv_tap", abs_norm=False
        )
        assert out.shape == (B, NUM_HEADS)

    def test_unregistered_layer_raises(self, relevance):
        c = HeadConcept()
        with pytest.raises(ValueError):
            c.attribute(relevance, layer_name="blocks.0.attn", abs_norm=False)

    def test_register_from_model_finds_qkv_attentions(self):
        """Synthesize a tiny model with two attention-like submodules."""
        import torch.nn as nn

        class FakeAttn(nn.Module):
            def __init__(self, num_heads, head_dim):
                super().__init__()
                self.qkv = nn.Linear(num_heads * head_dim, 3 * num_heads * head_dim)
                self.num_heads = num_heads
                self.head_dim = head_dim

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = FakeAttn(NUM_HEADS, HEAD_DIM)

        model = nn.Sequential(FakeBlock(), FakeBlock())
        c = HeadConcept()
        c.register_from_model(model)
        # both blocks registered, both under bare name and qkv_tap suffix
        assert c._resolve_dims("0.attn") == (NUM_HEADS, HEAD_DIM)
        assert c._resolve_dims("1.attn.qkv_tap") == (NUM_HEADS, HEAD_DIM)

    def test_invalid_dims_raise(self):
        c = HeadConcept()
        with pytest.raises(ValueError):
            c.register_layer(LAYER, 0, HEAD_DIM)
        with pytest.raises(ValueError):
            c.register_layer(LAYER, NUM_HEADS, -1)


# ── shape validation ──────────────────────────────────────────────────────────


def test_mask_validates_last_dim(relevance):
    """Mask raises when applied to a tensor with the wrong trailing dim."""
    c = _register(HeadConcept())
    bad_grad = torch.randn(B, N, 3 * D - 1)
    mask_fct = c.mask(0, [0], layer_name=LAYER)
    with pytest.raises(ValueError):
        mask_fct(bad_grad)
