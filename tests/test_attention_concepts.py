"""Unit tests for crp.attention_concepts.

Exercise mask shapes, aggregation axes, and conservation between the four
concept granularities. Pure tensor-level — no model load — so they run
anywhere torch is installed.

The four concepts come from crossing two orthogonal flags:

* ``KQV_SPLIT`` — qkv_tap (last dim ``3*D``) vs attn_out_tap (last dim ``D``).
* ``DIM_SPLIT`` — per-head ``head_dim`` axis kept (one concept per
  ``(head, dim)``) vs summed out (one concept per head).

Run with::

    uv run pytest tests/test_attention_concepts.py -v
"""

import pytest
import torch

from crp.attention_concepts import (
    HeadConcept,
    HeadDimConcept,
    KQVHeadConcept,
    KQVHeadDimConcept,
    PART_OFFSETS,
    PARTS,
)


# ── shared fixtures ───────────────────────────────────────────────────────────

QKV_LAYER = "blocks.0.attn.qkv_tap"
OUT_LAYER = "blocks.0.attn.attn_out_tap"
B, N, NUM_HEADS, HEAD_DIM = 2, 5, 4, 6
D = NUM_HEADS * HEAD_DIM


@pytest.fixture
def qkv_relevance():
    """Reproducible relevance tensor of shape ``(B, N, 3*D)`` for qkv_tap."""
    torch.manual_seed(0)
    return torch.randn(B, N, 3 * D)


@pytest.fixture
def out_relevance():
    """Reproducible relevance tensor of shape ``(B, N, D)`` for attn_out_tap."""
    torch.manual_seed(0)
    return torch.randn(B, N, D)


def _register(concept, layer):
    concept.register_layer(layer, NUM_HEADS, HEAD_DIM)
    return concept


def _apply_mask(concept, batch_id, concept_ids, grad, layer):
    mask_fct = concept.mask(batch_id, concept_ids, layer_name=layer)
    return mask_fct(grad.clone())


# ── HeadConcept (output side, per head) ──────────────────────────────────────


class TestHeadConcept:
    """``attn_out_tap``, mask = head stripe of ``D``-dim output tokens."""

    def test_mask_covers_one_head_stripe(self, out_relevance):
        c = _register(HeadConcept(), OUT_LAYER)
        head_id = 2
        masked = _apply_mask(c, 0, [head_id], out_relevance, OUT_LAYER)
        m = masked[0]  # (N, D)
        s, e = head_id * HEAD_DIM, (head_id + 1) * HEAD_DIM
        assert torch.equal(m[:, s:e], out_relevance[0, :, s:e])
        kept = m.clone()
        kept[:, s:e] = 0
        assert (kept == 0).all()

    def test_mask_other_batch_untouched(self, out_relevance):
        c = _register(HeadConcept(), OUT_LAYER)
        masked = _apply_mask(c, 0, [1], out_relevance, OUT_LAYER)
        assert torch.equal(masked[1], out_relevance[1])

    def test_mask_supports_int_and_tuple_ids(self, out_relevance):
        c = _register(HeadConcept(), OUT_LAYER)
        m_int = _apply_mask(c, 0, [0], out_relevance, OUT_LAYER)
        m_tuple = _apply_mask(c, 0, [(0,)], out_relevance, OUT_LAYER)
        assert torch.equal(m_int, m_tuple)

    def test_attribute_shape(self, out_relevance):
        c = _register(HeadConcept(), OUT_LAYER)
        out = c.attribute(out_relevance, layer_name=OUT_LAYER, abs_norm=False)
        assert out.shape == (B, NUM_HEADS)

    def test_attribute_abs_norm_sums_to_one(self, out_relevance):
        c = _register(HeadConcept(), OUT_LAYER)
        out = c.attribute(out_relevance, layer_name=OUT_LAYER, abs_norm=True)
        assert torch.allclose(
            out.abs().sum(dim=-1), torch.ones(B), atol=1e-5
        )

    def test_aggregation_matches_manual(self, out_relevance):
        c = _register(HeadConcept(), OUT_LAYER)
        out = c.attribute(out_relevance, layer_name=OUT_LAYER, abs_norm=False)
        manual = torch.zeros(B, NUM_HEADS)
        for h in range(NUM_HEADS):
            s, e = h * HEAD_DIM, (h + 1) * HEAD_DIM
            manual[:, h] = out_relevance[:, :, s:e].sum(dim=(1, 2))
        assert torch.allclose(out, manual, atol=1e-5)

    def test_out_of_range_head_raises(self):
        c = _register(HeadConcept(), OUT_LAYER)
        with pytest.raises(IndexError):
            c.mask(0, [NUM_HEADS], layer_name=OUT_LAYER)

    def test_tap_name(self):
        assert HeadConcept().tap_name == "attn_out_tap"


# ── HeadDimConcept (output side, per (head, dim)) ────────────────────────────


class TestHeadDimConcept:
    """``attn_out_tap``, mask = single ``(head, dim)`` column."""

    def test_mask_covers_single_column(self, out_relevance):
        c = _register(HeadDimConcept(), OUT_LAYER)
        masked = _apply_mask(c, 0, [(2, 4)], out_relevance, OUT_LAYER)
        col = 2 * HEAD_DIM + 4
        assert torch.equal(masked[0, :, col], out_relevance[0, :, col])
        m = masked[0].clone()
        m[:, col] = 0
        assert (m == 0).all()

    def test_attribute_shape(self, out_relevance):
        c = _register(HeadDimConcept(), OUT_LAYER)
        out = c.attribute(out_relevance, layer_name=OUT_LAYER, abs_norm=False)
        assert out.shape == (B, NUM_HEADS, HEAD_DIM)

    def test_aggregation_matches_manual(self, out_relevance):
        c = _register(HeadDimConcept(), OUT_LAYER)
        out = c.attribute(out_relevance, layer_name=OUT_LAYER, abs_norm=False)
        manual = out_relevance.view(B, N, NUM_HEADS, HEAD_DIM).sum(dim=1)
        assert torch.allclose(out, manual, atol=1e-5)

    def test_dim_out_of_range_raises(self):
        c = _register(HeadDimConcept(), OUT_LAYER)
        with pytest.raises(IndexError):
            c.mask(0, [(0, HEAD_DIM)], layer_name=OUT_LAYER)

    def test_head_out_of_range_raises(self):
        c = _register(HeadDimConcept(), OUT_LAYER)
        with pytest.raises(IndexError):
            c.mask(0, [(NUM_HEADS, 0)], layer_name=OUT_LAYER)

    def test_tap_name(self):
        assert HeadDimConcept().tap_name == "attn_out_tap"


# ── KQVHeadConcept (qkv side, per (part, head)) ──────────────────────────────


class TestKQVHeadConcept:
    """``qkv_tap``, mask = single ``(part, head)`` stripe of the
    ``3D``-tensor."""

    def test_mask_covers_part_head_stripe(self, qkv_relevance):
        c = _register(KQVHeadConcept(), QKV_LAYER)
        masked = _apply_mask(c, 0, [("k", 1)], qkv_relevance, QKV_LAYER)
        offset = PART_OFFSETS["k"] * D + 1 * HEAD_DIM
        kept = slice(offset, offset + HEAD_DIM)
        assert torch.equal(masked[0, :, kept], qkv_relevance[0, :, kept])
        m = masked[0].clone()
        m[:, kept] = 0
        assert (m == 0).all()

    def test_attribute_shape(self, qkv_relevance):
        c = _register(KQVHeadConcept(), QKV_LAYER)
        out = c.attribute(qkv_relevance, layer_name=QKV_LAYER, abs_norm=False)
        assert out.shape == (B, 3, NUM_HEADS)

    def test_part_axis_ordering(self, qkv_relevance):
        """Axis 1 is ordered ``(Q, K, V)``."""
        c = _register(KQVHeadConcept(), QKV_LAYER)
        out = c.attribute(qkv_relevance, layer_name=QKV_LAYER, abs_norm=False)
        for p_idx, p_str in enumerate(PARTS):
            for h in range(NUM_HEADS):
                s = PART_OFFSETS[p_str] * D + h * HEAD_DIM
                manual = qkv_relevance[:, :, s : s + HEAD_DIM].sum(dim=(1, 2))
                assert torch.allclose(out[:, p_idx, h], manual, atol=1e-5)

    def test_part_int_alias(self, qkv_relevance):
        c = _register(KQVHeadConcept(), QKV_LAYER)
        m_str = _apply_mask(c, 0, [("q", 2)], qkv_relevance, QKV_LAYER)
        m_int = _apply_mask(c, 0, [(0, 2)], qkv_relevance, QKV_LAYER)
        assert torch.equal(m_str, m_int)

    def test_invalid_arity_raises(self):
        c = _register(KQVHeadConcept(), QKV_LAYER)
        with pytest.raises(ValueError):
            c.mask(0, ["q"], layer_name=QKV_LAYER)
        with pytest.raises(ValueError):
            c.mask(0, [("q", 0, 0)], layer_name=QKV_LAYER)

    def test_tap_name(self):
        assert KQVHeadConcept().tap_name == "qkv_tap"


# ── KQVHeadDimConcept (qkv side, per (part, head, dim)) ──────────────────────


class TestKQVHeadDimConcept:
    """``qkv_tap``, mask = single ``(part, head, dim)`` column."""

    def test_mask_covers_single_column(self, qkv_relevance):
        c = _register(KQVHeadDimConcept(), QKV_LAYER)
        masked = _apply_mask(c, 0, [("v", 3, 5)], qkv_relevance, QKV_LAYER)
        col = PART_OFFSETS["v"] * D + 3 * HEAD_DIM + 5
        assert torch.equal(masked[0, :, col], qkv_relevance[0, :, col])
        m = masked[0].clone()
        m[:, col] = 0
        assert (m == 0).all()

    def test_attribute_shape(self, qkv_relevance):
        c = _register(KQVHeadDimConcept(), QKV_LAYER)
        out = c.attribute(qkv_relevance, layer_name=QKV_LAYER, abs_norm=False)
        assert out.shape == (B, 3, NUM_HEADS, HEAD_DIM)

    def test_aggregation_matches_manual(self, qkv_relevance):
        c = _register(KQVHeadDimConcept(), QKV_LAYER)
        out = c.attribute(qkv_relevance, layer_name=QKV_LAYER, abs_norm=False)
        manual = qkv_relevance.view(B, N, 3, NUM_HEADS, HEAD_DIM).sum(dim=1)
        assert torch.allclose(out, manual, atol=1e-5)

    def test_dim_out_of_range_raises(self):
        c = _register(KQVHeadDimConcept(), QKV_LAYER)
        with pytest.raises(IndexError):
            c.mask(0, [("q", 0, HEAD_DIM)], layer_name=QKV_LAYER)

    def test_tap_name(self):
        assert KQVHeadDimConcept().tap_name == "qkv_tap"


# ── conservation: finer concept summed over its extra axis = coarser ─────────


class TestConservation:
    """``HeadDim`` summed over ``dim`` axis should equal ``Head`` (output
    side); ``KQVHeadDim`` summed over ``dim`` should equal ``KQVHead``.

    Cross-tap conservation (e.g. KQVHead → Head) does **not** hold — the
    two taps see different relevance because they sit on different points
    in the attention forward."""

    def test_head_dim_sum_equals_head(self, out_relevance):
        fine = _register(HeadDimConcept(), OUT_LAYER).attribute(
            out_relevance, layer_name=OUT_LAYER, abs_norm=False
        )  # (B, num_heads, head_dim)
        coarse = _register(HeadConcept(), OUT_LAYER).attribute(
            out_relevance, layer_name=OUT_LAYER, abs_norm=False
        )  # (B, num_heads)
        assert torch.allclose(fine.sum(dim=-1), coarse, atol=1e-5)

    def test_kqv_head_dim_sum_equals_kqv_head(self, qkv_relevance):
        fine = _register(KQVHeadDimConcept(), QKV_LAYER).attribute(
            qkv_relevance, layer_name=QKV_LAYER, abs_norm=False
        )  # (B, 3, num_heads, head_dim)
        coarse = _register(KQVHeadConcept(), QKV_LAYER).attribute(
            qkv_relevance, layer_name=QKV_LAYER, abs_norm=False
        )  # (B, 3, num_heads)
        assert torch.allclose(fine.sum(dim=-1), coarse, atol=1e-5)


# ── reference_sampling (FeatureVisualization shim) ────────────────────────────


class TestReferenceSampling:
    """Per-concept ranking of batch samples — required by
    :class:`~crp.maximization.Maximization` and
    :class:`~crp.visualization.FeatureVisualization`."""

    def test_head_concept_shapes(self, out_relevance):
        c = _register(HeadConcept(), OUT_LAYER)
        d, r, n = c.reference_sampling(
            out_relevance, layer_name=OUT_LAYER, abs_norm=False
        )
        assert d.shape == (B, NUM_HEADS)
        assert r.shape == (B, NUM_HEADS)
        assert n.shape == (B, NUM_HEADS)
        assert (n >= 0).all() and (n < N).all()

    def test_head_dim_shapes(self, out_relevance):
        c = _register(HeadDimConcept(), OUT_LAYER)
        d, r, _ = c.reference_sampling(
            out_relevance, layer_name=OUT_LAYER, abs_norm=False
        )
        assert d.shape == (B, NUM_HEADS * HEAD_DIM)
        assert r.shape == (B, NUM_HEADS * HEAD_DIM)

    def test_kqv_head_shapes(self, qkv_relevance):
        c = _register(KQVHeadConcept(), QKV_LAYER)
        d, r, _ = c.reference_sampling(
            qkv_relevance, layer_name=QKV_LAYER, abs_norm=False
        )
        assert d.shape == (B, 3 * NUM_HEADS)
        assert r.shape == (B, 3 * NUM_HEADS)

    def test_kqv_head_dim_shapes(self, qkv_relevance):
        c = _register(KQVHeadDimConcept(), QKV_LAYER)
        d, r, _ = c.reference_sampling(
            qkv_relevance, layer_name=QKV_LAYER, abs_norm=False
        )
        assert d.shape == (B, 3 * NUM_HEADS * HEAD_DIM)
        assert r.shape == (B, 3 * NUM_HEADS * HEAD_DIM)

    def test_descending_order(self, out_relevance):
        c = _register(HeadConcept(), OUT_LAYER)
        _, r, _ = c.reference_sampling(
            out_relevance, layer_name=OUT_LAYER, abs_norm=False
        )
        diffs = r[:-1] - r[1:]
        assert (diffs >= 0).all()

    def test_aggregation_matches_attribute(self, qkv_relevance):
        """Sum-over-batch of ``rel_c_sorted`` equals sum-over-batch of
        ``attribute`` (modulo within-column reorder by argsort)."""
        c = _register(KQVHeadConcept(), QKV_LAYER)
        _, r_sorted, _ = c.reference_sampling(
            qkv_relevance, layer_name=QKV_LAYER, abs_norm=False
        )
        attr = c.attribute(qkv_relevance, layer_name=QKV_LAYER, abs_norm=False)
        attr_flat = attr.reshape(B, 3 * NUM_HEADS)
        assert torch.allclose(
            r_sorted.sum(dim=0), attr_flat.sum(dim=0), atol=1e-5
        )


# ── flat int IDs (FV passes these from argsort) ──────────────────────────────


class TestFlatIntegerIds:
    def test_head_flat_int_matches_tuple(self, out_relevance):
        c = _register(HeadConcept(), OUT_LAYER)
        # HeadConcept axes = (num_heads,) → flat int = head id.
        assert torch.equal(
            _apply_mask(c, 0, [2], out_relevance, OUT_LAYER),
            _apply_mask(c, 0, [(2,)], out_relevance, OUT_LAYER),
        )

    def test_head_dim_flat_int_matches_tuple(self, out_relevance):
        c = _register(HeadDimConcept(), OUT_LAYER)
        # axes = (num_heads, head_dim) = (4, 6). flat 14 → head=14//6=2, dim=2
        m_int = _apply_mask(c, 0, [14], out_relevance, OUT_LAYER)
        m_tuple = _apply_mask(c, 0, [(2, 2)], out_relevance, OUT_LAYER)
        assert torch.equal(m_int, m_tuple)

    def test_kqv_head_flat_int_matches_tuple(self, qkv_relevance):
        c = _register(KQVHeadConcept(), QKV_LAYER)
        # axes = (3, num_heads). flat 5 → part=5//4=1, head=5%4=1 → (k, 1).
        m_int = _apply_mask(c, 0, [5], qkv_relevance, QKV_LAYER)
        m_tuple = _apply_mask(c, 0, [("k", 1)], qkv_relevance, QKV_LAYER)
        assert torch.equal(m_int, m_tuple)

    def test_kqv_head_dim_flat_int_matches_tuple(self, qkv_relevance):
        c = _register(KQVHeadDimConcept(), QKV_LAYER)
        # axes = (3, num_heads, head_dim) = (3, 4, 6). flat 30 → part=1, head=1, dim=0
        m_int = _apply_mask(c, 0, [30], qkv_relevance, QKV_LAYER)
        m_tuple = _apply_mask(c, 0, [("k", 1, 0)], qkv_relevance, QKV_LAYER)
        assert torch.equal(m_int, m_tuple)

    def test_kqv_head_flat_out_of_range_raises(self):
        c = _register(KQVHeadConcept(), QKV_LAYER)
        with pytest.raises(IndexError):
            c.mask(0, [3 * NUM_HEADS], layer_name=QKV_LAYER)

    def test_head_dim_flat_out_of_range_raises(self):
        c = _register(HeadDimConcept(), OUT_LAYER)
        with pytest.raises(IndexError):
            c.mask(0, [NUM_HEADS * HEAD_DIM], layer_name=OUT_LAYER)


# ── registration ──────────────────────────────────────────────────────────────


class TestRegistration:
    def test_parent_fallback_attn_out_tap(self, out_relevance):
        c = HeadConcept()
        c.register_layer("blocks.7.attn", NUM_HEADS, HEAD_DIM)
        out = c.attribute(
            out_relevance,
            layer_name="blocks.7.attn.attn_out_tap",
            abs_norm=False,
        )
        assert out.shape == (B, NUM_HEADS)

    def test_parent_fallback_qkv_tap(self, qkv_relevance):
        c = KQVHeadConcept()
        c.register_layer("blocks.7.attn", NUM_HEADS, HEAD_DIM)
        out = c.attribute(
            qkv_relevance,
            layer_name="blocks.7.attn.qkv_tap",
            abs_norm=False,
        )
        assert out.shape == (B, 3, NUM_HEADS)

    def test_unregistered_layer_raises(self, out_relevance):
        c = HeadConcept()
        with pytest.raises(ValueError):
            c.attribute(out_relevance, layer_name="blocks.0.attn", abs_norm=False)

    def test_register_from_model_registers_both_taps(self):
        """A synthetic model with one attention-like submodule should
        register the dims under bare name AND both tap-suffixed names."""
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
        # Both blocks registered under bare AND both tap names.
        assert c._resolve_dims("0.attn") == (NUM_HEADS, HEAD_DIM)
        assert c._resolve_dims("0.attn.qkv_tap") == (NUM_HEADS, HEAD_DIM)
        assert c._resolve_dims("0.attn.attn_out_tap") == (NUM_HEADS, HEAD_DIM)
        assert c._resolve_dims("1.attn.attn_out_tap") == (NUM_HEADS, HEAD_DIM)

    def test_register_via_constructor(self):
        """Passing a model to the constructor calls register_from_model."""
        import torch.nn as nn

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(D, 3 * D)
                self.num_heads = NUM_HEADS
                self.head_dim = HEAD_DIM

        model = nn.Sequential(FakeAttn())
        c = HeadConcept(model=model)
        # named_modules yields name="0" for the FakeAttn child.
        assert c._resolve_dims("0") == (NUM_HEADS, HEAD_DIM)
        assert c._resolve_dims("0.attn_out_tap") == (NUM_HEADS, HEAD_DIM)
        assert c._resolve_dims("0.qkv_tap") == (NUM_HEADS, HEAD_DIM)

    def test_invalid_dims_raise(self):
        c = HeadConcept()
        with pytest.raises(ValueError):
            c.register_layer(OUT_LAYER, 0, HEAD_DIM)
        with pytest.raises(ValueError):
            c.register_layer(OUT_LAYER, NUM_HEADS, -1)


# ── shape validation ──────────────────────────────────────────────────────────


def test_head_concept_mask_validates_last_dim(out_relevance):
    """``HeadConcept`` (attn_out_tap) raises on ``3*D`` last dim — wrong tap."""
    c = _register(HeadConcept(), OUT_LAYER)
    bad_grad = torch.randn(B, N, 3 * D)
    mask_fct = c.mask(0, [0], layer_name=OUT_LAYER)
    with pytest.raises(ValueError):
        mask_fct(bad_grad)


def test_kqv_head_concept_mask_validates_last_dim(qkv_relevance):
    """``KQVHeadConcept`` (qkv_tap) raises on ``D`` last dim — wrong tap."""
    c = _register(KQVHeadConcept(), QKV_LAYER)
    bad_grad = torch.randn(B, N, D)
    mask_fct = c.mask(0, [(0, 0)], layer_name=QKV_LAYER)
    with pytest.raises(ValueError):
        mask_fct(bad_grad)
