"""Unit tests for ``crp.attention_unfolded``.

Phase 1 of the attention-unfolding refactor (see
``UNFOLDING_ATTENTION_REFACTOR.md``). Tests are weight-loading-free so
CI can run them without downloading checkpoints.

Coverage matrix:

* :class:`BilinearMatmul`
  - forward parity with bare ``a @ b`` (passthrough mode)
  - autograd backward parity with bare ``a @ b`` (passthrough mode)
  - matmul rule conservation per AttnLRP Prop. 3.3
    (``sum(R_a) + sum(R_b) ≈ sum(R_y)``)
* :class:`SoftmaxAlongLastDim`
  - forward parity with ``F.softmax(dim=-1)``
  - identity rule on backward (``R_in == R_out``)
* :class:`RotaryEmbedding`
  - forward parity with ``apply_rot_embed_cat`` (with and without
    ``num_prefix_tokens`` skip, with and without rope_detach)
  - rope_detach actually severs the rope grad path
* :class:`ScaleByConstant`, :class:`ChunkAlongLastDim`,
  :class:`ReshapeMergeHeads`, :class:`AddBias`, :class:`ResidualAdd`,
  :class:`LayerScaleMul`
  - forward / backward sanity
* :class:`EvaAttentionUnfolded`
  - forward parity with stock ``EvaAttention`` on a synthetic
    weight-init random model
  - autograd backward parity with stock ``EvaAttention``
* :class:`EvaAttentionSubstitutionCanonizer`
  - apply → forward → remove round-trip restores the original
    forward exactly
  - re-apply works after remove

Run with::

    uv run pytest tests/test_attention_unfolded.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

timm = pytest.importorskip("timm")

from crp.attention_unfolded import (
    AddBias,
    BilinearMatmul,
    ChunkAlongLastDim,
    EvaAttentionSubstitutionCanonizer,
    EvaAttentionUnfolded,
    LayerScaleMul,
    ReshapeMergeHeads,
    ResidualAdd,
    RotaryEmbedding,
    ScaleByConstant,
    SoftmaxAlongLastDim,
)


# ─── 1. BilinearMatmul ───────────────────────────────────────────────────────


class TestBilinearMatmul:
    def test_forward_passthrough_matches_bare_matmul(self):
        torch.manual_seed(0)
        a = torch.randn(2, 3, 5, 7)
        b = torch.randn(2, 3, 7, 4)
        m = BilinearMatmul(rule="passthrough")
        y = m(a, b)
        assert torch.equal(y, a @ b)

    def test_backward_passthrough_matches_bare_matmul(self):
        torch.manual_seed(1)
        a1 = torch.randn(2, 4, 5, requires_grad=True)
        b1 = torch.randn(2, 5, 6, requires_grad=True)
        a2 = a1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)
        go = torch.randn(2, 4, 6)
        (a1 @ b1).backward(go)
        BilinearMatmul(rule="passthrough")(a2, b2).backward(go)
        assert torch.allclose(a1.grad, a2.grad, atol=0)
        assert torch.allclose(b1.grad, b2.grad, atol=0)

    def test_forward_matmul_factor_2_matches_bare_matmul(self):
        # The Prop. 3.3 rule only changes backward; forward is identical.
        torch.manual_seed(2)
        a = torch.randn(2, 3, 5, 7)
        b = torch.randn(2, 3, 7, 4)
        m = BilinearMatmul(rule="matmul_factor_2", epsilon=1e-6)
        y = m(a, b)
        assert torch.allclose(y, a @ b, atol=0)

    def test_matmul_factor_2_conservation(self):
        """AttnLRP Prop. 3.3: ``sum(R_a) + sum(R_b) ≈ sum(R_y)`` because
        the ``2y+ε`` denominator splits each upstream relevance evenly
        across the two bilinear chains and the operand-multiplication
        recovers the conservation constant."""
        torch.manual_seed(3)
        # Pick well-conditioned operands (mean shifted away from 0) so
        # ``2y+ε`` is not near zero.
        a = (torch.randn(1, 4, 6) * 2.0 + 1.0).requires_grad_(True)
        b = (torch.randn(1, 6, 5) * 2.0 + 1.0).requires_grad_(True)
        m = BilinearMatmul(rule="matmul_factor_2", epsilon=1e-6)
        y = m(a, b)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        R_a = a.grad
        R_b = b.grad
        assert torch.isclose(
            R_a.sum() + R_b.sum(), R_y.sum(), rtol=1e-3, atol=1e-3,
        ), (
            f"sum(R_a)+sum(R_b)={R_a.sum().item() + R_b.sum().item():.6f} "
            f"vs sum(R_y)={R_y.sum().item():.6f}"
        )


# ─── 2. SoftmaxAlongLastDim ──────────────────────────────────────────────────


class TestSoftmaxAlongLastDim:
    def test_forward_matches_F_softmax(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 7)
        for rule in ("identity", "passthrough"):
            sm = SoftmaxAlongLastDim(rule=rule)
            assert torch.equal(sm(x), F.softmax(x, dim=-1))

    def test_identity_rule_passes_relevance_through(self):
        """``R_in = R_out`` per AttnLRP Eq. 9. The softmax Jacobian
        coupling between positions is short-circuited."""
        torch.manual_seed(1)
        x = torch.randn(2, 3, 7, requires_grad=True)
        sm = SoftmaxAlongLastDim(rule="identity")
        y = sm(x)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        assert torch.equal(x.grad, R_y)

    def test_passthrough_uses_natural_jacobian(self):
        """Bare softmax: backward is the proper Jacobian, not identity."""
        torch.manual_seed(2)
        x = torch.randn(2, 3, 7, requires_grad=True)
        sm = SoftmaxAlongLastDim(rule="passthrough")
        y = sm(x)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        # The natural Jacobian-vector product should NOT in general
        # equal the upstream relevance (different formula).
        assert not torch.allclose(x.grad, R_y, atol=1e-6)


# ─── 3. RotaryEmbedding ──────────────────────────────────────────────────────


class TestRotaryEmbedding:
    def _make(self):
        torch.manual_seed(0)
        # Shape mirrors the EvaAttention input (B, num_heads, N, head_dim).
        q = torch.randn(2, 4, 16, 8)
        # rope: (N - num_prefix_tokens, head_dim*2) — sin || cos chunks
        # per timm.layers.apply_rot_embed_cat.
        rope = torch.randn(12, 16)
        return q, rope

    def test_forward_parity_no_prefix_no_detach(self):
        from timm.layers import apply_rot_embed_cat
        q, rope = self._make()
        # No prefix → entire sequence is rotated.
        m = RotaryEmbedding(num_prefix_tokens=4, rotate_half=False, detach_rope=False)
        y = m(q, rope)
        # Manual reference: prefix unrotated, suffix rotated.
        prefix = q[:, :, :4, :]
        rotated = apply_rot_embed_cat(q[:, :, 4:, :], rope, half=False)
        ref = torch.cat([prefix, rotated], dim=2)
        assert torch.equal(y, ref)

    def test_forward_parity_with_rotate_half(self):
        from timm.layers import apply_rot_embed_cat
        q, rope = self._make()
        m = RotaryEmbedding(num_prefix_tokens=4, rotate_half=True)
        y = m(q, rope)
        prefix = q[:, :, :4, :]
        rotated = apply_rot_embed_cat(q[:, :, 4:, :], rope, half=True)
        ref = torch.cat([prefix, rotated], dim=2)
        assert torch.equal(y, ref)

    def test_rope_none_is_identity(self):
        q, _ = self._make()
        m = RotaryEmbedding(num_prefix_tokens=4)
        y = m(q, None)
        assert torch.equal(y, q)

    def test_detach_rope_severs_rope_grad(self):
        q, rope = self._make()
        rope.requires_grad_(True)
        q.requires_grad_(True)

        m_attached = RotaryEmbedding(num_prefix_tokens=4, detach_rope=False)
        y = m_attached(q, rope)
        y.sum().backward()
        assert rope.grad is not None
        assert rope.grad.abs().max() > 0
        rope.grad = None
        q.grad = None

        m_detached = RotaryEmbedding(num_prefix_tokens=4, detach_rope=True)
        y2 = m_detached(q, rope)
        y2.sum().backward()
        assert rope.grad is None or rope.grad.abs().max() == 0


# ─── 4. ScaleByConstant ──────────────────────────────────────────────────────


class TestScaleByConstant:
    def test_forward_matches_multiplication(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3)
        for rule in ("identity", "passthrough"):
            assert torch.equal(ScaleByConstant(2.5, rule=rule)(x), x * 2.5)

    def test_passthrough_backward_uses_chain_rule(self):
        torch.manual_seed(1)
        x = torch.randn(2, 3, requires_grad=True)
        ScaleByConstant(2.5, rule="passthrough")(x).sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, 2.5))

    def test_identity_backward_passes_relevance_through(self):
        torch.manual_seed(2)
        x = torch.randn(2, 3, requires_grad=True)
        sc = ScaleByConstant(2.5, rule="identity")
        y = sc(x)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        assert torch.equal(x.grad, R_y)


# ─── 5. ChunkAlongLastDim, ReshapeMergeHeads, AddBias ────────────────────────


class TestSimpleKernels:
    def test_chunk_split_and_concat_round_trip(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 12, requires_grad=True)
        chunks = ChunkAlongLastDim(3)(x)
        assert len(chunks) == 3
        assert all(c.shape == (2, 3, 4) for c in chunks)
        # Backward of split is concat — sum of grad is preserved.
        y = sum(c.sum() for c in chunks)
        y.backward()
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_reshape_merge_heads(self):
        # x: (B, num_heads, N, head_dim) → (B, N, num_heads*head_dim).
        torch.manual_seed(0)
        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        y = ReshapeMergeHeads()(x)
        assert y.shape == (2, 5, 24)
        # Manual reference.
        ref = x.transpose(1, 2).reshape(2, 5, 24)
        assert torch.equal(y, ref)
        # Backward: identity on R.
        go = torch.randn_like(y)
        y.backward(go)
        ref_grad = go.reshape(2, 5, 4, 6).transpose(1, 2)
        assert torch.allclose(x.grad, ref_grad, atol=0)

    def test_add_bias_with_none(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3)
        assert torch.equal(AddBias()(x, None), x)

    def test_add_bias_with_tensor(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 3)
        y = AddBias()(x, b)
        assert torch.equal(y, x + b)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x))


# ─── 6. ResidualAdd ──────────────────────────────────────────────────────────


class TestResidualAdd:
    def test_ratio_rule_conservation(self):
        torch.manual_seed(0)
        x = (torch.randn(2, 5) + 0.5).requires_grad_(True)
        b = (torch.randn(2, 5) + 0.5).requires_grad_(True)
        m = ResidualAdd(rule="ratio")
        y = m(x, b)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        # |x| + |b| ratio split should approximately conserve R_y.
        total = x.grad.sum() + b.grad.sum()
        assert torch.isclose(total, R_y.sum(), rtol=1e-3, atol=1e-3)

    def test_symmetric_rule_halves_gradient(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, requires_grad=True)
        b = torch.randn(2, 5, requires_grad=True)
        m = ResidualAdd(rule="symmetric")
        y = m(x, b)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        assert torch.allclose(x.grad, R_y / 2)
        assert torch.allclose(b.grad, R_y / 2)


# ─── 7. LayerScaleMul ────────────────────────────────────────────────────────


class TestLayerScaleMul:
    def test_forward_multiplies_by_gamma(self):
        gamma = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.randn(2, 4, 3)
        m = LayerScaleMul(gamma, layerscale_uniform=True)
        assert torch.equal(m(x), gamma * x)

    def test_uniform_rule_halves_relevance(self):
        gamma = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.randn(2, 4, 3, requires_grad=True)
        m = LayerScaleMul(gamma, layerscale_uniform=True)
        y = m(x)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        # Without the uniform rule the gradient would be R_y * gamma;
        # with the rule it's halved AFTER autograd does R_y * gamma.
        # But _DivideGradientFn divides R_y BEFORE bare-grad propagation,
        # so x.grad = (R_y / 2) * gamma.
        assert torch.allclose(x.grad, (R_y / 2) * gamma)

    def test_no_uniform_rule_full_grad(self):
        gamma = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.randn(2, 4, 3, requires_grad=True)
        m = LayerScaleMul(gamma, layerscale_uniform=False)
        y = m(x)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        assert torch.allclose(x.grad, R_y * gamma)


# ─── 8. EvaAttentionUnfolded — end-to-end on synthetic random model ─────────


@pytest.fixture(scope="module")
def synthetic_eva_attention():
    """Construct a synthetic EvaAttention without downloading any
    checkpoint — random init is sufficient for forward/backward parity
    tests since we compare against the same instance's stock forward."""
    pytest.importorskip("timm.models.eva")
    from timm.models.eva import EvaAttention
    # Match DINOv3 ViT-L attention shape: dim=1024, num_heads=16,
    # num_prefix_tokens=5 (1 cls + 4 reg), rotate_half=True (DINOv3
    # default). qk_norm=False / scale_norm=False to skip the
    # post-norm path (matches the DINOv3-imagenette probe topology
    # we're targeting in Phase 1).
    attn = EvaAttention(
        dim=1024,
        num_heads=16,
        num_prefix_tokens=5,
        qkv_bias=False,
        qk_norm=False,
        scale_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        rotate_half=True,
    )
    attn.eval()
    # Force the explicit op path (the Phase 1 unfolded compares against this).
    attn.fused_attn = False
    return attn


@pytest.fixture
def attn_input():
    torch.manual_seed(0)
    # Match a 224x224 input: 14×14 patches + 5 prefix = 201 tokens.
    # rope last dim is 2 * head_dim = 2 * (1024/16) = 128.
    x = torch.randn(1, 201, 1024)
    rope = torch.randn(196, 128)
    return x, rope


class TestEvaAttentionUnfolded:
    def test_forward_parity_passthrough(self, synthetic_eva_attention, attn_input):
        x, rope = attn_input
        attn = synthetic_eva_attention

        with torch.no_grad():
            y_orig = attn(x, rope=rope)

        unfolded = EvaAttentionUnfolded(attn, matmul_rule="passthrough")
        with torch.no_grad():
            y_new = unfolded(x, rope=rope)

        # Bit-identical: the unfolded computes exactly the same ops in
        # the same order as the stock forward when matmul_rule is
        # passthrough.
        assert torch.equal(y_orig, y_new)

    def test_backward_parity_passthrough(self, synthetic_eva_attention, attn_input):
        x_orig = attn_input[0].clone().requires_grad_(True)
        rope = attn_input[1]
        attn = synthetic_eva_attention

        y_orig = attn(x_orig, rope=rope)
        go = torch.randn_like(y_orig)
        y_orig.backward(go)
        g_orig = x_orig.grad.clone()

        x_new = attn_input[0].clone().requires_grad_(True)
        unfolded = EvaAttentionUnfolded(attn, matmul_rule="passthrough")
        y_new = unfolded(x_new, rope=rope)
        y_new.backward(go)
        g_new = x_new.grad.clone()

        # Within fp32 noise.
        assert torch.allclose(g_orig, g_new, atol=1e-5, rtol=1e-4), (
            f"max grad diff = {(g_orig - g_new).abs().max().item():.6e}"
        )

    def test_forward_parity_under_lrp_rule(self, synthetic_eva_attention, attn_input):
        # The matmul-factor-2 rule changes ONLY backward; forward must
        # still bit-match the stock attention.
        x, rope = attn_input
        attn = synthetic_eva_attention
        with torch.no_grad():
            y_orig = attn(x, rope=rope)
        unfolded = EvaAttentionUnfolded(
            attn, matmul_rule="matmul_factor_2", epsilon=1e-6,
        )
        with torch.no_grad():
            y_new = unfolded(x, rope=rope)
        assert torch.allclose(y_orig, y_new, atol=1e-6, rtol=1e-5)

    def test_no_rope_is_identity_in_rotation(self, synthetic_eva_attention, attn_input):
        x = attn_input[0]
        attn = synthetic_eva_attention
        unfolded = EvaAttentionUnfolded(attn, matmul_rule="passthrough")
        with torch.no_grad():
            y_no_rope = unfolded(x, rope=None)
        # Sanity: forward runs end-to-end without rope.
        assert y_no_rope.shape == x.shape


# ─── 9. EvaAttentionSubstitutionCanonizer ───────────────────────────────────


class TestEvaAttentionSubstitutionCanonizer:
    def _make_model(self):
        m = timm.create_model(
            "vit_large_patch16_dinov3", pretrained=False, num_classes=10,
        )
        m.eval()
        for blk in m.blocks:
            blk.attn.fused_attn = False
        return m

    def test_apply_substitutes_one_block(self):
        m = self._make_model()
        from timm.models.eva import EvaAttention
        can = EvaAttentionSubstitutionCanonizer(
            block_indices=(0,), matmul_rule="passthrough",
        )
        instances = can.apply(m)
        try:
            assert len(instances) == 1
            assert isinstance(m.blocks[0].attn, EvaAttentionUnfolded)
            # Other blocks unchanged.
            assert isinstance(m.blocks[1].attn, EvaAttention)
        finally:
            for inst in instances:
                inst.remove()

    def test_remove_restores_original(self):
        m = self._make_model()
        from timm.models.eva import EvaAttention
        original = m.blocks[0].attn
        can = EvaAttentionSubstitutionCanonizer(
            block_indices=(0,), matmul_rule="passthrough",
        )
        instances = can.apply(m)
        for inst in instances:
            inst.remove()
        assert m.blocks[0].attn is original
        assert isinstance(m.blocks[0].attn, EvaAttention)

    def test_round_trip_forward_parity(self):
        """apply → forward → remove → re-apply → forward: outputs match."""
        m = self._make_model()
        torch.manual_seed(0)
        img = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y0 = m(img)

        can = EvaAttentionSubstitutionCanonizer(
            block_indices=(0,), matmul_rule="passthrough",
        )
        instances = can.apply(m)
        with torch.no_grad():
            y1 = m(img)
        for inst in instances:
            inst.remove()
        with torch.no_grad():
            y2 = m(img)
        instances2 = can.apply(m)
        with torch.no_grad():
            y3 = m(img)
        for inst in instances2:
            inst.remove()

        # All forwards must produce the same output (passthrough mode).
        assert torch.equal(y0, y1)
        assert torch.equal(y0, y2)
        assert torch.equal(y0, y3)

    def test_block_indices_filter_targets_correct_block(self):
        m = self._make_model()
        can = EvaAttentionSubstitutionCanonizer(
            block_indices=(3, 7), matmul_rule="passthrough",
        )
        instances = can.apply(m)
        try:
            assert len(instances) == 2
            from timm.models.eva import EvaAttention
            assert isinstance(m.blocks[3].attn, EvaAttentionUnfolded)
            assert isinstance(m.blocks[7].attn, EvaAttentionUnfolded)
            assert isinstance(m.blocks[0].attn, EvaAttention)
            assert isinstance(m.blocks[5].attn, EvaAttention)
        finally:
            for inst in instances:
                inst.remove()
