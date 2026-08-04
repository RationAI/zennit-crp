"""Unit tests for ``zennit_ext``.

After the LRP-cleanup refactor, every module here has a vanilla PyTorch
forward and autograd's standard backward. LRP behaviour (custom backward
= relevance flow) is layered in EXCLUSIVELY by zennit Hook rules (assigned via a composite layer_map)
at attribution time.

So the test matrix is:

* Vanilla forward of each module = bare ``torch`` op (bit-identical).
* Vanilla backward of each module = autograd's standard chain rule.
* With the corresponding canonizer applied: forward unchanged
  (bit-identical to vanilla), backward implements the LRP rule
  (conservation / identity / etc.).

Tests are weight-loading-free so CI can run them without downloading
checkpoints.

Run with::

    uv run pytest tests/test_attention_unfolded.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

timm = pytest.importorskip("timm")

from zennit.rules import Pass

from zennit_extensions import (
    AddBias,
    AlphaBetaMatmul,
    BilinearMatmul,
    ChunkAlongLastDim,
    EvaAttentionSubstitutionCanonizer,
    EvaAttentionUnfolded,
    LayerScaleMul,
    ReshapeMergeHeads,
    ResidualAdd,
    ResidualRatio,
    RotaryEmbedding,
    ScaleByConstant,
    SoftmaxAlongLastDim,
    Uniform,
)


# ─── helpers ────────────────────────────────────────────────────────────────


def _apply_canonizer(canonizer, *modules):
    """Wrap modules in a Sequential and apply the canonizer to it.

    Returns the list of canonizer instances created — call ``.remove()``
    on each at teardown.
    """
    root = nn.Sequential(*modules)
    return canonizer.apply(root)


def _apply_hook(hook, module):
    """Register a zennit Hook (LRP rule) on a single module; return a
    one-element list whose ``.remove()`` tears the hook down at teardown."""
    return [hook.register(module)]


# ─── 1. BilinearMatmul ──────────────────────────────────────────────────────


class TestBilinearMatmul:
    def test_vanilla_forward_matches_bare_matmul(self):
        torch.manual_seed(0)
        a = torch.randn(2, 3, 5, 7)
        b = torch.randn(2, 3, 7, 4)
        m = BilinearMatmul()
        assert torch.equal(m(a, b), a @ b)

    def test_vanilla_backward_matches_bare_matmul(self):
        """Critical: head training relies on chain-rule gradients here."""
        torch.manual_seed(1)
        a1 = torch.randn(2, 4, 5, requires_grad=True)
        b1 = torch.randn(2, 5, 6, requires_grad=True)
        a2 = a1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)
        go = torch.randn(2, 4, 6)
        (a1 @ b1).backward(go)
        BilinearMatmul()(a2, b2).backward(go)
        assert torch.allclose(a1.grad, a2.grad, atol=0)
        assert torch.allclose(b1.grad, b2.grad, atol=0)

    def test_canonizer_remove_restores_vanilla_backward(self):
        """After ``remove()`` the module's backward must be vanilla again
        — critical so heads remain trainable after attribution sessions."""
        torch.manual_seed(4)
        m = BilinearMatmul()
        instances = _apply_hook(AlphaBetaMatmul(), m)
        for inst in instances:
            inst.remove()
        # Now retrain-style call: should match bare matmul.
        a1 = torch.randn(2, 4, 5, requires_grad=True)
        b1 = torch.randn(2, 5, 6, requires_grad=True)
        a2 = a1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)
        go = torch.randn(2, 4, 6)
        (a1 @ b1).backward(go)
        m(a2, b2).backward(go)
        assert torch.allclose(a1.grad, a2.grad, atol=0)
        assert torch.allclose(b1.grad, b2.grad, atol=0)

    def test_alpha_beta_canonizer_forward_unchanged(self):
        torch.manual_seed(5)
        a = torch.randn(2, 4, 6) + 0.5
        b = torch.randn(2, 6, 3) + 0.5
        m = BilinearMatmul()
        instances = _apply_hook(
            AlphaBetaMatmul(alpha=0.5, beta=0.5, epsilon=1e-6), m,
        )
        try:
            assert torch.allclose(m(a, b), a @ b, atol=0)
        finally:
            for inst in instances:
                inst.remove()

    def test_alpha_beta_canonizer_conservation(self):
        """AlphaBeta-on-bilinear: ``sum(R_a) + sum(R_b) = (α+β)·sum(R_y)``;
        with ``α + β = 1`` we get exact conservation modulo ε.

        Conservation is element-wise modulo ε and only tight when both
        ``Y^+`` and ``Y^-`` are well away from 0 at every position. With
        small random operands a single position can have ``Y^- ≈ 0``
        (structural degeneracy), and that position's β-contribution
        drops to 0, breaking total conservation. We avoid this by using
        larger operands (100×50 @ 50×80 = 8000 positions, degeneracy is
        ε-rare) and mean-zero so neither ``Y^+`` nor ``Y^-`` is dominated.
        """
        torch.manual_seed(6)
        a = (torch.randn(1, 100, 50) * 2.0).requires_grad_(True)
        b = (torch.randn(1, 50, 80) * 2.0).requires_grad_(True)
        m = BilinearMatmul()
        instances = _apply_hook(
            AlphaBetaMatmul(alpha=0.5, beta=0.5, epsilon=1e-6), m,
        )
        try:
            y = m(a, b)
            R_y = torch.randn_like(y)
            y.backward(R_y)
            assert torch.isclose(
                a.grad.sum() + b.grad.sum(), R_y.sum(),
                rtol=1e-3, atol=1e-3,
            ), (
                f"sum(R_a)+sum(R_b)={a.grad.sum().item() + b.grad.sum().item():.6f} "
                f"vs sum(R_y)={R_y.sum().item():.6f}"
            )
        finally:
            for inst in instances:
                inst.remove()


# ─── 2. SoftmaxAlongLastDim ─────────────────────────────────────────────────


class TestSoftmaxAlongLastDim:
    def test_vanilla_forward_matches_F_softmax(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 7)
        m = SoftmaxAlongLastDim()
        assert torch.equal(m(x), F.softmax(x, dim=-1))

    def test_vanilla_backward_uses_softmax_jacobian(self):
        torch.manual_seed(1)
        x1 = torch.randn(2, 5, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        go = torch.randn(2, 5)
        F.softmax(x1, dim=-1).backward(go)
        SoftmaxAlongLastDim()(x2).backward(go)
        assert torch.allclose(x1.grad, x2.grad, atol=0)

    def test_identity_canonizer_forward_unchanged(self):
        torch.manual_seed(2)
        x = torch.randn(2, 3, 7)
        m = SoftmaxAlongLastDim()
        instances = _apply_hook(Pass(), m)
        try:
            assert torch.allclose(m(x), F.softmax(x, dim=-1), atol=0)
        finally:
            for inst in instances:
                inst.remove()

    def test_identity_canonizer_passes_relevance_through(self):
        """AttnLRP Eq. 9: R_in == R_out (identity rule)."""
        torch.manual_seed(3)
        x = torch.randn(2, 5, requires_grad=True)
        m = SoftmaxAlongLastDim()
        instances = _apply_hook(Pass(), m)
        try:
            y = m(x)
            R_y = torch.randn_like(y)
            y.backward(R_y)
            assert torch.equal(x.grad, R_y)
        finally:
            for inst in instances:
                inst.remove()


# ─── 3. RotaryEmbedding (vanilla, no canonizer needed) ──────────────────────


class TestRotaryEmbedding:
    @pytest.fixture
    def rope_setup(self):
        from timm.layers import apply_rot_embed_cat
        torch.manual_seed(0)
        # (B, num_heads, N, head_dim) — RoPE last dim is 2 * head_dim.
        q = torch.randn(1, 2, 10, 8)
        rope = torch.randn(10, 16)
        return q, rope, apply_rot_embed_cat

    def test_forward_parity_no_prefix_no_detach(self, rope_setup):
        q, rope, apply_rot_embed_cat = rope_setup
        m = RotaryEmbedding(num_prefix_tokens=0, rotate_half=False, detach_rope=False)
        with torch.no_grad():
            y = m(q, rope)
            y_ref = apply_rot_embed_cat(q, rope, half=False)
        assert torch.equal(y, y_ref)

    def test_forward_parity_with_rotate_half(self, rope_setup):
        q, rope, apply_rot_embed_cat = rope_setup
        m = RotaryEmbedding(num_prefix_tokens=0, rotate_half=True, detach_rope=False)
        with torch.no_grad():
            y = m(q, rope)
            y_ref = apply_rot_embed_cat(q, rope, half=True)
        assert torch.equal(y, y_ref)

    def test_rope_none_is_identity(self, rope_setup):
        q, _, _ = rope_setup
        m = RotaryEmbedding(num_prefix_tokens=0)
        assert torch.equal(m(q, None), q)

    def test_detach_rope_severs_rope_grad(self, rope_setup):
        q, rope, _ = rope_setup
        rope = rope.clone().requires_grad_(True)
        q = q.clone().requires_grad_(True)
        m = RotaryEmbedding(num_prefix_tokens=0, detach_rope=True)
        m(q, rope).sum().backward()
        assert rope.grad is None  # detach severed it
        assert q.grad is not None


# ─── 4. ScaleByConstant ─────────────────────────────────────────────────────


class TestScaleByConstant:
    def test_vanilla_forward_matches_multiplication(self):
        x = torch.randn(2, 5)
        m = ScaleByConstant(0.5)
        assert torch.equal(m(x), x * 0.5)

    def test_vanilla_backward_uses_chain_rule(self):
        x = torch.randn(2, 5, requires_grad=True)
        m = ScaleByConstant(0.5)
        m(x).sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, 0.5))

    def test_identity_canonizer_forward_unchanged(self):
        x = torch.randn(2, 5)
        m = ScaleByConstant(0.5)
        instances = _apply_hook(Pass(), m)
        try:
            assert torch.equal(m(x), x * 0.5)
        finally:
            for inst in instances:
                inst.remove()

    def test_identity_canonizer_passes_relevance_through(self):
        """Constant absorbs no relevance: R_in == R_out."""
        x = torch.randn(2, 5, requires_grad=True)
        m = ScaleByConstant(0.5)
        instances = _apply_hook(Pass(), m)
        try:
            y = m(x)
            R_y = torch.randn_like(y)
            y.backward(R_y)
            assert torch.equal(x.grad, R_y)
        finally:
            for inst in instances:
                inst.remove()


# ─── 5. Simple kernels (no rules — tested for shape / parity only) ─────────


class TestSimpleKernels:
    def test_chunk_split_and_concat_round_trip(self):
        x = torch.randn(2, 3, 12)
        m = ChunkAlongLastDim(3)
        chunks = m(x)
        assert len(chunks) == 3
        assert chunks[0].shape == (2, 3, 4)
        assert torch.equal(torch.cat(chunks, dim=-1), x)

    def test_reshape_merge_heads(self):
        # (B=2, H=4, N=10, hd=6) → (2, 10, 24)
        x = torch.randn(2, 4, 10, 6)
        m = ReshapeMergeHeads()
        y = m(x)
        assert y.shape == (2, 10, 24)
        # Equivalent manual op:
        ref = x.transpose(1, 2).reshape(2, 10, 24)
        assert torch.equal(y, ref)

    def test_add_bias_with_none(self):
        x = torch.randn(2, 5)
        m = AddBias()
        assert torch.equal(m(x, None), x)

    def test_add_bias_with_tensor(self):
        x = torch.randn(2, 5)
        b = torch.randn(2, 5)
        m = AddBias()
        assert torch.equal(m(x, b), x + b)


# ─── 6. ResidualAdd ─────────────────────────────────────────────────────────


class TestResidualAdd:
    def test_vanilla_forward_matches_addition(self):
        x = torch.randn(2, 5)
        b = torch.randn(2, 5)
        m = ResidualAdd()
        assert torch.equal(m(x, b), x + b)

    def test_vanilla_backward_routes_grad_to_both(self):
        x = torch.randn(2, 5, requires_grad=True)
        b = torch.randn(2, 5, requires_grad=True)
        m = ResidualAdd()
        y = m(x, b)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        assert torch.equal(x.grad, R_y)
        assert torch.equal(b.grad, R_y)

    def test_ratio_canonizer_conservation(self):
        torch.manual_seed(0)
        x = (torch.randn(2, 5) + 0.5).requires_grad_(True)
        b = (torch.randn(2, 5) + 0.5).requires_grad_(True)
        m = ResidualAdd()
        instances = _apply_hook(ResidualRatio(epsilon=1e-6), m)
        try:
            y = m(x, b)
            R_y = torch.randn_like(y)
            y.backward(R_y)
            total = x.grad.sum() + b.grad.sum()
            assert torch.isclose(total, R_y.sum(), rtol=1e-3, atol=1e-3)
        finally:
            for inst in instances:
                inst.remove()


# ─── 7. LayerScaleMul ───────────────────────────────────────────────────────


class TestLayerScaleMul:
    def test_vanilla_forward_multiplies_by_gamma(self):
        gamma = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.randn(2, 4, 3)
        m = LayerScaleMul(gamma)
        assert torch.equal(m(x), gamma * x)

    def test_vanilla_backward_full_grad(self):
        gamma = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.randn(2, 4, 3, requires_grad=True)
        m = LayerScaleMul(gamma)
        y = m(x)
        R_y = torch.randn_like(y)
        y.backward(R_y)
        assert torch.allclose(x.grad, R_y * gamma)

    def test_uniform_canonizer_halves_relevance(self):
        gamma = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.randn(2, 4, 3, requires_grad=True)
        m = LayerScaleMul(gamma)
        instances = _apply_hook(Uniform(factor=2), m)
        try:
            y = m(x)
            R_y = torch.randn_like(y)
            y.backward(R_y)
            # Uniform rule: divide upstream R by 2 BEFORE bare-grad
            # propagation, so x.grad = (R_y / 2) * gamma.
            assert torch.allclose(x.grad, (R_y / 2) * gamma)
        finally:
            for inst in instances:
                inst.remove()


# ─── 8. EvaAttentionUnfolded ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_eva_attention():
    """Construct a synthetic EvaAttention without downloading any
    checkpoint — random init is sufficient for forward/backward parity
    tests since we compare against the same instance's stock forward."""
    pytest.importorskip("timm.models.eva")
    from timm.models.eva import EvaAttention
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
    attn.fused_attn = False
    return attn


@pytest.fixture
def attn_input():
    torch.manual_seed(0)
    x = torch.randn(1, 201, 1024)
    rope = torch.randn(196, 128)
    return x, rope


class TestEvaAttentionUnfolded:
    def test_vanilla_forward_parity_with_stock(self, synthetic_eva_attention, attn_input):
        """Vanilla unfolded must bit-match the stock attention forward."""
        x, rope = attn_input
        attn = synthetic_eva_attention
        with torch.no_grad():
            y_orig = attn(x, rope=rope)
        unfolded = EvaAttentionUnfolded(attn)
        with torch.no_grad():
            y_new = unfolded(x, rope=rope)
        assert torch.equal(y_orig, y_new)

    def test_vanilla_backward_parity_with_stock(self, synthetic_eva_attention, attn_input):
        """Critical for training: same chain-rule gradients as stock."""
        x_orig = attn_input[0].clone().requires_grad_(True)
        rope = attn_input[1]
        attn = synthetic_eva_attention
        y_orig = attn(x_orig, rope=rope)
        go = torch.randn_like(y_orig)
        y_orig.backward(go)
        g_orig = x_orig.grad.clone()

        x_new = attn_input[0].clone().requires_grad_(True)
        unfolded = EvaAttentionUnfolded(attn)
        y_new = unfolded(x_new, rope=rope)
        y_new.backward(go)
        g_new = x_new.grad.clone()

        assert torch.allclose(g_orig, g_new, atol=1e-5, rtol=1e-4), (
            f"max grad diff = {(g_orig - g_new).abs().max().item():.6e}"
        )

    def test_forward_unchanged_when_factor2_canonizer_applied(
        self, synthetic_eva_attention, attn_input,
    ):
        """The AlphaBeta bilinear rule changes ONLY backward; forward must
        still bit-match the stock attention."""
        x, rope = attn_input
        attn = synthetic_eva_attention
        with torch.no_grad():
            y_orig = attn(x, rope=rope)
        unfolded = EvaAttentionUnfolded(attn)
        instances = [
            AlphaBetaMatmul(alpha=0.5, beta=0.5, epsilon=1e-6).register(sm)
            for sm in unfolded.modules() if isinstance(sm, BilinearMatmul)
        ]
        try:
            with torch.no_grad():
                y_new = unfolded(x, rope=rope)
            assert torch.allclose(y_orig, y_new, atol=1e-6, rtol=1e-5)
        finally:
            for inst in instances:
                inst.remove()

    def test_no_rope_is_identity_in_rotation(self, synthetic_eva_attention, attn_input):
        x = attn_input[0]
        attn = synthetic_eva_attention
        unfolded = EvaAttentionUnfolded(attn)
        with torch.no_grad():
            y_no_rope = unfolded(x, rope=None)
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
        can = EvaAttentionSubstitutionCanonizer(block_indices=(0,))
        instances = can.apply(m)
        try:
            assert len(instances) == 1
            assert isinstance(m.blocks[0].attn, EvaAttentionUnfolded)
            assert isinstance(m.blocks[1].attn, EvaAttention)
        finally:
            for inst in instances:
                inst.remove()

    def test_remove_restores_original(self):
        m = self._make_model()
        from timm.models.eva import EvaAttention
        original = m.blocks[0].attn
        can = EvaAttentionSubstitutionCanonizer(block_indices=(0,))
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

        can = EvaAttentionSubstitutionCanonizer(block_indices=(0,))
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

        assert torch.equal(y0, y1)
        assert torch.equal(y0, y2)
        assert torch.equal(y0, y3)

    def test_block_indices_filter_targets_correct_block(self):
        m = self._make_model()
        can = EvaAttentionSubstitutionCanonizer(block_indices=(3, 7))
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


# ─── 10. TimmAttentionUnfolded — end-to-end on synthetic random model ───────


@pytest.fixture(scope="module")
def synthetic_timm_attention():
    """Construct a synthetic timm vision_transformer.Attention without
    downloading any checkpoint — random init is sufficient for forward /
    backward parity tests against the same instance's stock forward."""
    pytest.importorskip("timm.models.vision_transformer")
    from timm.models.vision_transformer import Attention as TimmAttention
    # vit_base_patch16_224-ish (dim=768, num_heads=12, head_dim=64).
    attn = TimmAttention(
        dim=768, num_heads=12, qkv_bias=True,
        qk_norm=False, attn_drop=0.0, proj_drop=0.0,
    )
    attn.eval()
    # Force the explicit-math path so we can compare against the stock
    # forward — the unfolded form takes the explicit path either way.
    attn.fused_attn = False
    return attn


@pytest.fixture
def timm_attn_input():
    torch.manual_seed(0)
    # vit_base_patch16_224 has 1 cls + 14*14 patches = 197 tokens, dim=768.
    x = torch.randn(1, 197, 768)
    return x


class TestTimmAttentionUnfolded:
    """Mirrors TestEvaAttentionUnfolded for the standard timm Attention path.

    Note on the ``attn.fused_attn = False`` flag in the fixture: modern timm
    Attention defaults to ``fused_attn=True`` and dispatches through
    ``F.scaled_dot_product_attention``. We need the explicit-math path on
    the source instance for ``torch.equal(stock(x), unfolded(x))`` parity
    assertions. In production this is irrelevant — after substitution the
    original instance is swapped out and never called.
    """

    def test_vanilla_forward_parity_with_stock(self, synthetic_timm_attention, timm_attn_input):
        x = timm_attn_input
        attn = synthetic_timm_attention
        with torch.no_grad():
            y_orig = attn(x)
        from zennit_extensions import TimmAttentionUnfolded
        unfolded = TimmAttentionUnfolded(attn)
        with torch.no_grad():
            y_new = unfolded(x)
        assert torch.equal(y_orig, y_new)

    def test_vanilla_backward_parity_with_stock(self, synthetic_timm_attention, timm_attn_input):
        """Critical for training: same chain-rule gradients as stock."""
        x_orig = timm_attn_input.clone().requires_grad_(True)
        attn = synthetic_timm_attention
        y_orig = attn(x_orig)
        go = torch.randn_like(y_orig)
        y_orig.backward(go)
        g_orig = x_orig.grad.clone()

        x_new = timm_attn_input.clone().requires_grad_(True)
        from zennit_extensions import TimmAttentionUnfolded
        unfolded = TimmAttentionUnfolded(attn)
        y_new = unfolded(x_new)
        y_new.backward(go)
        g_new = x_new.grad.clone()

        assert torch.allclose(g_orig, g_new, atol=1e-5, rtol=1e-4), (
            f"max grad diff = {(g_orig - g_new).abs().max().item():.6e}"
        )

    def test_forward_unchanged_when_alpha_beta_canonizer_applied(
        self, synthetic_timm_attention, timm_attn_input,
    ):
        """The AlphaBeta bilinear rule changes ONLY backward; forward must
        still bit-match the stock attention."""
        x = timm_attn_input
        attn = synthetic_timm_attention
        with torch.no_grad():
            y_orig = attn(x)
        from zennit_extensions import TimmAttentionUnfolded
        unfolded = TimmAttentionUnfolded(attn)
        instances = [
            AlphaBetaMatmul(alpha=0.5, beta=0.5, epsilon=1e-6).register(sm)
            for sm in unfolded.modules() if isinstance(sm, BilinearMatmul)
        ]
        try:
            with torch.no_grad():
                y_new = unfolded(x)
            assert torch.allclose(y_orig, y_new, atol=1e-6, rtol=1e-5)
        finally:
            for inst in instances:
                inst.remove()

    def test_exposes_concept_hook_submodules(self, synthetic_timm_attention):
        """The unfolded form must expose the named LRP-inspection submodules
        the concept classes target: q_lrp_probe / k_lrp_probe / v_lrp_probe
        (3D, post qkv split), proj_drop (3D, attention output), context
        (4D, pre-merge), plus qk_scores / softmax for diagnostics."""
        from zennit_extensions import (
            TimmAttentionUnfolded,
            LRPInspectionLayer,
        )
        unfolded = TimmAttentionUnfolded(synthetic_timm_attention)
        for name in (
            "context", "q_lrp_probe", "k_lrp_probe", "v_lrp_probe",
            "proj_drop", "qk_scores", "softmax",
        ):
            assert hasattr(unfolded, name), f"missing hookable submodule: {name}"
        for probe_name in ("q_lrp_probe", "k_lrp_probe", "v_lrp_probe"):
            assert isinstance(getattr(unfolded, probe_name), LRPInspectionLayer)


# ─── 11. TimmAttentionSubstitutionCanonizer ────────────────────────────────


class TestTimmAttentionSubstitutionCanonizer:
    """Mirrors TestEvaAttentionSubstitutionCanonizer for the standard timm
    Attention substitution canonizer.
    """

    def _make_model(self):
        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
        m.eval()
        for blk in m.blocks:
            blk.attn.fused_attn = False
        return m

    def test_apply_substitutes_one_block(self):
        m = self._make_model()
        from timm.models.vision_transformer import Attention as TimmAttention
        from zennit_extensions import (
            TimmAttentionSubstitutionCanonizer, TimmAttentionUnfolded,
        )
        can = TimmAttentionSubstitutionCanonizer(block_indices=(0,))
        instances = can.apply(m)
        try:
            assert len(instances) == 1
            assert isinstance(m.blocks[0].attn, TimmAttentionUnfolded)
            assert isinstance(m.blocks[1].attn, TimmAttention)
        finally:
            for inst in instances:
                inst.remove()

    def test_remove_restores_original(self):
        m = self._make_model()
        from timm.models.vision_transformer import Attention as TimmAttention
        from zennit_extensions import TimmAttentionSubstitutionCanonizer
        original = m.blocks[0].attn
        can = TimmAttentionSubstitutionCanonizer(block_indices=(0,))
        instances = can.apply(m)
        for inst in instances:
            inst.remove()
        assert m.blocks[0].attn is original
        assert isinstance(m.blocks[0].attn, TimmAttention)

    def test_round_trip_forward_parity(self):
        """apply → forward → remove → re-apply → forward: outputs match."""
        from zennit_extensions import TimmAttentionSubstitutionCanonizer
        m = self._make_model()
        torch.manual_seed(0)
        img = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y0 = m(img)

        can = TimmAttentionSubstitutionCanonizer(block_indices=(0,))
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

        assert torch.equal(y0, y1)
        assert torch.equal(y0, y2)
        assert torch.equal(y0, y3)

    def test_does_not_fire_on_eva_blocks(self):
        """Each substitution canonizer's isinstance filter must skip the
        other backbone's attention class — so both can be bundled into
        one composite without coupling."""
        from zennit_extensions import TimmAttentionSubstitutionCanonizer
        m_eva = timm.create_model(
            "vit_large_patch16_dinov3", pretrained=False, num_classes=10,
        )
        m_eva.eval()
        can = TimmAttentionSubstitutionCanonizer(block_indices=None)
        instances = can.apply(m_eva)
        # No standard timm Attention modules → no substitutions.
        assert len(instances) == 0
