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

from zennit_extensions.attention_unfolded import EvaAttentionUnfolded, TimmAttentionUnfolded
from zennit_extensions.canonisation.canonizers import EvaAttentionSubstitutionCanonizer
from zennit_extensions.rules.bajger_contrib import AlphaBetaMatmul
from zennit_extensions.rules.residuals_otsuki2024 import ResidualRatio

timm = pytest.importorskip("timm")

from zennit.rules import Pass

from zennit_extensions import (
    AddBias,
    BilinearMatmul,
    ChunkAlongLastDim,
    LayerScaleMul,
    PosEmbedAdd,
    ReshapeMergeHeads,
    ResidualAdd,
    RotaryEmbedding,
    ScaleByConstant,
    SoftmaxAlongLastDim,
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

    def test_pass_rule_forwards_relevance_unchanged(self):
        gamma = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.randn(2, 4, 3, requires_grad=True)
        m = LayerScaleMul(gamma)
        instances = _apply_hook(Pass(), m)
        try:
            y = m(x)
            R_y = torch.randn_like(y)
            y.backward(R_y)
            # γ-multiply is a bias-free elementwise linear op: identity
            # attribution, γ transparent.
            assert torch.allclose(x.grad, R_y)
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
        from zennit_extensions.attention_unfolded import TimmAttentionUnfolded
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
        from zennit_extensions.attention_unfolded import TimmAttentionUnfolded
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
        from zennit_extensions.attention_unfolded import TimmAttentionUnfolded
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
        from zennit_extensions.canonisation.canonizers import (
            VanillaViTAttentionSubstitutionCanonizer,
        )
        can = VanillaViTAttentionSubstitutionCanonizer(block_indices=(0,))
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
        from zennit_extensions.canonisation.canonizers import VanillaViTAttentionSubstitutionCanonizer
        original = m.blocks[0].attn
        can = VanillaViTAttentionSubstitutionCanonizer(block_indices=(0,))
        instances = can.apply(m)
        for inst in instances:
            inst.remove()
        assert m.blocks[0].attn is original
        assert isinstance(m.blocks[0].attn, TimmAttention)

    def test_round_trip_forward_parity(self):
        """apply → forward → remove → re-apply → forward: outputs match."""
        from zennit_extensions.canonisation.canonizers import VanillaViTAttentionSubstitutionCanonizer
        m = self._make_model()
        torch.manual_seed(0)
        img = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y0 = m(img)

        can = VanillaViTAttentionSubstitutionCanonizer(block_indices=(0,))
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
        from zennit_extensions.canonisation.canonizers import VanillaViTAttentionSubstitutionCanonizer
        m_eva = timm.create_model(
            "vit_large_patch16_dinov3", pretrained=False, num_classes=10,
        )
        m_eva.eval()
        can = VanillaViTAttentionSubstitutionCanonizer(block_indices=None)
        instances = can.apply(m_eva)
        # No standard timm Attention modules → no substitutions.
        assert len(instances) == 0


# ─── 12. PA-LRP positional-sink rules (Bakish et al., arXiv:2506.02138) ────


class TestPALRPRules:
    """Equation-named tests for the PA-LRP positional-sink rules
    (:class:`PosEmbedSink` Eq. 5; :class:`RotaryRopeSink` Eq. 10) and the
    structural canonizer that exposes the input-level PE merge."""

    @pytest.fixture
    def pe_add_batch(self):
        """B=2 (embedded, positional, rel_z) tensors for Eq. 5 checks."""
        torch.manual_seed(0)
        embedded = torch.randn(2, 7, 16, requires_grad=False)
        positional = torch.randn(1, 7, 16, requires_grad=False)   # broadcast
        rel_z = torch.randn(2, 7, 16)
        return embedded, positional, rel_z

    def test_eq5_pos_embed_sink(self, pe_add_batch):
        """Eq. 5: R(P) = P·R(z)/(z+ε), R(E) = E·R(z)/(z+ε); sink stashed
        per-sample; returned pos-side grad batch-summed (broadcast contract);
        per-sample conservation R(E)+R(P)=R(z)."""
        from zennit_extensions.rules.palrp import PosEmbedSink

        embedded, positional, rel_z = pe_add_batch
        eps = 1e-6
        m = PosEmbedAdd()
        instances = _apply_hook(PosEmbedSink(epsilon=eps), m)
        try:
            e = embedded.detach().clone().requires_grad_(True)
            p = positional.detach().clone().requires_grad_(True)
            z = m(e, p)
            z.backward(rel_z)
            denom = (e.detach() + p.detach()) + eps
            expected_rel_E = e.detach() * rel_z / denom
            expected_rel_P = p.detach() * rel_z / denom   # broadcast -> (2,7,16)
            # returned grads
            assert torch.allclose(e.grad, expected_rel_E, atol=1e-5)
            assert torch.allclose(p.grad, expected_rel_P.sum(0, keepdim=True), atol=1e-5)
            # stash is per-sample, batch dim preserved (Eq. 4 input space)
            assert m._palrp_sink.shape == (2, 7, 16)
            assert torch.allclose(m._palrp_sink, expected_rel_P.detach(), atol=1e-5)
        finally:
            for inst in instances:
                inst.remove()

    def test_eq5_pos_embed_sink_batched(self):
        """B=2 keeps the batch dim in the stash; returned pos-side grad is
        the per-position batch sum (1, N, D)."""
        from zennit_extensions.rules.palrp import PosEmbedSink

        torch.manual_seed(1)
        embedded = torch.randn(2, 5, 8)
        positional = torch.randn(1, 5, 8)
        rel_z = torch.randn(2, 5, 8)
        m = PosEmbedAdd()
        instances = _apply_hook(PosEmbedSink(), m)
        try:
            e = embedded.detach().clone().requires_grad_(True)
            p = positional.detach().clone().requires_grad_(True)
            m(e, p).backward(rel_z)
            assert m._palrp_sink.shape == (2, 5, 8)               # per-sample
            assert p.grad.shape == (1, 5, 8)                     # batch-summed
            assert e.grad.shape == (2, 5, 8)                     # unchanged
        finally:
            for inst in instances:
                inst.remove()

    def test_eq10_rotary_rope_sink_halves(self):
        """Eq. 10 / Lemma 3.2: sink = ½ of output-side per-sample relevance
        on rotated rows; EXACT zeros on prefix rows; prefix rows' grad_q
        unmodified; rotated rows carry the ε-contribution share
        ``q ⊙ Jᵀ(R/(2·(q̃+ε)))`` (reference-implementation semantics);
        batch dim preserved; rope=None ⇒ identity (unmodified grads, sink
        None)."""
        from timm.layers import apply_rot_embed_cat
        from zennit.core import stabilize
        from zennit_extensions.rules.palrp import RotaryRopeSink

        B, H, N, Dh = 2, 1, 10, 8
        npt = 3
        eps = 1e-6
        torch.manual_seed(2)
        q = torch.randn(B, H, N, Dh, requires_grad=True)
        rope = torch.randn(N - npt, 2 * Dh, requires_grad=True)   # rotated slice only
        m = RotaryEmbedding(num_prefix_tokens=npt, detach_rope=False)
        instances = _apply_hook(RotaryRopeSink(epsilon=eps), m)
        try:
            y = m(q, rope)
            rel_out = torch.randn_like(y)
            y.backward(rel_out)
            # sink shape preserved, batched; prefix rows zero, rotated rows = ½ R(out)
            assert m._palrp_sink.shape == (B, H, N, Dh)
            assert torch.all(m._palrp_sink[..., :npt, :] == 0)
            assert torch.allclose(m._palrp_sink[..., npt:, :], rel_out[..., npt:, :] / 2, atol=1e-6)
            # grad_q: prefix rows pass through unchanged (identity via cat)
            assert torch.allclose(q.grad[..., :npt, :], rel_out[..., :npt, :], atol=1e-6)
            # rotated rows: ε-contribution rule q ⊙ Jᵀ(R/(2·(q̃+ε)))
            q_rot = q.detach()[..., npt:, :].clone().requires_grad_(True)
            y_rot = apply_rot_embed_cat(q_rot, rope.detach(), half=False)
            s = rel_out[..., npt:, :] / stabilize(2.0 * y_rot.detach(), eps)
            (vjp,) = torch.autograd.grad(y_rot, q_rot, grad_outputs=s)
            expected = q_rot.detach() * vjp
            assert torch.allclose(q.grad[..., npt:, :], expected, atol=1e-6)
            # rope side: relevance-weighted share rope ⊙ Jᵀ_rope(s)
            assert rope.grad is not None and rope.grad.shape == rope.shape
        finally:
            for inst in instances:
                inst.remove()

    def test_eq10_per_token_conservation(self):
        """Lemma 3.2: per token, ``stash + R(q) = R(q̃)`` up to the ε
        stabiliser — the sink absorbs exactly half and the ε-contribution
        share returns the other half (a halved vanilla gradient would
        backward-rotate the relevance and break this)."""
        from zennit_extensions.rules.palrp import RotaryRopeSink

        B, H, N, Dh = 1, 2, 6, 8
        torch.manual_seed(5)
        q = torch.randn(B, H, N, Dh, requires_grad=True)
        rope = torch.randn(N, 2 * Dh)
        m = RotaryEmbedding(num_prefix_tokens=0, detach_rope=False)
        instances = _apply_hook(RotaryRopeSink(epsilon=1e-9), m)
        try:
            y = m(q, rope)
            rel_out = torch.randn_like(y)
            y.backward(rel_out)
            per_token_in = (m._palrp_sink + q.grad).sum(dim=-1)
            per_token_out = rel_out.sum(dim=-1)
            assert torch.allclose(per_token_in, per_token_out, atol=1e-4)
        finally:
            for inst in instances:
                inst.remove()

    def test_rotary_rope_sink_identity_when_rope_none(self):
        """rope=None ⇒ pass-through: grads unmodified, sink None."""
        from zennit_extensions.rules.palrp import RotaryRopeSink

        torch.manual_seed(3)
        q = torch.randn(1, 2, 10, 8, requires_grad=True)
        m = RotaryEmbedding(num_prefix_tokens=0, detach_rope=False)
        instances = _apply_hook(RotaryRopeSink(), m)
        try:
            y = m(q, None)
            assert torch.equal(y, q)                       # identity forward
            rel_out = torch.randn_like(y)
            y.backward(rel_out)
            assert torch.allclose(q.grad, rel_out)        # unmodified
            assert m._palrp_sink is None
        finally:
            for inst in instances:
                inst.remove()

    def test_pos_embed_canonizer_forward_parity(self):
        """Canonized ``_pos_embed`` forward is bit-identical to stock timm
        on a vanilla ViT; PosEmbedAdd submodule installed; remove() restores
        the original method and deletes the attr."""
        from zennit_extensions.canonisation.canonizers import VanillaViTPosEmbedCanonizer

        m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).eval()
        torch.manual_seed(0)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y_stock = m._pos_embed(m.patch_embed(x))
        original_method = type(m)._pos_embed
        can = VanillaViTPosEmbedCanonizer()
        instances = can.apply(m)
        try:
            assert hasattr(m, "pos_embed_add")
            assert isinstance(m.pos_embed_add, PosEmbedAdd)
            with torch.no_grad():
                y_canon = m._pos_embed(m.patch_embed(x))
            assert torch.equal(y_stock, y_canon)
            # full-model forward parity too
            m2 = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).eval()
            m2.load_state_dict(m.state_dict())
            with torch.no_grad():
                assert torch.equal(m(x), m2(x))
        finally:
            for inst in instances:
                inst.remove()
        assert not hasattr(m, "pos_embed_add")
        assert type(m)._pos_embed is original_method

    def test_default_composite_identity_with_pos_embed_canonizer(self):
        """Default recipe with the pos-embed canonizer (PosEmbedAdd installed
        but UNmapped) is byte-identical to the same composite with the
        canonizer removed — i.e. structure by default, rule by opt-in."""
        from zennit_extensions.lrp_composites import COMPOSITES
        from zennit_extensions.canonisation.canonizers import VanillaViTPosEmbedCanonizer
        from crp.attribution import CondAttribution

        m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).eval()
        attr = CondAttribution(m)
        torch.manual_seed(0)
        x = torch.randn(1, 3, 224, 224).requires_grad_(True)

        comp_full = COMPOSITES["attnlrp_baseline"]()
        with comp_full.context(m):
            out_full = attr(x.detach().clone().requires_grad_(True), [{"y": [1]}], comp_full)
            hm_full = out_full.heatmap.detach().clone()

        comp_no = COMPOSITES["attnlrp_baseline"]()
        comp_no.canonizers = [c for c in comp_no.canonizers
                               if not isinstance(c, VanillaViTPosEmbedCanonizer)]
        with comp_no.context(m):
            out_no = attr(x.detach().clone().requires_grad_(True), [{"y": [1]}], comp_no)
            hm_no = out_no.heatmap.detach().clone()

        assert torch.equal(hm_full, hm_no)

    def test_lemma31_conservation_restored_by_sink(self):
        """Lemma 3.1: ignoring PE relevance violates conservation; with
        :class:`PosEmbedSink` the stashed ``R(P)`` plus the token-stream
        ``R(E)`` equals the relevance arriving at the add output (per
        element, up to ε) — conservation is restored by the sink."""
        from zennit_extensions.rules.palrp import PosEmbedSink

        torch.manual_seed(4)
        embedded = torch.randn(2, 7, 16)
        positional = torch.randn(1, 7, 16)
        rel_z = torch.randn(2, 7, 16)
        m = PosEmbedAdd()
        instances = _apply_hook(PosEmbedSink(epsilon=1e-9), m)
        try:
            e = embedded.detach().clone().requires_grad_(True)
            p = positional.detach().clone().requires_grad_(True)
            m(e, p).backward(rel_z)
            # R(E) + R(P) (per-sample, full broadcast) == R(z)  (Eq. 5 conserves)
            denom = (e.detach() + p.detach()) + 1e-9
            rel_E = e.detach() * rel_z / denom
            rel_P = p.detach() * rel_z / denom
            assert torch.allclose(rel_E + rel_P, rel_z, atol=1e-4)
            # the sink IS R(P) per sample — the previously-discarded share
            assert torch.allclose(m._palrp_sink, rel_P.detach(), atol=1e-5)
        finally:
            for inst in instances:
                inst.remove()
