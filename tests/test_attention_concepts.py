"""Unit tests for crp.attention_concepts on the unfolded attention API.

Pure tensor-level — no model load, no real attention forward — so they
run anywhere torch is installed. The tests construct synthetic relevance
tensors of the shape each concept expects and exercise mask + attribute
+ shape semantics.

Concepts under test:
* :class:`HeadConcept` (per-head, optional dim_split) on `attn.context`
* :class:`QConcept`, :class:`KConcept`, :class:`VConcept`
  (Q/K/V per-head, optional dim_split) on `attn.rope_q`/`attn.rope_k`/`attn.v_id`
* :class:`AttnOutputDimConcept` on `attn.proj_drop` (per-channel,
  spatial aggregation)
"""
from __future__ import annotations

import pytest
import torch

from crp.attention_concepts import (
    HeadConcept,
    QConcept,
    KConcept,
    VConcept,
    AttnOutputDimConcept,
    RegisterTokenConcept,
)


# ── shared fixtures ──────────────────────────────────────────────────────────
#
# Token axis layout for the prefix-isolation tests: first NUM_PREFIX tokens
# are register / cls tokens (addressable via RegisterTokenConcept); remaining
# (N - NUM_PREFIX) tokens are spatial patches (addressable via the per-head
# and AttnOutputDim concepts). The default tests below use num_prefix_tokens=0
# so the prefix slice is empty — back-compat with the no-prefix model case.

B, NUM_HEADS, N, HEAD_DIM = 2, 4, 9, 6
NUM_PREFIX = 2
EMBED_DIM = NUM_HEADS * HEAD_DIM


@pytest.fixture
def per_head_relevance():
    """(B, num_heads, N, head_dim) — the shape used by HeadConcept and Q/K/V."""
    torch.manual_seed(0)
    return torch.randn(B, NUM_HEADS, N, HEAD_DIM)


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


# ── AttnOutputDimConcept ─────────────────────────────────────────────────────


class TestAttnOutputDimConcept:
    """Per-channel conditioning at the post-projection residual stream.
    Spatial token axis aggregated in attribute (no per-token conditioning).
    """

    LAYER = "blocks.0.attn.proj_drop"

    def _make(self):
        c = AttnOutputDimConcept()
        c.register_layer(self.LAYER, EMBED_DIM)
        return c

    def test_mask_zeros_other_channels(self, proj_output_relevance):
        c = self._make()
        m = c.mask(batch_id=0, concept_ids=[5, 7], layer_name=self.LAYER)
        masked = m(proj_output_relevance.clone())
        for ch in range(EMBED_DIM):
            if ch in (5, 7):
                # All tokens of selected channels preserved (channel-only mask).
                assert torch.equal(masked[0, :, ch], proj_output_relevance[0, :, ch])
            else:
                assert torch.all(masked[0, :, ch] == 0)

    def test_attribute_shape_sums_tokens(self, proj_output_relevance):
        c = self._make()
        rel = c.attribute(proj_output_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, EMBED_DIM)
        # Spatial aggregation: equals sum over the token (N) axis.
        expected = proj_output_relevance.sum(dim=1)
        assert torch.allclose(rel, expected)

    def test_invalid_channel_raises(self):
        c = self._make()
        with pytest.raises(IndexError):
            c.mask(batch_id=0, concept_ids=[EMBED_DIM], layer_name=self.LAYER)

    def test_layer_suffix(self):
        assert AttnOutputDimConcept.LAYER_SUFFIX == "proj_drop"


# ── Prefix-token isolation in spatial concepts ──────────────────────────────


class TestPrefixIsolation:
    """When num_prefix_tokens > 0, the spatial concepts must:
    1. zero out the prefix tokens in the mask (no relevance flows through them);
    2. exclude prefix tokens from the attribute() spatial sum.
    """

    LAYER_HEAD = "blocks.0.attn.context"
    LAYER_OUT = "blocks.0.attn.proj_drop"

    def test_head_concept_mask_zeros_prefix(self, per_head_relevance):
        c = HeadConcept()
        c.register_layer(self.LAYER_HEAD, NUM_HEADS, HEAD_DIM, num_prefix_tokens=NUM_PREFIX)
        m = c.mask(batch_id=0, concept_ids=[1], layer_name=self.LAYER_HEAD)
        masked = m(per_head_relevance.clone())
        # Selected head, prefix tokens — must be zero.
        assert torch.all(masked[0, 1, :NUM_PREFIX, :] == 0), (
            "HeadConcept did not zero prefix tokens of the selected head"
        )
        # Selected head, spatial tokens — preserved.
        assert torch.equal(
            masked[0, 1, NUM_PREFIX:, :], per_head_relevance[0, 1, NUM_PREFIX:, :],
        )
        # Other heads — entirely zero (incl. spatial).
        for h in (0, 2, 3):
            assert torch.all(masked[0, h] == 0)

    def test_head_concept_attribute_excludes_prefix(self, per_head_relevance):
        c = HeadConcept()
        c.register_layer(self.LAYER_HEAD, NUM_HEADS, HEAD_DIM, num_prefix_tokens=NUM_PREFIX)
        rel = c.attribute(per_head_relevance, layer_name=self.LAYER_HEAD, abs_norm=False)
        # Equals sum over (spatial tokens, head_dim).
        expected = per_head_relevance[:, :, NUM_PREFIX:, :].sum(dim=(2, 3))
        assert torch.allclose(rel, expected)
        # Sanity: differs from the no-prefix version.
        c0 = HeadConcept()
        c0.register_layer(self.LAYER_HEAD, NUM_HEADS, HEAD_DIM)
        rel0 = c0.attribute(per_head_relevance, layer_name=self.LAYER_HEAD, abs_norm=False)
        assert not torch.allclose(rel, rel0)

    def test_attn_output_dim_mask_zeros_prefix(self, proj_output_relevance):
        c = AttnOutputDimConcept()
        c.register_layer(self.LAYER_OUT, EMBED_DIM, num_prefix_tokens=NUM_PREFIX)
        m = c.mask(batch_id=0, concept_ids=[3], layer_name=self.LAYER_OUT)
        masked = m(proj_output_relevance.clone())
        # Channel 3, prefix tokens — must be zero.
        assert torch.all(masked[0, :NUM_PREFIX, 3] == 0)
        # Channel 3, spatial tokens — preserved.
        assert torch.equal(
            masked[0, NUM_PREFIX:, 3], proj_output_relevance[0, NUM_PREFIX:, 3],
        )

    def test_attn_output_dim_attribute_excludes_prefix(self, proj_output_relevance):
        c = AttnOutputDimConcept()
        c.register_layer(self.LAYER_OUT, EMBED_DIM, num_prefix_tokens=NUM_PREFIX)
        rel = c.attribute(proj_output_relevance, layer_name=self.LAYER_OUT, abs_norm=False)
        expected = proj_output_relevance[:, NUM_PREFIX:, :].sum(dim=1)
        assert torch.allclose(rel, expected)


# ── RegisterTokenConcept ─────────────────────────────────────────────────────


class TestRegisterTokenConcept:
    """Per-prefix-token conditioning at proj_drop."""

    LAYER = "blocks.0.attn.proj_drop"

    def _make(self, dim_split=False):
        c = RegisterTokenConcept(dim_split=dim_split)
        c.register_layer(self.LAYER, EMBED_DIM, num_prefix_tokens=NUM_PREFIX)
        return c

    def test_per_token_mask_zeros_other_tokens(self, proj_output_relevance):
        c = self._make()
        m = c.mask(batch_id=0, concept_ids=[1], layer_name=self.LAYER)
        masked = m(proj_output_relevance.clone())
        # Token 1 of prefix — preserved.
        assert torch.equal(masked[0, 1, :], proj_output_relevance[0, 1, :])
        # Other prefix token (0) — zeroed.
        assert torch.all(masked[0, 0, :] == 0)
        # All spatial tokens — zeroed (out of scope for register concept).
        assert torch.all(masked[0, NUM_PREFIX:, :] == 0)

    def test_per_token_attribute_shape_sums_channels(self, proj_output_relevance):
        c = self._make()
        rel = c.attribute(proj_output_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, NUM_PREFIX)
        expected = proj_output_relevance[:, :NUM_PREFIX, :].sum(dim=-1)
        assert torch.allclose(rel, expected)

    def test_dim_split_mask(self, proj_output_relevance):
        c = self._make(dim_split=True)
        m = c.mask(batch_id=0, concept_ids=[(1, 7)], layer_name=self.LAYER)
        masked = m(proj_output_relevance.clone())
        # Only (token=1, channel=7) preserved.
        assert masked[0, 1, 7].item() == proj_output_relevance[0, 1, 7].item()
        # All other entries zeroed.
        nonzero = (masked[0] != 0).sum().item()
        assert nonzero == 1

    def test_dim_split_attribute_shape(self, proj_output_relevance):
        c = self._make(dim_split=True)
        rel = c.attribute(proj_output_relevance, layer_name=self.LAYER, abs_norm=False)
        assert rel.shape == (B, NUM_PREFIX, EMBED_DIM)

    def test_dim_split_flat_int_id(self, proj_output_relevance):
        c = self._make(dim_split=True)
        # flat = token * EMBED_DIM + channel; (token=1, ch=3) → 1*EMBED_DIM + 3
        flat = 1 * EMBED_DIM + 3
        m = c.mask(batch_id=0, concept_ids=[flat], layer_name=self.LAYER)
        masked = m(proj_output_relevance.clone())
        assert masked[0, 1, 3].item() == proj_output_relevance[0, 1, 3].item()
        assert (masked[0] != 0).sum().item() == 1

    def test_invalid_token_raises(self):
        c = self._make()
        with pytest.raises(IndexError):
            c.mask(batch_id=0, concept_ids=[NUM_PREFIX], layer_name=self.LAYER)

    def test_register_layer_rejects_zero_prefix(self):
        c = RegisterTokenConcept()
        with pytest.raises(ValueError):
            c.register_layer(self.LAYER, EMBED_DIM, num_prefix_tokens=0)

    def test_layer_suffix(self):
        assert RegisterTokenConcept.LAYER_SUFFIX == "proj_drop"
