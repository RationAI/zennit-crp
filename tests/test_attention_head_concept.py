import torch
import pytest

from crp.concepts import AttentionHeadConcept, ChannelConcept


def _make_relevance(batch=2, seq_len=197, hidden_dim=768, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq_len, hidden_dim, generator=g)


def test_head_mode_shapes_sum():
    concept = AttentionHeadConcept()
    concept.register_num_heads("attn", 12)

    rel = _make_relevance(batch=2, seq_len=197, hidden_dim=768)

    d, r, rf = concept.reference_sampling(
        rel, layer_name="attn", max_target="sum", abs_norm=True, concept_mode="head")

    # batch=2 stored, num_heads=12
    assert d.shape == (2, 12)
    assert r.shape == (2, 12)
    assert rf.shape == (2, 12)
    # rf indices live in [0, seq_len * head_dim) = [0, 197*64) = [0, 12608)
    assert int(rf.max()) < 197 * 64
    assert int(rf.min()) >= 0


def test_head_mode_shapes_max_matches_gather():
    concept = AttentionHeadConcept()
    concept.register_num_heads("attn", 12)

    rel = _make_relevance(seed=1)

    d_max, r_max, rf_max = concept.reference_sampling(
        rel, layer_name="attn", max_target="max", abs_norm=False, concept_mode="head")

    # rebuild expected r_max manually
    batch, seq_len, hidden_dim = rel.shape
    nh, hd = 12, 64
    rel_h = rel.view(batch, seq_len, nh, hd).permute(0, 2, 1, 3).contiguous().view(batch, nh, seq_len * hd)
    expected_rf = torch.argmax(rel_h, dim=-1)
    expected_r = torch.gather(rel_h, -1, expected_rf.unsqueeze(-1)).squeeze(-1)
    # both sorted along batch dim same way -> compare sorted sets
    assert torch.allclose(r_max.sort(dim=0).values, expected_r.sort(dim=0).values, atol=1e-5)


def test_token_mode_falls_back_to_channel_shape():
    """concept_mode='token' on AttentionHeadConcept -> ChannelConcept shape (batch, seq_len)."""
    concept = AttentionHeadConcept()
    concept.register_num_heads("attn", 12)

    rel = _make_relevance(batch=2, seq_len=197, hidden_dim=768)

    d, r, rf = concept.reference_sampling(
        rel, layer_name="attn", max_target="sum", abs_norm=True, concept_mode="token")

    # ChannelConcept treats dim1 (=seq_len=197) as channel dim
    assert d.shape == (2, 197)
    assert r.shape == (2, 197)
    assert rf.shape == (2, 197)


def test_head_mode_fallback_when_num_heads_unset():
    """No registered num_heads -> warn and fall back to channel-style shape."""
    concept = AttentionHeadConcept()
    rel = _make_relevance(batch=2, seq_len=197, hidden_dim=768)
    with pytest.warns(UserWarning):
        d, r, rf = concept.reference_sampling(
            rel, layer_name="unknown", max_target="sum", abs_norm=True, concept_mode="head")
    assert d.shape == (2, 197)


def test_channel_concept_unchanged():
    concept = ChannelConcept()
    rel = _make_relevance(batch=2, seq_len=197, hidden_dim=768)
    d, r, rf = concept.reference_sampling(rel, max_target="sum", abs_norm=True)
    assert d.shape == (2, 197)
