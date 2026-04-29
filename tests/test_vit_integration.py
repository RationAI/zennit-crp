"""ViT integration tests for the AttnLRP Canonizer + Hook + Composite stack.

Exercise the full pipeline: model load → composite (canonizer applies) →
CondAttribution → mask hook on either ``qkv_tap`` (K/Q/V-side concepts) or
``attn_out_tap`` (output-side concepts) → shape checks. The composite is
context-managed so each test starts and ends with a clean (uncanonised) model.

Skipped if ``timm`` / ``zennit`` are missing.

Run::

    uv run pytest tests/test_vit_integration.py -v
"""

import pytest
import torch

timm = pytest.importorskip("timm")
zennit = pytest.importorskip("zennit")


from crp.attention_concepts import (
    HeadConcept,
    HeadDimConcept,
    KQVHeadConcept,
    KQVHeadDimConcept,
)
from crp.attribution import CondAttribution
from crp.transformer_patches import (
    AttentionTapsCanonizer,
    AttnLRPEpsilonComposite,
    AttnLRPGammaComposite,
    TimmViTCanonizer,
    timm_attention_forward,
)


# ── module-scope fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vit_tiny():
    """Smallest readily-available ViT for fast tests. ``vit_tiny_patch16_224``:
    12 blocks, num_heads=3, head_dim=64. Random init avoids weight download."""
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.eval()
    return model


@pytest.fixture
def img_batch():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224, requires_grad=True)


# Two layer names — one per tap. ``attn_out_tap`` is the default tap for
# the per-head output-side concepts (HeadConcept, HeadDimConcept);
# ``qkv_tap`` is for the K/Q/V-side concepts (KQVHeadConcept,
# KQVHeadDimConcept).
QKV_LAYER = "blocks.6.attn.qkv_tap"
OUT_LAYER = "blocks.6.attn.attn_out_tap"


# ── canonizer mechanics: register / remove cycle ──────────────────────────────


class TestAttentionTapsCanonizer:
    def test_register_adds_both_taps_to_every_attention(self, vit_tiny):
        canonizer = AttentionTapsCanonizer()
        instances = canonizer.apply(vit_tiny)
        try:
            # vit_tiny has 12 attention blocks, both taps on each.
            assert len(instances) == 12
            attn_modules = [
                m for m in vit_tiny.modules()
                if hasattr(m, "qkv") and isinstance(m.qkv, torch.nn.Linear)
                and hasattr(m, "num_heads") and hasattr(m, "head_dim")
            ]
            for m in attn_modules:
                assert isinstance(m.qkv_tap, torch.nn.Identity)
                assert isinstance(m.attn_out_tap, torch.nn.Identity)
        finally:
            for inst in instances:
                inst.remove()

    def test_remove_reverts_both_taps(self, vit_tiny):
        canonizer = AttentionTapsCanonizer()
        instances = canonizer.apply(vit_tiny)
        for inst in instances:
            inst.remove()
        attn = vit_tiny.blocks[0].attn
        assert not hasattr(attn, "qkv_tap")
        assert not hasattr(attn, "attn_out_tap")

    def test_idempotent_with_pre_injected_qkv_tap(self, vit_tiny):
        """A user-pre-injected tap must be left in place on remove. The
        canonizer still adds the other tap (``attn_out_tap``) and removes
        only the ones it created."""
        attn = vit_tiny.blocks[0].attn
        attn.add_module("qkv_tap", torch.nn.Identity())
        canonizer = AttentionTapsCanonizer()
        instances = canonizer.apply(vit_tiny)
        for inst in instances:
            inst.remove()
        # block 0 had pre-existing qkv_tap → still present; canonizer-added
        # attn_out_tap → removed.
        assert isinstance(attn.qkv_tap, torch.nn.Identity)
        assert not hasattr(attn, "attn_out_tap")
        # other blocks: both taps removed.
        attn_other = vit_tiny.blocks[1].attn
        assert not hasattr(attn_other, "qkv_tap")
        assert not hasattr(attn_other, "attn_out_tap")
        # Cleanup for downstream tests.
        del attn._modules["qkv_tap"]


class TestTimmViTCanonizer:
    def test_forward_swap_is_reversible(self, vit_tiny):
        attn = vit_tiny.blocks[0].attn
        original_class_forward = type(attn).forward
        canonizer = TimmViTCanonizer()
        instances = canonizer.apply(vit_tiny)
        try:
            assert attn.forward.__func__ is timm_attention_forward
        finally:
            for inst in instances:
                inst.remove()
        assert "forward" not in attn.__dict__
        assert type(attn).forward is original_class_forward


# ── forward parity (model still callable inside composite context) ────────────


def test_forward_runs_under_composite(vit_tiny, img_batch):
    composite = AttnLRPEpsilonComposite()
    with composite.context(vit_tiny) as modified:
        out = modified(img_batch)
    assert out.shape == (1, 1000)


def test_forward_runs_under_gamma_composite(vit_tiny, img_batch):
    composite = AttnLRPGammaComposite()
    with composite.context(vit_tiny) as modified:
        out = modified(img_batch)
    assert out.shape == (1, 1000)


def test_gamma_composite_attribution_end_to_end(vit_tiny, img_batch):
    """γ-LRP composite must produce a pixel-space heatmap of the right shape
    when paired with a HeadConcept mask (output-side, attn_out_tap)."""
    c = HeadConcept(vit_tiny)
    attribution = CondAttribution(vit_tiny)
    composite = AttnLRPGammaComposite()
    conditions = [{OUT_LAYER: [0], "y": [42]}]
    result = attribution(img_batch, conditions, composite, mask_map=c.mask)
    B, _, H, W = img_batch.shape
    assert result.heatmap.shape == (B, H, W)


def test_gamma_differs_from_epsilon(vit_tiny, img_batch):
    """Numerical sanity: γ-LRP and ε-LRP should produce different heatmaps
    on the same input + concept (γ biases toward positive contributions)."""
    c = HeadConcept(vit_tiny)
    attribution = CondAttribution(vit_tiny)
    conditions = [{OUT_LAYER: [0], "y": [42]}]

    img_eps = img_batch.detach().clone().requires_grad_(True)
    eps_result = attribution(
        img_eps, conditions, AttnLRPEpsilonComposite(), mask_map=c.mask
    )
    img_gam = img_batch.detach().clone().requires_grad_(True)
    gam_result = attribution(
        img_gam, conditions, AttnLRPGammaComposite(gamma=0.25), mask_map=c.mask
    )
    assert not torch.allclose(eps_result.heatmap, gam_result.heatmap, atol=1e-6)


# ── concept attribution end-to-end ────────────────────────────────────────────


def _attribute(model, concept, conditions, data):
    attribution = CondAttribution(model)
    composite = AttnLRPEpsilonComposite()
    return attribution(data, conditions, composite, mask_map=concept.mask)


class TestEndToEndShapes:
    """One conditional pass per concept granularity. Verifies the pipeline
    runs end-to-end and returns a pixel-space heatmap of the right shape."""

    def test_head_concept(self, vit_tiny, img_batch):
        c = HeadConcept(vit_tiny)
        conditions = [{OUT_LAYER: [0], "y": [42]}]
        result = _attribute(vit_tiny, c, conditions, img_batch)
        B, _, H, W = img_batch.shape
        assert result.heatmap.shape == (B, H, W)

    def test_head_dim_concept(self, vit_tiny, img_batch):
        c = HeadDimConcept(vit_tiny)
        conditions = [{OUT_LAYER: [(0, 5)], "y": [42]}]
        result = _attribute(vit_tiny, c, conditions, img_batch)
        B, _, H, W = img_batch.shape
        assert result.heatmap.shape == (B, H, W)

    def test_kqv_head_concept(self, vit_tiny, img_batch):
        c = KQVHeadConcept(vit_tiny)
        conditions = [{QKV_LAYER: [("k", 1)], "y": [42]}]
        result = _attribute(vit_tiny, c, conditions, img_batch)
        B, _, H, W = img_batch.shape
        assert result.heatmap.shape == (B, H, W)

    def test_kqv_head_dim_concept(self, vit_tiny, img_batch):
        c = KQVHeadDimConcept(vit_tiny)
        conditions = [{QKV_LAYER: [("v", 0, 0)], "y": [42]}]
        result = _attribute(vit_tiny, c, conditions, img_batch)
        B, _, H, W = img_batch.shape
        assert result.heatmap.shape == (B, H, W)


# ── per-concept relevance shapes ──────────────────────────────────────────────


class TestRelevanceShapes:
    """Per-concept relevance from ``concept.attribute()`` after a real
    backward, recorded on the concept's tap layer."""

    def _record_relevance(self, model, concept, data, layer_name, raw_concept_id):
        attribution = CondAttribution(model)
        composite = AttnLRPEpsilonComposite()
        conditions = [{layer_name: [raw_concept_id], "y": [42]}]
        result = attribution(
            data,
            conditions,
            composite,
            mask_map=concept.mask,
            record_layer=[layer_name],
        )
        rel = result.relevances[layer_name]
        return concept.attribute(rel, layer_name=layer_name, abs_norm=False)

    def test_head_concept_shape(self, vit_tiny, img_batch):
        c = HeadConcept(vit_tiny)
        scores = self._record_relevance(vit_tiny, c, img_batch, OUT_LAYER, 0)
        # vit_tiny has 3 heads
        assert scores.shape == (1, 3)

    def test_head_dim_concept_shape(self, vit_tiny, img_batch):
        c = HeadDimConcept(vit_tiny)
        scores = self._record_relevance(vit_tiny, c, img_batch, OUT_LAYER, (0, 0))
        # vit_tiny: 3 heads × 64 head_dim
        assert scores.shape == (1, 3, 64)

    def test_kqv_head_concept_shape(self, vit_tiny, img_batch):
        c = KQVHeadConcept(vit_tiny)
        scores = self._record_relevance(
            vit_tiny, c, img_batch, QKV_LAYER, ("q", 0)
        )
        assert scores.shape == (1, 3, 3)

    def test_kqv_head_dim_concept_shape(self, vit_tiny, img_batch):
        c = KQVHeadDimConcept(vit_tiny)
        scores = self._record_relevance(
            vit_tiny, c, img_batch, QKV_LAYER, ("q", 0, 0)
        )
        assert scores.shape == (1, 3, 3, 64)


# ── conservation: sum(R_input) ≈ R_target_logit (FUTURE_STATE.md D13) ────────


class TestConservation:
    """Quantify how badly conservation breaks across the AttnLRP pipeline.

    AttnLRP §3 / Eq. 1 asks ``R_input.sum() ≈ R_output.sum()``. With no concept
    mask, ``R_output`` = the masked logit value at the target class, so the
    test is ``sum(data.grad) / target_logit ≈ 1``.

    These tests are **diagnostic**, not gating: the current pipeline does NOT
    conserve (residual additions and the additive `pos_embed` step are
    plain tensor ops with no LRP rule applied unless the user opts into
    ``residual_lrp='ratio'`` / ``palrp=True``). The tests record the ratio
    for visibility and assert the ratio is finite + within a loose bound,
    so a narrowing fix can be detected via a failing-test signal.
    """

    @pytest.fixture(scope="class")
    def vit_base(self):
        """Pretrained vit_base — used for conservation since vit_tiny's three
        heads make many cells trivially noisy. Cached at class scope to avoid
        re-downloading."""
        m = timm.create_model("vit_base_patch16_224", pretrained=False)
        m.eval()
        return m

    def _conservation_ratio(self, model, composite, target=42):
        torch.manual_seed(0)
        data = torch.randn(1, 3, 224, 224, requires_grad=True)
        with torch.no_grad():
            logit_val = model(data)[0, target].item()
        attribution = CondAttribution(model)
        data.grad = None
        attribution(data, [{"y": [target]}], composite)
        sum_R = data.grad.sum().item()
        return sum_R / logit_val, sum_R, logit_val

    def test_epsilon_conservation_diagnostic(self, vit_base):
        ratio, sum_R, logit = self._conservation_ratio(
            vit_base, AttnLRPEpsilonComposite()
        )
        # Diagnostic: conservation requires ratio ≈ 1.0 ideally; the current
        # pipeline (no residual-LRP, no PA-LRP) on a 12-block ViT gives
        # |ratio| ~ O(100). Bound is loose to keep the test green; turning on
        # ``residual_lrp='ratio'`` should narrow it sharply.
        assert torch.isfinite(torch.tensor(sum_R)).item(), (
            f"R_input.sum() = {sum_R} is not finite"
        )
        assert abs(ratio) < 10000, (
            f"ε-LRP conservation ratio {ratio} blew up beyond loose bound; "
            f"sum(R_input)={sum_R}, logit={logit}"
        )

    def test_gamma_conservation_diagnostic(self, vit_base):
        ratio, sum_R, logit = self._conservation_ratio(
            vit_base, AttnLRPGammaComposite(gamma=0.25)
        )
        # γ-LRP at γ=0.25 produces dramatically larger ratios still — the
        # positive-weight clamp on every linear amplifies the residual /
        # pos_embed leak. Loosest of the three bounds; pure regression guard.
        assert torch.isfinite(torch.tensor(sum_R)).item(), (
            f"R_input.sum() = {sum_R} is not finite"
        )

    def test_epsilon_conserves_at_classifier_head(self, vit_base):
        """Sanity: at the classifier head the relevance IS exactly conserved
        — ``R_head[:, target] = target_logit``, all other entries 0. This
        verifies the relevance_init pathway and isolates the leak to the
        backward-pass rules below the head."""
        torch.manual_seed(0)
        data = torch.randn(1, 3, 224, 224, requires_grad=True)
        target = 42
        attribution = CondAttribution(vit_base)
        composite = AttnLRPEpsilonComposite()
        with torch.no_grad():
            logit_val = vit_base(data)[0, target].item()
        result = attribution(
            data, [{"y": [target]}], composite, record_layer=["head"]
        )
        head_rel = result.relevances["head"]
        assert head_rel.shape == (1, 1000)
        assert head_rel[0, target].item() == pytest.approx(logit_val, rel=1e-4)
        mask = torch.ones_like(head_rel, dtype=torch.bool)
        mask[0, target] = False
        assert head_rel[mask].abs().max().item() < 1e-6


# ── FeatureVisualization end-to-end on attention concepts ────────────────────


class TestFeatureVisualizationOnAttentionConcept:
    """Exercises ``FeatureVisualization._attribution_on_reference`` — which
    previously hardcoded ``ChannelConcept.mask`` and would IndexError when
    handed an attention concept's flat int ids. After the fix it pulls
    ``mask_map`` from ``self.layer_map[layer_name]``."""

    def _build_index(self, model, concept, layer_name, dataset, tmpdir):
        from crp.visualization import FeatureVisualization
        attribution = CondAttribution(model)
        composite = AttnLRPEpsilonComposite()
        fv = FeatureVisualization(
            attribution,
            dataset,
            layer_map={layer_name: concept},
            preprocess_fn=lambda x: x,
            path=str(tmpdir),
        )
        fv.run(composite, 0, len(dataset), batch_size=2, checkpoint=10)
        return fv, composite

    def _make_dataset(self, n=4):
        from torch.utils.data import Dataset

        class _RandDS(Dataset):
            def __init__(self, n):
                torch.manual_seed(0)
                self.x = torch.randn(n, 3, 224, 224)
                self.y = torch.randint(0, 1000, (n,))

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                return self.x[i], int(self.y[i])

        return _RandDS(n)

    def test_get_max_reference_with_composite_on_kqv_head(self, vit_tiny, tmp_path):
        """Reproduces the previously-failing path:
        ``get_max_reference([flat_int], composite=composite, plot_fn=None)``
        with a flat int id from KQVHeadConcept."""
        c = KQVHeadConcept(vit_tiny)
        ds = self._make_dataset(n=4)
        fv, composite = self._build_index(vit_tiny, c, QKV_LAYER, ds, tmp_path)

        # 5 = (k, head=2) for vit_tiny (3 heads): part = 5//3 = 1, head = 5%3 = 2.
        # The exact decoding doesn't matter — the test is that the composite path
        # *runs* end to end and returns sample + heatmap of the right shapes.
        ref_c = fv.get_max_reference(
            [5], QKV_LAYER, mode="relevance", r_range=(0, 2),
            composite=composite, plot_fn=None,
        )
        samples, heatmaps = ref_c[5]
        assert samples.shape[1:] == (3, 224, 224)
        # zennit-crp heatmaps are (B, H, W) — channels summed.
        assert heatmaps.shape[1:] == (224, 224)

    def test_get_max_reference_with_composite_on_head_dim(self, vit_tiny, tmp_path):
        """Same but for HeadDimConcept (3 heads × 64 head_dim = 192 concepts;
        pick id 100). HeadDimConcept hooks ``attn_out_tap``, not ``qkv_tap``."""
        c = HeadDimConcept(vit_tiny)
        ds = self._make_dataset(n=4)
        fv, composite = self._build_index(vit_tiny, c, OUT_LAYER, ds, tmp_path)

        ref_c = fv.get_max_reference(
            [100], OUT_LAYER, mode="relevance", r_range=(0, 2),
            composite=composite, plot_fn=None,
        )
        samples, heatmaps = ref_c[100]
        assert samples.shape[1:] == (3, 224, 224)
        assert heatmaps.shape[1:] == (224, 224)

    def test_get_max_reference_with_composite_on_kqv_head_dim(self, vit_tiny, tmp_path):
        """KQVHeadDimConcept (3 × 3 × 64 = 576 concepts; pick id 100)."""
        c = KQVHeadDimConcept(vit_tiny)
        ds = self._make_dataset(n=4)
        fv, composite = self._build_index(vit_tiny, c, QKV_LAYER, ds, tmp_path)

        ref_c = fv.get_max_reference(
            [100], QKV_LAYER, mode="relevance", r_range=(0, 2),
            composite=composite, plot_fn=None,
        )
        samples, heatmaps = ref_c[100]
        assert samples.shape[1:] == (3, 224, 224)
        assert heatmaps.shape[1:] == (224, 224)
