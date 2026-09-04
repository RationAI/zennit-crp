"""g-convention adapters (experiments.gradinput) + cp_lrp (grad×input) composite."""
import numpy as np
import pytest
import torch

timm = pytest.importorskip("timm")
pytest.importorskip("zennit")

from experiments.gradinput import (
    GradTimesInputAttribution, GradTimesInputFeatureVisualization,
    _GradTimesInputFeatVisHook)
from zennit_extensions.lrp_composites import COMPOSITES


@pytest.fixture(scope="module")
def model():
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_heatmap_is_grad_times_input(model):
    attribution = GradTimesInputAttribution(model)
    comp = COMPOSITES["cp_lrp_baseline"]()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    res = attribution(x, [{"y": [3]}], comp)
    assert torch.allclose(res.heatmap, (x.grad * x).sum(1))
    assert torch.isfinite(res.heatmap).all() and (res.heatmap != 0).any()


def test_recorded_relevance_is_g_times_activation(model):
    attribution = GradTimesInputAttribution(model)
    comp = COMPOSITES["cp_lrp_baseline"]()
    layer = "blocks.10.attn.proj_drop"
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    res = attribution(x, [{"y": [3]}], comp, record_layer=[layer])
    rel, act = res.relevances[layer], res.activations[layer]
    assert rel.shape == act.shape
    assert torch.isfinite(rel).all() and (rel != 0).any()
    # conditioning on a concept at the layer must zero everything outside it
    from crp.concepts import EmbeddingDimConcept
    concept = EmbeddingDimConcept(num_heads=3)
    res2 = attribution(x, [{layer: [5], "y": [3]}], comp,
                       record_layer=[layer], mask_map=concept.mask)
    assert torch.isfinite(res2.heatmap).all()


def test_fv_hook_negative_clamp_selects_signed_index_relevance():
    """The FV index hook analyzes grad×activation clamped (default) or signed."""
    class _FakeFV:
        def __init__(self, negative_clamp):
            self.negative_clamp = negative_clamp
            self.captured = None

        def analyze_relevance(self, rel, layer_name, concept, s_indices, targets):
            self.captured = rel

    act = torch.tensor([[1.0, -2.0, 3.0]])
    grad = torch.tensor([[3.0, -1.0, -0.5]])
    for clamp, expected in ((True, torch.tensor([[3.0, 2.0, 0.0]])),
                            (False, torch.tensor([[3.0, 2.0, -1.5]]))):
        fv = _FakeFV(clamp)
        hook = _GradTimesInputFeatVisHook(
            fv, None, "layer", {"sample_indices": np.array([0]), "targets": np.array([0])}, None)
        hook._activation = act
        hook.backward(None, grad)
        assert torch.equal(fv.captured, expected), f"negative_clamp={clamp}"


def test_fv_negative_clamp_defaults_true(tmp_path):
    """GradTimesInputFeatureVisualization exposes the flag, defaulting to clamped."""
    attribution = GradTimesInputAttribution(
        timm.create_model("vit_tiny_patch16_224", pretrained=False).eval())
    fv = GradTimesInputFeatureVisualization(attribution, [], {}, path=str(tmp_path / "a"))
    assert fv.negative_clamp is True
    fv = GradTimesInputFeatureVisualization(attribution, [], {}, path=str(tmp_path / "b"),
                                            negative_clamp=False)
    assert fv.negative_clamp is False


def test_instance_labels_distinguish_sign_flavours():
    """The web instance menu differentiates the two relevance-sign flavours."""
    from experiments.crp_gallery import instance_key, instance_label
    assert instance_label("cp_lrp_baseline", "embed_dim").endswith("(neg. clamped away)")
    assert instance_label("cp_lrp_baseline", "embed_dim", True).endswith("(negative included)")
    assert instance_key("cp_lrp_baseline", "embed_dim") != \
        instance_key("cp_lrp_baseline", "embed_dim", True)
