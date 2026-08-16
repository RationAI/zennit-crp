"""LXT-way LayerNorm (σ-detached + ε) and selectable bias handling
(``bias_mode``: absorb / omit / distribute, AttnLRP Appendix A.2.1) for
``LayerNormEpsilon`` and ``SoftmaxAttnLRP``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from zennit.rules import Epsilon

from zennit_extensions.attention_unfolded import LayerNormDetachedStd, SoftmaxAlongLastDim
from zennit_extensions.canonisation.canonizers import LayerNormSubstitutionCanonizer
from zennit_extensions.rules.attnlrp import LayerNormEpsilon, SoftmaxAttnLRP

torch.manual_seed(0)

DIM = 8
SHAPE = (2, 5, DIM)


def _layernorm_pair(affine=True):
    ln = nn.LayerNorm(DIM, elementwise_affine=affine).double()
    if affine:
        with torch.no_grad():
            ln.weight.copy_(torch.randn(DIM, dtype=torch.float64))
            ln.bias.copy_(torch.randn(DIM, dtype=torch.float64))
    return ln, LayerNormDetachedStd(ln)


def _hooked_relevance(module, hook, x, rel):
    x = x.clone().detach().requires_grad_()
    handles = hook.register(module)
    try:
        module(x).backward(gradient=rel)
    finally:
        handles.remove()
    return x.grad


class TestLayerNormDetachedStd:
    def test_forward_matches_stock_layernorm(self):
        ln, detached = _layernorm_pair()
        x = torch.randn(SHAPE, dtype=torch.float64)
        assert torch.allclose(detached(x), ln(x), rtol=1e-12, atol=1e-12)

    def test_std_detached_from_graph(self):
        _, detached = _layernorm_pair(affine=False)
        x = torch.randn(SHAPE, dtype=torch.float64, requires_grad=True)
        y = detached(x)
        (grad,) = torch.autograd.grad(y.sum(), x)
        # with σ detached the map is linear: gradient of the sum equals the
        # constant row-sums of the linear map, independent of x
        x2 = torch.randn(SHAPE, dtype=torch.float64, requires_grad=True)
        (grad2,) = torch.autograd.grad(detached(x2).sum(), x2)
        assert torch.allclose(grad, grad2, rtol=1e-10, atol=1e-12)


class TestLayerNormEpsilonBiasModes:
    def test_absorb_matches_stock_zennit_epsilon(self):
        # independent implementation path: zennit's BasicHook Epsilon on the
        # σ-detached module must equal LayerNormEpsilon(bias_mode='absorb')
        _, detached = _layernorm_pair()
        x = torch.randn(SHAPE, dtype=torch.float64)
        rel = torch.randn(SHAPE, dtype=torch.float64)
        ours = _hooked_relevance(detached, LayerNormEpsilon(), x, rel)
        stock = _hooked_relevance(detached, Epsilon(epsilon=1e-6), x, rel)
        assert torch.allclose(ours, stock, rtol=1e-10, atol=1e-12)

    def test_absorb_leaves_exactly_the_bias_share(self):
        _, detached = _layernorm_pair()
        x = torch.randn(SHAPE, dtype=torch.float64)
        rel = torch.randn(SHAPE, dtype=torch.float64)
        relevance = _hooked_relevance(detached, LayerNormEpsilon(), x, rel)
        y = detached(x)
        v = rel / (y + torch.sign(y) * 1e-6)
        bias_share = (detached.bias * v).sum()
        assert torch.allclose(
            relevance.sum() + bias_share, (y * v).sum(), rtol=1e-8, atol=1e-10
        )

    def test_distribute_conserves_up_to_stabilizer(self):
        # exact identity: Σ R_in = Σ rel·y/(y ± ε) — β's share is redistributed
        # to the inputs, only the ε share is absorbed
        from zennit.core import stabilize

        _, detached = _layernorm_pair()
        x = torch.randn(SHAPE, dtype=torch.float64)
        rel = torch.randn(SHAPE, dtype=torch.float64)
        relevance = _hooked_relevance(
            detached, LayerNormEpsilon(bias_mode="distribute"), x, rel
        )
        y = detached(x)
        expected = (rel * y / stabilize(y, 1e-6)).sum()
        assert torch.allclose(relevance.sum(), expected, rtol=1e-10, atol=1e-12)

    def test_omit_conserves_up_to_stabilizer(self):
        # exact identity: Σ R_in = Σ rel·y₀/(y₀ ± ε) with y₀ = y − β; the ε
        # share is all that is absorbed
        from zennit.core import stabilize

        _, detached = _layernorm_pair()
        x = torch.randn(SHAPE, dtype=torch.float64)
        rel = torch.randn(SHAPE, dtype=torch.float64)
        relevance = _hooked_relevance(
            detached, LayerNormEpsilon(bias_mode="omit"), x, rel
        )
        y0 = detached(x) - detached.bias
        expected = (rel * y0 / stabilize(y0, 1e-6)).sum()
        assert torch.allclose(relevance.sum(), expected, rtol=1e-10, atol=1e-12)

    def test_modes_coincide_without_affine(self):
        _, detached = _layernorm_pair(affine=False)
        x = torch.randn(SHAPE, dtype=torch.float64)
        rel = torch.randn(SHAPE, dtype=torch.float64)
        results = [
            _hooked_relevance(detached, LayerNormEpsilon(bias_mode=mode), x, rel)
            for mode in ("absorb", "omit", "distribute")
        ]
        assert torch.allclose(results[0], results[1], rtol=1e-12, atol=1e-12)
        assert torch.allclose(results[0], results[2], rtol=1e-12, atol=1e-12)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            LayerNormEpsilon(bias_mode="banana")


class TestSoftmaxBiasModes:
    def _run(self, bias_mode, x, rel):
        return _hooked_relevance(
            SoftmaxAlongLastDim(), SoftmaxAttnLRP(bias_mode=bias_mode), x, rel
        )

    def test_absorb_is_proposition_31(self):
        x = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        rel = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        s = torch.softmax(x, dim=-1)
        expected = x * (rel - s * rel.sum(dim=-1, keepdim=True))
        assert torch.allclose(self._run("absorb", x, rel), expected, rtol=1e-12, atol=1e-12)

    def test_distribute_conserves_exactly(self):
        x = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        rel = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        relevance = self._run("distribute", x, rel)
        assert torch.allclose(
            relevance.sum(dim=-1), rel.sum(dim=-1), rtol=1e-10, atol=1e-12
        )

    def test_omit_conserves_up_to_stabilizer(self):
        x = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        rel = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        relevance = self._run("omit", x, rel)
        assert torch.allclose(relevance.sum(dim=-1), rel.sum(dim=-1), rtol=1e-3, atol=1e-4)

    def test_modes_differ_from_absorb(self):
        x = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        rel = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        absorb = self._run("absorb", x, rel)
        assert not torch.allclose(absorb, self._run("distribute", x, rel))
        assert not torch.allclose(absorb, self._run("omit", x, rel))


class TestLayerNormSubstitutionCanonizer:
    def _toy_model(self):
        model = nn.Sequential(
            nn.Linear(DIM, DIM).double(),
            nn.LayerNorm(DIM).double(),
            nn.Linear(DIM, DIM).double(),
            nn.LayerNorm(DIM).double(),
        )
        with torch.no_grad():
            for ln in (model[1], model[3]):
                ln.weight.copy_(torch.randn(DIM, dtype=torch.float64))
                ln.bias.copy_(torch.randn(DIM, dtype=torch.float64))
        return model

    def test_substitutes_and_restores(self):
        model = self._toy_model()
        instances = LayerNormSubstitutionCanonizer().apply(model)
        assert len(instances) == 2
        assert isinstance(model[1], LayerNormDetachedStd)
        assert isinstance(model[3], LayerNormDetachedStd)
        for inst in instances:
            inst.remove()
        assert isinstance(model[1], nn.LayerNorm)
        assert isinstance(model[3], nn.LayerNorm)

    def test_forward_parity_under_substitution(self):
        model = self._toy_model()
        x = torch.randn(SHAPE, dtype=torch.float64)
        expected = model(x)
        instances = LayerNormSubstitutionCanonizer().apply(model)
        try:
            assert torch.allclose(model(x), expected, rtol=1e-12, atol=1e-12)
        finally:
            for inst in instances:
                inst.remove()

    def test_does_not_touch_layernorm_subclasses(self):
        class LayerNormSubclass(nn.LayerNorm):
            pass

        model = nn.Sequential(LayerNormSubclass(DIM))
        instances = LayerNormSubstitutionCanonizer().apply(model)
        assert instances == []
        assert isinstance(model[0], LayerNormSubclass)


class TestCompositeSmoke:
    @pytest.mark.parametrize("mode", ["absorb", "omit", "distribute"])
    def test_attnlrp_composite_accepts_bias_modes(self, mode):
        from zennit_extensions.lrp_composites.attnlrp import AttnLRPBaselineComposite
        from zennit_extensions.lrp_composites.cp_lrp import CPLRPComposite

        AttnLRPBaselineComposite(softmax_bias_mode=mode, layernorm_bias_mode=mode)
        CPLRPComposite(layernorm_bias_mode=mode)
