"""Tests for the grad×input adapter base class (rules/attnlrp.py):
``GradTimesInputMultiInputBasicHook`` and its single-inheritance rules
``EpsilonAddGradTimesInput`` / ``MatmulAttnLRPGradTimesInput``.

Run::

    uv run pytest tests/test_gradinput_adapter.py -v
"""
import pytest
import torch
import torch.nn as nn

pytest.importorskip("zennit")

from zennit.composites import LayerMapComposite
from zennit.core import stabilize

from zennit_extensions.attention_unfolded import BilinearMatmul, ResidualAdd
from zennit_extensions.rules.attnlrp import (
    EpsilonAddGradTimesInput, MatmulAttnLRPGradTimesInput,
)


class _AddNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.add = ResidualAdd()

    def forward(self, x, b):
        return self.add(x, b)


class _MatmulNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mm = BilinearMatmul()

    def forward(self, a, b):
        return self.mm(a, b)


def test_sandwiched_epsilon_add_equals_autograd_g():
    """In g-space the ε-split IS plain autograd through the add: the sandwich
    around the ε parameterisation must reproduce the unhooked gradient g on both
    operands (up to the ε stabilisers). This identity is why LXT needs no
    multi-input adapter at all."""
    torch.manual_seed(0)
    x = torch.randn(2, 7, requires_grad=True)
    b = torch.randn(2, 7, requires_grad=True)
    g_out = torch.randn(2, 7)

    net = _AddNet()
    comp = LayerMapComposite(layer_map=[(ResidualAdd, EpsilonAddGradTimesInput(epsilon=1e-9))])
    with comp.context(net):
        y = net(x, b)
        y.backward(g_out)
    # unhooked autograd in g-space: g passes to both operands unchanged
    assert torch.allclose(x.grad, g_out, rtol=1e-4, atol=1e-6)
    assert torch.allclose(b.grad, g_out, rtol=1e-4, atol=1e-6)


def test_sandwiched_matmul_matches_manual_g_reference():
    """MatmulAttnLRPGradTimesInput against the hand-computed g-space
    reference: R_out = g⊙y; Eq. 15 relevances per operand; g_i = R_i/x_i."""
    torch.manual_seed(1)
    a = torch.randn(3, 4, 5, requires_grad=True)
    b = torch.randn(3, 5, 6, requires_grad=True)
    g_out = torch.randn(3, 4, 6)
    eps = 1e-6

    net = _MatmulNet()
    comp = LayerMapComposite(layer_map=[(BilinearMatmul, MatmulAttnLRPGradTimesInput(epsilon=eps))])
    with comp.context(net):
        y = net(a, b)
        y.backward(g_out)

    with torch.no_grad():
        out = a.detach() @ b.detach()
        rel_out = g_out * out
        s = rel_out / stabilize(2.0 * out, eps)
        rel_a = a.detach() * (s @ b.detach().transpose(-1, -2))
        rel_b = b.detach() * (a.detach().transpose(-1, -2) @ s)
        ref_ga = rel_a / stabilize(a.detach(), epsilon=1e-10)
        ref_gb = rel_b / stabilize(b.detach(), epsilon=1e-10)
    assert torch.allclose(a.grad, ref_ga, rtol=1e-4, atol=1e-6)
    assert torch.allclose(b.grad, ref_gb, rtol=1e-4, atol=1e-6)


def test_positional_routing_same_shape_operands():
    """Square matmul: both operands share a shape, so any shape-matching
    heuristic would mis-route. Verify each operand receives ITS OWN g (they
    must differ for asymmetric a, b)."""
    torch.manual_seed(2)
    a = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(4, 4, requires_grad=True)
    g_out = torch.randn(4, 4)

    net = _MatmulNet()
    comp = LayerMapComposite(layer_map=[(BilinearMatmul, MatmulAttnLRPGradTimesInput(epsilon=1e-6))])
    with comp.context(net):
        y = net(a, b)
        y.backward(g_out)
    assert a.grad is not None and b.grad is not None
    assert not torch.allclose(a.grad, b.grad)


def test_non_differentiable_slot_gets_none():
    """A detached operand must yield no relevance path (None slot) without
    breaking the differentiable one."""
    torch.manual_seed(3)
    x = torch.randn(2, 7, requires_grad=True)
    b = torch.randn(2, 7)                      # no grad — constant operand
    g_out = torch.randn(2, 7)

    net = _AddNet()
    comp = LayerMapComposite(layer_map=[(ResidualAdd, EpsilonAddGradTimesInput(epsilon=1e-9))])
    with comp.context(net):
        y = net(x, b)
        y.backward(g_out)
    assert torch.allclose(x.grad, g_out, rtol=1e-4, atol=1e-6)
    assert b.grad is None
