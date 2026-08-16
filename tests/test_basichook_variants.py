"""Numerical parity of the BasicHook-based AttnLRP rule variants against the
raw-Hook originals in ``zennit_extensions/rules/attnlrp.py``.

Temporary by design: once one of the two implementations is chosen, this file
goes away together with the discarded set (see the note in ``attnlrp.py``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from zennit_extensions.attention_unfolded import (
    BilinearMatmul,
    ResidualAdd,
    SoftmaxAlongLastDim,
)
from zennit_extensions.rules.attnlrp import (
    EpsilonAdd,
    EpsilonAddBasicHook,
    MatmulAttnLRP,
    MatmulAttnLRPBasicHook,
    SoftmaxAttnLRP,
    SoftmaxAttnLRPBasicHook,
)

torch.manual_seed(0)


def _hooked_relevances(module, hook, inputs, out_relevance):
    """Forward `inputs` through `module` with `hook` registered, seed the
    backward with `out_relevance`, and return the per-input relevances."""
    inputs = [t.clone().detach().requires_grad_() for t in inputs]
    handles = hook.register(module)
    try:
        output = module(*inputs)
        output.backward(gradient=out_relevance)
    finally:
        handles.remove()
    return [t.grad for t in inputs]


def _assert_all_close(actual, expected, rtol, atol):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got is not None and want is not None
        assert torch.allclose(got, want, rtol=rtol, atol=atol), (
            f"max abs diff {(got - want).abs().max().item():.3e}"
        )


MATMUL_SHAPES = [
    # (a_shape, b_shape): attention-typical rectangular q@kᵀ and attn@v ...
    ((2, 3, 5, 8), (2, 3, 8, 5)),
    ((2, 3, 5, 5), (2, 3, 5, 8)),
    # ... and square, where stock BasicHook's shape-matching return would
    # silently hand operand-a relevance to operand b.
    ((2, 4, 4), (2, 4, 4)),
]


class TestMatmulParity:
    @pytest.mark.parametrize("a_shape,b_shape", MATMUL_SHAPES)
    def test_matches_original(self, a_shape, b_shape):
        a = torch.randn(a_shape, dtype=torch.float64)
        b = torch.randn(b_shape, dtype=torch.float64)
        rel = torch.randn(a_shape[:-1] + b_shape[-1:], dtype=torch.float64)
        original = _hooked_relevances(BilinearMatmul(), MatmulAttnLRP(), [a, b], rel)
        variant = _hooked_relevances(BilinearMatmul(), MatmulAttnLRPBasicHook(), [a, b], rel)
        _assert_all_close(variant, original, rtol=1e-12, atol=1e-12)

    def test_square_matmul_routes_per_operand(self):
        # both operands share a shape; make sure the variant does not hand the
        # same relevance tensor to both slots
        a = torch.randn(2, 4, 4, dtype=torch.float64)
        b = torch.randn(2, 4, 4, dtype=torch.float64)
        rel = torch.randn(2, 4, 4, dtype=torch.float64)
        rel_a, rel_b = _hooked_relevances(
            BilinearMatmul(), MatmulAttnLRPBasicHook(), [a, b], rel
        )
        assert not torch.allclose(rel_a, rel_b)


class TestEpsilonAddParity:
    def test_matches_original(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        branch = torch.randn(2, 5, 8, dtype=torch.float64)
        rel = torch.randn(2, 5, 8, dtype=torch.float64)
        original = _hooked_relevances(ResidualAdd(), EpsilonAdd(), [x, branch], rel)
        variant = _hooked_relevances(ResidualAdd(), EpsilonAddBasicHook(), [x, branch], rel)
        _assert_all_close(variant, original, rtol=1e-12, atol=1e-12)

    def test_conserves_signed_sum(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        branch = torch.randn(2, 5, 8, dtype=torch.float64)
        rel = torch.randn(2, 5, 8, dtype=torch.float64)
        rel_x, rel_branch = _hooked_relevances(
            ResidualAdd(), EpsilonAddBasicHook(), [x, branch], rel
        )
        assert torch.allclose(rel_x + rel_branch, rel, rtol=1e-4, atol=1e-6)


class TestSoftmaxParity:
    @pytest.mark.parametrize("scale", [1.0, 5.0])
    def test_matches_original(self, scale):
        x = scale * torch.randn(2, 3, 5, 5, dtype=torch.float64)
        rel = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        original = _hooked_relevances(SoftmaxAlongLastDim(), SoftmaxAttnLRP(), [x], rel)
        variant = _hooked_relevances(
            SoftmaxAlongLastDim(), SoftmaxAttnLRPBasicHook(), [x], rel
        )
        _assert_all_close(variant, original, rtol=1e-10, atol=1e-12)


class TestCopySemantics:
    """Composites register hooks via ``template.copy()`` — the inherited
    BasicHook.copy must preserve the closure-captured hyperparameters."""

    @pytest.mark.parametrize(
        "module_factory,hook,n_inputs",
        [
            (BilinearMatmul, MatmulAttnLRPBasicHook(epsilon=1e-3), 2),
            (ResidualAdd, EpsilonAddBasicHook(epsilon=1e-3), 2),
            (SoftmaxAlongLastDim, SoftmaxAttnLRPBasicHook(), 1),
        ],
    )
    def test_copy_matches_template(self, module_factory, hook, n_inputs):
        inputs = [torch.randn(2, 4, 4, dtype=torch.float64) for _ in range(n_inputs)]
        rel = torch.randn(2, 4, 4, dtype=torch.float64)
        direct = _hooked_relevances(module_factory(), hook, inputs, rel)
        copied = _hooked_relevances(module_factory(), hook.copy(), inputs, rel)
        _assert_all_close(copied, direct, rtol=0.0, atol=0.0)


class _ToyAttentionCore(nn.Module):
    """softmax(q @ kᵀ) @ v from the unfolded atomic modules."""

    def __init__(self):
        super().__init__()
        self.qk = BilinearMatmul()
        self.softmax = SoftmaxAlongLastDim()
        self.av = BilinearMatmul()

    def forward(self, q, k, v):
        return self.av(self.softmax(self.qk(q, k.transpose(-1, -2))), v)


class TestChainParity:
    def test_end_to_end_relevance_matches(self):
        q = torch.randn(2, 3, 5, 8, dtype=torch.float64)
        k = torch.randn(2, 3, 5, 8, dtype=torch.float64)
        v = torch.randn(2, 3, 5, 8, dtype=torch.float64)
        rel = torch.randn(2, 3, 5, 8, dtype=torch.float64)

        def run(matmul_hook_cls, softmax_hook_cls):
            core = _ToyAttentionCore()
            handles = [
                matmul_hook_cls().register(core.qk),
                softmax_hook_cls().register(core.softmax),
                matmul_hook_cls().register(core.av),
            ]
            inputs = [t.clone().detach().requires_grad_() for t in (q, k, v)]
            try:
                core(*inputs).backward(gradient=rel)
            finally:
                for handle in handles:
                    handle.remove()
            return [t.grad for t in inputs]

        original = run(MatmulAttnLRP, SoftmaxAttnLRP)
        variant = run(MatmulAttnLRPBasicHook, SoftmaxAttnLRPBasicHook)
        _assert_all_close(variant, original, rtol=1e-9, atol=1e-11)
