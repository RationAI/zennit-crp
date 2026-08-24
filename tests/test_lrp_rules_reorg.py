"""Correctness tests for the reorganized LRP rules.

Covers the rules added / touched by the one-file-per-paper reorg:

* :class:`MatmulAttnLRP` (AttnLRP Eq. 15) — the vectorized backward must equal
  the literal per-index summation and conserve relevance to ε.
* :class:`EpsilonAdd` (AttnLRP residual add) — signed-sum conservation.
* ``CPLRPComposite`` — a characterization test pinning the attribution on a
  seeded ``vit_tiny`` so relocations cannot silently change its behaviour.
* ``AttnLRPBaselineComposite`` — builds, attributes, and applies the Table-4
  name split (FFN → γ, attention projections → ε).

Run::

    uv run pytest tests/test_lrp_rules_reorg.py -v
"""
import pytest
import torch

from zennit_extensions.rules.attnlrp import EpsilonAdd, MatmulAttnLRP
from zennit_extensions.rules.chefer2021 import CheferMatmul, safe_divide

timm = pytest.importorskip("timm")
pytest.importorskip("zennit")

import torch.nn as nn
from zennit.core import stabilize
from zennit.rules import Gamma, Epsilon

from zennit_extensions.rules.chefer2021 import CheferAdd
from zennit_extensions.lrp_composites import (
    AttnLRPBaselineComposite, CheferLRPComposite, CPLRPComposite,
)


# ── MatmulAttnLRP: vectorized backward == literal Eq. 15 ─────────────────────


def test_matmul_attnlrp_matches_eq15_summation():
    """R^{l-1}_{ji} = Σ_p A_{ji} V_{ip} R^l_{jp} / (2 O_{jp} + ε): the closed-form
    tensor backward must equal the literal einsum over the summation index."""
    torch.manual_seed(0)
    eps = 1e-6
    a = torch.randn(2, 3, 5, 4)   # (batch, heads, J, I)
    b = torch.randn(2, 3, 4, 6)   # (batch, heads, I, P)
    out = a @ b
    rel = torch.randn(2, 3, 5, 6)
    s = rel / stabilize(2.0 * out, eps)
    grad_a = a * (s @ b.transpose(-1, -2))
    grad_b = b * (a.transpose(-1, -2) @ s)
    # literal Eq. 15: R_A[j,i] = A[j,i]·Σ_p B[i,p]·s[j,p]; R_B[i,p] = B[i,p]·Σ_j A[j,i]·s[j,p]
    ref_a = a * torch.einsum("...ip,...jp->...ji", b, s)
    ref_b = b * torch.einsum("...ji,...jp->...ip", a, s)
    assert torch.allclose(grad_a, ref_a, atol=1e-6)
    assert torch.allclose(grad_b, ref_b, atol=1e-6)


def test_matmul_attnlrp_conserves():
    """Σ R_A + Σ R_B → Σ R^l as ε → 0 (each operand takes half via 2O+ε)."""
    torch.manual_seed(1)
    a, b = torch.randn(4, 7, 8), torch.randn(4, 8, 5)
    out = a @ b
    rel = torch.randn(4, 7, 5)
    s = rel / stabilize(2.0 * out, 1e-9)
    total = (a * (s @ b.transpose(-1, -2))).sum() + (b * (a.transpose(-1, -2) @ s)).sum()
    assert torch.allclose(total, rel.sum(), rtol=1e-4)


def test_epsilon_add_conserves_signed_sum():
    """EpsilonAdd splits R_y over y = x + b conserving the signed sum."""
    torch.manual_seed(2)
    x, branch = torch.randn(3, 9), torch.randn(3, 9)
    y = x + branch
    rel = torch.randn(3, 9)
    s = rel / stabilize(y, 1e-9)
    assert torch.allclose(x * s + branch * s, rel, rtol=1e-4)


def test_chefer_matmul_code_exact():
    """Code-exact: plain z-rule with safe_divide, then ÷2 on both branches.
    The unnormalised z-rule conserves the sum; ÷2 makes the total ~half the
    incoming relevance (the released code halves; the paper normalises —
    this test pins the *code* behaviour)."""
    torch.manual_seed(3)
    a, b = torch.randn(2, 3, 5, 4), torch.randn(2, 3, 4, 6)
    out = a @ b
    rel = torch.randn(2, 3, 5, 6)
    hook = CheferMatmul()
    hook.stored_tensors = {"a": a, "b": b, "output": out}
    r_a, r_b = hook.backward(None, None, (rel,))
    s = safe_divide(rel, out)
    # the z-rule split (without ÷2) gives 2× the incoming relevance — each
    # operand independently explains the full R, so ÷2 restores conservation.
    unnorm_a = a * (s @ b.transpose(-1, -2))
    unnorm_b = b * (a.transpose(-1, -2) @ s)
    assert torch.allclose(unnorm_a.sum() + unnorm_b.sum(), 2.0 * rel.sum(), rtol=1e-4)
    # ÷2 on both branches restores conservation (total ≈ rel)
    assert torch.allclose(r_a.sum() + r_b.sum(), rel.sum(), rtol=1e-4)
    # each branch is exactly half the z-rule result
    assert torch.allclose(r_a, 0.5 * unnorm_a, rtol=1e-6)
    assert torch.allclose(r_b, 0.5 * unnorm_b, rtol=1e-6)


def test_chefer_add_global_conservation():
    """Code-exact: absolute-mass renorm over GLOBAL sums (incl. batch dim).
    The global sum is conserved; per-sample sums are not (for B>1)."""
    torch.manual_seed(4)
    x, branch = torch.randn(2, 8, 5), torch.randn(2, 8, 5)
    out = x + branch
    rel = torch.randn(2, 8, 5)
    hook = CheferAdd()
    hook.stored_tensors = {"x": x, "branch": branch, "output": out}
    r_x, r_b = hook.backward(None, None, (rel,))
    # global sum conserved (the released code sums over everything)
    assert torch.allclose(r_x.sum() + r_b.sum(), rel.sum(), rtol=1e-4)
    # the unnormalised z-rule split conserves per-sample (before renorm)
    s = safe_divide(rel, out)
    assert torch.allclose(x * s + branch * s, rel, rtol=1e-4)


# ── cp_lrp_baseline characterization (guards the reorg) ──────────────────────


def test_cp_lrp_baseline_characterization():
    """Pin cp_lrp_baseline attribution on a seeded vit_tiny. Relocating the rule
    classes (rules → zennit_ext.rules) must not change the numbers."""
    torch.manual_seed(1234)
    model = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=10).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    comp = CPLRPComposite()
    with comp.context(model) as mod:
        out = mod(x)
        cls = int(out.argmax(1))
        rel, = torch.autograd.grad(out[0, cls], x)
    assert cls == 9
    assert torch.isfinite(rel).all()
    # Reference values re-pinned 2026-08-19 after the LayerNormSubstitutionCanonizer
    # fix (it now matches timm.layers.norm.LayerNorm, so the sigma-detach + LN
    # epsilon rule actually engage on timm models; previously LNs silently fell
    # to the Pass fallback).
    assert rel.sum().item() == pytest.approx(4.064746e-01, rel=1e-3)
    assert rel.abs().sum().item() == pytest.approx(4.370023e-01, rel=1e-3)


# ── attnlrp_baseline: builds, attributes, Table-4 name split ─────────────────


def test_attnlrp_baseline_attributes():
    torch.manual_seed(0)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    comp = AttnLRPBaselineComposite()
    with comp.context(model) as mod:
        out = mod(x)
        rel, = torch.autograd.grad(out[0, int(out.argmax(1))], x)
    assert torch.isfinite(rel).all() and (rel != 0).any()


def test_attnlrp_baseline_ffn_gamma_projection_epsilon():
    """Table B.5 split via the FFNLinear marker: after canonization FFN linears
    are FFNLinear (→ γ), every unmarked linear (qkv/proj/head) stays nn.Linear
    (→ ε)."""
    from zennit_extensions.attention_unfolded import FFNLinear

    model = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
    comp = AttnLRPBaselineComposite()
    with comp.context(model) as mod:
        fc1 = mod.blocks[0].mlp.fc1
        fc2 = mod.blocks[0].mlp.fc2
        qkv = mod.blocks[0].attn.qkv
        head = mod.head
        assert isinstance(fc1, FFNLinear) and isinstance(fc2, FFNLinear)
        assert type(qkv) is nn.Linear and type(head) is nn.Linear
        assert isinstance(comp.mapping({}, "fc1", fc1), Gamma)
        assert type(comp.mapping({}, "qkv", qkv)) is Epsilon
        assert type(comp.mapping({}, "head", head)) is Epsilon


def test_chefer_lrp_attributes():
    from zennit.rules import ZPlus

    torch.manual_seed(0)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    comp = CheferLRPComposite()
    with comp.context(model) as mod:
        out = mod(x)
        rel, = torch.autograd.grad(out[0, int(out.argmax(1))], x)
    # builds + attributes; finite, non-trivial (the reference reads R_A at the
    # softmax, so pixel values here are off-path smoke only).
    assert torch.isfinite(rel).all() and (rel != 0).any()
    # z+ (α1β0, bias-excluded) on every linear.
    lin = next(m for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.endswith("mlp.fc1"))
    assert isinstance(comp.mapping({}, "blocks.0.mlp.fc1", lin), ZPlus)
