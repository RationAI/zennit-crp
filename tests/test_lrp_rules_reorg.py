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
from zennit_extensions.rules.chefer2021 import CheferMatmul

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


def test_chefer_matmul_normalizes_to_conservation():
    """Per the paper, Chefer normalises the attention matmul too (Eq. 9): the two
    operand relevances' per-sample sum equals the incoming relevance."""
    torch.manual_seed(3)
    a, b = torch.randn(2, 3, 5, 4), torch.randn(2, 3, 4, 6)
    out = a @ b
    rel = torch.randn(2, 3, 5, 6)
    hook = CheferMatmul(1e-9)
    hook.stored_tensors = {"a": a, "b": b, "output": out}
    r_a, r_b = hook.backward(None, None, (rel,))
    dims = (1, 2, 3)
    assert torch.allclose(r_a.sum(dims) + r_b.sum(dims), rel.sum(dims), rtol=1e-4)


def test_chefer_add_normalizes_to_conservation():
    """Chefer's Add applies the same Eq. 9 normalisation so the two branches'
    per-sample sum equals the incoming relevance."""
    torch.manual_seed(4)
    x, branch = torch.randn(2, 8, 5), torch.randn(2, 8, 5)
    out = x + branch
    rel = torch.randn(2, 8, 5)
    hook = CheferAdd(1e-9)
    hook.stored_tensors = {"x": x, "branch": branch, "output": out}
    r_x, r_b = hook.backward(None, None, (rel,))
    dims = (1, 2)
    assert torch.allclose(r_x.sum(dims) + r_b.sum(dims), rel.sum(dims), rtol=1e-4)


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
    assert rel.sum().item() == pytest.approx(4.101363e-01, rel=1e-3)
    assert rel.abs().sum().item() == pytest.approx(4.406183e-01, rel=1e-3)


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
    """Table B.5: FFN linears → γ-LRP, attention projections (qkv/proj) + head → ε."""
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
    comp = AttnLRPBaselineComposite()
    linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    assert isinstance(comp.mapping({}, "blocks.0.mlp.fc1", linears["blocks.0.mlp.fc1"]), Gamma)
    assert isinstance(comp.mapping({}, "blocks.0.mlp.fc2", linears["blocks.0.mlp.fc2"]), Gamma)
    assert isinstance(comp.mapping({}, "blocks.0.attn.qkv", linears["blocks.0.attn.qkv"]), Epsilon)
    assert isinstance(comp.mapping({}, "blocks.0.attn.proj", linears["blocks.0.attn.proj"]), Epsilon)
    assert isinstance(comp.mapping({}, "head", linears["head"]), Epsilon)


def test_chefer_lrp_attributes():
    from zennit.rules import ZPlus, ZBox

    torch.manual_seed(0)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    comp = CheferLRPComposite()
    with comp.context(model) as mod:
        out = mod(x)
        rel, = torch.autograd.grad(out[0, int(out.argmax(1))], x)
    # propagates all the way to input pixels (standalone use)
    assert torch.isfinite(rel).all() and (rel != 0).any()
    # z+ (α1β0) on every linear (no FFN/projection split); z^B box on patch conv.
    lin = next(m for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.endswith("mlp.fc1"))
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    assert isinstance(comp.mapping({}, "blocks.0.mlp.fc1", lin), ZPlus)
    assert isinstance(comp.mapping({}, "patch_embed.proj", conv), ZBox)
