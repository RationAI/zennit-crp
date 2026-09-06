"""FFNLinear marker + FFNLinearSubstitutionCanonizer: the paper's Table B.5
FFN-γ / projection-ε Linear split lifted from a name-substring match into the
type system.
"""

from __future__ import annotations

import pytest
import timm
import torch
import torch.nn as nn
from zennit.rules import Epsilon, Gamma

from zennit_extensions.attention_unfolded import FFNLinear
from zennit_extensions.canonisation.canonizers import FFNLinearSubstitutionCanonizer

torch.manual_seed(0)


@pytest.fixture(scope="module")
def vit_tiny():
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).eval()


class TestFFNLinearAlias:
    def test_forward_parity_and_shared_params(self):
        linear = nn.Linear(8, 16).double()
        alias = FFNLinear.from_linear(linear)
        x = torch.randn(4, 8, dtype=torch.float64)
        assert torch.equal(alias(x), linear(x))
        assert alias.weight is linear.weight
        assert alias.bias is linear.bias
        assert isinstance(alias, nn.Linear)


class TestFFNLinearSubstitutionCanonizer:
    def test_swaps_only_mlp_linears_and_restores(self, vit_tiny):
        instances = FFNLinearSubstitutionCanonizer().apply(vit_tiny)
        try:
            ffn, plain = [], []
            for name, mod in vit_tiny.named_modules():
                if isinstance(mod, nn.Linear):
                    (ffn if type(mod) is FFNLinear else plain).append(name)
            # 12 blocks × (fc1, fc2)
            assert len(ffn) == 24
            assert all(".mlp." in name for name in ffn)
            # qkv/proj/head stay plain
            assert all(".mlp." not in name for name in plain)
            assert "head" in plain
        finally:
            for inst in instances:
                inst.remove()
        assert not any(type(m) is FFNLinear for m in vit_tiny.modules())

    def test_forward_parity_under_substitution(self, vit_tiny):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected = vit_tiny(x)
        instances = FFNLinearSubstitutionCanonizer().apply(vit_tiny)
        try:
            with torch.no_grad():
                assert torch.allclose(vit_tiny(x), expected, rtol=1e-6, atol=1e-7)
        finally:
            for inst in instances:
                inst.remove()

    def test_idempotent_on_second_apply(self, vit_tiny):
        first = FFNLinearSubstitutionCanonizer().apply(vit_tiny)
        try:
            second = FFNLinearSubstitutionCanonizer().apply(vit_tiny)
            # exact-type match skips already-substituted FFNLinear modules
            assert second == []
        finally:
            for inst in first:
                inst.remove()
