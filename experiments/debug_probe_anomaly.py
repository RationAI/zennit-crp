"""Reproduce + diagnose the conservation-probe chain inconsistency.

`experiments/dinov3_conservation.py` showed that on DINOv3 with the
working composite:

    norm    | R_out=3.71  R_in=-0.0000   ← per-token sum-zero of LayerNorm
    blocks.23 | R_out=3.71  R_in=1.43    ← but the next module sees 3.71!

If norm's grad_input ≈ 0, blocks.23's grad_output should be ≈ 0 too
(since grad flows from norm.input to blocks[23].output via
``norm.input is blocks[23].output``). The chain ought to be consistent.

This script tests three hypotheses:

1. zennit's :class:`zennit.rules.Pass` rule (mapped to LayerNorm in the
   working composite's layer_map) returns ``grad_output`` as the new
   ``grad_input``. If this override IS happening, my probe should see
   norm.R_in = norm.R_out. If my probe shows R_in=0, the override is
   either not firing or my probe sees the value pre-override.
2. ``register_full_backward_hook`` ordering when both zennit and a
   custom probe are attached: PyTorch docs say "called in registration
   order"; the override return-value semantics may differ from
   intuition when multiple hooks chain.
3. Hook lifetime / context: zennit hooks attach via
   ``composite.context()``; if my probe is attached AFTER the context
   enters, ordering should be correct, but maybe the hook lookup table
   doesn't preserve order.

Each hypothesis tested as an isolated unit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Pass


def section(title: str):
    print("=" * 78)
    print(title)
    print("=" * 78)


def hypothesis_1_pass_rule_override():
    """Does zennit Pass rule actually override grad_input via return grad_output?"""
    section("H1: zennit's Pass rule on LayerNorm — does it override grad_input?")
    torch.manual_seed(0)
    ln = nn.LayerNorm(5)
    x = torch.randn(2, 5, requires_grad=True)
    grad_y = torch.ones(2, 5)

    # Without composite — natural autograd backward
    y = ln(x)
    x.grad = None
    y.backward(grad_y)
    g_natural = x.grad.clone()

    print(f"natural grad_x.sum() per row = {x.grad.sum(dim=-1)}")
    print(f"   expected to be ≈ 0 (LayerNorm's per-token sum-zero property)")
    print()

    # With Pass composite
    composite = LayerMapComposite(layer_map=[(nn.LayerNorm, Pass())])
    x.grad = None
    with composite.context(ln) as modified:
        y = modified(x)
        y.backward(grad_y)
    g_pass = x.grad.clone()

    print(f"with Pass: grad_x.sum() per row = {x.grad.sum(dim=-1)}")
    print(f"   expected to be ≈ sum(grad_y) per row = {grad_y.sum(dim=-1)}")
    print(f"   IF Pass returns grad_output as new grad_input, sum ≈ 5")
    print()

    print(f"natural grad_x:  {g_natural[0]}")
    print(f"Pass    grad_x:  {g_pass[0]}")
    print(f"grad_y:          {grad_y[0]}")
    print()


def hypothesis_2_hook_ordering():
    """Does my probe see post-Pass grad_input or pre-Pass grad_input?"""
    section("H2: hook ordering — my probe sees pre- or post-Pass grad_input?")
    torch.manual_seed(0)
    ln = nn.LayerNorm(5)
    x = torch.randn(2, 5, requires_grad=True)
    grad_y = torch.ones(2, 5)

    captured = {}

    def my_probe(module, grad_input, grad_output):
        # Standard probe — read what the previous hook (zennit Pass) returned
        captured["probe_grad_input_sum"] = grad_input[0].sum().item()
        captured["probe_grad_output_sum"] = grad_output[0].sum().item()

    composite = LayerMapComposite(layer_map=[(nn.LayerNorm, Pass())])

    # Attach probe AFTER the composite context is open
    with composite.context(ln) as modified:
        h = modified.register_full_backward_hook(my_probe)
        try:
            x.grad = None
            y = modified(x)
            y.backward(grad_y)
        finally:
            h.remove()

    print(f"probe sees grad_output.sum() = {captured['probe_grad_output_sum']:.4f}")
    print(f"probe sees grad_input.sum()  = {captured['probe_grad_input_sum']:.4f}")
    print(f"final  x.grad.sum()          = {x.grad.sum().item():.4f}")
    print()
    if abs(captured["probe_grad_input_sum"]) < 1e-3:
        print("→ probe sees grad_input ≈ 0  (LayerNorm's natural sum-zero)")
        print("  This means probe fires BEFORE Pass's override takes effect, OR")
        print("  Pass's override doesn't actually replace grad_input here.")
    else:
        print("→ probe sees grad_input ≈ grad_output  (post-Pass)")
    print()


def hypothesis_3_inverse_ordering():
    """Same as H2 but probe registered BEFORE the composite enters."""
    section("H3: probe registered BEFORE composite — does ordering flip?")
    torch.manual_seed(0)
    ln = nn.LayerNorm(5)
    x = torch.randn(2, 5, requires_grad=True)
    grad_y = torch.ones(2, 5)

    captured = {}

    def my_probe(module, grad_input, grad_output):
        captured["probe_grad_input_sum"] = grad_input[0].sum().item()
        captured["probe_grad_output_sum"] = grad_output[0].sum().item()

    h = ln.register_full_backward_hook(my_probe)
    try:
        composite = LayerMapComposite(layer_map=[(nn.LayerNorm, Pass())])
        with composite.context(ln) as modified:
            x.grad = None
            y = modified(x)
            y.backward(grad_y)
    finally:
        h.remove()

    print(f"probe sees grad_output.sum() = {captured['probe_grad_output_sum']:.4f}")
    print(f"probe sees grad_input.sum()  = {captured['probe_grad_input_sum']:.4f}")
    print(f"final  x.grad.sum()          = {x.grad.sum().item():.4f}")
    print()


def hypothesis_4_chain_inspection():
    """Set up Linear → LayerNorm → Linear; probe both modules; check chain consistency."""
    section("H4: chained modules — does L1.R_out match LN.R_in_post_pass?")
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(5, 8),
        nn.LayerNorm(8),
        nn.Linear(8, 3),
    )
    x = torch.randn(2, 5, requires_grad=True)
    target = torch.zeros(2, 3); target[:, 0] = 1.0

    captured = {}

    def make_probe(name):
        def probe(module, grad_input, grad_output):
            captured[f"{name}_R_out"] = grad_output[0].sum().item() if grad_output[0] is not None else None
            captured[f"{name}_R_in"] = grad_input[0].sum().item() if grad_input[0] is not None else None
        return probe

    composite = LayerMapComposite(layer_map=[
        (nn.Linear, Epsilon(epsilon=1e-6)),
        (nn.LayerNorm, Pass()),
    ])

    with composite.context(model) as modified:
        # Register probes AFTER zennit hooks are attached
        handles = [
            modified[0].register_full_backward_hook(make_probe("L1")),
            modified[1].register_full_backward_hook(make_probe("LN")),
            modified[2].register_full_backward_hook(make_probe("L2")),
        ]
        try:
            x.grad = None
            out = modified(x)
            out.backward(target)
        finally:
            for h in handles:
                h.remove()

    for name in ["L2", "LN", "L1"]:
        ro = captured.get(f"{name}_R_out", None)
        ri = captured.get(f"{name}_R_in", None)
        print(f"{name}:  R_out_sum = {ro:>12.4f}   R_in_sum = {ri:>12.4f}   absorbed = {(ro - ri) if ro is not None and ri is not None else float('nan'):>12.4f}")

    print()
    L2_in = captured.get("L2_R_in")
    LN_out = captured.get("LN_R_out")
    LN_in = captured.get("LN_R_in")
    L1_out = captured.get("L1_R_out")
    print("Chain consistency (forward order is L1 → LN → L2; backward goes L2 → LN → L1):")
    print(f"  L2's grad_input  ≟ LN's grad_output : {L2_in:.4f} vs {LN_out:.4f}  delta = {(L2_in - LN_out):.4e}")
    print(f"  LN's grad_input  ≟ L1's grad_output : {LN_in:.4f} vs {L1_out:.4f}  delta = {(LN_in - L1_out):.4e}")


def hypothesis_5_no_layer_norm_pass():
    """What if LayerNorm has NO rule (not even Pass)? Different from Pass-as-noop?"""
    section("H5: LayerNorm with NO rule at all (autograd backward through LN)")
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(5, 8),
        nn.LayerNorm(8),
        nn.Linear(8, 3),
    )
    x = torch.randn(2, 5, requires_grad=True)
    target = torch.zeros(2, 3); target[:, 0] = 1.0

    captured = {}

    def make_probe(name):
        def probe(module, grad_input, grad_output):
            captured[f"{name}_R_out"] = grad_output[0].sum().item() if grad_output[0] is not None else None
            captured[f"{name}_R_in"] = grad_input[0].sum().item() if grad_input[0] is not None else None
        return probe

    composite = LayerMapComposite(layer_map=[
        (nn.Linear, Epsilon(epsilon=1e-6)),
        # No rule for LayerNorm
    ])

    with composite.context(model) as modified:
        handles = [
            modified[0].register_full_backward_hook(make_probe("L1")),
            modified[1].register_full_backward_hook(make_probe("LN")),
            modified[2].register_full_backward_hook(make_probe("L2")),
        ]
        try:
            x.grad = None
            out = modified(x)
            out.backward(target)
        finally:
            for h in handles:
                h.remove()

    for name in ["L2", "LN", "L1"]:
        ro = captured.get(f"{name}_R_out", None)
        ri = captured.get(f"{name}_R_in", None)
        print(f"{name}:  R_out_sum = {ro:>12.4f}   R_in_sum = {ri:>12.4f}")

    print()
    L2_in = captured.get("L2_R_in")
    LN_out = captured.get("LN_R_out")
    LN_in = captured.get("LN_R_in")
    L1_out = captured.get("L1_R_out")
    print("Chain consistency:")
    print(f"  L2's grad_input  ≟ LN's grad_output : {L2_in:.4f} vs {LN_out:.4f}")
    print(f"  LN's grad_input  ≟ L1's grad_output : {LN_in:.4f} vs {L1_out:.4f}")


if __name__ == "__main__":
    hypothesis_1_pass_rule_override()
    hypothesis_2_hook_ordering()
    hypothesis_3_inverse_ordering()
    hypothesis_4_chain_inspection()
    hypothesis_5_no_layer_norm_pass()
