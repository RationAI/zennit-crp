"""Conservation + correctness audit of `GradientTimesInputBasicHook` (GTIEpsilon).

The standard LRP-ε rule (Bach et al. 2015; recommended on middle layers
per Montavon et al. 2019, iphome.hhi.de/samek/pdf/MonXAI19.pdf) is::

    R_i = sum_j (x_i · w_ij / (y_j + ε·sign(y_j))) · R_j
        = x_i · (W^T @ (R / stab(y)))_i

with the conservation property ``sum(R_in) ≈ sum(R_out)`` (modulo ε).

`GTIEpsilon` (our `GradientTimesInputBasicHook` subclass in
`crp/transformer_patches.py`) implements the same rule via gradient×input
plumbing, but with two extra factors::

    gti_grad     = grad_output * output           # ← "* output" inserted
    grad_outputs = gti_grad / stabilize(output)
    gradients    = autograd.grad(...)             # = W^T @ (R·y / stab(y)) ≈ W^T @ (R·sign(y))
    relevance    = inputs * gradients
    relevance    = relevance / stabilize(input)   # ← "/ stab(input)" inserted

These two extra factors do **not** appear in any LRP rule in the
literature. They make the rule equivalent to the standard ε-LRP only
when ``y/stab(y) ≈ sign(y)`` AND ``x/stab(x) ≈ sign(x)``, i.e. when both
input and output are well above ε. Whenever a Linear has near-ε inputs
(common in DINOv3's deep stack with LayerScale γ ≈ 1e-4), the
``1/stab(input)`` step inflates the relevance by ~1/ε per Linear.

This script verifies the bug numerically:

1. Fix a Linear ``y = Wx`` with synthetic small-magnitude input, compare
   `GTIEpsilon`-returned relevance to standard `zennit.rules.Epsilon` on
   the same input/grad_output.
2. Sum-conservation check: ``sum(R_in) ?= sum(R_out)``.
3. Sweep input magnitude across [1e-9, 1e+0] to see when the rules
   diverge.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Pass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crp.transformer_patches import GTIEpsilon  # noqa: E402


def relevance_through_hook(rule_factory, x: torch.Tensor, w: torch.Tensor,
                            R_out: torch.Tensor) -> torch.Tensor:
    """Run one backward pass through a single nn.Linear with the given rule
    and return the input-side relevance.

    The Linear is wrapped in a one-layer LayerMapComposite so the rule
    actually attaches.
    """
    layer = nn.Linear(w.shape[1], w.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(w)

    composite = LayerMapComposite(layer_map=[(nn.Linear, rule_factory())])
    x_run = x.clone().detach().requires_grad_(True)
    with composite.context(layer) as modified:
        y = modified(x_run)
        # Set the gradient of y w.r.t. itself to R_out and back-prop.
        y.backward(R_out)
    return x_run.grad.detach().clone()


def main():
    torch.manual_seed(0)
    print("=" * 78)
    print("LRP rule conservation audit: GTIEpsilon vs zennit.rules.Epsilon")
    print("=" * 78)

    in_dim, out_dim = 4, 3
    w = torch.randn(out_dim, in_dim)

    print(f"\nFixed test: Linear({in_dim} -> {out_dim}), W given.")
    print(f"\n{'input magnitude':<18} | {'rule':<12} | {'sum R_out':>10} | "
          f"{'sum R_in':>12} | {'max|R_in|':>12} | {'conservation':>14}")
    print("-" * 96)

    for mag in [1.0, 1e-2, 1e-4, 1e-6, 1e-8]:
        x = mag * torch.randn(in_dim).abs()  # positive input, controlled magnitude
        # Compute true output, use it as R_out (so R_out has the right shape).
        with torch.no_grad():
            y_true = torch.nn.functional.linear(x, w)
        R_out = y_true.clone()  # use y itself as relevance (canonical choice)

        for label, factory in [
            ("Epsilon (zennit)", lambda: Epsilon(epsilon=1e-6)),
            ("GTIEpsilon",       lambda: GTIEpsilon(epsilon=1e-6)),
        ]:
            R_in = relevance_through_hook(factory, x, w, R_out)
            cons = (R_in.sum().item() - R_out.sum().item()) / (R_out.sum().item() + 1e-30)
            print(f"|x|≈{mag:.0e}{'':<7} | {label:<12} | "
                  f"{R_out.sum().item():>10.4f} | {R_in.sum().item():>12.4e} | "
                  f"{R_in.abs().max().item():>12.4e} | {cons:>14.4e}")
        print()

    print("Conservation = (sum(R_in) - sum(R_out)) / sum(R_out). "
          "Standard LRP-ε should be ~0 (i.e., relevance is conserved across the layer). "
          "Big numbers = the rule violates conservation.")
    print()

    # Stress test: can we provoke the GTI inflation in isolation?
    print("=" * 78)
    print("Stress: input with one near-ε component (the explosion scenario)")
    print("=" * 78)
    for tiny in [1e-3, 1e-5, 1e-7, 1e-9]:
        x = torch.tensor([1.0, tiny, 0.5, 0.2])  # one tiny component
        with torch.no_grad():
            y_true = torch.nn.functional.linear(x, w)
        R_out = y_true.clone()
        R_zen = relevance_through_hook(lambda: Epsilon(epsilon=1e-6), x, w, R_out)
        R_gti = relevance_through_hook(lambda: GTIEpsilon(epsilon=1e-6), x, w, R_out)
        ratio = R_gti.abs().max().item() / (R_zen.abs().max().item() + 1e-30)
        print(f"  tiny={tiny:.0e}  R_zen.max={R_zen.abs().max().item():.3e}  "
              f"R_gti.max={R_gti.abs().max().item():.3e}  "
              f"ratio={ratio:.2e}  "
              f"R_gti tiny-comp={R_gti[1].item():.3e}")
    print()
    print("'ratio' is GTI's max|R| relative to zennit-Epsilon's. >> 1 means GTI is "
          "inflating relevance on near-ε inputs — the ``/ stab(input)`` step amplifies "
          "by ≈ 1/ε whenever any input component is near ε.")


if __name__ == "__main__":
    main()
