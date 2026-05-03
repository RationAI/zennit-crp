"""Audit `_IdentityRuleFn` against the LRP-0 derivation for elementwise
activations (Bach et al. 2015; recap in Montavon et al. 2019,
iphome.hhi.de/samek/pdf/MonXAI19.pdf).

Standard LRP-0 for ``y = f(x)`` elementwise (treat each element as a
1×1 "linear" with input x_i, weight f(x_i)/x_i, output y_i)::

    R_x_i = x_i · (f(x_i)/x_i) / y_i · R_y_i = R_y_i   (when y_i ≠ 0)

So the **true identity rule** is ``R_x = R_y``. Our implementation
saves ``f(x)/stab(x)`` on forward and returns ``saved · R_y`` on
backward, i.e. ``R_x = R_y · f(x)/stab(x)`` — *not* identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crp.transformer_patches import _IdentityRuleFn  # noqa: E402


def main():
    print("=" * 78)
    print("Identity-rule conservation audit (elementwise GELU)")
    print("=" * 78)
    print()

    gelu = nn.GELU()
    print(f"{'x':>10} | {'GELU(x)':>10} | {'R_y in':>10} | "
          f"{'R_x ours':>14} | {'R_x identity':>14} | {'ratio':>10}")
    print("-" * 90)

    # Sweep x across a range; set R_y = 1 (or some fixed value).
    R_y_value = 1.0
    for x_val in [3.0, 1.0, 0.5, 0.1, 0.01, 1e-4, 1e-7, -0.1, -1.0, -3.0]:
        x = torch.tensor([float(x_val)], requires_grad=True)
        R_y = torch.tensor([R_y_value])

        y = _IdentityRuleFn.apply(gelu.forward, x, 1e-6, False)
        x.grad = None
        y.backward(R_y)
        R_x_ours = x.grad.item()

        # True identity rule: R_x = R_y (when y is non-zero).
        R_x_identity = R_y.item() if abs(y.item()) > 1e-12 else 0.0
        ratio = R_x_ours / R_x_identity if abs(R_x_identity) > 1e-30 else float("nan")

        print(f"{x_val:>10.4f} | {y.item():>10.4f} | {R_y.item():>10.4f} | "
              f"{R_x_ours:>14.4e} | {R_x_identity:>14.4f} | {ratio:>10.4e}")

    print()
    print("'ratio' = R_x_ours / R_x_identity. The TRUE LRP identity rule (and the "
          "AttnLRP §3.4 form) gives R_x = R_y always. Our impl multiplies by "
          "f(x)/stab(x) — heavy dampening on small/negative x. Both are valid LRP "
          "behaviours (ours is closer to LRP-ε on a 1×1 'linear'), but they are "
          "NOT 'identity' and do NOT preserve conservation.")
    print()

    print("=" * 78)
    print("LayerNorm forward audit (stop_gradient(std) variant)")
    print("=" * 78)
    print()
    print("In `layer_norm_forward` we apply `stop_gradient(std)`. This is the")
    print("AttnLRP §3.5 'identity on normalisation' approach — std is treated as")
    print("a constant for the backward, so relevance flows through the affine")
    print("part as if std were fixed. For correctness, the rule on the resulting")
    print("affine `(x-mean) * (1/std) * weight + bias` reduces to standard LRP-ε")
    print("on the affine — as long as the LINEAR rule attached to that module is")
    print("itself a correct LRP-ε rule. With our broken GTIEpsilon, the LayerNorm")
    print("forward itself is fine; the failure is downstream at the next Linear.")


if __name__ == "__main__":
    main()
