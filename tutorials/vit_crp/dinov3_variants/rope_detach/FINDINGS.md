# `rope_detach` — `AttnLRPRopeDetachComposite()`

**Source.** RoFormer (Su et al. 2021, arXiv:2104.09864) — RoPE's cos/sin
table is computed from positional ids and has **no learnable
parameters**.

**Status:** ❌ no-op (numerically identical to the corresponding
non-detach composite across all samples).

**Why no effect.** `q_rot = q·cos + rotate_half(q)·sin` has the same
chain-rule derivative `∂q_rot/∂q = cos + rotate_half(·)·sin` whether
`cos`/`sin` are graph leaves or detached graph constants. Detach only
affects whether grad *also* flows back into the cos/sin tensor — but
RoPE's tensor is not a `nn.Parameter`, so that grad goes nowhere.
Strict no-op on q/k gradients.

Listed for parity with paper triage; kept as a labelled composite
class for future use on RoPE-with-learned-position variants if any
appear.

**No notebook in this folder.**
