# `signed_epsilon` — `AttnLRPSignedEpsilonComposite()`

**Source.** Achtibat et al. ICML 2024, Eq. 16 — sign-aware ε in the
identity rule's stabiliser (`input + ε·sign(input)` instead of plain
addition).

**Status:** ❌ no-op on DINOv3 with the current GELU rule.
Numerically *identical* to the corresponding non-signed composite
across all 5 samples and all per-layer trajectory entries.

**Why no effect.**

* zennit's `Stabilizer` (used inside `Epsilon` for Linear) is
  **already sign-aware by default** — does `input + sign(input)·ε`.
  So there is no additional knob to flip on the Linear rule.
* The remedy as wired here only flips the `_IdentityRuleFn`'s
  stabiliser. Post-fix, that rule uses `output/stab(output)` on
  GELU outputs, and GELU outputs are typically `O(1)`, far from ε.
  Sign-aware vs plain ε is invisible at this magnitude.

**Conclusion.** Listed as a remedy from the AttnLRP paper, but in our
configuration it has no measurable effect. Kept as a labelled composite
class for parity with the paper's notation; can be removed if we want
to slim the API surface.

**No notebook in this folder.**
