# `layerscale_uniform` — `AttnLRPLayerScaleUniformComposite()`

**Source.** AttnLRP §3 / Eq. 7 uniform allocation rule applied to the
LayerScale γ multiplication (Touvron et al. CaiT, 2021,
arXiv:2103.17239).

**Implementation.** `EvaBlockResidualCanonizer(layerscale_uniform=True)`
wraps `γ * branch` in `divide_gradient(., 2)` so γ (a learned scalar
with no upstream input) absorbs half the relevance under the uniform
rule.

**Status (alone):** ⚠️ produces a finite heatmap on every sample, but
at degenerate magnitude (`max|R| ≈ 5e+13` to `1e+16`). Better than
NaN but still 13–16 OOM too large. **Not usable on its own.**

**Status (combined with `matmul_factor_2` + `residual_lrp='ratio'`):**
✅ produces a usable heatmap (`max|R| ≈ 10–230`, focus 0.75–0.85).
This is the working DINOv3 recipe — see
[`working_combo/`](../working_combo/FINDINGS.md).

## What this rule actually does

Per backward step through one EvaBlock without this rule, LayerScale γ
contributes `* γ` to the backward (a *deflation* by γ ≈ 1e-4 per
LayerScale, ÷10⁻⁸ per block). With this rule, γ instead contributes
`* γ / 2`, so each block deflates by an additional ÷4 — a ~10× boost
to the deflation-vs-amplification balance per block. That's enough to
keep the magnitudes bounded under fp32 once the bilinear matmul rule
is also installed.

**Important:** this rule is most accurately viewed as **the AttnLRP
treatment of LayerScale's multiplication node**, not a "remedy." It
should always be on for any model with LayerScale (CaiT, EvaBlocks,
DINOv3, etc.).

**No standalone notebook** — see `working_combo/walkthrough.ipynb` for
the full demo.
