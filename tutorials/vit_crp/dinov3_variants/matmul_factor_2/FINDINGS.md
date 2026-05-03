# `matmul_factor_2` — `AttnLRPMatmulFactor2Composite()`

**Source.** Achtibat et al. ICML 2024 (arXiv:2402.05602), Proposition 3.3
& Eq. 14 — bilinear matmul rule with `2·Y + ε` stabiliser.

**Implementation.** Autograd Function `_MatmulFactor2Fn`
(`crp/transformer_patches.py`) wraps `Q@Kᵀ` and `attn@V`. Backward::

    scaled = R_Y / (2 · Y + ε·sign(Y))
    R_A    = A · (scaled @ B^T)
    R_B    = B · (A^T @ scaled)

Returns relevance in **pure R form** (with operand multiplication
`A ·` / `B ·`) — required because the upstream of these matmul operands
is softmax (Pass) which doesn't perform an operand-multiplication step.

> **Bug fix in this iteration.** The earlier `_MatmulFactor2Fn`
> implementation returned only `scaled @ B^T` etc. without the operand
> factor, and the diagnostic showed catastrophic 10²⁹ inflation on
> vit_tiny. Fixed during the rule-audit pass.

**Status (alone):** ❌ still all-NaN on DINOv3 — the bilinear rule on
its own conserves the matmul but the residual additions (still using
bare autograd) double the relevance per layer, which on a 24-block
stack overflows fp32. Need `residual_lrp='ratio'` alongside.

**Status (combined with `residual_lrp='ratio'`):** ✅ produces a
finite, sensible heatmap on DINOv3 (`max|R| ≈ 1.5e+5`, focus 0.77).
Adding `layerscale_uniform` brings `max|R|` down to **~10–230**
(a usable range) — see [`working_combo/`](../working_combo/FINDINGS.md).

**Notable.** This rule is what **fixes the transformer-specific
bilinears** that bare ε-LRP cannot address. It is NOT optional on a
transformer — it is the rule that AttnLRP introduces specifically for
this purpose. Earlier framing as a "remedy" was misleading; it is part
of the *correct* AttnLRP recipe and should be treated as such.

**No notebook in this folder** — see `working_combo/walkthrough.ipynb`
for the full demo (where this rule is one of the three required
ingredients).
