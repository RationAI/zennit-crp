# `linear_gamma_005` — `AttnLRPLinearGammaComposite(gamma=0.05)`

**Source.** Achtibat et al. ICML 2024 §3.2.1 — γ-LRP on `nn.Linear`,
`W' = W + γ·max(W, 0)`. Now uses zennit's stock
:class:`zennit.rules.Gamma` (the previous custom `GTIGamma` was a
buggy GTI subclass, removed during the rule audit).

**Status (alone):** ⚠️ finite (5/5 samples) but loose: `max|R| ≈ 1e+5`
to `1e+6`, focus ~0.5–0.6. Same shape of failure as
`baseline_gamma`, just less catastrophic — γ=0.05 is closer to ε-LRP
than the paper's γ=0.25, so the depth-driven amplification compounds
less aggressively.

**Status (with `layerscale_uniform`):** over-deflates
(`max|R| ≈ 1e-5`), focus 0.45–0.6. The combination of γ-on-linears +
uniform-on-LayerScale doubles up on the deflation (γ-LRP itself
already biases toward positive contributions); the result is a
heatmap with very small magnitudes that loses focus.

**Status (with `matmul + ratio + layerscale`):** would interact with
`combined_all` (where γ replaces ε on linears); not directly tested
in the sweep. Based on the per-sample stats from the standalone
γ-LRP runs, expect intermediate magnitudes between
`combined_all`'s O(100) and `layerscale+linear_gamma`'s 1e-5.

**Conclusion.** γ-LRP is a valid rule choice (zennit-stock) but does
not measurably improve over ε-LRP on DINOv3 once the matmul + residual
+ layerscale rules are in place. Keep available; not the default.

**No notebook in this folder.**
