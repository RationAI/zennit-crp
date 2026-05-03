# `combined_all` — `AttnLRPCombinedComposite(matmul_factor_2=True, signed_epsilon=True, rope_detach=True, layerscale_uniform=True, residual_lrp='ratio')`

**Status:** ✅ produces sane heatmaps on DINOv3 (5/5 classes).
Numerically equivalent to [`working_combo/`](../working_combo/FINDINGS.md)
since `signed_epsilon` and `rope_detach` are no-ops in our config.

| Sample | class | max\|R\| | focus@10% |
|---|---|---:|---:|
| 0 | 0 | 234.0 | 0.841 |
| 1 | 217 | 14.5 | 0.769 |
| 2 | 482 | 162.0 | 0.818 |
| 3 | 491 | 122.6 | 0.751 |
| 4 | 497 | 9.93 | 0.848 |

**Conclusion.** "Kitchen sink" stack works because the *active*
ingredients are `matmul_factor_2 + residual_lrp='ratio' +
layerscale_uniform` — the AttnLRP recipe minus the no-ops. Use the
slimmer [`working_combo/`](../working_combo/FINDINGS.md) composite for
production attribution; this folder demonstrates that adding the no-op
remedies on top doesn't break things.

**No standalone notebook** — see `working_combo/walkthrough.ipynb`.
