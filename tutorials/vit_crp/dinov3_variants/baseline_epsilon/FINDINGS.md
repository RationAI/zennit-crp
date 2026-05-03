# `baseline_epsilon` — `AttnLRPEpsilonComposite()`

**Status:** ❌ all-NaN heatmaps on DINOv3 ViT-L/16 (5/5 classes).

**Why.** The bare composite installs zennit's standard `Epsilon` rule on
`nn.Linear` / `nn.Conv2d` and our LayerNorm-stop-gradient + attention-tap
canonizers, but **does not** install:

* the bilinear matmul rule (`AttnLRPMatmulFactor2Composite`) — bare
  PyTorch matmul has no LRP rule, so the relevance through `Q@Kᵀ` and
  `attn@V` is just unstabilised raw gradient.
* the ratio residual rule (`residual_lrp='ratio'`) — bare residual
  `y = x + branch` has the standard autograd backward `grad_x = grad_y,
  grad_branch = grad_y`, which **doubles** the relevance per residual.
  Over 24 EvaBlocks × 2 residuals each that's `2⁴⁸ ≈ 3·10¹⁴`
  amplification before any other source kicks in.

The bare composite remained the documented baseline because the
*previous* `GTIEpsilon` hook (a buggy custom subclass; see
`experiments/audit_gti_hook.py` for the conservation audit that
exposed the bug) accidentally masked these issues with extra
`* output` and `/ stab(input)` factors that compensated for the
missing rules. Once we replaced `GTIEpsilon` with zennit's correct
`Epsilon`, the underlying gaps surfaced.

**Recommendation.** Use
`AttnLRPMatmulFactor2Composite(residual_lrp='ratio')` as the new
DINOv3 baseline (and, when LayerScale is present, add
`layerscale_uniform`). See [`working_combo/`](../working_combo/FINDINGS.md).

**No notebook in this folder** — would just show all-NaN heatmaps.
