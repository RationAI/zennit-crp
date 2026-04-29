# Future State — backlog for `transformer-multi-concept`

The implementation in `CURRENT_STATE.md` is complete enough to land as the first review-ready PR. This file enumerates what still needs to be done before the branch is review-ready, ranked by impact and ordered into milestones.

## Milestone A — close the AUC anomaly (investigated; escalated to D)

**Status**: ran the full sweep (iter 7); see `CURRENT_STATE.md` "Milestone A
— faithfulness sweep finding". Acceptance criterion (ordering holds for
**all four** granularities under one γ) is **not met by any composite**.
The strongest cell — `head_dim` under ε-LRP — has del_gap +0.060; the
weakest — `kqv_head` under ε-LRP — has del_gap **−0.009** (random wins),
reproducible regardless of γ. AttnLRP §3.2.1's γ ≈ 0.25 does not
transfer to attention-concept granularities under the union-of-top-k
Petsiuk methodology and consistently makes most cells worse.

Decision: **escalate to Milestone D** (conservation + PA-LRP) before any
further γ tuning. *(Iter-8 follow-up: D ran. PA-LRP halves the heatmap
uniformly so it cannot change rank → AUC; saturation is also disproven by
cross-scale data — vit_large is **worse** than vit_base on kqv_head, not
better. Next high-likelihood fix is residual-LRP, tracked as Milestone G.)*

Done in iter 7:

1. ✅ Sweep on `vit_base_patch16_224`, 64 imgs × 4 classes, γ ∈
   {0.0, 0.1, 0.25, 0.5} + ε-LRP.
2. ❌ Ordering does not hold for all four under any γ — escalated to D.
3. **Default**: `AttnLRPEpsilonComposite` (3/4 OK on the criterion). Keep
   `AttnLRPGammaComposite(gamma=0.25)` available; γ remains the AttnLRP
   §3.2.1 recommendation for pixel-attribution use cases.
4. ✅ Faithfulness table at `tutorials/vit_crp/data/milestone_a_table.md`,
   raw CSV + naïve-top-k=8 archive at `data/milestone_a_results*.csv`.

Open follow-ups within A (gated on Milestone D outcome):

* If D closes the kqv_head gap, re-run the γ sweep and re-pick the default.
* If D doesn't close it, investigate the methodological alternative noted
  in CURRENT_STATE.md: pixel-rank Petsiuk on the single top-1 conditional
  heatmap rather than union-of-top-k mask.

## Milestone B — additional faithfulness metrics + baselines

A richer benchmark beyond the current deletion / insertion AUC + random-concept baseline.

5. **Stability metric** — cosine similarity of pixel-space relevance maps under input Gaussian noise (σ ∈ {0.01, 0.05, 0.1}). One scalar per (granularity, σ); standard deviation across images. ~50 LOC, no new dataset.
6. **Gradient-only baseline** — plain input × gradient, no LRP. ~30 LOC.
7. **Grad-CAM at attention output** — adapt `pytorch-grad-cam` or implement directly. ~80 LOC.
8. **Occlusion (sliding window)** — slide a 16×16 zero patch, record target-class drop, normalise to a heatmap. Slow but a recognised baseline. ~60 LOC.
9. **CSV schema**: one row per `(model, image, granularity, composite, metric, baseline)` so it's pivotable. Plus a summary table for the PR description with mean ± std.

## Milestone C — localisation / pointing-game

10. **ImageNet-S bounding-box dataset** — pick a subset (the full thing is ~3 GB; a 50-image curated subset under `tutorials/vit_crp/data/imagenet_s_subset/` is enough for a sanity table).
11. **Pointing-game implementation** — heatmap argmax inside ground-truth bbox → 1, else 0. Aggregate as accuracy per granularity.
12. **Optional**: weighted variant (sum of relevance inside bbox / total relevance).

This is the heaviest milestone (~few hundred LOC + a non-trivial dataset call), and is somewhat independent of milestones A and B. Defer if Milestone A turns up surprises.

## Milestone D — conservation + PA-LRP (closed; PA-LRP shipped, AUC anomaly unaffected)

Done in iter 8 (`run_milestone_d.py`, 3072-row CSV, summary in
`CURRENT_STATE.md` "Milestone D — conservation + PA-LRP (iter 8)"):

13. ✅ Conservation diagnostic in `tests/test_vit_integration.py`. ε ratio
    on real `vit_base` + Imagenette image is **−223** (no PA-LRP), **−112**
    (with PA-LRP). Far from 1.0 — the leak is dominated by un-hooked
    residual additions in each block, not by `pos_embed`.
14. ✅ `PALRPCanonizer` shipped — `vit_pos_embed_palrp` forward swap,
    `palrp: bool` kwarg on `TimmViTCanonizer` and both composites. Defaults
    off (rationale below).
15. ✅ Empirically **PA-LRP halves the heatmap by an exact factor of 2**
    (Pearson 1.0000 with the no-PA-LRP heatmap, `argsort` identical) →
    Petsiuk AUC unchanged at every (model, granularity) row of the
    multi-model sweep. PA-LRP is a conservation-magnitude fix, not an AUC
    fix; left opt-in to avoid a silent ½× rescale of users' heatmaps.

The `kqv_head` AUC anomaly that motivated escalation A → D is **not**
fixed by PA-LRP, because PA-LRP cannot change ranking. The cross-scale
sweep also rules out the saturation hypothesis: vit_large (top-k coverage
17 %) has the **worst** kqv_head failure (`del_gap` = −0.0165), vit_small
(coverage 44 %) the **mildest** (−0.0011). Scale amplifies the AUC
inversion rather than diluting it, pointing at accumulated rule error
along the per-block backward chain.

## Milestone G — residual-LRP (highest-likelihood fix for the AUC anomaly)

Each transformer block does `x = x + attn(x)` and `x = x + mlp(x)` as
plain tensor `+`. zennit attaches no hook to a tensor op, so relevance
flowing back receives the full upstream relevance on **both** branches
(double-allocation; not conservative; ~2× per block). Conservation
ratio on a 12-block ViT drifts ~10²× and on a 24-block ViT ~10⁴×, which
matches the 24-block model's amplified AUC inversion.

PA-LRP at `pos_embed` doesn't address this — the residual additions are
inside each block, repeated 24 times for vit_large.

Sketch:

23. Add `_BlockResidualFn(Function)` — identity in forward, `divide_gradient`
    by **2** on the additive output (uniform rule allocates relevance
    equally between the identity branch `x` and the residual branch
    `f(x)`). One per `Block.forward`'s two adds.
24. `vit_block_forward(self, x)` — replicates timm `Block.forward` with
    each `x = x + branch(x)` rewritten as
    `x = divide_gradient(x + branch(x), 2)`. Versioned per timm-`Block`
    layout (`init_values`, `parallel`, `Mlp`/`SwiGLU` variants).
25. `BlockResidualCanonizer(AttributeCanonizer)` — swaps `forward` per
    `Block` instance.
26. Compose into `TimmViTCanonizer` behind a `residual_lrp: bool` kwarg
    (or default-on if conservation tests confirm it's the right fix).
27. Re-run `run_milestone_d.py` with the new canonizer enabled. Acceptance:
    kqv_head and head_dim del/ins gaps positive across all three model
    sizes.

Risk: residual-LRP can over-correct and flatten the heatmap. If it
collapses the discriminative signal, fall back to single-branch (only the
attn residual, not the mlp) and re-measure.

## Milestone H — methodology check (independent investigation)

Even if residual-LRP helps, the union-of-top-k Petsiuk variant is non-standard
and may have its own pathologies. Worth running once for triangulation:

28. **Pixel-rank Petsiuk on the single top-1 conditional heatmap** — instead
    of "union of 8 concepts", compute the heatmap conditional on the
    top-1 concept only, then rank pixels of *that* heatmap. Bypasses the
    union saturation hypothesis (which we ruled out via cross-scale, but
    confirmation never hurts).
29. **Signed top-k vs absolute top-k** — current ranking uses
    `scores.flatten().abs()`. Try `scores.flatten()` (positive contributions
    only) and report.

## Milestone E — visualisation polish

16. **Multi-block comparison figure** in the walkthrough notebook: same image, top-k concepts at blocks {3, 6, 9, 11} side-by-side. Demonstrates concept progression through the network.
17. **Class-conditional reference samples**: integrate `FeatureVisualization.get_stats_reference` into the notebook (currently uses only `get_max_reference`).
18. **Activation vs. relevance maximisation** comparison cell — the original CRP paper makes a point of this; we currently use only RelMax.

## Milestone F — release

19. **Refresh `tutorials/vit_crp/README.md`** for any API changes from milestones A–D.
20. **Update PR #2 description** with the faithfulness table from Milestone A.
21. **Mark PR ready for review.** Block on milestone A done + milestone D's conservation test passing or PA-LRP added.
22. **Final pass on `CURRENT_STATE.md`** before merge.

## Cross-cutting nice-to-haves

* **Multi-architecture canonizers**: HuggingFace ViT (`transformers.models.vit`) and torchvision ViT have different `Attention` modules. Each needs its own `Canonizer` analogous to `TimmViTCanonizer`. Currently `timm` only.
* **CP-LRP variant** for comparison — `GTIStopGradient` hook + `AttnLRPCPComposite`. Mostly mechanical given the existing GTI infrastructure.
* **Multi-branch Gamma rule** — `GTIGamma` is single-branch (positive-weight clamp). zennit's full `Gamma` rule has 5 branches handling positive/negative inputs and weights. Probably unnecessary for ViTs (post-LayerNorm features are mostly positive) but worth benchmarking.
* **Multi-target-class attribution** — the current pipeline runs one backward pass per class. For a top-5 class comparison, batch them.

## Open design questions

* **Single shared `qkv_tap` vs. three per-part taps (`q_out`, `k_out`, `v_out`)** — earlier spec called for three. Current implementation uses one tap covering all three concatenated. Functionally equivalent (mask isolates the part), simpler, fewer hooks. Worth keeping unless there's a strong reason to split.
* **Should `AttnLRPGammaComposite` be the default**? Per AttnLRP §3.2.1, yes. Once Milestone A confirms, swap the notebook and tutorials default and deprecate `AttnLRPEpsilonComposite` to "for comparison only".
* **What to do with `tests/test_attribution.py` / `tests/test_integration.py`** (the legacy, non-ViT tests) — they already fail under the current zennit version. Two distinct problems:
  * `test_integration.py` — `EpsilonPlus([SequentialMergeBatchNorm()])` → `EpsilonPlus(canonizers=[...])` (mechanical API drift). Cheap fix.
  * `test_attribution.py` — `test_parallel_attribution` / `test_parallel_cond_attribution` / `test_seq_cond_attribution` show real numerical regressions (e.g. `relevances["layer1"] == [1, 0]`, expected `[1, 4]`); likely a semantic change in `CondAttribution`'s handling of `{"layerN": []}` (empty-list condition). Not a one-liner; needs a bisect against zennit-crp upstream.
  Either fix or delete (we don't own those tests). Defer to release.
