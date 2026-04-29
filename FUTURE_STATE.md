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
further γ tuning. PA-LRP is the highest-likelihood fix for `kqv_head`
since the residual relevance leak through `pos_embed` is exactly the kind
of constant-energy term the union-of-top-8 mask would saturate first.

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

## Milestone D — conservation + PA-LRP (gated)

The original AttnLRP paper (Eq. 1) requires `R_input.sum() ≈ R_output.sum()`; positional encodings are treated as constants in the LXT reference impl, which can violate conservation when the positional energy is non-trivial. PA-LRP (Bakish et al., NeurIPS 2025; arXiv 2506.02138) defines an additive rule for positional encodings.

13. **Conservation quantitative test** — measure the ratio `|R_input.sum()| / |R_output.sum()|` per attribution; assert it's within 5 %. Add to `tests/test_vit_integration.py`.
14. **If 13 fails**: implement PA-LRP as a Canonizer that wraps the additive `pos_embed` step in a `_DivideGradientFn(factor=2)` (uniform-rule allocation between input embedding and positional embedding). New class `PALRPCanonizer`. Update `TimmViTCanonizer` to compose it.
15. **If 13 passes**: document the conservation check, mark PA-LRP as not needed.

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
