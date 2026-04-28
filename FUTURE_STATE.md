# Future State — backlog for `transformer-multi-concept`

The implementation in `CURRENT_STATE.md` is complete enough to land as the first review-ready PR. This file enumerates what still needs to be done before the branch is fully aligned with the original `IMPLEMENTATION_PLAN.md`, ranked by impact and ordered into milestones.

## Milestone A — close the AUC anomaly (next iteration)

Iteration 3's metrics smoke (vit_tiny + bare ε-LRP) showed the random-concept baseline beating relevance-ranked top-k on deletion / insertion AUC for the finer granularities (`kqv_head`, `head_dim`). This is the single most important open item — without a fix, the faithfulness story for the finer concepts is unconvincing.

1. **Re-run `metrics.py` with `AttnLRPGammaComposite`** on:
   * a non-trivial sample size (≥ 64 images, ≥ 4 classes per image),
   * `vit_base_patch16_224` rather than `vit_tiny_patch16_224` (the latter has only 3 heads, making the finer granularities trivially noisy),
   * a sweep of γ ∈ {0.0, 0.1, 0.25, 0.5}.
2. **Verify the ordering `deletion_AUC(true) < deletion_AUC(random)` holds** for all four granularities under the chosen γ. If it doesn't, escalate to milestone D (positional-encoding / PA-LRP) before declaring the rule correct.
3. **Pick the default γ** based on the sweep, document in `tutorials/vit_crp/README.md`, set as the `AttnLRPGammaComposite` default if different from 0.25.
4. **Acceptance**: a faithfulness table in the PR description with one row per (granularity, composite, γ) and the ordering called out explicitly. Random baseline must lose on deletion AND insertion across all four granularities.

## Milestone B — additional faithfulness metrics + baselines

`IMPLEMENTATION_PLAN.md` Phase 5 calls for a richer benchmark. Currently only deletion / insertion AUC + random-concept baseline exist.

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

16. **Generalise `FeatureVisualization._attribution_on_reference`** — it currently hardcodes `ChannelConcept.mask` / `ChannelConcept.mask_rf` regardless of the concept registered in `self.layer_map`, which means `get_max_reference(..., composite=composite, plot_fn=vis_opaque_img)` raises `IndexError` for our attention concepts (the flat int id is interpreted as a sequence-position index by `ChannelConcept.mask`). Structurally solvable (~5 LOC): pull the concept from `self.layer_map[layer_name]` and pass `mask_map=concept.mask` (or `concept.mask_rf` if `rf=True`). **Interpretability impact**: until this lands, the walkthrough notebook can't render per-reference-sample conditional heatmaps — which is the canonical "what does this concept look like" RelMax output (the eye-age example in the CRP paper). The workaround in the notebook (raw RGB samples + conditional heatmap only on the **target** image) is functional for ranking and target-image localisation but loses the localisation on each reference sample. Recommended priority: do this alongside Milestone A.
17. **Multi-block comparison figure** in the walkthrough notebook: same image, top-k concepts at blocks {3, 6, 9, 11} side-by-side. Demonstrates concept progression through the network. Closes Phase 4.4 of `IMPLEMENTATION_PLAN.md`.
18. **Class-conditional reference samples**: integrate `FeatureVisualization.get_stats_reference` into the notebook (currently uses only `get_max_reference`).
19. **Activation vs. relevance maximisation** comparison cell — the original CRP paper makes a point of this; we currently use only RelMax.

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

* **Single shared `qkv_tap` vs. three per-part taps (`q_out`, `k_out`, `v_out`)** — the original `IMPLEMENTATION_PLAN.md` Phase 2 spec called for three. The current implementation has one tap covering all three concatenated. Functionally equivalent (mask isolates the part), simpler, fewer hooks. Worth keeping unless there's a strong reason to split.
* **Should `AttnLRPGammaComposite` be the default**? Per AttnLRP §3.2.1, yes. Once Milestone A confirms, swap the notebook and tutorials default and deprecate `AttnLRPEpsilonComposite` to "for comparison only".
* **What to do with `tests/test_attribution.py` / `tests/test_integration.py`** (the legacy, non-ViT tests) — they already fail under the current zennit version. Either fix the `EpsilonPlusFlat([SequentialMergeBatchNorm()])` → `EpsilonPlusFlat(canonizers=[...])` API drift, or delete (we don't own those tests). Cheap fix; do as part of release.
