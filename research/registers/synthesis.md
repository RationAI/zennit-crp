# Register-token artifacts: synthesis of steps 1–4 (XAI-34)

*2026-07-24. Four parallel investigations (XAI-35..38); per-step notes in this
directory; arrays `data/results/registers/`; figures `figures/registers/`.*

## Question (Slack 2026-07-23)

Gallery heatmaps show high-relevance blobs on background patches, matching
Darcet et al. "ViTs Need Registers" (arXiv:2309.16588). Is it an LRP problem
(H_A) or real input sensitivity (H_B)? Avoid or explain?

## Findings

1. **Blobs = high-norm scratch-pad tokens** (step 1). Colocation decisive:
   89% of top-k |R| patches are outliers; 19–33% of |R| mass on 1.5–2% of
   area (×12–20). ViT-B/ImageNet: 1.8% of tokens, onset block 3–4, stable
   positions across blocks 5–11 (Jaccard 1.0). Surprise: ViT-S/FunnyBirds is
   NOT clean (2.6%, ×10 norms, corner-anchored (1,1) P=0.72) — presumably
   inherited from augreg-21k pretrain; no register-free in-repo baseline.
2. **Input content causally inert — H_B rejected** (step 3). Occluding the
   rank-1-relevance outlier patch: Δp ≈ −0.0005 (ns), indistinguishable from a
   random background patch, 35× below a top-relevance object patch; relevance
   mass uncorrelated with causal effect (ρ=−0.09). Hydra-guarded: relocation
   (48% of images) does not explain the null (p=0.47). 58% of occluded
   patches stay outliers with replaced content.
3. **Mechanism = ResidualRatio × high norms — H_A confirmed** (step 2).
   Attention-residual branch fraction collapses at outlier tokens exactly
   where norms explode: median f 0.07–0.13 vs 0.27–0.34 (blocks 5–10, p≈0).
   Write/read phases visible: MLP branch fraction HIGHER at outliers in
   blocks 2–5 (register write, 0.50–0.66 vs 0.31), collapsing at 7–9.
   Composite ablation (input-|R| concentration on register patches):
   cp_lrp_baseline 15.2 → attnlrp_gamma 8.7 → residual_symmetric 2.4
   (non-overlapping CIs). Residual-split rule is the dominant lever (6.3×);
   symmetric split trades artifact suppression for noisier heatmaps.
4. **Phenomenon is model-side and attention-mediated** (step 4). Raw CLS
   attention concentrates hardest (23.9×; 40% of attention mass), LRP 14.6×,
   rollout ranks outliers top-5 in 100% of images; gradient×input 3.5×, IG ≈
   chance. Not LRP-specific.

## Resolution of the Slack question

"Is it an LRP problem?" splits in two:

- **Relevance AT the scratch-pad token is faithful** — the token genuinely
  routes a large share of the decision (attention: 40%); every
  attention-aware method sees it.
- **Relevance at the token's INPUT-PATCH coordinates is an artifact** — the
  patch content is causally inert; the relevance lands there because the
  magnitude-proportional residual split parks the token's relevance in its
  token-local skip column instead of redistributing through attention. This
  projection error is rule-dependent: symmetric residual split suppresses it
  6.3× (at a noise cost).

Adam's intuition "LRP should redistribute it back before the input" is
correct in principle but requires the cross-token (attention) path to carry
the relevance; ResidualRatio starves exactly that path at high-norm tokens.

## Consequences

- Paper: this is a section, not a bug — mechanism + quantified rule-lever +
  cross-method comparison + causal falsification. Supersedes XAI-21's ad-hoc
  outlier thresholding item.
- Presentation choice (open): default composite for galleries — keep
  cp_lrp_baseline (sharp, artifact marked/masked using detection) vs
  residual-symmetric (artifact-suppressed, noisier). Candidate: annotate
  outlier patches in gallery heatmaps rather than change the rule.
- Step 5 (explain): TokenConcept conditioned on the (per-image stable)
  outlier position — show what the scratch-pad aggregated.
- Step 6 (registers): DINOv3 comparison still blocked on head checkpoint.
