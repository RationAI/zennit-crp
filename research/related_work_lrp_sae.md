# Related work — LRP × Sparse Autoencoders (for SAE-basis CRP)

> **2026-07-22 update:** partially superseded — the 2025 paper **CaFE**
> (arXiv:2509.00749) DOES propagate relevance through an SAE spliced into a
> CLIP-ViT (via AttnLRP, insertion-AUC eval). See `scout_novelty_crp_vit.md`
> for the full overlap table and the reformulated novelty claim.

Literature scan (2026-06-09) for the SAE-basis CRP project. Question: does anyone
already do **output→SAE-basis relevance decomposition** — initialise relevance at
the output class logit, use LRP/CRP to decompose it onto a trained SAE latent
basis (with conservation), and evaluate by concept-flipping/AOPC vs the
axis-aligned basis, on a ViT?

**Verdict: not found.** Our contribution holds. Closest works each diverge on a
load-bearing axis. Published gallery of this summary:
https://claude-bajger.dyn.cloud.e-infra.cz/zennit-litreview/

## Our approach (baseline for the diffs)
Train SAE on ViT activations at a probe site (`proj_drop` or residual stream);
splice as reconstruction pass-through exposing latents `f`; init relevance at the
**output logit**; LRP/CRP decomposes it **onto the SAE latents** (output→SAE),
conditionally maskable; eval = concept-flipping/AOPC (SAE vs axis-aligned at same
site) + conservation `ΣR(f_i)≈R(probe)`.

## Closest prior work (read in full)

- **CaFE — Causal Interpretation of SAE Features in Vision** — Han/Kim/Kwak, Aug
  2025, [arXiv:2509.00749](https://arxiv.org/abs/2509.00749). *The one paper with
  SAE + AttnLRP on a ViT.* But **opposite direction**: relevance starts at a single
  SAE latent and flows **to input pixels** (feature→input, "effective receptive
  field" on CLIP-ViT-L/14), validated by an insertion test on the feature
  activation. **No conservation, no axis-vs-SAE basis comparison, no
  concept-flipping.** → primary differentiation target (direction + target +
  conservation + basis comparison). Borrow: insertion-on-feature probe; "non-local
  feature" motivation.

- **ClassifSAE — Unveiling Decision-Making in LLMs … with SAEs** — Le Bail et al.,
  Jun 2025, [arXiv:2506.23951](https://arxiv.org/abs/2506.23951). *Same direction &
  goal* (attribute a classification decision onto SAE concepts) but by **ablation**
  (ΔAcc / label-flip / TVD, per concept), not conservative relevance propagation;
  no conservation; no ranked AOPC; **LLM** not ViT; modifies the SAE (classifier
  head + activation-rate sparsity). Borrow: ablation metrics as an independent
  causal cross-check of our relevance ranking.

- **Sparse but not Simpler — Multi-Level Interpretability of ViTs** — Zhang, Mar
  2026, [arXiv:2603.15919](https://arxiv.org/abs/2603.15919). IMPACT harness on
  DeiT-III B/16: BatchTopK SAEs + Chefer attribution + insertion/deletion. But
  **SAE and attribution are separate axes — relevance never flows through the
  SAE**; attribution lands on input pixels; IV = pruning level, not basis. Borrow:
  insertion/deletion protocol; "sparse ≠ more interpretable" framing; we close the
  loop they leave open.

## Adjacent / methodological

- **Interpreto — Explainability Library for Transformers** — Poché et al., Dec
  2025, ACL 2026 demo, [arXiv:2512.09730](https://arxiv.org/abs/2512.09730). Bundles
  attribution + a concept pipeline (incl. SAEs) but **no LRP** (gradient/perturb),
  concept importance via concept→output **gradients** (no conservation), modules
  **unlinked** (linking = future work), **NLP-only**. Evidence the fusion is a known
  tooling gap → supports novelty.

- **Unlocking LRP for Autoencoders** — Kobayashi et al. (Oracle), Mar 2023,
  [arXiv:2303.11734](https://arxiv.org/abs/2303.11734). LRP through an AE: a
  Deep-Taylor rule attributes **reconstruction error → input** (anomaly detection),
  generic AEs (not sparse). Does NOT solve our decoder-bias absorption, but its
  **DTD root-point recipe** is the principled template for a **bias-aware decoder
  rule** (root at the mean/bias so the residual `z−z̃` carries relevance, instead of
  the γ-denominator handing ~98% to the bias). → actionable for fixing our weak Q2
  conservation.

## Foundations & context (cite, no overlap)
- CRP — Achtibat et al., NMI 2023, [2206.03208](https://arxiv.org/abs/2206.03208)
  (our foundation; we swap axis-aligned → SAE basis).
- AttnLRP — Achtibat et al., ICML 2024, [2402.05602](https://arxiv.org/abs/2402.05602).
- Sparse Feature Circuits — Marks et al., ICLR 2025,
  [2403.19647](https://arxiv.org/abs/2403.19647) (SAE circuits via *gradient*, LM).
- Pruned-LRP sparse explanations — [2404.14271](https://arxiv.org/abs/2404.14271)
  (sparsity *in* LRP, not an SAE basis; tangential).
- TopK SAEs — Gao et al., [2406.04093](https://arxiv.org/abs/2406.04093).
- SAEs on ViT (no LRP): saev [2502.06755](https://arxiv.org/abs/2502.06755),
  Universal SAEs [2502.03714](https://arxiv.org/abs/2502.03714), VLM monosemantic
  [2504.02821](https://arxiv.org/abs/2504.02821).
- Superposition/monosemanticity: Elhage 2022
  [2209.10652](https://arxiv.org/abs/2209.10652), Bricken 2023.
- LLEXICORP — CRP + MLLM captioning, [2511.02720](https://arxiv.org/abs/2511.02720)
  (CRP extended, but not to an SAE basis).

## Publication status (venues)

Notable: the works *closest* to us are the *least* published — CaFE is workshop-only,
the two nearest preprints have no venue yet. Peer-reviewed papers are all foundations
or orthogonal context.

| Paper | Venue | Status |
|---|---|---|
| CaFE — 2509.00749 | eXCV Workshop (Explainable CV) @ ICCV 2025 | workshop, not main-track |
| ClassifSAE — 2506.23951 | — | preprint only |
| Sparse but not Simpler — 2603.15919 | — | preprint only (Mar 2026) |
| Interpreto — 2512.09730 | ACL 2026 (System Demonstrations) | published (demo) |
| LRP for Autoencoders — 2303.11734 | — | preprint only |
| CRP — 2206.03208 | Nature Machine Intelligence 5:1006–1019, 2023 | journal |
| AttnLRP — 2402.05602 | ICML 2024 (PMLR v235) | published |
| Sparse Feature Circuits — 2403.19647 | ICLR 2025 | published |
| Scaling/TopK SAEs — 2406.04093 | ICLR 2025 | published |
| Pruned-LRP — 2404.14271 | ECML PKDD 2024 (Springer LNCS 14944, 336–351) | published |
| VLM monosemantic SAE — 2504.02821 | NeurIPS 2025 | published |
| saev — 2502.06755 | — | preprint only |
| Universal SAEs — 2502.03714 | — | preprint only |
| LLEXICORP — 2511.02720 | — | preprint only |
| Toy Models / Towards Monosemanticity | Transformer Circuits | non-archival |

## Implications
1. **Novelty holds** — frame as conservative output→SAE-basis CRP on ViT +
   AOPC-vs-axis-aligned; differentiate from CaFE (direction), ClassifSAE (ablation
   vs conservation; LLM), Sparse-but-not-Simpler (separate axes).
2. **Fix conservation before claiming Q2** — use 2303.11734's root-point recipe for
   a bias-aware decoder rule.
3. **Strengthen eval** — add ablation cross-check (ClassifSAE) + insertion/deletion
   alongside AOPC for convergent faithfulness evidence.
