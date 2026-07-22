# Related-work brief: SemanticLens (2501.05398) & CaFE (2509.00749)

Read: full text, ar5iv (SemanticLens) / arXiv HTML v1 (CaFE), 2026-07-22.
Our paper: full CRP (conditional heatmaps + RelMax galleries) at ViT internal sites
(embed-dim / head / value / query / block-output), concept-flipping faithfulness benchmark
(MoRF/LeRF, AOPC on class prob), raw vs SAE bases.

---

## 1. SemanticLens — Dreyer, Berend et al., arXiv:2501.05398, Nat. Mach. Intell. 2025

### Method core
- Embeds each model **component** ("e.g., individual neurons") into the latent space of a
  multimodal foundation model (CLIP-family): component -> top-m **maximally activating**
  dataset samples ("concept examples" E) -> mean of foundation-model embeddings -> one
  semantic vector ϑ per component (Eq. 2, |E|=30 for ImageNet).
- CRP is used for two auxiliary things only: (a) **cropping** concept examples to the relevant
  region (attribution >1% of max, Gaussian-blurred heatmap), (b) **relevance scores** R of
  components per prediction (LRP εz+♭ composite via zennit-crp 0.6) to weight/filter components
  and build class-conditional attribution graphs.
- Operations in semantic space: **search** (cosine sim to text/image probe minus "null"
  embedding), **label** (nearest concept from user-defined set), **compare** models (avg max
  pairwise sim, Eq. 5), **audit** (valid vs spurious alignment scores per component, Sec. 3.4).
- **Interpretability metrics** from embedded concept examples: clarity (mean pairwise cos sim),
  similarity, redundancy, polysemanticity (1 − clarity of 2 cluster centroids); validated by
  AMT user study (218 participants, corr > 0.74).
- Selection of concept examples is **ActMax throughout** (mean- or max-pooled activations;
  Supp D.4, H.1). RelMax is never used. For ViTs they additionally collect *minimally*
  activating samples (no ReLU -> signed activations; Supp B.1).
- Framing: search/describe/compare/audit/evaluate "hidden knowledge"; debugging (ImageNet
  spurious concepts, melanoma model correction by pruning vs retraining).

### Models / datasets / metrics (enumerated)
- **Models analysed**: ResNet18/32/34/50/50v2/101/101v2 (torchvision + timm variants),
  VGG-13/16/19 ±BN; **ViTs (timm)**: `vit_small_patch16_224.augreg_in21k_ft_in1k`,
  `vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k`, `vit_large_patch16_224.augreg_in21k_ft_in1k`
  (Supp B.1).
- **Foundation models**: MobileCLIP-S2 (default ImageNet), DINOv2-base (interpretability
  metrics), WhyLesionCLIP (medical), CLIP-OpenAI ViT-B/32, CLIP-LAION ViT-B/32.
- **Datasets**: ImageNet-1k, ISIC-2019 (melanoma).
- **Experiments**: (4.1) search for bias/artefact/knowledge neurons in ResNet50v2 penultimate
  layer; UMAP knowledge maps; CRP attribution graphs for "Ox" (ResNet only); (4.2) audit of
  26 ImageNet classes, valid/spurious alignment scatter; behavioural test = AUROC separating
  class logits on real vs Stable-Diffusion spurious-only images (≈0.98 single concept, 0.91
  combined); (4.3) VGG-16 on ISIC: ABCDE-rule concepts, spurious neurons (ruler, band-aid,
  red skin), correction via pruning 40/512 penultimate neurons vs augmented retraining,
  artefact-poisoning accuracy drops; (4.4) interpretability ratings of pretrained zoo incl.
  ViTs (Fig. 5b), dropout / L1-sparsity training sweeps (VGG-13, ResNet-34/50); Supp D.4
  labelling-quality benchmark vs CLIP-Dissect / INVERT (200 final-layer neurons).
- **ViTs + CRP/RelMax**: CRP is **not** applied to ViTs. Sec. 3.1: attributions "approximated
  by up-sampled spatial maps"; Supp H.1: "CRP is not applicable to ViT yet. Thus, we use the
  upsampled spatial transformer tokens as heatmaps". ViTs appear only in ActMax-based
  interpretability ratings/comparisons; relevance scores R and attribution graphs are
  CNN-only. **Component granularity for ViTs** = feature dimension of a block output,
  spatially mean-pooled over tokens (± sign split). No heads, no queries/values, no per-site
  taxonomy.

### What they do NOT do (w.r.t. our claims)
- **No LRP/CRP on ViTs at all** — explicitly stated unavailable (Sec. 3.1; Supp H.1). Hence no
  conditional heatmaps for ViT components; ViT "heatmaps" are upsampled activations
  (NetDissect-style).
- **No site taxonomy**: components = neurons of a layer output ("after each layer block" for
  ResNet; block-output feature dims for ViT). Heads/values/queries/embedding dims never
  instrumented or compared.
- **No RelMax galleries**: concept examples selected by activation (ActMax, Eq. 1 top_m of
  activations; Supp D.4 studies ActMax pooling/count). CRP only crops them.
- **No attribution-faithfulness benchmark**: no deletion/insertion, no MoRF/LeRF, no
  concept-flipping. Their quantitative tests are label-quality scores, user-study correlation,
  and output-logit AUROC on generated spurious images.
- **No SAE experiments**: Sec. 2 claims applicability ("also applicable to SAEs or factorized
  activations"); Discussion lists SAEs as future work. No raw-vs-SAE basis comparison.
- Orthogonal goal: semantic labelling/search/auditing; not faithfulness of concept-level
  attribution at internal sites.

### Quotables (verbatim)
1. "For Vision Transformers (ViTs) the CRP method is not available yet, therefore we
   approximate attributions by up-sampled spatial maps" (Sec. 3.1).
2. "CRP is not applicable to ViT yet. Thus, we use the upsampled spatial transformer tokens
   as heatmaps to localize concepts" (Supp. Note H.1).
3. "Whereas we focus in this work on the neural basis, SemanticLens is thus also applicable
   to SAEs or factorized activations." (Sec. 2, Concept Discovery).

---

## 2. CaFE — Han, Kim, Kwak (SNU), arXiv:2509.00749, "Causal Interpretation of Sparse Autoencoder Features in Vision" (short paper, Aug 2025)

### Method core
- Problem: some vision-SAE features are **non-local** — top-activating patch is spatially
  displaced from the evidence (attention mixing); top-activation inspection then misleads.
- **CaFE** = for each SAE latent z_k, treat the scalar activation as attribution target and
  compute a patch-level input-attribution map = the feature's **Effective Receptive Field
  (ERF)** (Eq. 1: score map {(p, A(p|z_k, I))}).
- Propagation: relevance backpropagated **from the target SAE neuron through the SAE encoder,
  then through the ViT** with **AttnLRP** (Achtibat et al. 2024); plug-in alternatives
  evaluated: Integrated Gradients, KernelSHAP, plain Gradients.
- Baseline being displaced: naive activation-ranked patches (and top-activating images of
  prior SAE labelling work: [11]=Zaigrajew, [9]=Pach, [5]=PatchSAE/Lim).
- Qualitative product: per-image ERF heatmap per feature (e.g. "Despair" feature: max
  activation on background floor, ERF pinpoints spilled pills + frowning face).
- Census of non-locality: manual review of first 100 SAE features per layer; non-local
  features rare below layer 9 (CLS-token features), rising to ≈14% at layer 22.

### Models / datasets / metrics (enumerated)
- **Model**: CLIP-ViT-L/14 image encoder (OpenCLIP, Cherti et al. 2023). Single model.
- **SAE**: one **Matryoshka SAE per transformer layer**, trained on 5×10^8 ImageNet-1k
  training patches; ReLU SAE, L2 recon + λ‖z‖₁; site = the layer's hidden patch-embedding
  representation h (one site type per layer; no head/value/query/embed-dim sites; no raw
  basis analysed).
- **Faithfulness metric**: **patch-insertion test** (Samek et al. protocol) — start from a
  blank image, insert patches in importance order, measure recovery of the *feature's own
  activation* z_k; summary = **area under the insertion curve (AUC)**, aggregated over
  images/features, reported per layer (Fig. 3). Compared: CaFE-AttnLRP vs CaFE-IG vs
  CaFE-KernelSHAP vs CaFE-Gradients vs activation-ranked baseline; AttnLRP wins; modest gain
  even for local features. No deletion/MoRF curve, no class-output metric.
- **Other quantification**: per-layer counts of manually-flagged non-local features (Fig. 5).

### What they do NOT do (w.r.t. our claims)
- **Single site, single basis**: SAE latents on per-layer patch embeddings of one CLIP-ViT.
  No site taxonomy (embed-dim/head/value/query/block-output), no cross-site or
  cross-granularity comparison, no raw-neuron vs SAE comparison, no residual-stream-aware
  treatment (Sec. 4 setup: "For each transformer layer, we train a Matryoshka SAE").
- **No CRP apparatus**: direct backprop from one latent scalar; no conditioning sets /
  multi-concept conditional masking, no concept-composition, no relevance-filtered circuits.
- **No RelMax galleries**: reference images are still found via activation (they explain the
  activation post-hoc within an image); no dataset-wide relevance-based reference-sample
  selection or gallery construction (Secs. 3.2, 4.2).
- **Faithfulness is self-referential**: insertion recovers the latent's own activation, not
  the model's decision. No concept-flipping of latent detectors w.r.t. downstream class
  probability, no MoRF/LeRF pair, no AOPC-style area-between-curves, no
  perturbation-in-latent-space at all (their perturbation is input-patch insertion).
- Scale/scope: 4.5-page paper; manual non-locality annotation acknowledged subjective
  (Sec. 4.3 Limitations).

### Quotables (verbatim)
1. "the attribution A is obtanined [sic] by backpropagating relevance scores from the target
   SAE neuron through the SAE encoder and subsequently through the vision transformer."
   (Sec. 3.2).
2. "In an insertion test, we start with a blank image and gradually insert patches from the
   original image in order of their importance, measuring how quickly the feature activation
   is recovered." (Sec. 4.1).
3. "Patch insertion tests confirm that our CaFE more effectively recovers or suppresses
   feature activations than activation-ranked patches." (Abstract).

---

## 3. Differentiation table (draft)

| Capability | SemanticLens | CaFE | Ours |
|---|---|---|---|
| ViT site granularity | block-output feature dims only (spatially pooled); no heads/Q/V; CNN focus | per-layer patch-embedding SAE latents only (CLIP-ViT-L/14) | taxonomy: embed-dim / head / value / query / block-output, same apparatus at each |
| Conditional (concept-level) heatmaps on ViTs | no — "CRP not applicable to ViT yet"; upsampled activation maps | yes, single-latent ERF via AttnLRP backprop from SAE neuron | yes — CRP conditional heatmaps for any site/basis, incl. multi-concept conditioning |
| RelMax reference galleries | no (ActMax + CRP crop; CNNs only for crop) | no (activation-selected images, attribution only within image) | yes, relevance-selected galleries per concept per site |
| Flipping / faithfulness benchmark | none for attributions (output-AUROC on SD images; label-quality; user study) | insertion AUC on the latent's own activation | concept-flipping of latent detectors: MoRF+LeRF, Δprob on class output, AOPC, systematic across sites/layers |
| SAE basis | claimed applicable, not demonstrated (future work) | yes (Matryoshka SAE, only basis) | yes, and matched raw-vs-SAE comparison at identical sites |
| Residual-aware analysis | no | no | yes (residual-stream handling in propagation + site definitions) |
| Concept semantics / labelling | yes (core contribution: CLIP embedding, search, audit) | no (manual) | not core (orthogonal; could consume our galleries) |

## 4. Pitfalls & survivable phrasing

- **Cannot claim**: "first to propagate relevance from SAE latents through a ViT to the
  input" or "first conditional/latent-feature heatmaps for vision SAEs" — CaFE does exactly
  AttnLRP-from-SAE-neuron on CLIP-ViT with insertion-AUC validation. Also avoid "first
  faithfulness evaluation of SAE feature attributions" (their Fig. 3 is one).
- **Cannot claim**: "CRP has never touched ViTs/SAEs" without qualification — SemanticLens
  already runs the CRP toolchain (zennit-crp) around ViTs (for ActMax pipelines) and asserts
  SAE applicability in principle. Instead *use* their explicit gap statement (quote 1/2) as
  motivation: the machinery they declare missing is what we build.
- **"Conditional heatmap" wording**: an ERF map is a heatmap conditioned on one latent. Our
  differentiator is CRP-style conditioning *sets* + site taxonomy + RelMax, not the mere
  existence of a per-latent heatmap. Say "the full CRP apparatus (conditional multi-concept
  heatmaps and relevance-based reference galleries)", not "concept heatmaps".
- **Insertion vs flipping**: CaFE perturbs the *input* and reads the *latent*; we perturb the
  *latent/concept* and read the *class output* (MoRF & LeRF, AOPC). Make the direction of
  intervention explicit or a reviewer will conflate the two.
- **AttnLRP prior art**: propagation rules through attention are Achtibat et al. 2024 (CaFE's
  [1]); our novelty is instrumentation + benchmark, never the rules. Phrase: "building on
  AttnLRP".
- **SemanticLens Supp Tab. A.1** already contains a framework-capability comparison table
  (concept examples/labels/relevances/audit/...); our table should use different axes (sites,
  bases, faithfulness) to avoid looking derivative.
- Suggested positioning sentence (survives both): "Prior work has either embedded
  activation-selected concept examples of coarse components into a semantic space without any
  relevance propagation inside ViTs (SemanticLens), or attributed single SAE latents of one
  CLIP-ViT layer to input patches and validated them by recovery of the latent's own
  activation (CaFE); neither instruments multiple internal sites of a ViT with conditional
  relevance and relevance-based reference galleries, nor measures faithfulness as the effect
  of flipping concept-detectors on the model's decision, nor compares raw and SAE bases under
  one protocol."
