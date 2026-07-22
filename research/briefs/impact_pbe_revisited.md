# Brief: IMPACT ("Sparse but not Simpler") & Pruning-by-Explaining Revisited

Full-text read (ar5iv), 2026-07-22. Context: our paper = CRP concept detectors on ViTs
(sites x granularity x raw/SAE bases) + concept-flipping (unit-deletion AOPC) faithfulness benchmark.

---

## 1. "Sparse but not Simpler: A Multi-Level Interpretability Analysis of Vision Transformers" (IMPACT)
arXiv:2603.15919, Siyu Zhang (UT Austin), 2026. IMPACT = "Interpretability Multi-level Pipeline for Assessing Computational Transparency".

IMPORTANT REFRAME vs. our expectations: this is NOT an SAE-feature-deletion-curve paper. Its question is
"does weight sparsity (Wanda pruning) make ViTs more interpretable?" IMPACT is the *evaluation framework*
(4 levels). Insertion/deletion is **pixel-level only**; SAE latents are evaluated with one-shot top-K
ablation metrics, never with ordered deletion curves, and there is **no LRP/CRP anywhere except inside
Chefer-attribution**. No relevance-vs-activation-vs-random ordering comparison exists in the paper.

### Method core
- Backbone: DeiT-III B/16, ImageNet-1K; sparse variants via Wanda (and magnitude) pruning at 10/30/50/70/80/90%, each + 2 epochs fine-tune. Primary operating point: 70% Wanda (96% of dense top-1 retained; dense 83.71%, 70% 80.44%).
- Four levels: (1) neuron, (2) layer = BatchTopK SAE, (3) circuit = learnable node masking (Gao et al. weight-sparse-circuits style), (4) model = Chefer transformer attribution + insertion/deletion.
- Neuron level: for each of 1000 classes, top-50 most important neurons per layer found via **attribution patching** (Nanda; gradient-based approximation of activation patching — this is the only "importance ranking" method, no LRP). Layers probed: blocks 2, 7, 11. Attention heads treated as single units (activation = L2 norm of head output, D=64); patch-token metrics use max-pool over tokens 1..196.
- Four neuron/SAE metrics: Ablation Impact (simultaneously ablate top-K units, normalized logit change (f_c - f_c^abl)/||f||), Selectivity (AUROC of CLS activation as binary class detector), Class Variance (CV of per-image max patch activation), Label Entropy (Shannon entropy of class activation distribution).
- Circuit level: learnable binary masks (Heaviside + sigmoid straight-through, tau=1) at **8 sites per block**: norm1, q, k, v (per-head), attn_out, norm2, mlp_act (per-feature), mlp_out; masked-out nodes replaced with **dataset-mean activations** (96 hook points precomputed); objective = task loss + k*|active nodes|; 100 ImageNet classes; NCQ metric = (circuit acc / full acc) * (1 - node fraction).
- Model level: Chefer et al. 2021 (gradient-weighted attention + LRP-style relprop, adapted for DeiT-III: LayerScale-aware relprop R_in = R_out*(gamma+eps)^-1, fused QKV decoupled with standalone v_proj for CP-rule, pos_embed padded 196->197 for CLS).
- Headline result: 70%-sparse circuits have ~2.5x fewer edges but MORE active nodes (41.6% vs 29.0%) — "representation spreading"; no systematic gains in selectivity, SAE interpretability, or attribution faithfulness.

### Exact protocol
- Model: DeiT-III B/16 only (one architecture, one training regime).
- Insertion/deletion (only ordered-perturbation experiment, **pixel space**): order = decreasing Chefer-attribution relevance; deletion replaces pixels with **zero** (zero-valued reference image); metric = AUC of **predicted-class probability** curve; insertion AUC up / deletion AUC down; averaged over all 1000 ImageNet categories. No step count stated in text; no random or activation ordering baseline; single attribution method (no comparison of orderings at all).
- Neuron ablation: **mean ablation** (dataset mean), simultaneous top-K, normalized **logit** change; top-50 units/class; blocks 2/7/11; sparsities {0.1,0.3,0.5,0.7}; all 1000 classes.
- SAE: BatchTopK, k=128, expansion factor 32 (dict 24,576), trained on **block-11 residual stream**; hyperparameter sweep (App. B) selected **CLS token** at block 11 as probe (FVE 99.26% dense; all-token/patch SAEs much worse, FVE ~94.7%); identical SAE config reused for all sparsity levels. SAE-feature ablation uses **zero ablation** ("natural off state"), in contrast to mean ablation for neurons. Metrics = the same four; each latent = a unit "analogous to a neuron". Dead-latent %, NMSE, FVE reported. Active-feature count grows with model sparsity (6,092 -> 7,563 from 30%->70% sparse) — fixed-k SAE across bases is acknowledged as a limitation-ish observation ("adaptive k" = future work).
- Circuits: 100 classes (13 semantic groups), binary "class-vs-rest" task, 15 epochs, AdamW, sparsity coefficient 8e-5 dense / 4e-5 sparse (sparse models suffer "circuit collapse" at dense penalty); mean-replacement, masks shared across tokens and images.

### How SAE features are attributed
- Importance ranking: attribution patching (gradient x activation-difference proxy) to pick top-K units; then one-shot top-K zero-ablation impact. **Non-LRP**; no propagation-based per-latent relevance; no per-latent ordered deletion curve; no AOPC.
- Cross-basis dimensionality: no matched raw-vs-SAE faithfulness comparison. Neurons and SAE latents are evaluated with the same four metrics but at different sites (blocks 2/7/11 raw vs block-11 CLS SAE), different replacement (mean vs zero), and never on a common deletion-curve axis.

### Quotables
1. "Sparse models produce circuits with approximately 2.5x fewer edges than dense models, yet the fraction of active nodes remains similar or higher, indicating that pruning redistributes computation rather than isolating simpler functional modules."
2. "Because SAE latents are explicitly designed to be sparse with a natural 'off' state, we use zero ablation for all SAE feature evaluations, in contrast to the mean ablation used at the neuron level."
3. "These findings suggest that structural sparsity alone does not reliably yield more interpretable vision models."

---

## 2. "Pruning By Explaining Revisited: Optimizing Attribution Methods to Prune CNNs and Transformers"
arXiv:2408.12568, Hatefi, Dreyer, Achtibat, Wiegand, Samek, Lapuschkin (Fraunhofer HHI / TU Berlin / BIFOLD), ECCV-W 2024. Same group as CRP + AttnLRP. Code: github.com/erfanhatefi/Pruning-by-eXplaining-in-PyTorch.

### Method core
- Extends Yeom et al. 2021 "Pruning by Explaining" (CNN, LRP-z+) by (a) **optimizing LRP composite hyperparameters for the pruning objective** and (b) adding ViT-B-16 via transformer LRP (CP-LRP / AttnLRP rules).
- Component relevance: R_bar_psi = mean over n_ref reference samples of the component's LRP relevance, aggregated over all spatial/token axes (Eq. 1, App. B). ViT linear-layer neuron: sum over tokens; attention head: sum over query & key axes (magnitude variant: |sum over key axis| then sum).
- Pruning = ascending-relevance order ({c}_q = argsort(R)); pruned components **masked to zero** (Eq. 4, indicator mask), few-shot, **no retraining**.
- Key hyperparameter MAG: sort by |R| vs signed R. Finding: best to prune **near-zero-relevance components first** (MAG=True won nearly everywhere), not most-negative-first (Yeom's z+ sidesteps this by having no negative relevance).
- Optimization objective: A_PR = mean top-1 accuracy over m=20 pruning rates (0–95%), maximized over composite hyperparameters via Bayesian optimization (GP surrogate) then grid search.
- Composite search space: network split into 4 depth groups — LLL (first 25%), MLL, HLL (last 25% of hidden), FCL — one rule each from {eps(1e-6), alpha2beta1, gamma(0.25 conv/0.05 linear), z+}; ViTs add a 5th hyperparameter: softmax handling in {CP-LRP (attention matrix held constant, eps-rule on value path), AttnLRP DTD softmax rule, AttnLRP-z+ softmax variant}.
- Findings: ViT-B-16 markedly more over-parameterized than CNNs (~20% prunable at full 1000-class task with no meaningful loss); simple LRP-eps everywhere is a strong general recipe; composites faithful in **input** space (AttnLRP ViT composite) are NOT best for pruning, and vice versa — latent vs input faithfulness dissociate.

### Exact protocol
- Models: VGG-16, VGG-16-BN, ResNet-18, ResNet-50 (conv filters: 4224/4224/4800/26560), ViT-B-16 (torchvision pretrained), ImageNet val for evaluation, reference samples from ImageNet train.
- Deleted units (ViT): two structure classes, evaluated separately — (a) **neurons of the two MLP linear layers per block** (46,080 total), (b) **attention heads** (144). No token pruning, no SAE, no per-embedding-dim, no single-layer protocol — ordering/removal is **global across the whole network** within a structure class.
- Order criteria compared: optimized-LRP composite, "Faithful LRP" composites (Kohlbrenner CNN; AttnLRP Tab. A.1 ViT composite: gamma conv / gamma linear / eps QKV+O projections / AttnLRP-z+ softmax), plain LRP-eps, Yeom LRP-z+, Integrated Gradients (20 steps), random. **Weight-magnitude and activation orderings intentionally excluded**, citing Yeom 2021 (CNN evidence) — so "relevance beats magnitude on ViTs" is NOT actually shown here.
- Random baseline / seeds: 20 random seeds (main comparisons; SEM shaded in all figures).
- Replacement: zero (mask components out of the graph). No mean-replacement variant.
- Metric: **top-1 accuracy** vs pruning rate; 20 rates, 0–95%; summary stats A_PR (area under curve, mean of the 20 points) and Top-PR (highest pruning rate retaining 95% of baseline accuracy, +-3%). Accuracy, not prob/logit; direction = remove least-relevant-first (LeRF-style retention curve), never MoRF.
- Tasks/datasets: ImageNet; three task scales — 1000 classes, 100 classes, 3 classes (restricted output domain to expose over-parameterization); headline method comparisons on the 3-class task, **10 reference samples per class** for relevance (>=10 shown sufficient for CNNs, Fig. A.2; ViT insensitive to n_ref 1–100, Fig. 5).
- ViT numbers (3-class task, Tab. 2): linear neurons A_PR ours 0.87 vs eps 0.83, Yeom 0.77, IG 0.70, random 0.59, Top-PR 57%; heads ours 0.85 (=eps 0.85), Yeom **0.74 < random 0.76** — heuristic z+ composite underperforms random on head pruning.
- Sec. 4.4: heatmap-stability experiment — prune heads for corgi task, track cosine similarity between CRP-composite heatmaps (Achtibat 2023, ref [1]) of pruned vs unpruned model; heatmap change correlates 0.99 with confidence loss.

### CRP-on-ViT specifics (what they did / did not do)
- LRP rules used on ViT: eps / alpha-beta / gamma / z+ per depth-group; softmax via CP-LRP (softmax treated as constant, relevance through value path with eps on the A.V matmul) or AttnLRP (DTD softmax linearization, Eq. A.8; matmul rule Eq. A.9 splitting relevance 50/50 with 2*O denominator; z+ applied on the softmax Jacobian linearization for ViTs). Latent site = raw component outputs (MLP neurons, heads); relevance read out per component and summed over tokens.
- They use **plain LRP class-conditional latent relevance, not CRP conditional relevance**: no concept-conditioning masks in the backward pass, no conditional heatmaps for latent units, no concept visualization (RelMax etc.), no per-concept analysis. CRP [1] appears only as the heatmap composite in Sec. 4.4 and as a citation.
- Not done: SAE or any learned basis; per-layer/per-site deletion curves (only global pruning); MoRF deletion / AOPC; prob- or logit-target curves; comparison vs activation or weight magnitude on ViTs; faithfulness as the stated goal (goal = compression; faithfulness is a lens). BUT they explicitly frame pruning as latent-space faithfulness evaluation (quote 1 below) — this is the closest conceptual claim to our benchmark and must be cited.

### Quotables
1. "Attribution-based pruning in combination with measuring the model performance resembles a faithfulness evaluation scheme in latent space."
2. "Our experiments ... reveal that the highest pruning rates are achieved by first pruning components with near zero relevance, as their minimal contribution ensures low impact on the overall model performance."
3. "Optimizing an attributor for two different contexts of faithfulness (input or latent space/pruning) does not necessarily lead to an attributor that attributes faithfully in both input and latent space." (Sec. 4.3/0.F.2, lightly compressed)

---

## Protocol compatibility checklist (for our concept-flipping benchmark)

Copy (to be comparable):
- **Zero replacement** for deleted units: PbE-R zeroes components; IMPACT zero-ablates SAE latents. For raw-neuron flipping, add a mean-replacement robustness check (IMPACT's argument: mean avoids OOD artifacts; SAE latents have a "natural off state" so zero is principled there).
- **>=10 reference samples per class** for relevance estimation (PbE-R: stable at 10 for CNNs, ViT insensitive) — cite when justifying our sample counts.
- **Random baseline with ~20 seeds** + SEM shading (PbE-R convention).
- Report an **accuracy-retention LeRF curve + A_PR-style AUC + Top-PR-style threshold** as a secondary readout so numbers are directly comparable to PbE-R; x-axis = fraction of units removed on a fixed grid (theirs: 20 steps, 0–95%).
- Insertion/deletion AUC convention if we do pixel-space sanity checks: predicted-class **probability**, zero baseline (IMPACT).
- LRP-eps as the always-included reference composite (PbE-R: best simple recipe for latent attribution on ViTs), plus CP-LRP vs AttnLRP softmax handling as an explicit axis — matches our lrp_configs registry.

Deliberately differ (and say so):
- **MoRF flipping + AOPC on target prob** (our concept-flipping): both papers only do LeRF-style retention (PbE-R) or pixel-space MoRF (IMPACT). Running both MoRF and LeRF directions bridges to both.
- **Per-layer / per-site curves**: PbE-R orders globally across the network; IMPACT probes 3 blocks with one-shot ablation. Nobody produces per-site deletion curves.
- **CRP conditional relevance per concept** (conditioning masks, conditional heatmaps): absent from both.
- **SAE-vs-raw matched faithfulness comparison**: IMPACT evaluates SAE latents but never on a deletion curve and never matched against raw units; PbE-R has no SAE. Handle dimensionality by %-of-units and **%-of-total-relevance removed** x-axes (the latter is novel — neither paper uses it) since raw (768/3072) vs SAE (24k, k active) unit counts are incommensurable.
- **Per-embedding-dim granularity** (our EmbeddingDimConcept): neither paper deletes single residual-stream dims.
- **Include activation/magnitude orderings on ViTs**: PbE-R excluded them (citing CNN-only evidence from Yeom 2021) — an open gap we can fill cheaply and legitimately claim.

## Pitfalls for our novelty wording
- Do NOT claim latent-deletion-as-faithfulness is new: PbE-R states it verbatim ("resembles a faithfulness evaluation scheme in latent space"), and Yeom 2021 precedes both. Claim instead: first *concept-level, CRP-conditional, per-site, cross-basis (raw vs SAE)* flipping benchmark on ViTs.
- Do NOT cite PbE-R for "relevance beats magnitude on ViTs" — they never ran magnitude/activation baselines on ViTs (excluded, citing CNN results in Yeom 2021). They DO show relevance beats random and IG on ViTs — cite that; and that a bad composite (Yeom z+) is *worse than random* for head pruning — good motivation for our composite/site sweep.
- Do NOT describe IMPACT as an SAE-attribution-deletion benchmark: its insertion/deletion is pixel-space; SAE latents get one-shot top-K ablation metrics ranked by attribution patching, no ordered curves, no LRP.
- IMPACT's neuron/SAE importance = attribution patching (gradient proxy), not relevance propagation — if we compare "relevance vs gradient-proxy" orderings we are not replicating either paper.
- PbE-R authors = CRP/AttnLRP group; they used AttnLRP rules for pruning and the CRP composite for heatmaps but never conditional concept relevance — word our delta as extending *their own* stack from component-level pruning to concept-level faithfulness, not as introducing LRP-to-ViT (that's AttnLRP/CP-LRP prior work).
- IMPACT's fixed SAE config across models is criticized internally ("representation spreading", adaptive-k future work) — our cross-basis comparison should pre-empt the same critique (matched-k / dim-sweep, cf. experiments/sae.py dimsweep).
- Terminology collision: "IMPACT" is their framework acronym; avoid unqualified "IMPACT benchmark" phrasing implying a deletion benchmark.
