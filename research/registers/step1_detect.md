# Registers step 1 (XAI-35): high-norm outlier tokens — detection + colocation with CRP-gallery artifacts

Agent-1, 2026-07-24. Step 1 of XAI-34 (Darcet et al., arXiv:2309.16588 "Vision
Transformers Need Registers"). All arrays in `data/results/registers/`, figures in
`figures/registers/step1_detect/` (png+pdf).

## Setup

- **ViT-B/16 · ImageNet** — timm `vit_base_patch16_224` (pretrained, full head), the
  gallery model (`load_model(base="vit_base", dataset="imagenet", model_source="checkpoint")`).
- **ViT-S/16 · FunnyBirds** — finetuned linear probe on timm
  `vit_small_patch16_224.augreg_in21k_ft_in1k` (frozen backbone), intended as clean control.
- N=256 class-diverse images each (ViT-B: 1 img from each of 256 ImageNet classes out of
  the `n_per_class=10` val subset; ViT-S: round-robin over the 50 FunnyBirds classes),
  deterministic seed=0 selection; exact `ds_indices` + labels stored in each npz.
- Measured: L2 norm of every `backbone.blocks[i]` output token (i=0..11). Token 0 = CLS
  (excluded from all statistics), tokens 1..196 = 14x14 patch grid, row-major.
- Files: `norms_vit_base_imagenet.npz`, `norms_vit_small_funny_birds.npz`
  (`norms` (12, 256, 197) float32, `ds_indices`, `labels`, `meta`).

## Outlier criterion (documented choice)

**Primary: token is an outlier at block b iff `norm > tau_b = mean_b + 4*sd_b`**, computed
per block per model over all patch-token norms (CLS excluded).

- Justified by observed bimodality: at mid blocks the bulk and outlier modes are separated
  by roughly an order of magnitude (ViT-B blk6: p90 = 14.3 vs p99 = 92.5; ViT-S blk7:
  p90 = 27.1 vs p99 = 260.1), and mean+4sd lands inside the empty valley between them.
- Sensitivity: two alternatives (`2*median_b`, fixed 98th percentile) give the same
  plateau fractions and the same tokens at blocks 5–10 (fig `outlier_fraction_per_block`);
  the phenomenon is criterion-insensitive where it matters.
- Known edge: at block 11 the bulk mode itself grows (median 61 / 135), so per-block
  thresholds under-flag there (ViT-B blk11 Jaccard vs blk9 mask = 0.17). Therefore the
  **image-level outlier token** used everywhere downstream is a consensus:
  **flagged in >= 3 of blocks 6..11**. Masks per block AND image-level masks + thresholds
  are in `outlier_masks_<model>.npz`.

## Results — detection

| | ViT-B/16 · ImageNet | ViT-S/16 · FunnyBirds |
|---|---|---|
| onset (frac > 0.5%) | block 3 | block 3 |
| plateau fraction (blocks 6–9) | 1.8% | 2.6–2.7% |
| plateau norms (p99 vs median, blk8) | 113 vs 20 (x5.5) | 296 vs 31 (x9.7) |
| block consistency (Jaccard vs blk9, blocks 6–10) | 0.74–1.00 | 0.85–1.00 |
| outliers per image (consensus) | mean 3.6, range 2–6, 100% of images | mean 5.2, range 3–7, 100% of images |
| spatial pattern | weak preference for rows 1 and 12, max P(pos)=0.25, slight interior bias | strongly position-anchored near corners/edges of the interior ring, max P(pos)=0.72 at (1,1); border ring itself never |

**Key finding: the expected-clean control is NOT clean.** ViT-S/FunnyBirds (augreg21k
backbone) has *more and stronger* outliers than ViT-B (relative norm ratio ~10x vs ~5.5x),
and they are far more position-stereotyped — plausibly because FunnyBirds' uniform
backgrounds give many equally-uninformative patches, so the model reuses fixed slots.
Every single image in both models carries at least 2 outlier tokens. So there is no
within-repo "register-free" baseline; step-2+ comparisons must be
outlier-mask-conditioned rather than model-vs-model.

Figures: `norm_distributions` (percentile fan + tau per block, log scale),
`outlier_fraction_per_block` (3 criteria), `outlier_spatial_frequency` (14x14 P(outlier)),
`per_image_outlier_count`.

## Results — colocation with CRP-gallery relevance artifacts

For the 6 fixed gallery samples (`crp_gallery.IMAGENET_SAMPLES`; ds_indices persisted:
lizard 563, cheeseburger 471, goldfish 3232, sports_car 598, daisy 132,
golden_retriever 358), each sample's own outlier mask (own forward, consensus rule above)
was compared against the `cp_lrp_baseline` total relevance heatmap. The gallery PNGs are
colormapped, so quantification uses a bit-identical recomputation of the same heatmap
(raw values; stored in `gallery_samples_vit_base_imagenet.npz`); the overlay figure shows
the *existing* gallery PNG. |R| pooled per 16x16 cell to the 14x14 patch grid;
k = per-image outlier count.

| sample | n_out | top-k∩outlier | |R| mass on outliers (area) | conc. | mean |R|-rank /196 |
|---|---|---|---|---|---|
| lizard | 3 | 100% | 19.3% (1.5%) | x12.6 | 2.0 |
| cheeseburger | 3 | 100% | 24.7% (1.5%) | x16.1 | 2.0 |
| goldfish | 3 | 67% | 30.2% (1.5%) | x19.7 | 2.7 |
| sports_car | 4 | 100% | 32.9% (2.0%) | x16.1 | 2.5 |
| daisy | 3 | 67% | 21.9% (1.5%) | x14.3 | 2.7 |
| golden_retriever | 3 | 100% | 26.6% (1.5%) | x17.4 | 2.0 |

**Verdict: colocation confirmed, decisively.** The high-norm outlier tokens ARE the
isolated high-intensity blobs in the gallery heatmaps: mean 89% of the top-k |R| patches
are outlier tokens; outlier patches carry 19–33% of total |R| mass on 1.5–2% of the area
(concentration x12–x20); their mean |R| rank is 2.0–2.7 out of 196 — i.e. the top-2/top-3
most relevant patches of essentially every gallery sample are register-type outlier
tokens, sitting on semantically empty background (visible in
`gallery_colocation_overlays`). Signed relevance on outlier patches is positive in all 6
samples (+0.12 .. +0.51 of a ~1.0 total), so they inflate, not cancel, the explanation.

Metrics + masks: `colocation_vit_base_imagenet.npz`
(keys/ds_indices/targets/outlier_masks/heat_patch_abs/heat_patch_signed/metrics).

## Reuse for later steps

- Sample sets are fully deterministic: `ds_indices` in every npz; gallery samples are the
  canonical 6 with the indices above (`pick_samples("imagenet", ds)` reproduces them).
- Image-level masks: `outlier_masks_<model>.npz::image_level_mask` (N, 196) bool.
- Per-block thresholds: `thresholds_mean4sd` (12,) — same rule must be recomputed if a
  different image set is used (thresholds are distribution-dependent).
- Collection/analysis scripts kept out of the repo tree (scratchpad); the npz `meta`
  fields document all conventions (token 0 = CLS, row-major grid).
