# Registers step 1c (XAI-35): detection REDO — per-sample criterion, single-block flagging, both sites

Agent, 2026-07-24. Reviewer-ordered redo of step 1 (`step1_detect.md`). Script:
`experiments/scripts/registers_step1c_redo.py` (collect / analyze / figures /
colocation). Arrays `data/results/registers/step1c_*.npz`, figures
`figures/registers/step1c_redo/` (png+pdf), summary `step1c_analysis.json`.

## Criterion changes (vs step 1)

1. **Per-sample statistics** — `tau_b(sample) = mu_b(sample) + 4*sd_b(sample)`,
   mu/sd over the 196 patch tokens of that single sample at block b (CLS
   excluded). Old: population mu_b/sd_b over all N images.
2. **Single-block flagging** — outlier iff flagged at ANY single block (union
   over blocks 0..11). Old: consensus >= 3 of blocks 6..11. Per-block flags
   are kept (`step1c_masks_*.npz::masks_<site>`).
3. **Two inspection sites** — (a) `blocks[i]` output (residual stream, as
   before) and (b) `blocks[i].attn.proj_drop` output (attention output, before
   the residual add). The criterion applies per site.

## Setup

- ViT-B/16 · timm ImageNet val (`n_per_class=10`), N=256; ViT-S/16 ·
  FunnyBirds probe (`2026-06-03_000556/best.pt`), **test** split, N=256.
  Loaded via `crp_gallery.load_model` / `load_eval_dataset`,
  `model_io.backbone_transforms`; round-robin class-diverse selection, seed 0;
  `ds_indices` persisted in every npz.
- Note: the exact step-1 image sets are not reproducible from the repo (the
  step-1 collector was scratchpad-only and consumed the RNG differently; its
  FunnyBirds sample was TRAIN-clean, the redo uses TEST as ordered). All
  old-vs-new comparisons below therefore apply the OLD rule to the SAME redo
  norms — identical tokens, criterion is the only difference.

## Detection — per-site per-block outlier fractions (%)

ViT-B/16 · ImageNet (N=256):

| site | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 | b10 | b11 | union | per-img mean [min..max] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| residual | 0.03 | 0.14 | 0.25 | 0.92 | 1.25 | 1.55 | 1.77 | 1.81 | 1.83 | 1.82 | 1.52 | 1.09 | 2.00 | 3.9 [2..9] |
| proj_drop | 0.21 | 0.01 | 0.05 | 0.06 | 0.31 | 0.05 | 0.02 | 0.03 | 0.02 | 0.01 | 0.00 | 0.01 | 0.76 | 1.5 [0..8] |

ViT-S/16 · FunnyBirds test (N=256):

| site | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 | b10 | b11 | union | per-img mean [min..max] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| residual | 0.17 | 0.08 | 0.07 | 1.10 | 1.83 | 2.17 | 2.65 | 2.68 | 2.70 | 2.69 | 2.30 | 2.01 | 2.91 | 5.7 [4..12] |
| proj_drop | 0.13 | 0.56 | 0.18 | 0.13 | 0.13 | 1.87 | 0.01 | 0.03 | 0.06 | 0.00 | 0.00 | 0.00 | 2.81 | 5.5 [0..16] |

- **Onset** (first block with frac > 0.5%): residual — block 3 in both models
  (matches step 1). proj_drop — never crosses 0.5% for ViT-B; for ViT-S the
  formal onset is block 1 (0.56%), but that and the isolated block-5 spike
  (1.87%) are transient — proj_drop fractions collapse to ~0 at blocks 6-11
  in both models.
- **Site finding:** the high-norm outliers live in the RESIDUAL stream and are
  not visible at the attention output. At the blocks where residual outliers
  plateau (6-9), attn.proj_drop flags ~0.01-0.06% (near-noise): whatever
  writes the outlier mass into these tokens, it is not delivered through the
  attention output-projection at those blocks; the residual norms carry it.
- 100% of images have >= 1 flagged token (residual); ViT-B min 2, ViT-S min 4
  per image.

## Old vs new criterion (residual site, same samples/norms)

Jaccard between per-block masks (new per-sample rule vs old population rule):

| model | blk6 | blk7 | blk8 | blk9 | image-level (union vs >=3-of-6..11 consensus) |
|---|---|---|---|---|---|
| ViT-B · ImageNet | 0.98 | 0.98 | 0.98 | 0.99 | 0.90 |
| ViT-S · FunnyBirds | 0.99 | 0.99 | 0.99 | 0.99 | 0.92 |

The criterion change barely matters where it matters: at the plateau blocks
the per-sample and population thresholds flag near-identical token sets
(Jaccard 0.98-0.99). The image-level sets differ a little more (0.90/0.92)
because the any-block union additionally picks up tokens flagged only at
early/late blocks that the >=3-votes consensus dropped (union fraction 2.00%
vs 1.82% old image-level for ViT-B; 2.91% vs 2.68% for ViT-S). Old-rule masks
recomputed on the redo samples are stored as `old_*` keys in
`step1c_masks_*.npz`.

## Figures

- `norm_bimodality_{vit_base_imagenet,vit_small_funny_birds}.{png,pdf}` —
  2 (sites) x 4 (blocks 2, 6, 9, 11) histograms, log-x/log-y, with the
  per-sample tau distribution (median line + IQR band). Residual mid blocks
  show the two modes separated by an empty valley with tau inside it;
  proj_drop rows show a single mode with tau in the upper tail (consistent
  with the near-zero proj_drop fractions).
- `outlier_fraction_per_block.{png,pdf}` — per-block flagged fraction, both
  models x both sites, NO threshold lines (the old figure's duplicated 2%
  line is gone).

## Colocation redo (6 ImageNet gallery samples, NEW masks)

Masks: per-sample any-block rule on the stored gallery norms
(`gallery_samples_vit_base_imagenet.npz::norms`, residual site); heatmaps: the
existing full-model class-conditional `cp_lrp_baseline` arrays, aggregated as
sum |R| per 16x16 patch (identical to step 1). Arrays:
`step1c_colocation_vit_base_imagenet.npz`.

| sample | n_out | top-k∩outlier | |R| mass (area) | conc. | mean |R|-rank /196 |
|---|---|---|---|---|---|
| lizard | 4 | 75% | 20.0% (2.0%) | x9.8 | 8.0 |
| cheeseburger | 5 | 60% | 26.4% (2.6%) | x10.4 | 4.6 |
| goldfish | 4 | 50% | 30.6% (2.0%) | x15.0 | 21.5 |
| sports_car | 4 | 100% | 32.9% (2.0%) | x16.1 | 2.5 |
| daisy | 4 | 100% | 25.7% (2.0%) | x12.6 | 2.5 |
| golden_retriever | 4 | 100% | 28.5% (2.0%) | x13.9 | 2.5 |

Colocation conclusion unchanged: outlier patches carry 20-33% of total |R| on
2-2.6% of the area (concentration x10-x16), mean 81% of the top-k |R| patches
are outliers. The new any-block union adds ~1 extra token per sample vs the
step-1 consensus (typically an early-block-only flag with modest |R|), which
is what lowers the top-k overlap / raises the mean rank slightly (goldfish's
mean rank 21.5 is one such low-relevance extra token; its top patches remain
outliers).

Figure: `gallery_colocation_p{1,2}.{png,pdf}` in `figures/registers/step1c_redo/`
(rendered by `registers_step1_figures.py --colocation-npz
step1c_colocation_vit_base_imagenet.npz --out-dir figures/registers/step1c_redo`;
layout/labels identical to step 1). Letter order: (a) lizard 563,
(b) cheeseburger 471, (c) goldfish 3232, (d) sports_car 598, (e) daisy 132,
(f) golden_retriever 358.

## Paper copies

Copied to `crp-paper/iclr2026/journal-figures/`:
`norm_bimodality_imagenet.pdf`, `norm_bimodality_funnybirds.pdf`,
`outlier_fraction_per_block_v2.pdf`, `gallery_colocation_p1_v2.pdf`,
`gallery_colocation_p2_v2.pdf`.
