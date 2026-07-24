# Registers step 1b (XAI-35): per-position flag-frequency statistics

Agent, 2026-07-24. Response to the reviewer challenge on step 1
(`step1_detect.md`): the "corner-anchored at (1,1), P=0.72" claim rested on
N=256 and needed proper per-position statistics on a larger sample, or had to
be abandoned. Script: `experiments/scripts/registers_position_freq.py`;
arrays in `data/results/registers/step1b_position_freq_*.npz`
(+ `step1b_position_stats.json`); figures in `figures/registers/step1b_positions/`.

## Experiment card

- **RQ** — Are some spatial token positions flagged as high-norm outliers
  significantly more often than others in ViT-S/FunnyBirds; is (1,1) a
  persistent anomaly location?
- **H1** — Per-position flag frequency is strongly non-uniform; (1,1) flagged in
  a majority of images. **H0** — flag positions exchangeable across the 14x14 grid.
- **Falsified if** — no position beats uniform expectation after Holm correction
  across 196 positions, or (1,1) has corrected p >= 0.01 or frequency <= 0.5.
- **Detection rule** — IDENTICAL to step 1: L2 norm of each `blocks[i]` output
  token (CLS excluded); per-block threshold `tau_b = mean_b + 4*sd_b` over all
  patch-token norms of the sample; image-level flag = flagged in >= 3 of
  blocks 6–11.
- **Samples** — the card asked for FunnyBirds TEST N=2048, but **the official
  test split only contains 500 images**; deviation: the ENTIRE test split
  (N=500, all 50 classes) is the primary sample, plus a supplementary
  train-clean N=2048 sample at the requested scale
  (`step1b_position_freq_funny_birds_train.npz`). Contrast: ViT-B/ImageNet val
  (10/class) N=1024. Round-robin class-diverse, seed 0; `ds_indices` persisted.
- **Stats** — per-position count k_p out of N; H0 binomial with
  p0 = mean-flags-per-image / 196; exact binomial tail p-values,
  Holm-corrected over 196 positions; chi-square GOF (approximate — flags are
  not independent within an image). H1 supported for (1,1) iff corrected
  p < 0.01 AND frequency > 0.5.

## Results

| | FunnyBirds test (N=500) | FunnyBirds train-clean (N=2048) | ImageNet val (N=1024) |
|---|---|---|---|
| mean flags/image | 5.22 | 5.19 | 3.52 |
| p0 (uniform rate) | 0.0266 | 0.0265 | 0.0180 |
| chi-square GOF (df=195) | 42 069, p ≈ 0 | 170 608, p ≈ 0 | 15 949, p ≈ 0 |
| positions with Holm p < 0.01 | 14 | 15 | 20 |
| top position | **(1,1): 0.722** | **(1,1): 0.708** | (1,11): 0.275 |
| (1,1) | freq 0.722, p_holm ≈ 0 | freq 0.708, p_holm ≈ 0 | freq 0.004, p_holm = 1 |

FunnyBirds top-5 (test N=500 / train N=2048), Holm p < 1e-220 for all:

| pos | freq test | freq train |
|---|---|---|
| (1,1) | 0.722 | 0.708 |
| (1,12) | 0.704 | 0.697 |
| (1,2) | 0.562 | 0.563 |
| (1,11) | 0.558 | 0.567 |
| (6,1)/(6,12) | 0.474 / 0.470 | 0.458 / 0.462 |

ImageNet top-5: (1,11) 0.275, (1,2) 0.236, (12,2) 0.208, (2,5) 0.182,
(12,11) 0.178 — significant non-uniformity too, but max frequency 3.6x lower
and mass spread over ~20 positions; (1,1) itself is at chance.

## Verdict

**H1 SUPPORTED — the step-1 claim replicates and strengthens.** On the full
held-out test split, (1,1) is flagged in 72.2% of images (361/500, Holm
p ≈ 0, far below the 0.01 bar), and the same six interior-ring anchor slots
— (1,1), (1,12), (1,2), (1,11), (6,1), (6,12) — dominate identically at
N=2048 (train-clean) with frequencies stable to ±0.02 across splits. The
step-1 P=0.72 figure was not a small-sample artifact: the test-split estimate
is 0.722. ImageNet shows the expected contrast: non-uniform but weakly
position-preferring (max 0.275), no single persistent slot.

## Visual verification (reviewer's ask)

`figures/registers/step1b_positions/`:

- `actmap_<key>.{png,pdf}` — per-block (0–11) normalized token-norm maps
  (viridis, per-block [0,1]) with magenta borders on flagged tokens; the 6
  canonical gallery samples (c0_0, c1_603, c2_1222, c3_1810, c4_2402,
  c5_2988; train-clean, flags via test-sample thresholds) + 6 random flagged
  test images (ds_index 259, 33, 237, 349, 227, 178; rows persisted as
  `actmap_random_sample_rows`).
- `position_frequency.{png,pdf}` — 14x14 flag-frequency heatmaps, FunnyBirds
  test vs ImageNet, shared scale, top-5 annotated.
- Copied to `crp-paper/iclr2026/journal-figures/` as `position_frequency.pdf`,
  `actmap_example1.pdf` (= c0_0), `actmap_example2.pdf` (= test ds_index 259).

Clearest reading: blocks 5–9 of any sample — bird-shaped norm pattern in
blocks 0–2 collapses by block 4 into 4–7 isolated high-norm tokens parked at
the row-1/row-6 anchor slots, invariant across images; block 11 under-flags
(known step-1 edge: bulk norms grow). Web-gallery overlay integration remains
optional follow-up (not done here).
