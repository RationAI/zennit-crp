# E1 v2 — register-outlier detection: statistical tests (reviewer revision)
_Generated 2026-07-27 20:45 UTC by `experiments/scripts/registers_e1_counts.py` (analyze-stats / figures-v2 / report-v2). Supersedes the 'plateau within factor ~2' criterion of `e1_outlier_counts.md`; no plateau numbers are used anywhere below._

## What was recomputed vs reused
- **M1 recomputed** — the ViT-S/16 FunnyBirds probe changed to checkpoint `data/runs/finetune_vit_small_funny-birds-train-clean/2026-07-26_160337/best.pt` (loaded path asserted). Same N=256 FunnyBirds test indices as v1 (reused verbatim from the old npz, seed 0). → `e1_counts_m1_vit_small_fb.npz`.
- **M2 reused** — `e1_counts_vit_base_imagenet.npz` unchanged (timm ImageNet classifier did not change).
- **M3 recomputed** — now the FINETUNED DINOv3-S classifier backbone (`data/runs/finetune_vit_dinov3_small_funny-birds-train-clean/2026-07-25_200008/best.pt`), replacing the old pretrained-backbone row. Same FunnyBirds indices. → `e1_counts_m3_dinov3s_fb.npz`.
- **M4 reused** — `e1_counts_dinov3_base_imagenet.npz`. Justification: the old dinov3_base row recorded token norms of the pretrained `vit_base_patch16_dinov3` backbone, which is exactly M4's backbone (M4 adds only a frozen linear head, which cannot alter backbone activations). The arrays are therefore identical by construction.
- Old v1 npz files are untouched.

## Definitions
- Criterion (unchanged): per sample and per site, μ/σ over that sample's patch-token L2 norms; outlier iff norm > μ + 4σ. CLS excluded; DINOv3 register tokens excluded from patch statistics, tracked separately. 24 sites = 12 blocks × (post-attn-add, post-mlp-add).
- Unit of analysis: per-image normalized union outlier rate **r_i = |{patch tokens of image i flagged at ANY of the 24 sites}| / T**, T = number of patch tokens (196 standard @224², 256 DINOv3 @256²). Normalization by T makes the architectures comparable.
- Secondary variable: s_i = mean over the 24 sites of the per-site flagged fraction of image i.
- Tests: two-sided / one-sided Mann-Whitney U (asymptotic, tie-corrected; exact p infeasible at n=256 with ties), α=0.05; effect sizes: group medians, median difference with 95% bootstrap CI (10000 resamples), Cliff's δ; KS as robustness check for T3/T4.

## Per-model summary
| tag | model | N | T (patches) | union rate median | union rate mean | per-image union count mean | provenance |
|---|---|---|---|---|---|---|---|
| m1 | M1 ViT-S/16 (std) · FunnyBirds | 256 | 196 | 0.0255 | 0.0286 | 5.6 | RECOMPUTED: finetuned probe finetune_vit_small_funny-birds-train-clean/2026-07-26_160337 |
| m2 | M2 ViT-B/16 (std) · ImageNet | 256 | 196 | 0.0204 | 0.0204 | 4.0 | REUSED: timm ImageNet classifier unchanged |
| m3 | M3 DINOv3-S (+reg, finetuned) · FunnyBirds | 256 | 256 | 0.0234 | 0.0252 | 6.4 | RECOMPUTED: finetuned DINOv3-S classifier backbone finetune_vit_dinov3_small_funny-birds-train-clean/2026-07-25_200008 |
| m4 | M4 DINOv3-B (+reg, frozen bb) · ImageNet | 256 | 256 | 0.0117 | 0.0110 | 2.8 | REUSED: M4's frozen-head classifier uses the pretrained vit_base_patch16_dinov3 backbone verbatim — identical to the backbone these norms were recorded from (a frozen linear head cannot alter backbone activations) |

Full objective per-site table (24 sites × 4 models): `data/results/registers/e1_per_site_table_v2.csv`. The per-site figure `e1_fraction_per_site_v2` is a visual cue only — no numbers are derived from it.

## Test table (primary variable r_i)
| test | pair | alternative | U | p | median diff | 95% CI (bootstrap) | Cliff's δ | reject | KS check |
|---|---|---|---|---|---|---|---|---|---|
| T1 | m1 vs m2 | two-sided | 54624 | 3.4e-41 | 0.0051 | [0.0051, 0.0051] | 0.667 | yes | — |
| T2 | m3 vs m4 | two-sided | 56174 | 5.11e-45 | 0.0117 | [0.0117, 0.0156] | 0.714 | yes | — |
| T3 | m1 vs m3 | greater | 41768 | 3e-08 | 0.0021 | [0.0021, 0.0021] | 0.275 | yes | one-sided D=0.438, p=1.07e-22; two-sided D=0.438, p=2.14e-22 |
| T4 | m2 vs m4 | greater | 55676 | 1.59e-43 | 0.0087 | [0.0087, 0.0126] | 0.699 | yes | one-sided D=0.625, p=1.93e-47; two-sided D=0.625, p=3.86e-47 |

Secondary variable s_i (robustness, same one-sided direction):
| test | pair | alternative | U | p | median diff | 95% CI (bootstrap) | Cliff's δ | reject | KS check |
|---|---|---|---|---|---|---|---|---|---|
| T3s | m1 vs m3 | greater | 65530 | 1.13e-85 | 0.0132 | [0.0129, 0.0136] | 1.000 | yes | — |
| T4s | m2 vs m4 | greater | 65495 | 1.43e-85 | 0.0102 | [0.0099, 0.0107] | 0.999 | yes | — |

## Caveats
- T1/T2 are CROSS-DATASET comparisons: dataset and architecture scale are confounded; failure to reject is NOT evidence of equality.
- T3/T4 are the clean within-dataset architecture comparisons and carry the H1 decision.

## Verdict
- Decision rule: 'registers reduce outlier tokens' is supported iff BOTH T3 and T4 reject at alpha=0.05 with positive median difference.
- T3 reject=True (p=3e-08, median diff 0.0021); T4 reject=True (p=1.59e-43, median diff 0.0087).
- **'Registers reduce outlier tokens' is SUPPORTED** under the new rule.

## Register tokens (DINOv3)
- **m3** (M3 DINOv3-S (+reg, finetuned) · FunnyBirds): site-max median register norm 1118 vs site-max median patch norm 642; register tokens exceed the patch outlier threshold τ in 62% of (site, sample, register) triples.
- **m4** (M4 DINOv3-B (+reg, frozen bb) · ImageNet): site-max median register norm 6561 vs site-max median patch norm 1374; register tokens exceed the patch outlier threshold τ in 64% of (site, sample, register) triples.

## Files
- Arrays: `data/results/registers/e1_counts_m1_vit_small_fb.npz`, `e1_counts_vit_base_imagenet.npz` (reused), `e1_counts_m3_dinov3s_fb.npz`, `e1_counts_dinov3_base_imagenet.npz` (reused); stats `e1_stats_v2.json`; table `e1_per_site_table_v2.csv`.
- Figures: `figures/registers/e1_counts/e1_fraction_per_site_v2.{png,pdf}`, `e1_dinov3_registers_v2.{png,pdf}` (pdf copies in the paper's journal-figures).
