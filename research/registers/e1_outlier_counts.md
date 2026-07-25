# E1 — register-outlier detection: half-block counts, standard ViT vs DINOv3
_Generated 2026-07-25 18:55 UTC by `experiments/scripts/registers_e1_counts.py` (collect / analyze / figures / report)._

## Experiment card
**RQ.** How many high-norm outlier (register/scratch-pad) tokens appear after each attention half-block and each MLP half-block, and do DINOv3 backbones with built-in register tokens contain fewer of them than standard ViTs?

**H1.** DINOv3 backbones (register tokens present) contain substantially fewer patch-token outliers than standard AugReg ViTs at every depth. **H0.** comparable fractions. **Falsified if** DINOv3 patch-outlier plateau fractions are within a factor ~2 of the standard ViTs'.

**Criterion** (journal entry Ic, per-sample): at each of the 24 sites, μ/σ over that sample's patch-token L2 norms; outlier iff norm > μ + 4σ. CLS excluded everywhere; DINOv3's 4 register tokens are excluded from patch statistics and tracked separately.

## Method
- Sites: `blocks[b].norm2` INPUT = state after the attn residual add (forward pre-hook); `blocks[b]` OUTPUT = state after the MLP residual add. 12 blocks × 2 = 24 sites per model, network order.
- Site identity was verified numerically per model on the first batch: block output equals `norm2_input + ls2/γ₂·mlp(norm2(norm2_input))` (max abs deviation stored in each npz's meta; both timm `Block` (M1/M2) and `EvaBlock` (DINOv3, LayerScale `gamma_2`, rotary pos-emb) follow this structure).
- Models: ViT-S/16 FunnyBirds probe (test split), ViT-B/16 timm ImageNet val (n_per_class=10 pool), DINOv3 ViT-S/16 and ViT-B/16 timm pretrained backbones run headless via `forward_features` (norms only — no classification head needed for detection); DINOv3 preprocessing from timm `resolve_data_config` (256×256 → 256 patches vs 197-token 224×224 for the standard ViTs).
- N=256 images/model, seed 0, round-robin class-diverse; indices persisted in the npz. The two FunnyBirds models see the same images, likewise the two ImageNet models.
- Plateau % = mean outlier fraction over blocks [8, 9, 10, 11] of the given half (attn sites / MLP sites).

## Decision table
| model | plateau % (attn sites) | plateau % (MLP sites) | any-site union % | total flagged tokens | per-image mean | per-image min..max | register norms (DINOv3) |
|---|---|---|---|---|---|---|---|
| ViT-S (std) | 2.549 | 2.427 | 2.987 | 1499 | 5.9 | 4..12 | — |
| ViT-B (std) | 1.708 | 1.564 | 2.045 | 1026 | 4.0 | 2..12 | — |
| DINOv3-S (+reg) | 0.093 | 0.149 | 1.675 | 1098 | 4.3 | 0..16 | median reg 1174 vs patch 518 (site max); reg>τ in 100% of samples (plateau) |
| DINOv3-B (+reg) | 0.018 | 0.107 | 1.097 | 719 | 2.8 | 0..11 | median reg 6561 vs patch 1374 (site max); reg>τ in 100% of samples (plateau) |

## Verdict
- min(standard)/max(DINOv3) plateau ratio: attn sites 18.3×, MLP sites 10.5× (falsification bound: factor ~2).
- **H1 SUPPORTED**: DINOv3 patch-outlier plateaus are well beyond a factor 2 of the standard ViTs'.

## Files
- Arrays: `data/results/registers/e1_counts_<model>.npz` (norms 24×N×T, flags, τ, indices, criterion params), `e1_analysis.json`, `e1_per_site_table.csv` (24 sites × 4 models).
- Figures: `figures/registers/e1_counts/e1_fraction_per_site.{png,pdf}`, `e1_dinov3_registers.{png,pdf}` (copies in the paper's journal-figures).

## Per-site table
See `data/results/registers/e1_per_site_table.csv`; headline figure `e1_fraction_per_site` plots the same 24×4 numbers.
