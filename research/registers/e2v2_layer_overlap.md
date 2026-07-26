# E2v2 — layer-by-layer: block-b activation outliers ending up in the final saliency mask

Reviewer-ordered refinement of E2 (`e2_saliency_overlap.md`). Question (Adam):
layer-by-layer, how many of the outlier tokens identified at EACH block end up
significantly highlighted in the FINAL input saliency map — per XAI method,
per model+dataset.

Script: `experiments/scripts/registers_e2v2_layer_overlap.py`
(stages `norms` → `analyze` → `figures`, idempotent).
Data: `data/results/registers/e2v2_layer_norms_<model>.npz`,
`e2v2_layer_overlap_<model>.npz`, `e2v2_layer_table_<model>.csv`.
Figures: `figures/registers/e2v2_layer_overlap/e2v2_layer_overlap_<model>.{pdf,png}`
(copies in the paper `journal-figures/` dir). Run 2026-07-26.

## Definitions (exact, as implemented)

For image x with T = 196 patch tokens (224², /16; CLS excluded; DINOv3 n/a here):

- h_b(t) = residual-stream state of patch token t at the OUTPUT of block b
  (forward hook on `blocks[b]`), b = 0..11. Recorded in float32.
- Per-image per-block statistics: mu_b(x), sigma_b(x) = mean and std of
  { ||h_b(t)||_2 : t = 1..T }.
- Block-b activation outlier set: A_b(x) = { t : ||h_b(t)||_2 > mu_b(x) + 4·sigma_b(x) }.
- Saliency map s_m computed w.r.t. the model's PREDICTED class
  yhat(x) = argmax logit; all selected images are correctly classified, so
  yhat = y (conditioning stated as prediction for uniformity). Methods:
  - LRP: `cp_lrp_baseline` composite, `CondAttribution`, condition `[{"y": [yhat]}]`,
    full-model input heatmap;
  - Chefer transformer attribution (grad-weighted attention rollout, CVPR 2021;
    A captured on stock timm attention, `fused_attn=False`, hook on
    `attn.attn_drop` output; grad = d logit_yhat / dA);
  - attention rollout (Abnar & Zuidema; class-agnostic by nature);
  - occlusion (patch → image-mean color, Δp of the predicted class, positive part).
- Per-patch aggregation: P_m(p) = Σ over the 16×16 pixels (i,j) of patch p of |s_m(i,j)|.
  (Occlusion is natively patch-level: P_m(p) = Δp⁺(p).)
- Significant-highlight set: S_m(x) = { p : P_m(p) > mu^P_m(x) + 4·sigma^P_m(x) },
  mu^P/sigma^P over the image's own 196 values — the SAME 4σ rule as on the
  activation side, symmetric by design.
- Reported per (model, method m, block b): n_b = Σ_x |A_b(x)|,
  c_{b,m} = Σ_x |A_b(x) ∩ S_m(x)|, q_{b,m} = c_{b,m}/n_b where n_b > 0.

Models/data: same N=64 correctly-classified, class-diverse images as E2
(`e2_select_<model>.npz` indices) — ViT-B/16 · ImageNet val and
ViT-S/16 · FunnyBirds test (probe ckpt).

Reused vs recomputed: all four saliency per-patch arrays and masks S_m are
REUSED unchanged from `e2_overlap_<model>.npz` (occlusion maps not recomputed);
`analyze` re-derives S_m from the stored per-patch values and asserts identity.
Block-output norms were recomputed in one cheap fp32 forward sweep per model
(E2 stored them only in fp16); the fp32 A_b flags match the stored fp16 site
flags (site 2b+1) exactly — 0 mismatches of 150 528 per model.

## Table — ViT-B/16 · ImageNet val (N=64)

| block | n_b | LRP c (q) | Chefer c (q) | rollout c (q) | occlusion c (q) |
|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 0 (0.00) | 0 (0.00) | 0 (0.00) | 0 (0.00) |
| 1 | 15 | 0 (0.00) | 0 (0.00) | 1 (0.07) | 2 (0.13) |
| 2 | 35 | 29 (0.83) | 20 (0.57) | 20 (0.57) | 1 (0.03) |
| 3 | 120 | 112 (0.93) | 61 (0.51) | 64 (0.53) | 1 (0.01) |
| 4 | 155 | 133 (0.86) | 70 (0.45) | 70 (0.45) | 1 (0.01) |
| 5 | 195 | 148 (0.76) | 71 (0.36) | 70 (0.36) | 1 (0.01) |
| 6 | 223 | 151 (0.68) | 71 (0.32) | 70 (0.31) | 1 (0.00) |
| 7 | 224 | 151 (0.67) | 71 (0.32) | 70 (0.31) | 1 (0.00) |
| 8 | 225 | 151 (0.67) | 71 (0.32) | 70 (0.31) | 1 (0.00) |
| 9 | 224 | 151 (0.67) | 71 (0.32) | 70 (0.31) | 1 (0.00) |
| 10 | 184 | 140 (0.76) | 69 (0.38) | 68 (0.37) | 1 (0.01) |
| 11 | 122 | 108 (0.89) | 58 (0.48) | 58 (0.48) | 1 (0.01) |

Totals over all blocks (Σc / Σn): LRP 1274/1728 = 0.74, Chefer 633/1728 = 0.37,
rollout 631/1728 = 0.37, occlusion 12/1728 = 0.007.

## Table — ViT-S/16 · FunnyBirds test (N=64)

| block | n_b | LRP c (q) | Chefer c (q) | rollout c (q) | occlusion c (q) |
|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 5 (0.33) | 2 (0.13) | 1 (0.07) | 5 (0.33) |
| 1 | 7 | 1 (0.14) | 0 (0.00) | 0 (0.00) | 1 (0.14) |
| 2 | 7 | 1 (0.14) | 0 (0.00) | 0 (0.00) | 1 (0.14) |
| 3 | 142 | 0 (0.00) | 3 (0.02) | 0 (0.00) | 0 (0.00) |
| 4 | 222 | 0 (0.00) | 3 (0.01) | 0 (0.00) | 0 (0.00) |
| 5 | 262 | 0 (0.00) | 3 (0.01) | 0 (0.00) | 0 (0.00) |
| 6 | 330 | 0 (0.00) | 3 (0.01) | 0 (0.00) | 0 (0.00) |
| 7 | 336 | 0 (0.00) | 3 (0.01) | 0 (0.00) | 0 (0.00) |
| 8 | 337 | 0 (0.00) | 3 (0.01) | 0 (0.00) | 0 (0.00) |
| 9 | 337 | 0 (0.00) | 3 (0.01) | 0 (0.00) | 0 (0.00) |
| 10 | 284 | 0 (0.00) | 3 (0.01) | 0 (0.00) | 0 (0.00) |
| 11 | 248 | 0 (0.00) | 3 (0.01) | 0 (0.00) | 0 (0.00) |

Totals: LRP 7/2527 = 0.003, Chefer 29/2527 = 0.011, rollout 1/2527 = 0.000,
occlusion 7/2527 = 0.003.

## Observations

- Both models grow their activation outliers in the same window: n_b is tiny at
  blocks 0–1(–2), ramps steeply at block 3, plateaus over blocks 6–9
  (~224/image·64 for ViT-B, ~337 for ViT-S), and shrinks again toward block 11.
  Every one of the 64 images contributes outliers at the plateau
  (mean |A_8| = 3.5 ViT-B, 5.3 ViT-S).
- ViT-B/ImageNet: LRP picks up the mid-network outliers essentially as soon as
  they appear — q peaks exactly at the onset blocks 2–3 (0.83–0.93) and stays
  ≥ 0.67 everywhere from block 2 on. Of the union-outlier tokens LRP ends up
  highlighting, ALL are first flagged at blocks 2–6 (29/84/21/14/3 at blocks
  2/3/4/5/6); tokens first flagged at blocks 0–1 are never highlighted. The
  outliers LRP misses are dominated by later-onset tokens (blocks 4–6). Of the
  184 persistent tokens (flagged at ≥ 6 block outputs), 144 (78%) end in the LRP
  mask — the persistent register-like tokens are exactly what LRP lights up.
- Chefer and rollout track each other almost identically (c within ±2 at every
  block) at roughly half LRP's rate (~0.31–0.57): the grad-weighting adds almost
  nothing on top of raw rollout for these tokens. Occlusion is flat ≈ 0
  everywhere — flipping the pixels under an outlier token barely moves the
  predicted-class probability, consistent with E2 and with the
  register/scratch-pad reading (the activation outlier is not caused by the
  local patch content).
- ViT-S/FunnyBirds is the mirror image: from block 3 onward NO method highlights
  the outliers (LRP literally 0 of 2 425 tokens flagged at blocks 3–11;
  0 of the 272 persistent tokens are in S_lrp). The only hits are a handful of
  early-block (block-0) outliers — plausibly genuine content outliers, not
  registers. Chefer's constant c = 3 across blocks 3–11 is just 3 persistent
  (image, patch) pairs (images 18 and 58, corner-region patches 15/26).
- Together: whether a method "sees" the register tokens is model/dataset-
  dependent, not method-intrinsic — the same LRP recipe that inherits 74% of
  ViT-B/ImageNet outliers into its final mask inherits ~0% on ViT-S/FunnyBirds.
  Consistent with E2's per-image IoU story, now localized in depth: for ViT-B
  the outliers that end up in the LRP map are the early-onset (blocks 2–3),
  persistent ones.

Caveats: 4σ threshold on only 196 values is a hard cut — counts near the
threshold move with small numeric changes (fp32 used throughout here);
n_b totals differ per model, so compare q, not raw c, across models.
