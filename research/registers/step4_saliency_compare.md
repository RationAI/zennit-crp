# Registers step 4 (XAI-38): do non-LRP saliency methods hotspot the same outlier patches?

Question (XAI-34 step 4, Vojtech): are the relevance hotspots on visually
unremarkable patches an **LRP artifact** or a **model-side** phenomenon (the
high-norm "register" tokens of Darcet et al., arXiv:2309.16588)?

Script: `experiments/scripts/registers_saliency_compare.py` (stages
`detect` / `saliency` / `analyze`; all implementations in-repo, no external XAI
libs). Arrays: `data/results/registers/step4_{selection,saliency}.npz`,
metrics: `step4_metrics.json`, scan stats: `step4_scan_stats.json`.
Figures: `figures/registers/step4_methods/{qualitative_panel,summary_metrics}.{png,pdf}`.

## Setup

- Model: timm `vit_base_patch16_224` (ImageNet-1k supervised), ImageNet val
  subset `n_per_class=10`; un-normalized [0,1] images + canonical normalize at
  the forward boundary (repo convention).
- **Outlier detection**: forward hooks on every `backbone.blocks[b]` output;
  per-image, per-block L2 norm of each of the 196 patch tokens (CLS excluded).
  Token = outlier at block *b* iff `norm > mean + 4·sd` over that image's 196
  patch norms at that block. Image mask = **union over blocks 6–11** (high-norm
  outliers emerge from block ~3 and peak late; the flagged sets are nearly
  identical across blocks 6–11 — union 3.53 vs 3.53 patches at block 8 alone —
  i.e. *persistent* tokens, exactly register-like behavior).
- Scan of 2048 images (top-1 acc 0.831): max token norm grows to ~150–164 at
  blocks 8–11 against per-block means of ~20–65; mean mask size 3.53 patches =
  **1.8% of tokens** (Darcet report ~2%). Essentially 100% of images carry ≥1
  outlier patch under this criterion, so "images WITH outliers" is not a
  restrictive filter for this model.
- Selection: **N=64** correctly-classified images with ≥1 outlier patch
  (first 64 qualifying in a seed-0 random scan order). Mask sizes 2–5 patches.

## Methods compared (per image, aggregated to the 14×14 grid by sum of |values| per 16×16 patch where pixel-level)

1. **gradient × input** (true-class logit, w.r.t. normalized input);
2. **integrated gradients** (32 midpoint steps, black baseline);
3. **attention rollout** (Abnar & Zuidema: head-averaged per-block attention,
   0.5·A + 0.5·I, row-renormalized, chained over all 12 blocks; CLS row).
   Attention captured on the *stock* timm module (`fused_attn=False`,
   `attn_drop` hooks; the unfolded-attention substitution only exists inside
   composite contexts);
4. **raw last-block CLS attention** (head-averaged, CLS row);
5. **LRP reference** — `CondAttribution` + `lrp_configs.get("cp_lrp_baseline")`
   (CP-LRP: value-path only, γ=0.10 linears), condition `[{"y":[target]}]`,
   |R| summed per patch.

## Colocation with the outlier mask (n=64; chance: concentration = 1, mean rank = 98.5, top-5 = 0.088)

| method | concentration ratio (mean / median) | mass in mask | mean rank of outlier patches | ≥1 outlier in top-5 |
|---|---|---|---|---|
| gradient × input | 3.50 / 2.08 | 5.8% | 42.4 | 0.42 |
| integrated gradients (32) | 1.30 / 1.03 | 2.3% | 85.0 | 0.17 |
| attention rollout | 1.92 / 1.87 | 3.4% | 11.3 | 1.00 |
| **raw last-block CLS attention** | **23.93 / 22.77** | **40.5%** | **2.4** | **1.00** |
| **LRP (cp_lrp_baseline)** | **14.58 / 14.01** | **25.3%** | **3.7** | **1.00** |

(Concentration ratio = fraction of saliency mass inside the outlier mask ÷ mask
area fraction; masks average 3.53/196 patches.)

## Verdict: **model-side, attention-mediated — not an LRP-specific artifact** (graded across method families)

- The **hardest concentrator is raw last-block CLS attention** — no LRP
  involved at all: on average **40% of the CLS attention mass** sits on ~1.8% of
  the patches, the outlier patches are the top-2.4 ranked patches on average,
  and every one of the 64 images has an outlier patch in the attention top-5.
  This is precisely Darcet et al.'s register phenomenon, reproduced on
  supervised ViT-B/16.
- **LRP inherits it** (concentration 14.6, mean rank 3.7, top-5 100%) — and
  notably concentrates *less* than raw attention. Under CP-LRP the attention
  weights are constants and relevance flows through the value path, so the LRP
  hotspots mean the *value content* of the register tokens genuinely carries
  class-evidence flow, amplified by the huge CLS attention on those tokens. LRP
  is faithfully reporting a real routing property of the model, not inventing it.
- **Attention rollout** is graded: mass is diffuse (concentration 1.9) because
  chaining 12 row-stochastic maps spreads mass, but the *ranking* still puts an
  outlier patch in the top-5 of every image (mean rank 11.3).
- **Pure input-gradient methods largely bypass the artifact**: gradient×input is
  weakly elevated (median concentration 2.1, top-5 42%), IG is statistically
  near chance (median 1.03, mean rank 85, top-5 17% vs 8.8% chance). Consistent
  with the register interpretation: those patches' *pixel content* is
  low-information, so ∂logit/∂pixel there is small even though the token's
  latent state is a global aggregate the classifier reads via attention.
- Qualitative panels (`qualitative_panel.png`) match: outlier patches sit on
  sky/snow/background; last-block attention and LRP light exactly those patches
  (red boxes), IG lights the object instead.

**Implication for XAI-34**: treating the hotspots requires model-side handling
(e.g. registers à la Darcet, or excluding/renormalizing outlier-token
relevance at read-out), not a change of LRP propagation rules — every
attention-reading view of this model shows the same hotspots; switching
attribution method only hides them by ignoring the attention structure
(IG/G×I), at the cost of not explaining the model's actual routing.

## Caveats

- Single model (supervised ViT-B/16), N=64, one outlier criterion
  (per-image mean+4σ, union blocks 6–11); signed maps rectified (|·|) before
  patch aggregation; IG uses a black baseline; LRP variant is CP-LRP with the
  repo's γ composite (`cp_lrp_baseline`), not AttnLRP.
- Concentration ratios for attention-family methods are mask-size dependent;
  ranks/top-5 are the more robust comparisons.
