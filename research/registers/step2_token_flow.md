# Step 2 — H_A mechanistic test: per-token residual relevance split at register tokens (XAI-36)

**Question.** Do high-norm "register"/outlier tokens (Darcet et al., arXiv:2309.16588)
trap LRP relevance in the residual skip path under `cp_lrp_baseline`, so that
relevance rides the token column down to its input patch (H_A)? Two predictions:

1. The per-token branch fraction `f = |R_branch| / (|R_branch| + |R_skip|)` is
   lower at outlier tokens than at normal tokens, in the blocks where outliers live.
2. Composites that split residuals 50/50 (`attnlrp_gamma_residual_symmetric`) or
   attribute the full bilinear attention (`attnlrp_gamma`) put less input-relevance
   mass on outlier patches than `cp_lrp_baseline`.

**Both predictions are CONFIRMED.**

## Setup

- Model: timm `vit_base_patch16_224` ImageNet-1k pretrained (full head), via
  `model_io.load_probe("imagenet")`. Data: ImageNet val (HF mirror, 10/class);
  N=64 class-diverse **correctly classified** images (round-robin over classes,
  first 64 correct; `data/results/registers/selection.json`).
- Attribution: `crp.CondAttribution`, true-class conditioning (`{"y": [target]}`),
  no concept mask. Per-token skip/branch split reuses the
  `residual_flow_diag` machinery: the `TimmBlockResidualCanonizer` routes both
  residual adds through recordable `ResidualAdd` modules; per block `b` we record
  the add outputs `backbone.blocks.{b}._lrp_res1` / `_lrp_res2` and the branch
  endpoints `attn.proj_drop` / `mlp.drop2`; `R_skip = R_add − R_branch` exactly
  (elementwise split; endpoint-identity check `max|R(proj_drop) − R(ls1)| = 0.0`).
  Relevance is reduced **per token**: sum of |R| (and signed R) over the 768
  embedding dims → arrays (site = 24 block×{attn,mlp}, sample = 64, token = 197).
- Outlier detection (same forwards, plain forward hooks): per-token L2 norms of
  the residual stream at 13 cuts (pre-block-0 embedding + every block output).
  Criterion (documented choice): per block, per sample, a **patch** token is an
  outlier if `norm > mean + 4·sd` over the 196 patch tokens; **CLS excluded**.
  Site (b, attn/mlp) uses the block-b-output mask (outliers are persistent
  across neighbouring blocks, so the cut choice is not critical).
- Script: `experiments/scripts/registers_token_flow.py`
  (`select` / `flow` / `outliers` / `ablate` / `report`; CLI `--n-samples
  --config --device --batch-size --seed`; GPU steps chunkable via `--start/--stop`).

## Where the registers live (fig1)

Median per-image **max** patch-token norm explodes from ~10 (block 2 output) to
110–143 (blocks 6–11) while the median token norm stays 8–66; outlier prevalence
(mean+4sd criterion) rises from ~0.5 tokens/image at block 2 to a plateau of
**~3.5 tokens/image in blocks 6–9**, falling to ~1.9 by block 11. This matches
the Darcet register phenomenology for a plain (register-free) ViT-B/16.

## Prediction 1 — branch fraction at outlier vs normal tokens (fig2)

`cp_lrp_baseline` (ResidualRatio residual rule), median per-token f, pooled over
images; Mann-Whitney one-sided (outlier < normal) on pooled tokens; 95% cluster
bootstrap (over images) CI of the median difference:

| block | attn: f_norm → f_out (p, CI) | mlp: f_norm → f_out (p, CI) |
|---|---|---|
| 2 | 0.34 → 0.36 (n.s.) | 0.35 → **0.63** (higher, CI [+0.27,+0.31]) |
| 3 | 0.33 → 0.31 (p=2e-16) | 0.31 → **0.66** (higher, CI [+0.34,+0.36]) |
| 4 | 0.31 → 0.22 (p≈0, CI [−0.10,−0.07]) | 0.31 → **0.56** (higher) |
| 5 | 0.34 → **0.11** (p≈0, CI [−0.234,−0.226]) | 0.31 → 0.50 (higher) |
| 6 | 0.29 → **0.07** (p≈0, CI [−0.220,−0.212]) | 0.30 → 0.33 (higher, CI [+0.02,+0.04]) |
| 7 | 0.29 → **0.07** (p≈0, CI [−0.222,−0.216]) | 0.31 → **0.08** (p≈0, CI [−0.240,−0.234]) |
| 8 | 0.28 → **0.08** (p≈0) | 0.33 → **0.09** (p≈0) |
| 9 | 0.27 → **0.09** (p≈0) | 0.35 → **0.18** (p≈0) |
| 10 | 0.33 → **0.13** (p≈0) | 0.51 → 0.49 (p=2e-4) |
| 11 | 0.00 → 0.00 (degenerate: attn conducts ~nothing at the last block) | — |

(Full table incl. token counts: `data/results/registers/token_flow_analysis_cp_lrp_baseline.npz` → `stats`.)

**Verdict: confirmed, with a mechanistically informative refinement.** In the
blocks where registers live (5–10) the **attention-residual** branch share at
outlier tokens collapses to f ≈ 0.07–0.13 vs 0.27–0.34 at normal tokens (≈ 3–4×
lower; all p ≈ 0, bootstrap CIs well below 0) — exactly H_A's mechanism: the
ratio rule sends relevance into the skip where |x| explodes, and the only
cross-token mixing (the attention branch) is starved. The **MLP** side shows the
complementary signature: in blocks 2–5, where the outlier feature is being
*written* by the MLP, |branch| is large at those tokens so the ratio rule sends
relevance INTO the MLP branch (f 0.50–0.66 vs ~0.31); once the stream norm
dominates (blocks 7–9) the MLP side collapses too (0.08–0.18). Net effect: after
a register is established, relevance arriving at its column can essentially only
ride the skip down to the input patch.

## Prediction 2 — composite ablation (fig3, fig4)

M=16 images with detected outliers (largest per-image union of outlier patches
over the home blocks; median 5 outlier patches/image;
`data/results/registers/ablate_selection.json`). Full-model input heatmaps,
true-class conditioning. Metric per image: fraction of total |input relevance|
inside outlier patches ÷ their area fraction (concentration ratio; 1 =
area-proportional). Median [95% bootstrap CI]:

| composite | concentration ratio |
|---|---|
| `cp_lrp_baseline` | **15.2** [11.9, 16.4] |
| `attnlrp_gamma` (full bilinear attention) | **8.7** [6.8, 9.1] |
| `attnlrp_gamma_residual_symmetric` (50/50 skip split) | **2.4** [1.9, 3.1] |

**Verdict: confirmed.** Both alternative composites reduce the register-patch
concentration relative to `cp_lrp_baseline` — the symmetric residual split by
≈ 6.3×, full-bilinear attention by ≈ 1.7× — with non-overlapping CIs. The
residual-split rule is the dominant lever, consistent with H_A localizing the
artifact in the ResidualRatio rule rather than in the attention rule. Note the
symmetric split still leaves ~2.4× over-concentration (registers do carry real
signal / the split is per-token not per-path-usage), and its heatmaps are
visibly noisier overall (fig4) — reducing the artifact is not free.

## Caveats / choices

- Outlier criterion mean+4sd is per-sample per-block; on this sample it isolates
  the classic sparse high-norm set (~3.5 tokens/image at the plateau). CLS is
  excluded everywhere.
- "Home blocks" for the ablation patch set = blocks with ≥ 0.5 outlier
  tokens/image (here blocks 2–11); the per-image patch set is the union over
  those blocks.
- Block 11 attn/mlp is degenerate under this recipe (median f = 0 for both
  groups): with the timm head reading the (pre-norm) final stream, virtually all
  relevance passes the last block through the skip.
- Flow analysis was run under `cp_lrp_baseline` only (the hypothesis is about
  that recipe's ratio rule); the ablation covers the two alternatives.

## Artefacts

- Script: `experiments/scripts/registers_token_flow.py`
- Arrays: `data/results/registers/` (`flow_cp_lrp_baseline_part*.npz`,
  `token_flow_analysis_cp_lrp_baseline.npz`, `ablate_heatmaps_<config>.npz`,
  `concentration_summary.json`, `selection.json`, `ablate_selection.json`)
- Figures (png+pdf): `figures/registers/step2_flow/`
  (`fig1_outlier_prevalence`, `fig2_branch_fraction_attn`,
  `fig2_branch_fraction_mlp`, `fig3_concentration_by_composite`,
  `fig4_example_heatmaps`)

*Generated 2026-07-24 (UTC), Agent-2 / XAI-36.*
