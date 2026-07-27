# Residual relevance flow — 4-model redo (reviewer-ordered)

2026-07-27 · `experiments/scripts/residual_flow_diag.py` (extended with
`--checkpoint` / `--model-tag`, no fork) · config `cp_lrp_baseline` ·
N=96 class-diverse **correctly-classified** images per model (round-robin over
classes, seed 0, scanned 128 candidates → kept 96 for every model) ·
true-class conditioning `{"y": [target]}`.

## Models

| tag | model / data | checkpoint | blocks · D · tokens (prefix) |
|---|---|---|---|
| m1_vit_small_fb | vit_small / FunnyBirds test | `data/runs/finetune_vit_small_funny-birds-train-clean/2026-07-26_160337/best.pt` (pinned via `--checkpoint`; asserted == newest-run glob) | 12 · 384 · 197 (1) |
| m2_vit_base_in | vit_base / ImageNet val | `timm:vit_base_patch16_224` pretrained | 12 · 768 · 197 (1) |
| m3_dinov3s_fb | vit_dinov3_small / FunnyBirds test | `data/runs/finetune_vit_dinov3_small_funny-birds-train-clean/2026-07-25_200008/best.pt` | 12 · 384 · 261 (5) |
| m4_dinov3b_in | vit_base_patch16_dinov3.lvd1689m (img_size=256) + canvit IN1k CLS linear head / ImageNet val | `timm:vit_base_patch16_dinov3.lvd1689m + hf:canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe` | 12 · 768 · 261 (5) |

M4 assembly: one `nn.Module` whose forward =
`head(backbone.forward_features(x)[:, 0])` (final-norm CLS token → 1000
logits), so `CondAttribution` sees a single classifier. Input 256².

## Methods deltas vs the previous single-model (vit_small) run

1. **Eva (DINOv3) branch endpoints.** `EvaBlockResidualCanonizer`
   (already in `cp_lrp_baseline`, `layerscale_uniform=True`) routes
   `out = _lrp_res{1,2}(x, drop_path(LayerScaleMul(branch)))`. Because the
   LayerScaleMul carries the Uniform(factor=2) rule (absorbs half the
   relevance *below* it), the branch endpoints `attn.proj_drop` / `mlp.drop2`
   used for timm `Block` are WRONG for Eva — the branch summand of the
   elementwise residual split is the **LayerScaleMul output**, i.e.
   `backbone.blocks.{b}._lrp_ls1` (attn) and `._lrp_ls2` (mlp). Endpoint
   identity checked per Eva model as `max|R(drop_path{i}) − R(_lrp_ls{i})|`
   on blocks 0 and 11 (i = 1, 2); timm models keep
   `max|R(attn.proj_drop) − R(ls1)|` and `|R(mlp.drop2) − R(ls2)|`.
   **Endpoint-identity check = 0.0 exactly for all four models.**
2. **Token rows.** Per-dim token sums (signed + absolute) now use **patch
   tokens only** — prefix rows excluded: cls (M1/M2), cls + 4 register tokens
   (M3/M4). `tot_add` (used for conservation drift) keeps all rows. The old
   vit_small run summed all 197 rows incl. cls; recorded in npz meta
   (`num_prefix_tokens_excluded`, `token_rows`).
3. **Sample selection.** Now correctly-classified only (accuracy on sample =
   1.0 by construction for all four); previous run took any class-diverse
   images and reported the accuracy.
4. R_skip = R_add − R_branch stays exact (elementwise `ResidualRatio` split);
   no change.

## Per-model summary (mass-weighted total branch share F per site, 24 sites)

| tag | F min / median / max over sites | % sites F < 0.5 | median per-dim f range | drift within-block med/max | drift attn-side med/max |
|---|---|---|---|---|---|
| m1_vit_small_fb | 0.000 / 0.326 / 0.598 | 91.7 | 0.00–0.95 | 0.023 / 0.172 | 0.020 / 0.083 |
| m2_vit_base_in  | 0.000 / 0.302 / 0.549 | 91.7 | 0.00–0.95 | 0.042 / 0.222 | 0.024 / 0.107 |
| m3_dinov3s_fb   | 0.157 / 0.248 / 0.821 | 95.8 | 0.00–0.92 | 0.0016 / 0.590 | 0.0013 / 0.849 |
| m4_dinov3b_in   | 0.000 / 0.264 / 0.428 | 100.0 | 0.00–0.98 | 0.048 / 5.301 | 0.024 / 3.649 |

Drift = |Δ total relevance| between consecutive cuts relative to final-block
total (Gamma-rule bias absorption; property of the LRP recipe, not of the
skip/branch split). M4 medians are ordinary but a few samples have heavy-tail
drift (max 5.3); M3 likewise (max 0.85 at median 0.001).

Readings: skip dominates almost everywhere (median F ≈ 0.25–0.33 across
models). All models except M3 collapse to F ≈ 0 at blk 11 (attn+mlp): their
heads read the CLS token only, so last-block *patch* rows carry no branch
relevance. M3 does NOT collapse (blk 11 attn F = 0.82): the finetuned
dinov3_small probe's `extract_cls` goes through timm `forward_head`
(`global_pool='avg'` on Eva) — the head actually reads **mean patch tokens**,
so last-block patch rows stay relevant. Worth remembering when comparing
M3 vs M4 (M4's canvit head reads the true CLS token).

## Outputs

- Arrays: `data/results/residual_flow/residual_flow_{tag}_cp_lrp_baseline.npz`
  (4) + `rf_summary.json`.
- Figures: `figures/residual_flow/{tag}_branch_fraction_by_dim.{pdf,png}`,
  `{tag}_site_summary.{pdf,png}`, `rf_compare_models.{pdf,png}`; copies at
  `crp-paper/iclr2026/journal-figures/rf_{tag}_by_dim.pdf`,
  `rf_{tag}_sites.pdf` (8) + `rf_compare_models.pdf`.
- Web: `webapp/residual_flow/{tag}.html` (self-contained bokeh, INLINE
  resources) + `index.html` dropdown/iframe selector (default M1).
  Verified `https://claude-bajger.dyn.cloud.e-infra.cz/zennit-residual/`
  → 401 (auth wall up).

## Deviations / notes

- Benign `crp` warning "Some layer names not found" per run: endpoint-check
  layer names duplicate entries already in the record list; hooks register
  once and the script's own missing-layer check passed.
- `--model-tag` also parametrizes the bokeh page name (`--page-name`);
  `residual_flow_static_figures.py` gained `--stem-prefix` and a
  `--compare LABEL=NPZ ...` mode (no fork).
