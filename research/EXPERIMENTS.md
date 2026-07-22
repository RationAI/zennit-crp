# Experiment registry — setups, hyperparameters, reconstruction

One entry per experiment family: authoritative config source, key
hyperparameters, exact reconstruction command. Update on every new experiment.
Convention: parameters live in tracked code/specs; this file indexes them and
records only what is not derivable from the repo.

## Environment

- venv: `UV_PROJECT_ENVIRONMENT=/home/claude/venvs/zennit-crp UV_LINK_MODE=copy uv sync`;
  invoke `/home/claude/venvs/zennit-crp/bin/python` directly (`uv run` deadlocks).
  Pod bounce wipes the venv + scratch; repo (NFS) persists.
- Storage roots: `.env` / `.env.example` → `ZENNIT_PERSIST_ROOT` (durable),
  `ZENNIT_SCRATCH_ROOT` (fast, wiped). `experiments/storage.py`. FV caches build
  on scratch, mirror to `data/crp_gallery_cache/` (hydrate on start).
- GPU: single A40 46 GB.

## Models / checkpoints

| tag | model | source |
|---|---|---|
| funny_birds | vit_small + linear head | `data/runs/finetune_vit_small_funny-birds-train-clean/2026-06-03_000556/best.pt` |
| dsprites | vit_small + linear head | `data/runs/finetune_vit_small_dsprites/2026-06-02_183230/best.pt` |
| colored_mnist | vit_small + linear head | `data/runs/finetune_vit_small_colored-mnist-train/2026-06-02_175921/best.pt` |
| imagenet | `vit_base_patch16_224` timm ImageNet-1k pretrained, builtin head | no local ckpt (`experiments/model_io.py` tag branch) |
| (planned) DINOv3 | `vit_large_patch16_dinov3` timm wrapper exists (`experiments/models/bases/vit_dinov3.py`) | NO finetuned head ckpt in repo — must locate or train |

FunnyBirds eval loads clean-only filter (29,330/50,000 kept). ImageNet gallery
subset: `n_per_class=10` over all classes (FV-index size + NFS constraints).

## LRP profiles

`lrp_configs/` registry — one knob per named recipe; `cp_lrp_baseline` is the
paper-default (AttnLRP-style CP-LRP: value-path propagation). Composite
hyperparameters are IN the config source (shown verbatim on the gallery page via
`composite.json`).

## CRP gallery (webapp/crp_gallery, served as zennit-crp-gallery)

- **Authoritative spec: `webapp/crp_gallery/jobs.jsonl`** (tracked) — one line
  per job, full parameter set; `replay` regenerates everything from it.
- Current 4 jobs: {vit_small/funny_birds, vit_base/imagenet} × {embed_dim,
  sae m=1536}, config `cp_lrp_baseline`, site `proj_drop`, blocks 0–11, n=5
  detectors/block, n_ref=6, mode=relevance, rank=class_conditional (n_rank=8),
  plot=heat_rf (3 rows: image | class-conditional relevance | RF crop).
- FV index: `Maximization SAMPLE_SIZE=40`, `RelMax_sum_normed`, built by
  `fv.run(composite, 0, len(ds), batch_size=32)`.
- **Method note**: reference heatmaps are **class-conditional**
  (`class_conditional_references` in `experiments/crp_gallery.py`) — deliberate
  deviation from upstream `get_max_reference` (which seeds from layer
  activation, no class condition; contradicts CRP paper; found+fixed 2026-07-20).
  RF-crop row: sign-safe normalization by max|R|, display-range fixed before fade.
- Reconstruct: `python -m experiments.crp_gallery replay --plot heat_rf --device cuda`
  (add `--dataset` to filter).

## Concept flipping (public/, served as zennit-flip)

- Code `experiments/concept_flipping.py`; per-run parameters in each run's
  meta.json under `data/results/concept_flipping/`; parquet holds per-step
  deltas → metric re-scoring possible WITHOUT recompute.
- Protocol: MoRF/LeRF detector flipping, prob-target, AOPC; per-(dataset,
  layer, concept-basis) curves. Figures via
  `experiments/scripts/export_flipping_figures.py`.
- Planned metric upgrade (XAI-21): random baseline (state K, cite Hama/Mase/
  Owen JMLR 2023 closed form), %-relevance x-axis for cross-basis (SAE vs raw)
  comparison — see `scout_benchmark_cav_datasets.md` §3 + Q11 caveats
  (signed relevance; SAE latent capture fraction ~2% under folded-bias γ —
  report capture fraction alongside or fix decoder rule first).

## SAE training + splice (data/results/sae)

- Code `experiments/sae.py` (train + load, `sae_path` naming); L1 SAEs at sites
  {proj_drop, residual} × 12 blocks × 3 datasets = 72 SAEs; m=1536 for
  vit_small (384-dim) — dict size in checkpoint filename; FVU range
  0.005–0.044 (2026-06-09 run).
- Splice = reconstruction passthrough, γ-rule on decoder (NOT ε as originally
  planned); `.features` sublayer exposes (B, N, m) codes for CRP.
- Known issue: only ~2% of site relevance reaches latents under folded-bias
  γ-rule (see `sae_crp_plan.md` + slack DONE entry) — bias-aware decoder rule
  is an open item before conservation-based claims.
- Downstream fidelity: `experiments/sae_downstream.py` → webapp/sae_downstream
  (served as sae-downstream). MUST be promoted to a paper table (SAE splice
  fidelity: logit delta / accuracy drop).

## Residual flow diagnostic (webapp/residual_flow, served as zennit-residual)

- Code `experiments/scripts/residual_flow_diag.py` (CLI: compute|render|all,
  `--n-samples --batch-size --device --config --dataset --base`).
- Run 2026-07-22: vit_small funny_birds ckpt above, `cp_lrp_baseline`, N=96
  class-diverse test images (round-robin 50 classes), true-class conditioning,
  probe acc on sample 0.979. Raw arrays:
  `data/results/residual_flow/residual_flow_vit_small_funny_birds_cp_lrp_baseline.npz`.
- Findings + method: `residual_lrp_notes.md` (2026-07-22 entry).
- TODO: rerun on vit_base/imagenet (same CLI, `--base vit_base --dataset imagenet`).

## Planned (registered, not run)

- CAV self-consistency: `cav_selfconsistency_protocol.md`.
- Occlusion faithfulness: Chefer pos/neg token-perturbation AUC + ROAD variant
  (`scout_benchmark_cav_datasets.md` §2).
- Localization: relevance-mass-in-mask on funny_birds parts; ImageNet-S300 /
  PartImageNet for natural images (§1).
- Conservation audit: extend residual_flow_diag bookkeeping (per-module
  absorption table).

## Paper planning

- Story + scope: YouTrack XAI-21; novelty verdict `scout_novelty_crp_vit.md`;
  must-read briefs `briefs/` (AttnLRP+CRP, SemanticLens+CaFE, IMPACT+PbE-R).
