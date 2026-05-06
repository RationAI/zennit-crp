# Experiments

Sweeps, audits, and one-off analyses that drove the design choices in
`crp.attention_concepts` and `crp.transformer_patches`. None of these are
prerequisites for using the library — they are research artefacts. The
narrative findings live in [`CURRENT_STATE.md`](../CURRENT_STATE.md);
each entry below points at its corresponding milestone there.

All scripts read and write under [`<repo>/data/`](../data) (gitignored).
Run them from the repo root via `uv run python experiments/<script>.py`.

## Shared machinery

* [`datasets/`](datasets/) — per-dataset loaders, all auto-downloading
  on first use. ``load("imagenette", ...)`` (10-class fast.ai subset,
  ~98 MB), ``load("imagenet_val_hf", ...)`` (un-gated HF mirror of full
  ImageNet-1k val, ~830 MB), ``load("funny_birds", ...)`` (synthetic
  birds with GT part maps, ~1.5 GB), ``load("dsprites", ...)`` (2D
  shapes, ~26 MB). Each yields a Dataset suitable for both standard
  training and FV indexing.
* [`metrics.py`](metrics.py) — Petsiuk deletion / insertion AUC,
  random-concept baseline, per-granularity top-k resolver, image-class
  iteration helper, ε / γ composite factory. Imported by every milestone
  driver and also runnable standalone as a single-config CLI.

## Sweep drivers

* [`run_milestone_a.py`](run_milestone_a.py) — Milestone A
  (`CURRENT_STATE.md` § "Milestone A"). γ-LRP sweep on
  `vit_base_patch16_224`. Builds a 4-class × 16-image curated subset of
  Imagenette val (idempotent symlink farm), then evaluates ε-LRP plus
  γ ∈ {0.0, 0.1, 0.25, 0.5} across all four concept granularities.
  Output: `data/milestone_a_results.csv`.

* [`aggregate_milestone_a.py`](aggregate_milestone_a.py) — turns the
  Milestone A CSV into a markdown table for the PR description / state
  doc. Output: `data/milestone_a_table.md`.

* [`run_milestone_d.py`](run_milestone_d.py) — Milestone D
  (`CURRENT_STATE.md` § "Milestone D"). Multi-model PA-LRP sweep across
  `vit_small`, `vit_base`, `vit_large` × {palrp off, on} on ε-LRP. Tests
  whether PA-LRP changes Petsiuk AUC (it doesn't — see milestone notes)
  and how the kqv_head failure scales with model depth. Output:
  `data/milestone_d_results.csv`.

* [`run_milestone_g.py`](run_milestone_g.py) — Milestone G
  (`CURRENT_STATE.md` § "Milestone G"). Same multi-model layout × {none,
  ratio} residual-LRP rules. The only configuration that fixes the
  kqv_head AUC anomaly at vit_small / vit_base. Output:
  `data/milestone_g_results.csv`.

## Diagnostics

* [`conservation_check.py`](conservation_check.py) — companion CLI to
  `tests/test_vit_integration.py::TestConservation`. Measures
  `R_input.sum() / target_logit` per (model, composite, ± PA-LRP) on a
  real Imagenette image to surface the conservation drift the LRP rules
  introduce. Useful when verifying a residual-LRP / PA-LRP change moved
  the ratio in the right direction.

## Conventions

* All scripts target ε-LRP as default and accept knobs (`--gamma`,
  `--palrp`, `--rules`) for the milestone-specific sweep variants.
* `--dataset {imagenette|imagenet_val_hf}` selects the source. Defaults
  to `imagenette` everywhere; switch to `imagenet_val_hf` for a final
  pre-PR benchmark run (un-gated HF mirror, auto-downloaded).
* `--n-per-class` and `--classes` further subset the dataset.
  `--n-per-class` defaults to **16** for imagenette (4 default classes ×
  16 imgs ≈ 64-image sweep, ~10 min on an A100 MIG slice) and **1** for
  imagenet_val_hf (1000 classes × 1 img = 1000-image class-balanced
  sweep, ~1 hour for ε-LRP on vit_base).
* Top-k is per-granularity by default: `head: 4, head_dim: 8,
  kqv_head: 8, kqv_head_dim: 8`. Override with `--top-k <int>` to apply
  one value (clamped to the granularity's concept count). Petsiuk-style
  union heatmaps need `k ≪ num_concepts` — see
  [`metrics.PER_GRANULARITY_TOP_K`](metrics.py).
* Long sweeps print per-image lines; pipe through `tee data/<run>.log` if
  you want to keep them. Set `PYTHONUNBUFFERED=1` for live progress.
