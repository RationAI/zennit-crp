#!/usr/bin/env bash
# SAE-basis vs axis-aligned CRP × probe-site (proj_drop vs residual) study.
# Trains the SAEs, runs the 4 concept-flipping cases, renders the 4 grid figures.
# Default composite only (cp_lrp_baseline, the LXT value-path recipe).
set -euo pipefail
cd "$(dirname "$0")/../.."
# NOTE: `uv run` deadlocks on this GPFS-backed venv; invoke the interpreter directly.
export VIRTUAL_ENV="$PWD/.venv"
PY="$PWD/.venv/bin/python"

echo "=== 1/3 train SAEs (2 sites x 3 datasets x 12 blocks) ==="
$PY -m experiments.sae --datasets dsprites --datasets colored_mnist --datasets funny_birds \
    --site proj_drop --site residual --expansion 8 --l1-coeff 1e-3 --steps 3000

echo "=== 2/3 concept-flipping — 4 cases (embed_dim|sae × proj_drop|residual) ==="
# dsprites + colored_mnist: all classes
$PY -m experiments.concept_flipping --datasets dsprites --datasets colored_mnist \
    --config cp_lrp_baseline --concept embed_dim --concept sae \
    --site proj_drop --site residual --max-steps 48 --n-images 20 --perturbation zero
# funny_birds: first ten classes (typer needs the flag repeated per value)
$PY -m experiments.concept_flipping --datasets funny_birds \
    --config cp_lrp_baseline --concept embed_dim --concept sae \
    --site proj_drop --site residual --max-steps 48 --n-images 20 --perturbation zero \
    --classes 0 --classes 1 --classes 2 --classes 3 --classes 4 \
    --classes 5 --classes 6 --classes 7 --classes 8 --classes 9

echo "=== 3/3 render 4 grid figures ==="
$PY -m experiments.scripts.export_sae_site_figures --config cp_lrp_baseline

echo "=== DONE ==="
