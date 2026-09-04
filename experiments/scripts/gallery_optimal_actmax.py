"""Aggregate (consensus) activation-max gallery entries for the concept detectors
the heuristic-optimal search flagged as most important, per layer.

Model-agnostic: works for any (model, dataset) that has a heuristic npz
``data/results/benchmark/cdet_dapc_<model>_<dataset>__optimal.npz`` produced by
experiments.concept_detector_optimal (method "optimal", greedy). The consensus
top-k detectors per (site, block) = mean-rank over the ranked images. Each is
rendered as a gallery act-max entry under the labeled config
``cp_lrp_baseline_optimal_actmax`` (see crp_gallery.COMPOSITES / _CONFIG_LABELS).

The heuristic's 4 sites map to gallery --site:
    residual->residual, proj_drop->proj_drop, qk->query, value->value.

Two phases:
    build   — ONE FV-index pass over the dataset recording ALL 48 layers
              (4 sites x 12 blocks) at once — RelMax+ActMax sum stores; no entries.
    render  — one gallery entry per (site, block) for that block's consensus top-k
              (index reused, GPU-light; --no-samples skips per-sample saliency).

Usage::
    uv run python -m experiments.scripts.gallery_optimal_actmax --model vit_base --dataset imagenet --phase build
    uv run python -m experiments.scripts.gallery_optimal_actmax --model vit_base --dataset imagenet --phase render --k 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD = "optimal"
CONFIG = "cp_lrp_baseline_optimal_actmax"

# heuristic npz site -> gallery --site (SITE_LAYERS key)
SITE_MAP = {"residual": "residual", "proj_drop": "proj_drop", "qk": "query", "value": "value"}
BLOCKS = list(range(12))


def npz_path(tag: str) -> Path:
    return REPO_ROOT / f"data/results/benchmark/cdet_dapc_{tag}__optimal.npz"


def n_images(d) -> int:
    """Actual #images ranked per (site, block) — may be fewer than meta image_ids
    (e.g. M2/imagenet ranked 4/combo, not the full candidate list)."""
    import re
    js = [int(m.group(1)) for k in d.files
          if (m := re.match(rf"{METHOD}__residual__b0__img(\d+)__order", k))]
    return max(js) + 1 if js else 0


def consensus_top_k(d, D, nimg, npz_site, b, k):
    orders = np.stack([d[f"{METHOD}__{npz_site}__b{b}__img{j}__order"] for j in range(nimg)])
    ranks = np.empty_like(orders)
    for j in range(nimg):
        ranks[j, orders[j]] = np.arange(D)
    return [int(i) for i in np.argsort(ranks.mean(0))[:k]]


def phase_build(args, base, dataset, tag):
    """One FV pass over the whole pool recording all 48 layers at once."""
    import torch  # noqa: F401
    from experiments.crp_gallery import make_concept, SITE_LAYERS, COMPOSITES
    from experiments.gradinput import (
        GradTimesInputAttribution, GradTimesInputFeatureVisualization)
    from experiments.model_datasets import find

    device = args.device
    mdset = find(base, dataset, device=device)
    model, normalize, ds = mdset.model, mdset.normalize, mdset.dataset
    num_heads = model.backbone.blocks[0].attn.num_heads
    concept = make_concept("embed_dim", num_heads)
    attribution = GradTimesInputAttribution(model)
    comp_cls = COMPOSITES[CONFIG]
    layer_names = [SITE_LAYERS[SITE_MAP[s]][b] for s in SITE_MAP for b in BLOCKS]
    fv_dir = REPO_ROOT / "data/crp_gallery_cache/fv" / tag / CONFIG
    print(f"build: {len(layer_names)} layers over {len(ds)} images -> {fv_dir}")
    if args.dry_run:
        return
    fv = GradTimesInputFeatureVisualization(
        attribution, ds, {ln: concept for ln in layer_names},
        preprocess_fn=normalize, path=str(fv_dir), device=device, negative_clamp=True)
    end = len(ds) if not args.fv_end else min(args.fv_end, len(ds))
    fv.run(comp_cls(), 0, end, batch_size=args.batch_size)
    print(f"build done: index at {fv_dir}")


def _render_cmd(base, dataset, gsite, b, detectors, n_ref, device):
    cmd = [sys.executable, "-m", "experiments.crp_gallery", "compute",
           "--base", base, "--dataset", dataset, "--config", CONFIG,
           "--site", gsite, "--blocks", str(b), "--concept", "embed_dim",
           "--mode", "activation", "--fv-class", "original", "--rank", "fv_index",
           "--n", "0", "--n-ref", str(n_ref), "--plot", "heat_rf",
           "--no-samples", "--device", device]
    for det in detectors:
        cmd += ["--detectors", str(det)]
    return cmd


def phase_render(args, base, dataset, tag):
    d = np.load(npz_path(tag), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    D, nimg = meta["D"], n_images(d)
    for npz_site, gsite in SITE_MAP.items():
        for b in BLOCKS:
            dets = consensus_top_k(d, D, nimg, npz_site, b, args.k)
            cmd = _render_cmd(base, dataset, gsite, b, dets, args.n_ref, args.device)
            print(f"RENDER {gsite} b{b} top{args.k}={dets}")
            if not args.dry_run:
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="vit_small", help="model axis (M_* value)")
    ap.add_argument("--dataset", default="funny_birds", help="dataset axis (DS_* value)")
    ap.add_argument("--phase", choices=["build", "render"], required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-ref", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fv-end", type=int, default=0, help="cap FV pool (0=full)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tag = f"{args.model}_{args.dataset}"
    if not npz_path(tag).is_file():
        sys.exit(f"heuristic npz not found: {npz_path(tag)}")
    if args.phase == "build":
        phase_build(args, args.model, args.dataset, tag)
    else:
        phase_render(args, args.model, args.dataset, tag)


if __name__ == "__main__":
    main()
