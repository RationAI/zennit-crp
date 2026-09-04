"""Populate the CRP gallery with activation-maximization entries for the concept
detectors the heuristic-optimal search flagged as most important, per layer.

Model M1 = vit_small_funny_birds. The heuristic (experiments.concept_detector_optimal,
greedy "optimal") stores, per (site, block, image), a MoRF removal order over the
384 embedding channels. We take the *consensus* top-k detectors per (site, block)
(mean rank across the 16 test images) and render each as a gallery act-max entry
(reference images ranked by ActMax), reusing one FV index.

Sites: the heuristic's 4 sites map to gallery --site as
    residual -> residual, proj_drop -> proj_drop, qk -> query, value -> value.

Two phases (so the GPU-heavy index build can run in one window, e.g. a grid pause):
    build   — one FV index pass per site over all 12 blocks (no entries rendered)
    render  — one gallery entry per (site, block) for that block's consensus top-k
              (index already built -> cheap; --no-samples skips per-sample saliency)

Usage::
    uv run python -m experiments.scripts.gallery_optimal_actmax --phase build
    uv run python -m experiments.scripts.gallery_optimal_actmax --phase render --k 5
    uv run python -m experiments.scripts.gallery_optimal_actmax --phase render --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NPZ = REPO_ROOT / "data/results/benchmark/cdet_dapc_vit_small_funny_birds__optimal.npz"
METHOD = "optimal"                 # greedy (Variant A); the first, complete heuristic
BASE, DATASET, CONFIG = "vit_small", "funny_birds", "cp_lrp_baseline"

# heuristic npz site  ->  gallery --site
SITE_MAP = {
    "residual":  "residual",
    "proj_drop": "proj_drop",
    "qk":        "query",     # q_lrp_probe
    "value":     "value",     # v_lrp_probe
}
BLOCKS = list(range(12))


def _load():
    d = np.load(NPZ, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return d, meta["D"], len(meta["image_ids"])


def consensus_top_k(d, D, nimg, npz_site: str, b: int, k: int) -> list[int]:
    """Consensus most-important detectors for (site, block): invert each image's
    MoRF order to a rank vector, average across images, take the k smallest
    (= most-important-first)."""
    orders = np.stack([d[f"{METHOD}__{npz_site}__b{b}__img{j}__order"] for j in range(nimg)])
    ranks = np.empty_like(orders)
    for j in range(nimg):
        ranks[j, orders[j]] = np.arange(D)
    return [int(i) for i in np.argsort(ranks.mean(0))[:k]]


def _compute_cmd(gallery_site: str, blocks: list[int], detectors: list[int],
                 *, n_ref: int, device: str) -> list[str]:
    cmd = [sys.executable, "-m", "experiments.crp_gallery", "compute",
           "--base", BASE, "--dataset", DATASET, "--config", CONFIG,
           "--site", gallery_site, "--concept", "embed_dim",
           "--mode", "activation", "--fv-class", "original",
           "--rank", "fv_index", "--n", "0", "--n-ref", str(n_ref),
           "--plot", "heat_rf", "--no-samples", "--device", device]
    for b in blocks:
        cmd += ["--blocks", str(b)]
    for det in detectors:
        cmd += ["--detectors", str(det)]
    return cmd


def phase_build(args):
    """One FV-index pass per site over all 12 blocks (no detectors -> no entries,
    but the index — RelMax+ActMax — is built and mirrored)."""
    for npz_site, gsite in SITE_MAP.items():
        cmd = _compute_cmd(gsite, BLOCKS, [], n_ref=args.n_ref, device=args.device)
        print(f"\n=== BUILD site={gsite} (from heuristic '{npz_site}') ===\n{' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def phase_render(args):
    """One entry per (site, block) for that block's consensus top-k detectors."""
    d, D, nimg = _load()
    for npz_site, gsite in SITE_MAP.items():
        for b in BLOCKS:
            dets = consensus_top_k(d, D, nimg, npz_site, b, args.k)
            cmd = _compute_cmd(gsite, [b], dets, n_ref=args.n_ref, device=args.device)
            print(f"\n=== RENDER site={gsite} b{b} top{args.k}={dets} ===\n{' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "render"], required=True)
    ap.add_argument("--k", type=int, default=5, help="top-k detectors per (site, block)")
    ap.add_argument("--n-ref", type=int, default=12, help="reference images per detector")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args()
    if not NPZ.is_file():
        sys.exit(f"heuristic npz not found: {NPZ}")
    (phase_build if args.phase == "build" else phase_render)(args)


if __name__ == "__main__":
    main()
