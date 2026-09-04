"""Per-image "local view" gallery entries from the heuristic-optimal rankings.

The greedy-optimal heuristic (experiments.concept_detector_optimal) ranks the D
embedding channels per (site, block) SEPARATELY FOR EACH ranked image (the
``image_ids`` in the npz meta). This driver renders, for each of those images as
a selectable gallery *sample* and each (site, block) layer, the top-k detectors
from THAT image's own MoRF order — each as a local-view entry: the query image +
its conditional CRP heatmap, then the detector's activation-max representatives
(detector-intrinsic, from the FV pool).

Model-agnostic. Complements gallery_optimal_actmax.py (the Aggregate view) —
same labeled config ``cp_lrp_baseline_optimal_actmax`` and prebuilt FV index
(reused, no rebuild). The query images come from the SAME split the heuristic
ranked on (funny_birds test / imagenet full val), indexed by the npz image_ids.

Usage::
    uv run python -m experiments.scripts.gallery_optimal_local --model vit_small --dataset funny_birds --k 5
    uv run python -m experiments.scripts.gallery_optimal_local --model vit_base --dataset imagenet --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD = "optimal"
CONFIG = "cp_lrp_baseline_optimal_actmax"
SITE_MAP = {"residual": "residual", "proj_drop": "proj_drop", "qk": "query", "value": "value"}
BLOCKS = list(range(12))
REF_NAME = "actsum"          # activation-max references (fv.ActMax_sum_normed)

# ds_extra used by concept_detector_optimal.MODELS_CFG when it ranked, so the
# image_ids index the same pool. funny_birds ranked on the test split; imagenet
# on the full val (no extra).
HEURISTIC_QUERY_DS = {"funny_birds": {"split": "test"}, "imagenet": {}}


def npz_path(tag):
    return REPO_ROOT / f"data/results/benchmark/cdet_dapc_{tag}__optimal.npz"


def n_images(d):
    """Actual #images ranked per (site, block) — may be fewer than meta image_ids."""
    import re
    js = [int(m.group(1)) for k in d.files
          if (m := re.match(rf"{METHOD}__residual__b0__img(\d+)__order", k))]
    return max(js) + 1 if js else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="vit_small")
    ap.add_argument("--dataset", default="funny_birds")
    ap.add_argument("--k", type=int, default=5, help="top-k detectors per (image, layer)")
    ap.add_argument("--n-ref", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--limit-blocks", type=int, default=None)
    ap.add_argument("--sites", default=None, help="comma list of npz sites (debug)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base, dataset = args.model, args.dataset
    tag = f"{base}_{dataset}"
    if not npz_path(tag).is_file():
        sys.exit(f"heuristic npz not found: {npz_path(tag)}")

    from experiments.crp_gallery import (
        render_local_entry, make_concept, SITE_LAYERS, COMPOSITES, FIG_DIR,
        rebuild_manifest,
    )
    from experiments.gradinput import (
        GradTimesInputAttribution, GradTimesInputFeatureVisualization)
    from experiments.model_datasets import find
    from experiments.datasets import load_eval_dataset
    from crp.image import imgify

    d = np.load(npz_path(tag), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    image_ids, preds = meta["image_ids"], meta["preds"]

    device = args.device
    mdset = find(base, dataset, device=device)
    model, normalize, transform, ds = mdset.model, mdset.normalize, mdset.transform, mdset.dataset
    num_heads = model.backbone.blocks[0].attn.num_heads
    concept = make_concept("embed_dim", num_heads)
    attribution = GradTimesInputAttribution(model)
    comp_cls = COMPOSITES[CONFIG]

    # exact ranked images, from the split the heuristic used
    query_ds = load_eval_dataset(dataset, transform, HEURISTIC_QUERY_DS.get(dataset, {}))

    fv_dir = REPO_ROOT / "data/crp_gallery_cache/fv" / tag / CONFIG
    sites = args.sites.split(",") if args.sites else list(SITE_MAP)
    blocks = [args.limit_blocks] if args.limit_blocks is not None else BLOCKS
    layer_names = [SITE_LAYERS[SITE_MAP[s]][b] for s in sites for b in blocks]
    fv = GradTimesInputFeatureVisualization(
        attribution, ds, {ln: concept for ln in layer_names},
        preprocess_fn=normalize, path=str(fv_dir), device=device, negative_clamp=True)

    fig_model = FIG_DIR / tag
    entries_root = fig_model / CONFIG
    samples_dir = fig_model / "_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    avail = n_images(d)
    n_img = avail if args.limit_images is None else min(args.limit_images, avail)
    n_entries = 0
    for j in range(n_img):
        ds_idx, target = int(image_ids[j]), int(preds[j])
        x = query_ds[ds_idx][0]
        key = f"img{j}"
        label = f"image {j} · #{ds_idx} · class {target}"
        spath = samples_dir / f"{key}.png"
        if not args.dry_run and not spath.exists():
            imgify(x.detach().cpu()).save(spath)
        for s in sites:
            gsite = SITE_MAP[s]
            for b in blocks:
                layer = SITE_LAYERS[gsite][b]
                order = d[f"{METHOD}__{s}__b{b}__img{j}__order"]
                dets = [int(c) for c in order[:args.k]]
                print(f"img{j}(#{ds_idx},c{target}) {gsite} b{b}: {dets}")
                if args.dry_run:
                    continue
                for rank_local, cid in enumerate(dets):
                    out_dir = (entries_root / gsite / f"block{b}" / "embed_dim"
                               / "_img" / key / REF_NAME / str(cid))
                    render_local_entry(
                        fv, attribution, ds, x, target, layer, cid,
                        mode="activation", n_ref=args.n_ref, composite=comp_cls(),
                        concept=concept, normalize=normalize, device=device,
                        crop=False, plot="heat_rf", out_dir=out_dir,
                        fv_class="original", include_negative=False,
                        meta_extra={
                            "layer": layer, "site": gsite, "block": b,
                            "concept_kind": "embed_dim", "config": CONFIG,
                            "fv_class": "original", "include_negative": False,
                            "sample": key, "sample_label": label,
                            "ref": REF_NAME, "rank": rank_local})
                    n_entries += 1

    print(f"\nrendered {n_entries} local entries for {n_img} images")
    if not args.dry_run:
        rebuild_manifest()
        print("manifest rebuilt")


if __name__ == "__main__":
    main()
