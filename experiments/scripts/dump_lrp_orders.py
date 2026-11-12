"""Persist the LRP detector rankings for the cells already in the benchmark store.

The benchmark (:mod:`experiments.concept_detector_bench`) keeps only the MoRF/LeRF
curves and the DAPC of each cell, not the ranking that produced them, while the
search side-car (:mod:`experiments.concept_detector_optimal`) does store its
orders. Rank statistics (top-k overlap between rankings, rank frequency per
detector) need both, so this script fills the gap for the two LRP methods.

It is cheap: ψ for every site and block comes from a single backward pass per
(image, method), so no ablation curve is recomputed.

Writes ``data/results/benchmark/cdet_orders_<key>[__head].npz``:
    <method>__<site>__b<block>__psi     (n_images, D) float32, signed score
    <method>__<site>__b<block>__order   (n_images, D) int16/int32, psi descending

``--verify`` re-runs one ablation curve per model from a dumped order and
compares it against the curve the benchmark stored, which proves the dumped
ranking is the one the benchmark actually used.

Run:  uv run python -m experiments.scripts.dump_lrp_orders [--verify] [--models k1,k2]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from crp.concepts import EmbeddingDimConcept, HeadConcept
from experiments.concept_detector_bench import (
    MODELS_CFG, RES_DIR, ZeroChannelsHook, is_eva, layer_name, method_sites,
    prob_curve, rank_psi, select_correct)
from experiments.datasets import load_eval_dataset
from experiments.gradinput import GradTimesInputAttribution
from experiments.model_datasets import find_by_tag
from experiments.models import backbone_transforms
from zennit_extensions.canonisation.canonizers import (
    EvaAttentionSubstitutionCanonizer, VanillaViTAttentionSubstitutionCanonizer)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_IMAGES = 16
LRP_METHODS = ("cp_lrp", "chefer")


def stored_cells(key: str, concept: str):
    """(site, block) pairs the benchmark holds for each LRP method."""
    suffix = "__head" if concept == "head" else ""
    p = RES_DIR / f"cdet_dapc_{key}{suffix}.npz"
    if not p.exists():
        return None, None
    z = np.load(p, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    cells = {m: set() for m in LRP_METHODS}
    for k in z.files:
        m = re.match(r"(.+?)__(.+?)__b(\d+)__dapc$", k)
        if m and m.group(1) in cells:
            cells[m.group(1)].add((m.group(2), int(m.group(3))))
    return meta, cells


def verify_one(model, normalize, ds, picks, concept_obj, head_dim, D, key, concept,
               orders, cells):
    """Recompute one MoRF curve from a dumped order; compare to the stored curve."""
    suffix = "__head" if concept == "head" else ""
    z = np.load(RES_DIR / f"cdet_dapc_{key}{suffix}.npz", allow_pickle=True)
    method = next((m for m in LRP_METHODS if cells[m]), None)
    if method is None:
        return None
    site, b = sorted(cells[method])[-1]            # last block = strongest signal
    handles = []
    for c in (VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
              EvaAttentionSubstitutionCanonizer(block_indices=None)):
        handles += c.apply(model)
    try:
        hook = ZeroChannelsHook(head_dim=head_dim)
        hh = model.get_submodule(layer_name(site, b)).register_forward_hook(hook)
        try:
            order = torch.from_numpy(
                orders[f"{method}__{site}__b{b}__order"][0].astype(np.int64))
            x = ds[picks[0][0]][0].unsqueeze(0)
            got = prob_curve(model, normalize, x, picks[0][1], order, hook)
        finally:
            hh.remove()
    finally:
        for h in handles:
            h.remove()
    want = z[f"{method}__{site}__b{b}__morf"][0]
    err = float(np.abs(got - want).max())
    print(f"  [verify] {method} · {site} · b{b} · img0: max|recomputed - stored| "
          f"= {err:.3e}  {'OK' if err < 1e-4 else 'MISMATCH'}")
    return err


def run_model(key, dataset, extra, concept, verify=False):
    meta, cells = stored_cells(key, concept)
    if meta is None:
        print(f"-- {key} [{concept}]: no benchmark store, skipped")
        return
    out = RES_DIR / f"cdet_orders_{key}{'__head' if concept == 'head' else ''}.npz"
    model = find_by_tag(key, device=DEVICE).model.eval()
    transform, normalize = backbone_transforms(model.backbone)
    ds = load_eval_dataset(dataset, transform, extra)
    num_heads = model.backbone.blocks[0].attn.num_heads
    embed_dim = int(model.backbone.embed_dim)

    if concept == "head":
        head_dim, D = embed_dim // num_heads, num_heads
        concept_obj = HeadConcept(num_heads=num_heads)
    else:
        head_dim, D = None, embed_dim
        concept_obj = EmbeddingDimConcept(num_heads=num_heads)

    picks = select_correct(model, normalize, ds, N_IMAGES)
    ids = [i for i, _, _ in picks]
    if meta.get("image_ids") != ids:
        raise SystemExit(
            f"[{key}/{concept}] image sample differs from the benchmark store "
            f"({meta.get('image_ids')[:4]}… vs {ids[:4]}…); rankings would not "
            "correspond to the stored curves.")

    attribution = GradTimesInputAttribution(model)
    store = {}
    for method in LRP_METHODS:
        if not cells[method]:
            continue
        sites = sorted({s for s, _ in cells[method]})
        blocks = sorted({b for _, b in cells[method]})
        sites = method_sites(method, sites)
        psi_buf = {(s, b): np.zeros((len(picks), D), np.float32)
                   for s in sites for b in blocks}
        for j, (idx, pred, _) in enumerate(picks):
            x = ds[idx][0].unsqueeze(0).to(DEVICE)
            psi = rank_psi(attribution, normalize, x, pred, method, num_heads,
                           sites, blocks, concept_obj)
            for sb, v in psi.items():
                psi_buf[sb][j] = v.numpy()
        idt = np.int16 if D <= 32767 else np.int32
        for (s, b), arr in psi_buf.items():
            if (s, b) not in cells[method]:
                continue                      # method_sites widened beyond the store
            store[f"{method}__{s}__b{b}__psi"] = arr
            store[f"{method}__{s}__b{b}__order"] = np.argsort(
                -arr, axis=1, kind="stable").astype(idt)
        print(f"[{key}/{concept}] {method}: {len(cells[method])} cells, "
              f"{len(picks)} images, D={D}")

    if verify:
        verify_one(model, normalize, ds, picks, concept_obj, head_dim, D, key,
                   concept, store, cells)

    out_meta = {"key": key, "concept": concept, "D": D, "embed_dim": embed_dim,
                "num_heads": int(num_heads), "image_ids": ids,
                "preds": [p for _, p, _ in picks],
                "source": f"cdet_dapc_{key}{'__head' if concept == 'head' else ''}.npz"}
    np.savez(out, meta=json.dumps(out_meta), **store)
    print(f"[{key}/{concept}] saved {out.name} ({len(store)//2} cells)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--models", default="", help="comma-separated zoo keys")
    a = ap.parse_args()
    want = {m.strip() for m in a.models.split(",") if m.strip()}
    for key, _tag, dataset, extra, _label in MODELS_CFG:
        if want and key not in want:
            continue
        for concept in ("embed_dim", "head"):
            run_model(key, dataset, extra, concept, verify=a.verify)


if __name__ == "__main__":
    main()
