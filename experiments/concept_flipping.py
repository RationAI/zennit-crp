"""Concept-flipping experiment (paper §Concept-flipping).

For each correctly-classified image, rank a layer's concept detectors
(HeadConcept = attention heads, read at ``proj_drop``) by LRP relevance,
then cumulatively perturb them most-relevant-first and least-relevant-first,
recording the target-class logit after each step. The relative logit
``Δ(n) = y'_c / y_c`` is the concept-flipping curve.

Relevance is computed with the CP-LRP (``composite_lxt``) recipe — the
"nicer heatmaps" composite. Under CP-LRP only the value path carries
relevance, so head detectors are read at ``proj_drop`` (head output).

Three perturbation methods are run and stored as a column for comparison:
``zero`` (zero-ablate), ``mean`` (replace with per-image token-mean at the
site), ``sign_flip`` (negate).

Output: one long-format parquet per dataset under
``data/results/concept_flipping/`` (+ ``meta.json``), separate from
``fv_cache`` and training checkpoints. Columns let you group by dataset /
class / layer / perturbation / ordering and recover the curves.

Usage::

    uv run python -m experiments.concept_flipping --n-images 50
    uv run python -m experiments.concept_flipping --datasets dsprites --n-images 4   # quick
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from timm.data import resolve_data_config, create_transform
from torch.utils.data import DataLoader
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma, Pass

from crp.attribution import CondAttribution
from crp.concepts import HeadConcept
from zennit_ext import (
    QInspectionLayer, KInspectionLayer, StopGradient,
    SoftmaxAlongLastDim, ScaleByConstant, ResidualAdd, UniformAdd, LayerScaleMul,
    ResidualRatio, Uniform,
    LayerNormForwardCanonizer, DropoutPassthroughCanonizer,
    TimmBlockResidualCanonizer, EvaBlockResidualCanonizer,
    EvaAttentionSubstitutionCanonizer, TimmAttentionSubstitutionCanonizer,
)
from experiments.models import build_probe

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "results" / "concept_flipping"

# train-ds → (dataset name, loader kwargs, run-dir tag)
DATASETS = {
    "funny_birds":   ("funny_birds",   {"split": "train", "clean_only": True},
                      "funny-birds-train-clean"),
    "dsprites":      ("dsprites",       {"target": "shape"}, "dsprites"),
    "colored_mnist": ("colored_mnist",  {"split": "train"}, "colored-mnist-train"),
}


def build_lxt_composite() -> LayerMapComposite:
    """CP-LRP composite (walkthrough ``composite_lxt``): γ-rule linears,
    Q/K probes → StopGradient (value-path-only attribution)."""
    layer_map = [
        (nn.Linear,           Gamma(gamma=0.10)),
        (nn.Conv2d,           Gamma(gamma=0.25)),
        (nn.GELU,             Pass()),
        (nn.LayerNorm,        Pass()),
        (nn.Dropout,          Pass()),
        (SoftmaxAlongLastDim, Pass()),
        (ScaleByConstant,     Pass()),
        (ResidualAdd,         ResidualRatio(epsilon=1e-6)),
        (UniformAdd,          Uniform(factor=2)),
        (LayerScaleMul,       Uniform(factor=2)),
        (QInspectionLayer,    StopGradient()),
        (KInspectionLayer,    StopGradient()),
        (nn.Identity,         Pass()),
    ]
    canonizers = [
        LayerNormForwardCanonizer(), DropoutPassthroughCanonizer(),
        TimmBlockResidualCanonizer(residual_rule="ratio"),
        EvaBlockResidualCanonizer(residual_rule="ratio", layerscale_uniform=True),
        EvaAttentionSubstitutionCanonizer(block_indices=None),
        TimmAttentionSubstitutionCanonizer(block_indices=None),
    ]
    return LayerMapComposite(layer_map=layer_map, canonizers=canonizers)


def load_probe(tag: str, device: str):
    runs = sorted((REPO_ROOT / "data" / "runs" / f"finetune_vit_small_{tag}").glob("*/best.pt"))
    if not runs:
        raise FileNotFoundError(f"no probe for {tag} under data/runs/finetune_vit_small_{tag}/")
    ck = torch.load(runs[-1], map_location=device, weights_only=False)
    model = build_probe(base=ck["base"], head=ck["head"], num_classes=ck["num_classes"],
                        head_kwargs=ck.get("head_kwargs", {})).eval().to(device)
    if "backbone_state_dict" in ck:
        model.backbone.load_state_dict(ck["backbone_state_dict"])
    model.head.load_state_dict(ck["head_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck, str(runs[-1])


def select_correct(model, ds, num_classes, n_per_class, device, batch_size=128, seed=0):
    """Random-order scan (so grouped-by-class datasets like dSprites find
    every class fast). Subset(ds, perm) keeps the running counter mapped to
    the true dataset index via ``perm``."""
    from torch.utils.data import Subset
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    loader = DataLoader(Subset(ds, perm), batch_size=batch_size, shuffle=False, num_workers=2)
    sel = {c: [] for c in range(num_classes)}
    pos, done = 0, set()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(-1).cpu()
            for j in range(len(y)):
                c = int(y[j])
                if pred[j] == c and len(sel[c]) < n_per_class:
                    sel[c].append(perm[pos + j])
                    if len(sel[c]) >= n_per_class:
                        done.add(c)
            pos += len(y)
            if len(done) == num_classes:
                break
    return sel


METHOD_ID = {"zero": 0, "mean": 1, "sign_flip": 2}


class _PerturbHook:
    """Forward hook on a block's ``proj_drop``. Per-batch-element it zeros /
    mean-replaces / sign-flips the embed_dim slices of selected heads, with a
    per-element method id so all (method × ordering × n) configs run in one
    batched forward."""
    def __init__(self):
        self.mask = None       # (B, embed_dim) bool — features to perturb
        self.method = None     # (B,) long — 0 zero, 1 mean, 2 sign_flip
        self.meanvec = None    # (embed_dim,) per-image token-mean

    def __call__(self, module, inp, out):
        if self.mask is None:
            return out
        meth = self.method
        z = (self.mask & (meth == 0).unsqueeze(1)).unsqueeze(1)   # (B,1,embed)
        mm = (self.mask & (meth == 1).unsqueeze(1)).unsqueeze(1)
        sf = (self.mask & (meth == 2).unsqueeze(1)).unsqueeze(1)
        out = torch.where(z, torch.zeros_like(out), out)
        if self.meanvec is not None:
            out = torch.where(mm, self.meanvec.view(1, 1, -1).expand_as(out), out)
        out = torch.where(sf, -out, out)
        return out


def run_dataset(key, n_images, device, eps_logit=1e-3, max_classes=None):
    ds_name, ds_kw, tag = DATASETS[key]
    model, ck, probe_path = load_probe(tag, device)
    num_classes = int(ck["num_classes"])
    num_heads = int(model.backbone.blocks[0].attn.num_heads)
    embed_dim = int(model.backbone.embed_dim)
    head_dim = embed_dim // num_heads
    n_blocks = len(model.backbone.blocks)
    proj_layers = [f"backbone.blocks.{b}.attn.proj_drop" for b in range(n_blocks)]
    print(f"[{key}] probe={Path(probe_path).parent.name} classes={num_classes} "
          f"heads={num_heads} blocks={n_blocks}")

    from experiments.datasets import load as load_ds
    tf = create_transform(**resolve_data_config({}, model=model.backbone), is_training=False)
    ds = load_ds(ds_name, root=REPO_ROOT / "data", transform=tf, **ds_kw)
    sel = select_correct(model, ds, num_classes, n_images, device)
    if max_classes is not None:
        sel = {c: v for c, v in sel.items() if c < max_classes}
    counts = {c: len(v) for c, v in sel.items()}
    print(f"[{key}] images/class min={min(counts.values())} max={max(counts.values())}")

    attribution = CondAttribution(model)
    composite = build_lxt_composite()
    concept = HeadConcept(num_heads=num_heads)

    # install perturbation hooks (one per block proj_drop)
    proj_mods = [model.backbone.blocks[b].attn.proj_drop for b in range(n_blocks)]
    hooks = [_PerturbHook() for _ in range(n_blocks)]
    handles = [pm.register_forward_hook(h) for pm, h in zip(proj_mods, hooks)]

    # capture hooks for baseline proj_drop outputs (mean-replace values)
    captured = {}
    def _mk_cap(b):
        def cap(module, inp, out): captured[b] = out.detach()
        return cap

    methods = ["zero", "mean", "sign_flip"]
    orderings = ["most", "least"]
    rows = []
    try:
        for c in sorted(sel):
            for image_idx in sel[c]:
                x = ds[image_idx][0].unsqueeze(0).to(device)
                # 1) baseline + capture proj_drop outputs
                for h in hooks:
                    h.mask = None
                cap_handles = [pm.register_forward_hook(_mk_cap(b)) for b, pm in enumerate(proj_mods)]
                with torch.no_grad():
                    base_logits = model(x)[0]
                for ch in cap_handles:
                    ch.remove()
                y_logit = float(base_logits[c])
                y_prob = float(torch.softmax(base_logits, -1)[c])
                meanvecs = [captured[b][0].mean(dim=0) for b in range(n_blocks)]  # (embed_dim,)

                # 2) relevance ranking (CP-LRP), all blocks from one backward
                xg = x.clone().requires_grad_(True)
                res = attribution(xg, [{"y": [c]}], composite, record_layer=proj_layers)
                # per-block per-head relevance (raw)
                rank = {}     # block -> (heads_desc, rel_per_head)
                for b in range(n_blocks):
                    rel = concept.attribute(res.relevances[proj_layers[b]], abs_norm=False)[0]  # (num_heads,)
                    order_desc = torch.argsort(rel, descending=True).tolist()
                    rank[b] = (order_desc, rel.detach().cpu().numpy())

                # 3) perturbed forwards — ALL (method × ordering × block × n)
                #    configs in ONE batched forward. Element index:
                #    e = (((mi*O + oi)*n_blocks) + b)*num_heads + (n-1)
                M, O = len(methods), len(orderings)
                B = M * O * n_blocks * num_heads
                xb = x.expand(B, -1, -1, -1).contiguous()
                method_t = torch.zeros(B, dtype=torch.long, device=device)
                block_masks = [torch.zeros(B, embed_dim, dtype=torch.bool, device=device)
                               for _ in range(n_blocks)]
                for mi, method in enumerate(methods):
                    for oi, ordering in enumerate(orderings):
                        for b in range(n_blocks):
                            heads_desc = rank[b][0]
                            order_heads = heads_desc if ordering == "most" else heads_desc[::-1]
                            for nn_ in range(1, num_heads + 1):
                                e = (((mi * O + oi) * n_blocks) + b) * num_heads + (nn_ - 1)
                                method_t[e] = mi
                                for h in order_heads[:nn_]:
                                    block_masks[b][e, h * head_dim:(h + 1) * head_dim] = True
                for b in range(n_blocks):
                    hooks[b].mask = block_masks[b]
                    hooks[b].method = method_t
                    hooks[b].meanvec = meanvecs[b]
                with torch.no_grad():
                    pert_logits = model(xb)                # (B, num_classes)
                for h in hooks:
                    h.mask = None
                pert_c = pert_logits[:, c]
                pert_prob = torch.softmax(pert_logits, -1)[:, c]
                for mi, method in enumerate(methods):
                    for oi, ordering in enumerate(orderings):
                        for b in range(n_blocks):
                            heads_desc, rel = rank[b]
                            order_heads = heads_desc if ordering == "most" else heads_desc[::-1]
                            rel_total = float(np.abs(rel).sum())
                            rows.append(dict(
                                dataset=key, perturbation=method, ordering=ordering,
                                block=b, layer=proj_layers[b], **{"class": c},
                                image_idx=int(image_idx), n=0,
                                logit_target=y_logit, prob_target=y_prob,
                                logit_baseline=y_logit, prob_baseline=y_prob,
                                cum_relevance=0.0, rel_total=rel_total, num_heads=num_heads,
                            ))
                            for nn_ in range(1, num_heads + 1):
                                e = (((mi * O + oi) * n_blocks) + b) * num_heads + (nn_ - 1)
                                cum_rel = float(rel[order_heads[:nn_]].sum())
                                rows.append(dict(
                                    dataset=key, perturbation=method, ordering=ordering,
                                    block=b, layer=proj_layers[b], **{"class": c},
                                    image_idx=int(image_idx), n=nn_,
                                    logit_target=float(pert_c[e]), prob_target=float(pert_prob[e]),
                                    logit_baseline=y_logit, prob_baseline=y_prob,
                                    cum_relevance=cum_rel, rel_total=rel_total, num_heads=num_heads,
                                ))
            print(f"[{key}] class {c:>2} done ({len(sel[c])} imgs) — rows so far {len(rows)}")
    finally:
        for hd in handles:
            hd.remove()

    df = pl.DataFrame(rows)
    df = df.with_columns([
        (pl.col("logit_target") / pl.col("logit_baseline")).alias("delta_logit"),
        (pl.col("prob_target") / pl.col("prob_baseline")).alias("delta_prob"),
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"flipping_{key}.parquet"
    df.write_parquet(out)
    print(f"[{key}] wrote {len(df)} rows → {out}")
    return out, dict(probe=probe_path, num_classes=num_classes, num_heads=num_heads,
                     n_blocks=n_blocks, n_images=n_images, counts=counts)


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    datasets: List[str] = typer.Option(list(DATASETS), "--datasets"),
    n_images: int = typer.Option(50, "--n-images"),
    device: Optional[str] = typer.Option(None, "--device"),
    max_classes: Optional[int] = typer.Option(None, "--max-classes", help="smoke: limit classes"),
):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  datasets={datasets}  n_images={n_images}")
    meta = dict(
        experiment="concept_flipping", composite="lxt_cplrp",
        concept="HeadConcept", site="proj_drop",
        perturbations=["zero", "mean", "sign_flip"],
        orderings=["most", "least"], split="train-clean",
        n_images_per_class=n_images, device=dev, datasets={},
    )
    for key in datasets:
        _, dmeta = run_dataset(key, n_images, dev, max_classes=max_classes)
        meta["datasets"][key] = dmeta
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"meta → {OUT_DIR / 'meta.json'}")


if __name__ == "__main__":
    app()
