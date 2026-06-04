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
from crp.concepts import HeadConcept, EmbeddingDimConcept
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


def select_correct(model, ds, num_classes, n_per_class, device, batch_size=128, seed=0,
                   num_workers=0):
    """Random-order scan (so grouped-by-class datasets like dSprites find
    every class fast). Subset(ds, perm) keeps the running counter mapped to
    the true dataset index via ``perm``. ``num_workers=0`` by default — the
    scan is GPU-forward-bound and one-shot, so it avoids ~18 s of DataLoader
    worker (forkserver) spawn."""
    from torch.utils.data import Subset
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    loader = DataLoader(Subset(ds, perm), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
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
        # Fused: out' = out * mul + add, with per-(elem,feature) mul/add built
        # from the per-element method. Avoids three full-tensor torch.where
        # allocations over (B, N, embed).
        meth = self.method
        mz = self.mask & (meth == 0).unsqueeze(1)     # zero
        mm = self.mask & (meth == 1).unsqueeze(1)     # mean-replace
        ms = self.mask & (meth == 2).unsqueeze(1)     # sign-flip
        dt = out.dtype
        mul = 1.0 - (mz | mm).to(dt) - 2.0 * ms.to(dt)          # (B, embed): 1 / 0 / -1
        out = out * mul.unsqueeze(1)
        if self.meanvec is not None and bool(mm.any()):
            out = out + (mm.to(dt) * self.meanvec.to(dt).unsqueeze(0)).unsqueeze(1)
        return out


def concept_spec(concept_name, num_heads, embed_dim, device):
    """Return (concept, n_detectors, D, n_grid) for the chosen formulation.

    ``D`` is a (n_detectors, embed_dim) bool matrix mapping each detector to
    the embed_dim features it owns (head → contiguous head_dim slice;
    embed_dim → a single index). ``n_grid`` is the set of cumulative-flip
    counts (all heads for HeadConcept; log-spaced for EmbeddingDimConcept,
    which has embed_dim detectors)."""
    head_dim = embed_dim // num_heads
    if concept_name == "head":
        concept = HeadConcept(num_heads=num_heads)
        ndet = num_heads
        D = torch.zeros(ndet, embed_dim, dtype=torch.bool, device=device)
        for h in range(num_heads):
            D[h, h * head_dim:(h + 1) * head_dim] = True
        n_grid = list(range(1, num_heads + 1))
    elif concept_name == "embed_dim":
        concept = EmbeddingDimConcept(num_heads=num_heads)
        ndet = embed_dim
        D = torch.eye(embed_dim, dtype=torch.bool, device=device)
        n_grid = list(range(1, embed_dim + 1))   # one-by-one, no sub-sampling
    else:
        raise ValueError(f"unknown concept {concept_name!r}")
    return concept, ndet, D, n_grid


def run_dataset(key, n_images, device, concept_name="head", max_classes=None,
                chunk_size=None, precision="bf16", methods=("zero",)):
    ds_name, ds_kw, tag = DATASETS[key]
    model, ck, probe_path = load_probe(tag, device)
    num_classes = int(ck["num_classes"])
    num_heads = int(model.backbone.blocks[0].attn.num_heads)
    embed_dim = int(model.backbone.embed_dim)
    n_blocks = len(model.backbone.blocks)
    proj_layers = [f"backbone.blocks.{b}.attn.proj_drop" for b in range(n_blocks)]

    from experiments.datasets import load as load_ds
    tf = create_transform(**resolve_data_config({}, model=model.backbone), is_training=False)
    ds = load_ds(ds_name, root=REPO_ROOT / "data", transform=tf, **ds_kw)
    sel = select_correct(model, ds, num_classes, n_images, device)
    if max_classes is not None:
        sel = {c: v for c, v in sel.items() if c < max_classes}
    counts = {c: len(v) for c, v in sel.items()}

    attribution = CondAttribution(model)
    composite = build_lxt_composite()
    concept, ndet, D, n_grid = concept_spec(concept_name, num_heads, embed_dim, device)
    K = len(n_grid)
    print(f"[{key}/{concept_name}] probe={Path(probe_path).parent.name} "
          f"classes={num_classes} heads={num_heads} embed={embed_dim} blocks={n_blocks} "
          f"detectors={ndet} n_grid={n_grid}")
    print(f"[{key}/{concept_name}] images/class min={min(counts.values())} "
          f"max={max(counts.values())}")

    # install perturbation hooks (one per block proj_drop)
    proj_mods = [model.backbone.blocks[b].attn.proj_drop for b in range(n_blocks)]
    hooks = [_PerturbHook() for _ in range(n_blocks)]
    handles = [pm.register_forward_hook(h) for pm, h in zip(proj_mods, hooks)]

    # capture hooks for baseline proj_drop outputs (mean-replace values)
    captured = {}
    def _mk_cap(b):
        def cap(module, inp, out): captured[b] = out.detach()
        return cap

    methods = list(methods)
    orderings = ["most", "least"]
    M, O = len(methods), len(orderings)
    method_ids = np.asarray([METHOD_ID[m] for m in methods])   # absolute ids for the hook
    B = M * O * n_blocks * K
    # fixed per-element decode (same every image): e=(((mi*O+oi)*n_blocks)+b)*K+ki
    e_arr = np.arange(B)
    ki_arr = e_arr % K
    rest = e_arr // K
    b_arr = (rest % n_blocks).astype(np.int16)
    rest2 = rest // n_blocks
    oi_arr = rest2 % O
    mi_arr = rest2 // O
    n_arr = np.asarray(n_grid, dtype=np.int32)[ki_arr]
    pert_name = np.asarray(methods)[mi_arr]
    ord_name = np.asarray(orderings)[oi_arr]
    layer_arr = np.asarray(proj_layers)[b_arr]
    method_t = torch.as_tensor(method_ids[mi_arr], device=device, dtype=torch.long)
    # global config indices per block (each config perturbs exactly one block, so
    # batching per block keeps only that block's hook active — others fast-path).
    idx_by_block = [torch.as_tensor(np.where(b_arr == b)[0], device=device)
                    for b in range(n_blocks)]
    # baseline (n=0) index arrays (M*O*n_blocks rows)
    nb = M * O * n_blocks
    bmi, boi, bb = [], [], []
    for mi in range(M):
        for oi in range(O):
            for b in range(n_blocks):
                bmi.append(mi); boi.append(oi); bb.append(b)
    bb = np.asarray(bb, dtype=np.int16)
    base_pert = np.asarray(methods)[np.asarray(bmi)]
    base_ord = np.asarray(orderings)[np.asarray(boi)]
    base_layer = np.asarray(proj_layers)[bb]

    img_dfs = []
    try:
        for c in sorted(sel):
            for image_idx in sel[c]:
                x = ds[image_idx][0].unsqueeze(0).to(device)
                amp = torch.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(precision == "bf16" and device == "cuda"))
                # 1) baseline logit + capture proj_drop outputs (mean-replace values)
                for h in hooks:
                    h.mask = None
                cap_handles = [pm.register_forward_hook(_mk_cap(b)) for b, pm in enumerate(proj_mods)]
                with torch.no_grad(), amp:
                    base_logits = model(x).float()[0]
                for ch in cap_handles:
                    ch.remove()
                y_logit = float(base_logits[c])
                y_prob = float(torch.softmax(base_logits, -1)[c])
                meanvecs = [captured[b][0].mean(dim=0) for b in range(n_blocks)]

                # 2) relevance ranking (CP-LRP), all blocks from one backward;
                #    build cumulative masks (cummax over ranked detectors) + cum relevance
                xg = x.clone().requires_grad_(True)
                res = attribution(xg, [{"y": [c]}], composite, record_layer=proj_layers)
                block_masks = [torch.zeros(B, embed_dim, dtype=torch.bool, device=device)
                               for _ in range(n_blocks)]
                cumrel = np.zeros((O, n_blocks, K), dtype=np.float64)
                rel_total_b = np.zeros(n_blocks)
                for b in range(n_blocks):
                    rel = concept.attribute(res.relevances[proj_layers[b]], abs_norm=False)[0]  # (ndet,)
                    rel_np = rel.detach().cpu().numpy()
                    rel_total_b[b] = float(np.abs(rel_np).sum())
                    od = torch.argsort(rel, descending=True)
                    for oi, ordering in enumerate(orderings):
                        order = od if ordering == "most" else od.flip(0)
                        cm = torch.cummax(D[order].to(torch.int8), dim=0).values.bool()  # (K, embed)
                        cumrel[oi, b] = np.cumsum(rel_np[order.cpu().numpy()])
                        for mi in range(M):
                            bi = (((mi * O + oi) * n_blocks) + b) * K
                            block_masks[b][bi:bi + K] = cm

                # 3) perturbed forwards — batched PER BLOCK (chunked), so only the
                #    target block's proj_drop hook is active (others fast-path None).
                #    Logits accumulate on-GPU; single host transfer per image.
                pc = torch.empty(B, device=device); pp = torch.empty(B, device=device)
                cs = chunk_size or B
                with torch.no_grad(), amp:
                    for b in range(n_blocks):
                        idxb = idx_by_block[b]
                        for s in range(0, idxb.numel(), cs):
                            cidx = idxb[s:s + cs]; bs = cidx.numel()
                            hooks[b].mask = block_masks[b][cidx]
                            hooks[b].method = method_t[cidx]
                            hooks[b].meanvec = meanvecs[b]
                            out = model(x.expand(bs, -1, -1, -1)).float()
                            hooks[b].mask = None
                            pc[cidx] = out[:, c]
                            pp[cidx] = torch.softmax(out, -1)[:, c]
                pert_c = pc.cpu(); pert_prob = pp.cpu()

                # 4) columnar rows (n>0 then n=0 baselines) — avoid per-row dicts
                main = pl.DataFrame(dict(
                    perturbation=pert_name, ordering=ord_name, block=b_arr,
                    layer=layer_arr, n=n_arr,
                    image_idx=np.full(B, image_idx, np.int64),
                    logit_target=pert_c.numpy().astype(np.float32),
                    prob_target=pert_prob.numpy().astype(np.float32),
                    cum_relevance=cumrel[oi_arr, b_arr, ki_arr].astype(np.float32),
                    rel_total=rel_total_b[b_arr].astype(np.float32),
                ))
                base = pl.DataFrame(dict(
                    perturbation=base_pert, ordering=base_ord, block=bb,
                    layer=base_layer, n=np.zeros(nb, np.int32),
                    image_idx=np.full(nb, image_idx, np.int64),
                    logit_target=np.full(nb, y_logit, np.float32),
                    prob_target=np.full(nb, y_prob, np.float32),
                    cum_relevance=np.zeros(nb, np.float32),
                    rel_total=rel_total_b[bb].astype(np.float32),
                ))
                df_img = pl.concat([main, base]).with_columns([
                    pl.lit(c).cast(pl.Int32).alias("class"),
                    pl.lit(key).alias("dataset"), pl.lit(concept_name).alias("concept"),
                    pl.lit(ndet).cast(pl.Int32).alias("n_detectors"),
                    pl.lit(num_heads).cast(pl.Int16).alias("num_heads"),
                    pl.lit(y_logit).cast(pl.Float32).alias("logit_baseline"),
                    pl.lit(y_prob).cast(pl.Float32).alias("prob_baseline"),
                ])
                img_dfs.append(df_img)
            print(f"[{key}/{concept_name}] class {c:>2} done ({len(sel[c])} imgs) "
                  f"— images {sum(len(v) for cc, v in sel.items() if cc <= c)}")
    finally:
        for hd in handles:
            hd.remove()

    df = pl.concat(img_dfs)
    df = df.with_columns([
        (pl.col("logit_target") / pl.col("logit_baseline")).alias("delta_logit"),
        (pl.col("prob_target") / pl.col("prob_baseline")).alias("delta_prob"),
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"flipping_{concept_name}_{key}.parquet"
    df.write_parquet(out)
    print(f"[{key}/{concept_name}] wrote {len(df)} rows → {out}")
    return out, dict(probe=probe_path, num_classes=num_classes, num_heads=num_heads,
                     n_detectors=ndet, n_grid=n_grid, n_blocks=n_blocks,
                     n_images=n_images, counts=counts)


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    datasets: List[str] = typer.Option(list(DATASETS), "--datasets"),
    concept: List[str] = typer.Option(["head"], "--concept",
                                      help="head | embed_dim (repeatable)"),
    n_images: int = typer.Option(50, "--n-images"),
    device: Optional[str] = typer.Option(None, "--device"),
    max_classes: Optional[int] = typer.Option(None, "--max-classes", help="smoke: limit classes"),
    chunk_size: Optional[int] = typer.Option(4096, "--chunk-size",
                                             help="configs per forward chunk (GPU batch)"),
    precision: str = typer.Option("bf16", "--precision", help="bf16 (model-native) | fp32"),
    perturbation: List[str] = typer.Option(["zero"], "--perturbation",
                                           help="zero | mean | sign_flip (repeatable)"),
):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  datasets={datasets}  concepts={concept}  n_images={n_images} "
          f"chunk_size={chunk_size}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta.update(dict(experiment="concept_flipping", composite="lxt_cplrp", site="proj_drop",
                     perturbations=list(perturbation), orderings=["most", "least"],
                     split="train-clean", n_images_per_class=n_images, device=dev,
                     precision=precision))
    meta.setdefault("concepts", {})
    for cname in concept:
        meta["concepts"].setdefault(cname, {})
        for key in datasets:
            _, dmeta = run_dataset(key, n_images, dev, concept_name=cname,
                                   max_classes=max_classes, chunk_size=chunk_size,
                                   precision=precision, methods=perturbation)
            meta["concepts"][cname][key] = dmeta
            meta_path.write_text(json.dumps(meta, indent=2))  # checkpoint after each
    print(f"meta → {meta_path}")


if __name__ == "__main__":
    app()
