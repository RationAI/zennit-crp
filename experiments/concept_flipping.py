"""Concept-flipping experiment (paper §Concept-flipping).

Question: is the LRP relevance assigned to a transformer's concept detectors
(attention heads, or embedding dimensions) representative of their causal
influence on the prediction, and is that influence concentrated in a few
detectors?

Method, per correctly-classified image and per attention block:
  1. rank the block's concept detectors by their (signed) LRP relevance for the
     target class — relevance is initialised at the target logit, zero on every
     other class (conditional LRP, ``crp.attribution.relevance_init``);
  2. cumulatively perturb them in two orderings — most-relevant-first (MoRF) and
     least-relevant-first (LeRF) — re-running the model after each step and
     recording the target-class probability;
  3. the relative response  Δ(n) = p'_c(n) / p_c  is the concept-flipping curve.
If relevance ∝ influence, Δ_most drops faster than Δ_least; the gap between them
is the concept-flipping score (an AOPC difference; computed in the results
notebook, not here). Ranking is *signed*: negatively-relevant detectors are
removed last under MoRF and *raise* p_c, so Δ_most can dip then recover — the
sign crossing is informative, not a bug.

Everything is measured in PROBABILITY space (the logit ratio is ill-defined when
the target logit is small/negative). Relevance propagation is pluggable: pass any
recipe from ``COMPOSITES`` via ``--config`` (default ``cp_lrp_baseline``, the
LXT value-path recipe). Detectors are read at the config's probe site
(``proj_drop``). Perturbation is ``zero`` / ``mean`` / ``sign_flip``.

This script only *gathers* the perturbation data; all metrics (AOPC, scores),
statistics (Wilcoxon) and plots live in ``tutorials/concept_flipping_results.ipynb``.
Output: one long-format parquet per (config, concept, dataset) under
``data/results/concept_flipping/`` (+ ``meta.json``) — group by
config/dataset/class/block/perturbation/ordering and recover each curve from the
``n`` column.

Examples::

    uv run python -m experiments.concept_flipping --config cp_lrp_baseline --concept embed_dim
    uv run python -m experiments.concept_flipping --datasets funny_birds \
        --config cp_lrp_baseline --concept embed_dim --classes 0 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from timm.data import resolve_data_config, create_transform

from zennit_extensions.lrp_composites import (
    AttnLRPBaselineComposite, CheferLRPComposite, CPLRPComposite,
)

# The composites this experiment uses, tracked explicitly by name. Gathered
# data references these name strings only; the definitions live in
# zennit_extensions.lrp_composites and their provenance in the experiment journal.
COMPOSITES = {
    "cp_lrp_baseline": CPLRPComposite,
    "attnlrp_baseline": AttnLRPBaselineComposite,
    "chefer_lrp": CheferLRPComposite,
}
from crp.attribution import CondAttribution
from crp.concepts import HeadConcept, EmbeddingDimConcept
# Model + data loading is shared in experiments.model_io; re-exported here so
# existing imports (sae.py, scripts) keep working.
from experiments.model_io import DATASETS, load_probe, select_correct, site_layer_names
from experiments import sae as sae_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "results" / "concept_flipping"

# perturbation name → integer id consumed by PerturbHook
PERTURBATIONS = {"zero": 0, "mean": 1, "sign_flip": 2}
ORDERINGS = ("most", "least")

# Deferred random reference (NOT a cumulative random ordering — that was dropped
# as redundant with LeRF and only meaningful relative to other configs): the
# principled control is a per-n random-SUBSET variance band — for each n draw
# many random subsets of size n, perturb, and record the effect distribution to
# get E[effect|n] ± variance. It separates "how many detectors removed" from
# "which ones". Build it as a separate arm if/when needed.


# ─────────────────────────────────────────────────────────────────────────────
# General, reusable building blocks  (load_probe / select_correct / DATASETS are
# imported from experiments.model_io and re-exported above.)
# ─────────────────────────────────────────────────────────────────────────────

def _subsample_grid(n_det: int, max_steps: Optional[int]) -> List[int]:
    """Cumulative flip counts. One detector at a time (``1..n_det``) unless
    ``max_steps`` caps it, in which case ~``max_steps`` evenly-spaced unique
    counts spanning ``1..n_det`` (always including the last). The AOPC metric is
    a mean over the normalised fraction ``n/N``, so a coarse grid still estimates
    it — this keeps large bases (SAE ``m=3072``) tractable."""
    if not max_steps or n_det <= max_steps:
        return list(range(1, n_det + 1))
    return sorted(set(int(round(x)) for x in np.linspace(1, n_det, max_steps)))


def concept_detectors(concept_name: str, num_heads: int, feat_dim: int,
                      device, max_steps: Optional[int] = None,
                      ) -> Tuple[object, torch.Tensor, List[int]]:
    """Resolve a concept formulation into the pieces the flip needs:

    * ``concept`` — the :mod:`crp.concepts` instance whose ``attribute`` gives
      per-detector relevance from a recorded ``(B, N, feat_dim)`` tensor.
    * ``D`` — a ``(n_detectors, feat_dim)`` bool matrix mapping each detector
      to the feature components it owns (a head → its contiguous ``head_dim``
      slice; an embedding dim / SAE latent → a single index). Drives the masks.
    * ``n_grid`` — cumulative flip counts (see :func:`_subsample_grid`).

    ``sae`` is per-latent like ``embed_dim`` but over the spliced SAE feature
    space, so ``feat_dim`` is the dictionary size ``m`` (not ``embed_dim``).
    """
    if concept_name == "head":
        head_dim = feat_dim // num_heads
        D = torch.zeros(num_heads, feat_dim, dtype=torch.bool, device=device)
        for h in range(num_heads):
            D[h, h * head_dim:(h + 1) * head_dim] = True
        return HeadConcept(num_heads=num_heads), D, list(range(1, num_heads + 1))
    if concept_name in ("embed_dim", "sae"):
        D = torch.eye(feat_dim, dtype=torch.bool, device=device)
        return (EmbeddingDimConcept(num_heads=num_heads), D,
                _subsample_grid(feat_dim, max_steps))
    raise ValueError(f"unknown concept {concept_name!r}")


class PerturbHook:
    """Forward hook for a block's probe site (``proj_drop``). For each batch
    element it perturbs the embedding features flagged in ``mask`` according to
    that element's ``method`` id (0 zero-ablate, 1 mean-replace, 2 sign-flip),
    fused as ``out' = out * mul + add`` to avoid per-method ``torch.where``
    allocations. Inactive when ``mask is None``. ``meanvec`` (per-image
    token-mean at the site) is only needed when an element mean-replaces.
    """
    def __init__(self):
        self.mask: Optional[torch.Tensor] = None    # (B, embed_dim) bool
        self.method: Optional[torch.Tensor] = None   # (B,) long, see PERTURBATIONS
        self.meanvec: Optional[torch.Tensor] = None  # (embed_dim,) or None

    def __call__(self, module, inp, out):
        if self.mask is None:
            return out
        flip_zero = self.mask & (self.method == 0).unsqueeze(1)
        flip_mean = self.mask & (self.method == 1).unsqueeze(1)
        flip_sign = self.mask & (self.method == 2).unsqueeze(1)
        dt = out.dtype
        mul = 1.0 - (flip_zero | flip_mean).to(dt) - 2.0 * flip_sign.to(dt)  # 1 / 0 / -1
        out = out * mul.unsqueeze(1)
        if self.meanvec is not None:
            out = out + (flip_mean.to(dt) * self.meanvec.to(dt)).unsqueeze(1)
        return out


def cumulative_masks(D: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Cumulative perturbation masks for one detector ordering. Row ``k`` is the
    union of the embedding features owned by the top-``(k+1)`` detectors in
    ``order`` (a ``(len(order), embed_dim)`` bool matrix). ``cummax`` over the
    reordered ``D`` accumulates the union one detector at a time."""
    return torch.cummax(D[order].to(torch.int8), 0).values.bool()


def forward_prob(model, x, c: int, hooks, masks: torch.Tensor, method_t: torch.Tensor,
                 idx_by_block, chunk_size: int, amp, meanvecs) -> torch.Tensor:
    """Run the perturbed forwards for a set of config rows and return the
    target-class probability of each. ``masks`` is ``(E, embed_dim)`` and
    ``idx_by_block[b]`` lists the rows whose perturbation lands on block ``b``;
    only block ``b``'s hook is armed while its rows run, so a single batched
    forward covers many cumulative-flip steps. Chunked to bound memory."""
    pp = torch.empty(masks.shape[0], device=x.device)
    with torch.no_grad(), amp:
        for b, idxb in enumerate(idx_by_block):
            for s in range(0, idxb.numel(), chunk_size):
                cidx = idxb[s:s + chunk_size]
                hooks[b].mask = masks[cidx]
                hooks[b].method = method_t[cidx]
                hooks[b].meanvec = meanvecs[b] if meanvecs is not None else None
                out = model(x.expand(cidx.numel(), -1, -1, -1)).float()
                hooks[b].mask = None
                pp[cidx] = out.softmax(-1)[:, c]
    return pp


def setup_sites(model, site: str, concept_name: str, dataset_key: str, device,
                sae_m: Optional[int] = None,
                ) -> Tuple[List[str], List[nn.Module], int, Optional[List[float]]]:
    """Resolve probe site + (optional) SAE splice into the pieces ``run_dataset``
    needs, per block:

    * ``record_layers[b]`` — layer name whose recorded relevance feeds
      ``concept.attribute`` (and which is perturbed).
    * ``hook_mods[b]`` — module to attach the :class:`PerturbHook` / capture to;
      its OUTPUT is the perturbed tensor (B, N, feat_dim).
    * ``feat_dim`` — ``embed_dim`` for axis-aligned concepts, the SAE dictionary
      size ``m`` for ``--concept sae``.
    * ``fvu`` — per-block reconstruction FVU when an SAE is spliced (else None),
      reported as the splice-faithfulness control.

    For ``--concept sae`` the trained SAE is spliced in as a reconstruction
    pass-through (``experiments.sae.SAESplice``) at the site, exposing the
    feature tensor at a ``features`` sublayer; record/perturb point there. At
    the ``residual`` site the splice wraps the whole block (``inner=block``); at
    ``proj_drop`` it replaces the dropout module.
    """
    blocks = model.backbone.blocks
    n_blocks = len(blocks)
    is_sae = concept_name == "sae"
    fvu = [] if is_sae else None

    base_name = site_layer_names(model, site)  # canonical site → layer-name map

    if not is_sae:
        mods = sae_mod.site_modules(model, site)
        return base_name, mods, model.backbone.embed_dim, None

    record_layers, hook_mods, m = [], [], None
    for b in range(n_blocks):
        ck = torch.load(sae_mod.sae_path(site, dataset_key, b, m=sae_m), map_location=device,
                        weights_only=False)
        raw = {k: ck["raw"][k].to(device) for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
        fvu.append(float(ck["fvu"]))
        if site == "proj_drop":
            splice = sae_mod.SAESplice(raw, inner=None).to(device).eval()
            blocks[b].attn.proj_drop = splice
            record_layers.append(f"backbone.blocks.{b}.attn.proj_drop.features")
        else:  # residual: wrap the block
            splice = sae_mod.SAESplice(raw, inner=blocks[b]).to(device).eval()
            blocks[b] = splice
            record_layers.append(f"backbone.blocks.{b}.features")
        m = splice.m
        hook_mods.append(splice.features)
    return record_layers, hook_mods, m, fvu


# ─────────────────────────────────────────────────────────────────────────────
# The experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(key: str, config_name: str, concept_name: str, methods: Sequence[str],
                classes: Optional[Sequence[int]], n_images: int,
                device: str, chunk_size: int, precision: str, site: str = "proj_drop",
                max_steps: Optional[int] = None, sae_m: Optional[int] = None) -> Tuple[Path, dict]:
    """Run the concept-flipping experiment for one (config, concept, dataset,
    site) and write a long-format parquet. Stages are marked inline below."""
    methods = list(methods)
    method_ids = np.array([PERTURBATIONS[m] for m in methods])
    need_mean = "mean" in methods
    amp = torch.autocast("cuda", dtype=torch.bfloat16,
                         enabled=(precision == "bf16" and device == "cuda"))

    # ── Stage 1 · model, dataset, relevance composite, concept detectors ──────
    # Read geometry BEFORE any SAE splice (which replaces site modules).
    model, ck, probe_path = load_probe(DATASETS[key][2], device)
    num_heads = model.backbone.blocks[0].attn.num_heads
    embed_dim = model.backbone.embed_dim
    n_blocks = len(model.backbone.blocks)
    # site / splice → record-layer names, hook modules, feature dim (m for sae).
    sites, hook_mods, feat_dim, fvu = setup_sites(model, site, concept_name, key, device, sae_m=sae_m)
    attribution = CondAttribution(model)
    composite = COMPOSITES[config_name]()
    concept, D, n_grid = concept_detectors(concept_name, num_heads, feat_dim, device, max_steps)
    K = len(n_grid)
    grid_rows = np.asarray(n_grid, dtype=np.int64) - 1            # rows into the
    grid_rows_t = torch.as_tensor(grid_rows, device=device)      # full cumulative arrays

    ds_name, ds_kw, _ = DATASETS[key]
    transform = create_transform(**resolve_data_config({}, model=model.backbone), is_training=False)
    from experiments.datasets import load as load_dataset
    ds = load_dataset(ds_name, root=REPO_ROOT / "data", transform=transform, **ds_kw)

    # ── Stage 2 · pick N correctly-classified images per (chosen) class ───────
    target_classes = sorted(set(classes) & set(range(int(ck["num_classes"])))) if classes \
        else list(range(int(ck["num_classes"])))
    # Cap the scan: an SAE splice can break a class's predictions (esp. dSprites,
    # 737k imgs), which would otherwise crawl the whole dataset (~20 min). 30k is
    # ample to fill healthy classes; unfillable ones return partial (handled below).
    sel = select_correct(model, ds, target_classes, n_images, device,
                         max_scan=min(len(ds), 30000))
    counts = {c: len(v) for c, v in sel.items()}
    print(f"[{config_name}/{key}/{concept_name}/{site}] heads={num_heads} embed={embed_dim} "
          f"feat_dim={feat_dim} blocks={n_blocks} detectors={D.shape[0]} steps={K} "
          f"methods={methods} classes={len(sel)} imgs/class={min(counts.values())}..{max(counts.values())}"
          + (f" sae_fvu={np.mean(fvu):.3f}" if fvu else ""))

    # ── Stage 3 · config grid for the two orderings. Every
    #    (method, ordering∈{most,least}, block, n) is one batch element
    #    e = (((mi*O + oi)*n_blocks) + b)*K + ki. Decode arrays + per-block row
    #    lists are image-independent, so they are built once. ────────────────────
    M, O = len(methods), len(ORDERINGS)  # O == 2 (most, least)
    e = np.arange(M * O * n_blocks * K)
    ki_arr, rest = e % K, e // K
    b_arr = (rest % n_blocks).astype(np.int16)
    oi_arr, mi_arr = (rest // n_blocks) % O, (rest // n_blocks) // O
    n_arr = np.array(n_grid, dtype=np.int32)[ki_arr]
    pert_arr, ord_arr = np.array(methods)[mi_arr], np.array(ORDERINGS)[oi_arr]
    layer_arr = np.array(sites)[b_arr]
    method_t = torch.as_tensor(method_ids[mi_arr], device=device)
    idx_by_block = [torch.as_tensor(np.where(b_arr == b)[0], device=device) for b in range(n_blocks)]

    # n=0 baseline rows, one per (method, ordering, block)
    base = [(m, o, b) for m in methods for o in ORDERINGS for b in range(n_blocks)]
    base_pert = np.array([g[0] for g in base])
    base_ord = np.array([g[1] for g in base])
    base_block = np.array([g[2] for g in base], dtype=np.int16)
    base_layer = np.array(sites)[base_block]

    hooks = [PerturbHook() for _ in range(n_blocks)]
    handles = [pm.register_forward_hook(h) for pm, h in zip(hook_mods, hooks)]

    # ── Stage 4 · per image: baseline → rank → most/least flips → rows ────────
    img_dfs = []
    try:
        for c in target_classes:
            for image_idx in sel[c]:
                x = ds[image_idx][0].unsqueeze(0).to(device)

                # 4a. baseline prob + capture proj_drop activations (mean ref)
                for h in hooks:
                    h.mask = None
                caps: Dict[int, torch.Tensor] = {}
                cap_handles = [pm.register_forward_hook(
                    lambda m, i, o, b=b: caps.__setitem__(b, o.detach()))
                    for b, pm in enumerate(hook_mods)]
                with torch.no_grad(), amp:
                    base_logits = model(x).float()[0]
                for ch in cap_handles:
                    ch.remove()
                p_base = float(base_logits.softmax(-1)[c])
                meanvecs = [caps[b][0].mean(0) for b in range(n_blocks)] if need_mean else None

                # 4b. rank detectors by conditional LRP relevance (one backward,
                #     all blocks); build cumulative masks for most/least + the
                #     cumulative signed relevance along each ordering
                xg = x.clone().requires_grad_(True)
                rel_layers = attribution(xg, [{"y": [c]}], composite, record_layer=sites).relevances
                ranked_masks = torch.zeros(e.size, feat_dim, dtype=torch.bool, device=device)
                cumrel = np.zeros((O, n_blocks, K))
                rel_total = np.zeros(n_blocks)
                for b in range(n_blocks):
                    rel = concept.attribute(rel_layers[sites[b]], abs_norm=False)[0]  # (n_det,)
                    rel_np = rel.cpu().numpy()
                    rel_total[b] = np.abs(rel_np).sum()
                    order_desc = torch.argsort(rel, descending=True)
                    for oi in range(O):  # 0=most (desc), 1=least (asc)
                        order = order_desc if oi == 0 else order_desc.flip(0)
                        # cumulative arrays span all detectors; sample at n_grid.
                        cumrel[oi, b] = np.cumsum(rel_np[order.cpu().numpy()])[grid_rows]
                        cm = cumulative_masks(D, order)[grid_rows_t]
                        for mi in range(M):
                            start = (((mi * O + oi) * n_blocks) + b) * K
                            ranked_masks[start:start + K] = cm

                # 4c. perturbed forwards for both orderings (prob space)
                pp = forward_prob(model, x, c, hooks, ranked_masks, method_t,
                                  idx_by_block, chunk_size, amp, meanvecs)

                # 4d. collect rows: ranked arms (n>0) + the shared n=0 baseline
                ranked_df = pl.DataFrame(dict(
                    perturbation=pert_arr, ordering=ord_arr, block=b_arr, layer=layer_arr,
                    n=n_arr, prob_target=pp.cpu().numpy().astype(np.float32),
                    cum_relevance=cumrel[oi_arr, b_arr, ki_arr].astype(np.float32),
                    rel_total=rel_total[b_arr].astype(np.float32)))
                baseline_df = pl.DataFrame(dict(
                    perturbation=base_pert, ordering=base_ord, block=base_block, layer=base_layer,
                    n=np.zeros(len(base), np.int32),
                    prob_target=np.full(len(base), p_base, np.float32),
                    cum_relevance=np.zeros(len(base), np.float32),
                    rel_total=rel_total[base_block].astype(np.float32)))
                img_dfs.append(pl.concat([ranked_df, baseline_df]).with_columns(
                    image_idx=pl.lit(image_idx, pl.Int64), **{"class": pl.lit(c, pl.Int32)},
                    dataset=pl.lit(key), concept=pl.lit(concept_name), config=pl.lit(config_name),
                    site=pl.lit(site), n_detectors=pl.lit(D.shape[0], pl.Int32),
                    prob_baseline=pl.lit(p_base, pl.Float32)))
            print(f"[{config_name}/{key}/{concept_name}] class {c} done ({len(sel[c])} imgs)")
    finally:
        for hd in handles:
            hd.remove()

    # ── Stage 5 · assemble, derive Δ in probability space, write ──────────────
    df = pl.concat(img_dfs).with_columns(
        delta_prob=pl.col("prob_target") / pl.col("prob_baseline"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    msuff = f"_m{int(feat_dim)}" if concept_name == "sae" else ""
    out_path = OUT_DIR / f"flipping_{config_name}_{concept_name}{msuff}_{site}_{key}.parquet"
    df.write_parquet(out_path)
    print(f"[{config_name}/{key}/{concept_name}/{site}] wrote {len(df)} rows → {out_path}")
    return out_path, dict(probe=str(probe_path), config=config_name, site=site,
                          concept=concept_name, feat_dim=int(feat_dim),
                          n_detectors=int(D.shape[0]), n_blocks=n_blocks, n_grid=n_grid,
                          sae_fvu=fvu, counts=counts)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    datasets: List[str] = typer.Option(list(DATASETS), "--datasets", help="dataset keys"),
    config: List[str] = typer.Option(["cp_lrp_baseline"], "--config",
                                     help=f"LRP recipe(s): {sorted(COMPOSITES)}"),
    concept: List[str] = typer.Option(["head"], "--concept", help="head | embed_dim | sae"),
    site: List[str] = typer.Option(["proj_drop"], "--site", help="proj_drop | residual"),
    perturbation: List[str] = typer.Option(["zero"], "--perturbation",
                                           help="zero | mean | sign_flip"),
    classes: List[int] = typer.Option([], "--classes",
                                      help="class subset (default: all classes of each dataset)"),
    n_images: int = typer.Option(50, "--n-images", help="correctly-classified images per class"),
    max_steps: int = typer.Option(0, "--max-steps", help="cap cumulative-flip grid (0=one-by-one; "
                                  "use e.g. 48 for the large SAE basis)"),
    chunk_size: int = typer.Option(4096, "--chunk-size", help="configs per GPU forward"),
    precision: str = typer.Option("bf16", "--precision", help="bf16 (model-native) | fp32"),
    sae_m: List[int] = typer.Option([], "--sae-m", help="SAE dictionary size(s) for --concept sae; "
                                    "selects which trained SAE to splice (empty=legacy m=3072). "
                                    "Pass several to sweep dimensionality."),
    device: Optional[str] = typer.Option(None, "--device"),
):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ms = max_steps or None
    sae_ms: List[Optional[int]] = list(sae_m) or [None]
    print(f"device={dev} configs={config} datasets={datasets} concepts={concept} sites={site} "
          f"perturbations={perturbation} classes={classes or 'all'} n_images={n_images} "
          f"max_steps={ms} sae_m={sae_ms} precision={precision}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta.update(experiment="concept_flipping", metric="delta_prob", split="train",
                perturbations=list(perturbation), orderings=list(ORDERINGS),
                n_images_per_class=n_images, classes=list(classes) or "all",
                max_steps=ms, precision=precision)
    runs = meta.setdefault("runs", {})
    for cfg_name in config:
        runs.setdefault(cfg_name, {})
        for cname in concept:
            runs[cfg_name].setdefault(cname, {})
            for s in site:
                runs[cfg_name][cname].setdefault(s, {})
                # SAE dim sweep only applies to --concept sae; other concepts run once.
                dims = sae_ms if cname == "sae" else [None]
                for m in dims:
                    for key in datasets:
                        # Resume across pod bounces: skip a run whose parquet exists.
                        msuff = f"_m{m}" if (cname == "sae" and m is not None) else ""
                        out_p = OUT_DIR / f"flipping_{cfg_name}_{cname}{msuff}_{s}_{key}.parquet"
                        if out_p.is_file():
                            print(f"[skip] {out_p.name} exists")
                            continue
                        _, dmeta = run_dataset(key, cfg_name, cname, perturbation, classes or None,
                                               n_images, dev, chunk_size, precision, site=s,
                                               max_steps=ms, sae_m=m)
                        rkey = key if m is None else f"{key}_m{dmeta['feat_dim']}"
                        runs[cfg_name][cname][s][rkey] = dmeta
                        meta_path.write_text(json.dumps(meta, indent=2))  # checkpoint per dataset
    print(f"meta → {meta_path}")


if __name__ == "__main__":
    app()
