"""CRP gallery — compute CRP saliency maps + representative images for a static web.

This is the single, idiomatic generator behind the static gallery at
``webapp/crp_gallery/`` (served via ``webshare``). Given a model (backbone+head,
the systematic ``experiments.models`` way), an LRP/CRP recipe from
``COMPOSITES``, a probe site and a set of blocks, it produces — per concept
detector — the standard CRP-paper presentation: the top reference images and
their *conditional* relevance/saliency maps, as matplotlib figures (png + pdf),
exactly like the notebooks.

Design (see the approved plan / AGENTS.md):

* **One ENTRY per concept detector** at a (model+dataset, config, layer). Entries
  are amendable: rerunning with a higher ``--n`` or explicit ``--detectors`` only
  ADDS/updates entry dirs (merge-not-wipe). ``manifest.json`` is always rebuilt by
  *scanning the output tree* — it lists exactly what is present, so an empty tree
  ⇒ empty selects on the web.
* **Track only the recompute metadata.** Every ``compute`` appends/merges a line
  in ``jobs.jsonl`` (the only tracked output) carrying the full spec; ``replay``
  re-runs those lines to regenerate the gallery after a restart/redeploy. The
  figures / manifest / FV indices are gitignored (regenerable).
* **Composites are taken AS-IS** from the python source; the web only displays
  their summary + hyperparameters (``composite.json``). Nothing here defines or
  mutates a composite.
* **Relevance-sign flavours.** By default negative relevance is dropped from the
  FV index, the rankings, the scores and the displayed maps (the "clamped"
  flavour); ``--include-negative`` computes a parallel flavour that keeps it. The
  two live in separate caches (``<config>--negincl``) and entry subtrees
  (``<config>/negincl/``) and the web suffixes the instance labels accordingly.

Compute nothing on your own — only the combinations explicitly requested.

Run (GPFS-safe; ``uv run`` deadlocks on this venv)::

    VIRTUAL_ENV=$PWD/.venv .venv/bin/python -m experiments.crp_gallery compute \
        --base vit_small --dataset dsprites --config cp_lrp_baseline \
        --site proj_drop --blocks 10 --blocks 11 --concept embed_dim --n 5

    VIRTUAL_ENV=$PWD/.venv .venv/bin/python -m experiments.crp_gallery replay
    VIRTUAL_ENV=$PWD/.venv .venv/bin/python -m experiments.crp_gallery manifest
"""
from __future__ import annotations

import inspect
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

from torchvision.transforms.functional import gaussian_blur

from zennit_extensions.lrp_composites import COMPOSITES as _BASE_COMPOSITES

# Gallery config aliases: same composite math, distinct config name so a set of
# entries lands in its own labeled gallery instance. ``cp_lrp_baseline_optimal_actmax``
# is the heuristic-optimal activation-max view (experiments/scripts/gallery_optimal_actmax.py).
COMPOSITES = {**_BASE_COMPOSITES,
              "cp_lrp_baseline_optimal_actmax": _BASE_COMPOSITES["cp_lrp_baseline"]}
from experiments import storage
from experiments import fv_aligned
from experiments.gradinput import (
    GradTimesInputAttribution, GradTimesInputFeatureVisualization)
from crp.concepts import HeadConcept, EmbeddingDimConcept
from crp.helper import load_maximization
from crp.image import get_crop_range, imgify, plot_grid, vis_img_heatmap, vis_opaque_img
from experiments.model_datasets import find
from experiments.models import (
    select_correct,
)
from experiments.datasets import EVAL_DATASETS

REPO_ROOT = Path(__file__).resolve().parents[1]
GALLERY_DIR = REPO_ROOT / "webapp" / "crp_gallery"
FIG_DIR = GALLERY_DIR / "figures"
JOBS_PATH = GALLERY_DIR / "jobs.jsonl"
MANIFEST_PATH = GALLERY_DIR / "manifest.json"
# FV-index cache. The index is an intermediate (regenerable) artefact and a large
# vit_base m=1536 index stalls/races when written straight to network storage, so
# it is BUILT on fast scratch and mirrored to the persistent root (see
# experiments.storage; roots are deploy-configured, not detected). CACHE_ROOT is
# the working/scratch location (CRP_GALLERY_CACHE overrides it); CACHE_MIRROR is
# the durable copy under the persistent root that survives a pod bounce. Figures
# themselves live in the repo (persistent) — only this cache is scratch-built.
CACHE_ROOT = Path(os.environ.get("CRP_GALLERY_CACHE", str(storage.SCRATCH_ROOT / "crp_gallery_cache")))
CACHE_MIRROR = storage.PERSIST_ROOT / "crp_gallery_cache"

CONCEPTS = ("head", "embed_dim")
# FV index flavours: "original" conditions every image on its ground-truth label
# (stock CRP); "aligned" conditions on the top-3 predicted classes with p > 0.10
# and serves representatives only under a matching conditioning class (see
# experiments.fv_aligned).
FV_CLASS_LABELS = {
    "original": "original (ground-truth conditioning)",
    "aligned": "condition-class-aligned (top-3 predicted)",
}

# Reference-ranking modes: which per-concept score orders the representative
# samples in the FV index. quantity ∈ {relevance, activation} × reduction over
# the concept's neurons ∈ {sum, max} — served from the corresponding
# ``{Rel|Act}Max_{sum|max}_normed`` store.
REF_MODES = {
    "relsum": ("relevance", "sum", "RelSum — relevance, token-sum"),
    "relmax": ("relevance", "max", "RelMax — relevance, token-max"),
    "actsum": ("activation", "sum", "ActSum — activation, token-sum"),
    "actmax": ("activation", "max", "ActMax — activation, token-max"),
}

app = typer.Typer(add_completion=False, help=__doc__)


# ─────────────────────────────────────────────────────────────────────────────
# Concepts / layers / ranking
# ─────────────────────────────────────────────────────────────────────────────

def make_concept(kind: str, num_heads: int):
    if kind == "head":
        return HeadConcept(num_heads=num_heads)
    if kind == "embed_dim":
        return EmbeddingDimConcept(num_heads=num_heads)
    raise typer.BadParameter(f"--concept must be one of {CONCEPTS}, got {kind!r}")


# Probe-site layer names, written out in full. Every zoo backbone is a 12-block
# ViT, so these ARE the probed layers — stated explicitly, never derived from
# the model at runtime (the derivation repeatedly probed layers that did not
# match the actual architecture). resolve_layers validates every name against
# the model, so a backbone these don't fit fails at startup, not mid-run.
# Mirrored in experiments/concept_flipping.py — keep the two copies in sync.
SITE_LAYERS = {
    "proj_drop": [
        "backbone.blocks.0.attn.proj_drop",
        "backbone.blocks.1.attn.proj_drop",
        "backbone.blocks.2.attn.proj_drop",
        "backbone.blocks.3.attn.proj_drop",
        "backbone.blocks.4.attn.proj_drop",
        "backbone.blocks.5.attn.proj_drop",
        "backbone.blocks.6.attn.proj_drop",
        "backbone.blocks.7.attn.proj_drop",
        "backbone.blocks.8.attn.proj_drop",
        "backbone.blocks.9.attn.proj_drop",
        "backbone.blocks.10.attn.proj_drop",
        "backbone.blocks.11.attn.proj_drop",
    ],
    "residual": [
        "backbone.blocks.0",
        "backbone.blocks.1",
        "backbone.blocks.2",
        "backbone.blocks.3",
        "backbone.blocks.4",
        "backbone.blocks.5",
        "backbone.blocks.6",
        "backbone.blocks.7",
        "backbone.blocks.8",
        "backbone.blocks.9",
        "backbone.blocks.10",
        "backbone.blocks.11",
    ],
    # Q/K probe sites — the QInspectionLayer/KInspectionLayer identities the
    # attention-unfolding canonizer inserts on the (B, N, embed_dim) q/k tensors.
    # These exist ONLY inside a canonized composite context, so resolve_layers
    # validates them against the canonized model (see PROBE_SITES).
    "query": [f"backbone.blocks.{b}.attn.q_lrp_probe" for b in range(12)],
    "key": [f"backbone.blocks.{b}.attn.k_lrp_probe" for b in range(12)],
    # V probe site — the VInspectionLayer identity the attention-unfolding
    # canonizer inserts on the (B, N, embed_dim) value tensor, alongside q/k
    # (attention_unfolded.py). Same canonized-context caveat as query/key.
    "value": [f"backbone.blocks.{b}.attn.v_lrp_probe" for b in range(12)],
}

# Sites whose layer names materialize only after canonization (Q/K/V probes).
PROBE_SITES = frozenset({"query", "key", "value"})


def resolve_layers(model, site: str, blocks: List[int], *,
                   known: Optional[set] = None) -> List[Tuple[int, str]]:
    """``(block, layer_name)`` per requested block at the probe site. The names
    are the explicit :data:`SITE_LAYERS` strings, validated against ``known`` (the
    bare model's module names by default; the caller passes the canonized set for
    :data:`PROBE_SITES`) — a name that does not resolve is a hard error."""
    if site not in SITE_LAYERS:
        raise typer.BadParameter(
            f"--site must be one of {tuple(SITE_LAYERS)}, got {site!r}")
    names = SITE_LAYERS[site]
    known = known if known is not None else set(dict(model.named_modules()))
    missing = [n for n in names if n not in known]
    if missing:
        raise typer.BadParameter(
            f"site {site!r} names layers this model does not have: {missing}. "
            f"SITE_LAYERS is written for the 12-block zoo ViTs — extend the "
            f"list for other architectures.")
    out = []
    for b in blocks:
        if not 0 <= b < len(names):
            raise typer.BadParameter(f"block {b} out of range 0..{len(names) - 1}")
        out.append((b, names[b]))
    return out


def rank_scores(rank_mode: str, *, attribution, ds, sel, layer, concept, composite,
                normalize, device, fv, batch_size: int = 32,
                include_negative: bool = False) -> np.ndarray:
    """Per-detector relevance score vector for one layer (higher = more relevant).

    * ``class_conditional`` (default) — mean over a sample of correctly-classified
      images of ``concept.attribute(R[layer])`` with relevance initialised at the
      true target logit (``{"y":[c]}``). Idiom from ``head_relevance_by_class``.
    * ``fv_index`` — mean over the FV RelMax index (whole-dataset, target-agnostic).

    ``include_negative=False`` (default) scores by positive relevance only.
    """
    if rank_mode == "fv_index":
        _, rel_c_sorted, _ = load_maximization(fv.RelMax.PATH, layer)
        return np.asarray(rel_c_sorted).mean(axis=0)
    if rank_mode != "class_conditional":
        raise typer.BadParameter(f"--rank must be class_conditional|fv_index, got {rank_mode!r}")
    total, n_imgs = None, 0
    for c, idxs in sel.items():
        if not idxs:
            continue
        x = torch.stack([ds[i][0] for i in idxs]).to(device)
        x = normalize(x).requires_grad_(True)
        res = attribution(x, [{"y": [int(c)]}], composite, record_layer=[layer])
        rel = res.relevances[layer]
        det = concept.attribute(rel if include_negative else rel.clamp(min=0),
                                abs_norm=False)  # (B, n_det)
        s = det.sum(0).detach().cpu().numpy()
        total = s if total is None else total + s
        n_imgs += det.shape[0]
    if total is None:
        raise RuntimeError("no correctly-classified images to rank from")
    return total / max(n_imgs, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Output tree: composite meta, per-entry figures, jobs, manifest
# ─────────────────────────────────────────────────────────────────────────────

def composite_meta(name: str, comp_cls, site: str) -> dict:
    """Human-readable summary of a composite, pulled straight from the source
    (no duplication): class docstring first line + the class source text."""
    try:
        build_source = inspect.getsource(comp_cls).strip()
    except (OSError, TypeError):
        build_source = ""
    doc = (comp_cls.__doc__ or "").strip()
    return {
        "name": name,
        "class": comp_cls.__name__,
        "description": doc.splitlines()[0] if doc else "",
        "site": site,
        "build_source": build_source,
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def concept_kind_desc(concept_kind: str, site: str) -> str:
    """One-line description of the concept *basis* for the web (the composite
    panel is config-level and shared, so the basis distinction must live per
    layer)."""
    if concept_kind == "embed_dim":
        return ("Axis-aligned basis — one concept per embedding dimension (the standard CRP basis). "
                f"Relevance read directly at the {site} site.")
    if concept_kind == "head":
        return f"Attention-head basis — one concept per attention head, read at the {site} site."
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Single-image samples: a small FIXED set of inputs shown across every layer and
# instance so the user can compare "how does concept #k respond to *this* image"
# (lizard vs cheeseburger …) — the local/instance-conditional CRP view, next to
# the dataset-aggregate reference-image view.
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet-1k class idx → display name. Six visually distinct classes; an
# optional third element pins an explicit dataset index (else: first val image
# of that class).
IMAGENET_SAMPLES: List[Tuple] = [
    (39, "lizard"), (933, "cheeseburger"), (1, "goldfish"),
    (817, "sports_car"), (985, "daisy"), (207, "golden_retriever"),
    # tutorials/images/lizard.jpg == val index 2339 (true label 46, models
    # predict 40) — pinned for visual parity with the lxt_reference notebooks,
    # conditioned on the predicted class like the notebooks.
    (40, "lizard_nb", 2339),
]


def _ds_labels(ds) -> Optional[List[int]]:
    """Cheap per-sample class labels without decoding images (parquet/file lists
    expose them directly). ``None`` if the dataset has no cheap label list."""
    if hasattr(ds, "rows"):
        return [int(c) for _, c in ds.rows]
    if hasattr(ds, "items"):
        return [int(c) for _, c in ds.items]
    if hasattr(ds, "labels"):
        return [int(c) for c in ds.labels]
    return None


def pick_samples(dataset: str, ds) -> List[dict]:
    """The fixed comparison images for a dataset: ``[{key,label,ds_index,target}]``.

    * ImageNet — the six named classes (:data:`IMAGENET_SAMPLES`), one val image each.
    * other datasets — up to six images spread across classes (round-robin), so the
      set is diverse for datasets with few classes (funny_birds, dsprites)."""
    labels = _ds_labels(ds)
    if labels is None:
        return []
    if dataset == "imagenet":
        out = []
        for entry in IMAGENET_SAMPLES:
            cls, name = entry[0], entry[1]
            idx = entry[2] if len(entry) > 2 else                 next((i for i, l in enumerate(labels) if l == cls), None)
            if idx is not None:
                out.append({"key": name, "label": f"{name} · class {cls}",
                            "ds_index": idx, "target": cls})
        return out
    by_class: Dict[int, List[int]] = {}
    for i, l in enumerate(labels):
        by_class.setdefault(l, []).append(i)
    classes = sorted(by_class)
    out: List[dict] = []
    while len(out) < 6 and any(by_class[c] for c in classes):
        for c in classes:
            if by_class[c]:
                i = by_class[c].pop(0)
                out.append({"key": f"c{c}_{i}", "label": f"class {c} · #{i}",
                            "ds_index": i, "target": c})
                if len(out) >= 6:
                    break
    return out


def local_relevances(attribution, x, target: int, layer: str, *, concept, composite,
                     normalize, device: str, include_negative: bool = False) -> np.ndarray:
    """Per-detector relevance of ONE input image at ``layer`` (local analysis):
    initialise relevance at the image's true class and read it on each concept.
    Returns a ``(n_det,)`` vector — argsort gives the detectors most relevant to
    *this* image. ``include_negative=False`` (default) scores positive relevance
    only."""
    xin = normalize(x[None].to(device)).requires_grad_(True)
    res = attribution(xin, [{"y": [int(target)]}], composite, record_layer=[layer],
                      mask_map=concept.mask)
    rel = res.relevances[layer]
    rel = rel if include_negative else rel.clamp(min=0)
    return concept.attribute(rel, abs_norm=False)[0].detach().cpu().numpy()


def render_local_entry(fv, attribution, ds, x, target: int, layer: str, cid: int, *,
                       mode: str, n_ref: int, composite, concept, normalize, device: str,
                       crop: bool, plot: str, out_dir: Path, meta_extra: dict,
                       fv_class: str = "original", include_negative: bool = False) -> float:
    """Local analysis of one detector for one input image: the leftmost column is
    the query image + its *conditional* CRP heatmap; the remaining columns are the
    detector's dataset **representatives** so the reader can tell what the locally-
    relevant concept actually is. Both are class-conditional (see
    :func:`class_conditional_references`); under the aligned index the
    representatives come from the bucket of the query's own conditioning class,
    so index conditioning matches the explanation. png+pdf + meta.json.
    Returns the query image's relevance on the concept."""
    ref_s, ref_h = class_conditional_references(
        attribution, fv, ds, layer, cid, n_ref=n_ref, mode=mode, composite=composite,
        concept=concept, normalize=normalize, device=device, fv_class=fv_class,
        query_target=(int(target) if fv_class == "aligned" else None),
        include_negative=include_negative)
    xin = normalize(x[None].to(device)).requires_grad_(True)
    res = attribution(xin, [{layer: [int(cid)], "y": [int(target)]}], composite,
                      record_layer=[layer], mask_map=concept.mask)
    local_h = res.heatmap.detach().cpu()                     # (1, H, W)
    rel_vec = res.relevances[layer]
    if not include_negative:
        local_h = local_h.clamp(min=0)                       # clamped flavour: positive only
        rel_vec = rel_vec.clamp(min=0)
    rel = float(concept.attribute(rel_vec, abs_norm=False)[0, int(cid)])
    # Column 0 = query image + local heatmap; columns 1.. = global representatives.
    imgs = torch.cat([x[None].detach().cpu(), ref_s.detach().cpu()], dim=0)
    heats = torch.cat([local_h, ref_h.detach().cpu()], dim=0)
    rows, nsub, row_lbl = build_rows(imgs, heats, plot=plot, crop=crop,
                                     signed=include_negative)
    ref = {cid: rows}
    ncols = len(rows[0]) if nsub > 1 else len(rows)
    fig = plot_grid(ref, figsize=(1.9 * ncols, 2.1 * nsub + 0.5))
    _entry_title(fig, f"#{cid} · block {meta_extra['block']} · "
                      f"local rank {meta_extra['rank']} · [query | representatives]")
    _row_labels(fig, ncols, row_lbl)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "entry.png", dpi=130, bbox_inches="tight")
    fig.savefig(out_dir / "entry.pdf", bbox_inches="tight")
    plt.close(fig)
    meta = {"concept_id": int(cid), "mode": mode, "n_ref": n_ref, "crop": crop, "plot": plot,
            "relevance": rel, "generated": _now(), **meta_extra}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return rel


def save_sample_image(ds, sample: dict, out_path: Path) -> None:
    """Save the raw (un-normalized) sample image once for the web thumbnail."""
    if out_path.exists():
        return
    x = ds[sample["ds_index"]][0].detach().cpu()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgify(x).save(out_path)


def save_sample_heat(attribution, x, target: int, *, composite, normalize, device: str,
                     out_path: Path, include_negative: bool = False) -> None:
    """Save the sample input's OWN overall relevance heatmap — the full-model LRP
    attribution to its true class (all concepts, input space), the standard CRP
    saliency for that image. Instance-specific (the composite/model differ per
    basis and relevance-sign flavour), so stored per concept_kind and flavour.
    Always (re)written."""
    xin = normalize(x[None].to(device)).requires_grad_(True)
    res = attribution(xin, [{"y": [int(target)]}], composite)   # no layer cond → total heatmap
    heat = res.heatmap.detach().cpu()[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ViT input relevance is sparse — a few extreme pixels wash out a plain
    # norm. Clip to a high percentile so the structure is visible.
    if include_negative:
        v = float(np.quantile(heat.abs().numpy(), 0.995)) or 1.0
        imgify(heat, cmap="bwr", vmin=-v, vmax=v, symmetric=True).save(out_path)
    else:
        heat = heat.clamp(min=0)                                 # clamped flavour: positive only
        vmax = float(np.quantile(heat.numpy(), 0.995))
        if vmax <= 0:
            vmax = float(heat.max()) or 1.0
        imgify(heat, cmap="wred", vmin=0, vmax=vmax, symmetric=False).save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Competing XAI saliency (Chefer / rollout / occlusion) next to LRP, per sample.
# Model-level (composite-independent) → stored once at md level under _sample_xai/
# <method>/<key>.png. All maps are class-conditional on the PREDICTED class and
# patch-aggregated to the model's patch grid (see experiments.xai_methods).
# ─────────────────────────────────────────────────────────────────────────────

def save_sample_xai(model, attribution, x, *, composite, normalize, device: str,
                    out_root: Path, key: str) -> int:
    """Render the competing input-saliency maps for one image next to the LRP
    baseline: ``_sample_xai/{lrp,chefer,rollout,occlusion}/<key>.png``. Every map
    is conditioned on the model's OWN predicted class (what it actually used).
    Idempotent-friendly (always rewritten). Returns the predicted class."""
    import experiments.xai_methods as xm
    x = x.detach()
    with torch.no_grad():
        pred = int(model(normalize(x[None].to(device))).argmax(-1))
    n_prefix, grid, patch = xm.model_geometry(model, x[None])
    xn = normalize(x[None].to(device))
    res = int(x.shape[-1])
    maps = {"lrp": xm.lrp_patch(attribution, xn, pred, composite, grid=grid, patch=patch)}
    _, attns = xm.capture_attention(model, xn)                 # detached (rollout)
    maps["rollout"] = xm.attention_rollout(attns, n_prefix, grid)[0]
    maps["chefer"] = xm.chefer_relevance(model, xn, [pred], n_prefix=n_prefix, grid=grid)[0]
    maps["occlusion"] = xm.occlusion_deltap(model, normalize, x, pred, grid=grid, patch=patch)
    for method, pm in maps.items():
        xm.render_patch_map(pm, out_root / method / f"{key}.png", res=res)
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# OOD (outlier) token maps: dual-site (proj_drop + residual) per-block token L2
# norms with per-sample μ+4σ outlier flags, DINOv3-aware (patch stats exclude
# cls+registers; register tokens shown distinctly). Renders _normmaps/<key>.png
# and returns the OOD patch-token count (union over sites). Class-agnostic.
# ─────────────────────────────────────────────────────────────────────────────

class _DualSiteNormRecorder:
    """Forward hooks capturing per-token L2 norms at both concept sites for all
    blocks in one pass: block output (residual stream) and ``attn.proj_drop``."""

    def __init__(self, backbone):
        self.norms: Dict[str, Dict[int, np.ndarray]] = {"residual": {}, "proj_drop": {}}
        self.handles = []
        for b, blk in enumerate(backbone.blocks):
            def hook_res(mod, args, out, b=b):
                self.norms["residual"][b] = out.detach().norm(dim=-1).float().cpu().numpy()
            def hook_proj(mod, args, out, b=b):
                self.norms["proj_drop"][b] = out.detach().norm(dim=-1).float().cpu().numpy()
            self.handles.append(blk.register_forward_hook(hook_res))
            self.handles.append(blk.attn.proj_drop.register_forward_hook(hook_proj))

    def stack(self, site: str) -> np.ndarray:                  # (n_blocks, B, T)
        d = self.norms[site]
        return np.stack([d[b] for b in sorted(d)])

    def remove(self):
        for h in self.handles:
            h.remove()


def save_sample_ood(model, x, *, normalize, device: str, out_path: Path,
                    n_prefix: int, n_reg: int, k: float = 4.0) -> int:
    """Dual-site token-norm map for one image + OOD flag overlay; returns the
    number of OOD patch tokens (union over sites of tokens flagged at any block).

    Patch statistics (μ+kσ) exclude the ``n_prefix`` prefix tokens (cls +
    registers); register tokens (indices ``1..n_reg``) are drawn distinctly in a
    small strip so the reader can tell them apart from spatial patches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rec = _DualSiteNormRecorder(model.backbone)
    try:
        with torch.no_grad():
            model(normalize(x[None].to(device)))
        norms = {s: rec.stack(s)[:, 0] for s in ("residual", "proj_drop")}  # (nblk, T)
    finally:
        rec.remove()

    n_blk = norms["residual"].shape[0]
    grid = int(round(math.sqrt(norms["residual"].shape[1] - n_prefix)))
    site_desc = {"residual": "residual stream (block output)",
                 "proj_drop": "attention output projection (proj_drop)"}
    union = np.zeros(grid * grid, dtype=bool)
    fig = plt.figure(figsize=(13.5, 10.2))
    subfigs = fig.subfigures(2, 1, hspace=0.04)
    for sf, site in zip(subfigs, ("residual", "proj_drop")):
        sf.suptitle(site_desc[site], fontsize=11, fontweight="bold")
        axes = sf.subplots(2, (n_blk + 1) // 2)
        for b, ax in enumerate(axes.ravel()):
            if b >= n_blk:
                ax.axis("off"); continue
            row = norms[site][b]
            patch = row[n_prefix:n_prefix + grid * grid]
            mu, sd = patch.mean(), patch.std()
            flags = patch > mu + k * sd
            union |= flags
            v = patch.reshape(grid, grid)
            vn = (v - v.min()) / (v.max() - v.min() + 1e-12)
            ax.imshow(vn, cmap="viridis", vmin=0, vmax=1)
            for r, c in zip(*np.nonzero(flags.reshape(grid, grid))):
                ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                                edgecolor="magenta", linewidth=1.6))
            # Register tokens (DINOv3): distinct strip below the patch grid.
            reg_note = ""
            if n_reg > 0:
                reg = row[1:1 + n_reg]
                reg_out = int((reg > mu + k * sd).sum())
                strip = ax.inset_axes([0.0, -0.20, 1.0, 0.12])
                strip.imshow(reg[None] / (patch.max() + 1e-12), cmap="magma",
                             vmin=0, vmax=1, aspect="auto")
                for j in range(n_reg):
                    strip.add_patch(mpatches.Rectangle((j - 0.5, -0.5), 1, 1, fill=False,
                                                       edgecolor="cyan", linewidth=1.4))
                strip.set_xticks([]); strip.set_yticks([])
                strip.set_ylabel("reg", fontsize=6, rotation=0, labelpad=8, va="center")
                reg_note = f" · reg×{reg_out}"
            ax.set_title(f"block {b} · {int(flags.sum())} flagged{reg_note}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    prefix_desc = "cls" + (f"+{n_reg} registers" if n_reg else "")
    fig.suptitle(f"Per-block token L2 norms, normalized to [0,1] per block/site (viridis); "
                 f"magenta = patch flagged (norm > mean + {k:g}·sd over this sample's "
                 f"{grid * grid} patch tokens; {prefix_desc} excluded from the stats"
                 + (". Cyan strip = register-token norms (magma)." if n_reg else "."),
                 fontsize=10, y=1.045)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return int(union.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Instance = (config, concept basis) — one instance per basis, so the layer
# layer — so the layer dropdown lists each block exactly once per instance.
# ─────────────────────────────────────────────────────────────────────────────

def instance_key(config: str, concept_kind: str, include_negative: bool = False) -> str:
    key = f"{config}::{concept_kind}"
    return f"{key}::negincl" if include_negative else key


# Human-readable instance names for configs whose raw string is opaque.
_CONFIG_LABELS = {
    "cp_lrp_baseline_optimal_actmax": "CP-LRP · heuristic-optimal detectors · activation-max",
}


def instance_label(config: str, concept_kind: str, include_negative: bool = False) -> str:
    suffix = " (negative included)" if include_negative else " (neg. clamped away)"
    base = _CONFIG_LABELS.get(config, config)
    if concept_kind == "embed_dim":
        return f"{base} · axis-aligned{suffix}"
    if concept_kind == "head":
        return f"{base} · heads{suffix}"
    return f"{base} · {concept_kind}{suffix}"


def _entry_title(fig, text: str) -> None:
    """Short caption placed ABOVE the axes (y>1) so it never overlaps the image;
    ``bbox_inches='tight'`` at save time keeps it in frame."""
    fig.suptitle(text, fontsize=9, y=1.02)


def _row_labels(fig, ncols: int, labels: Tuple[str, ...]) -> None:
    """Name each sub-row at its leftmost axis. ``plot_grid`` adds axes row-major
    (sub-row outer, column inner), so axis ``sr * ncols`` starts sub-row ``sr``.
    Overwrites plot_grid's concept-id ylabel — the id is already in the title."""
    for sr, text in enumerate(labels):
        i = sr * ncols
        if i < len(fig.axes):
            fig.axes[i].set_ylabel(text, fontsize=7, labelpad=2)


def class_conditional_references(attribution, fv, ds, layer: str, cid: int, *, n_ref: int,
                                 mode: str, composite, concept, normalize, device: str,
                                 fv_class: str = "original",
                                 query_target: Optional[int] = None,
                                 include_negative: bool = False):
    """Top-``n_ref`` reference samples for a concept **with class-conditional CRP
    heatmaps**, as defined in the CRP paper.

    Why not ``FeatureVisualization.get_max_reference``: it computes its heatmaps via
    ``_attribution_on_reference`` with ``start_layer=<layer>``, and with a start
    layer :class:`CondAttribution` seeds the backward pass with the layer
    ACTIVATION and drops the output-class condition entirely::

        pred = layer_out[start_layer]
        grad_mask = self.relevance_init(pred.detach().clone(), None, init_rel)

    The paper propagates "backwards through the network, starting from the output
    until the input layer" under a condition set that carries the class alongside
    the concept — ``theta = {L:{dog}, l:{fur}}`` — and Fig. 4a distinguishes
    "per-channel activation maps" from "respective **true class** CRP relevance
    maps". The FV index is already built the paper's way (``run_distributed`` uses
    ``conditions = [{MODEL_OUTPUT_NAME: [t]}]`` over the dataset targets), so the
    activation seeding also made the *displayed* heatmap inconsistent with the
    relevance that ranked the sample in the first place.

    Here the reference indices come from the index, but each heatmap is recomputed
    the paper's way: initialise at the output on that sample's true class, mask at
    the concept, propagate to the input. Conditions are per batch element
    (``CondAttribution.broadcast`` maps ``conditions[i]`` to sample ``i``), so all
    ``n_ref`` references are done in one batched pass.

    ``fv_class`` selects the index flavour. ``"original"`` (stock): indices from
    the target-agnostic RelMax store, heatmaps conditioned on each reference's
    ground-truth label. ``"aligned"``: indices from the per-conditioning-class
    RelStats store (see :mod:`experiments.fv_aligned`) and each heatmap is
    conditioned on the class the reference was INDEXED under — with
    ``query_target`` set (local view) only the bucket of that class is served,
    so the conditioning matches the explanation's class exactly."""
    if fv_class == "aligned":
        if mode != "relevance":
            raise ValueError("the aligned index serves relevance references only")
        if query_target is None:
            idxs, ys = fv_aligned.aligned_references_merged(fv, layer, cid, n_ref)
        else:
            idxs, ys = fv_aligned.aligned_references_for_class(
                fv, layer, cid, int(query_target), n_ref)
    else:
        path = fv.RelMax.PATH if mode == "relevance" else fv.ActMax.PATH
        d_sorted, _, _ = load_maximization(path, layer)
        idxs = [int(i) for i in np.asarray(d_sorted)[:n_ref, int(cid)]]
        ys = [int(ds[di][1]) for di in idxs]
    batch = torch.stack([ds[di][0] for di in idxs])
    xin = normalize(batch.to(device)).requires_grad_(True)
    conds = [{layer: [int(cid)], "y": [y]} for y in ys]
    res = attribution(xin, conds, composite, mask_map=concept.mask)
    heat = res.heatmap.detach().cpu()
    # clamped flavour (default): only the positive part of the map is shown
    return batch.detach().cpu(), (heat if include_negative else heat.clamp(min=0))


def rf_crop_row(samples, heatmaps, *, vis_th: float = 0.2, crop_th: float = 0.1,
                kernel_size: int = 19, alpha: float = 0.3):
    """Receptive-field crop row, sign-safe and magnitude-based.

    Same recipe as :func:`crp.image.vis_opaque_img` with ``rf=True`` — blur the
    conditional heatmap, keep the box where it exceeds ``crop_th``, fade pixels
    below ``vis_th`` — with TWO changes:

    1. Normalise by ``max(|R|)`` instead of ``crp.helper.max_norm``'s
       ``R / R.max()``. ``max_norm`` is unsafe for signed bases: a ViT embedding
       dimension can produce a conditional heatmap that is negative almost
       everywhere (its input-space attribution is net inhibitory); the blurred
       map's max is then itself negative, so dividing by it FLIPS every pixel
       positive. The mask becomes all-True and the crop box the full frame —
       the panel then reads as "this concept covers the whole image" when the
       truth is "there is no positive evidence here".
    2. Crop and fade on ``|fh|``. High-**negative** relevance is evidence too —
       the CRP paper displays strong inhibitory regions alongside excitatory
       ones — so the crop box and the unfaded region localise the strongest
       conditional evidence of EITHER sign. The sign itself remains readable in
       the bwr relevance row above the crop. A zero/structureless map still
       fades dark (no evidence at all)."""
    out = []
    for i in range(len(samples)):
        img, heat = samples[i], heatmaps[i]
        blurred = gaussian_blur(heat.unsqueeze(0), kernel_size=kernel_size)[0]
        fh = blurred / (blurred.abs().max() + 1e-10)      # sign-safe normalisation
        fa = fh.abs()                                     # |R| decides crop + fade
        vis_mask = fa > vis_th
        r1, r2, c1, c2 = get_crop_range(fa, crop_th)
        img_t, mask_t = img[..., r1:r2, c1:c2], vis_mask[r1:r2, c1:c2]
        if img_t.sum() != 0 and mask_t.sum() != 0:
            img, vis_mask = img_t, mask_t
        # Fix the display range BEFORE fading. ``zennit.image.imgify`` min-max
        # normalises, which exactly undoes a *uniform* scale — so an all-False mask
        # (nothing passes vis_th) would render identical to the un-faded image
        # instead of fading out. Mapping to [0,1] first and pinning vmin/vmax keeps
        # the fade visible, so "no positive evidence" reads as a dark panel.
        img = img.detach().cpu().float()
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / (hi - lo + 1e-10)
        img = img * vis_mask + img * (~vis_mask) * alpha
        out.append(imgify(img, vmin=0.0, vmax=1.0, symmetric=False))
    return out


def build_rows(samples, heatmaps, *, plot: str, crop: bool, signed: bool = False):
    """Sub-rows of one detector's figure. Returns ``(rows, nsub, row_labels)``.

    ``heat_rf`` (default) shows the three views together: the reference image, its
    conditional relevance heatmap, and the **receptive-field crop** — the image
    clipped to the heatmap's high-relevance box (``crp.image.get_crop_range``) with
    low-relevance pixels faded (``vis_opaque_img``). The crop answers "which part of
    the image does this concept latch onto?" without having to read heatmap colours.
    The RF row is always cropped (that is what the row *is*); ``crop`` still governs
    whether the image/heatmap rows are clipped too.

    ``signed=False`` renders positive-only maps (red, the clamped flavour);
    ``signed=True`` renders the signed map (bwr, the include-negative flavour)."""
    if plot == "opaque":
        return vis_opaque_img(samples, heatmaps, rf=crop), 1, ("concept",)
    vis_kw = {"cmap": "bwr", "symmetric": True} if signed else \
             {"cmap": "wred", "vmin": 0, "symmetric": False}
    imgs, heats = vis_img_heatmap(samples, heatmaps, rf=crop, **vis_kw)
    if plot != "heat_rf":
        return (imgs, heats), 2, ("image", "relevance")
    rf = rf_crop_row(samples, heatmaps)
    return (imgs, heats, rf), 3, ("image", "relevance", "RF crop")


def render_entry(fv, attribution, ds, layer: str, cid: int, *, mode: str, n_ref: int,
                 composite, concept, normalize, device: str, crop: bool, plot: str,
                 out_dir: Path, meta_extra: dict, fv_class: str = "original",
                 include_negative: bool = False) -> None:
    """Render + write one detector's figure (png+pdf) and meta.json (merge-not-wipe).

    Retrieve the reference images + their **class-conditional** CRP heatmaps (see
    :func:`class_conditional_references`), then present them. ``crop=True`` clips
    each reference to the high-relevance region of its saliency map (the standard
    CRP "receptive field" crop — ``crp.image.get_crop_range`` on the heatmap, via
    the ``rf`` flag of the vis functions; NOT the conv-neuron ``mask_rf`` path).
    """
    samples, heatmaps = class_conditional_references(
        attribution, fv, ds, layer, cid, n_ref=n_ref, mode=mode, composite=composite,
        concept=concept, normalize=normalize, device=device, fv_class=fv_class,
        include_negative=include_negative)
    rows, nsub, row_lbl = build_rows(samples, heatmaps, plot=plot, crop=crop,
                                     signed=include_negative)
    ref = {cid: rows}
    ncols = len(rows[0]) if nsub > 1 else len(rows)
    fig = plot_grid(ref, figsize=(1.9 * ncols, 2.1 * nsub + 0.5))
    _entry_title(fig, f"#{cid} · block {meta_extra['block']}")
    _row_labels(fig, ncols, row_lbl)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "entry.png", dpi=130, bbox_inches="tight")
    fig.savefig(out_dir / "entry.pdf", bbox_inches="tight")
    plt.close(fig)
    meta = {"concept_id": int(cid), "mode": mode, "n_ref": n_ref, "crop": crop, "plot": plot,
            "generated": _now(), **meta_extra}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def record_job(spec: dict) -> None:
    """Append/merge a job line in jobs.jsonl (dedup by
    base,dataset,config,site,concept,fv_class,include_negative)."""
    key = (spec["base"], spec["dataset"], spec["config"], spec["site"],
           spec["concept"], spec.get("fv_class", "original"),
           bool(spec.get("include_negative", False)))
    jobs = []
    if JOBS_PATH.exists():
        for line in JOBS_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            jk = (j["base"], j["dataset"], j["config"], j["site"],
                  j["concept"], j.get("fv_class", "original"),
                  bool(j.get("include_negative", False)))
            if jk != key:
                jobs.append(j)
    jobs.append(spec)
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOBS_PATH.write_text("\n".join(json.dumps(j) for j in jobs) + "\n")


def rebuild_manifest() -> dict:
    """Scan figures/ for entry meta.json + composite.json → manifest.json. The
    tree is the single source of truth, so the web lists exactly what exists.

    Schema: ``models → instances → samples → layers → entries``. An *instance* is
    (config, concept basis) — separate instances per basis, and
    the layer list under each holds every block exactly once. A *sample* is either
    ``"aggregate"`` (dataset reference images) or one fixed input image."""
    models: Dict[str, dict] = {}
    if FIG_DIR.exists():
        for comp_json in sorted(FIG_DIR.glob("*/*/composite.json")):
            config_dir = comp_json.parent
            md = config_dir.parent.name          # <base>_<dataset>
            config = config_dir.name
            mhead = config_dir.parent / "model.json"   # figures/<md>/model.json
            minfo = json.loads(mhead.read_text()) if mhead.exists() else {}
            m = models.setdefault(md, {**minfo, "instances": {}})
            composite = json.loads(comp_json.read_text())
            samples_dir = config_dir.parent / "_samples"
            sample_imgs = ({p.stem: str(p.relative_to(GALLERY_DIR)) for p in samples_dir.glob("*.png")}
                           if samples_dir.exists() else {})
            # Per-sample dual-site (proj_drop+residual) token-norm maps with
            # per-sample μ+4σ outlier flags (md-level, composite-independent;
            # produced by generate_sample_extras / save_sample_ood).
            norm_dir = config_dir.parent / "_normmaps"
            sample_norms = ({p.stem: str(p.relative_to(GALLERY_DIR)) for p in norm_dir.glob("*.png")}
                            if norm_dir.exists() else {})
            # Per-sample OOD patch-token counts (union over sites): _ood.json.
            ood_path = config_dir.parent / "_ood.json"
            ood_counts: Dict[str, dict] = {}
            if ood_path.exists():
                try:
                    ood_counts = json.loads(ood_path.read_text())
                except json.JSONDecodeError:
                    ood_counts = {}
            # Competing-XAI saliency maps per sample (md-level, composite-independent):
            # _sample_xai/<method>/<key>.png → {key: {method: relpath}}.
            xai_root = config_dir.parent / "_sample_xai"
            sample_xai: Dict[str, Dict[str, str]] = {}
            if xai_root.exists():
                for p in xai_root.glob("*/*.png"):
                    sample_xai.setdefault(p.stem, {})[p.parent.name] = str(p.relative_to(GALLERY_DIR))
            # Per-instance (concept_kind) sample relevance heatmaps: _sample_heat/<ck>/<key>.png
            heat_root = config_dir / "_sample_heat"
            sample_heats: Dict[str, Dict[str, str]] = {}
            if heat_root.exists():
                for p in heat_root.glob("*/*.png"):
                    sample_heats.setdefault(p.parent.name, {})[p.stem] = str(p.relative_to(GALLERY_DIR))
            for meta_path in sorted(config_dir.rglob("meta.json")):
                meta = json.loads(meta_path.read_text())
                ck, layer = meta["concept_kind"], meta["layer"]
                sample = meta.get("sample", "aggregate")
                fvc = meta.get("fv_class", "original")
                neg = bool(meta.get("include_negative", False))
                rel = meta_path.parent.relative_to(GALLERY_DIR)
                inst = m["instances"].setdefault(instance_key(config, ck, neg), {
                    "config": config, "basis": ck, "label": instance_label(config, ck, neg),
                    "composite": composite,
                    "concept_desc": concept_kind_desc(ck, meta["site"]),
                    "fv": {}})
                fvrec = inst["fv"].setdefault(fvc, {
                    "label": FV_CLASS_LABELS[fvc], "samples": {}})
                srec = fvrec["samples"].setdefault(sample, {
                    "label": meta.get("sample_label") or ("Aggregate" if sample == "aggregate" else sample),
                    "image": sample_imgs.get(sample),
                    "heat": sample_heats.get(ck + ("--negincl" if neg else ""), {}).get(sample),
                    "normmap": sample_norms.get(sample),
                    "xai": sample_xai.get(sample),
                    "ood_tokens": (ood_counts.get(sample) or {}).get("ood_tokens"),
                    "layers": {}})
                lrec = srec["layers"].setdefault(layer, {
                    "site": meta["site"], "block": meta["block"], "concept_kind": ck,
                    "entries": []})
                lrec["entries"].append({
                    "id": meta["concept_id"], "rank": meta.get("rank"),
                    "ref": meta.get("ref", "relsum"),
                    "relevance": meta.get("relevance"),
                    "png": str(rel / "entry.png"), "pdf": str(rel / "entry.pdf")})
            for inst in m["instances"].values():
                for fvrec in inst["fv"].values():
                    for srec in fvrec["samples"].values():
                        for lrec in srec["layers"].values():
                            lrec["entries"].sort(key=lambda e: (e["rank"] is None, e["rank"]))
    from experiments.xai_methods import METHODS, METHOD_CAPTIONS
    manifest = {"generated": _now(), "models": models,
                "xai_order": list(METHODS), "xai_captions": METHOD_CAPTIONS}
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def _model_prefix_reg(model) -> Tuple[int, int]:
    """``(n_prefix, n_reg)`` for a ViT backbone: ``n_prefix`` non-patch tokens
    (1 cls + registers); ``n_reg`` register tokens (0 for a standard ViT, 4 for
    DINOv3). Assumes exactly one cls token, which holds for M1–M4."""
    n_prefix = int(getattr(model.backbone, "num_prefix_tokens", 1))
    return n_prefix, max(n_prefix - 1, 0)


def generate_sample_extras(model, attribution, ds, samples, *, composite, normalize,
                           device: str, md_dir: Path, force: bool = False) -> None:
    """Model-level per-sample sub-figures shared across every instance of a model:
    the competing-XAI saliency row (``_sample_xai/<method>/<key>.png``: LRP,
    Chefer, rollout, occlusion) and the dual-site OOD token-norm map
    (``_normmaps/<key>.png``) plus the OOD patch-token count (``_ood.json``).

    Idempotent: skips a sample whose outputs already exist unless ``force``.
    Defensive — a failure here is logged but never aborts the core CRP build."""
    if not samples:
        return
    from experiments.xai_methods import METHODS
    n_prefix, n_reg = _model_prefix_reg(model)
    xai_root, norm_dir, ood_path = md_dir / "_sample_xai", md_dir / "_normmaps", md_dir / "_ood.json"
    ood: Dict[str, dict] = {}
    if ood_path.exists():
        try:
            ood = json.loads(ood_path.read_text())
        except json.JSONDecodeError:
            ood = {}
    for s in samples:
        key = s["key"]
        x = ds[s["ds_index"]][0]
        xai_done = all((xai_root / m / f"{key}.png").exists() for m in METHODS)
        ood_done = (norm_dir / f"{key}.png").exists() and key in ood
        if not force and xai_done and ood_done:
            continue
        try:
            if force or not xai_done:
                pred = save_sample_xai(model, attribution, x, composite=composite,
                                       normalize=normalize, device=device,
                                       out_root=xai_root, key=key)
                print(f"    xai[{key}] pred={pred}")
            if force or not ood_done:
                n_ood = save_sample_ood(model, x, normalize=normalize, device=device,
                                        out_path=norm_dir / f"{key}.png",
                                        n_prefix=n_prefix, n_reg=n_reg)
                ood[key] = {"ood_tokens": int(n_ood)}
                print(f"    ood[{key}] ood_tokens={n_ood}")
        except Exception as e:                       # never break the build over an extra
            print(f"    [warn] sample extras for {key!r} failed: {type(e).__name__}: {e}")
    if ood:
        ood_path.write_text(json.dumps(ood, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Core compute
# ─────────────────────────────────────────────────────────────────────────────

def run_spec(spec: dict, device: str) -> None:
    """Compute all entries for one job spec, then leave manifest rebuild to caller."""
    base, dataset, config = spec["base"], spec["dataset"], spec["config"]
    site, concept_kind = spec["site"], spec["concept"]
    blocks = list(spec["blocks"])
    n, detectors = int(spec["n"]), [int(d) for d in spec.get("detectors", [])]
    n_ref, mode, plot = int(spec["n_ref"]), spec["mode"], spec["plot"]
    crop, rank_mode = bool(spec.get("crop", False)), spec["rank"]
    classes = [int(c) for c in spec.get("classes", [])]
    n_rank = int(spec["n_rank"])
    fv_end = int(spec.get("fv_end", 0))
    fv_class = spec.get("fv_class", "original")
    if fv_class not in FV_CLASS_LABELS:
        raise typer.BadParameter(f"--fv-class must be one of {FV_CLASS_LABELS}, got {fv_class!r}")
    # Relevance-sign flavour: default drops negative relevance (index, rankings,
    # scores, displayed maps); include_negative keeps the fully signed quantities.
    include_negative = bool(spec.get("include_negative", False))

    # Model + dataset via the ModelDataset registry (experiments/model_datasets).
    # The pair's flat tag (mdset.tag == f"{base}_{dataset}") names the FV cache /
    # figure tree, unchanged from the old zoo tag.
    ckpt = spec.get("checkpoint")
    try:
        mdset = find(base, dataset, device=device, checkpoint=ckpt)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    model = mdset.model
    num_classes, head_name = model.num_classes, model.head_name
    label = f"{base} · {head_name} · {dataset}"

    num_heads = model.backbone.blocks[0].attn.num_heads
    transform, normalize = mdset.transform, mdset.normalize
    # ImageNet: the gallery indexes and serves the FULL 50k val split (FV builds
    # on local scratch, so the old NFS small-file wedge no longer forces a subset).
    ds = mdset.dataset
    concept = make_concept(concept_kind, num_heads)

    md = f"{base}_{dataset}"
    comp_cls = COMPOSITES[config]
    if site in PROBE_SITES:
        # Probe layers exist only after the composite canonizes the model — validate
        # the names against the canonized module set, not the bare model.
        with comp_cls().context(model.eval()) as canon:
            known = set(dict(canon.named_modules()))
        layers = resolve_layers(model, site, blocks, known=known)
    else:
        layers = resolve_layers(model, site, blocks)
    model_tag = md
    # All composites are grad×input (g-convention): the read-out is uniformly
    # heatmap = x.grad·x, per-layer relevance = g×activation.
    if fv_class != "original":
        raise typer.BadParameter(
            "the aligned FV path has no g-convention adapter yet")
    attribution = GradTimesInputAttribution(model)
    fv_cls = GradTimesInputFeatureVisualization
    layer_names = [ln for _, ln in layers]
    # The FV index feeds the AGGREGATE view (reference sample indices), fv_index ranking,
    # AND the single-image local view (each locally-relevant detector is shown with
    # its representatives). --only-samples skips only the aggregate render, not the
    # index — the local view needs representatives too.
    only_samples = bool(spec.get("only_samples", False))
    want_samples = bool(spec.get("samples", True))
    only_extras = bool(spec.get("only_extras", False))
    need_fv = ((not only_samples) or rank_mode == "fv_index" or want_samples) \
        and not spec.get("only_heat", False) and not only_extras
    fv = None
    if need_fv:
        # Build on scratch, mirror to the persistent root. Refill scratch from the
        # mirror first so an index built before a bounce is reused, not recomputed.
        # The aligned index lives in a sibling directory (suffix --aligned) so the
        # two flavours never mix on disk.
        fv_dirname = config if fv_class == "original" else f"{config}--aligned"
        if include_negative:
            fv_dirname += "--negincl"
        fv_dir = Path(CACHE_ROOT) / "fv" / model_tag / fv_dirname
        rel = fv_dir.relative_to(CACHE_ROOT)                 # fv/<model_tag>/<dirname>
        storage.sync(CACHE_MIRROR / rel, fv_dir)             # hydrate (no-op if scratch already has it)
        if fv_class == "aligned":
            if fv_aligned.targets_path(fv_dir).exists():
                pred_targets = fv_aligned.load_predicted_targets(fv_dir)
            else:
                print(f"[{md}/{config}] predicting conditioning targets over {len(ds)} samples…")
                pred_targets = fv_aligned.predict_targets(model, ds, normalize, device)
                fv_aligned.save_predicted_targets(
                    fv_dir, pred_targets,
                    provenance={"model_tag": model_tag, "config": config})
            fv_ds = fv_aligned.PredTargetsDataset(ds, pred_targets)
            fv = fv_aligned.AlignedFeatureVisualization(
                attribution, fv_ds, {ln: concept for ln in layer_names},
                preprocess_fn=normalize, path=str(fv_dir), device=device)
            # per-class broadcast triples the effective batch — keep it bounded
            fv_batch = 12
        else:
            fv = fv_cls(attribution, ds, {ln: concept for ln in layer_names},
                        preprocess_fn=normalize, path=str(fv_dir), device=device,
                        negative_clamp=not include_negative)
            fv_batch = 32
        fv_path = Path(fv.RelMax.PATH)
        have = fv_path.exists() and all(any(fv_path.glob(f"{ln}_data.npy")) for ln in layer_names)
        if fv_class == "aligned":
            stats_path = Path(fv.RelStats.PATH)
            have = have and (stats_path / "targets.npy").exists() \
                and all((stats_path / ln).is_dir() for ln in layer_names)
        if not have:
            end = fv_end if fv_end > 0 else len(ds)
            print(f"[{md}/{config}] building {fv_class} FV index over {end} samples "
                  f"for {len(layer_names)} layer(s)…")
            fv.run(comp_cls(), 0, end, batch_size=fv_batch)
            storage.sync(fv_dir, CACHE_MIRROR / rel)         # persist the fresh build
        # Sibling index ranked by token-MAX (stores {Rel|Act}Max_max_normed) —
        # the sum/max pair serves the four REF_MODES. Original fv_class only.
        fv_max = None
        if fv_class == "original":
            fv_max = fv_cls(attribution, ds, {ln: concept for ln in layer_names},
                            preprocess_fn=normalize, path=str(fv_dir),
                            max_target="max", device=device,
                            negative_clamp=not include_negative)
            max_path = Path(fv_max.RelMax.PATH)
            have_max = max_path.exists() and all(any(max_path.glob(f"{ln}_data.npy")) for ln in layer_names)
            if not have_max:
                end = fv_end if fv_end > 0 else len(ds)
                print(f"[{md}/{config}] building max-ranked FV index over {end} samples "
                      f"for {len(layer_names)} layer(s)…")
                fv_max.run(comp_cls(), 0, end, batch_size=fv_batch)
                storage.sync(fv_dir, CACHE_MIRROR / rel)

    # Correctly-classified sample for class-conditional ranking.
    target_classes = sorted(set(classes) & set(range(num_classes))) if classes else list(range(num_classes))
    sel = select_correct(model, ds, target_classes, n_rank, device, normalize=normalize) \
        if rank_mode == "class_conditional" and not only_extras else {}

    config_dir = FIG_DIR / md / config
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir.parent / "model.json").write_text(json.dumps(
        {"base": base, "head": head_name, "dataset": dataset, "label": label}, indent=2))
    (config_dir / "composite.json").write_text(json.dumps(composite_meta(config, comp_cls, site), indent=2))

    # Fixed comparison images (shown across every layer of this instance). Saved
    # once at md level; their raw thumbnail goes to the web. Empty ⇒ aggregate-only.
    samples = pick_samples(dataset, ds) if spec.get("samples", True) else []
    heat_key = concept_kind + ("--negincl" if include_negative else "")
    heat_dir = config_dir / "_sample_heat" / heat_key
    for s in samples:
        save_sample_image(ds, s, config_dir.parent / "_samples" / f"{s['key']}.png")
        save_sample_heat(attribution, ds[s["ds_index"]][0], s["target"],
                         composite=comp_cls(), normalize=normalize, device=device,
                         out_path=heat_dir / f"{s['key']}.png",
                         include_negative=include_negative)
    if samples:
        print(f"[{md}/{config}] {len(samples)} single-image sample(s): "
              f"{[s['key'] for s in samples]}")
    if spec.get("only_heat"):
        return   # sample relevance heatmaps only — skip FV + entry rendering

    # Model-level competing-XAI saliency row + dual-site OOD token-norm maps,
    # shared across every instance of this model (composite-independent). Written
    # once at md level; idempotent for normal jobs, force-refreshed by sample-xai.
    try:
        generate_sample_extras(model, attribution, ds, samples, composite=comp_cls(),
                               normalize=normalize, device=device, md_dir=config_dir.parent,
                               force=only_extras)
    except Exception as e:                       # extras must never block the CRP render
        print(f"[{md}/{config}] [warn] sample-extras step failed: {type(e).__name__}: {e}")
    if only_extras:
        return   # extras backfill only — skip FV + entry rendering

    for b, layer in layers:
        scores = rank_scores(rank_mode, attribution=attribution, ds=ds, sel=sel, layer=layer,
                             concept=concept, composite=comp_cls(), normalize=normalize,
                             device=device, fv=fv, include_negative=include_negative)
        order = list(np.argsort(scores)[::-1])               # descending
        rank_of = {int(cid): r for r, cid in enumerate(order)}
        ids = list(dict.fromkeys([int(c) for c in order[:n]] + detectors))
        print(f"[{md}/{config}] block {b} ({layer}): {len(ids)} detector(s) → {ids}")
        base_meta = {"layer": layer, "site": site, "block": b,
                     "concept_kind": concept_kind, "config": config, "fv_class": fv_class,
                     "include_negative": include_negative}
        # Entries of the two FV flavours live in disjoint subtrees so recomputes
        # of one never touch the other; the manifest keys them by meta fv_class.
        # The negincl sign flavour nests one level deeper inside that subtree.
        entries_root = config_dir if fv_class == "original" else config_dir / "fv_aligned"
        if include_negative:
            entries_root = entries_root / "negincl"
        # Aggregate view: top reference images across the dataset (needs FV).
        if not only_samples:
            for cid in ids:
                out_dir = entries_root / site / f"block{b}" / concept_kind / str(cid)
                render_entry(fv, attribution, ds, layer, cid, mode=mode, n_ref=n_ref,
                             composite=comp_cls(), concept=concept, normalize=normalize,
                             device=device, crop=crop, plot=plot, out_dir=out_dir,
                             fv_class=fv_class, include_negative=include_negative,
                             meta_extra={
                                 **base_meta, "sample": "aggregate", "sample_label": "Aggregate",
                                 "rank": rank_of.get(int(cid)), "relevance": float(scores[int(cid)]),
                             })
        # Local analysis per fixed input image: rank detectors on THAT image, then
        # show each with the query heatmap + its dataset representatives (needs FV).
        img_root = entries_root / site / f"block{b}" / concept_kind / "_img"
        for s in samples:
            shutil.rmtree(img_root / s["key"], ignore_errors=True)   # drop stale detectors
            x = ds[s["ds_index"]][0]
            det = local_relevances(attribution, x, s["target"], layer, concept=concept,
                                   composite=comp_cls(), normalize=normalize, device=device,
                                   include_negative=include_negative)
            l_ids = list(dict.fromkeys([int(c) for c in np.argsort(det)[::-1][:n]] + detectors))
            print(f"[{md}/{config}] block {b} · {s['key']}: local detectors → {l_ids}")
            ref_fvs = {"sum": fv, "max": fv_max}
            for r_local, cid in enumerate(l_ids):
                for ref_name, (ref_quantity, ref_reduction, _lbl) in REF_MODES.items():
                    ref_fv = ref_fvs[ref_reduction]
                    if ref_fv is None:
                        continue                     # max stores absent (aligned fv_class)
                    if fv_class == "aligned" and ref_quantity != "relevance":
                        continue                     # aligned index is relevance-only
                    render_local_entry(ref_fv, attribution, ds, x, s["target"], layer, cid,
                                       mode=ref_quantity,
                                       n_ref=n_ref, composite=comp_cls(), concept=concept,
                                       normalize=normalize, device=device, crop=crop, plot=plot,
                                       out_dir=img_root / s["key"] / ref_name / str(cid),
                                       fv_class=fv_class, include_negative=include_negative,
                                       meta_extra={
                                           **base_meta, "sample": s["key"], "ref": ref_name,
                                           "sample_label": s["label"], "rank": r_local})


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def compute(
    base: str = typer.Option("vit_small", "--base", help="model axis; (base, dataset) must be a registered model_datasets pair"),
    dataset: str = typer.Option(..., "--dataset", help=f"dataset key: {sorted(EVAL_DATASETS)}"),
    config: str = typer.Option("cp_lrp_baseline", "--config", help=f"composite name: {sorted(COMPOSITES)}"),
    site: str = typer.Option("proj_drop", "--site", help=f"probe site: {tuple(SITE_LAYERS)}"),
    blocks: List[int] = typer.Option(..., "--blocks", help="block indices (repeat the flag)"),
    concept: str = typer.Option("embed_dim", "--concept", help=f"concept kind: {CONCEPTS}"),
    n: int = typer.Option(5, "--n", help="auto top-n most-relevant detectors per layer"),
    detectors: List[int] = typer.Option([], "--detectors", help="extra explicit detector ids (additive)"),
    n_ref: int = typer.Option(12, "--n-ref", help="number of representative images per detector"),
    mode: str = typer.Option("relevance", "--mode", help="relevance | activation"),
    plot: str = typer.Option("heat_rf", "--plot", help="heat_rf (img+saliency+receptive-field crop) | heatmap (img+saliency) | opaque (masked crop)"),
    crop: bool = typer.Option(False, "--crop", help="clip each reference to its saliency map's high-relevance region (CRP receptive-field crop)"),
    samples: bool = typer.Option(True, "--samples/--no-samples", help="also render the fixed single-image comparison views (lizard, cheeseburger, …)"),
    only_samples: bool = typer.Option(False, "--only-samples", help="render ONLY the single-image views (skip FV index + aggregate; reuse existing aggregate entries)"),
    rank: str = typer.Option("class_conditional", "--rank", help="class_conditional | fv_index"),
    fv_class: str = typer.Option("original", "--fv-class", help=f"FV index flavour: {', '.join(FV_CLASS_LABELS.keys())}"),
    classes: List[int] = typer.Option([], "--classes", help="restrict ranking to these classes"),
    n_rank: int = typer.Option(8, "--n-rank", help="correct images per class for ranking"),
    fv_end: int = typer.Option(0, "--fv-end", help="cap FV-index samples (0 = full dataset)"),
    include_negative: bool = typer.Option(False, "--include-negative/--no-include-negative", help="keep negative relevance in the FV index, rankings, scores and displayed maps (default: dropped); renders a parallel '<config>/negincl' flavour"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="explicit best.pt path (finetuned-probe models only)"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device"),
):
    """Compute one (model, dataset, composite, site, blocks, concept) spec, record
    the job, render entries, and rebuild the manifest."""
    if dataset not in EVAL_DATASETS:
        raise typer.BadParameter(f"--dataset must be one of {sorted(EVAL_DATASETS)}")
    spec = {
        "base": base, "dataset": dataset, "config": config, "site": site,
        "blocks": list(blocks), "concept": concept, "n": n, "detectors": list(detectors),
        "n_ref": n_ref, "mode": mode, "plot": plot, "crop": crop, "samples": samples,
        "only_samples": only_samples, "rank": rank, "fv_class": fv_class,
        "classes": list(classes), "n_rank": n_rank, "fv_end": fv_end,
        "include_negative": include_negative,
        "checkpoint": checkpoint,
        "created": _now(),
    }
    run_spec(spec, device)
    record_job(spec)
    rebuild_manifest()
    print(f"done · manifest → {MANIFEST_PATH}")


@app.command()
def replay(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="filter jobs by dataset"),
    config: Optional[str] = typer.Option(None, "--config", help="filter jobs by config"),
    base: Optional[str] = typer.Option(None, "--base", help="filter jobs by base"),
    plot: Optional[str] = typer.Option(None, "--plot", help="override each job's plot mode (e.g. heat_rf) — re-renders existing entries in the new layout"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device"),
):
    """Re-run tracked jobs.jsonl (regenerate gallery after a restart/redeploy)."""
    if not JOBS_PATH.exists():
        print("no jobs.jsonl — nothing to replay")
        return
    jobs = [json.loads(l) for l in JOBS_PATH.read_text().splitlines() if l.strip()]
    sel = [j for j in jobs
           if (dataset is None or j["dataset"] == dataset)
           and (config is None or j["config"] == config)
           and (base is None or j["base"] == base)]
    print(f"replaying {len(sel)}/{len(jobs)} job(s)" + (f" · plot={plot}" if plot else ""))
    for j in sel:
        run_spec({**j, "plot": plot} if plot else j, device)
    rebuild_manifest()
    print(f"done · manifest → {MANIFEST_PATH}")


@app.command()
def samples(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="filter jobs by dataset"),
    config: Optional[str] = typer.Option(None, "--config", help="filter jobs by config"),
    base: Optional[str] = typer.Option(None, "--base", help="filter jobs by base"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device"),
):
    """Backfill the single-image local views onto EXISTING tracked jobs without
    re-rendering the aggregate entries (``--only-samples``). Still needs the FV
    index (hydrated from the persistent mirror, or built if absent) because each
    locally-relevant detector is shown with its dataset representatives."""
    if not JOBS_PATH.exists():
        print("no jobs.jsonl — nothing to do")
        return
    jobs = [json.loads(l) for l in JOBS_PATH.read_text().splitlines() if l.strip()]
    sel = [j for j in jobs
           if (dataset is None or j["dataset"] == dataset)
           and (config is None or j["config"] == config)
           and (base is None or j["base"] == base)]
    print(f"backfilling samples for {len(sel)}/{len(jobs)} job(s)")
    for j in sel:
        run_spec({**j, "only_samples": True, "samples": True}, device)
    rebuild_manifest()
    print(f"done · manifest → {MANIFEST_PATH}")


@app.command("sample-heat")
def sample_heat(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="filter jobs by dataset"),
    config: Optional[str] = typer.Option(None, "--config", help="filter jobs by config"),
    base: Optional[str] = typer.Option(None, "--base", help="filter jobs by base"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device"),
):
    """Backfill each fixed sample input's OWN overall relevance heatmap onto
    EXISTING tracked jobs (one full-model LRP backward per sample/instance). No FV,
    no entry re-render — just the per-sample saliency shown next to the thumbnail."""
    if not JOBS_PATH.exists():
        print("no jobs.jsonl — nothing to do")
        return
    jobs = [json.loads(l) for l in JOBS_PATH.read_text().splitlines() if l.strip()]
    sel = [j for j in jobs
           if (dataset is None or j["dataset"] == dataset)
           and (config is None or j["config"] == config)
           and (base is None or j["base"] == base)]
    print(f"sample-heat backfill for {len(sel)}/{len(jobs)} job(s)")
    for j in sel:
        run_spec({**j, "only_heat": True, "samples": True}, device)
    rebuild_manifest()
    print(f"done · manifest → {MANIFEST_PATH}")


@app.command("sample-xai")
def sample_xai(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="filter jobs by dataset"),
    config: Optional[str] = typer.Option(None, "--config", help="filter jobs by config"),
    base: Optional[str] = typer.Option(None, "--base", help="filter jobs by base"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device"),
):
    """(Re)generate the model-level per-sample extras — competing-XAI saliency row
    (LRP · Chefer · rollout · occlusion) and dual-site OOD token-norm maps + counts
    — for EXISTING tracked jobs. Deduped per model (base,dataset): these figures are
    composite-independent, so they are computed once per model, not per instance."""
    if not JOBS_PATH.exists():
        print("no jobs.jsonl — nothing to do")
        return
    jobs = [json.loads(l) for l in JOBS_PATH.read_text().splitlines() if l.strip()]
    sel = [j for j in jobs
           if (dataset is None or j["dataset"] == dataset)
           and (config is None or j["config"] == config)
           and (base is None or j["base"] == base)]
    seen = set()
    uniq = []
    for j in sel:                                    # one job per model (extras are md-level)
        key = (j["base"], j["dataset"], j.get("checkpoint"))
        if key not in seen:
            seen.add(key)
            uniq.append(j)
    print(f"sample-xai backfill for {len(uniq)} model(s) (from {len(sel)}/{len(jobs)} job(s))")
    for j in uniq:                                   # embed_dim/proj_drop: extras only, no FV
        run_spec({**j, "concept": "embed_dim", "site": "proj_drop",
                  "only_extras": True, "samples": True}, device)
    rebuild_manifest()
    print(f"done · manifest → {MANIFEST_PATH}")


@app.command()
def manifest():
    """Rebuild manifest.json by scanning the figures tree (no compute)."""
    m = rebuild_manifest()
    print(f"manifest → {MANIFEST_PATH} ({len(m['models'])} model(s))")


if __name__ == "__main__":
    matplotlib.use("Agg")  # headless render; importers (notebooks) keep their backend
    app()
