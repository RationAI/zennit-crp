"""CRP gallery — compute CRP saliency maps + representative images for a static web.

This is the single, idiomatic generator behind the static gallery at
``webapp/crp_gallery/`` (served via ``webshare``). Given a model (backbone+head,
the systematic ``experiments.models`` way), an LRP/CRP recipe from
:mod:`lrp_configs`, a probe site and a set of blocks, it produces — per concept
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
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

from torchvision.transforms.functional import gaussian_blur

import lrp_configs
from experiments import storage
from crp.attribution import CondAttribution
from crp.concepts import HeadConcept, EmbeddingDimConcept
from crp.helper import load_maximization
from crp.image import get_crop_range, imgify, plot_grid, vis_img_heatmap, vis_opaque_img
from crp.visualization import FeatureVisualization
from experiments.models import build_probe, BASES, HEADS
from experiments.model_io import (
    DATASETS, SITES, load_probe, select_correct, site_layer_names, backbone_transforms,
)

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

CONCEPTS = ("head", "embed_dim", "sae")

app = typer.Typer(add_completion=False, help=__doc__)


# ─────────────────────────────────────────────────────────────────────────────
# Model + data (reuse experiments.models boilerplate; un-normalized + normalize)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(base: str, dataset: str, *, model_source: str, checkpoint: Optional[str],
               head: str, num_classes: Optional[int], head_kwargs: dict, device: str,
               ) -> Tuple[torch.nn.Module, int, str, str]:
    """Build the probe the systematic way (reuses ``experiments.model_io``).
    Returns ``(model, num_classes, head_name, label)``.

    * ``checkpoint`` source — load a finetuned probe via
      :func:`model_io.load_probe` (``finetune_<base>_<tag>/<ts>/best.pt`` or an
      explicit ``--checkpoint`` path); reconstruction uses the saved spec.
    * ``fresh`` source — ImageNet-pretrained backbone + (untrained) head from the
      registries, for inspecting raw pretrained features.
    """
    if model_source == "checkpoint":
        tag = DATASETS[dataset][2]
        model, ck, _ = load_probe(tag, device, base=base,
                                  path=Path(checkpoint) if checkpoint else None)
        label = f"{ck['base']} · {ck['head']} · {dataset}"
        return model, int(ck["num_classes"]), ck["head"], label
    if model_source == "fresh":
        if num_classes is None:
            raise typer.BadParameter("--num-classes is required for --model-source fresh")
        model = build_probe(base=base, head=head, num_classes=num_classes,
                            head_kwargs=head_kwargs).eval().to(device)
        model.requires_grad_(False)
        return model, int(num_classes), head, f"{base} · {head} · {dataset} (fresh)"
    raise typer.BadParameter(f"--model-source must be checkpoint|fresh, got {model_source!r}")


def load_eval_dataset(dataset: str, transform, extra_kwargs: Optional[dict] = None):
    """Un-normalized eval dataset for the given key (reuses the dataset registry).
    ``extra_kwargs`` is merged into the loader kwargs — used to restrict ImageNet
    (50k val) to the gallery's ranking classes so the FV index stays small (the
    full-dataset index for a vit_base ``m=1536`` SAE is huge and stalls on NFS)."""
    ds_name, ds_kw, _ = DATASETS[dataset]
    from experiments.datasets import load as load_dataset
    return load_dataset(ds_name, root=REPO_ROOT / "data", transform=transform,
                        **{**ds_kw, **(extra_kwargs or {})})


# ─────────────────────────────────────────────────────────────────────────────
# Concepts / layers / ranking
# ─────────────────────────────────────────────────────────────────────────────

def make_concept(kind: str, num_heads: int):
    if kind == "head":
        return HeadConcept(num_heads=num_heads)
    if kind in ("embed_dim", "sae"):
        # 'sae' is per-latent over the spliced SAE feature space, same machinery
        # as embed_dim but applied to the SAESplice '.features' sublayer.
        return EmbeddingDimConcept(num_heads=num_heads)
    raise typer.BadParameter(f"--concept must be one of {CONCEPTS}, got {kind!r}")


def splice_sae(model, site: str, dataset: str, blocks: List[int], m: int,
               device: str) -> Dict[int, str]:
    """Splice the trained SAE (dictionary size ``m``) in as a reconstruction
    pass-through at ``site`` for each requested block, exactly like
    ``concept_flipping.setup_sites``. Returns ``{block: features_layer_name}`` —
    the recordable ``.features`` sublayer whose output is the (B, N, m) SAE codes
    that CRP decomposes the logit relevance onto. Mutates ``model`` in place."""
    from experiments.sae import load_sae, sae_path
    bl = model.backbone.blocks
    names: Dict[int, str] = {}
    for b in blocks:
        if not sae_path(site, dataset, b, m=m).is_file():
            raise typer.BadParameter(
                f"no SAE checkpoint {sae_path(site, dataset, b, m=m)} — train it first (experiments.sae)")
        sp = load_sae(site, dataset, b, device, m=m)   # reuse: ckpt-load + SAESplice build
        if site == "proj_drop":
            bl[b].attn.proj_drop = sp
            names[b] = f"backbone.blocks.{b}.attn.proj_drop.features"
        elif site == "residual":
            sp.inner = bl[b]                            # rewire for the residual site
            bl[b] = sp
            names[b] = f"backbone.blocks.{b}.features"
        else:
            raise typer.BadParameter(f"--site must be one of {SITES} for --concept sae, got {site!r}")
    return names


def resolve_layers(model, site: str, blocks: List[int]) -> List[Tuple[int, str]]:
    """``(block, layer_name)`` per requested block at the probe site (canonical
    site → layer mapping from :func:`model_io.site_layer_names`)."""
    if site not in SITES:
        raise typer.BadParameter(f"--site must be one of {SITES}, got {site!r}")
    names = site_layer_names(model, site)
    out = []
    for b in blocks:
        if not 0 <= b < len(names):
            raise typer.BadParameter(f"block {b} out of range 0..{len(names) - 1}")
        out.append((b, names[b]))
    return out


def rank_scores(rank_mode: str, *, attribution, ds, sel, layer, concept, composite,
                normalize, device, fv, batch_size: int = 32) -> np.ndarray:
    """Per-detector relevance score vector for one layer (higher = more relevant).

    * ``class_conditional`` (default) — mean over a sample of correctly-classified
      images of ``concept.attribute(R[layer])`` with relevance initialised at the
      true target logit (``{"y":[c]}``). Idiom from ``head_relevance_by_class``.
    * ``fv_index`` — mean over the FV RelMax index (whole-dataset, target-agnostic).
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
        det = concept.attribute(res.relevances[layer], abs_norm=False)  # (B, n_det)
        s = det.sum(0).detach().cpu().numpy()
        total = s if total is None else total + s
        n_imgs += det.shape[0]
    if total is None:
        raise RuntimeError("no correctly-classified images to rank from")
    return total / max(n_imgs, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Output tree: composite meta, per-entry figures, jobs, manifest
# ─────────────────────────────────────────────────────────────────────────────

def composite_meta(cfg) -> dict:
    """Human-readable summary + hyperparameters of a config, pulled straight from
    the source (no duplication): registry fields + the build() source text."""
    try:
        build_source = inspect.getsource(cfg.build).strip()
    except (OSError, TypeError):
        build_source = ""
    comp_class = type(cfg.composite()).__name__
    return {
        "name": cfg.name,
        "class": comp_class,
        "description": cfg.description,
        "isolates": cfg.isolates,
        "site": cfg.site,
        "build_source": build_source,
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def concept_kind_desc(concept_kind: str, site: str) -> str:
    """One-line description of the concept *basis* for the web (the composite
    panel is config-level and shared, so the SAE vs axis-aligned distinction must
    live per layer)."""
    if concept_kind.startswith("sae"):
        m = concept_kind.split("_m")[-1] if "_m" in concept_kind else "?"
        return (f"SAE-basis CRP — the logit relevance is decomposed onto a trained "
                f"sparse-autoencoder dictionary ({m} latents) spliced in as a reconstruction "
                f"pass-through at the {site} site. Concepts = SAE latents (learned, ~monosemantic), "
                f"not raw embedding axes. Same recipe/composite as the axis-aligned case — only the "
                f"concept basis differs.")
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

# ImageNet-1k class idx → display name. Six visually distinct classes.
IMAGENET_SAMPLES: List[Tuple[int, str]] = [
    (39, "lizard"), (933, "cheeseburger"), (1, "goldfish"),
    (817, "sports_car"), (985, "daisy"), (207, "golden_retriever"),
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
        for cls, name in IMAGENET_SAMPLES:
            idx = next((i for i, l in enumerate(labels) if l == cls), None)
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
                     normalize, device: str) -> np.ndarray:
    """Per-detector relevance of ONE input image at ``layer`` (local analysis):
    initialise relevance at the image's true class and read it on each concept.
    Returns a ``(n_det,)`` vector — argsort gives the detectors most relevant to
    *this* image."""
    xin = normalize(x[None].to(device)).requires_grad_(True)
    res = attribution(xin, [{"y": [int(target)]}], composite, record_layer=[layer],
                      mask_map=concept.mask)
    return concept.attribute(res.relevances[layer], abs_norm=False)[0].detach().cpu().numpy()


def render_local_entry(fv, attribution, ds, x, target: int, layer: str, cid: int, *,
                       mode: str, n_ref: int, composite, concept, normalize, device: str,
                       crop: bool, plot: str, out_dir: Path, meta_extra: dict) -> float:
    """Local analysis of one detector for one input image: the leftmost column is
    the query image + its *conditional* CRP heatmap; the remaining columns are the
    detector's dataset **representatives** so the reader can tell what the locally-
    relevant concept actually is. Both are class-conditional (see
    :func:`class_conditional_references`). png+pdf + meta.json.
    Returns the query image's relevance on the concept."""
    ref_s, ref_h = class_conditional_references(
        attribution, fv, ds, layer, cid, n_ref=n_ref, mode=mode, composite=composite,
        concept=concept, normalize=normalize, device=device)
    xin = normalize(x[None].to(device)).requires_grad_(True)
    res = attribution(xin, [{layer: [int(cid)], "y": [int(target)]}], composite,
                      record_layer=[layer], mask_map=concept.mask)
    local_h = res.heatmap.detach().cpu()                     # (1, H, W)
    rel = float(concept.attribute(res.relevances[layer], abs_norm=False)[0, int(cid)])
    # Column 0 = query image + local heatmap; columns 1.. = global representatives.
    imgs = torch.cat([x[None].detach().cpu(), ref_s.detach().cpu()], dim=0)
    heats = torch.cat([local_h, ref_h.detach().cpu()], dim=0)
    rows, nsub, row_lbl = build_rows(imgs, heats, plot=plot, crop=crop)
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
                     out_path: Path) -> None:
    """Save the sample input's OWN overall relevance heatmap — the full-model LRP
    attribution to its true class (all concepts, input space), the standard CRP
    saliency for that image. Instance-specific (the composite/model differ per
    basis), so stored per concept_kind. Always (re)written."""
    xin = normalize(x[None].to(device)).requires_grad_(True)
    res = attribution(xin, [{"y": [int(target)]}], composite)   # no layer cond → total heatmap
    heat = res.heatmap.detach().cpu()[0]                         # (H, W)
    # ViT input relevance is sparse — a few extreme pixels wash out a plain
    # symmetric norm. Clip to a high percentile of |R| so the structure is visible.
    vmax = float(np.quantile(heat.abs().numpy(), 0.995))
    if vmax <= 0:
        vmax = float(heat.abs().max()) or 1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgify(heat, cmap="bwr", vmin=-vmax, vmax=vmax, symmetric=False).save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Instance = (config, concept basis). SAE vs axis-aligned is an *instance*, not a
# layer — so the layer dropdown lists each block exactly once per instance.
# ─────────────────────────────────────────────────────────────────────────────

def instance_key(config: str, concept_kind: str) -> str:
    return f"{config}::{concept_kind}"


def instance_label(config: str, concept_kind: str) -> str:
    if concept_kind.startswith("sae_m"):
        return f"{config} · SAE (m={concept_kind.split('_m')[-1]})"
    if concept_kind == "embed_dim":
        return f"{config} · axis-aligned"
    if concept_kind == "head":
        return f"{config} · heads"
    return f"{config} · {concept_kind}"


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
                                 mode: str, composite, concept, normalize, device: str):
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
    ``n_ref`` references are done in one batched pass."""
    path = fv.RelMax.PATH if mode == "relevance" else fv.ActMax.PATH
    d_sorted, _, _ = load_maximization(path, layer)
    idxs = [int(i) for i in np.asarray(d_sorted)[:n_ref, int(cid)]]
    xs, ys = [], []
    for di in idxs:
        x, y = ds[di]
        xs.append(x)
        ys.append(int(y))
    batch = torch.stack(xs)
    xin = normalize(batch.to(device)).requires_grad_(True)
    conds = [{layer: [int(cid)], "y": [y]} for y in ys]
    res = attribution(xin, conds, composite, mask_map=concept.mask)
    return batch.detach().cpu(), res.heatmap.detach().cpu()


def rf_crop_row(samples, heatmaps, *, vis_th: float = 0.2, crop_th: float = 0.1,
                kernel_size: int = 19, alpha: float = 0.3):
    """Receptive-field crop row, sign-safe.

    Same recipe as :func:`crp.image.vis_opaque_img` with ``rf=True`` — blur the
    conditional heatmap, keep the box where it exceeds ``crop_th``, fade pixels
    below ``vis_th`` — with ONE change: normalise by ``max(|R|)`` instead of
    ``crp.helper.max_norm``'s ``R / R.max()``.

    ``max_norm`` is unsafe for signed bases. A ViT embedding dimension can produce
    a conditional heatmap that is negative almost everywhere (its input-space
    attribution is net inhibitory); the blurred map's max is then itself negative,
    so dividing by it FLIPS every pixel positive. The mask becomes all-True and the
    crop box the full frame — the panel then reads as "this concept covers the whole
    image" when the truth is "there is no positive evidence here". Normalising by
    the absolute max keeps the sign, so such a panel correctly fades out instead.

    For positive-dominant maps (the normal case) ``max(|R|) == R.max()``, so this is
    identical to the published behaviour."""
    out = []
    for i in range(len(samples)):
        img, heat = samples[i], heatmaps[i]
        blurred = gaussian_blur(heat.unsqueeze(0), kernel_size=kernel_size)[0]
        fh = blurred / (blurred.abs().max() + 1e-10)      # sign-safe normalisation
        vis_mask = fh > vis_th
        r1, r2, c1, c2 = get_crop_range(fh, crop_th)
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


def build_rows(samples, heatmaps, *, plot: str, crop: bool):
    """Sub-rows of one detector's figure. Returns ``(rows, nsub, row_labels)``.

    ``heat_rf`` (default) shows the three views together: the reference image, its
    conditional relevance heatmap, and the **receptive-field crop** — the image
    clipped to the heatmap's high-relevance box (``crp.image.get_crop_range``) with
    low-relevance pixels faded (``vis_opaque_img``). The crop answers "which part of
    the image does this concept latch onto?" without having to read heatmap colours.
    The RF row is always cropped (that is what the row *is*); ``crop`` still governs
    whether the image/heatmap rows are clipped too."""
    if plot == "opaque":
        return vis_opaque_img(samples, heatmaps, rf=crop), 1, ("concept",)
    imgs, heats = vis_img_heatmap(samples, heatmaps, rf=crop)
    if plot != "heat_rf":
        return (imgs, heats), 2, ("image", "relevance")
    rf = rf_crop_row(samples, heatmaps)
    return (imgs, heats, rf), 3, ("image", "relevance", "RF crop")


def render_entry(fv, attribution, ds, layer: str, cid: int, *, mode: str, n_ref: int,
                 composite, concept, normalize, device: str, crop: bool, plot: str,
                 out_dir: Path, meta_extra: dict) -> None:
    """Render + write one detector's figure (png+pdf) and meta.json (merge-not-wipe).

    Retrieve the reference images + their **class-conditional** CRP heatmaps (see
    :func:`class_conditional_references`), then present them. ``crop=True`` clips
    each reference to the high-relevance region of its saliency map (the standard
    CRP "receptive field" crop — ``crp.image.get_crop_range`` on the heatmap, via
    the ``rf`` flag of the vis functions; NOT the conv-neuron ``mask_rf`` path).
    """
    samples, heatmaps = class_conditional_references(
        attribution, fv, ds, layer, cid, n_ref=n_ref, mode=mode, composite=composite,
        concept=concept, normalize=normalize, device=device)
    rows, nsub, row_lbl = build_rows(samples, heatmaps, plot=plot, crop=crop)
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
    """Append/merge a job line in jobs.jsonl (dedup by base,dataset,config,site,concept)."""
    key = (spec["base"], spec["dataset"], spec["config"], spec["site"],
           spec["concept"], spec.get("sae_m", 0))
    jobs = []
    if JOBS_PATH.exists():
        for line in JOBS_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            jk = (j["base"], j["dataset"], j["config"], j["site"],
                  j["concept"], j.get("sae_m", 0))
            if jk != key:
                jobs.append(j)
    jobs.append(spec)
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOBS_PATH.write_text("\n".join(json.dumps(j) for j in jobs) + "\n")


def rebuild_manifest() -> dict:
    """Scan figures/ for entry meta.json + composite.json → manifest.json. The
    tree is the single source of truth, so the web lists exactly what exists.

    Schema: ``models → instances → samples → layers → entries``. An *instance* is
    (config, concept basis) — so SAE and axis-aligned are separate instances and
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
            # Per-sample per-block token-norm maps with flagged register outliers
            # (md-level, composite-independent; produced by
            # experiments/scripts/registers_position_freq.py).
            norm_dir = config_dir.parent / "_normmaps"
            sample_norms = ({p.stem: str(p.relative_to(GALLERY_DIR)) for p in norm_dir.glob("*.png")}
                            if norm_dir.exists() else {})
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
                rel = meta_path.parent.relative_to(GALLERY_DIR)
                inst = m["instances"].setdefault(instance_key(config, ck), {
                    "config": config, "basis": ck, "label": instance_label(config, ck),
                    "composite": composite,
                    "concept_desc": concept_kind_desc(ck, meta["site"]),
                    "samples": {}})
                srec = inst["samples"].setdefault(sample, {
                    "label": meta.get("sample_label") or ("Aggregate" if sample == "aggregate" else sample),
                    "image": sample_imgs.get(sample),
                    "heat": sample_heats.get(ck, {}).get(sample),
                    "normmap": sample_norms.get(sample), "layers": {}})
                lrec = srec["layers"].setdefault(layer, {
                    "site": meta["site"], "block": meta["block"], "concept_kind": ck,
                    "entries": []})
                lrec["entries"].append({
                    "id": meta["concept_id"], "rank": meta.get("rank"),
                    "relevance": meta.get("relevance"),
                    "png": str(rel / "entry.png"), "pdf": str(rel / "entry.pdf")})
            for inst in m["instances"].values():
                for srec in inst["samples"].values():
                    for lrec in srec["layers"].values():
                        lrec["entries"].sort(key=lambda e: (e["rank"] is None, e["rank"]))
    manifest = {"generated": _now(), "models": models}
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


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

    model, num_classes, head_name, label = load_model(
        base, dataset, model_source=spec["model_source"], checkpoint=spec.get("checkpoint"),
        head=spec.get("head", "linear"), num_classes=spec.get("num_classes"),
        head_kwargs=spec.get("head_kwargs", {}), device=device)

    num_heads = model.backbone.blocks[0].attn.num_heads
    transform, normalize = backbone_transforms(model.backbone)
    # ImageNet (50k val): load a moderate, class-diverse subset (n per class over
    # ALL classes). Full-50k indexing of a vit_base m=1536 SAE wedges NFS; too few
    # images (single-class) starves get_max_reference (latents lack n_ref samples).
    # Ranking still uses the spec's --classes; references come from the whole subset.
    ds_extra = {"n_per_class": 10} if dataset == "imagenet" else None
    ds = load_eval_dataset(dataset, transform, ds_extra)
    concept = make_concept(concept_kind, num_heads)
    cfg = lrp_configs.get(config)

    md = f"{base}_{dataset}"
    # 'sae': splice the trained dictionary at each block and record its .features
    # sublayer; label/cache are dict-size-specific so the SAE case coexists with
    # the axis-aligned (embed_dim) case under the same model+config.
    if concept_kind == "sae":
        sae_m = int(spec.get("sae_m") or 0)
        if sae_m <= 0:
            raise typer.BadParameter("--sae-m (dictionary size) is required for --concept sae")
        if site not in SITES:
            raise typer.BadParameter(f"--site must be one of {SITES}, got {site!r}")
        feat_names = splice_sae(model, site, dataset, blocks, sae_m, device)
        layers = [(b, feat_names[b]) for b in blocks]
        concept_kind = f"sae_m{sae_m}"          # display + output-dir + manifest label
        model_tag = f"{md}_sae_m{sae_m}_{site}"  # SAE-specific FV index cache
    else:
        layers = resolve_layers(model, site, blocks)
        model_tag = md
    attribution = CondAttribution(model)
    layer_names = [ln for _, ln in layers]
    # The FV index feeds the AGGREGATE view (reference sample indices), fv_index ranking,
    # AND the single-image local view (each locally-relevant detector is shown with
    # its representatives). --only-samples skips only the aggregate render, not the
    # index — the local view needs representatives too.
    only_samples = bool(spec.get("only_samples", False))
    want_samples = bool(spec.get("samples", True))
    need_fv = ((not only_samples) or rank_mode == "fv_index" or want_samples) \
        and not spec.get("only_heat", False)
    fv = None
    if need_fv:
        # Build on scratch, mirror to the persistent root. Refill scratch from the
        # mirror first so an index built before a bounce is reused, not recomputed.
        fv_dir = Path(cfg.fv_path(CACHE_ROOT, model_tag))
        rel = fv_dir.relative_to(CACHE_ROOT)                 # fv/<model_tag>/<config>
        storage.sync(CACHE_MIRROR / rel, fv_dir)             # hydrate (no-op if scratch already has it)
        fv = FeatureVisualization(attribution, ds, {ln: concept for ln in layer_names},
                                  preprocess_fn=normalize, path=str(fv_dir), device=device)
        fv_path = Path(fv.RelMax.PATH)
        have = fv_path.exists() and all(any(fv_path.glob(f"{ln}_data.npy")) for ln in layer_names)
        if not have:
            end = fv_end if fv_end > 0 else len(ds)
            print(f"[{md}/{config}] building FV index over {end} samples for {len(layer_names)} layer(s)…")
            fv.run(cfg.composite(), 0, end, batch_size=32)
            storage.sync(fv_dir, CACHE_MIRROR / rel)         # persist the fresh build

    # Correctly-classified sample for class-conditional ranking.
    target_classes = sorted(set(classes) & set(range(num_classes))) if classes else list(range(num_classes))
    sel = select_correct(model, ds, target_classes, n_rank, device, normalize=normalize) \
        if rank_mode == "class_conditional" else {}

    config_dir = FIG_DIR / md / config
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir.parent / "model.json").write_text(json.dumps(
        {"base": base, "head": head_name, "dataset": dataset, "label": label}, indent=2))
    (config_dir / "composite.json").write_text(json.dumps(composite_meta(cfg), indent=2))

    # Fixed comparison images (shown across every layer of this instance). Saved
    # once at md level; their raw thumbnail goes to the web. Empty ⇒ aggregate-only.
    samples = pick_samples(dataset, ds) if spec.get("samples", True) else []
    heat_dir = config_dir / "_sample_heat" / concept_kind
    for s in samples:
        save_sample_image(ds, s, config_dir.parent / "_samples" / f"{s['key']}.png")
        save_sample_heat(attribution, ds[s["ds_index"]][0], s["target"],
                         composite=cfg.composite(), normalize=normalize, device=device,
                         out_path=heat_dir / f"{s['key']}.png")
    if samples:
        print(f"[{md}/{config}] {len(samples)} single-image sample(s): "
              f"{[s['key'] for s in samples]}")
    if spec.get("only_heat"):
        return   # sample relevance heatmaps only — skip FV + entry rendering

    for b, layer in layers:
        scores = rank_scores(rank_mode, attribution=attribution, ds=ds, sel=sel, layer=layer,
                             concept=concept, composite=cfg.composite(), normalize=normalize,
                             device=device, fv=fv)
        order = list(np.argsort(scores)[::-1])               # descending
        rank_of = {int(cid): r for r, cid in enumerate(order)}
        ids = list(dict.fromkeys([int(c) for c in order[:n]] + detectors))
        print(f"[{md}/{config}] block {b} ({layer}): {len(ids)} detector(s) → {ids}")
        base_meta = {"layer": layer, "site": site, "block": b,
                     "concept_kind": concept_kind, "config": config}
        # Aggregate view: top reference images across the dataset (needs FV).
        if not only_samples:
            for cid in ids:
                out_dir = config_dir / site / f"block{b}" / concept_kind / str(cid)
                render_entry(fv, attribution, ds, layer, cid, mode=mode, n_ref=n_ref,
                             composite=cfg.composite(), concept=concept, normalize=normalize,
                             device=device, crop=crop, plot=plot, out_dir=out_dir, meta_extra={
                                 **base_meta, "sample": "aggregate", "sample_label": "Aggregate",
                                 "rank": rank_of.get(int(cid)), "relevance": float(scores[int(cid)]),
                             })
        # Local analysis per fixed input image: rank detectors on THAT image, then
        # show each with the query heatmap + its dataset representatives (needs FV).
        img_root = config_dir / site / f"block{b}" / concept_kind / "_img"
        for s in samples:
            shutil.rmtree(img_root / s["key"], ignore_errors=True)   # drop stale detectors
            x = ds[s["ds_index"]][0]
            det = local_relevances(attribution, x, s["target"], layer, concept=concept,
                                   composite=cfg.composite(), normalize=normalize, device=device)
            l_ids = list(dict.fromkeys([int(c) for c in np.argsort(det)[::-1][:n]] + detectors))
            print(f"[{md}/{config}] block {b} · {s['key']}: local detectors → {l_ids}")
            for r_local, cid in enumerate(l_ids):
                render_local_entry(fv, attribution, ds, x, s["target"], layer, cid, mode=mode,
                                   n_ref=n_ref, composite=cfg.composite(), concept=concept,
                                   normalize=normalize, device=device, crop=crop, plot=plot,
                                   out_dir=img_root / s["key"] / str(cid), meta_extra={
                                       **base_meta, "sample": s["key"],
                                       "sample_label": s["label"], "rank": r_local})


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def compute(
    base: str = typer.Option("vit_small", "--base", help=f"backbone: {sorted(BASES)}"),
    dataset: str = typer.Option(..., "--dataset", help=f"dataset key: {sorted(DATASETS)}"),
    config: str = typer.Option("cp_lrp_baseline", "--config", help="lrp_configs recipe name"),
    site: str = typer.Option("proj_drop", "--site", help=f"probe site: {SITES}"),
    blocks: List[int] = typer.Option(..., "--blocks", help="block indices (repeat the flag)"),
    concept: str = typer.Option("embed_dim", "--concept", help=f"concept kind: {CONCEPTS}"),
    sae_m: int = typer.Option(0, "--sae-m", help="(concept=sae) SAE dictionary size to splice"),
    n: int = typer.Option(5, "--n", help="auto top-n most-relevant detectors per layer"),
    detectors: List[int] = typer.Option([], "--detectors", help="extra explicit detector ids (additive)"),
    n_ref: int = typer.Option(6, "--n-ref", help="number of representative images per detector"),
    mode: str = typer.Option("relevance", "--mode", help="relevance | activation"),
    plot: str = typer.Option("heat_rf", "--plot", help="heat_rf (img+saliency+receptive-field crop) | heatmap (img+saliency) | opaque (masked crop)"),
    crop: bool = typer.Option(False, "--crop", help="clip each reference to its saliency map's high-relevance region (CRP receptive-field crop)"),
    samples: bool = typer.Option(True, "--samples/--no-samples", help="also render the fixed single-image comparison views (lizard, cheeseburger, …)"),
    only_samples: bool = typer.Option(False, "--only-samples", help="render ONLY the single-image views (skip FV index + aggregate; reuse existing aggregate entries)"),
    rank: str = typer.Option("class_conditional", "--rank", help="class_conditional | fv_index"),
    classes: List[int] = typer.Option([], "--classes", help="restrict ranking to these classes"),
    n_rank: int = typer.Option(8, "--n-rank", help="correct images per class for ranking"),
    fv_end: int = typer.Option(0, "--fv-end", help="cap FV-index samples (0 = full dataset)"),
    # fresh-model only:
    model_source: str = typer.Option("checkpoint", "--model-source", help="checkpoint | fresh"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="explicit best.pt path"),
    head: str = typer.Option("linear", "--head", help=f"(fresh) head: {sorted(HEADS)}"),
    num_classes: Optional[int] = typer.Option(None, "--num-classes", help="(fresh) classes"),
    head_kwargs_json: str = typer.Option("{}", "--head-kwargs", help="(fresh) JSON head kwargs"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device"),
):
    """Compute one (model, dataset, composite, site, blocks, concept) spec, record
    the job, render entries, and rebuild the manifest."""
    if dataset not in DATASETS:
        raise typer.BadParameter(f"--dataset must be one of {sorted(DATASETS)}")
    spec = {
        "base": base, "dataset": dataset, "config": config, "site": site,
        "blocks": list(blocks), "concept": concept, "sae_m": sae_m, "n": n, "detectors": list(detectors),
        "n_ref": n_ref, "mode": mode, "plot": plot, "crop": crop, "samples": samples,
        "only_samples": only_samples, "rank": rank,
        "classes": list(classes), "n_rank": n_rank, "fv_end": fv_end,
        "model_source": model_source, "checkpoint": checkpoint, "head": head,
        "num_classes": num_classes, "head_kwargs": json.loads(head_kwargs_json),
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


@app.command()
def manifest():
    """Rebuild manifest.json by scanning the figures tree (no compute)."""
    m = rebuild_manifest()
    print(f"manifest → {MANIFEST_PATH} ({len(m['models'])} model(s))")


if __name__ == "__main__":
    app()
