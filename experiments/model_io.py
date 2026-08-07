"""Shared model + data loading for the explainability experiments.

Single home for the bits that used to be re-derived in every experiment script
(``concept_flipping`` re-exports these for back-compat):

* :data:`DATASETS` — the eval-dataset registry (key → loader, kwargs, probe tag).
* :func:`load_probe` — load a finetuned probe for **any** ``base`` (not just
  ``vit_small``), rebuilding it via the ``experiments.models`` registry.
* :func:`select_correct` — sample correctly-classified images per class, with an
  optional ``normalize`` so it works whether the dataset is pre-normalized or
  the un-normalized [0,1] convention (:func:`models.backbone_transforms`).
* :data:`SITES` / :func:`site_layer_names` — canonical concept probe sites and
  their per-block layer-name strings.

Transforms come from :func:`models.backbone_transforms` (re-exported here).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments.models import build_probe, backbone_transforms

REPO_ROOT = Path(__file__).resolve().parents[1]

# dataset key → (loader name, loader kwargs, finetune-run tag for the probe).
# The special tag ``imagenet`` is NOT a finetune run: it loads a full timm
# ImageNet-1k-pretrained vit_base via :func:`load_probe` (see the tag branch).
DATASETS: Dict[str, Tuple[str, dict, str]] = {
    "funny_birds":   ("funny_birds",  {"split": "train", "clean_only": True}, "funny-birds-train-clean"),
    "dsprites":      ("dsprites",      {"target": "shape"},                    "dsprites"),
    "colored_mnist": ("colored_mnist", {"split": "train"},                     "colored-mnist-train"),
    "imagenet":      ("imagenet_val_hf", {},                                   "imagenet"),
}

# timm model used for the ``imagenet`` dataset (ImageNet-1k pretrained, full head).
IMAGENET_TIMM = "vit_base_patch16_224"


class _TimmFullProbe(nn.Module):
    """Adapter exposing a *full* timm classifier (with its pretrained head) under
    the same surface the experiments expect from a :class:`Probe`: a ``backbone``
    submodule carrying ``.blocks`` / ``.embed_dim`` (so ``backbone.blocks.{b}``
    attribution layer-names and the site modules resolve) and a ``forward``
    that returns class logits directly. Unlike ``Probe`` the classification head
    lives *inside* ``backbone`` (timm's ``norm`` + ``head``); the LRP composite is
    type-based so it handles those modules the same way."""

    def __init__(self, timm_model: nn.Module):
        super().__init__()
        self.backbone = timm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# concept probe sites with a stable per-block module whose OUTPUT is the
# (B, N, embed_dim) site tensor. ``proj_drop`` = attention output-projection
# dropout; ``residual`` = the block output (full residual stream).
SITES = ("proj_drop", "residual")


def load_probe(tag: str, device: str, base: str = "vit_small",
               path: Optional[Path] = None) -> Tuple[nn.Module, dict, Path]:
    """Load a finetuned probe and rebuild the full (backbone + head) frozen eval
    model via the training registry.

    Looks up ``data/runs/finetune_<base>_<tag>/<ts>/best.pt`` (latest) unless an
    explicit ``path`` is given. The checkpoint records its own ``base``/``head``/
    ``head_kwargs``/``num_classes`` — those drive reconstruction, so ``base`` here
    only selects which run directory to glob.
    """
    if tag == "imagenet":
        import timm
        tm = timm.create_model(IMAGENET_TIMM, pretrained=True, num_classes=1000)
        tm = tm.eval().to(device)
        model = _TimmFullProbe(tm).eval().to(device)
        model.requires_grad_(False)
        ck = {"base": "vit_base", "head": "timm_builtin", "head_kwargs": {},
              "num_classes": 1000, "dataset": "imagenet_val_hf"}
        return model, ck, Path(f"timm:{IMAGENET_TIMM}")
    if path is None:
        runs = sorted((REPO_ROOT / "data" / "runs" / f"finetune_{base}_{tag}").glob("*/best.pt"))
        if not runs:
            raise FileNotFoundError(f"no probe under data/runs/finetune_{base}_{tag}/")
        path = runs[-1]
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_probe(base=ck["base"], head=ck["head"], num_classes=ck["num_classes"],
                        head_kwargs=ck.get("head_kwargs", {})).eval().to(device)
    model.backbone.load_state_dict(ck["backbone_state_dict"])
    model.head.load_state_dict(ck["head_state_dict"])
    model.requires_grad_(False)
    return model, ck, Path(path)


def select_correct(model, ds, classes: Sequence[int], n_per_class: int, device,
                   *, normalize=None, batch_size: int = 128, seed: int = 0,
                   max_scan: Optional[int] = None,
                   ) -> Dict[int, List[int]]:
    """Return ``{class: [dataset indices]}`` of up to ``n_per_class`` images per
    target class that the model classifies correctly. Scans the dataset in a
    fixed random order (so class-grouped datasets like dSprites are covered
    quickly) and stops once every target class is filled.

    ``max_scan`` caps how many images are examined: if a class is unfillable
    (e.g. a class the model never predicts correctly) the scan would
    otherwise crawl the *entire* dataset — for dSprites (737k images) that is
    ~20 min of CPU image decoding. With a cap the scan stops early and returns
    whatever filled (partial classes are expected and handled downstream).

    ``normalize`` (optional) is applied to each batch before the forward — pass
    it when the dataset yields un-normalized [0,1] images; leave ``None`` when the
    dataset transform already normalizes.
    """
    targets = set(classes)
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()
    loader = DataLoader(Subset(ds, perm), batch_size=batch_size, num_workers=0)
    sel: Dict[int, List[int]] = {c: [] for c in targets}
    pos = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x = normalize(x) if normalize is not None else x
            pred = model(x).argmax(-1).cpu()
            for j in range(len(y)):
                c = int(y[j])
                if c in targets and pred[j] == c and len(sel[c]) < n_per_class:
                    sel[c].append(perm[pos + j])
            pos += len(y)
            if all(len(sel[c]) >= n_per_class for c in targets):
                break
            if max_scan is not None and pos >= max_scan:
                break
    return sel


def site_layer_names(model, site: str) -> List[str]:
    """Per-block attribution layer-name strings for a probe ``site`` (one per
    block, indexed by block number). Single source of truth for the site → layer
    mapping used by the gallery / concept-flipping."""
    n_blocks = len(model.backbone.blocks)
    if site == "proj_drop":
        return [f"backbone.blocks.{b}.attn.proj_drop" for b in range(n_blocks)]
    if site == "residual":
        return [f"backbone.blocks.{b}" for b in range(n_blocks)]
    raise ValueError(f"unknown site {site!r}; pick from {SITES}")


def site_modules(model, site: str) -> list:
    """Per-block module objects for a probe ``site`` (module-object counterpart
    of :func:`site_layer_names`)."""
    blocks = model.backbone.blocks
    if site == "proj_drop":
        return [blocks[b].attn.proj_drop for b in range(len(blocks))]
    if site == "residual":
        return [blocks[b] for b in range(len(blocks))]
    raise ValueError(f"unknown site {site!r}; pick from {SITES}")


__all__ = [
    "REPO_ROOT", "DATASETS", "SITES",
    "load_probe", "select_correct", "site_layer_names", "site_modules", "backbone_transforms",
]
