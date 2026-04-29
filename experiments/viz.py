"""Visualisation helpers for the walkthrough notebook.

Each top-level function runs the attribution loops it needs and produces a
matplotlib ``Figure``. Notebook cells just call the helper and `plt.show()`
the result, so the notebook stays focused on layout / config rather than
plumbing. All numeric data (relevance scores, concept ids) appears as
subplot titles / labels — never as printed tables.

Conventions:

* ``image`` — a preprocessed `(1, 3, H, W)` tensor with ``requires_grad=True``.
* ``image_np`` — the de-normalised RGB image for display, shape `(H, W, 3)`.
* ``model`` — a ``timm`` ViT (anything providing ``model.blocks[i].attn``).
* ``attribution`` — ``crp.attribution.CondAttribution(model)``.
* ``composite`` — ``AttnLRPEpsilonComposite()`` or ``AttnLRPGammaComposite()``.

All concept classes auto-resolve their tap from ``concept.tap_name``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from timm.data import resolve_data_config

from crp.attention_concepts import (
    HeadConcept,
    HeadDimConcept,
    KQVHeadConcept,
    KQVHeadDimConcept,
    PARTS,
)


# ── canonical concept registry (used by atlas / reference-sample helpers) ────


CONCEPT_CLASSES: dict[str, type] = {
    "head":         HeadConcept,
    "head_dim":     HeadDimConcept,
    "kqv_head":     KQVHeadConcept,
    "kqv_head_dim": KQVHeadDimConcept,
}


# ── primitives ───────────────────────────────────────────────────────────────


def denormalize(image: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """Reverse the timm preprocessing for display. Returns ``(H, W, 3)``
    in ``[0, 1]``."""
    cfg = resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, -1, 1, 1)
    img = image.detach().cpu()
    img = img * std + mean
    return img.clamp(0, 1)[0].permute(1, 2, 0).numpy()


def overlay(
    ax: plt.Axes,
    image_np: np.ndarray,
    heatmap_np: np.ndarray,
    *,
    alpha_img: float = 0.4,
    alpha_hm: float = 0.6,
    cmap: str = "bwr",
) -> None:
    """Show ``image_np`` + ``heatmap_np`` (symmetric BWR by default) on ``ax``."""
    vmax = float(np.abs(heatmap_np).max()) or 1.0
    ax.imshow(image_np, alpha=alpha_img)
    ax.imshow(heatmap_np, cmap=cmap, alpha=alpha_hm, vmin=-vmax, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])


def _heatmap_2d(result_heatmap: torch.Tensor) -> np.ndarray:
    """Reduce ``result.heatmap[0]`` to a 2D ``(H, W)`` ndarray, summing the
    channel dim if present."""
    hm = result_heatmap
    if hm.dim() == 3:
        hm = hm.sum(dim=0)
    return hm.detach().cpu().numpy()


def _attribute_concept(
    attribution,
    composite,
    image: torch.Tensor,
    layer_name: str,
    concept,
    concept_id,
    target_class: int,
) -> np.ndarray:
    """One conditional attribution → 2D heatmap."""
    image.grad = None
    res = attribution(
        image, [{layer_name: [concept_id], "y": [target_class]}],
        composite, mask_map=concept.mask,
    )
    return _heatmap_2d(res.heatmap[0])


def _per_concept_scores(
    attribution,
    composite,
    image: torch.Tensor,
    layer_name: str,
    concept,
    target_class: int,
) -> torch.Tensor:
    """Per-concept relevance under the target class — no concept mask in
    the condition; record at the concept's tap and let
    ``concept.attribute`` reduce to scores."""
    image.grad = None
    res = attribution(
        image, [{"y": [target_class]}], composite,
        mask_map=concept.mask, record_layer=[layer_name],
    )
    rel = res.relevances[layer_name]
    return concept.attribute(rel, layer_name=layer_name, abs_norm=False)[0]


def _enumerate_ids(name: str, num_heads: int, head_dim: int) -> list:
    if name == "head":
        return list(range(num_heads))
    if name == "head_dim":
        return [(h, d) for h in range(num_heads) for d in range(head_dim)]
    if name == "kqv_head":
        return [(p, h) for p in PARTS for h in range(num_heads)]
    if name == "kqv_head_dim":
        return [
            (p, h, d) for p in PARTS for h in range(num_heads) for d in range(head_dim)
        ]
    raise ValueError(name)


def _label_id(name: str, cid) -> str:
    if name == "head":
        return f"h{int(cid)}"
    if name == "head_dim":
        return f"h{int(cid[0])}/d{int(cid[1])}"
    if name == "kqv_head":
        return f"{cid[0]}/h{int(cid[1])}"
    if name == "kqv_head_dim":
        return f"{cid[0]}/h{int(cid[1])}/d{int(cid[2])}"
    raise ValueError(name)


# ── concept atlas (single image, all 4 granularities × multiple blocks) ──────


def plot_concept_atlas(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    *,
    blocks: Sequence[int],
    target_class: int,
    top_k: int = 4,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """Single-image multi-granularity multi-block atlas.

    Rows: ``(granularity, block)`` pairs, ordered (head, head_dim, kqv_head,
    kqv_head_dim) × (block_0, block_1, …). Columns: input + the ``top_k``
    most-relevant concepts of that granularity at that block, ranked by
    absolute relevance under ``target_class``. Each heatmap is overlaid on
    the input; subplot titles show the concept id and its raw relevance.
    """
    image_np = denormalize(image, model)
    n_rows = len(CONCEPT_CLASSES) * len(blocks)
    n_cols = top_k + 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols, 2.0 * n_rows + 0.5),
        squeeze=False,
    )

    row = 0
    for name, cls in CONCEPT_CLASSES.items():
        concept = cls(model)
        for block_idx in blocks:
            num_heads = model.blocks[block_idx].attn.num_heads
            head_dim = model.blocks[block_idx].attn.head_dim
            layer = f"blocks.{block_idx}.attn.{concept.tap_name}"

            scores = _per_concept_scores(
                attribution, composite, image, layer, concept, target_class
            )
            all_ids = _enumerate_ids(name, num_heads, head_dim)
            k = min(top_k, len(all_ids))
            top_idx = torch.topk(scores.flatten().abs(), k=k).indices.tolist()
            top_pairs = [(all_ids[i], scores.flatten()[i].item()) for i in top_idx]

            axes[row, 0].imshow(image_np)
            axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
            axes[row, 0].set_ylabel(
                f"{name}\nblk {block_idx}", fontsize=9, rotation=0,
                ha="right", va="center", labelpad=22,
            )
            for c_i, (cid, score) in enumerate(top_pairs, start=1):
                hm = _attribute_concept(
                    attribution, composite, image, layer, concept, cid, target_class
                )
                overlay(axes[row, c_i], image_np, hm)
                axes[row, c_i].set_title(
                    f"{_label_id(name, cid)}\nr={score:+.2g}",
                    fontsize=8,
                )
            for c_i in range(k + 1, n_cols):
                axes[row, c_i].axis("off")
            row += 1

    # column 0 has no title; label "input"
    axes[0, 0].set_title("input", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


# ── all heads of one block (HeadConcept) ─────────────────────────────────────


def plot_per_head(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    *,
    block_idx: int,
    target_class: int,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """Show every head's HeadConcept heatmap on the same image, sorted by
    descending |relevance|. Useful to see how heads attend to different
    spatial structures of the same target class."""
    concept = HeadConcept(model)
    layer = f"blocks.{block_idx}.attn.{concept.tap_name}"
    num_heads = model.blocks[block_idx].attn.num_heads
    image_np = denormalize(image, model)

    scores = _per_concept_scores(
        attribution, composite, image, layer, concept, target_class
    )
    order = torch.argsort(scores.abs(), descending=True).tolist()

    n_cols = min(6, num_heads + 1)
    n_rows = (num_heads + n_cols) // n_cols  # +1 cell for the input image
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.7 * n_cols, 1.7 * n_rows + 0.5), squeeze=False,
    )
    axes_flat = axes.flatten()

    axes_flat[0].imshow(image_np)
    axes_flat[0].set_xticks([]); axes_flat[0].set_yticks([])
    axes_flat[0].set_title("input", fontsize=8)

    for slot, h in enumerate(order, start=1):
        hm = _attribute_concept(
            attribution, composite, image, layer, concept, h, target_class
        )
        overlay(axes_flat[slot], image_np, hm)
        axes_flat[slot].set_title(
            f"h{h}  r={scores[h].item():+.2g}", fontsize=8,
        )
    for slot in range(num_heads + 1, len(axes_flat)):
        axes_flat[slot].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


# ── one concept across multiple blocks ───────────────────────────────────────


def plot_layer_evolution(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    *,
    concept_class: type = HeadConcept,
    concept_id=0,
    blocks: Sequence[int] = (0, 3, 6, 9, 11),
    target_class: int = 0,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """One concept's heatmap across multiple blocks. Demonstrates how the
    same head/dim/(part,head) shifts spatial focus through the network."""
    concept = concept_class(model)
    image_np = denormalize(image, model)

    n_cols = len(blocks) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(1.9 * n_cols, 2.5))
    axes[0].imshow(image_np); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("input", fontsize=9)

    for c_i, block_idx in enumerate(blocks, start=1):
        layer = f"blocks.{block_idx}.attn.{concept.tap_name}"
        # Per-image relevance score for this block (so the title is honest).
        scores = _per_concept_scores(
            attribution, composite, image, layer, concept, target_class
        )
        # Get the score for this concept_id specifically.
        if isinstance(concept_id, int):
            score = scores.flatten()[concept_id].item()
        else:
            # Tuple → look up in the enumerate-ids order.
            num_heads = model.blocks[block_idx].attn.num_heads
            head_dim = model.blocks[block_idx].attn.head_dim
            ids = _enumerate_ids(
                {v: k for k, v in CONCEPT_CLASSES.items()}[concept_class],
                num_heads, head_dim,
            )
            try:
                idx = ids.index(tuple(concept_id))
                score = scores.flatten()[idx].item()
            except (ValueError, IndexError):
                score = float("nan")

        hm = _attribute_concept(
            attribution, composite, image, layer, concept, concept_id, target_class
        )
        overlay(axes[c_i], image_np, hm)
        name = {v: k for k, v in CONCEPT_CLASSES.items()}[concept_class]
        axes[c_i].set_title(
            f"blk {block_idx}\n{_label_id(name, concept_id)}  r={score:+.2g}",
            fontsize=8,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


# ── K vs Q vs V split, one head ─────────────────────────────────────────────


def plot_kqv_split(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    *,
    block_idx: int,
    head_id: int,
    target_class: int,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """For one (block, head): three heatmaps comparing Q / K / V using
    KQVHeadConcept. Different parts of the same head can highlight different
    aspects of the same image (e.g. K = where to look, V = what to convey)."""
    concept = KQVHeadConcept(model)
    layer = f"blocks.{block_idx}.attn.{concept.tap_name}"
    image_np = denormalize(image, model)

    scores = _per_concept_scores(
        attribution, composite, image, layer, concept, target_class
    )

    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.5))
    axes[0].imshow(image_np); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("input", fontsize=9)
    for c_i, part in enumerate(PARTS, start=1):
        score = scores[c_i - 1, head_id].item()
        hm = _attribute_concept(
            attribution, composite, image, layer, concept,
            (part, head_id), target_class,
        )
        overlay(axes[c_i], image_np, hm)
        axes[c_i].set_title(f"{part}/h{head_id}  r={score:+.2g}", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


# ── selected head_dim concepts on one head ──────────────────────────────────


def plot_head_dim_grid(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    *,
    block_idx: int,
    head_id: int,
    target_class: int,
    n_dims: int = 8,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """For one (block, head): the top-``n_dims`` head_dim concepts ranked
    by |relevance|. Each cell shows one (head, dim)'s heatmap. Demonstrates
    the per-dim granularity inside one head."""
    concept = HeadDimConcept(model)
    layer = f"blocks.{block_idx}.attn.{concept.tap_name}"
    image_np = denormalize(image, model)

    head_dim = model.blocks[block_idx].attn.head_dim
    scores = _per_concept_scores(
        attribution, composite, image, layer, concept, target_class
    )
    head_scores = scores[head_id]  # (head_dim,)
    order = torch.argsort(head_scores.abs(), descending=True).tolist()[:n_dims]

    n_cols = min(n_dims + 1, 5)
    n_rows = (n_dims + n_cols) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.7 * n_cols, 1.7 * n_rows + 0.5), squeeze=False,
    )
    axes_flat = axes.flatten()
    axes_flat[0].imshow(image_np); axes_flat[0].set_xticks([]); axes_flat[0].set_yticks([])
    axes_flat[0].set_title(f"h{head_id}\ninput", fontsize=8)

    for slot, dim_id in enumerate(order, start=1):
        hm = _attribute_concept(
            attribution, composite, image, layer, concept,
            (head_id, dim_id), target_class,
        )
        overlay(axes_flat[slot], image_np, hm)
        axes_flat[slot].set_title(
            f"h{head_id}/d{dim_id}  r={head_scores[dim_id].item():+.2g}",
            fontsize=8,
        )
    for slot in range(n_dims + 1, len(axes_flat)):
        axes_flat[slot].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


# ── reference samples (top-N images per concept, with overlaid heatmap) ─────


def plot_reference_samples(
    fv,
    dataset,
    *,
    concept_name: str,
    block_idx: int,
    target_image: torch.Tensor,
    target_class: int,
    model: torch.nn.Module,
    attribution,
    composite,
    n_top_concepts: int = 4,
    n_refs_per_concept: int = 4,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """For the top-``n_top_concepts`` concepts of ``concept_name`` (ranked
    on ``target_image``'s relevance), fetch the top-``n_refs_per_concept``
    dataset images that maximise each concept's relevance, and render each
    reference image with its conditional heatmap overlaid.

    Rows: top concepts. Cols: reference samples (left) + the target image
    with this concept's heatmap on the right.
    """
    cls = CONCEPT_CLASSES[concept_name]
    concept = cls(model)
    layer = f"blocks.{block_idx}.attn.{concept.tap_name}"
    num_heads = model.blocks[block_idx].attn.num_heads
    head_dim = model.blocks[block_idx].attn.head_dim

    target_np = denormalize(target_image, model)

    # 1. Pick the top concepts on the target image.
    scores = _per_concept_scores(
        attribution, composite, target_image, layer, concept, target_class
    )
    flat = scores.flatten()
    flat_top = torch.topk(flat.abs(), k=min(n_top_concepts, flat.numel())).indices.tolist()

    # 2. Plot.
    n_cols = n_refs_per_concept + 1   # +1 for the target heatmap
    n_rows = len(flat_top)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.8 * n_cols, 1.8 * n_rows + 0.5),
        squeeze=False,
    )

    for row_i, flat_idx in enumerate(flat_top):
        # Use FV.get_max_reference for actual reference images.
        ref_dict = fv.get_max_reference(
            [flat_idx], layer, mode="relevance", r_range=(0, n_refs_per_concept),
            composite=composite, plot_fn=None,
        )
        samples, ref_heatmaps = ref_dict[flat_idx]
        # samples: (N, 3, H, W); ref_heatmaps: (N, H, W)
        for c_i in range(n_refs_per_concept):
            if c_i >= len(samples):
                axes[row_i, c_i].axis("off")
                continue
            ref_img = samples[c_i].cpu().numpy().transpose(1, 2, 0)
            ref_img = (ref_img - ref_img.min()) / max(ref_img.max() - ref_img.min(), 1e-9)
            ref_hm = ref_heatmaps[c_i].cpu().numpy()
            overlay(axes[row_i, c_i], ref_img, ref_hm)
            axes[row_i, c_i].set_title(f"ref #{c_i + 1}", fontsize=8)

        # Right column: the target image with this concept's heatmap.
        all_ids = _enumerate_ids(concept_name, num_heads, head_dim)
        cid = all_ids[flat_idx]
        target_hm = _attribute_concept(
            attribution, composite, target_image, layer, concept, cid, target_class,
        )
        overlay(axes[row_i, -1], target_np, target_hm)
        score = flat[flat_idx].item()
        axes[row_i, -1].set_title(
            f"target\n{_label_id(concept_name, cid)}  r={score:+.2g}",
            fontsize=8,
        )
        # Row label.
        axes[row_i, 0].set_ylabel(
            _label_id(concept_name, cid), fontsize=9, rotation=0,
            ha="right", va="center", labelpad=22,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


__all__ = [
    "CONCEPT_CLASSES",
    "denormalize",
    "overlay",
    "plot_concept_atlas",
    "plot_per_head",
    "plot_layer_evolution",
    "plot_kqv_split",
    "plot_head_dim_grid",
    "plot_reference_samples",
]
