"""Visualisation helpers for the unfolded-attention CRP walkthrough.

These adapt the patterns from ``experiments/viz.py`` (the legacy walkthrough)
to the new attention-unfolded concept classes:

* :class:`crp.attention_concepts.HeadConcept` (per-head, optional dim_split)
* :class:`crp.attention_concepts.QConcept` / :class:`KConcept` / :class:`VConcept`
  (per-head, optional dim_split, target rope_q / rope_k / v_id)
* :class:`crp.attention_concepts.AttnOutputDimConcept` (per-channel, spatial-aggregated)
* :class:`crp.attention_concepts.RegisterTokenConcept` (per prefix token)

The functions are deliberately concept-agnostic where possible — they
read each concept's ``LAYER_SUFFIX`` and ``_dims`` registration to figure
out the right hook point and id-space. The atlas helpers list concept ids,
attribute one image, then plot one image+heatmap panel per id.

All paths assume the model has been substituted with
``EvaAttentionUnfolded`` via :class:`EvaAttentionSubstitutionCanonizer`
(installed automatically by :class:`AttnLRPCombinedComposite` when
``use_unfolded_attention=True``).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as _PIL

from crp.attention_concepts import (
    HeadConcept,
    QConcept,
    KConcept,
    VConcept,
    AttnOutputDimConcept,
    RegisterTokenConcept,
)


# ─── image / heatmap helpers (lifted from experiments/viz.py) ────────────────


def to_display(image: torch.Tensor) -> np.ndarray:
    """Convert a model-input tensor to a display-ready ``(H, W, 3)``
    ndarray in ``[0, 1]``. No model dependency, no denormalize step —
    dataset transforms emit unnormalized tensors, so display is just
    clamp + permute. Accepts shape ``(1, 3, H, W)`` or ``(3, H, W)``.
    """
    img = image.detach().cpu()
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img.clamp(0, 1)[0].permute(1, 2, 0).numpy()


def _heatmap_to_rgb(heatmap_np: np.ndarray, cmap: str = "seismic") -> np.ndarray:
    vmax = float(np.abs(heatmap_np).max()) or 1.0
    norm = np.clip(heatmap_np / vmax, -1.0, 1.0) / 2.0 + 0.5
    cmap_obj = plt.get_cmap(cmap)
    return cmap_obj(norm)[..., :3]


def panel(ax: plt.Axes, image_np: np.ndarray, heatmap_np: np.ndarray, *, cmap="seismic"):
    """Image + heatmap side-by-side on one axis."""
    H, W = image_np.shape[:2]
    if heatmap_np.shape != (H, W):
        heatmap_np = np.array(
            _PIL.fromarray(heatmap_np.astype(np.float32), mode="F").resize(
                (W, H), resample=_PIL.NEAREST,
            )
        )
    img = np.clip(image_np, 0, 1)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    hm_rgb = _heatmap_to_rgb(heatmap_np, cmap=cmap)
    spacer = np.ones((H, 4, 3))
    ax.imshow(np.concatenate([img, spacer, hm_rgb], axis=1))
    ax.set_xticks([]); ax.set_yticks([])


def _heatmap_2d(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 3:
        t = t.sum(dim=0)
    return t.detach().cpu().numpy()


# ─── attribution + per-concept scoring ───────────────────────────────────────


def attribute_at_concept(
    attribution, composite, image: torch.Tensor, layer_name: str,
    concept, concept_id, target_class: int,
    extra_conditions: Optional[dict] = None,
    exclude_parallel: bool = True,
    preprocess_fn: Optional[Callable] = None,
) -> np.ndarray:
    """One conditional attribution → 2D input-space heatmap.

    ``cond = {layer_name: [concept_id], "y": [target_class]}``. The
    ``mask_map`` is the concept's ``mask`` method, which zeros out
    non-selected dimensions per the concept class's semantics.

    ``preprocess_fn`` is the per-batch normalize step ``(x - mean) / std``;
    applied to ``image`` before the forward pass. ``None`` means inputs
    are already in the model's expected distribution (e.g. visinf
    vit_base trained on raw ``[0, 1]``).

    When ``preprocess_fn`` is supplied, we make the **normalized** tensor
    a fresh leaf (``detach().requires_grad_(True)``) so that
    ``CondAttribution`` can populate ``.grad`` on it during backward.
    The heatmap lands in normalize-output space, which is spatially
    equivalent to input space for any per-channel affine normalize
    (the operation is just a per-channel rescale + shift; channels are
    summed in the heatmap reduction).
    """
    cond = {layer_name: [concept_id], "y": [target_class]}
    if extra_conditions:
        cond.update(extra_conditions)
    if preprocess_fn is None:
        image.grad = None
        x = image
    else:
        x = preprocess_fn(image).detach().requires_grad_(True)
    res = attribution(
        x, [cond], composite, mask_map=concept.mask,
        exclude_parallel=exclude_parallel,
    )
    return _heatmap_2d(res.heatmap[0])


def per_concept_scores(
    attribution, composite, image: torch.Tensor, layer_name: str,
    concept, target_class: int,
    extra_conditions: Optional[dict] = None,
    exclude_parallel: bool = True,
    preprocess_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """Attribute toward target class, record at the concept's tap, reduce
    via ``concept.attribute``. Returns a flattened 1D tensor of per-id
    scores in the concept's row-major id order.

    See :func:`attribute_at_concept` for the ``preprocess_fn`` contract.
    """
    cond = {"y": [target_class]}
    if extra_conditions:
        cond.update(extra_conditions)
    if preprocess_fn is None:
        image.grad = None
        x = image
    else:
        x = preprocess_fn(image).detach().requires_grad_(True)
    res = attribution(
        x, [cond], composite,
        mask_map=concept.mask, record_layer=[layer_name],
        exclude_parallel=exclude_parallel,
    )
    rel = res.relevances[layer_name]
    scores = concept.attribute(rel, layer_name=layer_name, abs_norm=False)[0]
    return scores.flatten()


# ─── id enumeration per concept type ─────────────────────────────────────────


def enumerate_ids(concept, layer_name: str) -> List:
    """List all concept ids for a given concept in row-major order.

    Reads dims from the concept's captured attributes (``num_heads``,
    ``head_dim``, ``num_prefix_tokens``) — set at concept construction;
    the ``layer_name`` argument is accepted for API symmetry but
    unused (dims are model-wide invariants).
    """
    num_heads = int(concept.num_heads)
    head_dim = int(concept.head_dim)
    npt = int(concept.num_prefix_tokens)
    if isinstance(concept, (HeadConcept, QConcept, KConcept, VConcept)):
        if concept.dim_split:
            return [(h, d) for h in range(num_heads) for d in range(head_dim)]
        return list(range(num_heads))
    if isinstance(concept, AttnOutputDimConcept):
        return list(range(num_heads * head_dim))
    if isinstance(concept, RegisterTokenConcept):
        if concept.dim_split:
            return [(t, c) for t in range(npt) for c in range(num_heads * head_dim)]
        return list(range(npt))
    raise TypeError(f"Unknown concept type: {type(concept).__name__}")


def label_id(concept, cid) -> str:
    """Short concept-specific identifier for plot titles / row labels.

    Format conventions:

    * ``HeadConcept``        : ``H{h}``      (``H{h},D{d}`` if dim_split)
    * ``QConcept``           : ``H{h},Q``    (``H{h},Q,D{d}`` if dim_split)
    * ``KConcept``           : ``H{h},K``    (``H{h},K,D{d}`` if dim_split)
    * ``VConcept``           : ``H{h},V``    (``H{h},V,D{d}`` if dim_split)
    * ``AttnOutputDimConcept``: ``D{d}``
    * ``RegisterTokenConcept``: ``cls`` / ``reg{i}``  (``,D{d}`` if dim_split)

    Use :func:`row_label` to pair this with a trimmed layer path for the
    two-line row labels used in reference-sample grids.
    """
    if isinstance(concept, HeadConcept):
        if concept.dim_split:
            return f"H{int(cid[0])},D{int(cid[1])}"
        return f"H{int(cid) if isinstance(cid, int) else int(cid[0])}"
    if isinstance(concept, (QConcept, KConcept, VConcept)):
        letter = type(concept).__name__[0]  # 'Q' / 'K' / 'V'
        if concept.dim_split:
            return f"H{int(cid[0])},{letter},D{int(cid[1])}"
        return f"H{int(cid) if isinstance(cid, int) else int(cid[0])},{letter}"
    if isinstance(concept, AttnOutputDimConcept):
        return f"D{int(cid) if isinstance(cid, int) else int(cid[0])}"
    if isinstance(concept, RegisterTokenConcept):
        if concept.dim_split:
            tid, dim = int(cid[0]), int(cid[1])
            base = "cls" if tid == 0 else f"reg{tid - 1}"
            return f"{base},D{dim}"
        tid = int(cid) if isinstance(cid, int) else int(cid[0])
        return "cls" if tid == 0 else f"reg{tid - 1}"
    raise TypeError(f"Unknown concept: {type(concept).__name__}")


def row_label(concept, cid, layer_name: str) -> str:
    """Two-line row label: trimmed layer path on line 1, concept id on
    line 2. Used by reference-sample grids and the cascade row headers.

    Example outputs::

        blocks.22.attn.context
        H1,D15

        blocks.5.attn.q_id
        H0,Q
    """
    # Strip the ``backbone.`` prefix and any leading module wrapping —
    # produces ``blocks.{i}.attn.{leaf}`` from ``backbone.blocks.{i}.attn.{leaf}``
    short = layer_name.split("backbone.", 1)[-1]
    return f"{short}\n{label_id(concept, cid)}"


# ─── atlases (one image, all/top-K concepts at one layer) ────────────────────


def plot_concept_atlas(
    image: torch.Tensor, model: torch.nn.Module,
    attribution, composite, *,
    concept, layer_name: str, target_class: int,
    top_k: Optional[int] = None,
    cell_size: float = 4.0,
    title_prefix: Optional[str] = None,
    preprocess_fn: Optional[Callable] = None,
) -> plt.Figure:
    """Generic atlas: one image+heatmap panel per (top-K-by-relevance)
    concept id, in concept-id order. Works for any of the 6 concept
    classes.

    ``top_k=None`` shows every concept id (atlas may get wide); pass an
    int to keep only the top-K by absolute per-concept relevance.

    ``preprocess_fn`` is the normalize step applied to ``image`` before
    each forward pass. See :func:`attribute_at_concept`.
    """
    image_np = to_display(image)
    all_ids = enumerate_ids(concept, layer_name)

    # Per-concept relevance scores under no concept conditioning — used
    # to rank ids for the top-K selection.
    scores = per_concept_scores(
        attribution, composite, image, layer_name, concept, target_class,
        preprocess_fn=preprocess_fn,
    )
    ranked_idx = torch.argsort(scores.abs(), descending=True).cpu().tolist()
    if top_k is not None:
        ranked_idx = ranked_idx[:top_k]
    # Re-sort the kept ids by id for a consistent left-to-right reading.
    kept = sorted(ranked_idx)
    ids = [all_ids[i] for i in kept]
    scores_kept = [float(scores[i]) for i in kept]

    n = len(ids)
    fig, axes = plt.subplots(1, n, figsize=(cell_size * n * 2, cell_size))
    if n == 1:
        axes = [axes]
    for ax, cid, sc in zip(axes, ids, scores_kept):
        hm = attribute_at_concept(
            attribution, composite, image, layer_name, concept, cid, target_class,
            preprocess_fn=preprocess_fn,
        )
        panel(ax, image_np, hm)
        ax.set_title(
            f"{label_id(concept, cid)}\nscore={sc:.2e}",
            fontsize=8, family="monospace",
        )
    title = f"{type(concept).__name__} @ {layer_name}"
    if title_prefix:
        title = f"{title_prefix} — {title}"
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


# ─── FV reference samples per concept (top-K activating images) ──────────────


def build_layer_map(concept_layer_pairs: Sequence[Tuple[object, str]]) -> Dict[str, object]:
    """Convenience: turn a list of ``(concept_instance, layer_name)`` into the
    ``layer_map`` dict that :class:`crp.visualization.FeatureVisualization`
    expects (mapping layer_name → concept). One concept per layer; if you
    need multiple concepts at the same layer, use multiple FV instances."""
    return {layer: concept for concept, layer in concept_layer_pairs}


def plot_reference_samples(
    fv, *, concept_layer: str, concept, top_concept_ids: Sequence,
    n_refs: int = 4, cell_size: float = 4.0,
    title_prefix: Optional[str] = None,
) -> plt.Figure:
    """For each id in ``top_concept_ids``, fetch the top-``n_refs`` images
    from the FV cache and plot them in one row.

    ``fv`` is a populated :class:`FeatureVisualization`; ``concept_layer``
    is the layer key in ``fv.layer_map``; ``concept`` is the same concept
    used to build the index (used here only for label formatting).
    """
    n = len(top_concept_ids)
    fig, axes = plt.subplots(n, n_refs, figsize=(cell_size * n_refs, cell_size * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for row, cid in enumerate(top_concept_ids):
        try:
            ref = fv.get_max_reference(
                [cid] if not isinstance(cid, list) else cid,
                concept_layer, mode="relevance", r_range=(0, n_refs),
                composite=fv._cache_composite if hasattr(fv, "_cache_composite") else None,
                rf=False, plot_fn=None,
            )
        except Exception as e:
            for ax in axes[row]:
                ax.text(0.5, 0.5, f"err: {e!s}"[:30], ha="center", va="center", fontsize=6)
                ax.axis("off")
            continue
        # ``ref`` is dict {cid: [image_tensors]} or similar. Normalize to flat list.
        imgs = ref[cid] if isinstance(ref, dict) else ref
        for ax, img in zip(axes[row], imgs[:n_refs]):
            if hasattr(img, "detach"):
                img_np = to_display(img)
            else:
                img_np = np.asarray(img)
            ax.imshow(np.clip(img_np, 0, 1))
            ax.axis("off")
        axes[row, 0].set_ylabel(
            row_label(concept, cid, concept_layer),
            fontsize=7, rotation=0, ha="right", va="center",
            family="monospace",
        )
    title = f"{type(concept).__name__} top-{n_refs} reference samples @ {concept_layer}"
    if title_prefix:
        title = f"{title_prefix} — {title}"
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


# ─── conditional propagation cascade ─────────────────────────────────────────


def plot_cascade(
    image: torch.Tensor, model: torch.nn.Module,
    attribution, composite, *,
    concept, layer_names: Sequence[str], target_class: int,
    top_k: int = 4, cell_size: float = 4.0,
    preprocess_fn: Optional[Callable] = None,
) -> Tuple[plt.Figure, Dict[str, List]]:
    """Incremental conditional cascade: walk ``layer_names`` (a list of
    fully-qualified hookable submodule paths, deep → shallow). At each
    layer, pick the top-``k`` concept ids by per-concept relevance under
    the *cumulative* conditioning of all already-selected deeper-layer
    ids; render each as a heatmap row; pass the union of selections
    forward to the next (shallower) layer.

    The caller is responsible for picking the right paths — they vary
    per model (wrapped vs bare ViT) and per concept type (``context``,
    ``rope_q``, ``proj_drop``, …). Use the discovery cell in the
    walkthrough notebook to enumerate the available paths.

    Returns ``(fig, selected)`` where ``selected[layer_name]`` is the
    list of concept ids chosen at that depth (in order of relevance).
    """
    image_np = to_display(image)
    selected: Dict[str, List] = {}
    extra_conditions: Dict[str, List] = {}

    n_rows = len(layer_names)
    fig, axes = plt.subplots(
        n_rows, top_k, figsize=(cell_size * top_k * 2, cell_size * n_rows),
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, layer_name in enumerate(layer_names):
        all_ids = enumerate_ids(concept, layer_name)
        # Cumulative conditioning: pass `extra_conditions` (deeper layer ids)
        # so the cascade narrows progressively.
        scores = per_concept_scores(
            attribution, composite, image, layer_name, concept, target_class,
            extra_conditions=extra_conditions,
            exclude_parallel=False,  # cascade needs unified backward
            preprocess_fn=preprocess_fn,
        )
        ranked = torch.argsort(scores.abs(), descending=True).cpu().tolist()
        kept_idx = ranked[:top_k]
        kept_ids = [all_ids[i] for i in kept_idx]
        selected[layer_name] = kept_ids
        # Render this row.
        for ax, cid in zip(axes[row], kept_ids):
            hm = attribute_at_concept(
                attribution, composite, image, layer_name, concept, cid, target_class,
                extra_conditions=extra_conditions, exclude_parallel=False,
                preprocess_fn=preprocess_fn,
            )
            panel(ax, image_np, hm)
            ax.set_title(f"{layer_name.split('.')[-2]} {label_id(concept, cid)}", fontsize=7)
        # Add THIS layer's selection to the extra conditioning for the next layer.
        extra_conditions[layer_name] = kept_ids

    fig.suptitle(
        f"Cascade — {type(concept).__name__} (top-{top_k} per layer, deep→shallow)",
        fontsize=10,
    )
    plt.tight_layout()
    return fig, selected


__all__ = [
    "to_display", "panel",
    "attribute_at_concept", "per_concept_scores",
    "enumerate_ids", "label_id", "row_label",
    "plot_concept_atlas",
    "build_layer_map", "plot_reference_samples",
    "plot_cascade",
]
