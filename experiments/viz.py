"""Visualisation helpers for the walkthrough notebook.

Each top-level plotter runs the attribution loops it needs and produces
matplotlib ``Figure`` objects. Notebook cells just call the helper and
render. All numeric data (relevance scores, concept ids) appears as
subplot titles / labels — never as printed tables.

Conventions
-----------

* ``image`` — preprocessed ``(1, 3, H, W)`` tensor with
  ``requires_grad=True``.
* ``image_np`` — denormalised RGB image for display, shape ``(H, W, 3)``.
* ``model`` — ``timm`` ViT (anything providing ``model.blocks[i].attn``).
* ``attribution`` — ``crp.attribution.CondAttribution(model)``.
* ``composite`` — ``AttnLRPEpsilonComposite()`` or
  ``AttnLRPGammaComposite()``.

Layout primitives
-----------------

* :func:`panel` — draws ``image_np`` and ``heatmap_np`` as a single
  *horizontally concatenated* image. Replaces alpha-blended overlays so
  the input stays vivid and the heatmap stays fully saturated.
* All plotters accept ``cell_size`` (per-panel width in inches) and
  ``figsize`` (overrides the auto-computed size).
* Cells within a row are always ordered by **concept id** (``head_id``,
  ``(part, head)``, ``dim_id``, …) — top-K filtering selects the cells
  but the displayed order is deterministic by id.

Concept hierarchy
-----------------

The walkthrough uses these helpers in increasing specificity:

1. :func:`plot_head_atlas` — one figure per layer, all (or top-K) heads
   sorted by id. The coarsest, most informative view.
2. :func:`plot_kqv_head_atlas` — one figure per layer, three rows
   (K, Q, V) × heads. Couples the K/Q/V triple per head for direct
   comparison.
3. :func:`plot_head_dim_closeup` — fixes one (layer, head), shows
   top-K head_dim concepts in id order.
4. :func:`plot_kqv_head_dim_closeup` — same fix, three rows × top-K
   dims.
5. :func:`plot_conditional_cascade` — backward conditional CRP through
   a list of layers, picking top-K heads per layer, fetching FV
   reference samples for each picked head.
6. :func:`plot_reference_samples` — for one granularity at one layer,
   show the top concepts' reference samples + their conditional
   heatmaps on the target image.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

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


# ── canonical concept registry ───────────────────────────────────────────────


CONCEPT_CLASSES: dict[str, type] = {
    "head":         HeadConcept,
    "head_dim":     HeadDimConcept,
    "kqv_head":     KQVHeadConcept,
    "kqv_head_dim": KQVHeadDimConcept,
}


# ── primitives ───────────────────────────────────────────────────────────────


def denormalize(image: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """Reverse the timm preprocessing. Returns ``(H, W, 3)`` in ``[0, 1]``."""
    cfg = resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, -1, 1, 1)
    img = image.detach().cpu()
    img = img * std + mean
    return img.clamp(0, 1)[0].permute(1, 2, 0).numpy()


def _heatmap_to_rgb(heatmap_np: np.ndarray, cmap: str = "seismic") -> np.ndarray:
    """Map a 2D relevance heatmap to an RGB ``(H, W, 3)`` array using a
    symmetric colormap. ``±vmax`` saturate the colour extremes; zero is the
    midpoint."""
    vmax = float(np.abs(heatmap_np).max()) or 1.0
    norm = np.clip(heatmap_np / vmax, -1.0, 1.0) / 2.0 + 0.5  # → [0, 1]
    cmap_obj = plt.get_cmap(cmap)
    return cmap_obj(norm)[..., :3]  # drop alpha


def panel(
    ax: plt.Axes,
    image_np: np.ndarray,
    heatmap_np: np.ndarray,
    *,
    cmap: str = "seismic",
    spacer_px: int = 4,
) -> None:
    """Draw ``image_np`` and ``heatmap_np`` side-by-side as a single concatenated
    image on ``ax``. Resizes the heatmap to the image's HxW if they differ
    (heatmaps from CRP are already at input resolution; this is a safety net).

    The image stays unaltered (no alpha) and the heatmap stays fully saturated
    on a symmetric colormap. A thin white spacer separates the two halves so
    the boundary is obvious.
    """
    if heatmap_np.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got {heatmap_np.shape}")
    H, W = image_np.shape[:2]
    if heatmap_np.shape != (H, W):
        # Resize via PIL (nearest, since heatmaps are typically already on the
        # input grid). Avoids requiring scipy.
        from PIL import Image as _PILImage
        heatmap_np = np.array(
            _PILImage.fromarray(heatmap_np.astype(np.float32), mode="F").resize(
                (W, H), resample=_PILImage.NEAREST
            )
        )
    img_clipped = np.clip(image_np, 0, 1)
    if img_clipped.ndim == 2:
        img_clipped = np.stack([img_clipped] * 3, axis=-1)
    hm_rgb = _heatmap_to_rgb(heatmap_np, cmap=cmap)
    spacer = np.ones((H, max(spacer_px, 1), 3))
    combined = np.concatenate([img_clipped, spacer, hm_rgb], axis=1)
    ax.imshow(combined)
    ax.set_xticks([])
    ax.set_yticks([])


# ── attribution helpers (private) ────────────────────────────────────────────


def _heatmap_2d(result_heatmap: torch.Tensor) -> np.ndarray:
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
    extra_conditions: Optional[dict] = None,
    exclude_parallel: bool = True,
) -> np.ndarray:
    """One conditional attribution → 2D heatmap.

    ``extra_conditions`` is folded into the condition dict alongside
    ``{layer_name: [concept_id], "y": [target_class]}`` — used by the
    conditional-cascade helper to pass deeper-layer masks. The cascade
    sets ``exclude_parallel=False`` to avoid zennit-crp's
    ``partial_backward`` path, which can't chain multi-layer conditions
    through our custom autograd Functions; one unified backward fires
    every ``MaskHook`` in topological order.
    """
    image.grad = None
    cond = {layer_name: [concept_id], "y": [target_class]}
    if extra_conditions:
        cond.update(extra_conditions)
    res = attribution(
        image, [cond], composite, mask_map=concept.mask,
        exclude_parallel=exclude_parallel,
    )
    return _heatmap_2d(res.heatmap[0])


def _per_concept_scores(
    attribution,
    composite,
    image: torch.Tensor,
    layer_name: str,
    concept,
    target_class: int,
    extra_conditions: Optional[dict] = None,
    exclude_parallel: bool = True,
) -> torch.Tensor:
    """Per-concept relevance under target class — no concept mask in
    the recording condition; record at the concept's tap and let
    ``concept.attribute`` reduce to scores. ``extra_conditions`` allows
    masking at deeper layers (cascade); cascade callers pass
    ``exclude_parallel=False`` so a single unified backward fires every
    ``MaskHook`` in topological order (avoids the
    ``partial_backward`` "tensor not in graph" failure when the chain
    crosses our custom autograd Functions)."""
    image.grad = None
    cond = {"y": [target_class]}
    if extra_conditions:
        cond.update(extra_conditions)
    res = attribution(
        image, [cond], composite,
        mask_map=concept.mask, record_layer=[layer_name],
        exclude_parallel=exclude_parallel,
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


def _layer_name_for(layer_idx: int, concept) -> str:
    return f"blocks.{layer_idx}.attn.{concept.tap_name}"


# ── HeadConcept atlas (one fig per layer; rows × heads sorted by id) ─────────


def plot_head_atlas(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    *,
    layers: Sequence[int],
    target_class: int,
    top_k: Optional[int] = None,
    cell_size: float = 1.6,
    figsize: Optional[tuple] = None,
    suptitle_prefix: Optional[str] = None,
) -> list[plt.Figure]:
    """One figure per layer in ``layers``. Each figure has one row, with
    ``num_heads`` (or ``top_k`` if set) cells, **sorted by ascending head
    id**. Cells are image+heatmap side-by-side panels.

    ``top_k=None`` shows every head; otherwise the top-``top_k`` heads by
    absolute relevance are kept and then reordered by id.
    """
    image_np = denormalize(image, model)
    figs: list[plt.Figure] = []
    concept = HeadConcept(model)

    for layer_idx in layers:
        num_heads = model.blocks[layer_idx].attn.num_heads
        layer_name = _layer_name_for(layer_idx, concept)

        scores = _per_concept_scores(
            attribution, composite, image, layer_name, concept, target_class
        )
        # Filter then re-sort by id.
        if top_k is None or top_k >= num_heads:
            head_ids = list(range(num_heads))
        else:
            head_ids = sorted(
                int(h) for h in torch.topk(scores.abs(), k=top_k).indices.tolist()
            )

        n_cols = len(head_ids)
        if figsize is None:
            fs = (cell_size * 2.0 * n_cols, cell_size + 0.6)
        else:
            fs = figsize
        fig, axes = plt.subplots(1, n_cols, figsize=fs, squeeze=False)
        for c_i, h in enumerate(head_ids):
            hm = _attribute_concept(
                attribution, composite, image, layer_name, concept, h, target_class
            )
            panel(axes[0, c_i], image_np, hm)
            axes[0, c_i].set_title(
                f"h{h}  r={scores[h].item():+.2g}", fontsize=8,
            )
        title = f"HeadConcept  •  layer {layer_idx}"
        if suptitle_prefix:
            title = f"{suptitle_prefix}  •  {title}"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        figs.append(fig)
    return figs


# ── KQVHeadConcept atlas (one fig per layer; 3 rows × heads, by id) ─────────


def plot_kqv_head_atlas(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    *,
    layers: Sequence[int],
    target_class: int,
    top_k: Optional[int] = None,
    cell_size: float = 1.4,
    figsize: Optional[tuple] = None,
    suptitle_prefix: Optional[str] = None,
) -> list[plt.Figure]:
    """One figure per layer. Each figure has **3 rows (K, Q, V) × heads**
    sorted by ascending head id, so the K/Q/V triple of every head is
    column-aligned for direct comparison.

    ``top_k=None`` shows every head; otherwise the top-``top_k`` heads
    (by ``|score|`` summed across K/Q/V) are kept and re-sorted by id.
    """
    image_np = denormalize(image, model)
    figs: list[plt.Figure] = []
    concept = KQVHeadConcept(model)

    for layer_idx in layers:
        num_heads = model.blocks[layer_idx].attn.num_heads
        layer_name = _layer_name_for(layer_idx, concept)

        scores = _per_concept_scores(
            attribution, composite, image, layer_name, concept, target_class
        )  # shape (3, num_heads)

        if top_k is None or top_k >= num_heads:
            head_ids = list(range(num_heads))
        else:
            head_score = scores.abs().sum(dim=0)  # rank by aggregate KQV magnitude
            head_ids = sorted(
                int(h) for h in torch.topk(head_score, k=top_k).indices.tolist()
            )

        n_cols = len(head_ids)
        if figsize is None:
            fs = (cell_size * 2.0 * n_cols, cell_size * 3 + 0.6)
        else:
            fs = figsize
        fig, axes = plt.subplots(3, n_cols, figsize=fs, squeeze=False)
        for r_i, part in enumerate(PARTS):
            for c_i, h in enumerate(head_ids):
                hm = _attribute_concept(
                    attribution, composite, image, layer_name, concept,
                    (part, h), target_class,
                )
                panel(axes[r_i, c_i], image_np, hm)
                axes[r_i, c_i].set_title(
                    f"{part}/h{h}  r={scores[r_i, h].item():+.2g}", fontsize=7,
                )
        title = f"KQVHeadConcept  •  layer {layer_idx}"
        if suptitle_prefix:
            title = f"{suptitle_prefix}  •  {title}"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        figs.append(fig)
    return figs


# ── HeadDim closeup (one fig per layer; one head, top-K dims by id) ─────────


def plot_head_dim_closeup(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    *,
    layers: Sequence[int],
    head_id: int,
    target_class: int,
    top_k: int = 8,
    cell_size: float = 1.6,
    figsize: Optional[tuple] = None,
    suptitle_prefix: Optional[str] = None,
) -> list[plt.Figure]:
    """One figure per layer. Each figure shows one row × ``top_k``
    head_dim concepts within the chosen ``head_id``, sorted by ascending
    dim id (after top-K filtering by |relevance|).
    """
    image_np = denormalize(image, model)
    figs: list[plt.Figure] = []
    concept = HeadDimConcept(model)

    for layer_idx in layers:
        head_dim = model.blocks[layer_idx].attn.head_dim
        layer_name = _layer_name_for(layer_idx, concept)

        scores = _per_concept_scores(
            attribution, composite, image, layer_name, concept, target_class
        )  # (num_heads, head_dim)
        head_scores = scores[head_id]  # (head_dim,)
        k = min(top_k, head_dim)
        top_dims = sorted(
            int(d) for d in torch.topk(head_scores.abs(), k=k).indices.tolist()
        )

        n_cols = len(top_dims)
        if figsize is None:
            fs = (cell_size * 2.0 * n_cols, cell_size + 0.6)
        else:
            fs = figsize
        fig, axes = plt.subplots(1, n_cols, figsize=fs, squeeze=False)
        for c_i, d in enumerate(top_dims):
            hm = _attribute_concept(
                attribution, composite, image, layer_name, concept,
                (head_id, d), target_class,
            )
            panel(axes[0, c_i], image_np, hm)
            axes[0, c_i].set_title(
                f"h{head_id}/d{d}  r={head_scores[d].item():+.2g}", fontsize=8,
            )
        title = f"HeadDimConcept (head {head_id})  •  layer {layer_idx}"
        if suptitle_prefix:
            title = f"{suptitle_prefix}  •  {title}"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        figs.append(fig)
    return figs


# ── KQVHeadDim closeup (one fig per layer; one head, 3 rows × top-K dims) ───


def plot_kqv_head_dim_closeup(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    *,
    layers: Sequence[int],
    head_id: int,
    target_class: int,
    top_k: int = 8,
    cell_size: float = 1.4,
    figsize: Optional[tuple] = None,
    suptitle_prefix: Optional[str] = None,
) -> list[plt.Figure]:
    """One figure per layer. **3 rows (K, Q, V) × top-``top_k`` dims** within
    the chosen ``head_id``. Top-K dims are picked by aggregate
    ``|score|`` summed across K/Q/V, then displayed in ascending dim id —
    so K[d]/Q[d]/V[d] are column-aligned for direct comparison.
    """
    image_np = denormalize(image, model)
    figs: list[plt.Figure] = []
    concept = KQVHeadDimConcept(model)

    for layer_idx in layers:
        head_dim = model.blocks[layer_idx].attn.head_dim
        layer_name = _layer_name_for(layer_idx, concept)

        scores = _per_concept_scores(
            attribution, composite, image, layer_name, concept, target_class
        )  # (3, num_heads, head_dim)
        head_scores = scores[:, head_id, :]  # (3, head_dim)
        k = min(top_k, head_dim)
        # Pick dims by aggregate |relevance| across K/Q/V.
        agg = head_scores.abs().sum(dim=0)  # (head_dim,)
        top_dims = sorted(
            int(d) for d in torch.topk(agg, k=k).indices.tolist()
        )

        n_cols = len(top_dims)
        if figsize is None:
            fs = (cell_size * 2.0 * n_cols, cell_size * 3 + 0.6)
        else:
            fs = figsize
        fig, axes = plt.subplots(3, n_cols, figsize=fs, squeeze=False)
        for r_i, part in enumerate(PARTS):
            for c_i, d in enumerate(top_dims):
                hm = _attribute_concept(
                    attribution, composite, image, layer_name, concept,
                    (part, head_id, d), target_class,
                )
                panel(axes[r_i, c_i], image_np, hm)
                axes[r_i, c_i].set_title(
                    f"{part}/h{head_id}/d{d}  r={head_scores[r_i, d].item():+.2g}",
                    fontsize=7,
                )
        title = f"KQVHeadDimConcept (head {head_id})  •  layer {layer_idx}"
        if suptitle_prefix:
            title = f"{suptitle_prefix}  •  {title}"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        figs.append(fig)
    return figs


# ── conditional layer cascade (backward CRP through a layer list) ───────────


def plot_conditional_cascade(
    image: torch.Tensor,
    model: torch.nn.Module,
    attribution,
    composite,
    fv,
    *,
    layers: Sequence[int],
    target_class: int,
    top_k_per_layer: int = 4,
    n_refs: int = 4,
    cell_size: float = 1.4,
    figsize: Optional[tuple] = None,
    suptitle: Optional[str] = None,
) -> Tuple[plt.Figure, dict]:
    """Backward conditional CRP cascade.

    For each layer in ``layers`` (taken **deepest first**):

    1. Compute ``HeadConcept`` relevance at this layer, conditioned on the
       union of heads selected at deeper layers (``y=target_class`` plus
       ``{deeper_layer_taps: selected_heads}``).
    2. Pick the top-``top_k_per_layer`` heads by absolute relevance and
       sort them by ascending head id.
    3. For each picked head, fetch the top-``n_refs`` reference samples
       from ``fv`` (which must index ``HeadConcept`` at every layer in
       ``layers``).

    Layout: rows = layers (late → early), each row contains
    ``top_k_per_layer`` head-blocks side-by-side; each head-block holds
    ``1`` panel for the target image (with this head's heatmap) plus
    ``n_refs`` panels for the reference samples.

    Returns ``(figure, selected_heads_per_layer)`` — the dict maps
    ``layer_idx → [head_ids]`` so the caller can re-use the selection.
    """
    head_concept = HeadConcept(model)
    target_np = denormalize(image, model)

    # Run the cascade and collect display data per layer.
    selected: dict[int, list[int]] = {}
    accumulated: dict = {}  # extra conditions to fold in
    layer_rows: list[tuple[int, list]] = []

    for layer_idx in layers:
        layer_name = _layer_name_for(layer_idx, head_concept)
        # exclude_parallel=False once any extra condition is present —
        # see the docstrings on _per_concept_scores / _attribute_concept.
        # The first cascade step (no extras) keeps the cheaper default.
        exclude_parallel = not bool(accumulated)
        scores = _per_concept_scores(
            attribution, composite, image, layer_name, head_concept,
            target_class, extra_conditions=(accumulated or None),
            exclude_parallel=exclude_parallel,
        )
        k = min(top_k_per_layer, scores.numel())
        top_heads = sorted(
            int(h) for h in torch.topk(scores.abs(), k=k).indices.tolist()
        )
        selected[layer_idx] = top_heads

        # Fetch FV references for each picked head.
        head_blocks = []
        for h in top_heads:
            ref_dict = fv.get_max_reference(
                [h], layer_name, mode="relevance", r_range=(0, n_refs),
                composite=composite, plot_fn=None,
            )
            samples, ref_heatmaps = ref_dict[h]
            head_blocks.append((h, scores[h].item(), samples, ref_heatmaps))
        layer_rows.append((layer_idx, head_blocks))

        # Accumulate this layer's selection into the conditioning of the
        # next (shallower) layer.
        accumulated = dict(accumulated)
        accumulated[layer_name] = top_heads

    # Layout: rows = layers; each row has top_k_per_layer × (1 + n_refs)
    # panels. Each panel is a side-by-side image+heatmap.
    panels_per_head = 1 + n_refs
    n_cols = top_k_per_layer * panels_per_head
    n_rows = len(layer_rows)
    if figsize is None:
        figsize = (cell_size * 2.0 * n_cols, cell_size * n_rows + 0.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for row_i, (layer_idx, head_blocks) in enumerate(layer_rows):
        # Pad if fewer heads picked than top_k_per_layer (rare, only if
        # num_heads < top_k_per_layer at this layer).
        for h_i in range(top_k_per_layer):
            base_col = h_i * panels_per_head
            if h_i >= len(head_blocks):
                for col_offset in range(panels_per_head):
                    axes[row_i, base_col + col_offset].axis("off")
                continue
            h, score, samples, ref_heatmaps = head_blocks[h_i]
            # Target image with this head's conditional heatmap, masked
            # also at every deeper cascade layer (full-history option).
            deeper_extras = {
                _layer_name_for(deeper, head_concept): selected[deeper]
                for deeper in layers if deeper != layer_idx
                and layers.index(deeper) < layers.index(layer_idx)
            }
            target_hm = _attribute_concept(
                attribution, composite, image,
                _layer_name_for(layer_idx, head_concept),
                head_concept, h, target_class,
                extra_conditions=(deeper_extras or None),
                exclude_parallel=not bool(deeper_extras),
            )
            panel(axes[row_i, base_col], target_np, target_hm)
            axes[row_i, base_col].set_title(
                f"L{layer_idx} h{h}\nr={score:+.2g}", fontsize=7,
            )
            for ref_i in range(n_refs):
                ax = axes[row_i, base_col + 1 + ref_i]
                if ref_i >= len(samples):
                    ax.axis("off")
                    continue
                ref_img = samples[ref_i].cpu().numpy().transpose(1, 2, 0)
                ref_img = (ref_img - ref_img.min()) / max(
                    ref_img.max() - ref_img.min(), 1e-9
                )
                hm = ref_heatmaps[ref_i].cpu().numpy()
                panel(ax, ref_img, hm)
                ax.set_title(f"ref #{ref_i + 1}", fontsize=7)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig, selected


# ── reference samples per granularity ────────────────────────────────────────


def plot_reference_samples(
    fv,
    *,
    concept_name: str,
    layer_idx: int,
    target_image: torch.Tensor,
    target_class: int,
    model: torch.nn.Module,
    attribution,
    composite,
    n_top_concepts: int = 4,
    n_refs_per_concept: int = 4,
    cell_size: float = 1.5,
    figsize: Optional[tuple] = None,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """For the top-``n_top_concepts`` concepts of ``concept_name`` on the
    target image (sorted by ascending concept id after top-K filtering),
    fetch the top-``n_refs_per_concept`` reference samples that maximise
    each concept's relevance and render them with the concept's
    conditional heatmap overlaid.

    Rows: top concepts (by id). Cols: target panel (concept on target
    image) + ``n_refs_per_concept`` reference panels.
    """
    cls = CONCEPT_CLASSES[concept_name]
    concept = cls(model)
    layer_name = _layer_name_for(layer_idx, concept)
    num_heads = model.blocks[layer_idx].attn.num_heads
    head_dim = model.blocks[layer_idx].attn.head_dim
    target_np = denormalize(target_image, model)

    scores = _per_concept_scores(
        attribution, composite, target_image, layer_name, concept, target_class
    )
    flat = scores.flatten()
    k = min(n_top_concepts, flat.numel())
    top_idx_by_rel = torch.topk(flat.abs(), k=k).indices.tolist()
    flat_top = sorted(int(i) for i in top_idx_by_rel)  # sorted by flat id

    n_cols = n_refs_per_concept + 1  # +1 for the target panel
    n_rows = len(flat_top)
    if figsize is None:
        figsize = (cell_size * 2.0 * n_cols, cell_size * n_rows + 0.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    all_ids = _enumerate_ids(concept_name, num_heads, head_dim)
    for row_i, flat_idx in enumerate(flat_top):
        cid = all_ids[flat_idx]
        ref_dict = fv.get_max_reference(
            [flat_idx], layer_name, mode="relevance",
            r_range=(0, n_refs_per_concept),
            composite=composite, plot_fn=None,
        )
        samples, ref_heatmaps = ref_dict[flat_idx]
        # Target column.
        target_hm = _attribute_concept(
            attribution, composite, target_image, layer_name, concept,
            cid, target_class,
        )
        panel(axes[row_i, 0], target_np, target_hm)
        score = flat[flat_idx].item()
        axes[row_i, 0].set_title(
            f"target  •  {_label_id(concept_name, cid)}\nr={score:+.2g}",
            fontsize=8,
        )
        # Reference columns.
        for c_i in range(n_refs_per_concept):
            ax = axes[row_i, c_i + 1]
            if c_i >= len(samples):
                ax.axis("off")
                continue
            ref_img = samples[c_i].cpu().numpy().transpose(1, 2, 0)
            ref_img = (ref_img - ref_img.min()) / max(
                ref_img.max() - ref_img.min(), 1e-9
            )
            ref_hm = ref_heatmaps[c_i].cpu().numpy()
            panel(ax, ref_img, ref_hm)
            ax.set_title(f"ref #{c_i + 1}", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


__all__ = [
    "CONCEPT_CLASSES",
    "denormalize",
    "panel",
    "plot_head_atlas",
    "plot_kqv_head_atlas",
    "plot_head_dim_closeup",
    "plot_kqv_head_dim_closeup",
    "plot_conditional_cascade",
    "plot_reference_samples",
]
