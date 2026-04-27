"""Vision-transformer CRP demo.

Compares the four attention-concept granularities on a single image:

* ``HeadConcept``    — one concept per head
* ``KQVConcept``     — three concepts per block (whole Q / K / V)
* ``KQVHeadConcept`` — 3 × num_heads concepts (per (part, head))
* ``HeadDimConcept`` — 3 × num_heads × head_dim concepts

For each granularity, the script computes per-concept relevance scores at a
chosen attention block under the target class, picks the top-k concepts,
runs a conditional attribution per concept to obtain a pixel-space heatmap,
and renders a comparison grid.

Usage::

    uv run python tutorials/vit_crp/demo.py \\
        --image path/to/image.jpg --target-class 281 --block 6 --out out.png

Defaults to ``timm.vit_base_patch16_224`` (pretrained) and target class 281
(tabby cat). Requires the ``vit`` and ``dev`` extras of this package.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import timm
from timm.data import resolve_data_config, create_transform
from zennit.composites import EpsilonPlusFlat

from crp.attention_concepts import (
    HeadConcept,
    KQVConcept,
    KQVHeadConcept,
    HeadDimConcept,
    PARTS,
)
from crp.attribution import CondAttribution
from crp.image import imgify
from crp.transformer_patches import prepare_timm_vit


# ── concept registry for the demo ─────────────────────────────────────────────


CONCEPT_DEFS = {
    "head": HeadConcept,
    "kqv": KQVConcept,
    "kqv_head": KQVHeadConcept,
    "head_dim": HeadDimConcept,
}


def _id_label(name: str, cid) -> str:
    """Pretty-print a concept id for figure labels."""
    if name == "head":
        return f"head={int(cid)}"
    if name == "kqv":
        return f"part={cid}"
    if name == "kqv_head":
        part, h = cid
        return f"{part}/h{int(h)}"
    if name == "head_dim":
        part, h, d = cid
        return f"{part}/h{int(h)}/d{int(d)}"
    raise ValueError(name)


def _enumerate_ids(name: str, num_heads: int, head_dim: int) -> list:
    """All concept ids for this granularity and ViT geometry."""
    if name == "head":
        return list(range(num_heads))
    if name == "kqv":
        return list(PARTS)
    if name == "kqv_head":
        return [(p, h) for p in PARTS for h in range(num_heads)]
    if name == "head_dim":
        return [
            (p, h, d)
            for p in PARTS
            for h in range(num_heads)
            for d in range(head_dim)
        ]
    raise ValueError(name)


# ── pipeline ──────────────────────────────────────────────────────────────────


def load_image(path: str, model: torch.nn.Module) -> torch.Tensor:
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    tensor.requires_grad_(True)
    return tensor


def per_concept_scores(
    attribution: CondAttribution,
    concept,
    layer_name: str,
    data: torch.Tensor,
    target_class: int,
    composite,
) -> torch.Tensor:
    """Run a single backward pass under the target class with NO mask, record
    relevance at the qkv_tap layer, aggregate via the concept's ``attribute``.
    Returns scores of shape ``concept.attribute(...)``-shape (batch dim 1).
    """
    conditions = [{"y": [target_class]}]
    result = attribution(
        data,
        conditions,
        composite,
        mask_map=concept.mask,           # no mask is applied if no condition keys match
        record_layer=[layer_name],
    )
    rel = result.relevances[layer_name]
    return concept.attribute(rel, layer_name=layer_name, abs_norm=False)[0]


def top_k_ids(scores: torch.Tensor, ids: Sequence, k: int) -> list:
    flat = scores.flatten()
    k = min(k, flat.numel())
    top = torch.topk(flat.abs(), k=k).indices.tolist()
    # Map back to id list — `ids` is in the same row-major order as `flat`.
    return [ids[i] for i in top]


def conditional_heatmap(
    attribution: CondAttribution,
    concept,
    layer_name: str,
    concept_id,
    data: torch.Tensor,
    target_class: int,
    composite,
) -> np.ndarray:
    conditions = [{layer_name: [concept_id], "y": [target_class]}]
    result = attribution(
        data, conditions, composite, mask_map=concept.mask
    )
    heatmap = result.heatmap[0]  # (3, H, W) or (H, W) depending on the model
    if heatmap.dim() == 3:
        heatmap = heatmap.sum(dim=0)
    return heatmap.detach().cpu().numpy()


def make_figure(
    image: torch.Tensor,
    rows: dict,
    out_path: Path,
    layer_name: str,
    target_class: int,
):
    """rows: {concept_def_name: [(label, heatmap_np), ...]}"""
    n_cols = max(len(items) for items in rows.values()) + 1  # +1 for input image col
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows + 0.5))
    if n_rows == 1:
        axes = np.array([axes])

    img_pil = imgify(image[0].detach().cpu(), denormalize=True)

    for r, (name, items) in enumerate(rows.items()):
        axes[r, 0].imshow(img_pil)
        axes[r, 0].set_title(name, fontsize=10)
        axes[r, 0].axis("off")
        for c, (label, hm) in enumerate(items, start=1):
            axes[r, c].imshow(img_pil, alpha=0.4)
            axes[r, c].imshow(hm, cmap="bwr", alpha=0.6, vmin=-np.abs(hm).max(), vmax=np.abs(hm).max())
            axes[r, c].set_title(label, fontsize=9)
            axes[r, c].axis("off")
        for c in range(len(items) + 1, n_cols):
            axes[r, c].axis("off")

    fig.suptitle(f"layer={layer_name}  target_class={target_class}", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="path to input image")
    p.add_argument("--target-class", type=int, default=281, help="ImageNet class index (default 281, tabby cat)")
    p.add_argument("--block", type=int, default=6, help="ViT block index (default 6, mid-network)")
    p.add_argument("--top-k", type=int, default=4, help="top-K concepts per granularity (default 4)")
    p.add_argument("--model", default="vit_base_patch16_224", help="timm model name")
    p.add_argument(
        "--concepts",
        nargs="+",
        default=list(CONCEPT_DEFS.keys()),
        choices=list(CONCEPT_DEFS.keys()),
        help="which concept definitions to include",
    )
    p.add_argument("--out", default="figures/comparison.png", help="output figure path")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.set_grad_enabled(True)

    print(f"loading {args.model} (pretrained=True) on {device}")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    prepare_timm_vit(model)

    print(f"loading image: {args.image}")
    data = load_image(args.image, model).to(device)

    # Sanity: model prediction
    with torch.no_grad():
        pred = model(data)[0].softmax(dim=-1)
    top5 = pred.topk(5)
    print("top-5 predictions:")
    for prob, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        marker = " ← target" if idx == args.target_class else ""
        print(f"  cls={idx:4d} p={prob:.3f}{marker}")

    # Resolve geometry from a representative attention block
    block = model.blocks[args.block].attn
    num_heads, head_dim = block.num_heads, block.head_dim
    layer_name = f"blocks.{args.block}.attn.qkv_tap"
    print(f"layer={layer_name}  num_heads={num_heads}  head_dim={head_dim}")

    composite = EpsilonPlusFlat()
    attribution = CondAttribution(model, device=torch.device(device))

    rows = {}
    for name in args.concepts:
        concept_cls = CONCEPT_DEFS[name]
        concept = concept_cls()
        concept.register_from_model(model)

        scores = per_concept_scores(
            attribution, concept, layer_name, data, args.target_class, composite
        )
        all_ids = _enumerate_ids(name, num_heads, head_dim)
        top_ids = top_k_ids(scores, all_ids, args.top_k)
        print(f"[{name}] top-{args.top_k} ids: {top_ids}")

        items = []
        for cid in top_ids:
            hm = conditional_heatmap(
                attribution, concept, layer_name, cid, data, args.target_class, composite
            )
            items.append((_id_label(name, cid), hm))
        rows[name] = items

    out_path = Path(args.out)
    make_figure(data, rows, out_path, layer_name, args.target_class)


if __name__ == "__main__":
    main()
