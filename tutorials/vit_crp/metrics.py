"""Quantitative comparison of attention-concept granularities.

Metrics
-------

* **Faithfulness — deletion / insertion AUC** (Petsiuk et al., BMVC 2018). The
  heatmap-ranked top-k% of input patches are progressively masked (deletion)
  or revealed from a blurred baseline (insertion); the area under the
  resulting target-class-probability curve is reported. Lower deletion AUC
  and higher insertion AUC indicate a more faithful heatmap.

* **Random-concept baseline.** For each concept definition, the k true top-k
  concepts' union heatmap is compared against a same-cardinality random
  sample of concept ids. If the structure is meaningful, the true top-k
  heatmap should achieve materially better deletion / insertion AUC than the
  random one.

Usage
-----

::

    uv run python tutorials/vit_crp/metrics.py \\
        --image-dir path/to/images --target-class 281 --block 6 \\
        --top-k 8 --out results.csv

Each row of ``results.csv`` is one ``(image, concept_def, mode)`` triple,
where ``mode ∈ {true, random}``.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from crp.attention_concepts import (
    HeadConcept,
    KQVConcept,
    KQVHeadConcept,
    HeadDimConcept,
    PARTS,
)
from crp.attribution import CondAttribution
from crp.transformer_patches import AttnLRPEpsilonComposite


CONCEPT_DEFS = {
    "head": HeadConcept,
    "kqv": KQVConcept,
    "kqv_head": KQVHeadConcept,
    "head_dim": HeadDimConcept,
}


def _enumerate_ids(name: str, num_heads: int, head_dim: int) -> list:
    if name == "head":
        return list(range(num_heads))
    if name == "kqv":
        return list(PARTS)
    if name == "kqv_head":
        return [(p, h) for p in PARTS for h in range(num_heads)]
    if name == "head_dim":
        return [
            (p, h, d) for p in PARTS for h in range(num_heads) for d in range(head_dim)
        ]
    raise ValueError(name)


# ── core attribution helpers ──────────────────────────────────────────────────


def per_concept_scores(
    attribution: CondAttribution,
    concept,
    layer_name: str,
    data: torch.Tensor,
    target_class: int,
    composite,
) -> torch.Tensor:
    conditions = [{"y": [target_class]}]
    result = attribution(
        data, conditions, composite, mask_map=concept.mask, record_layer=[layer_name]
    )
    rel = result.relevances[layer_name]
    return concept.attribute(rel, layer_name=layer_name, abs_norm=False)[0]


def union_heatmap(
    attribution: CondAttribution,
    concept,
    layer_name: str,
    cid_list: Sequence,
    data: torch.Tensor,
    target_class: int,
    composite,
) -> np.ndarray:
    """Single backward pass with the union of all ``cid_list`` masks."""
    conditions = [{layer_name: list(cid_list), "y": [target_class]}]
    result = attribution(data, conditions, composite, mask_map=concept.mask)
    heatmap = result.heatmap[0]
    if heatmap.dim() == 3:
        heatmap = heatmap.sum(dim=0)
    return heatmap.detach().cpu().numpy()


# ── deletion / insertion ──────────────────────────────────────────────────────


def _patch_grid_relevance(heatmap: np.ndarray, patch_size: int) -> np.ndarray:
    """Pool the heatmap to ViT-patch-resolution by absolute-value sum.

    ViT-base operates on 14×14 patches of a 224×224 image. Faithfulness is
    most informative at this granularity rather than per-pixel.
    """
    H, W = heatmap.shape
    Ph, Pw = H // patch_size, W // patch_size
    grid = np.abs(heatmap[: Ph * patch_size, : Pw * patch_size]).reshape(
        Ph, patch_size, Pw, patch_size
    ).sum(axis=(1, 3))
    return grid


def _apply_patch_mask(
    image: torch.Tensor, mask_grid: np.ndarray, baseline: torch.Tensor
) -> torch.Tensor:
    """Replace patches where ``mask_grid == 1`` with the baseline image."""
    Ph, Pw = mask_grid.shape
    H, W = image.shape[-2:]
    patch_size = H // Ph
    mask_full = (
        torch.from_numpy(mask_grid)
        .repeat_interleave(patch_size, 0)
        .repeat_interleave(patch_size, 1)
        .to(image.device, dtype=image.dtype)
    )  # (H, W)
    mask_full = mask_full.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return image * (1 - mask_full) + baseline * mask_full


def deletion_curve(
    model: torch.nn.Module,
    image: torch.Tensor,
    heatmap: np.ndarray,
    target_class: int,
    patch_size: int = 16,
    n_steps: int = 14,
) -> np.ndarray:
    """Probability-of-target-class as patches are progressively masked
    (highest-relevance first) with a black baseline.
    """
    grid = _patch_grid_relevance(heatmap, patch_size)
    Ph, Pw = grid.shape
    order = np.argsort(-grid.flatten())  # most-relevant patches first
    n_patches = Ph * Pw
    step_size = max(1, n_patches // n_steps)
    baseline = torch.zeros_like(image)

    probs = []
    mask = np.zeros(n_patches, dtype=np.float32)
    with torch.no_grad():
        # initial probability (full image)
        prob0 = F.softmax(model(image)[0], dim=-1)[target_class].item()
        probs.append(prob0)
        for step in range(1, n_steps + 1):
            mask[order[: step * step_size]] = 1
            mask_grid = mask.reshape(Ph, Pw)
            perturbed = _apply_patch_mask(image, mask_grid, baseline)
            p = F.softmax(model(perturbed)[0], dim=-1)[target_class].item()
            probs.append(p)
    return np.asarray(probs, dtype=np.float64)


def insertion_curve(
    model: torch.nn.Module,
    image: torch.Tensor,
    heatmap: np.ndarray,
    target_class: int,
    patch_size: int = 16,
    n_steps: int = 14,
    blur_sigma: float = 5.0,
) -> np.ndarray:
    """Probability-of-target-class as patches are progressively revealed
    (highest-relevance first) from a blurred baseline.
    """
    grid = _patch_grid_relevance(heatmap, patch_size)
    Ph, Pw = grid.shape
    order = np.argsort(-grid.flatten())
    n_patches = Ph * Pw
    step_size = max(1, n_patches // n_steps)

    # blurred baseline via simple Gaussian on each channel
    baseline = _gaussian_blur(image, sigma=blur_sigma)

    probs = []
    mask = np.zeros(n_patches, dtype=np.float32)
    with torch.no_grad():
        prob0 = F.softmax(model(baseline)[0], dim=-1)[target_class].item()
        probs.append(prob0)
        for step in range(1, n_steps + 1):
            mask[order[: step * step_size]] = 1
            mask_grid = mask.reshape(Ph, Pw)
            # invert: reveal patches where mask_grid == 1
            revealed = _apply_patch_mask(baseline, mask_grid, image)
            p = F.softmax(model(revealed)[0], dim=-1)[target_class].item()
            probs.append(p)
    return np.asarray(probs, dtype=np.float64)


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Cheap separable Gaussian blur via conv2d. Channel-wise."""
    if sigma <= 0:
        return image.clone()
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    C = image.shape[1]
    kx = kernel_1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = kernel_1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    blurred = F.conv2d(image, kx, padding=(0, radius), groups=C)
    blurred = F.conv2d(blurred, ky, padding=(radius, 0), groups=C)
    return blurred


def auc(curve: np.ndarray) -> float:
    """Trapezoidal AUC on a unit-x-axis."""
    n = len(curve) - 1
    if n <= 0:
        return float(curve[0])
    return float(np.trapezoid(curve, dx=1.0 / n))


# ── runner ────────────────────────────────────────────────────────────────────


def load_image(path: Path, model: torch.nn.Module) -> torch.Tensor:
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    tensor.requires_grad_(True)
    return tensor


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image-dir", required=True, help="directory of input images")
    p.add_argument("--target-class", type=int, default=281)
    p.add_argument("--block", type=int, default=6)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument("--steps", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    rng = random.Random(args.seed)

    print(f"loading {args.model}")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)

    block = model.blocks[args.block].attn
    num_heads, head_dim = block.num_heads, block.head_dim
    layer_name = f"blocks.{args.block}.attn.qkv_tap"
    print(f"layer={layer_name}  num_heads={num_heads}  head_dim={head_dim}")

    # Composite bundles TimmViTCanonizer (qkv_tap injection + AttnLRP forward
    # swaps) — applied scoped on composite.context(), reverted on exit.
    composite = AttnLRPEpsilonComposite()
    attribution = CondAttribution(model, device=torch.device(device))

    image_paths = sorted(
        p
        for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not image_paths:
        raise SystemExit(f"no images found in {args.image_dir}")

    rows = []
    for img_path in image_paths:
        print(f"\n=== {img_path.name} ===")
        data = load_image(img_path, model).to(device)

        for name, cls in CONCEPT_DEFS.items():
            concept = cls()
            concept.register_from_model(model)

            scores = per_concept_scores(
                attribution, concept, layer_name, data, args.target_class, composite
            )
            all_ids = _enumerate_ids(name, num_heads, head_dim)
            k = min(args.top_k, len(all_ids))
            flat_top = torch.topk(scores.flatten().abs(), k=k).indices.tolist()
            true_top_k = [all_ids[i] for i in flat_top]
            random_k = rng.sample(all_ids, k)

            for mode, ids in (("true", true_top_k), ("random", random_k)):
                hm = union_heatmap(
                    attribution, concept, layer_name, ids, data, args.target_class, composite
                )
                d_curve = deletion_curve(
                    model, data, hm, args.target_class, n_steps=args.steps
                )
                i_curve = insertion_curve(
                    model, data, hm, args.target_class, n_steps=args.steps
                )
                rows.append({
                    "image": img_path.name,
                    "concept_def": name,
                    "mode": mode,
                    "k": k,
                    "deletion_auc": auc(d_curve),
                    "insertion_auc": auc(i_curve),
                    "p_initial": float(d_curve[0]),
                    "p_final_deletion": float(d_curve[-1]),
                    "p_final_insertion": float(i_curve[-1]),
                })
                print(
                    f"  {name:>9} {mode:6} del-AUC={rows[-1]['deletion_auc']:.4f} "
                    f"ins-AUC={rows[-1]['insertion_auc']:.4f}"
                )

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
