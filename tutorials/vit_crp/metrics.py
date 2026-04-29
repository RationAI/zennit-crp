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

The four supported concepts each pick their own hook tap: ``HeadConcept`` /
``HeadDimConcept`` read at the per-head output tokens (``attn_out_tap``);
``KQVHeadConcept`` / ``KQVHeadDimConcept`` read at the K/Q/V projections
(``qkv_tap``). The caller passes a block index and the layer name is
resolved per-concept as ``blocks.{block}.attn.{concept.tap_name}``.

Usage
-----

Flat directory (single target class for every image)::

    uv run python tutorials/vit_crp/metrics.py \\
        --image-dir path/to/images --target-class 281 --block 6 \\
        --composite gamma --gamma 0.25 --top-k 8 --out results.csv

Class-keyed directory tree (subdir name = ImageNet-1k class index, target
class is auto-resolved per image)::

    uv run python tutorials/vit_crp/metrics.py \\
        --image-dir path/to/curated --block 6 \\
        --composite gamma --gamma 0.25 --top-k 8 --out results.csv

Each row of ``results.csv`` is one ``(image, target_class, concept_def,
composite, gamma, mode)`` tuple, where ``mode ∈ {true, random}``.
"""

from __future__ import annotations

import argparse
import csv
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
    HeadDimConcept,
    KQVHeadConcept,
    KQVHeadDimConcept,
    PARTS,
)
from crp.attribution import CondAttribution
from crp.transformer_patches import (
    AttnLRPEpsilonComposite,
    AttnLRPGammaComposite,
)


CONCEPT_DEFS = {
    "head":         HeadConcept,
    "head_dim":     HeadDimConcept,
    "kqv_head":     KQVHeadConcept,
    "kqv_head_dim": KQVHeadDimConcept,
}


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


def _layer_name(block_idx: int, concept) -> str:
    """Resolve the recording layer name for a concept on a given block.

    Each concept knows which tap it reads from (``concept.tap_name`` —
    either ``qkv_tap`` for K/Q/V-side concepts or ``attn_out_tap`` for the
    output-side ones), so the caller passes the block index and the layer
    name follows.
    """
    return f"blocks.{block_idx}.attn.{concept.tap_name}"


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
    # γ-LRP relevance can produce |R| values that overflow float32 when summed
    # over a patch; cast to float64 for the reduction.
    cropped = heatmap[: Ph * patch_size, : Pw * patch_size].astype(np.float64)
    grid = np.abs(cropped).reshape(Ph, patch_size, Pw, patch_size).sum(axis=(1, 3))
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


# ── image / class iteration ───────────────────────────────────────────────────


def iter_image_classes(
    image_dir: Path, target_class: int | None = None
) -> list[tuple[Path, int]]:
    """Build a list of ``(image_path, target_class)`` pairs from ``image_dir``.

    Two layouts:

    * **Class-keyed**: ``image_dir/<int>/<image>``. Auto-detected when every
      subdirectory of ``image_dir`` is named with an integer class index;
      ``target_class`` arg is ignored. Yields one (path, int(subdir)) per
      image.
    * **Flat**: every image under ``image_dir`` (recursively). Requires
      ``target_class``.
    """
    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".JPEG")

    subdirs = [p for p in image_dir.iterdir() if p.is_dir()]
    int_named = subdirs and all(p.name.isdigit() for p in subdirs)

    pairs: list[tuple[Path, int]] = []
    if int_named:
        for sd in sorted(subdirs, key=lambda p: int(p.name)):
            cls = int(sd.name)
            for p in sorted(sd.iterdir()):
                if p.suffix in image_exts:
                    pairs.append((p, cls))
    else:
        if target_class is None:
            raise SystemExit(
                f"{image_dir} is a flat directory; pass --target-class."
            )
        for p in sorted(image_dir.rglob("*")):
            if p.suffix in image_exts and p.is_file():
                pairs.append((p, target_class))

    if not pairs:
        raise SystemExit(f"no images found in {image_dir}")
    return pairs


# ── composite construction ────────────────────────────────────────────────────


def build_composite(name: str, gamma: float, epsilon: float):
    """Map a CLI string + γ to an AttnLRP composite.

    ``name='epsilon'``        → :class:`AttnLRPEpsilonComposite` (γ ignored).
    ``name='gamma'``          → :class:`AttnLRPGammaComposite` (γ argument used).
    """
    if name == "epsilon":
        return AttnLRPEpsilonComposite(epsilon=epsilon)
    if name == "gamma":
        return AttnLRPGammaComposite(gamma=gamma, epsilon=epsilon)
    raise ValueError(f"unknown composite {name!r}; expected 'epsilon' or 'gamma'")


# ── reusable evaluator ────────────────────────────────────────────────────────


def resolve_top_k(
    name: str, num_heads: int, head_dim: int, top_k: int | dict[str, int]
) -> int:
    """Resolve top-k for ``name`` from a flat int or a per-granularity map.

    A flat int is clamped to the granularity's concept count.

    For Petsiuk-style true-vs-random union heatmaps to discriminate, ``k``
    must be << ``num_concepts`` — otherwise the random union covers most of
    the same concepts as the true union and the gap collapses to noise. The
    canonical defaults below pick ~⅓ of concepts at the coarse granularities
    and stay sparse at the fine ones:

      ``head`` (12)              → 4
      ``head_dim`` (12·64=768)   → 8
      ``kqv_head`` (3·12=36)     → 8
      ``kqv_head_dim`` (2304)    → 8
    """
    if isinstance(top_k, dict):
        if name not in top_k:
            raise KeyError(f"top_k dict missing entry for granularity {name!r}")
        k = int(top_k[name])
    else:
        k = int(top_k)
    n = len(_enumerate_ids(name, num_heads, head_dim))
    return max(1, min(k, n))


PER_GRANULARITY_TOP_K: dict[str, int] = {
    "head": 4,
    "head_dim": 8,
    "kqv_head": 8,
    "kqv_head_dim": 8,
}


def run_one_config(
    model: torch.nn.Module,
    attribution: CondAttribution,
    composite,
    composite_label: str,
    gamma_label: float | None,
    image_class_pairs: list[tuple[Path, int]],
    block_idx: int,
    num_heads: int,
    head_dim: int,
    top_k: int | dict[str, int],
    n_steps: int,
    rng: random.Random,
    device: str,
) -> list[dict]:
    """Run faithfulness benchmark for one (composite, γ) over all images.

    Returns a flat list of result rows, one per
    ``(image, granularity, mode)`` triple. ``top_k`` may be a single int
    (applied to every granularity, clamped) or a per-granularity dict.

    Each concept's recording layer is resolved from ``block_idx`` and the
    concept's own ``tap_name`` (``qkv_tap`` for K/Q/V-side concepts,
    ``attn_out_tap`` for output-side concepts).
    """
    rows: list[dict] = []
    for img_path, target_class in image_class_pairs:
        print(f"  [{composite_label} γ={gamma_label}] {img_path.name} (cls={target_class})")
        data = load_image(img_path, model).to(device)

        for name, cls in CONCEPT_DEFS.items():
            concept = cls(model)
            layer_name = _layer_name(block_idx, concept)

            scores = per_concept_scores(
                attribution, concept, layer_name, data, target_class, composite
            )
            all_ids = _enumerate_ids(name, num_heads, head_dim)
            k = resolve_top_k(name, num_heads, head_dim, top_k)
            flat_top = torch.topk(scores.flatten().abs(), k=k).indices.tolist()
            true_top_k = [all_ids[i] for i in flat_top]
            random_k = rng.sample(all_ids, k)

            for mode, ids in (("true", true_top_k), ("random", random_k)):
                hm = union_heatmap(
                    attribution, concept, layer_name, ids, data, target_class, composite
                )
                d_curve = deletion_curve(
                    model, data, hm, target_class, n_steps=n_steps
                )
                i_curve = insertion_curve(
                    model, data, hm, target_class, n_steps=n_steps
                )
                rows.append({
                    "composite": composite_label,
                    "gamma": gamma_label,
                    "image": img_path.name,
                    "target_class": target_class,
                    "concept_def": name,
                    "mode": mode,
                    "k": k,
                    "deletion_auc": auc(d_curve),
                    "insertion_auc": auc(i_curve),
                    "p_initial": float(d_curve[0]),
                    "p_final_deletion": float(d_curve[-1]),
                    "p_final_insertion": float(i_curve[-1]),
                })
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image-dir", required=True, help="directory of input images (flat or class-keyed)")
    p.add_argument("--target-class", type=int, default=None,
                   help="ImageNet class index for a flat image dir (ignored for class-keyed)")
    p.add_argument("--block", type=int, default=6)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument("--steps", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--composite", choices=("epsilon", "gamma"), default="gamma",
                   help="LRP composite (default: gamma — AttnLRP §3.2.1)")
    p.add_argument("--gamma", type=float, default=0.25,
                   help="γ for AttnLRPGammaComposite (ignored when --composite=epsilon)")
    p.add_argument("--epsilon", type=float, default=1e-6)
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    rng = random.Random(args.seed)

    print(f"loading {args.model}")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)

    block = model.blocks[args.block].attn
    num_heads, head_dim = block.num_heads, block.head_dim
    print(f"block={args.block}  num_heads={num_heads}  head_dim={head_dim}")

    composite = build_composite(args.composite, args.gamma, args.epsilon)
    composite_label = type(composite).__name__
    gamma_label = args.gamma if args.composite == "gamma" else None
    print(f"composite={composite_label}  gamma={gamma_label}")
    attribution = CondAttribution(model, device=torch.device(device))

    image_class_pairs = iter_image_classes(Path(args.image_dir), args.target_class)
    print(f"images: {len(image_class_pairs)} pair(s); "
          f"{len(set(c for _, c in image_class_pairs))} class(es)")

    rows = run_one_config(
        model=model,
        attribution=attribution,
        composite=composite,
        composite_label=composite_label,
        gamma_label=gamma_label,
        image_class_pairs=image_class_pairs,
        block_idx=args.block,
        num_heads=num_heads,
        head_dim=head_dim,
        top_k=args.top_k,
        n_steps=args.steps,
        rng=rng,
        device=device,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
