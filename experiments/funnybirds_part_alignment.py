"""FunnyBirds part-alignment evaluation: does CRP attribution localise on
the *correct* semantic parts?

FunnyBirds is unique among XAI testbeds because each image ships with a
ground-truth per-pixel part map (eye01, eye02, beak, foot01, foot02,
wing01, wing02, tail) on a black background. For an attribution method
to be considered faithful, its heatmap should concentrate on the
semantic part(s) the model uses to make its decision — not on the
background, not on irrelevant parts.

This script computes, per concept and per attention layer, the fraction
of total |R| that lands on (a) the union of named bird parts vs (b) the
black background, vs (c) part-by-part. The closer to 1.0 the
"on-parts" fraction, the more the attribution respects the
ground-truth structure. The closer the per-part distribution matches
which parts the *prediction* depended on (e.g. heads typically need
beak shape), the more faithful the attribution.

Usage::

    uv run python experiments/funnybirds_part_alignment.py \\
        --probe data/vit_large_patch16_dinov3_probe_funny_birds.pt \\
        --n-images 20 --layer 12

Outputs a markdown table of per-concept "on-parts" fractions and
per-part relevance distributions. Useful as a quantitative check on
the attribution recipe.

Status: minimal first cut. Future work: per-class breakdown, comparison
across concept granularities (HeadConcept vs Q/K/V/AttnOutputDim), and
the formal FunnyBirds explainability protocol from the paper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform

from crp.attribution import CondAttribution
from zennit_extensions import AttnLRPBaselineComposite
from experiments.datasets import FunnyBirdsDataset
from experiments.datasets.funny_birds import (
    PART_COLORS_TO_NAME, BACKGROUND_COLOR,
)
from experiments.models import FinetunedProbe

REPO_ROOT = Path(__file__).resolve().parents[1]


def part_mask_from_pm(part_map_uint8: torch.Tensor):
    """Convert (3, H, W) uint8 part map → dict[part_name → bool mask (H, W)]
    plus a background mask. PART_COLORS_TO_NAME has the canonical RGB→name
    mapping from the official FunnyBirds release."""
    arr = part_map_uint8.permute(1, 2, 0).numpy()  # (H, W, 3)
    H, W, _ = arr.shape
    masks = {name: np.zeros((H, W), dtype=bool) for name in PART_COLORS_TO_NAME.values()}
    for rgb, name in PART_COLORS_TO_NAME.items():
        m = (arr == np.array(rgb, dtype=np.uint8)).all(axis=-1)
        masks[name] |= m
    bg = (arr == np.array(BACKGROUND_COLOR, dtype=np.uint8)).all(axis=-1)
    return masks, bg


def heatmap_2d(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 3:
        t = t.sum(dim=0)
    return t.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True,
                   help="path to the FunnyBirds probe checkpoint")
    p.add_argument("--n-images", type=int, default=20,
                   help="how many test images to evaluate")
    p.add_argument("--layer", type=int, default=12,
                   help="which block index to inspect")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = FinetunedProbe(checkpoint=args.probe, device=device)
    print(f"probe: {args.probe}  (val_acc={model.meta.get('val_acc', '?')})")

    cfg = resolve_data_config({}, model=model.backbone)
    transform = create_transform(**cfg, is_training=False)

    # Load FunnyBirds *test* split with part maps so we can measure alignment
    # on samples the probe never trained on.
    ds = FunnyBirdsDataset(
        root=REPO_ROOT / "data", split="test", transform=transform,
        with_part_map=True,
    )
    print(f"loaded {len(ds)} test images, taking the first {args.n_images}")

    composite = AttnLRPBaselineComposite()
    attribution = CondAttribution(model)

    # Aggregate per-image fractions.
    on_parts_fracs = []
    per_part_means = {name: [] for name in PART_COLORS_TO_NAME.values()}
    bg_fracs = []
    for i in range(args.n_images):
        image, cls, part_map = ds[i]
        x = image.unsqueeze(0).to(device).requires_grad_(True)
        # Heatmap from PLAIN attribution toward the predicted class.
        with torch.no_grad():
            pred = model(x).argmax(-1).item()
        x.grad = None
        res = attribution(x, [{"y": [pred]}], composite)
        hm = heatmap_2d(res.heatmap[0])
        # Resize part_map to match heatmap if needed.
        H_hm, W_hm = hm.shape
        if part_map.shape[-2:] != (H_hm, W_hm):
            from PIL import Image as _PIL
            part_map_pil = _PIL.fromarray(part_map.permute(1, 2, 0).numpy())
            part_map_pil = part_map_pil.resize((W_hm, H_hm), _PIL.NEAREST)
            part_map_resized = torch.from_numpy(
                np.array(part_map_pil)).permute(2, 0, 1)
        else:
            part_map_resized = part_map
        masks, bg = part_mask_from_pm(part_map_resized)

        abs_hm = np.abs(hm)
        total = abs_hm.sum() + 1e-12
        on_parts = sum(abs_hm[m].sum() for m in masks.values()) / total
        on_bg = abs_hm[bg].sum() / total
        on_parts_fracs.append(float(on_parts))
        bg_fracs.append(float(on_bg))
        for name, m in masks.items():
            per_part_means[name].append(float(abs_hm[m].sum() / total))
        print(
            f"  image {i:>3} class={cls:>3} pred={pred:>3}  "
            f"on_parts={on_parts:.3f}  on_bg={on_bg:.3f}"
        )

    print()
    print("## Aggregate part-alignment results")
    print()
    print(f"| metric | mean | std |")
    print(f"|---|---:|---:|")
    print(f"| on-parts fraction | {np.mean(on_parts_fracs):.3f} | {np.std(on_parts_fracs):.3f} |")
    print(f"| on-background fraction | {np.mean(bg_fracs):.3f} | {np.std(bg_fracs):.3f} |")
    print(f"| sum check (parts + bg) | {np.mean(on_parts_fracs) + np.mean(bg_fracs):.3f} | — |")
    print()
    print(f"### Per-part mean fraction (averaged over {args.n_images} images)")
    print()
    print(f"| part | mean fraction |")
    print(f"|---|---:|")
    for name, vs in per_part_means.items():
        print(f"| {name} | {np.mean(vs):.3f} |")
    print()
    print(
        "**Interpretation.** `on-parts ≫ on-bg` means the attribution "
        "respects the bird/background structure — a necessary (not "
        "sufficient) condition for faithful XAI on this synthetic "
        "testbed. Per-part fractions show which parts the network "
        "actually uses for its decisions."
    )


if __name__ == "__main__":
    main()
