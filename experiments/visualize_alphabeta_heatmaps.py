"""Visualize AlphaBeta-on-bilinear heatmaps to confirm spatial pattern preservation.

The eval matrix shows AlphaBeta variants give ~20 OOM smaller magnitudes
than the baseline 2Y+ε rule. This script visualizes the actual heatmaps
side-by-side so we can verify the spatial localization is preserved
(focus@10% says yes, but a visual check is reassuring).

Output: PNG with 4 rows (one per variant) × 3 columns (image, heatmap,
rank-normalized heatmap).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from timm.data import resolve_data_config, create_transform

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from crp.attribution import CondAttribution
from crp.attention_unfolded import EvaAttentionSubstitutionCanonizer
from crp.transformer_patches import AttnLRPCombinedComposite
from datasets import load as load_dataset


VARIANTS = [
    ("baseline_2y_eps",       dict(matmul_rule="matmul_factor_2", epsilon=1e-6)),
    ("alphabeta_1_0",         dict(matmul_rule="alpha_beta", alpha=1.0, beta=0.0)),
    ("alphabeta_2_-1",        dict(matmul_rule="alpha_beta", alpha=2.0, beta=-1.0)),
    ("alphabeta_05_05",       dict(matmul_rule="alpha_beta", alpha=0.5, beta=0.5)),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(
        REPO_ROOT / "data/vit_large_patch16_dinov3_probe_imagenette.pt",
        map_location=device, weights_only=False,
    )
    model = timm.create_model(
        ckpt["model_name"], pretrained=True, num_classes=ckpt["num_classes"],
    ).eval().to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=False)

    dataset = load_dataset("imagenette", split="val", transform=transform)
    # Pick sample 0 (class 0).
    x, y = dataset[0]
    x_dev = x.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_dev).argmax(-1).item()
    print(f"sample class = {y}, pred = {pred}")

    attribution = CondAttribution(model)
    n_blocks = len(model.blocks)

    # Denormalize image for display.
    mean = torch.tensor(cfg["mean"], device=device).view(3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(3, 1, 1)
    img = (x_dev.squeeze(0) * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(len(VARIANTS), 3, figsize=(11, 3.5 * len(VARIANTS)))
    for row, (label, kw) in enumerate(VARIANTS):
        composite = AttnLRPCombinedComposite(
            layerscale_uniform=True, residual_lrp="ratio",
        )
        sub = EvaAttentionSubstitutionCanonizer(
            block_indices=tuple(range(n_blocks)), **kw,
        )
        composite.canonizers = list(composite.canonizers) + [sub]
        x_run = x_dev.detach().clone().requires_grad_(True)
        result = attribution(x_run, [{"y": [pred]}], composite)
        hm = result.heatmap[0]
        if hm.dim() == 3 and hm.shape[0] == 3:
            hm = hm.sum(dim=0)
        hm = hm.detach().cpu().numpy()
        finite = np.isfinite(hm).all()
        vmax = float(np.abs(hm).max()) if finite else float("nan")

        ax_im, ax_raw, ax_rank = axes[row]
        ax_im.imshow(img); ax_im.axis("off")
        ax_im.set_title(f"{label}", fontsize=10)

        if finite and vmax > 0:
            ax_raw.imshow(hm, cmap="seismic", vmin=-vmax, vmax=vmax)
            ax_raw.set_title(f"raw heatmap (max|R|={vmax:.2e})", fontsize=9)
            # Rank-normalised by absolute value, with original sign.
            flat = hm.flatten()
            order = np.argsort(np.abs(flat))
            rank = np.empty_like(order, dtype=float)
            rank[order] = np.linspace(0, 1, flat.size)
            normed = (rank * np.sign(flat)).reshape(hm.shape)
            ax_rank.imshow(normed, cmap="seismic", vmin=-1, vmax=1)
            ax_rank.set_title("rank-normalised |R| (signed)", fontsize=9)
        else:
            ax_raw.text(0.5, 0.5, "NaN", ha="center", va="center", fontsize=18, color="red")
            ax_rank.text(0.5, 0.5, "—", ha="center", va="center", fontsize=18)
        ax_raw.axis("off"); ax_rank.axis("off")

    plt.tight_layout()
    out = REPO_ROOT / "tutorials/vit_crp/dinov3_variants/alphabeta/heatmaps_sample0.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
