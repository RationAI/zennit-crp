"""Journal figures for registers step 1 (XAI-35): outlier-token / LRP colocation.

Remakes ``gallery_colocation_overlays`` as two page-fitting figures of 3 sample
rows each (rows labeled a..c and d..f; the name<->letter mapping is printed to
stdout for the journal caption). Columns per row:

    input | class-conditional LRP heatmap | overlay (outliers + top-|N| by |R|)

- Outlier patches (consensus mask from step 1): magenta borders.
- Top-|N| patches by |R|: cyan borders (inset), N = that image's outlier count.
- Per-patch relevance aggregation (identical to the step-1 arrays,
  ``colocation_*.npz::heat_patch_abs``): R_patch = sum over the patch's 16x16
  pixels of |R(pixel)| of the class-conditional LRP heatmap.

Data: ``data/results/registers/gallery_samples_vit_base_imagenet.npz`` (raw
inputs in [0,1] + raw heatmaps) and ``colocation_vit_base_imagenet.npz``
(consensus outlier masks, per-patch |R|). 14x14 patch grid, row-major,
patch (r, c) -> pixels [16r:16r+16, 16c:16c+16].

Usage (idempotent, CPU only):
    python -m experiments.scripts.registers_step1_figures \
        [--data-dir data/results/registers] [--out-dir figures/registers/step1_detect]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "data/results/registers"
DEFAULT_OUT = REPO / "figures/registers/step1_detect"

PATCH = 16          # pixels per patch side
GRID = 14           # patches per image side
C_OUT = "#ff00c8"   # magenta — outlier tokens
C_TOP = "#00e5ff"   # cyan — top-|N| patches by |R|
COL_TITLES = ("input", "class-conditional LRP heatmap",
              "$N$ outliers (magenta)\n+ top-$|N|$ patches by $|R|$ (cyan)")


def _rect(ax, r: int, c: int, color: str, lw: float, inset: float) -> None:
    ax.add_patch(Rectangle(
        (c * PATCH - 0.5 + inset, r * PATCH - 0.5 + inset),
        PATCH - 2 * inset, PATCH - 2 * inset,
        fill=False, edgecolor=color, linewidth=lw))


def _render_page(keys, inputs, heatmaps, masks, heat_patch, letters,
                 out_dir: Path, stem: str) -> None:
    n = len(keys)
    fig, axes = plt.subplots(n, 3, figsize=(6.6, 2.2 * n + 0.4))
    fig.subplots_adjust(left=0.035, right=0.995, top=1 - 0.42 / (2.2 * n + 0.4),
                        bottom=0.005, wspace=0.04, hspace=0.04)
    for j, t in enumerate(COL_TITLES):
        axes[0, j].set_title(t, fontsize=8.5, pad=5)
    for i in range(n):
        img = np.transpose(inputs[i], (1, 2, 0)).clip(0, 1)
        hm = heatmaps[i]
        vmax = np.percentile(np.abs(hm), 99.5)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(hm, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[i, 2].imshow(img)

        n_out = int(masks[i].sum())
        out_rc = [divmod(int(t_), GRID) for t_ in np.flatnonzero(masks[i])]
        top_flat = np.argsort(-heat_patch[i].ravel())[:n_out]
        top_rc = [divmod(int(t_), GRID) for t_ in top_flat]
        for r, c in out_rc:
            _rect(axes[i, 2], r, c, C_OUT, lw=2.0, inset=0.0)
        for r, c in top_rc:
            _rect(axes[i, 2], r, c, C_TOP, lw=1.6, inset=2.5)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(f"({letters[i]})", fontsize=10,
                              fontweight="bold", rotation=0, labelpad=14,
                              va="center")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/{stem}.{{pdf,png}}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    g = np.load(args.data_dir / "gallery_samples_vit_base_imagenet.npz",
                allow_pickle=False)
    c = np.load(args.data_dir / "colocation_vit_base_imagenet.npz",
                allow_pickle=True)
    keys = [str(k) for k in g["keys"]]
    assert keys == [str(k) for k in c["keys"]], "sample order mismatch"

    # Sanity: the stored per-patch |R| equals sum of |R| over each 16x16 block.
    blocks = g["heatmaps"].reshape(-1, GRID, PATCH, GRID, PATCH)
    assert np.allclose(np.abs(blocks).sum(axis=(2, 4)), c["heat_patch_abs"],
                       atol=1e-5), "heat_patch_abs is not sum(|R|) per patch"

    letters = "abcdef"
    for page, (lo, hi, stem) in enumerate(
            [(0, 3, "gallery_colocation_p1"), (3, 6, "gallery_colocation_p2")]):
        _render_page(keys[lo:hi], g["inputs"][lo:hi], g["heatmaps"][lo:hi],
                     c["outlier_masks"][lo:hi], c["heat_patch_abs"][lo:hi],
                     letters[lo:hi], args.out_dir, stem)

    print("sample order (letter -> name, ds_index, n_outliers):")
    for i, k in enumerate(keys):
        print(f"  ({letters[i]}) {k}  ds_index={int(g['ds_indices'][i])}"
              f"  n_outliers={int(c['outlier_masks'][i].sum())}")


if __name__ == "__main__":
    main()
