"""Static (matplotlib) journal figures for the residual skip-vs-branch LRP flow.

Reads the npz written by ``experiments/scripts/residual_flow_diag.py compute``
(per residual site x sample x embedding dim absolute/signed relevance for the
branch and skip paths) and renders two publication PDFs/PNGs:

1. ``branch_fraction_by_dim`` — 24-row heatmap: rows = residual sites in
   network order (blk b attn / blk b mlp), x = embedding dim index (0..383),
   color = mean over samples of the branch fraction
   f = |R_branch| / (|R_branch| + |R_skip|), diverging around 0.5.
2. ``site_summary`` — per site, the distribution over dims of the per-dim
   median f (median over samples; box = IQR, whiskers = 5-95%), overlaid with
   the site's mass-weighted total branch share
   F = sum_d |R_branch| / (sum_d |R_branch| + sum_d |R_skip|), averaged over
   samples.

Usage (idempotent, CPU only):
    python -m experiments.scripts.residual_flow_static_figures \
        [--npz PATH] [--out-dir figures/residual_flow]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (REPO / "data/results/residual_flow/"
               "residual_flow_vit_small_funny_birds_cp_lrp_baseline.npz")
DEFAULT_OUT = REPO / "figures/residual_flow"

# Diverging: blue = skip-dominated (f<0.5), red = branch-dominated (f>0.5),
# neutral light gray-white at the 0.5 midpoint.
CMAP = "RdBu_r"


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/{stem}.{{pdf,png}}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    ba, sa = z["branch_abs"], z["skip_abs"]          # (24, S, 384)
    site_block, site_kind = z["site_block"], z["site_kind"]
    n_sites, n_samples, n_dim = ba.shape
    labels = [f"blk {b} {k}" for b, k in zip(site_block, site_kind)]

    f = ba / (ba + sa + 1e-12)                       # (24, S, D)
    mean_f = f.mean(axis=1)                          # (24, D)
    med_f = np.median(f, axis=1)                     # (24, D) per-dim median over samples
    tot_share = (ba.sum(axis=2)
                 / (ba.sum(axis=2) + sa.sum(axis=2) + 1e-12)).mean(axis=1)  # (24,)

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # ---- Figure 1: per-dim mean branch fraction heatmap ----------------
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.imshow(mean_f, aspect="auto", cmap=CMAP, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_yticks(np.arange(n_sites))
    ax.set_yticklabels(labels)
    ax.set_xlabel("embedding dimension index")
    ax.set_xticks(list(np.arange(0, n_dim - 1, 64)) + [n_dim - 1])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    # light separators between blocks (every 2 rows = attn+mlp)
    for y in np.arange(1.5, n_sites - 1, 2):
        ax.axhline(y, color="white", lw=0.5, alpha=0.6)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label(r"mean branch fraction  $f=\frac{|R_\mathrm{branch}|}"
                 r"{|R_\mathrm{branch}|+|R_\mathrm{skip}|}$", fontsize=9)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    _save(fig, args.out_dir, "branch_fraction_by_dim")

    # ---- Figure 2: per-site distribution + mass-weighted total ---------
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(n_sites)
    bp = ax.boxplot(
        [med_f[i] for i in range(n_sites)], positions=x, widths=0.62,
        whis=(5, 95), showfliers=False, patch_artist=True, zorder=2,
        medianprops=dict(color="#1f2430", lw=1.4),
        boxprops=dict(facecolor="#aec7e0", edgecolor="#4d6a86", lw=0.8),
        whiskerprops=dict(color="#4d6a86", lw=0.9),
        capprops=dict(color="#4d6a86", lw=0.9),
    )
    ax.plot(x, tot_share, color="#c4442a", lw=1.4, marker="o", ms=4.5,
            zorder=3, label="total branch share (mass-weighted)")
    ax.axhline(0.5, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("branch fraction")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color="0.9", lw=0.7)
    ax.set_axisbelow(True)
    handles, lab = ax.get_legend_handles_labels()
    handles.append(bp["boxes"][0])
    lab.append("per-dim median $f$ (IQR box, 5–95% whiskers)")
    ax.legend(handles, lab, loc="upper left", frameon=False)
    _save(fig, args.out_dir, "site_summary")

    print(f"n_sites={n_sites} n_samples={n_samples} n_dim={n_dim}")


if __name__ == "__main__":
    main()
