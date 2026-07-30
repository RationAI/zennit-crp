"""Static (matplotlib) journal figures for the residual skip-vs-branch flow.

Reads npz files written by ``experiments/scripts/residual_flow_diag.py
compute`` (per residual site x sample x embedding dim absolute/signed
relevance for the branch and skip paths) and renders publication figures:

* per model — every row/slot comes from the site list stored in that model's
  own npz, so each figure shows exactly the layers the model has:
  1. ``<stem>branch_fraction_by_dim`` — heatmap of the sample-mean branch
     fraction f = |R_branch| / (|R_branch| + |R_skip|) per site x dim,
     diverging around 0.5;
  2. ``<stem>site_summary`` — per-site distribution over dims of the per-dim
     median f (IQR box, 5–95% whiskers), overlaid with the mass-weighted
     total branch share;
* ``--compare LABEL=NPZ ...`` — ``rf_compare_models``: the total branch share
  per site, one line per model, on the residual sites ALL compared models
  share — a model's line never lands on a slot for a layer it does not have.

Usage (idempotent, CPU only)::

    python -m experiments.scripts.residual_flow_static_figures \
        [--npz PATH] [--out-dir figures/residual_flow] [--stem-prefix TAG_]
    python -m experiments.scripts.residual_flow_static_figures \
        --compare "M1 label=path1.npz" "M2 label=path2.npz" ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (REPO / "data/results/residual_flow/"
               "residual_flow_m1_vit_small_fb_cp_lrp_baseline.npz")
DEFAULT_OUT = REPO / "figures/residual_flow"

# Diverging: blue = skip-dominated (f<0.5), red = branch-dominated (f>0.5),
# neutral at the 0.5 midpoint.
CMAP = "RdBu_r"
# Validated categorical order (dataviz reference palette, light mode,
# adjacent-pairlist CVD-safe): blue, green, magenta, yellow.
COMPARE_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
COMPARE_MARKERS = ["o", "s", "^", "D"]

SiteKey = Tuple[int, str]  # (block, "attn" | "mlp")


def site_label(key: SiteKey) -> str:
    return f"blk {key[0]} {key[1]}"


# ---------------------------------------------------------------------------
# Step 1 — per-model statistics, always keyed by the model's own site list
# ---------------------------------------------------------------------------

class SiteStats(NamedTuple):
    """Per-site branch-fraction summaries of ONE model (npz site order)."""
    keys: List[SiteKey]      # this model's residual sites, in network order
    mean_f: np.ndarray       # (n_sites, n_dims) sample-mean of f
    median_f: np.ndarray     # (n_sites, n_dims) sample-median of f
    total_share: np.ndarray  # (n_sites,) mass-weighted branch share

    @property
    def labels(self) -> List[str]:
        return [site_label(k) for k in self.keys]


def branch_fraction(z) -> np.ndarray:
    """f = |R_branch| / (|R_branch| + |R_skip|) per site x sample x dim."""
    branch, skip = z["branch_abs"], z["skip_abs"]
    return branch / (branch + skip + 1e-12)


def total_branch_share(z) -> np.ndarray:
    """Per-site mass-weighted share F = sum|R_branch| / (sum|R_branch| +
    sum|R_skip|) over dims, averaged over samples."""
    branch, skip = z["branch_abs"], z["skip_abs"]
    return (branch.sum(2) / (branch.sum(2) + skip.sum(2) + 1e-12)).mean(1)


def site_keys(z) -> List[SiteKey]:
    """The model's residual sites, straight from its npz."""
    return [(int(b), str(k)) for b, k in zip(z["site_block"], z["site_kind"])]


def load_site_stats(npz: Path) -> SiteStats:
    z = np.load(npz, allow_pickle=False)
    f = branch_fraction(z)
    return SiteStats(site_keys(z), f.mean(axis=1), np.median(f, axis=1),
                     total_branch_share(z))


def load_branch_share_by_site(npz: Path) -> Dict[SiteKey, float]:
    """{(block, kind): mass-weighted total branch share} for one model."""
    z = np.load(npz, allow_pickle=False)
    share = total_branch_share(z)
    return {key: float(share[i]) for i, key in enumerate(site_keys(z))}


# ---------------------------------------------------------------------------
# Step 2 — figure plumbing
# ---------------------------------------------------------------------------

def _journal_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/{stem}.{{pdf,png}}")


def _label_site_axis_rows(ax, stats: SiteStats) -> None:
    """One y-tick per residual site of this model."""
    ax.set_yticks(np.arange(len(stats.keys)))
    ax.set_yticklabels(stats.labels)


def _draw_block_separators(ax, stats: SiteStats) -> None:
    """Light white separator wherever the block index increments."""
    blocks = [k[0] for k in stats.keys]
    for i in range(1, len(blocks)):
        if blocks[i] != blocks[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=0.5, alpha=0.6)


# ---------------------------------------------------------------------------
# Step 3 — the per-model figures
# ---------------------------------------------------------------------------

def plot_branch_fraction_by_dim(stats: SiteStats, out_dir: Path,
                                stem: str) -> None:
    """Heatmap: one row per residual site of this model, x = embedding dim."""
    n_dim = stats.mean_f.shape[1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.imshow(stats.mean_f, aspect="auto", cmap=CMAP, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    _label_site_axis_rows(ax, stats)
    ax.set_xlabel("embedding dimension index")
    ax.set_xticks(list(np.arange(0, n_dim - 1, 64)) + [n_dim - 1])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    _draw_block_separators(ax, stats)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label(r"mean branch fraction  $f=\frac{|R_\mathrm{branch}|}"
                 r"{|R_\mathrm{branch}|+|R_\mathrm{skip}|}$", fontsize=9)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    _save(fig, out_dir, f"{stem}branch_fraction_by_dim")


def plot_site_summary(stats: SiteStats, out_dir: Path, stem: str) -> None:
    """Per site: distribution over dims of the per-dim median f (IQR box,
    5–95% whiskers) + the mass-weighted total branch share."""
    n_sites = len(stats.keys)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(n_sites)
    bp = ax.boxplot(
        [stats.median_f[i] for i in range(n_sites)], positions=x, widths=0.62,
        whis=(5, 95), showfliers=False, patch_artist=True, zorder=2,
        medianprops=dict(color="#1f2430", lw=1.4),
        boxprops=dict(facecolor="#aec7e0", edgecolor="#4d6a86", lw=0.8),
        whiskerprops=dict(color="#4d6a86", lw=0.9),
        capprops=dict(color="#4d6a86", lw=0.9),
    )
    ax.plot(x, stats.total_share, color="#c4442a", lw=1.4, marker="o", ms=4.5,
            zorder=3, label="total branch share (mass-weighted)")
    ax.axhline(0.5, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(stats.labels, rotation=90)
    ax.set_ylabel("branch fraction")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color="0.9", lw=0.7)
    ax.set_axisbelow(True)
    handles, lab = ax.get_legend_handles_labels()
    handles.append(bp["boxes"][0])
    lab.append("per-dim median $f$ (IQR box, 5–95% whiskers)")
    ax.legend(handles, lab, loc="upper left", frameon=False)
    _save(fig, out_dir, f"{stem}site_summary")


# ---------------------------------------------------------------------------
# Step 4 — cross-model comparison on the shared sites only
# ---------------------------------------------------------------------------

def shared_site_keys(models: List[Tuple[str, Dict[SiteKey, float]]]
                     ) -> List[SiteKey]:
    """Sites present in EVERY compared model, in the first model's site order.

    The comparison axis gets one slot per shared site — never a slot for a
    layer some evaluated model does not have. Sites a model has outside the
    shared set are reported, not plotted under another model's name.
    """
    shared = [k for k in models[0][1] if all(k in m for _, m in models[1:])]
    for name, per_site in models:
        dropped = [site_label(k) for k in per_site if k not in shared]
        if dropped:
            print(f"  {name}: {len(dropped)} site(s) not shared by all "
                  f"models, excluded from comparison: {dropped}")
    if not shared:
        raise RuntimeError("the compared models share no residual sites")
    return shared


def compare(pairs: List[str], out_dir: Path) -> None:
    """Per-site mass-weighted total branch share, one line per model, on the
    residual sites all compared models share."""
    models = [(pair.partition("=")[0],
               load_branch_share_by_site(Path(pair.partition("=")[2])))
              for pair in pairs]
    shared = shared_site_keys(models)

    _journal_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(shared))
    for i, (label, per_site) in enumerate(models):
        ax.plot(x, [per_site[k] for k in shared],
                color=COMPARE_COLORS[i], lw=1.6, zorder=3,
                marker=COMPARE_MARKERS[i], ms=4, label=label)
    ax.axhline(0.5, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([site_label(k) for k in shared], rotation=90)
    ax.set_ylabel("total branch share (mass-weighted)")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color="0.9", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncols=2)
    _save(fig, out_dir, "rf_compare_models")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stem-prefix", default="",
                    help="prefix for output stems, e.g. 'm1_vit_small_fb_'")
    ap.add_argument("--compare", nargs="+", default=None,
                    metavar="LABEL=NPZ",
                    help="cross-model mode: per-site mass-weighted branch "
                         "share, one line per model -> rf_compare_models")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare, args.out_dir)
        return

    stats = load_site_stats(args.npz)
    _journal_style()
    plot_branch_fraction_by_dim(stats, args.out_dir, args.stem_prefix)
    plot_site_summary(stats, args.out_dir, args.stem_prefix)
    print(f"n_sites={len(stats.keys)} n_dims={stats.mean_f.shape[1]}")


if __name__ == "__main__":
    main()
