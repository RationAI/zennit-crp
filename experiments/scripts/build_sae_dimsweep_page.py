"""Render the SAE-dimensionality-sweep concept-flipping figures + inject the
interactive section into the ``/zennit-flip/`` page (``public/index.html``).

This extends the original 4-case study (``public/index.html``): same concept-
flipping protocol (rank a block's detectors by signed conditional LRP relevance,
cumulatively ablate MoRF/LeRF, track Δ = p'_c / p_c), but now the SAE concept
basis is swept across dictionary sizes ``m`` and compared, per probe site, to the
axis-aligned ("no SAE") baseline.

Outputs (convention: png+pdf, committed, paper-ready — AGENTS.md §Figures):

* ``figures/concept_flipping_dimsweep/curves_<ds>_<site>_<variant>.{png,pdf}``
  — 12-block MoRF/LeRF curve grid, one per (dataset, site, variant). ``variant``
  is ``none`` (axis-aligned embed-dim baseline) or the SAE size ``m``.
* ``figures/concept_flipping_dimsweep/metric_<ds>.{png,pdf}`` — the *integral*
  metric: concept-flipping score (AOPC_most − AOPC_least) per block, one line per
  variant, faceted by site. Shows how faithfulness changes with SAE dimensionality.

PNGs are also copied to ``public/dimsweep/`` (webshare serves ``public/``); the
HTML ``<select>`` (hand-authored in ``public/index.html``) swaps the curve images.

Usage::

    python -m experiments.scripts.build_sae_dimsweep_page \
        --datasets dsprites --datasets funny_birds
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import polars as pl
import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
RES = REPO_ROOT / "data" / "results" / "concept_flipping"
FIG_ROOT = REPO_ROOT / "figures" / "concept_flipping_dimsweep"   # committed, paper-ready
WEB_DIR = REPO_ROOT / "public" / "dimsweep"                       # served by webshare
CONFIG = "cp_lrp_baseline"
COLORS = {"most": "tab:red", "least": "tab:green"}
RNG = np.random.default_rng(0)

# SAE dictionary sizes swept per dataset (toy d=384, imagenet vit_base d=768).
# "none" = axis-aligned embed-dim baseline (N = embed_dim of that model).
DATASET_DIMS = {
    "dsprites":    [154, 384, 768, 1536],
    "funny_birds": [154, 384, 768, 1536],
    "imagenet":    [384, 768, 1536, 3072],
}
EMBED_DIM = {"dsprites": 384, "funny_birds": 384, "imagenet": 768}
# distinct colours for the metric-vs-dim lines (one per dict size; "none"=black)
VARIANT_COLOR = {
    "none": "k", "154": "tab:blue", "384": "tab:green", "768": "tab:orange",
    "1536": "tab:red", "3072": "tab:purple",
}


def variants_for(ds: str):
    return ["none"] + [str(m) for m in DATASET_DIMS[ds]]


def variant_label(v: str, ds: str = "dsprites") -> str:
    return f"no SAE (axis-aligned, {EMBED_DIM[ds]})" if v == "none" else f"SAE m={v}"


def parquet_path(ds: str, site: str, variant: str) -> Path:
    if variant == "none":
        return RES / f"flipping_{CONFIG}_embed_dim_{site}_{ds}.parquet"
    return RES / f"flipping_{CONFIG}_sae_m{variant}_{site}_{ds}.parquet"


# ── shared analysis primitives (mirrors export_flipping_figures.py) ────────────

def _curve_matrix(d: pl.DataFrame, block: int, ordering: str):
    sub = d.filter((pl.col("block") == block) & (pl.col("ordering") == ordering) & (pl.col("n") > 0))
    if sub.is_empty():
        return np.array([]), np.empty((0, 0))
    ns = sorted(sub["n"].unique().to_list())
    piv = (sub.pivot(values="delta_prob", index="image_idx", on="n", aggregate_function="mean")
              .sort("image_idx"))
    return np.array(ns), piv.select([pl.col(str(n)) for n in ns]).to_numpy()


def _boot(M: np.ndarray, B: int = 2000):
    idx = RNG.integers(0, M.shape[0], size=(B, M.shape[0]))
    means = M[idx].mean(1)
    lo, hi = np.percentile(means, (2.5, 97.5), axis=0)
    return M.mean(0), lo, hi, M.std(0)


def _aopc(d: pl.DataFrame, block: int, ordering: str) -> float:
    sub = d.filter((pl.col("block") == block) & (pl.col("ordering") == ordering) & (pl.col("n") > 0))
    if sub.is_empty():
        return float("nan")
    return float(sub.group_by("image_idx").agg((1 - pl.col("delta_prob")).mean().alias("a"))["a"].mean())


def _save(fig, path_noext: Path):
    path_noext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_noext.with_suffix(".png"), dpi=110, bbox_inches="tight")
    fig.savefig(path_noext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure 1 · per-(dataset,site,variant) 12-block curve grid ──────────────────

def render_curves(ds: str, site: str, variant: str) -> Optional[Path]:
    f = parquet_path(ds, site, variant)
    if not f.is_file():
        print(f"  [skip] missing {f.name}")
        return None
    d = pl.read_parquet(f)
    N = int(d["n_detectors"][0])
    blocks = sorted(d["block"].unique().to_list())
    n_img = d["image_idx"].n_unique()
    cols = 4
    rows = -(-len(blocks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, b in enumerate(blocks):
        ax = axes.flat[i]
        ax.axis("on")
        for o in ("most", "least"):
            ns, M = _curve_matrix(d, b, o)
            if M.size == 0:
                continue
            mean, lo, hi, sd = _boot(M)
            x = ns / N
            ax.plot(x, mean, color=COLORS[o], lw=1.6)
            ax.fill_between(x, lo, hi, color=COLORS[o], alpha=0.30, lw=0)
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[o], alpha=0.08, lw=0)
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.3)
        ax.set_title(f"block {b}", fontsize=9)
        ax.set_xlabel("fraction perturbed  n/N", fontsize=8)
        ax.set_ylabel("Δ = p'_c / p_c", fontsize=8)
    legend_handles = [
        Line2D([], [], color=COLORS["most"], lw=1.6, label="most-relevant-first (MoRF): mean Δ"),
        Line2D([], [], color=COLORS["least"], lw=1.6, label="least-relevant-first (LeRF): mean Δ"),
        Patch(facecolor="gray", alpha=0.30, label="dark band: 95% bootstrap CI of the mean"),
        Patch(facecolor="gray", alpha=0.12, label="faint band: ±1 std across images"),
        Line2D([], [], color="k", lw=0.5, alpha=0.3, label="Δ = 1 (no change)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        f"Concept-flipping curves — {ds} · {site} · {variant_label(variant, ds)}  "
        f"(n={n_img} imgs, N={N} detectors)",
        fontsize=11, y=1.002)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = FIG_ROOT / f"curves_{ds}_{site}_{variant}"
    _save(fig, p)
    return p


# ── figure 2 · integral metric: score per block, line per variant, by site ─────

def render_metric(ds: str, sites: List[str]) -> Optional[Path]:
    fig, axes = plt.subplots(1, len(sites), figsize=(6.2 * len(sites), 3.6), squeeze=False)
    any_data = False
    for si, site in enumerate(sites):
        ax = axes[0][si]
        allscores = []
        for v in variants_for(ds):
            f = parquet_path(ds, site, v)
            if not f.is_file():
                continue
            d = pl.read_parquet(f)
            blocks = sorted(d["block"].unique().to_list())
            score = [_aopc(d, b, "most") - _aopc(d, b, "least") for b in blocks]
            any_data = True
            allscores += [s for s in score if s == s]  # drop NaN
            ax.plot(blocks, score, marker="o", ms=4, lw=1.6,
                    color=VARIANT_COLOR[v], label=variant_label(v, ds))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{site}", fontsize=11)
        ax.set_xlabel("transformer block")
        if si == 0:
            ax.set_ylabel("score = AOPC(most) − AOPC(least)")
        # Robust y-limits: an early-block large-m SAE can give a wildly negative
        # score (few correctly-classified spliced images ⇒ unstable Δ ratio) that
        # would squash the informative range. Clip to a padded robust window and
        # flag if any plotted point falls outside it.
        if allscores:
            a = np.array(allscores)
            lo, hi = np.percentile(a, [5, 95])
            pad = 0.12 * (hi - lo + 1e-6)
            ylo, yhi = min(lo - pad, -0.05), hi + pad
            if (a < ylo).any() or (a > yhi).any():
                ax.text(0.02, 0.02, "(extreme early-block points clipped)",
                        transform=ax.transAxes, fontsize=7, color="0.5")
            ax.set_ylim(ylo, yhi)
        ax.legend(fontsize=8, frameon=False)
    if not any_data:
        plt.close(fig)
        print(f"  [skip] no metric data for {ds}")
        return None
    fig.suptitle(
        f"Concept-flipping score per block vs SAE dimensionality — {ds}\n"
        "higher ⇒ MoRF collapses faster than LeRF (more faithful & concentrated relevance)",
        fontsize=11)
    fig.tight_layout()
    p = FIG_ROOT / f"metric_{ds}"
    _save(fig, p)
    return p


def main(
    datasets: List[str] = typer.Option(["dsprites", "funny_birds"], "--datasets"),
    sites: List[str] = typer.Option(["proj_drop", "residual"], "--sites"),
):
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for ds in datasets:
        for site in sites:
            for v in variants_for(ds):
                p = render_curves(ds, site, v)
                if p:
                    written.append(p)
        p = render_metric(ds, sites)
        if p:
            written.append(p)
    # copy PNGs to the webshare dir
    for p in written:
        shutil.copy(p.with_suffix(".png"), WEB_DIR / (p.name + ".png"))
    print(f"wrote {len(written)} figures (png+pdf) → {FIG_ROOT}")
    print(f"copied {len(written)} PNGs → {WEB_DIR}")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    typer.run(main)
