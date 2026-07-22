"""Render concept-flipping figures (PNG + PDF) from the gathered parquets.

Convention (see AGENTS.md §Figures): figures live OUTSIDE the gitignored
``data/`` tree so they survive accidental data loss and are paper-ready. Every
figure is written in BOTH ``.png`` (quick view / sharing) and ``.pdf`` (durable,
vector, drop-straight-into-the-paper) form. The output subdir is namespaced by
``<config>_<concept>`` and is WIPED before each render so stale figures from
removed datasets / superseded runs never linger.

Usage::

    uv run python -m experiments.scripts.export_flipping_figures \
        --config cp_lrp_baseline --concept embed_dim

Source of truth for the analysis is the notebook
``tutorials/concept_flipping_results.ipynb``; this script mirrors its plots for
non-interactive, reproducible figure export.
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path
from typing import List

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
FIG_ROOT = REPO_ROOT / "figures" / "concept_flipping"   # committed, survives data loss
RNG = np.random.default_rng(0)
COLORS = {"most": "tab:red", "least": "tab:green"}


def _curve_matrix(d: pl.DataFrame, block: int, ordering: str):
    """(ns, images×ns matrix of Δ) for one block+ordering, n>0."""
    sub = d.filter((pl.col("block") == block) & (pl.col("ordering") == ordering) & (pl.col("n") > 0))
    if sub.is_empty():
        return np.array([]), np.empty((0, 0))
    ns = sorted(sub["n"].unique().to_list())
    piv = sub.pivot(values="delta_prob", index="image_idx", on="n", aggregate_function="mean").sort("image_idx")
    return np.array(ns), piv.select([pl.col(str(n)) for n in ns]).to_numpy()


def _boot(M: np.ndarray, B: int = 2000):
    """Mean curve + bootstrap-95% CI band + ±1 std over the image axis."""
    idx = RNG.integers(0, M.shape[0], size=(B, M.shape[0]))
    means = M[idx].mean(1)
    lo, hi = np.percentile(means, (2.5, 97.5), axis=0)
    return M.mean(0), lo, hi, M.std(0)


def _crossing(d: pl.DataFrame, block: int, ordering: str):
    """The sign-crossing n along one ordering: the n at which mean cumulative
    relevance reverses — its PEAK for 'most' (we have just added the last
    positive-relevance detector; negatives follow) or its TROUGH for 'least'
    (last negative added; positives follow). By the sort symmetry the two land
    at mirror positions. Returns that n, or None."""
    cr = (d.filter((pl.col("block") == block) & (pl.col("ordering") == ordering) & (pl.col("n") > 0))
            .group_by("n").agg(pl.col("cum_relevance").mean().alias("c"))
            .sort("c", descending=(ordering == "most")))
    return cr["n"][0] if not cr.is_empty() else None


def _aopc(d: pl.DataFrame, block: int, ordering: str) -> float:
    """Normalised AOPC = mean_n (1 − Δ) over the curve, averaged across images."""
    sub = d.filter((pl.col("block") == block) & (pl.col("ordering") == ordering) & (pl.col("n") > 0))
    if sub.is_empty():
        return float("nan")
    return float(sub.group_by("image_idx").agg((1 - pl.col("delta_prob")).mean().alias("a"))["a"].mean())


def _save(fig, path_noext: Path):
    """Write a figure as both PNG (quick) and PDF (durable, paper-ready)."""
    fig.savefig(path_noext.with_suffix(".png"), dpi=110, bbox_inches="tight")
    fig.savefig(path_noext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render(config: str, concept: str, out_dir: Path) -> List[Path]:
    """Render the flipping figures per dataset into ``out_dir`` (png+pdf):
    per-block curves with CI band, and the concept-flipping score per block."""
    written: List[Path] = []
    files = sorted(glob.glob(str(RES / f"flipping_{config}_{concept}_*.parquet")))
    for f in files:
        ds = Path(f).stem[len(f"flipping_{config}_{concept}_"):]
        d = pl.read_parquet(f)
        N = int(d["n_detectors"][0])
        blocks = sorted(d["block"].unique().to_list())
        n_img = d["image_idx"].n_unique()

        # (1) per-block flipping curves with bootstrap-CI band
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
                # sign-crossing n★ marked AS A POINT ON this ordering's curve,
                # in matching colour (mirror positions for most vs least).
                nc = _crossing(d, b, o)
                if nc and ns.min() <= nc <= ns.max():
                    y_nc = float(np.interp(nc, ns, mean))
                    ax.plot(nc / N, y_nc, marker="o", color=COLORS[o], ms=6,
                            mec="k", mew=0.7, zorder=5)
            ax.axhline(1.0, color="k", lw=0.5, alpha=0.3)
            ax.set_title(f"block {b}", fontsize=9)
            ax.set_xlabel("fraction of dims perturbed  n/N", fontsize=8)
            ax.set_ylabel("Δ = p'_c / p_c", fontsize=8)
        # one comprehensive figure-level legend spelling out every element,
        # including exactly what the two shaded bands are
        legend_handles = [
            Line2D([], [], color=COLORS["most"], lw=1.6,
                   label="most-relevant-first: mean Δ over images"),
            Line2D([], [], color=COLORS["least"], lw=1.6,
                   label="least-relevant-first: mean Δ over images"),
            Patch(facecolor="gray", alpha=0.30,
                  label="dark band: 95% bootstrap CI of the mean curve"),
            Patch(facecolor="gray", alpha=0.12,
                  label="faint band: ±1 std across images (population spread)"),
            Line2D([], [], color="w", marker="o", ms=6, mec="k", mew=0.7, ls="",
                   markerfacecolor="0.5",
                   label="● sign-crossing n★ on each curve (positive→negative relevance), matching colour"),
            Line2D([], [], color="k", lw=0.5, alpha=0.3,
                   label="Δ = 1 (no change from baseline prediction)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(
            f"Concept-flipping curves — {config} · {concept} · {ds}  (n={n_img} imgs, N={N} detectors)\n"
            "Δ(n) = target-class prob after perturbing the n most/least relevant detectors, "
            "÷ baseline prob",
            fontsize=10, y=1.003)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        p = out_dir / f"curves_{ds}"
        _save(fig, p)
        written.append(p)

        # (2) concept-flipping score per block
        sc = [(b, _aopc(d, b, "most") - _aopc(d, b, "least")) for b in blocks]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar([s[0] for s in sc], [s[1] for s in sc], color="tab:purple")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("transformer block")
        ax.set_ylabel("score = AOPC(most) − AOPC(least)")
        ax.set_title(
            f"Concept-flipping score per block — {config} · {concept} · {ds}\n"
            ">0 ⇒ most-relevant-first collapses faster (faithful); peak ⇒ relevance concentrated there",
            fontsize=10)
        fig.tight_layout()
        p = out_dir / f"score_{ds}"
        _save(fig, p)
        written.append(p)
    return written


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    config: str = typer.Option("cp_lrp_baseline", "--config"),
    concept: str = typer.Option("embed_dim", "--concept"),
):
    out_dir = FIG_ROOT / f"{config}_{concept}"
    if out_dir.exists():
        shutil.rmtree(out_dir)   # wipe stale figures before re-render
    out_dir.mkdir(parents=True, exist_ok=True)
    written = render(config, concept, out_dir)
    if not written:
        raise SystemExit(f"no parquets matched flipping_{config}_{concept}_*.parquet under {RES}")
    print(f"wrote {len(written)} figures (png+pdf) → {out_dir}")
    for p in written:
        print(f"  {p.name}.{{png,pdf}}")


if __name__ == "__main__":
    app()
