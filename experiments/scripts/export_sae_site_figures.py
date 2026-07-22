"""Render the 4 SAE-vs-axis-aligned × probe-site concept-flipping figures.

Four cases = ``{proj_drop, residual} × {embed_dim (axis-aligned), sae}``. One
figure per case. Each figure joins the **three dataset-trained models
horizontally** (dsprites · colored_mnist · funny_birds) as side-by-side
sub-grids; within each sub-grid every transformer **block** gets its own
concept-flipping curve panel (most-relevant-first vs least, bootstrap-CI band,
sign-crossing marker) — same per-block curve style as
``export_flipping_figures.py``, which this reuses.

Reads ``flipping_{config}_{concept}_{site}_{dataset}.parquet`` from
``data/results/concept_flipping/``. Writes png+pdf under
``figures/concept_flipping/sae_site/`` (AGENTS.md §Figures convention).

Usage::

    uv run python -m experiments.scripts.export_sae_site_figures --config cp_lrp_baseline
"""
from __future__ import annotations

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

from experiments.scripts.export_flipping_figures import (
    REPO_ROOT, RES, COLORS, _curve_matrix, _boot, _crossing,
)

FIG_DIR = REPO_ROOT / "figures" / "concept_flipping" / "sae_site"

# the four cases, in display order
CASES = [
    ("proj_drop", "embed_dim", "proj_drop · axis-aligned (embed dims)"),
    ("proj_drop", "sae",       "proj_drop · SAE dictionary"),
    ("residual",  "embed_dim", "residual stream · axis-aligned (embed dims)"),
    ("residual",  "sae",       "residual stream · SAE dictionary"),
]
DATASETS = ["dsprites", "colored_mnist", "funny_birds"]


def _panel(ax, d: pl.DataFrame, block: int, N: int):
    """One block's most/least flipping curves with CI band + n★ marker."""
    for o in ("most", "least"):
        ns, M = _curve_matrix(d, block, o)
        if M.size == 0:
            continue
        mean, lo, hi, sd = _boot(M)
        x = ns / N
        ax.plot(x, mean, color=COLORS[o], lw=2.0)
        ax.fill_between(x, lo, hi, color=COLORS[o], alpha=0.30, lw=0)
        ax.fill_between(x, mean - sd, mean + sd, color=COLORS[o], alpha=0.08, lw=0)
        nc = _crossing(d, block, o)
        if nc and ns.min() <= nc <= ns.max():
            ax.plot(nc / N, float(np.interp(nc, ns, mean)), marker="o",
                    color=COLORS[o], ms=7, mec="k", mew=0.8, zorder=5)
    ax.axhline(1.0, color="k", lw=0.7, alpha=0.3)
    ax.set_title(f"block {block}", fontsize=12)
    ax.tick_params(labelsize=9)


def render_case(config: str, site: str, concept: str, label: str, out: Path) -> Path | None:
    files = {ds: RES / f"flipping_{config}_{concept}_{site}_{ds}.parquet" for ds in DATASETS}
    present = {ds: p for ds, p in files.items() if p.is_file()}
    if not present:
        print(f"  [skip] no parquets for site={site} concept={concept}")
        return None

    fig = plt.figure(figsize=(9.6 * len(present), 11.0))
    subfigs = fig.subfigures(1, len(present), wspace=0.04)
    if len(present) == 1:
        subfigs = [subfigs]
    for sf, (ds, p) in zip(subfigs, present.items()):
        d = pl.read_parquet(p)
        N = int(d["n_detectors"][0])
        blocks = sorted(d["block"].unique().to_list())
        n_img = d["image_idx"].n_unique()
        cols = 3
        rows = -(-len(blocks) // cols)
        axes = sf.subplots(rows, cols, squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for i, b in enumerate(blocks):
            ax = axes.flat[i]
            ax.axis("on")
            _panel(ax, d, b, N)
        sf.suptitle(f"{ds}   (N={N}, n={n_img} imgs)", fontsize=14, y=0.975)
        sf.supxlabel("fraction perturbed  n/N", fontsize=12)
        sf.supylabel("Δ = p'_c / p_c", fontsize=12)

    legend_handles = [
        Line2D([], [], color=COLORS["most"], lw=1.6, label="most-relevant-first (MoRF)"),
        Line2D([], [], color=COLORS["least"], lw=1.6, label="least-relevant-first (LeRF)"),
        Patch(facecolor="gray", alpha=0.30, label="95% bootstrap CI of mean"),
        Patch(facecolor="gray", alpha=0.12, label="±1 std across images"),
        Line2D([], [], color="w", marker="o", ms=5, mec="k", mew=0.6, ls="",
               markerfacecolor="0.5", label="sign-crossing n★"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Concept-flipping — {config} · {label}\n"
                 "per-block curves; 3 dataset-trained models joined horizontally",
                 fontsize=17, y=1.05)

    p_noext = out / f"flip_{site}_{concept}"
    fig.savefig(p_noext.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(p_noext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p_noext.name}.{{png,pdf}}")
    return p_noext


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(config: str = typer.Option("cp_lrp_baseline", "--config")):
    if FIG_DIR.exists():
        shutil.rmtree(FIG_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for site, concept, label in CASES:
        print(f"case: {site} / {concept}")
        p = render_case(config, site, concept, label, FIG_DIR)
        if p:
            written.append(p)
    print(f"\nwrote {len(written)}/4 case figures → {FIG_DIR}")


if __name__ == "__main__":
    app()
