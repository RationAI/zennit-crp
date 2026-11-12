"""Emit the rank-level results of the concept-detector benchmark: normalised
DAPC and top-k rank agreement, as figures, LaTeX fragments and a CSV of every
number, so the paper never carries a hand-copied value.

Statistics. The independent unit is the **image**: every cell = (model,
concept, site, block) is aggregated over its own images and nothing is pooled
across blocks, because the same images recur in every block and blocks are not
replicates of one another. Point estimates carry a percentile bootstrap CI over
images and the image count that backs them.

Scope. Main outputs cover the attention-internal sites (value, query/key, head
output), which is what the paper claims about; residual and proj_drop are
post-projection and go to a separate supplementary fragment.

Block choice is fixed in advance: the figures show every block, and the tables
report the **last block**, nearest the readout. Picking the best-scoring block
per cell after seeing the numbers would be selection on the outcome.

Controls. Normalised DAPC has chance at 0 by construction (DAPC is
antisymmetric, so a random ordering scores 0 in expectation) and the best
ordering found at 1. Rank agreement is read against two references: the chance
floor k/D for two independent k-subsets, and the agreement *between two
independent searches*, which is the level a ranking could reach without being
wrong — without it an overlap number has no scale.

Run:  uv run python -m experiments.scripts.emit_rank_outputs
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import rank_analysis as ra

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "figures" / "rank_analysis"
CSV = REPO / "data" / "results" / "benchmark" / "rank_analysis.csv"
TEX = Path("/home/claude/workspaces/crp-paper/iclr2026/tables")

MODELS = [("vit_small_funny_birds", "M1 ViT-S/16 · FunnyBirds"),
          ("vit_base_imagenet", "M2 ViT-B/16 · ImageNet"),
          ("vit_dinov3_small_funny_birds", "M5 DINOv3-S/16 · FunnyBirds"),
          ("vit_dinov3_base_imagenet", "M6 DINOv3-B/16 · ImageNet")]
CONCEPTS = [("embed_dim", "embedding dimension"), ("head", "head")]
COLOR = {"cp_lrp": "#1f77b4", "chefer": "#d62728", "optimal": "#7f7f7f",
         "optimal_dual": "#bcbd22", "optimal_dyna": "#2ca02c", "random": "#9467bd"}
TEX_SITE = {"value": r"\texttt{value}", "qk": r"\texttt{q/k}",
            "query": r"\texttt{query}", "key": r"\texttt{key}",
            "head_out": r"\texttt{head\_out}", "proj_drop": r"\texttt{proj\_drop}",
            "residual": r"\texttt{residual}"}


# ── figures ───────────────────────────────────────────────────────────────────
def fig_normdapc_by_block(c: ra.Cells, site, outdir, label):
    blocks = c.blocks(site)
    if not blocks:
        return
    methods = [m for m in ("cp_lrp", "chefer", "optimal", "optimal_dual", "optimal_dyna")
               if any((m, site, b) in c.dapc for b in blocks)]
    series = []
    for m in methods:
        xs, pt, lo, hi = [], [], [], []
        for b in blocks:
            r = ra.normalised_dapc(c, m, site, b)
            if r is None or not np.isfinite(r[0]):
                continue
            xs.append(b); pt.append(r[0]); lo.append(r[1]); hi.append(r[2])
        if xs and not all(v >= 0.999 for v in pt):
            series.append((m, xs, pt, lo, hi))   # a search that *is* the bound
                                                 # plots as a flat 1.0 and says nothing
    if not series:
        return                     # nothing measurable here; emit no empty figure
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m, xs, pt, lo, hi in series:
        ax.plot(xs, pt, "o-", ms=3.5, lw=1.4, color=COLOR[m], label=ra.METHOD_LABEL[m])
        ax.fill_between(xs, lo, hi, color=COLOR[m], alpha=0.15, lw=0)
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.axhline(0.0, color="k", lw=0.9, ls=":")
    ax.text(0.995, 1.0, " best ordering found", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7.5, color="k")
    ax.text(0.995, 0.0, " chance (random ordering)", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7.5, color="k")
    ax.set_xlabel("transformer block")
    ax.set_ylabel("DAPC / best DAPC found")
    ax.set_title(f"Normalised deletion-area score · {label}\n"
                 f"site {ra.SITE_LABEL[site]}, {c.concept} detectors (D={c.D}), "
                 f"n={c.n_images} images; median, 95% bootstrap CI",
                 fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"normdapc_by_block_{site}.{ext}", dpi=160)
    plt.close(fig)


def fig_overlap_vs_k(c: ra.Cells, site, block, outdir, label):
    ks = [k for k in ra.k_grid(c.D) if k <= c.D]
    if not ks:
        return
    if ra.best_search_order(c, site, block) is None:
        return                     # no reference ordering: nothing to agree with
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in ra.LRP_METHODS:
        pt, lo, hi, got = [], [], [], []
        for k in ks:
            r = ra.agreement_with_search(c, m, site, block, k, stat=np.mean)
            if r is None:
                continue
            got.append(k); pt.append(r[0]); lo.append(r[1]); hi.append(r[2])
        if not got:
            continue
        ax.plot(got, pt, "o-", ms=3.5, lw=1.4, color=COLOR[m], label=ra.METHOD_LABEL[m])
        ax.fill_between(got, lo, hi, color=COLOR[m], alpha=0.15, lw=0)
    # control: agreement between two independent searches = the reachable level
    sp0 = ra.search_pair_overlap(c, site, block, ks[0])
    if sp0 is not None:
        a, b2 = sp0[1]
        pt = [ra.search_pair_overlap(c, site, block, k)[0] for k in ks]
        ax.plot(ks, pt, "s--", ms=3.5, lw=1.2, color="#555555",
                label=f"{ra.METHOD_LABEL[a]} vs {ra.METHOD_LABEL[b2]}")
    ax.plot(ks, [k / c.D for k in ks], ":", lw=1.2, color="k", label="chance (k/D)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("k (top-k detectors)")
    ax.set_ylabel("overlap fraction with the best ordering")
    ax.set_title(f"Top-k rank agreement · {label}\n"
                 f"site {ra.SITE_LABEL[site]}, block {block}, {c.concept} "
                 f"detectors (D={c.D}), n={c.n_images} images; mean, 95% CI",
                 fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"overlap_vs_k_{site}_b{block}.{ext}", dpi=160)
    plt.close(fig)


# ── tables + csv ──────────────────────────────────────────────────────────────
def _cell_rows(c: ra.Cells, label, scope):
    """One CSV row per (site, block, method) with both estimands."""
    rows = []
    for site in c.sites(scope):
        for b in c.blocks(site):
            for m in ra.LRP_METHODS + ra.SEARCH_METHODS:
                nd = ra.normalised_dapc(c, m, site, b)
                if nd is None:
                    continue
                raw = c.dapc.get((m, site, b))
                row = {"model": c.key, "model_label": label, "concept": c.concept,
                       "site": site, "block": b, "method": m, "D": c.D,
                       "n_images": c.n_images,
                       "cell_n": ra.cell_n(c, site, b),
                       "signal_threshold": ra.signal_threshold(c, site, b),
                       "dapc_mean": float(np.nanmean(raw)) if raw is not None else "",
                       "norm_dapc": nd[0], "norm_lo": nd[1], "norm_hi": nd[2],
                       "norm_n": nd[3], "norm_undefined": nd[4]}
                for k in ra.k_grid(c.D):
                    r = ra.agreement_with_search(c, m, site, b, k, stat=np.mean)
                    row[f"overlap_k{k}"] = "" if r is None else r[0]
                    row[f"overlap_k{k}_lo"] = "" if r is None else r[1]
                    row[f"overlap_k{k}_hi"] = "" if r is None else r[2]
                    row[f"chance_k{k}"] = ra.chance_overlap(k, c.D)[0]
                rows.append(row)
    return rows


def _fmt(pt, lo, hi, n):
    if not np.isfinite(pt):
        return "---"
    if not np.isfinite(lo):
        return f"${pt:+.2f}$"
    return f"${pt:+.2f}$ {{\\tiny$[{lo:+.2f},{hi:+.2f}]$}}"


def table_normdapc(scope, caption_sites):
    """Normalised DAPC at the last block, every model and concept."""
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"model & site & CP-LRP & Chefer & dual greedy & $n$ \\", r"\midrule"]
    for concept, cname in CONCEPTS:
        L.append(r"\multicolumn{6}{l}{\emph{" + cname + r" detectors}} \\")
        for key, label in MODELS:
            try:
                c = ra.load_cells(key, concept)
            except FileNotFoundError:
                continue
            first = True
            for site in c.sites(scope):
                blocks = c.blocks(site)
                if not blocks:
                    continue
                b = blocks[-1]
                cells = []
                for m in ("cp_lrp", "chefer", "optimal_dual"):
                    r = ra.normalised_dapc(c, m, site, b)
                    cells.append("---" if r is None else _fmt(*r[:4]))
                if all(x == "---" for x in cells):
                    continue          # no measurable cell here; emit no dash row
                L.append(f"{label if first else ''} & {TEX_SITE[site]} & "
                         + " & ".join(cells)
                         + f" & {ra.cell_n(c, site, b)}" + r" \\")
                first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    return "\n".join(L)


def table_overlap(scope):
    """Top-k agreement with the best ordering, last block, with both controls."""
    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"model & site & $k$ & CP-LRP & Chefer & search vs search & chance \\",
         r"\midrule"]
    for concept, cname in CONCEPTS:
        L.append(r"\multicolumn{7}{l}{\emph{" + cname + r" detectors}} \\")
        for key, label in MODELS:
            try:
                c = ra.load_cells(key, concept)
            except FileNotFoundError:
                continue
            k = ra.report_k(c.D)
            first = True
            for site in c.sites(scope):
                blocks = c.blocks(site)
                if not blocks:
                    continue
                b = blocks[-1]
                cells = []
                for m in ra.LRP_METHODS:
                    r = ra.agreement_with_search(c, m, site, b, k, stat=np.mean)
                    cells.append("---" if r is None
                                 else f"${r[0]:.2f}$ {{\\tiny$[{r[1]:.2f},{r[2]:.2f}]$}}")
                sp = ra.search_pair_overlap(c, site, b, k)
                if all(x == "---" for x in cells) and sp is None:
                    continue
                ss = "---" if sp is None else f"${sp[0]:.2f}$"
                L.append(f"{label if first else ''} & {TEX_SITE[site]} & {k} & "
                         + " & ".join(cells) + f" & {ss} & ${k / c.D:.3f}$" + r" \\")
                first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    return "\n".join(L)


NOTE_NORM = (
    "% Normalised deletion-area score: per-image DAPC divided by the best DAPC any\n"
    "% search found for that image and cell, then the median over images with a\n"
    "% 95%% percentile bootstrap CI. 0 = a random ordering (exact, since DAPC is\n"
    "% antisymmetric); 1 = the best ordering found, which is an estimate of the\n"
    "% attainable bound from below, so values above 1 mean the ranking beat the\n"
    "% search. Cells whose bound does not clear 3 sd of the random control are\n"
    "% excluded; n is the number of images that remain. Last block of each site.\n"
    "% CP-LRP is undefined at query/key: its StopGradient severs that path.\n"
    "% Generated by experiments/scripts/emit_rank_outputs.py -- do not hand-edit.\n")
NOTE_OVL = (
    "% Top-k rank agreement: mean over images of |top_k(ranking) & top_k(best\n"
    "% ordering found)| / k, with a 95%% percentile bootstrap CI over images.\n"
    "% Two references bound the scale: 'chance' = k/D, the expectation for two\n"
    "% independent uniform k-subsets, and 'search vs search' = the same overlap\n"
    "% between the greedy and dual-greedy orderings, i.e. how much two\n"
    "% independent searches agree with each other on the same cell.\n"
    "% Last block of each site. CP-LRP is undefined at query/key (StopGradient).\n"
    "% Generated by experiments/scripts/emit_rank_outputs.py -- do not hand-edit.\n")


def coverage_report():
    """What each cell actually has, so a gap is visible instead of silent."""
    lines = ["# Rank-analysis coverage",
             "",
             "One row per (model, concept, site). `blocks` counts the blocks with a",
             "stored search ordering; `n` is the images that clear the cell's signal",
             "threshold at the last block. A missing search means the cell carries no",
             "bound, so no normalised score and no rank agreement can be computed.",
             ""]
    for scope, title in ((ra.ATTENTION_SITES, "Attention-internal sites (main)"),
                         (ra.POST_PROJECTION_SITES, "Post-projection sites (supplementary)")):
        lines += [f"## {title}", "",
                  "| model | concept | site | blocks w/ search | LRP rankings | n at last block |",
                  "|---|---|---|---|---|---|"]
        for key, label in MODELS:
            for concept, _ in CONCEPTS:
                try:
                    c = ra.load_cells(key, concept)
                except FileNotFoundError:
                    continue
                for site in scope:
                    blocks = c.blocks(site)
                    if not blocks:
                        continue
                    wsearch = [b for b in blocks
                               if ra.bound_per_image(c, site, b) is not None]
                    lrp = [ra.METHOD_LABEL[m] for m in ra.LRP_METHODS
                           if (m, site, blocks[-1]) in c.order]
                    lines.append(
                        f"| {label} | {concept} | {site} | {len(wsearch)}/{len(blocks)} "
                        f"| {', '.join(lrp) or '--'} "
                        f"| {ra.cell_n(c, site, blocks[-1])}/{c.n_images} |")
                lines.append("")
    return "\n".join(lines)


def main():
    if FIG.exists():
        shutil.rmtree(FIG)           # stale figures must not outlive their data
    FIG.mkdir(parents=True)
    rows = []
    for key, label in MODELS:
        for concept, _cname in CONCEPTS:
            try:
                c = ra.load_cells(key, concept)
            except FileNotFoundError:
                print(f"-- {key} [{concept}]: no store, skipped")
                continue
            outdir = FIG / f"{key}__{concept}"
            outdir.mkdir(parents=True, exist_ok=True)
            for site in c.sites(ra.ATTENTION_SITES) + c.sites(ra.POST_PROJECTION_SITES):
                fig_normdapc_by_block(c, site, outdir, label)
                blocks = c.blocks(site)
                if blocks:
                    fig_overlap_vs_k(c, site, blocks[-1], outdir, label)
            rows += _cell_rows(c, label, ra.ATTENTION_SITES)
            rows += _cell_rows(c, label, ra.POST_PROJECTION_SITES)
            print(f"[{key}/{concept}] figures -> {outdir.relative_to(REPO)}")

    if rows:
        cols = sorted({k for r in rows for k in r},
                      key=lambda s: (s not in ("model", "model_label", "concept",
                                               "site", "block", "method"), s))
        with CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {CSV.relative_to(REPO)} ({len(rows)} rows)")

    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / "cdet_normdapc_attention.tex").write_text(
        NOTE_NORM + table_normdapc(ra.ATTENTION_SITES, "attention") + "\n")
    (TEX / "cdet_normdapc_supplementary.tex").write_text(
        NOTE_NORM + table_normdapc(ra.POST_PROJECTION_SITES, "post-projection") + "\n")
    (TEX / "cdet_overlap_attention.tex").write_text(
        NOTE_OVL + table_overlap(ra.ATTENTION_SITES) + "\n")
    (TEX / "cdet_overlap_supplementary.tex").write_text(
        NOTE_OVL + table_overlap(ra.POST_PROJECTION_SITES) + "\n")
    for n in ("cdet_normdapc_attention", "cdet_normdapc_supplementary",
              "cdet_overlap_attention", "cdet_overlap_supplementary"):
        print(f"wrote {TEX / (n + '.tex')}")
    cov = FIG / "COVERAGE.md"
    cov.write_text(coverage_report() + "\n")
    print(f"wrote {cov.relative_to(REPO)}")


if __name__ == "__main__":
    main()
