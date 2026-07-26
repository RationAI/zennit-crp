"""Registers E2v2 (reviewer-ordered refinement of E2): layer-by-layer, how many
of the outlier tokens identified at EACH block end up significantly highlighted
in the FINAL input saliency map — per XAI method, per model+dataset.

Definitions (verbatim protocol; journal copy in
``research/registers/e2v2_layer_overlap.md``):

* ``h_b(t)`` = residual-stream state of patch token t at the OUTPUT of block b
  (forward hook on ``blocks[b]``), b = 0..11. CLS excluded; T = 196 patches.
* Per-image per-block: ``mu_b(x), sigma_b(x)`` = mean/std of
  ``{||h_b(t)||_2 : t = 1..T}``.
* Block-b activation outlier set ``A_b(x) = {t : ||h_b(t)||_2 > mu_b + 4*sigma_b}``.
* Saliency maps: the four E2 methods (LRP cp_lrp_baseline / Chefer / attention
  rollout / occlusion Δp⁺), REUSED from ``e2_overlap_<model>.npz`` (per-patch
  values ``patch_<m>`` = sum of |s| over each 16x16 pixel patch, and masks
  ``S_<m>`` = same per-image mu+4*sd rule on the 196 per-patch values).
  Class conditioning: predicted class = true class (all E2 images are correctly
  classified); rollout is class-agnostic by nature.
* Reported per (model, method m, block b): ``n_b = sum_x |A_b(x)|``,
  ``c_bm = sum_x |A_b(x) ∩ S_m(x)|``, ``q_bm = c_bm / n_b`` (where n_b > 0).

Stages (norms needs GPU + the shared lock; the rest is CPU)::

    P=/home/claude/venvs/zennit-crp/bin/python
    $P -m experiments.scripts.registers_e2v2_layer_overlap norms   --model vit_base_imagenet
    $P -m experiments.scripts.registers_e2v2_layer_overlap norms   --model vit_small_funny_birds
    $P -m experiments.scripts.registers_e2v2_layer_overlap analyze --model vit_base_imagenet
    $P -m experiments.scripts.registers_e2v2_layer_overlap analyze --model vit_small_funny_birds
    $P -m experiments.scripts.registers_e2v2_layer_overlap figures --model vit_base_imagenet --copy-paper
    $P -m experiments.scripts.registers_e2v2_layer_overlap figures --model vit_small_funny_birds --copy-paper

Outputs:
``data/results/registers/e2v2_layer_norms_<model>.npz`` (fp32 block-output
norms), ``e2v2_layer_overlap_<model>.npz`` (A_b/S_m sets + per-image counts),
``e2v2_layer_table_<model>.csv`` (flat model,block,method,n_b,c_bm,q_bm),
figures ``figures/registers/e2v2_layer_overlap/e2v2_layer_overlap_<model>.{pdf,png}``
(+ copies to the paper journal-figures dir with --copy-paper).
All stages are idempotent (recompute + overwrite).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import typer

from experiments.scripts.registers_e2_overlap import (
    METHODS, METHOD_COLORS, METHOD_LABELS, N_PATCH, SD_K,
    _load_model_ds, overlap_path, select_path)
from experiments.scripts.registers_step1c_redo import MODELS

REPO = Path(__file__).resolve().parents[2]
RES_DIR = REPO / "data" / "results" / "registers"
FIG_DIR = REPO / "figures" / "registers" / "e2v2_layer_overlap"
PAPER_FIG_DIR = Path("/home/claude/workspaces/crp-paper/iclr2026/journal-figures")

COL_TOTAL = "#d4d4d4"          # neutral light bar: n_b (total identified)

app = typer.Typer(add_completion=False, help=__doc__)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def norms_path(m): return RES_DIR / f"e2v2_layer_norms_{m}.npz"
def out_path(m): return RES_DIR / f"e2v2_layer_overlap_{m}.npz"
def csv_path(m): return RES_DIR / f"e2v2_layer_table_{m}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# stage 1: norms — fp32 block-output token norms for the E2 image selection
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def norms(model: str = typer.Option(...), batch: int = typer.Option(32),
          device: str = typer.Option("cuda")):
    """One forward sweep over the 64 E2 images; forward hook on ``blocks[b]``
    records ||h_b(t)||_2 in float32 for every block output b = 0..11."""
    sel = np.load(select_path(model), allow_pickle=True)
    idxs = sel["ds_indices"].tolist()
    mdl, normalize, ds, label = _load_model_ds(model, device)
    blocks = mdl.backbone.blocks
    store: Dict[int, torch.Tensor] = {}
    handles = []
    for b, blk in enumerate(blocks):
        def post(mod, args, out, b=b):
            store[b] = out.detach().float().norm(dim=-1).cpu()   # (B, 197)
        handles.append(blk.register_forward_hook(post))

    per_batch: List[torch.Tensor] = []
    n_wrong = 0
    with torch.no_grad():
        for s in range(0, len(idxs), batch):
            x = torch.stack([ds[i][0] for i in idxs[s:s + batch]]).to(device)
            logits = mdl(normalize(x)).cpu()
            n_wrong += int((logits.argmax(-1) !=
                            torch.as_tensor(sel["targets"][s:s + batch])).sum())
            per_batch.append(torch.stack([store[b] for b in range(len(blocks))]))
    for h in handles:
        h.remove()
    if n_wrong:                       # E2 selection = correctly classified only
        raise RuntimeError(f"{n_wrong} images no longer correctly classified")

    arr = torch.cat(per_batch, dim=1).numpy().astype(np.float32)  # (12, 64, 197)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        norms_path(model), norms=arr, ds_indices=sel["ds_indices"],
        targets=sel["targets"],
        meta=np.array([
            f"model={label}", "h_b = blocks[b] forward-hook OUTPUT (after-mlp "
            "residual add), fp32 token L2 norms, token0=CLS",
            "images = E2 selection (e2_select ds_indices, all correctly classified)",
            f"collected={_now()}"]))
    print(f"saved {norms_path(model)} {arr.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# stage 2: analyze (CPU) — A_b sets, reuse S_m, counts + CSV
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def analyze(model: str = typer.Option(...)):
    """A_b via per-image mu+4sd on fp32 block-output norms; S_m reused from
    e2_overlap npz (identity re-checked against patch values); writes
    e2v2_layer_overlap_<model>.npz + e2v2_layer_table_<model>.csv."""
    nz = np.load(norms_path(model), allow_pickle=True)
    ov = np.load(overlap_path(model), allow_pickle=True)
    assert (nz["ds_indices"] == ov["ds_indices"]).all()
    patch_norms = nz["norms"][:, :, 1:]                       # (12, N, 196), CLS out
    n_blocks, n_img, _ = patch_norms.shape
    mu = patch_norms.mean(-1, keepdims=True)
    sd = patch_norms.std(-1, keepdims=True)
    A_b = patch_norms > mu + SD_K * sd                        # (12, N, 196) bool

    # sanity vs the fp16 E2 site flags (site 2b+1 = same block output)
    if select_path(model).exists():
        sf = np.load(select_path(model), allow_pickle=True)["site_flags"]
        mism = int((A_b != sf[1::2]).sum())
        print(f"fp32 vs stored-fp16 flag mismatches: {mism} "
              f"of {A_b.size} ({mism / A_b.size:.2e})")

    S = {}
    for m in METHODS:
        S[m] = ov[f"S_{m}"]                                   # (N, 196) bool
        p = ov[f"patch_{m}"]
        mu_p = p.mean(-1, keepdims=True)
        sd_p = p.std(-1, keepdims=True)
        assert (S[m] == (p > mu_p + SD_K * sd_p)).all(), f"S_{m} rule drifted"

    nb_img = A_b.sum(-1)                                      # (12, N)
    c_img = np.stack([(A_b & S[m][None]).sum(-1) for m in METHODS], 1)  # (12, 4, N)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path(model), ds_indices=nz["ds_indices"], targets=nz["targets"],
        A_b=A_b, nb_img=nb_img.astype(np.int16),
        c_img=c_img.astype(np.int16), methods=np.array(METHODS),
        **{f"S_{m}": S[m] for m in METHODS},
        meta=np.array([
            "A_b = per-image mu+4sd on fp32 ||h_b(t)|| (blocks[b] output), CLS excluded",
            "S_m = E2 saliency mask (same mu+4sd rule on 196 per-patch values), reused",
            "nb_img (12,N)=|A_b(x)|; c_img (12,4,N)=|A_b ∩ S_m| in METHODS order",
            f"analyzed={_now()}"]))
    print(f"saved {out_path(model)}")

    with open(csv_path(model), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "block", "method", "n_b", "c_bm", "q_bm"])
        for b in range(n_blocks):
            n_b = int(nb_img[b].sum())
            for j, m in enumerate(METHODS):
                c = int(c_img[b, j].sum())
                q = c / n_b if n_b > 0 else float("nan")
                w.writerow([model, b, m, n_b, c, f"{q:.4f}" if n_b else ""])
    print(f"saved {csv_path(model)}")

    print(f"{'block':>5} {'n_b':>5} " + " ".join(f"{m:>10}" for m in METHODS))
    for b in range(n_blocks):
        n_b = int(nb_img[b].sum())
        cs = [int(c_img[b, j].sum()) for j in range(len(METHODS))]
        print(f"{b:>5} {n_b:>5} " + " ".join(f"{c:>4} ({c / n_b:.2f})" if n_b
                                             else f"{'—':>10}" for c in cs))


# ─────────────────────────────────────────────────────────────────────────────
# stage 3: figures (CPU)
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def figures(model: str = typer.Option(...),
            copy_paper: bool = typer.Option(False, "--copy-paper")):
    """Rows = blocks 0..11 (top→bottom), columns = 4 methods; per cell a
    horizontal paired bar: light n_b + method-colored c_bm overlay, annotated
    'c/n'; shared x-scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(out_path(model), allow_pickle=True)
    nb = d["nb_img"].sum(-1)                                  # (12,)
    c = d["c_img"].sum(-1)                                    # (12, 4)
    n_blocks = nb.shape[0]
    y = np.arange(n_blocks)
    xmax = nb.max() * 1.30

    fig, axes = plt.subplots(1, len(METHODS),
                             figsize=(11.5, 4.4), sharey=True)
    for j, (ax, m) in enumerate(zip(axes, METHODS)):
        ax.barh(y, nb, height=0.72, color=COL_TOTAL, zorder=2,
                label=r"$n_b$ (identified at block $b$)")
        ax.barh(y, c[:, j], height=0.72, color=METHOD_COLORS[m], zorder=3,
                label=r"$c_{b,m}$ (in final saliency mask)")
        for b in range(n_blocks):
            ax.text(nb[b] + 0.012 * xmax, b, f"{c[b, j]}/{nb[b]}",
                    va="center", ha="left", fontsize=8, color="#333333")
        ax.set_title(METHOD_LABELS[m], fontsize=9.5,
                     color="#222222", pad=6)
        ax.set_xlim(0, xmax)
        ax.set_ylim(n_blocks - 0.5, -0.5)                     # block 0 on top
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("outlier tokens (sum over 64 images)", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if j == 0:
            ax.set_yticks(y)
            ax.set_ylabel("block $b$ (outlier set $A_b$ at block output)",
                          fontsize=9)
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.98),
                      fontsize=8, frameon=False)

    label = MODELS[model]["label"]
    fig.suptitle(
        f"E2v2 — {label}: block-$b$ activation outliers ending up in the final "
        f"input-saliency mask, per method\n"
        r"$A_b$: $\|h_b(t)\|_2 > \mu_b+4\sigma_b$ over the image's 196 patch "
        r"tokens at the output of block $b$;  $S_m$: same $\mu+4\sigma$ rule on "
        r"the method's 196 per-patch saliency values", fontsize=10)
    fig.text(0.5, 0.005,
             "Annotation per bar: $c_{b,m}/n_b$ summed over N=64 correctly-classified images "
             "(prediction = label, so class-conditioning w.r.t. the predicted class). "
             "LRP = cp_lrp_baseline (CondAttribution, cond {y:[$\\hat{y}$]}, sum|R| per 16$\\times$16 patch); "
             "Chefer = grad-weighted attention rollout; attention rollout is class-agnostic by nature; "
             "occlusion = patch$\\to$image-mean, $\\Delta$p($\\hat{y}$) clamped at 0. "
             "CLS excluded everywhere; $h_b$ = forward hook on blocks[$b$] output.",
             fontsize=7, ha="center", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"e2v2_layer_overlap_{model}"
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"fig {FIG_DIR / stem}.png/.pdf")
    if copy_paper:
        import shutil
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            shutil.copy2(FIG_DIR / f"{stem}.{ext}", PAPER_FIG_DIR / f"{stem}.{ext}")
            print(f"copied {PAPER_FIG_DIR / f'{stem}.{ext}'}")


if __name__ == "__main__":
    app()
