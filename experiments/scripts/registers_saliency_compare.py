"""Registers step 4 (XAI-38, part of XAI-34): do non-LRP saliency methods
hotspot the same high-norm outlier ("register") patches as LRP?

Darcet et al. (arXiv:2309.16588) show that large ViTs recycle a few
low-information patch tokens as global scratchpads ("registers"); those tokens
have outlier L2 norms in the later blocks. Our LRP/CRP relevance maps on
ViT-B/16 show hotspots on visually unremarkable patches — this script tests
whether NON-LRP saliency methods place mass on the same outlier patches. If
they all do, the artifact is model-side; if only LRP does, it is
propagation-rule-specific.

Protocol (ViT-B/16 timm ``vit_base_patch16_224``, ImageNet val subset
``n_per_class=10``, un-normalized [0,1] images + canonical normalize at the
forward boundary):

1. ``detect`` — forward hooks on every ``backbone.blocks[b]`` output record the
   L2 norm of each of the 196 patch tokens (CLS excluded, 14x14 grid). A token
   is an *outlier at block b* iff ``norm > mean + sigma*sd`` over that image's
   196 patch norms at that block (per-image, per-block criterion; sigma=4 by
   default). The image's outlier MASK is the union of the per-block flags over
   ``--mask-start..--mask-end`` (default 6..11 — high-norm outliers emerge in
   the mid/late blocks, see the printed per-block stats). Scans ``--n-scan``
   images in a seeded random order and selects the first ``--n-select``
   correctly-classified images with a non-empty mask.

2. ``saliency`` — per selected image, five patch-aggregated maps (sum of
   absolute values per 16x16 patch where the method is pixel-level):
   * ``gxi``     — gradient x input at the true-class logit;
   * ``ig``      — integrated gradients, 32 midpoint steps, black baseline;
   * ``rollout`` — attention rollout (Abnar & Zuidema): per-block attention
                   averaged over heads (captured via ``attn_drop`` hooks with
                   ``fused_attn=False`` — plain forward, stock timm module),
                   0.5*A + 0.5*I per block, row-renormalized, chained; CLS row;
   * ``attn``    — raw last-block CLS attention row, averaged over heads;
   * ``lrp``     — reference CP-LRP heatmap (``CondAttribution`` +
                   ``CPLRPComposite`` (cp_lrp_baseline), condition
                   ``[{"y": [target]}]``), |R| summed per patch.

3. ``analyze`` (CPU, no GPU needed) — colocation metrics per method:
   * concentration ratio = (saliency mass inside mask / total) / (mask area /
     196); 1.0 = chance;
   * mean rank of the outlier patches in the method's descending patch ranking
     (1 = most salient; chance = 98.5);
   * fraction of images with >=1 outlier patch in the method's top-5 (reported
     next to the per-image analytic chance level).
   Plus qualitative 8-image panels and a summary figure (png+pdf).

Run (GPU stages under the shared lock):

    /home/claude/venvs/zennit-crp/bin/python -m experiments.scripts.registers_saliency_compare detect --n-scan 256 --probe-only
    /home/claude/venvs/zennit-crp/bin/python -m experiments.scripts.registers_saliency_compare detect
    /home/claude/venvs/zennit-crp/bin/python -m experiments.scripts.registers_saliency_compare saliency
    /home/claude/venvs/zennit-crp/bin/python -m experiments.scripts.registers_saliency_compare analyze
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

REPO = Path(__file__).resolve().parents[2]
RES_DIR = REPO / "data" / "results" / "registers"
FIG_DIR = REPO / "figures" / "registers" / "step4_methods"
SELECT_NPZ = RES_DIR / "step4_selection.npz"
SCAN_STATS = RES_DIR / "step4_scan_stats.json"
SALIENCY_NPZ = RES_DIR / "step4_saliency.npz"
METRICS_JSON = RES_DIR / "step4_metrics.json"

GRID = 14           # 14x14 patch grid (ViT-B/16 @ 224)
PATCH = 16
N_PATCH = GRID * GRID
METHODS = ("gxi", "ig", "rollout", "attn", "lrp")
METHOD_LABELS = {
    "gxi": "gradient x input", "ig": "integrated gradients (32)",
    "rollout": "attention rollout", "attn": "last-block CLS attention",
    "lrp": "LRP (cp_lrp_baseline)",
}

app = typer.Typer(add_completion=False, help=__doc__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_ds(device: str):
    from experiments.crp_gallery import load_model, load_eval_dataset
    from experiments.model_io import backbone_transforms
    model, _, _, _ = load_model(
        "vit_base", "imagenet", model_source="checkpoint", checkpoint=None,
        head="linear", num_classes=None, head_kwargs={}, device=device)
    transform, normalize = backbone_transforms(model.backbone)
    ds = load_eval_dataset("imagenet", transform, {"n_per_class": 10})
    return model, normalize, ds


def _load_ds_only():
    """Dataset without instantiating the model on GPU (analyze stage)."""
    import timm
    from experiments.model_io import IMAGENET_TIMM
    from experiments.models import backbone_transforms
    from experiments.crp_gallery import load_eval_dataset
    tm = timm.create_model(IMAGENET_TIMM, pretrained=True, num_classes=1000)
    transform, _ = backbone_transforms(tm)
    return load_eval_dataset("imagenet", transform, {"n_per_class": 10})


def _to_patch(sal: torch.Tensor) -> torch.Tensor:
    """(B, 224, 224) pixel map -> (B, 196) per-patch sum of |values|."""
    b = sal.shape[0]
    return sal.abs().reshape(b, GRID, PATCH, GRID, PATCH).sum(dim=(2, 4)).reshape(b, N_PATCH)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: outlier detection + image selection
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def detect(
    n_scan: int = typer.Option(2048, help="images to scan (seeded random order)"),
    n_select: int = typer.Option(64, help="correctly-classified outlier images to keep"),
    sigma: float = typer.Option(4.0, help="outlier criterion: norm > mean + sigma*sd"),
    mask_start: int = typer.Option(6, help="first block of the mask union"),
    mask_end: int = typer.Option(11, help="last block of the mask union (inclusive)"),
    seed: int = typer.Option(0),
    batch: int = typer.Option(32),
    probe_only: bool = typer.Option(False, "--probe-only", help="print stats, save nothing"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Scan → per-block token norms → per-image outlier flags → select images."""
    model, normalize, ds = _load_model_ds(device)
    n_blocks = len(model.backbone.blocks)
    assert int(model.backbone.num_prefix_tokens) == 1, "expected exactly one CLS prefix token"

    store: Dict[int, torch.Tensor] = {}
    hooks = []
    for b, blk in enumerate(model.backbone.blocks):
        def f(m, i, o, b=b):
            store[b] = o.detach().float().norm(dim=-1)[:, 1:].cpu()   # (B, 196), CLS dropped
        hooks.append(blk.register_forward_hook(f))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g)[:n_scan].tolist()

    all_norms, all_flags, all_correct, all_targets = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(perm), batch):
            idxs = perm[s:s + batch]
            xs, ys = zip(*[ds[i] for i in idxs])
            x = torch.stack(xs).to(device)
            y = torch.tensor(ys)
            pred = model(normalize(x)).argmax(-1).cpu()
            norms = torch.stack([store[b] for b in range(n_blocks)], dim=1)  # (B, 12, 196)
            mu = norms.mean(-1, keepdim=True)
            sd = norms.std(-1, keepdim=True)
            flags = norms > mu + sigma * sd                                   # (B, 12, 196)
            all_norms.append(norms.half())
            all_flags.append(flags)
            all_correct.append(pred == y)
            all_targets.append(y)
    for h in hooks:
        h.remove()

    norms = torch.cat(all_norms).float()          # (n, 12, 196)
    flags = torch.cat(all_flags)                  # (n, 12, 196)
    correct = torch.cat(all_correct)
    targets = torch.cat(all_targets)

    print(f"scanned {len(perm)} images · top-1 acc {correct.float().mean():.3f} · sigma={sigma}")
    print(f"{'block':>5} {'mean#outl/img':>13} {'frac imgs>=1':>13} {'max norm':>9} {'mean norm':>10}")
    for b in range(n_blocks):
        fb = flags[:, b]
        print(f"{b:>5} {fb.sum(-1).float().mean():13.2f} "
              f"{(fb.any(-1)).float().mean():13.3f} {norms[:, b].max():9.1f} "
              f"{norms[:, b].mean():10.1f}")

    mask = flags[:, mask_start:mask_end + 1].any(dim=1)                       # (n, 196)
    has = mask.any(-1)
    print(f"mask = union blocks {mask_start}..{mask_end}: "
          f"frac imgs with >=1 outlier {has.float().mean():.3f} · "
          f"mean mask size {mask.sum(-1).float().mean():.2f} patches · "
          f"among correct: {(has & correct).float().sum() / correct.float().sum():.3f}")
    if probe_only:
        return

    qual = (correct & has).nonzero().flatten().tolist()
    if len(qual) < n_select:
        raise RuntimeError(f"only {len(qual)} qualifying images (< {n_select}); raise --n-scan")
    keep = qual[:n_select]
    sel_ds = [perm[i] for i in keep]
    RES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SELECT_NPZ,
        ds_indices=np.array(sel_ds), targets=targets[keep].numpy(),
        masks=mask[keep].numpy(), flags=flags[keep].numpy(),
        norms=norms[keep].numpy().astype(np.float16),
        sigma=sigma, mask_start=mask_start, mask_end=mask_end, seed=seed)
    stats = {
        "n_scan": len(perm), "sigma": sigma, "seed": seed,
        "mask_blocks": [mask_start, mask_end],
        "top1_acc": float(correct.float().mean()),
        "frac_images_with_outlier": float(has.float().mean()),
        "frac_correct_with_outlier": float((has & correct).float().sum() / correct.float().sum()),
        "mean_mask_patches": float(mask.sum(-1).float().mean()),
        "per_block_mean_outliers": [float(flags[:, b].sum(-1).float().mean()) for b in range(n_blocks)],
        "per_block_frac_images": [float(flags[:, b].any(-1).float().mean()) for b in range(n_blocks)],
        "n_selected": len(keep),
        "selected_mean_mask_patches": float(mask[keep].sum(-1).float().mean()),
    }
    SCAN_STATS.write_text(json.dumps(stats, indent=2))
    print(f"selected {len(keep)} images → {SELECT_NPZ}\nstats → {SCAN_STATS}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: the five saliency methods
# ─────────────────────────────────────────────────────────────────────────────

def _capture_attention(model, xn: torch.Tensor) -> List[torch.Tensor]:
    """Plain forward with ``fused_attn=False``; hooks on each ``attn_drop``
    return the post-softmax attention (B, heads, 197, 197) per block. Outside
    any composite context this is the stock timm module (the unfolded-attention
    substitution only exists inside ``composite.context``)."""
    store: Dict[int, torch.Tensor] = {}
    hooks, prev = [], []
    for b, blk in enumerate(model.backbone.blocks):
        prev.append(blk.attn.fused_attn)
        blk.attn.fused_attn = False

        def f(m, i, o, b=b):
            store[b] = o.detach().float().cpu()
        hooks.append(blk.attn.attn_drop.register_forward_hook(f))
    try:
        with torch.no_grad():
            model(xn)
    finally:
        for h in hooks:
            h.remove()
        for blk, p in zip(model.backbone.blocks, prev):
            blk.attn.fused_attn = p
    return [store[b] for b in range(len(model.backbone.blocks))]


def _rollout(attns: List[torch.Tensor]) -> torch.Tensor:
    """Abnar & Zuidema attention rollout → CLS row over patches (B, 196)."""
    r = None
    for a in attns:
        a = a.mean(dim=1)                                   # heads → (B, N, N)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1]).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True)                 # renormalize rows
        r = a if r is None else a @ r
    return r[:, 0, 1:]                                      # CLS attends-to row, drop CLS col


@app.command()
def saliency(
    ig_steps: int = typer.Option(32),
    batch: int = typer.Option(8),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Compute the five patch-aggregated saliency maps for the selected images."""
    from zennit_extensions.lrp_composites import CPLRPComposite
    from crp.attribution import CondAttribution

    sel = np.load(SELECT_NPZ)
    ds_indices, targets = sel["ds_indices"].tolist(), sel["targets"].tolist()
    model, normalize, ds = _load_model_ds(device)
    attribution = CondAttribution(model)
    composite_cls = CPLRPComposite

    out = {m: [] for m in METHODS}
    pix = {m: [] for m in ("gxi", "ig", "lrp")}             # keep pixel maps for panels
    for s in range(0, len(ds_indices), batch):
        idxs = ds_indices[s:s + batch]
        tg = targets[s:s + batch]
        x = torch.stack([ds[i][0] for i in idxs]).to(device)
        xn = normalize(x)
        bsz = x.shape[0]

        # attention rollout + raw last-block CLS attention (plain forward)
        attns = _capture_attention(model, xn)
        out["rollout"].append(_rollout(attns))
        out["attn"].append(attns[-1].mean(dim=1)[:, 0, 1:])

        # gradient x input (w.r.t. the normalized model input)
        xg = xn.clone().requires_grad_(True)
        logit = model(xg)[range(bsz), tg].sum()
        (grad,) = torch.autograd.grad(logit, xg)
        gxi = (grad * xn).sum(dim=1).detach().cpu()          # signed pixel map
        out["gxi"].append(_to_patch(gxi))
        pix["gxi"].append(gxi)

        # integrated gradients: black baseline (zeros in [0,1] space, normalized)
        xb = normalize(torch.zeros_like(x))
        acc = torch.zeros_like(xn)
        for k in range(ig_steps):
            a = (k + 0.5) / ig_steps                          # midpoint rule
            xi = (xb + a * (xn - xb)).requires_grad_(True)
            logit = model(xi)[range(bsz), tg].sum()
            (grad,) = torch.autograd.grad(logit, xi)
            acc += grad.detach()
        ig = ((xn - xb) * acc / ig_steps).sum(dim=1).detach().cpu()
        out["ig"].append(_to_patch(ig))
        pix["ig"].append(ig)

        # reference LRP heatmap (class-conditional, cp_lrp_baseline)
        xl = xn.clone().requires_grad_(True)
        conds = [{"y": [int(t)]} for t in tg]
        res = attribution(xl, conds, composite_cls())
        lrp = res.heatmap.detach().cpu()                      # (B, 224, 224) signed
        out["lrp"].append(_to_patch(lrp))
        pix["lrp"].append(lrp)
        print(f"batch {s // batch + 1}/{(len(ds_indices) + batch - 1) // batch} done")

    RES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SALIENCY_NPZ,
        ds_indices=np.array(ds_indices), targets=np.array(targets),
        masks=sel["masks"],
        **{f"patch_{m}": torch.cat(out[m]).numpy() for m in METHODS},
        **{f"pix_{m}": torch.cat(pix[m]).numpy().astype(np.float16) for m in pix},
        ig_steps=ig_steps)
    print(f"saliency → {SALIENCY_NPZ}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: metrics + figures (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(p: np.ndarray, masks: np.ndarray) -> dict:
    """Colocation of one method's patch maps (n, 196) with the outlier masks."""
    n = p.shape[0]
    conc, mrank, top5, mass = [], [], [], []
    order = np.argsort(-p, axis=1)                          # descending saliency
    ranks = np.empty_like(order)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(1, N_PATCH + 1)[None, :]  # 1-based rank per patch
    for i in range(n):
        m = masks[i].astype(bool)
        tot = p[i].sum()
        frac_mass = float(p[i][m].sum() / tot) if tot > 0 else 0.0
        area = m.sum() / N_PATCH
        conc.append(frac_mass / area)
        mass.append(frac_mass)
        mrank.append(float(ranks[i][m].mean()))
        top5.append(bool((ranks[i][m] <= 5).any()))
    return {
        "concentration_mean": float(np.mean(conc)), "concentration_median": float(np.median(conc)),
        "concentration_sd": float(np.std(conc)),
        "mass_in_mask_mean": float(np.mean(mass)),
        "mean_rank": float(np.mean(mrank)), "mean_rank_sd": float(np.std(mrank)),
        "top5_frac": float(np.mean(top5)),
        "per_image_concentration": [float(v) for v in conc],
        "per_image_mean_rank": [float(v) for v in mrank],
        "per_image_top5": [bool(v) for v in top5],
    }


def _chance_top5(masks: np.ndarray) -> float:
    """Mean over images of P(>=1 of k outlier patches in a random top-5)."""
    out = []
    for m in masks:
        k = int(m.sum())
        p_miss = 1.0
        for j in range(5):
            p_miss *= (N_PATCH - k - j) / (N_PATCH - j)
        out.append(1.0 - p_miss)
    return float(np.mean(out))


def _patch_img(p: np.ndarray) -> np.ndarray:
    return p.reshape(GRID, GRID)


@app.command()
def analyze(n_panel: int = typer.Option(8)):
    """Colocation metrics, verdict numbers, qualitative panels + summary figure."""
    d = np.load(SALIENCY_NPZ)
    masks = d["masks"]
    metrics = {m: _metrics(d[f"patch_{m}"], masks) for m in METHODS}
    chance5 = _chance_top5(masks)
    mask_sizes = masks.sum(1)

    print(f"n={masks.shape[0]} images · mean mask {mask_sizes.mean():.2f} patches "
          f"(min {mask_sizes.min()}, max {mask_sizes.max()}) · top-5 chance {chance5:.3f}")
    print(f"{'method':>26} {'conc.ratio':>10} {'median':>8} {'mass%':>7} {'meanrank':>9} {'top5':>6}")
    for m in METHODS:
        r = metrics[m]
        print(f"{METHOD_LABELS[m]:>26} {r['concentration_mean']:10.2f} "
              f"{r['concentration_median']:8.2f} {100 * r['mass_in_mask_mean']:7.1f} "
              f"{r['mean_rank']:9.1f} {r['top5_frac']:6.2f}")

    METRICS_JSON.write_text(json.dumps(
        {"chance_top5": chance5, "chance_mean_rank": (N_PATCH + 1) / 2,
         "mask_sizes": [int(v) for v in mask_sizes],
         "methods": metrics}, indent=2))
    print(f"metrics → {METRICS_JSON}")

    # ── qualitative panels ────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ds = _load_ds_only()
    idxs = list(range(min(n_panel, masks.shape[0])))
    ncols = 2 + len(METHODS)
    fig, axes = plt.subplots(len(idxs), ncols, figsize=(2.1 * ncols, 2.15 * len(idxs)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idxs):
        x = ds[int(d["ds_indices"][i])][0].permute(1, 2, 0).numpy()
        m2 = _patch_img(masks[i].astype(float))
        axes[r, 0].imshow(x)
        axes[r, 0].set_ylabel(f"img {i}\nclass {int(d['targets'][i])}", fontsize=7)
        over = x.copy()
        big = np.kron(m2, np.ones((PATCH, PATCH)))[..., None]
        over = np.clip(over * (1 - 0.6 * big) + 0.6 * big * np.array([1.0, 0.1, 0.1]), 0, 1)
        axes[r, 1].imshow(over)
        for c, meth in enumerate(METHODS):
            p = _patch_img(d[f"patch_{meth}"][i])
            axes[r, 2 + c].imshow(p, cmap="viridis")
            for (yy, xx) in zip(*np.where(m2 > 0)):
                axes[r, 2 + c].add_patch(plt.Rectangle(
                    (xx - 0.5, yy - 0.5), 1, 1, fill=False, edgecolor="red", lw=1.2))
            axes[r, 2 + c].set_title(
                f"cr={metrics[meth]['per_image_concentration'][i]:.1f}", fontsize=7, pad=2)
        if r == 0:
            axes[r, 0].set_title("input", fontsize=8)
            axes[r, 1].set_title("outlier mask", fontsize=8)
            for c, meth in enumerate(METHODS):
                axes[r, 2 + c].set_title(
                    METHOD_LABELS[meth] + f"\ncr={metrics[meth]['per_image_concentration'][i]:.1f}",
                    fontsize=7, pad=2)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Register/outlier patches vs saliency methods — ViT-B/16, ImageNet val\n"
                 "red boxes = high-norm outlier patches (union blocks "
                 f"{int(np.load(SELECT_NPZ)['mask_start'])}–{int(np.load(SELECT_NPZ)['mask_end'])}, "
                 "mean+4sd) · patch maps = per-16x16-patch |saliency| · cr = concentration ratio "
                 "(1 = chance)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(FIG_DIR / "qualitative_panel.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "qualitative_panel.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── summary figure ────────────────────────────────────────────────────
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 3.6))
    xs = np.arange(len(METHODS))
    labels = ["grad x input", "IG (32)", "attn rollout", "last-block\nCLS attn", "LRP\n(cp_lrp_baseline)"]
    conc = [metrics[m]["concentration_mean"] for m in METHODS]
    for ax, key, base, title in (
            (a1, "per_image_concentration", 1.0, "concentration ratio (log; 1 = chance)"),
            (a2, "per_image_mean_rank", (N_PATCH + 1) / 2, "mean rank of outlier patches\n(lower = more salient)")):
        vals = [metrics[m][key] for m in METHODS]
        means = [np.mean(v) for v in vals]
        ax.bar(xs, means, color="#4878a8", zorder=2)
        for j, v in enumerate(vals):
            ax.scatter(np.full(len(v), xs[j]) + np.random.default_rng(0).uniform(-0.18, 0.18, len(v)),
                       v, s=6, color="#222222", alpha=0.35, zorder=3)
        ax.axhline(base, color="#b04030", ls="--", lw=1, label="chance")
        ax.set_xticks(xs, labels, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    a1.set_yscale("log")
    a3.bar(xs, [metrics[m]["top5_frac"] for m in METHODS], color="#4878a8", zorder=2)
    a3.axhline(chance5, color="#b04030", ls="--", lw=1, label=f"chance {chance5:.2f}")
    a3.set_xticks(xs, labels, fontsize=7)
    a3.set_ylim(0, 1.05)
    a3.set_title("frac. images with >=1 outlier patch in top-5", fontsize=9)
    a3.legend(fontsize=7)
    fig.suptitle(f"Saliency colocation with high-norm outlier patches — "
                 f"ViT-B/16 ImageNet, n={masks.shape[0]} images", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "summary_metrics.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "summary_metrics.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"figures → {FIG_DIR}")


if __name__ == "__main__":
    app()
