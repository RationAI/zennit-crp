"""Registers step 1c (XAI-35): REDO of outlier-token detection per reviewer.

Changes vs step 1 (research/registers/step1_detect.md), as ordered:

1. **Per-sample statistics** — mu_b and sigma_b are computed from the patch
   tokens of a SINGLE sample at block b (196 tokens, CLS excluded), so the
   threshold ``tau_b(sample) = mu_b(sample) + 4*sd_b(sample)`` is sensitive to
   local out-of-distribution deviation (old: population stats over all N
   images).
2. **Single-block flagging** — a token is an outlier if flagged at ANY single
   block (old: consensus >= 3 of blocks 6..11). Per-block flags are kept for
   the per-block figures.
3. **Bimodality figure** — per model, histogram grid at representative blocks
   with the PER-SAMPLE tau distribution (median + IQR band), log-x.
4. **Outlier-fraction-per-block figure** regenerated WITHOUT any horizontal
   threshold line (the old figure carried a duplicated 2% line).
5. **Both inspection sites** — token norms recorded at (a) ``blocks[i]``
   output (residual stream, as before) and (b) ``blocks[i].attn.proj_drop``
   output (attention output, before the residual add); the per-sample
   criterion and all detection stats are reported per site.

Models / samples (identical selection scheme to step 1, seed 0, indices
persisted): ViT-B/16 timm ImageNet val (n_per_class=10) N=256; ViT-S/16
FunnyBirds probe, TEST split, N=256.

Run (collect needs the GPU; everything else is CPU)::

    python -m experiments.scripts.registers_step1c_redo collect --model vit_base_imagenet
    python -m experiments.scripts.registers_step1c_redo collect --model vit_small_funny_birds
    python -m experiments.scripts.registers_step1c_redo analyze
    python -m experiments.scripts.registers_step1c_redo figures
    python -m experiments.scripts.registers_step1c_redo colocation

Outputs: ``data/results/registers/step1c_*.npz``,
``figures/registers/step1c_redo/`` (png+pdf),
report ``research/registers/step1c_redo.md`` (written by `analyze`+`colocation`).
The colocation figure pages are rendered by ``registers_step1_figures.py``
via its ``--colocation-npz`` flag (same layout/labels as the step-1 originals).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "results" / "registers"
FIG_DIR = REPO_ROOT / "figures" / "registers" / "step1c_redo"

CKPT_FUNNY = (REPO_ROOT / "data" / "runs"
              / "finetune_vit_small_funny-birds-train-clean" / "2026-06-03_000556" / "best.pt")

SD_K = 4.0                       # criterion: norm > mu_sample + 4*sd_sample
GRID = 14
N_PATCH = GRID * GRID            # 196; token 0 = CLS
SITES = ("residual", "proj_drop")
ONSET_FRAC = 0.005               # step-1 onset definition: first block with frac > 0.5%
JACCARD_BLOCKS = (6, 7, 8, 9)    # old-vs-new comparison blocks

MODELS = {
    "vit_base_imagenet": dict(base="vit_base", dataset="imagenet",
                              checkpoint=None, extra={"n_per_class": 10},
                              label="ViT-B/16 · ImageNet val"),
    "vit_small_funny_birds": dict(base="vit_small", dataset="funny_birds",
                                  checkpoint=str(CKPT_FUNNY), extra={"split": "test"},
                                  label="ViT-S/16 · FunnyBirds test"),
}
BIMODAL_BLOCKS = (2, 6, 9, 11)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def norms_path(model_key: str) -> Path:
    return OUT_DIR / f"step1c_norms_{model_key}.npz"


def masks_path(model_key: str) -> Path:
    return OUT_DIR / f"step1c_masks_{model_key}.npz"


# ─────────────────────────────────────────────────────────────────────────────
# criterion (the one thing this redo changes)
# ─────────────────────────────────────────────────────────────────────────────

def per_sample_flags(norms: np.ndarray):
    """NEW criterion. norms (12, N, 197) -> (masks (12, N, 196), union (N, 196),
    tau (12, N)). mu/sd over the 196 patch tokens of EACH sample at EACH block
    (CLS excluded); token flagged at block b iff norm > mu + 4*sd; image-level
    outlier = flagged at ANY single block."""
    patch = norms[:, :, 1:]                                    # (12, N, 196)
    mu = patch.mean(-1)                                        # (12, N)
    sd = patch.std(-1)
    tau = mu + SD_K * sd
    masks = patch > tau[:, :, None]                            # (12, N, 196)
    union = masks.any(0)                                       # (N, 196)
    return masks, union, tau


def population_flags(norms: np.ndarray):
    """OLD (step-1) criterion on the same norms: population mu_b/sd_b over ALL
    patch tokens of ALL samples; image-level = >= 3 of blocks 6..11."""
    patch = norms[:, :, 1:]
    mean_b = patch.reshape(12, -1).mean(1)
    sd_b = patch.reshape(12, -1).std(1)
    tau = mean_b + SD_K * sd_b
    masks = patch > tau[:, None, None]
    image_level = masks[6:12].sum(0) >= 3
    return masks, image_level, tau


# ─────────────────────────────────────────────────────────────────────────────
# collect (GPU) — forwards, per-block token norms at BOTH sites
# ─────────────────────────────────────────────────────────────────────────────

def pick_class_diverse(ds, n: int, seed: int = 0) -> List[int]:
    """Round-robin over classes (identical to step 1 / step 1b)."""
    labels = _ds_labels(ds)
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for i in rng.permutation(len(labels)):
        by_class.setdefault(labels[i], []).append(int(i))
    classes = sorted(by_class)
    out: List[int] = []
    while len(out) < n and any(by_class[c] for c in classes):
        for c in classes:
            if by_class[c]:
                out.append(by_class[c].pop(0))
                if len(out) >= n:
                    break
    return out


def _ds_labels(ds) -> List[int]:
    if hasattr(ds, "items"):
        return [int(c) for _, c in ds.items]
    if hasattr(ds, "rows"):
        return [int(c) for _, c in ds.rows]
    return [int(ds[i][1]) for i in range(len(ds))]


class TwoSiteNormRecorder:
    """Forward hooks recording L2 token norms at both inspection sites of every
    block: ``blocks[i]`` output (residual stream) and ``blocks[i].attn.proj_drop``
    output (attention output, pre residual-add)."""

    def __init__(self, backbone):
        self.norms: Dict[str, Dict[int, np.ndarray]] = {s: {} for s in SITES}
        self.handles = []
        for b, blk in enumerate(backbone.blocks):
            def hook_res(mod, args, out, b=b):
                self.norms["residual"][b] = out.detach().norm(dim=-1).float().cpu().numpy()

            def hook_proj(mod, args, out, b=b):
                self.norms["proj_drop"][b] = out.detach().norm(dim=-1).float().cpu().numpy()

            self.handles.append(blk.register_forward_hook(hook_res))
            self.handles.append(blk.attn.proj_drop.register_forward_hook(hook_proj))

    def stack(self, site: str) -> np.ndarray:                  # (12, B, 197)
        d = self.norms[site]
        return np.stack([d[b] for b in sorted(d)])

    def remove(self):
        for h in self.handles:
            h.remove()


def cmd_collect(args):
    import torch
    from experiments.models import MODELS as MODEL_ZOO, backbone_transforms
    from experiments.datasets import load_eval_dataset

    spec = MODELS[args.model]
    device = args.device
    ckpt = spec["checkpoint"]
    model = MODEL_ZOO[f"{spec['base']}_{spec['dataset']}"](
        **({"checkpoint": ckpt} if ckpt else {}), device=device)
    label = f"{spec['base']} · {model.head_name} · {spec['dataset']}"
    transform, normalize = backbone_transforms(model.backbone)
    ds = load_eval_dataset(spec["dataset"], transform, extra_kwargs=spec["extra"])

    indices = pick_class_diverse(ds, args.n, seed=args.seed)
    labels = np.array([_ds_labels(ds)[i] for i in indices])
    print(f"{label}: N={len(indices)} of {len(ds)}")

    rec = TwoSiteNormRecorder(model.backbone)
    chunks = {s: [] for s in SITES}
    with torch.no_grad():
        for s0 in range(0, len(indices), args.batch_size):
            batch = [ds[i][0] for i in indices[s0:s0 + args.batch_size]]
            x = normalize(torch.stack(batch).to(device))
            model(x)
            for s in SITES:
                chunks[s].append(rec.stack(s))
            print(f"  {s0 + len(batch)}/{len(indices)}", flush=True)
    rec.remove()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        norms_path(args.model),
        norms_residual=np.concatenate(chunks["residual"], axis=1).astype(np.float32),
        norms_proj_drop=np.concatenate(chunks["proj_drop"], axis=1).astype(np.float32),
        ds_indices=np.array(indices, dtype=np.int64),
        labels=labels,
        meta=np.array([
            f"model={label}", f"checkpoint={spec['checkpoint']}",
            "sites: residual = L2 token norm of blocks[i] output; "
            "proj_drop = L2 token norm of blocks[i].attn.proj_drop output (pre residual-add)",
            "token0=CLS (excluded from stats), tokens1..196=patches row-major 14x14",
            f"selection=round-robin class-diverse seed={args.seed}",
            f"dataset_kwargs={spec['extra']}",
            f"collected={_now()}",
        ]),
    )
    print(f"saved {norms_path(args.model)}")


# ─────────────────────────────────────────────────────────────────────────────
# analyze (CPU) — per-site stats, old-vs-new Jaccard
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def cmd_analyze(args):
    report = {}
    for model_key in MODELS:
        d = np.load(norms_path(model_key), allow_pickle=True)
        rep: dict = {"N": int(d["labels"].shape[0]), "sites": {}}
        out_npz: dict = dict(ds_indices=d["ds_indices"], labels=d["labels"])

        for site in SITES:
            norms = d[f"norms_{site}"]
            masks, union, tau = per_sample_flags(norms)
            frac = masks.mean(axis=(1, 2))                     # (12,)
            onset = next((b for b in range(12) if frac[b] > ONSET_FRAC), None)
            cnt = union.sum(1)
            rep["sites"][site] = dict(
                frac_per_block=[float(f) for f in frac],
                onset_block=onset,
                union_frac=float(union.mean()),
                per_image=dict(mean=float(cnt.mean()), median=float(np.median(cnt)),
                               min=int(cnt.min()), max=int(cnt.max()),
                               frac_images_flagged=float((cnt > 0).mean())),
            )
            out_npz.update({f"masks_{site}": masks, f"union_mask_{site}": union,
                            f"tau_{site}": tau.astype(np.float32),
                            f"frac_per_block_{site}": frac,
                            f"per_image_count_{site}": cnt.astype(np.int64)})

        # old-vs-new criterion overlap, residual site, same samples
        norms_res = d["norms_residual"]
        old_masks, old_img, old_tau = population_flags(norms_res)
        new_masks, new_union, _ = per_sample_flags(norms_res)
        jac = {b: _jaccard(new_masks[b], old_masks[b]) for b in JACCARD_BLOCKS}
        jac_img = _jaccard(new_union, old_img)
        rep["old_vs_new_residual"] = dict(
            jaccard_per_block={str(b): jac[b] for b in JACCARD_BLOCKS},
            jaccard_image_level_union_vs_consensus=jac_img,
            old_image_level_frac=float(old_img.mean()),
            new_union_frac=float(new_union.mean()),
        )
        out_npz.update(old_masks_residual=old_masks, old_image_level_residual=old_img,
                       old_tau_residual=old_tau.astype(np.float32))

        # cross-check against the stored step-1 arrays where the sample set matches
        old_path = OUT_DIR / f"outlier_masks_{model_key}.npz"
        if old_path.exists():
            o = np.load(old_path, allow_pickle=True)
            same = (o["ds_indices"].shape == d["ds_indices"].shape
                    and bool((o["ds_indices"] == d["ds_indices"]).all()))
            rep["old_vs_new_residual"]["same_samples_as_step1"] = same
            if same:
                rep["old_vs_new_residual"]["stored_old_mask_match"] = \
                    bool((o["image_level_mask"] == old_img).all())

        out_npz["meta"] = np.array([
            f"criterion=norm > mu_sample + {SD_K}*sd_sample per block "
            "(per-sample stats over the 196 patch tokens, CLS excluded)",
            "image-level = flagged at ANY single block (union over blocks 0..11)",
            "old_* arrays = step-1 criterion (population stats, >=3 of blocks 6..11) "
            "recomputed on the SAME samples for comparison",
            f"analyzed={_now()}",
        ])
        np.savez_compressed(masks_path(model_key), **out_npz)
        print(f"saved {masks_path(model_key)}")
        report[model_key] = rep

    (OUT_DIR / "step1c_analysis.json").write_text(json.dumps(report, indent=2))
    print(f"saved {OUT_DIR / 'step1c_analysis.json'}")

    for mk, rep in report.items():
        print(f"\n== {mk} (N={rep['N']})")
        print("   block:    " + "  ".join(f"{b:>5d}" for b in range(12)))
        for site in SITES:
            f = rep["sites"][site]["frac_per_block"]
            print(f"   {site:>9s} " + "  ".join(f"{100*x:5.2f}" for x in f) + "  (%)")
        for site in SITES:
            s = rep["sites"][site]
            print(f"   {site}: onset blk {s['onset_block']}, union {100*s['union_frac']:.2f}%, "
                  f"per-image mean {s['per_image']['mean']:.1f} "
                  f"[{s['per_image']['min']}..{s['per_image']['max']}]")
        ov = rep["old_vs_new_residual"]
        print("   old-vs-new Jaccard (residual): " +
              ", ".join(f"blk{b}={ov['jaccard_per_block'][str(b)]:.2f}" for b in JACCARD_BLOCKS) +
              f"; image-level {ov['jaccard_image_level_union_vs_consensus']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# figures (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  fig {FIG_DIR / stem}.png/.pdf")


SITE_LABEL = {"residual": "residual stream (block output)",
              "proj_drop": "attention output (attn.proj_drop)"}


def bimodality_figure(model_key: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.ticker import NullFormatter

    d = np.load(norms_path(model_key), allow_pickle=True)
    label = MODELS[model_key]["label"]
    fig, axes = plt.subplots(len(SITES), len(BIMODAL_BLOCKS),
                             figsize=(3.1 * len(BIMODAL_BLOCKS), 2.6 * len(SITES)),
                             sharey="row")
    for r, site in enumerate(SITES):
        norms = d[f"norms_{site}"]
        _, _, tau = per_sample_flags(norms)                    # (12, N)
        for c, b in enumerate(BIMODAL_BLOCKS):
            ax = axes[r, c]
            v = norms[b, :, 1:].ravel()
            v = v[v > 0]
            bins = np.geomspace(v.min(), v.max() * 1.02, 80)
            ax.hist(v, bins=bins, color="#4878a8", alpha=0.9)
            ax.set_xscale("log")
            ax.set_yscale("log")
            # clean fixed 1-2-5 ticks: log minor labels collide on <1-decade spans
            lo, hi = v.min(), v.max() * 1.02
            cand = [m * 10.0 ** e for e in range(-1, 4) for m in (1, 2, 5)]
            ticks = [t for t in cand if lo <= t <= hi][-4:]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.xaxis.set_minor_formatter(NullFormatter())
            t = tau[b]
            q1, med, q3 = np.percentile(t, [25, 50, 75])
            ax.axvspan(q1, q3, color="#d62728", alpha=0.18, lw=0)
            ax.axvline(med, color="#d62728", lw=1.6,
                       label=r"per-sample $\tau_b$: median (line), IQR (band)")
            ax.set_title(f"block {b}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{SITE_LABEL[site]}\ntoken count (log)", fontsize=8.5)
            if r == len(SITES) - 1:
                ax.set_xlabel("token L2 norm (log)", fontsize=9)
            ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.suptitle(
        f"{label} — patch-token norm distributions (all {d['labels'].shape[0]} images, "
        f"CLS excluded)\nbimodality motivating the outlier threshold "
        r"$\tau_b(\mathrm{sample}) = \mu_b(\mathrm{sample}) + 4\,\sigma_b(\mathrm{sample})$",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, f"norm_bimodality_{model_key}")


def fraction_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    site_color = {"residual": "#1f77b4", "proj_drop": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, model_key in zip(axes, MODELS):
        d = np.load(masks_path(model_key), allow_pickle=True)
        for site in SITES:
            frac = d[f"frac_per_block_{site}"] * 100.0
            ax.plot(range(12), frac, marker="o", ms=4, color=site_color[site],
                    label=SITE_LABEL[site])
        ax.set_title(MODELS[model_key]["label"], fontsize=10)
        ax.set_xlabel("block", fontsize=9)
        ax.set_xticks(range(12))
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("outlier-token fraction (%)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Fraction of patch tokens flagged per block — per-sample criterion "
                 r"(norm $> \mu_b + 4\sigma_b$ of the sample's own 196 patch tokens)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "outlier_fraction_per_block")


def cmd_figures(args):
    for model_key in MODELS:
        bimodality_figure(model_key)
    fraction_figure()


# ─────────────────────────────────────────────────────────────────────────────
# colocation (CPU) — redo on the 6 gallery samples with the NEW masks
# ─────────────────────────────────────────────────────────────────────────────

PATCH = 16


def cmd_colocation(args):
    g = np.load(OUT_DIR / "gallery_samples_vit_base_imagenet.npz", allow_pickle=True)
    keys = [str(k) for k in g["keys"]]
    norms = g["norms"]                                         # (12, 6, 197) residual site
    heat = g["heatmaps"]                                       # (6, 224, 224) cp_lrp_baseline

    _, union, _ = per_sample_flags(norms)                      # NEW masks (6, 196)

    blocks = heat.reshape(-1, GRID, PATCH, GRID, PATCH)
    heat_abs = np.abs(blocks).sum(axis=(2, 4))                 # (6, 14, 14) sum|R| per patch
    heat_signed = blocks.sum(axis=(2, 4))

    cols = ["key", "n_outliers", "topk_overlap", "abs_mass_share",
            "concentration_vs_area", "mean_abs_rank"]
    rows = []
    print("colocation (NEW per-sample any-block masks, residual site):")
    for i, key in enumerate(keys):
        m = union[i]
        n_out = int(m.sum())
        flat = heat_abs[i].ravel()
        order = np.argsort(-flat)
        rank = np.empty(N_PATCH, dtype=int)
        rank[order] = np.arange(1, N_PATCH + 1)
        out_idx = np.flatnonzero(m)
        topk = set(order[:n_out].tolist())
        overlap = len(topk & set(out_idx.tolist())) / max(n_out, 1)
        mass = float(flat[out_idx].sum() / flat.sum())
        area = n_out / N_PATCH
        conc = mass / area if area else float("nan")
        mrank = float(rank[out_idx].mean()) if n_out else float("nan")
        rows.append([key, n_out, overlap, np.float32(mass), np.float32(conc), mrank])
        print(f"  {key:16s} n_out={n_out:2d} topk∩={overlap:.2f} "
              f"|R|mass={100*mass:.1f}% (area {100*area:.1f}%) conc=x{conc:.1f} "
              f"mean-rank={mrank:.1f}")

    np.savez_compressed(
        OUT_DIR / "step1c_colocation_vit_base_imagenet.npz",
        keys=g["keys"], ds_indices=g["ds_indices"], targets=g["targets"],
        outlier_masks=union, heat_patch_abs=heat_abs.astype(np.float32),
        heat_patch_signed=heat_signed.astype(np.float32),
        metrics=np.array(rows, dtype=object), metrics_columns=np.array(cols),
        meta=np.array([
            "masks: NEW step-1c criterion (per-sample mu+4sd per block, union over "
            "blocks 0..11), residual site, from gallery_samples norms",
            "heatmaps: existing full-model class-conditional cp_lrp_baseline "
            "(gallery_samples_vit_base_imagenet.npz), sum|R| per 16x16 patch",
            f"computed={_now()}",
        ]),
    )
    print(f"saved {OUT_DIR / 'step1c_colocation_vit_base_imagenet.npz'}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.set_defaults(fn=cmd_collect)
    c.add_argument("--model", required=True, choices=list(MODELS))
    c.add_argument("--n", type=int, default=256)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--batch-size", type=int, default=64)
    c.add_argument("--device", default="cuda")
    a = sub.add_parser("analyze"); a.set_defaults(fn=cmd_analyze)
    f = sub.add_parser("figures"); f.set_defaults(fn=cmd_figures)
    co = sub.add_parser("colocation"); co.set_defaults(fn=cmd_colocation)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
