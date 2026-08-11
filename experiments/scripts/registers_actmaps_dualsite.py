"""Registers: DUAL-SITE per-sample activation maps for the CRP-gallery samples.

Reviewer (Adam) found a site mismatch in the CRP gallery: the concept
visualisations existed only for site ``proj_drop`` (attention output-projection,
BEFORE the residual add), while the register-outlier activation maps were
recorded at the ``blocks[i]`` output (the residual stream AFTER the adds).
Relevance heatmaps at proj_drop do not show the register artifacts the
residual-stream activations show — so the gallery norm-maps must present BOTH
sites side by side.

For each fixed gallery sample (the canonical 6 per model — FunnyBirds
c0_0..c5_2988 and ImageNet lizard..golden_retriever, exactly
``crp_gallery.pick_samples``), this script:

* runs ONE forward pass recording the L2 token norms at both sites per block:
  - ``residual``  = ``backbone.blocks[i]`` output (full residual stream),
  - ``proj_drop`` = ``backbone.blocks[i].attn.proj_drop`` output (pre-add);
* flags outlier tokens PER SAMPLE at each site/block with the reviewer's
  per-sample criterion: token norm > mean + 4*sd over that sample's own 196
  patch tokens at that site/block (single-block flags, CLS excluded);
* renders one figure per sample with two row-groups (residual on top,
  proj_drop below), per-block [0,1]-normalized viridis maps, magenta borders
  on flagged tokens → ``figures/registers/actmaps_dualsite/actmap2_<key>.{png,pdf}``;
* with ``--deploy`` (default) also installs the pngs as the gallery web
  norm-maps: ``webapp/crp_gallery/figures/<base>_<dataset>/_normmaps/<key>.png``
  (overwriting the old single-site version for funny_birds; new for imagenet).
  Rebuild the manifest afterwards: ``python -m experiments.crp_gallery manifest``.

Raw norms are persisted to
``data/results/registers/actmaps_dualsite_<dataset>.npz``.

Run (CPU is fine — 12 forward passes total)::

    python -m experiments.scripts.registers_actmaps_dualsite --dataset both --device cpu
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "results" / "registers"
FIG_DIR = REPO_ROOT / "figures" / "registers" / "actmaps_dualsite"
WEBAPP_FIG = REPO_ROOT / "webapp" / "crp_gallery" / "figures"

SD_K = 4.0        # per-sample criterion: norm > mean + 4*sd over the sample's patch tokens
GRID = 14
SITES = ("residual", "proj_drop")
SITE_DESC = {
    "residual": "site residual — blocks[i] output (residual stream, AFTER the adds)",
    "proj_drop": "site proj_drop — attn output projection (BEFORE the residual add)",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class DualSiteNormRecorder:
    """Forward hooks recording L2 token norms of every block's OUTPUT (residual
    stream) and its ``attn.proj_drop`` output (pre-residual-add), per block."""

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

    def stack(self, site: str) -> np.ndarray:            # (12, B, 197)
        d = self.norms[site]
        return np.stack([d[b] for b in sorted(d)])

    def remove(self):
        for h in self.handles:
            h.remove()


def collect(dataset: str, device: str):
    """Forward the 6 gallery samples once, norms at both sites.
    Returns ``(samples, {site: (12, 6, 197)})``."""
    import torch
    from experiments.models import FunnyBirdsViTSmall, ImagenetViTBase, backbone_transforms
    from experiments.datasets import load as load_dataset
    from experiments.crp_gallery import pick_samples
    from experiments.scripts.registers_position_freq import CKPT_FUNNY

    if dataset == "funny_birds":
        model = FunnyBirdsViTSmall(checkpoint=CKPT_FUNNY, device=device)
        transform, normalize = backbone_transforms(model.backbone)
        ds = load_dataset("funny_birds", root=REPO_ROOT / "data",
                          transform=transform, split="train", clean_only=True)
    elif dataset == "imagenet":
        model = ImagenetViTBase(device=device)
        transform, normalize = backbone_transforms(model.backbone)
        ds = load_dataset("imagenet_val_hf", root=REPO_ROOT / "data",
                          transform=transform).subsample(10)
    else:
        raise SystemExit(f"unknown dataset {dataset!r}")

    samples = pick_samples(dataset, ds)
    assert len(samples) == 6, f"expected 6 gallery samples, got {len(samples)}"
    print(f"{dataset}: samples {[s['key'] for s in samples]} "
          f"(ds_indices {[s['ds_index'] for s in samples]})")

    rec = DualSiteNormRecorder(model.backbone)
    with torch.no_grad():
        x = torch.stack([ds[s["ds_index"]][0] for s in samples]).to(device)
        model(normalize(x))
    norms = {site: rec.stack(site) for site in SITES}    # (12, 6, 197)
    rec.remove()
    return samples, norms


def per_sample_flags(norms_site: np.ndarray) -> np.ndarray:
    """Per-sample mean+4sd criterion. norms (12, B, 197) -> masks (12, B, 196):
    thresholds over each SAMPLE's own 196 patch tokens at that block (CLS
    excluded), single-block flags."""
    patch = norms_site[:, :, 1:]                          # (12, B, 196)
    mu = patch.mean(axis=2, keepdims=True)
    sd = patch.std(axis=2, keepdims=True)
    return patch > mu + SD_K * sd


def actmap2_figure(norms_1: Dict[str, np.ndarray], masks_1: Dict[str, np.ndarray],
                   title: str, stem: str):
    """One sample, both sites: two row-groups of 2x6 per-block maps.
    norms_1[site] (12, 197), masks_1[site] (12, 196)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(13.5, 10.2))
    subfigs = fig.subfigures(2, 1, hspace=0.04)
    for sf, site in zip(subfigs, SITES):
        sf.suptitle(SITE_DESC[site], fontsize=11, fontweight="bold")
        axes = sf.subplots(2, 6)
        for b, ax in enumerate(axes.ravel()):
            v = norms_1[site][b, 1:].reshape(GRID, GRID)
            vn = (v - v.min()) / (v.max() - v.min() + 1e-12)
            ax.imshow(vn, cmap="viridis", vmin=0, vmax=1)
            m = masks_1[site][b].reshape(GRID, GRID)
            for r, c in zip(*np.nonzero(m)):
                ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                                edgecolor="magenta", linewidth=1.6))
            ax.set_title(f"block {b} · {int(m.sum())} flagged", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{title}\nper-block token L2 norms, normalized to [0,1] per block/site "
                 f"(viridis); magenta = flagged (norm > mean + {SD_K:g}·sd over this "
                 f"sample's 196 patch tokens at that site/block; CLS excluded)",
                 fontsize=11, y=1.045)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig {FIG_DIR / stem}.png/.pdf")


def crosscheck_residual(dataset: str, samples: List[dict], res_norms: np.ndarray):
    """Sanity: residual-site norms must match the previously stored ones."""
    if dataset == "imagenet":
        path, key = OUT_DIR / "gallery_samples_vit_base_imagenet.npz", "norms"
    else:
        path, key = OUT_DIR / "step1b_position_freq_funny_birds.npz", "gallery_norms"
    if not path.exists():
        print(f"  [crosscheck skipped] {path} missing")
        return
    ref = np.load(path, allow_pickle=True)[key]           # (12, 6, 197)
    ok = np.allclose(ref, res_norms, rtol=1e-3, atol=1e-2)
    print(f"  crosscheck residual norms vs {path.name}: "
          f"{'OK' if ok else 'MISMATCH — max |d|=%.4g' % np.abs(ref - res_norms).max()}")


def run_dataset(dataset: str, device: str, deploy: bool):
    md = {"funny_birds": "vit_small_funny_birds", "imagenet": "vit_base_imagenet"}[dataset]
    samples, norms = collect(dataset, device)
    crosscheck_residual(dataset, samples, norms["residual"])
    masks = {site: per_sample_flags(norms[site]) for site in SITES}
    model_lbl = "ViT-S/16 · FunnyBirds" if dataset == "funny_birds" else "ViT-B/16 · ImageNet val"
    for j, s in enumerate(samples):
        n1 = {site: norms[site][:, j] for site in SITES}
        m1 = {site: masks[site][:, j] for site in SITES}
        actmap2_figure(n1, m1, f"{model_lbl} — gallery sample {s['key']} "
                       f"(ds_index {s['ds_index']})", f"actmap2_{s['key']}")
        if deploy:
            dst = WEBAPP_FIG / md / "_normmaps" / f"{s['key']}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(FIG_DIR / f"actmap2_{s['key']}.png", dst)
            print(f"  deployed → {dst}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"actmaps_dualsite_{dataset}.npz",
        norms_residual=norms["residual"].astype(np.float32),
        norms_proj_drop=norms["proj_drop"].astype(np.float32),
        masks_residual=masks["residual"], masks_proj_drop=masks["proj_drop"],
        keys=np.array([s["key"] for s in samples]),
        ds_indices=np.array([s["ds_index"] for s in samples], dtype=np.int64),
        meta=np.array([
            f"model={model_lbl}",
            "norms: (block, sample, token) L2 over embed_dim; token0=CLS",
            "residual=blocks[i] output; proj_drop=blocks[i].attn.proj_drop output",
            f"masks: per-sample per-site per-block, norm > mean + {SD_K:g}*sd over the "
            "sample's 196 patch tokens (CLS excluded), single-block flags",
            f"collected={_now()}",
        ]))
    print(f"saved {OUT_DIR / f'actmaps_dualsite_{dataset}.npz'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="both",
                    choices=["funny_birds", "imagenet", "both"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deploy", action=argparse.BooleanOptionalAction, default=True,
                    help="copy pngs into webapp/crp_gallery/figures/<md>/_normmaps/")
    args = ap.parse_args()
    for ds in (["funny_birds", "imagenet"] if args.dataset == "both" else [args.dataset]):
        run_dataset(ds, args.device, args.deploy)


if __name__ == "__main__":
    main()
