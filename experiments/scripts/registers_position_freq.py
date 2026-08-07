"""Registers step 1b (XAI-35): per-position outlier-flag frequency at scale.

Reviewer challenge to step 1 (research/registers/step1_detect.md): the claim that
ViT-S/FunnyBirds outlier tokens are corner-anchored at grid position (1,1)
(P=0.72) rested on N=256. Step 1b re-measures per-position flag frequencies on a
larger sample with proper statistics.

EXPERIMENT CARD
  RQ  Are some spatial token positions flagged as high-norm outliers
      significantly more often than others in ViT-S/FunnyBirds — specifically,
      is (1,1) a persistent anomaly location?
  H1  Per-position flag frequency is strongly non-uniform; (1,1) is flagged in
      a majority of images.
  H0  Flag positions are exchangeable across the 14x14 grid.
  Falsified if no position beats uniform expectation after Holm correction, or
  (1,1) is not an extreme position (corrected p >= 0.01 or frequency <= 0.5).

  Detection rule — IDENTICAL to step 1:
    * L2 norm of every ``backbone.blocks[i]`` output token, i = 0..11;
      token 0 = CLS (excluded), tokens 1..196 = 14x14 patch grid, row-major.
    * per-block threshold tau_b = mean_b + 4*sd_b over ALL patch-token norms of
      the whole image sample at block b (population-level, CLS excluded);
    * image-level flag = token flagged in >= 3 of blocks 6..11.

  Samples: FunnyBirds TEST split — the card asked N=2048 but the official test
  split only has 500 images, so the ENTIRE test split (N=500) is used, plus a
  supplementary train-clean N=2048 sample at the requested scale
  (``--split train``, saved as ``..._funny_birds_train.npz``); ImageNet val
  subset (n_per_class=10) N=1024 (ViT-B, contrast). Class-diverse round-robin,
  seed 0, indices persisted in each npz.

  Stats: per-position flag count k_p out of N; H0 binomial with
  p0 = (mean flags per image)/196; exact binomial p-values, Holm-corrected
  across 196 positions; chi-square GOF over the grid (approximate — flags are
  not independent within an image). H1 supported for (1,1) iff corrected
  p < 0.01 AND frequency > 0.5.

Run (GPU for collect, CPU for the rest)::

    python -m experiments.scripts.registers_position_freq collect --dataset funny_birds --n 2048
    python -m experiments.scripts.registers_position_freq collect --dataset imagenet --n 1024
    python -m experiments.scripts.registers_position_freq analyze
    python -m experiments.scripts.registers_position_freq figures

Outputs: ``data/results/registers/step1b_position_freq_<dataset>.npz``,
figures in ``figures/registers/step1b_positions/`` (png+pdf).
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
FIG_DIR = REPO_ROOT / "figures" / "registers" / "step1b_positions"

CKPT_FUNNY = (REPO_ROOT / "data" / "runs"
              / "finetune_vit_small_funny-birds-train-clean" / "2026-06-03_000556" / "best.pt")

SD_K = 4.0                       # step-1 primary criterion: norm > mean + 4*sd
CONSENSUS_BLOCKS = list(range(6, 12))   # blocks 6..11
MIN_VOTES = 3                    # image-level flag: >= 3 of blocks 6..11
GRID = 14
N_PATCH = GRID * GRID            # 196; token 0 = CLS


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def npz_path(dataset: str) -> Path:
    return OUT_DIR / f"step1b_position_freq_{dataset}.npz"


# ─────────────────────────────────────────────────────────────────────────────
# collect (GPU) — forward passes, per-block token norms
# ─────────────────────────────────────────────────────────────────────────────

def pick_class_diverse(ds, n: int, seed: int = 0) -> List[int]:
    """Round-robin over classes (same scheme as registers_token_flow)."""
    if hasattr(ds, "items"):
        labels = [int(c) for _, c in ds.items]
    elif hasattr(ds, "rows"):
        labels = [int(c) for _, c in ds.rows]
    else:
        labels = [int(ds[i][1]) for i in range(len(ds))]
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


class BlockNormRecorder:
    """Forward hooks: L2 norm of every blocks[i] OUTPUT token (step-1 measure)."""

    def __init__(self, backbone):
        self.norms: Dict[int, np.ndarray] = {}
        self.handles = []
        for b, blk in enumerate(backbone.blocks):
            def hook(mod, args, out, b=b):
                self.norms[b] = out.detach().norm(dim=-1).float().cpu().numpy()
            self.handles.append(blk.register_forward_hook(hook))

    def stack(self) -> np.ndarray:                    # (12, B, 197)
        return np.stack([self.norms[b] for b in sorted(self.norms)])

    def remove(self):
        for h in self.handles:
            h.remove()


def _forward_norms(model, normalize, ds, indices, device, batch_size=64) -> np.ndarray:
    import torch
    rec = BlockNormRecorder(model.backbone)
    chunks = []
    with torch.no_grad():
        for s in range(0, len(indices), batch_size):
            batch = [ds[i][0] for i in indices[s:s + batch_size]]
            x = normalize(torch.stack(batch).to(device))
            model(x)
            chunks.append(rec.stack())
            print(f"  {s + len(batch)}/{len(indices)}", flush=True)
    rec.remove()
    return np.concatenate(chunks, axis=1)             # (12, N, 197)


def flags_from_norms(norms: np.ndarray):
    """Step-1 rule. norms (12, N, 197) -> per-block masks, image-level mask,
    per-block thresholds. Thresholds over ALL patch tokens of the sample."""
    patch = norms[:, :, 1:]                            # (12, N, 196), CLS excluded
    mean_b = patch.reshape(12, -1).mean(1)
    sd_b = patch.reshape(12, -1).std(1)
    tau = mean_b + SD_K * sd_b                         # (12,)
    masks = patch > tau[:, None, None]                 # (12, N, 196)
    votes = masks[CONSENSUS_BLOCKS].sum(0)             # (N, 196)
    image_level = votes >= MIN_VOTES                   # (N, 196)
    return masks, image_level, tau


def cmd_collect(args):
    import torch
    device = args.device
    from experiments.model_io import load_probe
    from experiments.models import backbone_transforms
    from experiments.datasets import load as load_dataset

    if args.dataset == "funny_birds":
        model, ck, ck_path = load_probe("funny-birds-train-clean", device,
                                        base="vit_small", path=CKPT_FUNNY)
        transform, normalize = backbone_transforms(model.backbone)
        kw = {"split": "test"} if args.split == "test" else \
             {"split": "train", "clean_only": True}
        ds = load_dataset("funny_birds", root=REPO_ROOT / "data",
                          transform=transform, **kw)
        label = f"vit_small · funny_birds {args.split.upper()}"
    elif args.dataset == "imagenet":
        model, ck, ck_path = load_probe("imagenet", device, base="vit_base")
        transform, normalize = backbone_transforms(model.backbone)
        ds = load_dataset("imagenet_val_hf", root=REPO_ROOT / "data",
                          transform=transform).subsample(10)
        label = "vit_base · imagenet val (10/class)"
    else:
        raise SystemExit(f"unknown dataset {args.dataset!r}")

    indices = pick_class_diverse(ds, args.n, seed=args.seed)
    if hasattr(ds, "items"):
        labels = np.array([int(ds.items[i][1]) for i in indices])
    elif hasattr(ds, "rows"):
        labels = np.array([int(ds.rows[i][1]) for i in indices])
    else:
        labels = np.array([int(ds[i][1]) for i in indices])
    print(f"{label}: N={len(indices)} of {len(ds)}")

    norms = _forward_norms(model, normalize, ds, indices, device, args.batch_size)
    masks, image_level, tau = flags_from_norms(norms)

    out = dict(
        norms=norms.astype(np.float32),
        ds_indices=np.array(indices, dtype=np.int64),
        labels=labels,
        thresholds_mean4sd=tau.astype(np.float32),
        masks=masks, image_level_mask=image_level,
        meta=np.array([
            f"model={label}", f"checkpoint={ck_path}",
            "norm=L2 over embed_dim of blocks[i] output, i=0..11",
            "token0=CLS (excluded), tokens1..196=patches row-major 14x14",
            f"criterion=norm > mean_b + {SD_K}*sd_b over all patch tokens of the sample; "
            f"image-level = flagged in >= {MIN_VOTES} of blocks {CONSENSUS_BLOCKS}",
            f"split={'test' if args.dataset == 'funny_birds' else 'val'}",
            f"selection=round-robin class-diverse seed={args.seed}",
            f"collected={_now()}",
        ]),
    )

    if args.dataset == "funny_birds" and args.split == "test":
        # Fixed gallery samples (TRAIN clean split — the canonical 6 of the CRP
        # gallery) for the visual-verification act-map figures. Flags for them
        # use the test-sample thresholds computed above (noted in meta).
        from experiments.crp_gallery import pick_samples
        ds_train = load_dataset("funny_birds", root=REPO_ROOT / "data",
                                transform=transform, split="train", clean_only=True)
        samples = pick_samples("funny_birds", ds_train)
        g_norms = _forward_norms(model, normalize, ds_train,
                                 [s["ds_index"] for s in samples], device, args.batch_size)
        out.update(
            gallery_norms=g_norms.astype(np.float32),
            gallery_keys=np.array([s["key"] for s in samples]),
            gallery_ds_indices=np.array([s["ds_index"] for s in samples], dtype=np.int64),
        )
        print("gallery samples:", [s["key"] for s in samples])

    key = args.dataset if not (args.dataset == "funny_birds" and args.split == "train") \
        else "funny_birds_train"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path(key), **out)
    print(f"saved {npz_path(key)}")


# ─────────────────────────────────────────────────────────────────────────────
# analyze (CPU) — per-position binomial + Holm, chi-square GOF
# ─────────────────────────────────────────────────────────────────────────────

def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def position_stats(image_level: np.ndarray) -> dict:
    from scipy import stats
    N = image_level.shape[0]
    k = image_level.sum(0).astype(int)                # (196,)
    total = int(k.sum())
    p0 = total / (N * N_PATCH)                        # uniform per-position rate
    pvals = stats.binom.sf(k - 1, N, p0)              # P(X >= k)
    padj = holm(pvals)
    expected = N * p0
    chisq = float(((k - expected) ** 2 / expected).sum())
    chi_p = float(stats.chi2.sf(chisq, N_PATCH - 1))
    return dict(N=N, k=k, freq=k / N, p0=p0, pvals=pvals, padj=padj,
                chisq=chisq, chi_p=chi_p,
                flags_per_image=total / N)


def fmt_pos(t: int) -> str:
    return f"({t // GRID},{t % GRID})"


def cmd_analyze(args):
    report = {}
    for dataset in ("funny_birds", "funny_birds_train", "imagenet"):
        path = npz_path(dataset)
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        d = np.load(path, allow_pickle=True)
        st = position_stats(d["image_level_mask"])
        top = np.argsort(-st["k"])[:10]
        n_sig = int((st["padj"] < 0.01).sum())
        pos11 = 1 * GRID + 1
        print(f"\n== {dataset} — N={st['N']}, mean flags/image "
              f"{st['flags_per_image']:.2f}, p0={st['p0']:.5f}")
        print(f"   chi-square GOF: chi2={st['chisq']:.0f} (df={N_PATCH - 1}), "
              f"p={st['chi_p']:.3g}   |   positions with Holm p<0.01: {n_sig}")
        print("   pos      freq     k      p_holm")
        for t in top:
            print(f"   {fmt_pos(t):8s} {st['freq'][t]:.3f}  {st['k'][t]:5d}  "
                  f"{st['padj'][t]:.3g}")
        print(f"   (1,1):   freq={st['freq'][pos11]:.3f}  "
              f"p_holm={st['padj'][pos11]:.3g}")
        report[dataset] = dict(
            N=st["N"], flags_per_image=st["flags_per_image"], p0=st["p0"],
            chisq=st["chisq"], chi_p=st["chi_p"], n_sig_holm01=n_sig,
            top10=[dict(pos=fmt_pos(int(t)), token=int(t), freq=float(st["freq"][t]),
                        k=int(st["k"][t]), p_holm=float(st["padj"][t])) for t in top],
            pos_1_1=dict(freq=float(st["freq"][pos11]), k=int(st["k"][pos11]),
                         p_holm=float(st["padj"][pos11])),
        )
        # persist stats next to the raw arrays
        out = {kk: d[kk] for kk in d.files}
        out.update(flag_count=st["k"], flag_freq=st["freq"],
                   binom_p=st["pvals"], binom_p_holm=st["padj"],
                   chi2=np.array([st["chisq"], st["chi_p"]]))
        np.savez_compressed(path, **out)
    (OUT_DIR / "step1b_position_stats.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {OUT_DIR / 'step1b_position_stats.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# figures (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  fig {stem}.png/.pdf")


def actmap_figure(norms_1img: np.ndarray, masks_1img: np.ndarray, title: str, stem: str):
    """One sample: 3x4 grid of per-block normalized token-norm maps (viridis),
    magenta borders on tokens flagged at that block.
    norms_1img (12, 197), masks_1img (12, 196)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6))
    for b, ax in enumerate(axes.ravel()):
        v = norms_1img[b, 1:].reshape(GRID, GRID)
        vn = (v - v.min()) / (v.max() - v.min() + 1e-12)
        ax.imshow(vn, cmap="viridis", vmin=0, vmax=1)
        m = masks_1img[b].reshape(GRID, GRID)
        for r, c in zip(*np.nonzero(m)):
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                            edgecolor="magenta", linewidth=1.6))
        ax.set_title(f"block {b} · {int(m.sum())} flagged", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{title}\nper-block token-activation L2 norms, normalized to [0,1] "
                 f"per block (viridis); magenta = flagged (norm > mean_b + 4·sd_b)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, stem)


def cmd_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dF = np.load(npz_path("funny_birds"), allow_pickle=True)

    # 1 — act-map panels: 6 gallery samples + 6 random flagged test images
    g_norms = dF["gallery_norms"]                       # (12, 6, 197)
    g_keys = [str(k) for k in dF["gallery_keys"]]
    tau = dF["thresholds_mean4sd"]
    for j, key in enumerate(g_keys):
        n1 = g_norms[:, j]                              # (12, 197)
        m1 = n1[:, 1:] > tau[:, None]                   # (12, 196)
        actmap_figure(n1, m1, f"FunnyBirds gallery sample {key} (train-clean split, "
                      f"test-sample thresholds)", f"actmap_{key}")

    norms = dF["norms"]; masks = dF["masks"]
    flagged_imgs = np.flatnonzero(dF["image_level_mask"].any(1))
    rng = np.random.default_rng(1)
    rand6 = rng.choice(flagged_imgs, size=6, replace=False)
    for i in rand6:
        dsi = int(dF["ds_indices"][i])
        actmap_figure(norms[:, i], masks[:, i],
                      f"FunnyBirds TEST image ds_index={dsi} (random flagged sample)",
                      f"actmap_test{dsi}")
    # persist which random test images were used
    d_all = {k: dF[k] for k in dF.files}
    d_all["actmap_random_sample_rows"] = rand6
    np.savez_compressed(npz_path("funny_birds"), **d_all)

    # 2 — position-frequency summary heatmap, FunnyBirds + ImageNet side by side
    dI = np.load(npz_path("imagenet"), allow_pickle=True)
    freqs = {"ViT-S/16 · FunnyBirds test": (dF["flag_freq"], dF["image_level_mask"].shape[0]),
             "ViT-B/16 · ImageNet val": (dI["flag_freq"], dI["image_level_mask"].shape[0])}
    vmax = max(f.max() for f, _ in freqs.values())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (name, (f, N)) in zip(axes, freqs.items()):
        grid = f.reshape(GRID, GRID)
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=vmax)
        for t in np.argsort(-f)[:5]:
            r, c = t // GRID, t % GRID
            ax.text(c, r, f"{f[t]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if grid[r, c] < 0.6 * vmax else "black")
        ax.set_title(f"{name} (N={N})", fontsize=10)
        ax.set_xticks(range(0, GRID, 2)); ax.set_yticks(range(0, GRID, 2))
        ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("P(image-level outlier flag)", fontsize=9)
    fig.suptitle("Per-position outlier-flag frequency (image-level rule: flagged in "
                 ">=3 of blocks 6–11; top-5 positions annotated)", fontsize=11)
    _save(fig, "position_frequency")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.set_defaults(fn=cmd_collect)
    c.add_argument("--dataset", required=True, choices=["funny_birds", "imagenet"])
    c.add_argument("--split", default="test", choices=["test", "train"],
                   help="funny_birds only; train = supplementary large sample")
    c.add_argument("--n", type=int, default=2048)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--batch-size", type=int, default=64)
    c.add_argument("--device", default="cuda")
    a = sub.add_parser("analyze"); a.set_defaults(fn=cmd_analyze)
    f = sub.add_parser("figures"); f.set_defaults(fn=cmd_figures)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
