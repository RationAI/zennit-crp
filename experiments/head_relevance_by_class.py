"""Class-conditioned per-head relevance distributions (HeadConcept).

Experiment: for a trained probe, at one attention block, measure how much
each attention head (HeadConcept detector) contributes — per class.

Procedure
---------
1. Pick N images per class that the model classifies *correctly* into
   that class.
2. For each such image, run LRP attribution **conditioned on that class**
   and read the relevance recorded at ``backbone.blocks.<L>.attn.proj_drop``.
3. Aggregate with ``HeadConcept`` → one raw relevance value per head per
   image (sum over head_dim and tokens, ``abs_norm=False``).
4. Repeat for every class → distribution of per-head relevance, per class.

Raw values are stored unmodified. For visualisation a *single global*
scale factor is applied (divide by global max|R|) so that cells stay
comparable — no per-cell normalisation. The grid is rows=class,
cols=head, each cell a small violin on a shared axis.

Usage::

    uv run python -m experiments.head_relevance_by_class \\
        --probe data/runs/finetune_vit_small_funny-birds-train-clean/<ts>/best.pt \\
        --split funny-birds-train-clean --n-images 30 --block -1

Re-plot from cached raw values (fast iteration on the scale)::

    uv run python -m experiments.head_relevance_by_class --from-cache data/head_relevance/<run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from timm.data import resolve_data_config, create_transform

from crp.attribution import CondAttribution
from zennit_extensions import AttnLRPBaselineComposite
from crp.concepts import HeadConcept
from experiments.datasets import load as load_dataset
from experiments.models import FinetunedProbe

REPO_ROOT = Path(__file__).resolve().parents[1]

# train-ds choice → (dataset name, loader kwargs). Mirrors train_probe.
SPLITS = {
    "funny-birds-train-clean": ("funny_birds", {"split": "train", "clean_only": True}),
    "funny-birds-test":        ("funny_birds", {"split": "test"}),
}


# ── compute ────────────────────────────────────────────────────────────────

def _load_probe(path: Path, device: str):
    model = FinetunedProbe(checkpoint=path, device=device)
    return model, model.meta


def _select_correct(model, ds, num_classes, n_per_class, device, batch_size=64):
    """Return dict class → list of dataset indices, of images correctly
    classified into that class (up to n_per_class each)."""
    selected = {c: [] for c in range(num_classes)}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    idx = 0
    done = set()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(-1).cpu()
            for j in range(len(y)):
                c = int(y[j])
                if pred[j] == c and len(selected[c]) < n_per_class:
                    selected[c].append(idx + j)
                    if len(selected[c]) >= n_per_class:
                        done.add(c)
            idx += len(y)
            if len(done) == num_classes:
                break
    return selected


def compute(args, device):
    model, ck = _load_probe(args.probe, device)
    num_classes = int(ck["num_classes"])
    num_heads = int(model.backbone.blocks[0].attn.num_heads)
    n_blocks = len(model.backbone.blocks)
    block = n_blocks - 1 if args.block < 0 else args.block
    layer = f"backbone.blocks.{block}.attn.proj_drop"
    print(f"probe={args.probe.name} val_acc={ck.get('val_acc')} "
          f"classes={num_classes} heads={num_heads} block={block}")
    print(f"layer={layer}")

    cfg = resolve_data_config({}, model=model.backbone)
    transform = create_transform(**cfg, is_training=False)  # includes normalize
    ds_name, ds_kw = SPLITS[args.split]
    ds = load_dataset(ds_name, root=REPO_ROOT / "data", transform=transform, **ds_kw)
    print(f"dataset={args.split} ({ds_name}) size={len(ds)}")

    print("pass 1: finding correctly-classified images per class ...")
    selected = _select_correct(model, ds, num_classes, args.n_images, device)
    counts = {c: len(v) for c, v in selected.items()}
    short = {c: n for c, n in counts.items() if n < args.n_images}
    print(f"  collected per-class min={min(counts.values())} "
          f"max={max(counts.values())} target={args.n_images}")
    if short:
        print(f"  WARNING {len(short)} classes under target: {short}")

    composite = AttnLRPBaselineComposite()
    attribution = CondAttribution(model)
    concept = HeadConcept(num_heads=num_heads)

    print("pass 2: class-conditioned attribution ...")
    per_class = {}  # class → (n_c, num_heads) raw relevance
    bs = args.batch_size
    for c in range(num_classes):
        idxs = selected[c]
        if not idxs:
            per_class[c] = np.zeros((0, num_heads), dtype=np.float32)
            continue
        vals = []
        for k in range(0, len(idxs), bs):
            batch_idx = idxs[k:k + bs]
            x = torch.stack([ds[i][0] for i in batch_idx]).to(device)
            x.requires_grad_(True)
            conditions = [{"y": [c]} for _ in batch_idx]
            res = attribution(x, conditions, composite, record_layer=[layer])
            rel = res.relevances[layer]               # (B, N, embed_dim)
            head_rel = concept.attribute(rel, abs_norm=False)  # (B, num_heads)
            vals.append(head_rel.detach().float().cpu().numpy())
        per_class[c] = np.concatenate(vals, axis=0)
        print(f"  class {c:>2}: {per_class[c].shape[0]} images")

    # persist raw
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "head_relevance_raw.npz",
        **{f"class_{c}": per_class[c] for c in range(num_classes)},
    )
    meta = dict(
        probe=str(args.probe), split=args.split, num_classes=num_classes,
        num_heads=num_heads, block=block, layer=layer,
        n_images=args.n_images,
        counts=counts,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved raw → {out}/head_relevance_raw.npz")
    return per_class, meta, out


# ── plot ─────────────────────────────────────────────────────────────────

def load_cache(run_dir: Path):
    npz = np.load(run_dir / "head_relevance_raw.npz")
    meta = json.loads((run_dir / "meta.json").read_text())
    per_class = {int(k.split("_")[1]): npz[k] for k in npz.files}
    return per_class, meta


def plot_grid(per_class, meta, out: Path, clip_pct: float = 100.0, row_h: float = 0.46):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = meta["num_classes"]
    num_heads = meta["num_heads"]

    # Global normalisation: ONE scale factor over all data points (keeps
    # cells comparable). clip_pct<100 divides by the p-th percentile of
    # |R| instead of the max, so a few extreme cells don't stretch the
    # axis and crush the bulk; values beyond are clipped (count reported).
    allv = np.concatenate([v.ravel() for v in per_class.values() if v.size])
    gmax = float(np.percentile(np.abs(allv), clip_pct)) or 1.0
    normed_all = allv / gmax
    n_clipped = int((np.abs(normed_all) > 1.0).sum())
    lo = max(float(normed_all.min()), -1.0)
    hi = min(float(normed_all.max()), 1.0)
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ylim = (lo - pad, hi + pad)
    suffix = "max" if clip_pct >= 100 else f"p{clip_pct:g}"
    print(f"global |R| {suffix}={gmax:.4g}  normed range=[{lo:.3f},{hi:.3f}]  "
          f"clipped={n_clipped}/{allv.size}")

    def prep(d):
        return np.clip(d / gmax, ylim[0], ylim[1])

    fig, axes = plt.subplots(
        num_classes, num_heads,
        figsize=(num_heads * 1.4, num_classes * row_h),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.07, right=0.99,
                        top=0.965, bottom=0.02)

    for c in range(num_classes):
        for h in range(num_heads):
            ax = axes[c][h]
            data = prep(per_class[c][:, h]) if per_class[c].size else np.array([])
            ax.axhline(0.0, color="0.85", lw=0.5, zorder=0)
            if data.size >= 2 and np.ptp(data) > 0:
                parts = ax.violinplot(data, positions=[0], widths=0.85,
                                      showmeans=True, showextrema=False)
                for b in parts["bodies"]:
                    b.set_facecolor("#4C72B0")
                    b.set_edgecolor("#26456e")
                    b.set_alpha(0.8)
                parts["cmeans"].set_color("#C44E52")
                parts["cmeans"].set_linewidth(1.2)
            elif data.size:
                ax.scatter([0] * data.size, data, s=4, color="#4C72B0", alpha=0.7)
                ax.hlines(data.mean(), -0.4, 0.4, color="#C44E52", lw=1.2)
            ax.set_ylim(*ylim)
            ax.set_xlim(-0.6, 0.6)
            ax.set_xticks([])
            for s in ("top", "right", "bottom"):
                ax.spines[s].set_visible(False)
            if h == 0:
                ax.set_ylabel(f"{c}", rotation=0, ha="right", va="center",
                              fontsize=6, labelpad=2)
                ax.tick_params(axis="y", labelsize=5, length=2)
            else:
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0, labelleft=False)
            if c == 0:
                ax.set_title(f"head {h}", fontsize=8, pad=4)

    fig.suptitle(
        f"HeadConcept relevance per head × class — {meta['layer']}  "
        f"(global ÷{suffix}={gmax:.3g}, clipped {n_clipped}/{allv.size}; "
        f"N≈{meta['n_images']})",
        fontsize=10, y=0.997,
    )
    fig.text(0.012, 0.5, "class", rotation=90, va="center", fontsize=9)
    png = out / "head_relevance_grid.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"saved plot → {png}")
    return png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path)
    p.add_argument("--split", default="funny-birds-train-clean", choices=list(SPLITS))
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--block", type=int, default=-1, help="block index (-1 = last)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "head_relevance" / "run")
    p.add_argument("--from-cache", type=Path, default=None,
                   help="skip compute; re-plot from this run dir")
    p.add_argument("--clip-pct", type=float, default=100.0,
                   help="global scale = this percentile of |R| (100=max). "
                        "Lower (e.g. 99) de-emphasises extreme cells.")
    p.add_argument("--row-h", type=float, default=0.46,
                   help="figure height per class row (inches)")
    args = p.parse_args()

    if args.from_cache:
        per_class, meta = load_cache(args.from_cache)
        plot_grid(per_class, meta, args.from_cache,
                  clip_pct=args.clip_pct, row_h=args.row_h)
        return

    assert args.probe is not None, "--probe required unless --from-cache"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    per_class, meta, out = compute(args, device)
    plot_grid(per_class, meta, out, clip_pct=args.clip_pct, row_h=args.row_h)


if __name__ == "__main__":
    main()
