"""Register/outlier-token residual relevance flow — per-TOKEN skip/branch split.

Step 2 of the register-artifact investigation (XAI-34 / XAI-36): the H_A
mechanistic test. Reuses the recording machinery of
:mod:`experiments.scripts.residual_flow_diag` — the canonized block forward
routes both residual additions through recordable ``ResidualAdd`` modules
(``backbone.blocks.{b}._lrp_res1`` / ``_lrp_res2``), the branch endpoints are
``attn.proj_drop`` / ``mlp.drop2``, and ``R_skip = R_add - R_branch`` exactly —
but reduces relevance **per token** (sum over embedding dims; signed and
absolute) instead of per dim, giving arrays of shape (site, sample, token).

In the same attribution forward, plain forward hooks record the per-token L2
norm of the residual stream at 13 cuts (pre-block-0 embedding + every block
output); high-norm "register" tokens are detected from these norms
(per block, per sample: patch token is an outlier if
``norm > mean + 4*sd`` over the 196 patch tokens; CLS excluded).

Predictions tested (hypothesis H_A):

1. Under ``cp_lrp_baseline`` the branch fraction
   ``f = |R_branch| / (|R_branch| + |R_skip|)`` at outlier tokens is even lower
   than at normal tokens, in the blocks where outliers live (the ResidualRatio
   rule pushes relevance into the skip where the stream norm explodes).
2. Attributing the full bilinear attention (``attnlrp_baseline``) reduces the
   input-relevance mass concentrated on outlier patches relative to
   ``cp_lrp_baseline`` (metric: concentration ratio = share of total |input R|
   inside outlier patches / their area share).

Model: timm ViT-B/16 ImageNet-1k-pretrained (``model_io.load_probe`` tag
``imagenet``); data: ImageNet val (HF mirror), ``n_per_class=10``; N
class-diverse correctly-classified images, true-class conditioning.

Pipeline (GPU steps are chunkable so a shared GPU can interleave)::

    python -m experiments.scripts.registers_token_flow select   --n-samples 64
    python -m experiments.scripts.registers_token_flow flow     --start 0 --stop 16
    ...                                                         # more chunks
    python -m experiments.scripts.registers_token_flow outliers  # CPU: merge+stats
    python -m experiments.scripts.registers_token_flow ablate   --config cp_lrp_baseline
    ...                                                         # other composites
    python -m experiments.scripts.registers_token_flow report    # CPU: figures+note
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
FIG_DIR = REPO_ROOT / "figures" / "registers" / "step2_flow"
NOTE_PATH = REPO_ROOT / "research" / "registers" / "step2_token_flow.md"

ABLATE_CONFIGS = ("cp_lrp_baseline", "attnlrp_baseline")
SD_K = 4.0          # outlier criterion: norm > mean + SD_K * sd over patch tokens
N_TOK = 197         # ViT-B/16 @224: CLS + 14*14 patches
N_PATCH = 196


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ─────────────────────────────────────────────────────────────────────────────
# Shared GPU-side setup
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_data(device: str):
    """ViT-B/16 ImageNet-pretrained probe + un-normalized ImageNet val subset
    (10 per class), exactly the crp_gallery loading path."""
    from experiments.model_io import DATASETS, load_probe, backbone_transforms
    from experiments.datasets import load as load_dataset

    model, ck, ck_path = load_probe("imagenet", device, base="vit_base")
    transform, normalize = backbone_transforms(model.backbone)
    ds_name, ds_kw, _ = DATASETS["imagenet"]
    ds = load_dataset(ds_name, root=REPO_ROOT / "data", transform=transform,
                      **{**ds_kw, "n_per_class": 10})
    return model, ck, ck_path, ds, normalize


def pick_class_diverse(ds, n: int, seed: int = 0) -> List[int]:
    """Round-robin over classes (copied from residual_flow_diag) so candidates
    are class-diverse."""
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


def sites_for(n_blocks: int):
    sites = []
    for b in range(n_blocks):
        sites.append((b, "attn", f"backbone.blocks.{b}._lrp_res1",
                      f"backbone.blocks.{b}.attn.proj_drop"))
        sites.append((b, "mlp", f"backbone.blocks.{b}._lrp_res2",
                      f"backbone.blocks.{b}.mlp.drop2"))
    return sites


class StreamNormRecorder:
    """Forward hooks capturing per-token L2 norms of the residual stream at
    13 cuts: cut 0 = input to block 0 (patch embed + CLS + pos), cut b+1 =
    output of block b. Values are pure forward quantities (composite
    independent); hooks overwrite on repeated forwards."""

    def __init__(self, backbone):
        self.norms: Dict[int, np.ndarray] = {}
        self.handles = []
        blocks = backbone.blocks

        def pre_hook(mod, args):
            x = args[0]
            self.norms[0] = x.detach().norm(dim=-1).cpu().numpy()

        self.handles.append(blocks[0].register_forward_pre_hook(pre_hook))
        for b in range(len(blocks)):
            def hook(mod, args, out, b=b):
                self.norms[b + 1] = out.detach().norm(dim=-1).cpu().numpy()
            self.handles.append(blocks[b].register_forward_hook(hook))

    def stack(self) -> np.ndarray:                     # (n_cuts, B, N_TOK)
        return np.stack([self.norms[c] for c in sorted(self.norms)])

    def remove(self):
        for h in self.handles:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# select — N class-diverse correctly-classified images (GPU, quick)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_select(args):
    import torch
    device = args.device
    model, ck, ck_path, ds, normalize = load_model_and_data(device)
    cand = pick_class_diverse(ds, args.n_candidates, seed=args.seed)
    keep_idx, keep_y = [], []
    bs = args.batch_size
    with torch.no_grad():
        for i0 in range(0, len(cand), bs):
            chunk = cand[i0:i0 + bs]
            xs, ys = zip(*[(ds[i][0], int(ds[i][1])) for i in chunk])
            x = normalize(torch.stack(list(xs)).to(device))
            pred = model(x).argmax(-1).cpu()
            for j, (i, y) in enumerate(zip(chunk, ys)):
                if int(pred[j]) == y and len(keep_idx) < args.n_samples:
                    keep_idx.append(i)
                    keep_y.append(y)
            if len(keep_idx) >= args.n_samples:
                break
    if len(keep_idx) < args.n_samples:
        raise RuntimeError(f"only {len(keep_idx)} correct of {len(cand)} candidates")
    sel = {"ds_indices": keep_idx, "targets": keep_y, "n_samples": len(keep_idx),
           "seed": args.seed, "checkpoint": str(ck_path),
           "note": "class-diverse round-robin candidates, first n correctly classified",
           "generated": _now()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "selection.json"
    p.write_text(json.dumps(sel, indent=2))
    print(f"selected {len(keep_idx)} correctly-classified class-diverse images -> {p}")


# ─────────────────────────────────────────────────────────────────────────────
# flow — per-token skip/branch relevance + stream norms (GPU, chunked)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_flow(args):
    import torch
    import lrp_configs
    from crp.attribution import CondAttribution

    device = args.device
    sel = json.loads((OUT_DIR / "selection.json").read_text())
    idxs = sel["ds_indices"][args.start:args.stop]
    ys_all = sel["targets"][args.start:args.stop]
    model, ck, ck_path, ds, normalize = load_model_and_data(device)
    n_blocks = len(model.backbone.blocks)
    cfg = lrp_configs.get(args.config)
    attribution = CondAttribution(model)
    sites = sites_for(n_blocks)
    record = sorted({l for _, _, a, br in sites for l in (a, br)})
    check_layers = ["backbone.blocks.0.ls1", f"backbone.blocks.{n_blocks - 1}.ls1"]

    S = len(idxs)
    branch_abs = np.zeros((len(sites), S, N_TOK), np.float32)
    skip_abs = np.zeros_like(branch_abs)
    branch_signed = np.zeros_like(branch_abs)
    skip_signed = np.zeros_like(branch_abs)
    tot_add = np.zeros((len(sites), S), np.float32)
    stream_norm = np.zeros((n_blocks + 1, S, N_TOK), np.float32)
    logits = np.zeros(S, np.float32)
    endpoint_err = -1.0

    rec_norm = StreamNormRecorder(model.backbone)
    bs = args.batch_size
    for i0 in range(0, S, bs):
        chunk = idxs[i0:i0 + bs]
        ys = ys_all[i0:i0 + bs]
        xs = [ds[i][0] for i in chunk]
        x = torch.stack(xs).to(device)
        xin = normalize(x).requires_grad_(True)
        conds = [{"y": [y]} for y in ys]
        rec = record + (check_layers if (i0 == 0 and args.start == 0) else [])
        res = attribution(xin, conds, cfg.composite(), record_layer=rec)
        missing = [l for l in record if l not in res.relevances]
        if missing:
            raise RuntimeError(f"recording failed for layers: {missing}")
        if i0 == 0 and args.start == 0:
            e1 = (res.relevances["backbone.blocks.0.attn.proj_drop"]
                  - res.relevances["backbone.blocks.0.ls1"]).abs().max()
            e2 = (res.relevances[f"backbone.blocks.{n_blocks-1}.attn.proj_drop"]
                  - res.relevances[f"backbone.blocks.{n_blocks-1}.ls1"]).abs().max()
            endpoint_err = float(torch.maximum(e1, e2))
            print(f"endpoint identity err: {endpoint_err:.3e}")
        pred = res.prediction.detach()
        for j, y in enumerate(ys):
            logits[i0 + j] = float(pred[j, y])
        sl = slice(i0, i0 + len(chunk))
        for si, (b, kind, add_l, br_l) in enumerate(sites):
            r_add = res.relevances[add_l]              # (B, N, D)
            r_br = res.relevances[br_l]
            r_skip = r_add - r_br                      # exact elementwise split
            branch_abs[si, sl] = r_br.abs().sum(-1).cpu().numpy()
            skip_abs[si, sl] = r_skip.abs().sum(-1).cpu().numpy()
            branch_signed[si, sl] = r_br.sum(-1).cpu().numpy()
            skip_signed[si, sl] = r_skip.sum(-1).cpu().numpy()
            tot_add[si, sl] = r_add.sum((1, 2)).cpu().numpy()
        stream_norm[:, sl] = rec_norm.stack()
        print(f"  batch {i0 // bs + 1}/{(S + bs - 1) // bs} done", flush=True)
    rec_norm.remove()

    meta = {"config": args.config, "checkpoint": str(ck_path),
            "start": args.start, "stop": args.start + S,
            "n_blocks": n_blocks, "endpoint_identity_err": endpoint_err,
            "composite_desc": cfg.description, "generated": _now()}
    out = OUT_DIR / f"flow_{args.config}_part{args.start:03d}.npz"
    np.savez_compressed(
        out, branch_abs=branch_abs, skip_abs=skip_abs,
        branch_signed=branch_signed, skip_signed=skip_signed, tot_add=tot_add,
        stream_norm=stream_norm, logits=logits,
        site_block=np.array([b for b, _, _, _ in sites], np.int64),
        site_kind=np.array([k for _, k, _, _ in sites]),
        sample_ds_index=np.array(idxs, np.int64),
        sample_target=np.array(ys_all, np.int64),
        meta=np.array(json.dumps(meta)))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# outliers — merge chunks, detect register tokens, Prediction-1 stats (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def merge_flow(config: str):
    parts = sorted(OUT_DIR.glob(f"flow_{config}_part*.npz"))
    if not parts:
        raise FileNotFoundError(f"no flow parts for {config} in {OUT_DIR}")
    zs = [np.load(p, allow_pickle=False) for p in parts]
    keys = ["branch_abs", "skip_abs", "branch_signed", "skip_signed",
            "tot_add", "stream_norm", "logits", "sample_ds_index", "sample_target"]
    axis = {"branch_abs": 1, "skip_abs": 1, "branch_signed": 1, "skip_signed": 1,
            "tot_add": 1, "stream_norm": 1, "logits": 0,
            "sample_ds_index": 0, "sample_target": 0}
    merged = {k: np.concatenate([z[k] for z in zs], axis=axis[k]) for k in keys}
    merged["site_block"] = zs[0]["site_block"]
    merged["site_kind"] = zs[0]["site_kind"]
    metas = [json.loads(str(z["meta"])) for z in zs]
    return merged, metas


def detect_outliers(stream_norm: np.ndarray, sd_k: float = SD_K):
    """Per block, per sample outlier mask over the 196 patch tokens.
    ``stream_norm``: (n_blocks+1, S, N_TOK); cut b+1 = output of block b.
    Returns mask (n_blocks, S, N_PATCH) plus per-block norm stats."""
    n_cuts, S, _ = stream_norm.shape
    n_blocks = n_cuts - 1
    patch = stream_norm[1:, :, 1:]                     # (n_blocks, S, 196)
    mu = patch.mean(-1, keepdims=True)
    sd = patch.std(-1, keepdims=True)
    mask = patch > (mu + sd_k * sd)
    return mask, patch


def bootstrap_median_diff(f_out_by_sample, f_nrm_by_sample, n_boot=2000, seed=0):
    """Cluster bootstrap over samples of median(f_outlier) - median(f_normal).
    Inputs: lists (len S) of per-sample 1-D arrays (possibly empty)."""
    rng = np.random.default_rng(seed)
    S = len(f_out_by_sample)
    diffs = np.empty(n_boot)
    for it in range(n_boot):
        pick = rng.integers(0, S, S)
        fo = np.concatenate([f_out_by_sample[i] for i in pick]) if S else np.array([])
        fn = np.concatenate([f_nrm_by_sample[i] for i in pick])
        diffs[it] = (np.median(fo) - np.median(fn)) if fo.size else np.nan
    diffs = diffs[~np.isnan(diffs)]
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) \
        if diffs.size else (float("nan"), float("nan"))


def cmd_outliers(args):
    from scipy.stats import mannwhitneyu
    merged, metas = merge_flow(args.config)
    ba, sa = merged["branch_abs"], merged["skip_abs"]  # (n_sites, S, 197)
    site_block, site_kind = merged["site_block"], merged["site_kind"]
    stream_norm = merged["stream_norm"]
    n_sites, S, _ = ba.shape
    n_blocks = stream_norm.shape[0] - 1
    mask, patch_norm = detect_outliers(stream_norm)    # (n_blocks, S, 196)

    f = ba / (ba + sa + 1e-12)                         # (n_sites, S, 197)
    f_patch = f[:, :, 1:]                              # exclude CLS

    rows = []
    for si in range(n_sites):
        b, kind = int(site_block[si]), str(site_kind[si])
        m = mask[b]                                    # (S, 196)
        f_o = f_patch[si][m]
        f_n = f_patch[si][~m]
        n_out = int(m.sum())
        row = {"block": b, "kind": kind, "n_outlier_tokens": n_out,
               "n_images_with_outliers": int(m.any(1).sum()),
               "median_f_normal": float(np.median(f_n))}
        if n_out >= 8:
            u, p = mannwhitneyu(f_o, f_n, alternative="less")
            fo_s = [f_patch[si, s][mask[b, s]] for s in range(S)]
            fn_s = [f_patch[si, s][~mask[b, s]] for s in range(S)]
            lo, hi = bootstrap_median_diff(fo_s, fn_s, seed=args.seed)
            row.update(median_f_outlier=float(np.median(f_o)),
                       mw_p_less=float(p),
                       diff_ci95=[lo, hi])
        rows.append(row)

    # Outlier "home" blocks: where the average image has >= 0.5 outlier tokens.
    per_block_rate = mask.sum(-1).mean(-1)             # (n_blocks,) tokens/image
    home_blocks = [int(b) for b in range(n_blocks) if per_block_rate[b] >= 0.5]
    # Per-image union of outlier patches over home blocks -> ablation set.
    union = mask[home_blocks].any(0) if home_blocks else mask.any(0)   # (S,196)
    n_union = union.sum(-1)
    order = np.argsort(-n_union)
    chosen = [int(i) for i in order[:args.n_ablate] if n_union[i] > 0]
    ablate = {
        "sample_pos": chosen,
        "ds_indices": [int(merged["sample_ds_index"][i]) for i in chosen],
        "targets": [int(merged["sample_target"][i]) for i in chosen],
        "outlier_patches": [np.flatnonzero(union[i]).tolist() for i in chosen],
        "home_blocks": home_blocks,
        "criterion": f"norm > mean + {SD_K}*sd over 196 patch tokens, per block "
                     f"per sample, CLS excluded; union over home blocks "
                     f"(>=0.5 outlier tokens/image)",
        "generated": _now(),
    }
    (OUT_DIR / "ablate_selection.json").write_text(json.dumps(ablate, indent=2))

    out = OUT_DIR / f"token_flow_analysis_{args.config}.npz"
    np.savez_compressed(
        out, outlier_mask=mask, patch_norm=patch_norm,
        f_patch=f_patch, site_block=site_block, site_kind=site_kind,
        per_block_rate=per_block_rate,
        stats=np.array(json.dumps(rows)),
        meta=np.array(json.dumps({"config": args.config, "n_samples": S,
                                  "sd_k": SD_K, "home_blocks": home_blocks,
                                  "generated": _now()})))
    print(json.dumps(rows, indent=2))
    print(f"home blocks (>=0.5 outlier tokens/image): {home_blocks}")
    print(f"ablation set: {len(chosen)} images, "
          f"outlier patches/image median {np.median(n_union[chosen]):.1f}")
    print(f"saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ablate — full-model input heatmaps under one composite (GPU, per config)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ablate(args):
    import torch
    import lrp_configs
    from crp.attribution import CondAttribution

    device = args.device
    ab = json.loads((OUT_DIR / "ablate_selection.json").read_text())
    idxs, ys_all = ab["ds_indices"], ab["targets"]
    model, ck, ck_path, ds, normalize = load_model_and_data(device)
    cfg = lrp_configs.get(args.config)
    attribution = CondAttribution(model)

    M = len(idxs)
    heat = np.zeros((M, 224, 224), np.float32)
    imgs = np.zeros((M, 3, 224, 224), np.float32)
    bs = args.batch_size
    for i0 in range(0, M, bs):
        chunk = idxs[i0:i0 + bs]
        ys = ys_all[i0:i0 + bs]
        xs = [ds[i][0] for i in chunk]
        x = torch.stack(xs).to(device)
        xin = normalize(x).requires_grad_(True)
        conds = [{"y": [y]} for y in ys]
        res = attribution(xin, conds, cfg.composite())
        heat[i0:i0 + len(chunk)] = res.heatmap.detach().cpu().numpy()
        imgs[i0:i0 + len(chunk)] = x.detach().cpu().numpy()
        print(f"  batch {i0 // bs + 1}/{(M + bs - 1) // bs} done", flush=True)

    out = OUT_DIR / f"ablate_heatmaps_{args.config}.npz"
    np.savez_compressed(out, heatmap=heat, image=imgs,
                        ds_indices=np.array(idxs), targets=np.array(ys_all),
                        meta=np.array(json.dumps({"config": args.config,
                                                  "generated": _now()})))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# report — Prediction 1+2 figures & markdown note (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def patch_abs(heat: np.ndarray) -> np.ndarray:
    """(M, 224, 224) signed pixel heatmap -> (M, 196) per-patch sum of |R|."""
    M = heat.shape[0]
    h = np.abs(heat).reshape(M, 14, 16, 14, 16).sum((2, 4))
    return h.reshape(M, 196)


def concentration(heat: np.ndarray, outlier_patches: List[List[int]]):
    """Per image: (share of |R| in outlier patches) / (their area share)."""
    pa = patch_abs(heat)                               # (M, 196)
    out = []
    for i, patches in enumerate(outlier_patches):
        k = len(patches)
        if k == 0:
            out.append(np.nan)
            continue
        share = pa[i, patches].sum() / (pa[i].sum() + 1e-12)
        out.append(share / (k / N_PATCH))
    return np.array(out)


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(OUT_DIR / f"token_flow_analysis_{args.config}.npz", allow_pickle=False)
    stats = json.loads(str(z["stats"]))
    meta = json.loads(str(z["meta"]))
    mask, patch_norm = z["outlier_mask"], z["patch_norm"]
    per_block_rate = z["per_block_rate"]
    n_blocks = mask.shape[0]
    S = mask.shape[1]
    ab = json.loads((OUT_DIR / "ablate_selection.json").read_text())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # palette: categorical blue / red / amber (CVD-checked: min pairwise
    # OKLab dE*100 = 19.7 normal, 14.1 deutan, 21.0 protan); ink/chrome
    C_NRM, C_OUT, C_ACC = "#2a78d6", "#e34948", "#e8a13c"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {FIG_DIR / name}.png/.pdf")

    # ── fig 1: where outliers live + how extreme their norms are ────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.bar(range(n_blocks), per_block_rate, color=C_OUT, width=0.7)
    ax.set_xlabel("block", color=INK2)
    ax.set_ylabel("outlier tokens / image", color=INK2)
    ax.set_title(f"register-token prevalence (norm > mean+{SD_K:g}sd, CLS excl.)",
                 fontsize=10, color=INK)
    style(ax)
    ax = axes[1]
    med_all = np.median(patch_norm, axis=(1, 2))
    mx = np.array([np.median([patch_norm[b, s].max() for s in range(S)])
                   for b in range(n_blocks)])
    ax.plot(range(n_blocks), med_all, color=C_NRM, lw=2, label="median token norm")
    ax.plot(range(n_blocks), mx, color=C_OUT, lw=2, label="median per-image max")
    ax.set_yscale("log")
    ax.set_xlabel("block (output)", color=INK2)
    ax.set_ylabel("token L2 norm", color=INK2)
    ax.set_title("residual-stream token norms", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    style(ax)
    fig.suptitle(f"ViT-B/16 ImageNet, N={S} images", fontsize=11, color=INK, y=1.02)
    save(fig, "fig1_outlier_prevalence")

    # ── fig 2: Prediction 1 — branch fraction f, outlier vs normal ──────────
    for kind in ("attn", "mlp"):
        rows = [r for r in stats if r["kind"] == kind]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        xs = [r["block"] for r in rows]
        ax.plot(xs, [r["median_f_normal"] for r in rows], "o-", color=C_NRM,
                lw=2, ms=4, label="normal tokens")
        xo = [r["block"] for r in rows if "median_f_outlier" in r]
        yo = [r["median_f_outlier"] for r in rows if "median_f_outlier" in r]
        ax.plot(xo, yo, "o-", color=C_OUT, lw=2, ms=5, label="outlier tokens")
        for r in rows:
            if "diff_ci95" in r:
                lo, hi = r["diff_ci95"]
                base = r["median_f_normal"]
                ax.vlines(r["block"], base + lo, base + hi, color=C_OUT,
                          alpha=0.45, lw=3)
        ax.set_xlabel("block", color=INK2)
        ax.set_ylabel("median branch fraction f", color=INK2)
        ax.set_ylim(0, None)
        ax.set_title(f"{kind} residual: f = |R_branch| / (|R_branch|+|R_skip|) "
                     f"per token — {args.config}", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
        style(ax)
        save(fig, f"fig2_branch_fraction_{kind}")

    # ── Prediction 2: concentration ratios per composite ────────────────────
    conc, heatmaps, images = {}, {}, None
    for cfgname in ABLATE_CONFIGS:
        p = OUT_DIR / f"ablate_heatmaps_{cfgname}.npz"
        if not p.exists():
            print(f"missing {p}, skipping {cfgname}")
            continue
        za = np.load(p, allow_pickle=False)
        conc[cfgname] = concentration(za["heatmap"], ab["outlier_patches"])
        heatmaps[cfgname] = za["heatmap"]
        images = za["image"]

    if conc:
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        names = list(conc)
        colors = {"cp_lrp_baseline": C_OUT, "attnlrp_baseline": C_ACC}
        summary = {}
        for k, name in enumerate(names):
            v = conc[name][~np.isnan(conc[name])]
            med = float(np.median(v))
            boots = [np.median(rng.choice(v, v.size)) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            summary[name] = {"median": med, "ci95": [float(lo), float(hi)],
                             "per_image": [float(x) for x in conc[name]]}
            jit = (rng.random(v.size) - 0.5) * 0.25
            ax.scatter(k + jit, v, s=16, color=colors.get(name, C_NRM), alpha=0.55)
            ax.hlines(med, k - 0.22, k + 0.22, color=INK, lw=2.4)
            ax.vlines(k, lo, hi, color=INK, lw=1.2)
        ax.axhline(1.0, color=INK2, lw=1, ls="--")
        ax.text(len(names) - 0.45, 1.0, "area-proportional", fontsize=8,
                color=INK2, va="bottom", ha="right")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("cp_lrp_baseline", "cp_lrp\nbaseline")
                            .replace("attnlrp_baseline", "attnlrp\nbaseline")
                            for n in names], fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel("|R| concentration on outlier patches\n(share / area share)",
                      color=INK2)
        ax.set_title(f"input-relevance mass on register patches, "
                     f"M={len(ab['ds_indices'])} images", fontsize=10, color=INK)
        style(ax)
        save(fig, "fig3_concentration_by_composite")
        (OUT_DIR / "concentration_summary.json").write_text(
            json.dumps(summary, indent=2))
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_image"}
                          for k, v in summary.items()}, indent=2))

    # ── fig 4: qualitative examples — heatmaps with outlier patches boxed ───
    if conc and images is not None:
        n_ex = min(4, images.shape[0])
        order = np.argsort(-np.nan_to_num(conc["cp_lrp_baseline"]))[:n_ex]
        fig, axes = plt.subplots(n_ex, 1 + len(heatmaps),
                                 figsize=(2.3 * (1 + len(heatmaps)), 2.3 * n_ex))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(order):
            axes[r, 0].imshow(np.transpose(images[i], (1, 2, 0)).clip(0, 1))
            axes[r, 0].set_ylabel(f"img {ab['ds_indices'][i]}", fontsize=8,
                                  color=INK2)
            for c, name in enumerate(heatmaps):
                h = heatmaps[name][i]
                lim = np.percentile(np.abs(h), 99.5) + 1e-12
                axes[r, c + 1].imshow(h, cmap="bwr", vmin=-lim, vmax=lim)
                for pidx in ab["outlier_patches"][i]:
                    py, px = divmod(pidx, 14)
                    axes[r, c + 1].add_patch(plt.Rectangle(
                        (px * 16 - .5, py * 16 - .5), 16, 16, fill=False,
                        edgecolor="#0b0b0b", lw=1.1))
                if r == 0:
                    axes[r, c + 1].set_title(name, fontsize=7.5, color=INK2)
        if n_ex:
            axes[0, 0].set_title("input", fontsize=7.5, color=INK2)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        save(fig, "fig4_example_heatmaps")

    print("report done")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "flow", "outliers", "ablate", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="cp_lrp_baseline")
        p.add_argument("--device", default="cuda")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--n-samples", type=int, default=64)
        p.add_argument("--batch-size", type=int, default=8)
        if name == "select":
            p.add_argument("--n-candidates", type=int, default=256)
        if name == "flow":
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--stop", type=int, default=int(1e9))
        if name == "outliers":
            p.add_argument("--n-ablate", type=int, default=16)
    args = ap.parse_args()
    {"select": cmd_select, "flow": cmd_flow, "outliers": cmd_outliers,
     "ablate": cmd_ablate, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
