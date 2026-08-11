"""Registers E2 (clean redo): which explainability methods highlight the
register/scratch-pad tokens, quantified by set overlap.

RQ: when a method produces an input saliency map, how strongly does its set of
"outlier-hot" patches overlap the register set identified from activations —
and does LRP overlap more than other standard methods?

Protocol (symmetric detection rule on both sides, for comparability):

* Activation register set ``A`` — per-sample ``mu + 4*sd`` rule on
  residual-stream token L2 norms at ANY of the 24 sites (per block:
  after-attn-add = forward PRE-hook on ``blocks[i].norm2``; after-mlp-add =
  ``blocks[i]`` output). mu/sd over the sample's own 196 patch tokens (CLS
  excluded); a patch is in ``A`` iff flagged at >= 1 site.
* Saliency outlier set ``S_m`` for method m — the SAME rule applied to the 196
  per-patch saliency values of that image (mu/sd over the 196 values; patch
  flagged iff value > mu + 4*sd).
* Per image: IoU(A, S_m), recall |A∩S|/|A|, precision |A∩S|/|S|.

Methods (all class-conditional on the true class where applicable; per-patch
aggregation of pixel-space maps = sum of |values| over the 16x16 patch):

* ``lrp``       — cp_lrp_baseline composite, ``CondAttribution``, condition
                  ``[{"y": [target]}]``, full-model input heatmap.
* ``chefer``    — Chefer, Gur & Wolf (CVPR 2021) transformer attribution:
                  per block ``A_bar = I + mean_heads((grad_A * A)^+)``,
                  row-normalized, chained over blocks; CLS row over patches.
                  A captured on the stock timm attention (``fused_attn=False``,
                  forward hook on ``attn.attn_drop`` output, graph kept);
                  grad_A = d(target logit)/dA via ``autograd.grad``.
* ``rollout``   — Abnar & Zuidema attention rollout: same chaining with the raw
                  attention (no gradients).
* ``occlusion`` — per patch, the 16x16 pixel patch replaced by the image mean
                  color; saliency = (p_clean - p_occluded) of the true class,
                  clamped at 0.

Models: vit_base_imagenet (PRIMARY, timm ViT-B/16, val subset n_per_class=10)
and vit_small_funny_birds (probe ckpt, test split); N=64 correctly-classified
images each, round-robin class-diverse order (seed 0, step-1c scheme).

Run (GPU stages one at a time, each under the shared GPU lock)::

    P=/home/claude/venvs/zennit-crp/bin/python
    $P -m experiments.scripts.registers_e2_overlap select  --model vit_base_imagenet
    $P -m experiments.scripts.registers_e2_overlap attn    --model vit_base_imagenet
    $P -m experiments.scripts.registers_e2_overlap lrp     --model vit_base_imagenet
    $P -m experiments.scripts.registers_e2_overlap occlude --model vit_base_imagenet --start 0 --end 32
    $P -m experiments.scripts.registers_e2_overlap occlude --model vit_base_imagenet --start 32 --end 64
    $P -m experiments.scripts.registers_e2_overlap analyze --model vit_base_imagenet   # CPU
    $P -m experiments.scripts.registers_e2_overlap figures --model vit_base_imagenet   # CPU

Outputs: ``data/results/registers/e2_overlap_<model>.npz`` + ``e2_summary.json``,
figures ``figures/registers/e2_overlap/`` (png+pdf), report
``research/registers/e2_saliency_overlap.md`` (written separately).
All stages are idempotent (recompute + overwrite).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import typer

from experiments.scripts.registers_step1c_redo import (
    MODELS, pick_class_diverse, _ds_labels)

REPO = Path(__file__).resolve().parents[2]
RES_DIR = REPO / "data" / "results" / "registers"
FIG_DIR = REPO / "figures" / "registers" / "e2_overlap"
PAPER_FIG_DIR = Path("/home/claude/workspaces/crp-paper/iclr2026/journal-figures")

GRID, PATCH = 14, 16
N_PATCH = GRID * GRID
SD_K = 4.0
METHODS = ("lrp", "chefer", "rollout", "occlusion")
METHOD_LABELS = {
    "lrp": "LRP (cp_lrp_baseline)",
    "chefer": "Chefer attribution",
    "rollout": "attention rollout",
    "occlusion": "occlusion (Δp⁺)",
}
# fixed categorical order/colors (colorblind-safe, validated repo-adjacent set)
METHOD_COLORS = {"lrp": "#4878a8", "chefer": "#d1615d",
                 "rollout": "#6a9f58", "occlusion": "#967662"}
COL_A, COL_S = "#ff00ff", "#00c8e0"          # magenta = A, cyan = S_m

app = typer.Typer(add_completion=False, help=__doc__)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def select_path(m): return RES_DIR / f"e2_select_{m}.npz"
def sal_path(m, meth): return RES_DIR / f"e2_sal_{m}_{meth}.npz"
def occl_part_path(m, s, e): return RES_DIR / f"e2_occl_{m}_{s:03d}_{e:03d}.npz"
def overlap_path(m): return RES_DIR / f"e2_overlap_{m}.npz"


# ─────────────────────────────────────────────────────────────────────────────
# shared loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_ds(model_key: str, device: str):
    from experiments.models import MODELS as MODEL_ZOO, backbone_transforms
    from experiments.datasets import load_eval_dataset
    spec = MODELS[model_key]
    ckpt = spec["checkpoint"]
    model = MODEL_ZOO[f"{spec['base']}_{spec['dataset']}"](
        **({"checkpoint": ckpt} if ckpt else {}), device=device)
    label = f"{spec['base']} · {model.head_name} · {spec['dataset']}"
    transform, normalize = backbone_transforms(model.backbone)
    ds = load_eval_dataset(spec["dataset"], transform, extra_kwargs=spec["extra"])
    assert int(model.backbone.num_prefix_tokens) == 1, "expected exactly one CLS token"
    return model, normalize, ds, label


def _load_ds_only(model_key: str):
    """Dataset + transform without putting the model on GPU (figure stages)."""
    import timm
    from experiments.datasets import load_eval_dataset
    from experiments.models import backbone_transforms
    spec = MODELS[model_key]
    arch = {"vit_base": "vit_base_patch16_224", "vit_small": "vit_small_patch16_224"}
    tm = timm.create_model(arch[spec["base"]], pretrained=False)
    transform, _ = backbone_transforms(tm)
    return load_eval_dataset(spec["dataset"], transform, extra_kwargs=spec["extra"])


def _sel_batches(model_key: str, ds, batch: int):
    sel = np.load(select_path(model_key))
    idxs, tgts = sel["ds_indices"].tolist(), sel["targets"].tolist()
    for s in range(0, len(idxs), batch):
        x = torch.stack([ds[i][0] for i in idxs[s:s + batch]])
        yield s, x, tgts[s:s + batch]


def _to_patch(sal: torch.Tensor) -> torch.Tensor:
    """(B, 224, 224) pixel map -> (B, 196) sum of |values| per 16x16 patch."""
    b = sal.shape[0]
    return sal.abs().reshape(b, GRID, PATCH, GRID, PATCH).sum(dim=(2, 4)).reshape(b, N_PATCH)


def _save_sal(model_key: str, meth: str, maps: torch.Tensor, note: str):
    RES_DIR.mkdir(parents=True, exist_ok=True)
    sel = np.load(select_path(model_key))
    np.savez_compressed(sal_path(model_key, meth),
                        patch=maps.numpy().astype(np.float32),
                        ds_indices=sel["ds_indices"], targets=sel["targets"],
                        meta=np.array([note, f"computed={_now()}"]))
    print(f"saved {sal_path(model_key, meth)} ({maps.shape[0]} images)")


# ─────────────────────────────────────────────────────────────────────────────
# stage 1: select — correctly-classified images + activation register set A
# ─────────────────────────────────────────────────────────────────────────────

class ResidualSiteRecorder:
    """24 residual-stream sites: per block, after-attn-add (forward PRE-hook on
    ``blocks[i].norm2`` — its input is x after the attention residual add) and
    after-mlp-add (``blocks[i]`` forward output). Records token L2 norms."""

    def __init__(self, backbone):
        self.n_blocks = len(backbone.blocks)
        self.store: Dict[int, torch.Tensor] = {}
        self.handles = []
        for b, blk in enumerate(backbone.blocks):
            def pre(mod, args, b=b):
                self.store[2 * b] = args[0].detach().float().norm(dim=-1).cpu()

            def post(mod, args, out, b=b):
                self.store[2 * b + 1] = out.detach().float().norm(dim=-1).cpu()

            self.handles.append(blk.norm2.register_forward_pre_hook(pre))
            self.handles.append(blk.register_forward_hook(post))

    def stack(self) -> torch.Tensor:                     # (24, B, 197)
        return torch.stack([self.store[s] for s in range(2 * self.n_blocks)])

    def remove(self):
        for h in self.handles:
            h.remove()


def register_flags(norms: np.ndarray):
    """norms (24, N, 197) -> (site flags (24, N, 196), union A (N, 196)).
    Per-sample per-site mu+4sd over the 196 patch tokens, CLS excluded."""
    patch = norms[:, :, 1:]
    mu = patch.mean(-1, keepdims=True)
    sd = patch.std(-1, keepdims=True)
    flags = patch > mu + SD_K * sd
    return flags, flags.any(0)


def saliency_flags(patch_maps: np.ndarray) -> np.ndarray:
    """SAME rule on the saliency side: (N, 196) -> bool (N, 196),
    value > mu + 4*sd of the image's own 196 per-patch values."""
    mu = patch_maps.mean(-1, keepdims=True)
    sd = patch_maps.std(-1, keepdims=True)
    return patch_maps > mu + SD_K * sd


@app.command()
def select(model: str = typer.Option(...), n_samples: int = typer.Option(64),
           seed: int = typer.Option(0), batch: int = typer.Option(32),
           device: str = typer.Option("cuda")):
    """Scan class-diverse order, keep first N correctly-classified images,
    record 24-site token norms and the activation register set A."""
    mdl, normalize, ds, label = _load_model_ds(model, device)
    order = pick_class_diverse(ds, min(len(ds), 8 * n_samples), seed=seed)
    labels = _ds_labels(ds)

    rec = ResidualSiteRecorder(mdl.backbone)
    kept_idx: List[int] = []
    kept_norms, kept_probs = [], []
    n_seen = 0
    with torch.no_grad():
        for s in range(0, len(order), batch):
            idxs = order[s:s + batch]
            x = torch.stack([ds[i][0] for i in idxs]).to(device)
            y = torch.tensor([labels[i] for i in idxs])
            logits = mdl(normalize(x)).cpu()
            probs = logits.softmax(-1)[range(len(idxs)), y]
            correct = logits.argmax(-1) == y
            norms = rec.stack()                                  # (24, B, 197)
            n_seen += len(idxs)
            for j in range(len(idxs)):
                if correct[j] and len(kept_idx) < n_samples:
                    kept_idx.append(idxs[j])
                    kept_norms.append(norms[:, j])
                    kept_probs.append(float(probs[j]))
            if len(kept_idx) >= n_samples:
                break
    rec.remove()
    if len(kept_idx) < n_samples:
        raise RuntimeError(f"only {len(kept_idx)} correct of {n_seen} scanned")

    norms = torch.stack(kept_norms, dim=1).numpy()               # (24, 64, 197)
    site_flags, A = register_flags(norms)
    tg = np.array([labels[i] for i in kept_idx], dtype=np.int64)
    print(f"{label}: kept {len(kept_idx)}/{n_seen} scanned; "
          f"|A| mean {A.sum(1).mean():.2f} (min {A.sum(1).min()}, max {A.sum(1).max()}), "
          f"images with empty A: {(A.sum(1) == 0).sum()}")

    RES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        select_path(model),
        ds_indices=np.array(kept_idx, dtype=np.int64), targets=tg,
        clean_prob=np.array(kept_probs, dtype=np.float32),
        norms=norms.astype(np.float16), site_flags=site_flags, A=A,
        meta=np.array([
            f"model={label}", f"seed={seed}", f"selection=round-robin class-diverse,"
            f" first {n_samples} correctly classified",
            "sites: 2i = after-attn-add (pre-hook blocks[i].norm2 input), "
            "2i+1 = after-mlp-add (blocks[i] output); token0=CLS excluded from stats",
            f"A = per-sample per-site norm > mu+{SD_K}*sd, union over 24 sites",
            f"collected={_now()}"]))
    print(f"saved {select_path(model)}")


# ─────────────────────────────────────────────────────────────────────────────
# stage 2a: chefer + rollout (one forward+backward per batch)
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def attn(model: str = typer.Option(...), batch: int = typer.Option(16),
         limit: Optional[int] = typer.Option(None),
         device: str = typer.Option("cuda")):
    """Chefer transformer attribution + attention rollout from one attention
    capture per batch (stock timm attention, fused_attn=False)."""
    mdl, normalize, ds, _ = _load_model_ds(model, device)
    blocks = mdl.backbone.blocks
    out = {"chefer": [], "rollout": []}
    for s, x, tg in _sel_batches(model, ds, batch):
        if limit is not None and s >= limit:
            break
        xn = normalize(x.to(device)).requires_grad_(True)   # force graph
        store: Dict[int, torch.Tensor] = {}
        hooks, prev = [], []
        for b, blk in enumerate(blocks):
            prev.append(blk.attn.fused_attn)
            blk.attn.fused_attn = False

            def f(m, i, o, b=b):
                store[b] = o                                # keep graph, no detach
            hooks.append(blk.attn.attn_drop.register_forward_hook(f))
        try:
            logits = mdl(xn)
            attns = [store[b] for b in range(len(blocks))]  # (B, H, 197, 197) each
            logit = logits[range(len(tg)), tg].sum()
            grads = torch.autograd.grad(logit, attns)
        finally:
            for h in hooks:
                h.remove()
            for blk, p in zip(blocks, prev):
                blk.attn.fused_attn = p

        n_tok = attns[0].shape[-1]
        eye = torch.eye(n_tok, device=device)
        r_ch = r_ro = None
        for a, g in zip(attns, grads):
            a = a.detach()
            cam = (g * a).clamp(min=0).mean(1)              # (B, N, N)
            ab = eye + cam
            ab = ab / ab.sum(-1, keepdim=True)
            r_ch = ab if r_ch is None else ab @ r_ch
            ar = eye + a.mean(1)
            ar = ar / ar.sum(-1, keepdim=True)              # == 0.5A + 0.5I
            r_ro = ar if r_ro is None else ar @ r_ro
        out["chefer"].append(r_ch[:, 0, 1:].float().cpu())
        out["rollout"].append(r_ro[:, 0, 1:].float().cpu())
        print(f"  attn batch @{s} done", flush=True)
    _save_sal(model, "chefer",
              torch.cat(out["chefer"]),
              "Chefer/Gur/Wolf CVPR21 grad-weighted rollout: per block "
              "I + mean_heads((dLogit/dA * A)^+), row-normalized, chained; CLS row. "
              "A from attn.attn_drop output hooks, fused_attn=False, true-class logit")
    _save_sal(model, "rollout",
              torch.cat(out["rollout"]),
              "Abnar&Zuidema rollout: per block row-normalized I + mean_heads(A) "
              "(= 0.5A+0.5I), chained; CLS row; raw attention, no gradients")


# ─────────────────────────────────────────────────────────────────────────────
# stage 2b: LRP
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def lrp(model: str = typer.Option(...), batch: int = typer.Option(8),
        limit: Optional[int] = typer.Option(None),
        device: str = typer.Option("cuda")):
    """Full-model class-conditional cp_lrp_baseline input heatmap, sum|R|/patch."""
    from zennit_extensions.lrp_composites import CPLRPComposite
    from crp.attribution import CondAttribution
    mdl, normalize, ds, _ = _load_model_ds(model, device)
    attribution = CondAttribution(mdl)
    composite_cls = CPLRPComposite
    maps = []
    for s, x, tg in _sel_batches(model, ds, batch):
        if limit is not None and s >= limit:
            break
        xn = normalize(x.to(device)).requires_grad_(True)
        res = attribution(xn, [{"y": [int(t)]} for t in tg], composite_cls())
        maps.append(_to_patch(res.heatmap.detach().cpu()))
        print(f"  lrp batch @{s} done", flush=True)
    _save_sal(model, "lrp", torch.cat(maps),
              "cp_lrp_baseline composite, CondAttribution, condition [{'y':[target]}], "
              "full-model input heatmap, sum|R| per 16x16 patch")


# ─────────────────────────────────────────────────────────────────────────────
# stage 2c: occlusion (chunk with --start/--end to keep lock holds short)
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def occlude(model: str = typer.Option(...), start: int = typer.Option(0),
            end: int = typer.Option(64), batch: int = typer.Option(98),
            device: str = typer.Option("cuda")):
    """Per patch: 16x16 pixels -> image mean color; Δp = p_clean - p_occl of the
    true class, clamped at 0. 196 forwards per image."""
    mdl, normalize, ds, _ = _load_model_ds(model, device)
    sel = np.load(select_path(model))
    idxs, tgts = sel["ds_indices"].tolist(), sel["targets"].tolist()
    end = min(end, len(idxs))
    maps = []
    with torch.no_grad():
        for i in range(start, end):
            x = ds[idxs[i]][0].to(device)                    # (3, 224, 224) in [0,1]
            t = int(tgts[i])
            p_clean = mdl(normalize(x[None])).softmax(-1)[0, t]
            mean_col = x.mean(dim=(1, 2))                    # per-image mean color
            occ = x[None].repeat(N_PATCH, 1, 1, 1)
            for p in range(N_PATCH):
                r, c = divmod(p, GRID)
                occ[p, :, r * PATCH:(r + 1) * PATCH,
                    c * PATCH:(c + 1) * PATCH] = mean_col[:, None, None]
            probs = torch.cat([mdl(normalize(occ[s:s + batch])).softmax(-1)[:, t]
                               for s in range(0, N_PATCH, batch)])
            maps.append((p_clean - probs).clamp(min=0).cpu())
            if (i - start) % 8 == 7:
                print(f"  occlusion image {i + 1}/{end}", flush=True)
    np.savez_compressed(occl_part_path(model, start, end),
                        patch=torch.stack(maps).numpy().astype(np.float32),
                        start=start, end=end)
    print(f"saved {occl_part_path(model, start, end)}")


def _merge_occlusion(model: str, n: int) -> np.ndarray:
    parts = sorted(RES_DIR.glob(f"e2_occl_{model}_*.npz"))
    out = np.full((n, N_PATCH), np.nan, dtype=np.float32)
    for p in parts:
        d = np.load(p)
        out[int(d["start"]):int(d["end"])] = d["patch"]
    if np.isnan(out).any():
        missing = np.flatnonzero(np.isnan(out).any(1))
        raise RuntimeError(f"occlusion incomplete: images {missing[:8]}… missing")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# stage 3: analyze (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _overlap_metrics(A: np.ndarray, S: np.ndarray):
    """Per-image IoU/recall/precision of bool sets (N, 196). NaN where undefined."""
    inter = (A & S).sum(1).astype(float)
    union = (A | S).sum(1).astype(float)
    nA, nS = A.sum(1).astype(float), S.sum(1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
        rec = np.where(nA > 0, inter / nA, np.nan)
        prec = np.where(nS > 0, inter / nS, np.nan)
    return iou, rec, prec


def _boot_ci(v: np.ndarray, n_boot: int = 10000, seed: int = 0):
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@app.command()
def analyze(model: str = typer.Option(...)):
    """Symmetric-rule S_m sets, per-image overlap metrics, bootstrap CIs;
    writes e2_overlap_<model>.npz and updates e2_summary.json."""
    sel = np.load(select_path(model), allow_pickle=True)
    A = sel["A"]
    n = A.shape[0]
    patch_maps = {m: np.load(sal_path(model, m))["patch"] for m in ("lrp", "chefer", "rollout")}
    patch_maps["occlusion"] = _merge_occlusion(model, n)
    for m in METHODS:
        assert patch_maps[m].shape == (n, N_PATCH), (m, patch_maps[m].shape)

    out_npz: dict = dict(ds_indices=sel["ds_indices"], targets=sel["targets"], A=A)
    summary = {"n_images": int(n), "mean_A_size": float(A.sum(1).mean()),
               "n_empty_A": int((A.sum(1) == 0).sum()), "methods": {}}
    print(f"{model}: n={n}, |A| mean {A.sum(1).mean():.2f}, empty A: {summary['n_empty_A']}")
    print(f"{'method':>22} {'IoU mean':>9} {'IoU med':>8} {'recall':>7} "
          f"{'prec':>6} {'|S|':>6} {'IoU 95% CI':>16}")
    for m in METHODS:
        S = saliency_flags(patch_maps[m])
        iou, rec, prec = _overlap_metrics(A, S)
        lo, hi = _boot_ci(iou)
        summary["methods"][m] = dict(
            iou_mean=float(np.nanmean(iou)), iou_median=float(np.nanmedian(iou)),
            iou_ci95=[lo, hi], recall_mean=float(np.nanmean(rec)),
            precision_mean=float(np.nanmean(prec)),
            mean_S_size=float(S.sum(1).mean()), n_empty_S=int((S.sum(1) == 0).sum()))
        out_npz.update({f"S_{m}": S, f"patch_{m}": patch_maps[m].astype(np.float32),
                        f"iou_{m}": iou, f"recall_{m}": rec, f"precision_{m}": prec})
        r = summary["methods"][m]
        print(f"{METHOD_LABELS[m]:>22} {r['iou_mean']:9.3f} {r['iou_median']:8.3f} "
              f"{r['recall_mean']:7.3f} {r['precision_mean']:6.3f} "
              f"{r['mean_S_size']:6.2f} [{lo:.3f}, {hi:.3f}]")

    out_npz["meta"] = np.array([
        "A = activation register set (24-site per-sample mu+4sd union, see e2_select meta)",
        "S_m = same mu+4sd rule on the 196 per-patch saliency values of method m",
        "iou/recall/precision per image; NaN where the denominator set is empty",
        f"analyzed={_now()}"])
    np.savez_compressed(overlap_path(model), **out_npz)
    print(f"saved {overlap_path(model)}")

    sp = RES_DIR / "e2_summary.json"
    full = json.loads(sp.read_text()) if sp.exists() else {}
    full[model] = summary
    full["meta"] = {"updated": _now(), "rule": f"mu+{SD_K}sd per-sample, both sides",
                    "bootstrap": "10k resamples, percentile 95% CI on mean IoU"}
    sp.write_text(json.dumps(full, indent=2))
    print(f"updated {sp}")


# ─────────────────────────────────────────────────────────────────────────────
# stage 4: figures (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _outline(ax, mask2d: np.ndarray, color: str, lw: float, scale: float = 1.0,
             off: float = -0.5, inset: float = 0.0):
    import matplotlib.pyplot as plt
    for (yy, xx) in zip(*np.where(mask2d)):
        ax.add_patch(plt.Rectangle(
            (xx * scale + off + inset, yy * scale + off + inset),
            scale - 2 * inset, scale - 2 * inset,
            fill=False, edgecolor=color, lw=lw))


def _example_page(model: str, rows: List[int], d, ds, stem: str, page_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = MODELS[model]
    ncols = 1 + len(METHODS)
    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(2.35 * ncols, 2.45 * len(rows) + 0.9))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(rows):
        x = ds[int(d["ds_indices"][i])][0].permute(1, 2, 0).numpy()
        A2 = d["A"][i].reshape(GRID, GRID)
        ax = axes[r, 0]
        ax.imshow(x)
        _outline(ax, A2, COL_A, 1.4, scale=PATCH)
        ax.set_ylabel(f"img {i} · class {int(d['targets'][i])}", fontsize=8)
        for c, m in enumerate(METHODS):
            ax = axes[r, 1 + c]
            p2 = d[f"patch_{m}"][i].reshape(GRID, GRID)
            S2 = d[f"S_{m}"][i].reshape(GRID, GRID)
            ax.imshow(p2, cmap="viridis")
            _outline(ax, A2, COL_A, 1.6)
            _outline(ax, S2, COL_S, 1.4, inset=0.12)
            iou = d[f"iou_{m}"][i]
            ax.set_title(f"IoU {iou:.2f}" if np.isfinite(iou) else "IoU n/a",
                         fontsize=8, pad=3)
        if r == 0:
            axes[r, 0].set_title("input + A", fontsize=9)
            for c, m in enumerate(METHODS):
                t = axes[r, 1 + c].get_title()
                axes[r, 1 + c].set_title(f"{METHOD_LABELS[m]}\n{t}", fontsize=8, pad=2)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"E2 {page_label} — {spec['label']}: activation registers (magenta = A) vs "
        f"saliency outliers (cyan = S$_m$), both per-sample $\\mu+4\\sigma$", fontsize=10)
    fig.text(0.5, 0.005,
             "Provenance: A = token-norm outliers, union over 24 residual-stream sites "
             "(after-attn-add = blocks[i].norm2 pre-hook; after-mlp-add = blocks[i] output), "
             "per-sample $\\mu+4\\sigma$ over 196 patches, CLS excluded. Saliency panels = per-patch maps, "
             "true-class conditional: LRP = cp_lrp_baseline composite (CondAttribution, cond {y:[target]}, "
             "sum|R|/16$\\times$16 patch); Chefer = grad-weighted attention rollout "
             "(I + mean$_h$((dlogit/dA$\\odot$A)$^+$), row-norm., chained, CLS row); rollout = raw-attention "
             "rollout (Abnar & Zuidema); occlusion = patch$\\to$image-mean, $\\Delta$p(true class) clamped at 0. "
             "S$_m$ = same $\\mu+4\\sigma$ rule on the 196 saliency values.",
             fontsize=6, ha="center", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=2.4)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig {FIG_DIR / stem}.png/.pdf")


def _iou_box():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in MODELS if overlap_path(m).exists()]
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.5),
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for a, model in enumerate(models):
        ax = axes[0, a]
        d = np.load(overlap_path(model))
        for j, m in enumerate(METHODS):
            v = d[f"iou_{m}"]
            v = v[~np.isnan(v)]
            bp = ax.boxplot(v, positions=[j], widths=0.55, showfliers=False,
                            medianprops=dict(color="#222222", lw=1.6),
                            boxprops=dict(color=METHOD_COLORS[m], lw=1.4),
                            whiskerprops=dict(color=METHOD_COLORS[m], lw=1.1),
                            capprops=dict(color=METHOD_COLORS[m], lw=1.1))
            ax.scatter(np.full(len(v), j) + rng.uniform(-0.16, 0.16, len(v)), v,
                       s=7, color=METHOD_COLORS[m], alpha=0.45, lw=0, zorder=3)
            ax.text(j, 1.03, f"{v.mean():.2f}", ha="center", fontsize=8,
                    color="#333333")
        ax.set_title(MODELS[model]["label"], fontsize=10)
        ax.set_xticks(range(len(METHODS)),
                      ["LRP\n(cp_lrp_baseline)", "Chefer", "attention\nrollout",
                       "occlusion"], fontsize=8)
        ax.set_ylim(-0.04, 1.12)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("per-image IoU(A, S$_m$)", fontsize=9)
    fig.suptitle("E2 — overlap of saliency-outlier patches with activation registers "
                 "(mean above each box; per-sample $\\mu+4\\sigma$ rule on both sides)",
                 fontsize=10)
    fig.text(0.5, 0.0,
             "A = 24-site residual-stream token-norm outliers; S$_m$ = same rule on the method's "
             "196 per-patch saliency values; all methods true-class conditional; N=64 correctly-"
             "classified images per model.", fontsize=6.5, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"e2_iou_box.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig {FIG_DIR / 'e2_iou_box'}.png/.pdf")


@app.command()
def figures(model: str = typer.Option("vit_base_imagenet"),
            copy_paper: bool = typer.Option(False, "--copy-paper")):
    """Example panels (2 pages x 3 images) for --model, IoU box plot over all
    analyzed models; --copy-paper copies the three journal PDFs (exact names)."""
    d = np.load(overlap_path(model))
    ds = _load_ds_only(model)
    sizes = d["A"].sum(1)
    cand = [int(i) for i in range(len(sizes)) if sizes[i] >= 1][:6]
    pre = "e2_examples" if model == "vit_base_imagenet" else f"e2_examples_{model}"
    _example_page(model, cand[:3], d, ds, f"{pre}_p1", "examples p1")
    _example_page(model, cand[3:6], d, ds, f"{pre}_p2", "examples p2")
    _iou_box()
    if copy_paper:
        import shutil
        for name in ("e2_examples_p1.pdf", "e2_examples_p2.pdf", "e2_iou_box.pdf"):
            shutil.copy2(FIG_DIR / name, PAPER_FIG_DIR / name)
            print(f"  copied {PAPER_FIG_DIR / name}")


if __name__ == "__main__":
    app()
