"""Insertion-Deletion saliency benchmark (\\AOPCWorkName{} = DAPC) — the
faithfulness metric of the crp-paper experiment journal
(``exp:insertion-deletion-bench``). See the journal's "Insertion-Deletion
benchmark for saliency masks" section for the formal definition; this module is
its reference implementation + CLI.

WHAT IS MEASURED (as implemented here)
--------------------------------------
For a saliency method ``ψ``, model ``M`` and image ``x`` we split ``x`` into the
model's non-overlapping ``patch×patch`` grid (aligned to the patch-embedding),
aggregate the pixel saliency to one score per patch by **MAX**, and sort patches
by descending score ``S_0 ≥ S_1 ≥ … ≥ S_{N-1}``. ``M(·)`` is the
**predicted-class softmax probability**. We occlude patches by **image-mean fill**
and build two prediction curves in **predicted-class probability**:

* **MoRF** — occlude most-salient-first (``{p_0..p_{k-1}}`` at step ``k``); a
  faithful map makes this drop fastest → smallest area.
* **LeRF** — occlude least-salient-first; drops slowest → largest area.

Each stored curve has length ``N+1`` (step ``k = 0..N`` = #patches occluded;
index 0 = clean prediction ``M(x)``; index ``N`` = all-occluded, identical for MoRF & LeRF).

    DAPC(ψ, M, x) = area_under(LeRF curve) − area_under(MoRF curve)      (≥0 good)

with area = trapezoid over occlusion-fraction ``∈[0,1]`` (``np.trapz(curve,
dx=1/N)``). Higher = better. Raw probability curves are stored so any AOPC/AUC/sign variant is recomputable
offline without rerunning.

METHODS (Benchmark run 1)
-------------------------
``lrp``     — CP-LRP (``cp_lrp_baseline``), signed input relevance, max/patch.
``chefer``  — Chefer/Gur/Wolf CVPR'21 Transformer Attribution (faithful; the
              attention-map LRP relevance comes from our ``attnlrp_gamma``).
``rollout`` — Abnar & Zuidema attention rollout.
``rise``    — RISE (Petsiuk et al. BMVC'18), N random masks, max/patch.
``random``  — mean over ``K`` random saliency maps (theoretical floor).

MODELS: M1 ViT-S/FunnyBirds, M2 ViT-B/ImageNet, M3 DINOv3-S/FunnyBirds,
M4 DINOv3-B/ImageNet. FunnyBirds uses the TEST split (zero part-ablations),
ImageNet the HF val (``n_per_class=10`` pool). ``N=64`` correctly-classified
images/model, seed 0, indices persisted.

STORAGE — ``data/results/benchmark/iddapc_<model>.npz``: for each method the raw
``curve_morf__<m>`` / ``curve_lerf__<m>`` (n, N+1) and ``dapc__<m>`` (n,), plus
``image_ids``, ``preds``, ``clean_prob``, geometry and a JSON ``provenance``
(model spec, checkpoint, git commit, params). Designed for extension: adding a
method or model never reruns existing ones — results are keyed by
``(model_id, image_id, method)`` and merged into the per-model npz.

CLI (each GPU stage should be run under the shared A40 lock)::

    py -m experiments.insertion_deletion_bench select   --model M1
    py -m experiments.insertion_deletion_bench run      --model M1 --methods lrp,chefer,rollout,rise,random
    py -m experiments.insertion_deletion_bench summarize
    py -m experiments.insertion_deletion_bench figures
    py -m experiments.insertion_deletion_bench verify-chefer --model M1   # repo-match check
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = REPO_ROOT / "data" / "results" / "benchmark"
FIG_DIR = REPO_ROOT / "figures" / "benchmark"
JOURNAL_FIG = Path("/home/claude/workspaces/crp-paper/iclr2026/journal-figures")

CKPT_M1 = REPO_ROOT / "data/runs/finetune_vit_small_funny-birds-train-clean/2026-07-26_160337/best.pt"
CKPT_M3 = REPO_ROOT / "data/runs/finetune_vit_dinov3_small_funny-birds-train-clean/2026-07-25_200008/best.pt"

# ── Model registry (design for extension: add a row, nothing else reruns) ──────
@dataclass(frozen=True)
class ModelSpec:
    tag: str                      # M1..M4
    base: str
    dataset: str
    model_source: str             # checkpoint | dinov3_in1k
    checkpoint: Optional[str]
    label: str
    ds_extra: dict = field(default_factory=dict)

MODELS: Dict[str, ModelSpec] = {
    "M1": ModelSpec("M1", "vit_small", "funny_birds", "checkpoint", str(CKPT_M1),
                    "ViT-S/16 · FunnyBirds (test)", {"split": "test"}),
    "M2": ModelSpec("M2", "vit_base", "imagenet", "checkpoint", None,
                    "ViT-B/16 · ImageNet (val)", {"n_per_class": 10}),
    "M3": ModelSpec("M3", "vit_dinov3_small", "funny_birds", "checkpoint", str(CKPT_M3),
                    "DINOv3-S/16 (+reg, finetuned) · FunnyBirds (test)", {"split": "test"}),
    "M4": ModelSpec("M4", "vit_base_patch16_dinov3", "imagenet", "dinov3_in1k", None,
                    "DINOv3-B/16 (+reg, canvit head) · ImageNet (val)", {"n_per_class": 10}),
}

METHODS: Tuple[str, ...] = ("lrp", "chefer", "rollout", "rise", "random")
N_IMAGES = 64
SEED = 0
RISE_N_MASKS = 2000
RISE_S = 8
RISE_P = 0.5
RANDOM_K = 5
CHEFER_COMPOSITE = "attnlrp_gamma"     # full-bilinear AttnLRP → attention-map relevance R_A
LRP_COMPOSITE = "cp_lrp_baseline"

app = typer.Typer(add_completion=False, help=__doc__)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def npz_path(model: str) -> Path:
    return RES_DIR / f"iddapc_{model}.npz"


# ── model + data loading (reuses crp_gallery.load_model) ───────────────────────
def load(model_id: str, device: str):
    from experiments.crp_gallery import load_model, load_eval_dataset
    from experiments.model_io import backbone_transforms
    spec = MODELS[model_id]
    model, ncls, head, label = load_model(
        spec.base, spec.dataset, model_source=spec.model_source,
        checkpoint=spec.checkpoint, head="linear", num_classes=None,
        head_kwargs={}, device=device)
    transform, normalize = backbone_transforms(model.backbone)
    ds = load_eval_dataset(spec.dataset, transform, spec.ds_extra)
    return model, normalize, ds, ncls, label


def select_indices(model, normalize, ds, device, *, n: int = N_IMAGES,
                   seed: int = SEED, batch: int = 64) -> Tuple[List[int], List[int], List[float]]:
    """First ``n`` correctly-classified dataset indices in seeded random order.
    Returns (indices, preds, clean predicted-class probabilities)."""
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()
    idxs: List[int] = []
    preds: List[int] = []
    probs: List[float] = []
    with torch.no_grad():
        for s0 in range(0, len(perm), batch):
            chunk = perm[s0:s0 + batch]
            x = torch.stack([ds[i][0] for i in chunk]).to(device)
            y = torch.tensor([ds[i][1] for i in chunk])
            logits = model(normalize(x))
            p = logits.softmax(-1)
            pred = p.argmax(-1).cpu()
            for j in range(len(chunk)):
                if int(pred[j]) == int(y[j]):
                    idxs.append(int(chunk[j]))
                    preds.append(int(pred[j]))
                    probs.append(float(p[j, pred[j]]))
                    if len(idxs) >= n:
                        return idxs, preds, probs
    return idxs, preds, probs


# ── saliency dispatch → (grid, grid) patch-saliency (MAX-aggregated) ───────────
def saliency_patch(method: str, *, model, normalize, attribution, lrp_comp, chefer_comp,
                   softmax_layers, x01, xn, target, n_prefix, grid, patch, input_size,
                   img_seed: int) -> torch.Tensor:
    """One method's (grid, grid) patch saliency for a single image. ``x01`` is the
    un-normalised (3,H,W) image in [0,1]; ``xn`` its normalised (1,3,H,W)."""
    import experiments.xai_methods as xm
    if method == "lrp":
        xin = xn.clone().detach().requires_grad_(True)
        res = attribution(xin, [{"y": [int(target)]}], lrp_comp)
        heat = res.heatmap.detach().cpu()[0]                       # (H,W) signed
        return xm.to_patch_max(heat, grid, patch)
    if method == "rollout":
        _, attns = xm.capture_attention(model, xn)
        return xm.attention_rollout(attns, n_prefix, grid)[0].cpu()
    if method == "chefer":
        return xm.chefer_transformer_attribution(
            model, attribution, chefer_comp, xn, int(target),
            n_prefix=n_prefix, grid=grid, softmax_layers=softmax_layers)[0].cpu()
    if method == "rise":
        sal = xm.rise_saliency(model, normalize, x01, int(target), input_size=input_size,
                               n_masks=RISE_N_MASKS, s=RISE_S, p=RISE_P, seed=SEED)
        return xm.to_patch_max(sal, grid, patch)
    if method == "random":
        g = torch.Generator().manual_seed(SEED * 100003 + img_seed)
        return torch.rand(grid, grid, generator=g)
    raise ValueError(f"unknown method {method!r}")


# ── perturbation engine → MoRF / LeRF predicted-class probability curves ──────
def perturbation_curves(model, normalize, x01, target, sal_grid, *, grid, patch,
                        batch: int = 128) -> Tuple[np.ndarray, np.ndarray, float]:
    """(morf_curve, lerf_curve, clean_prob). Curves length ``N+1``, in
    predicted-class probability. Mean-fill occlusion; cumulative descending / ascending."""
    device = next(model.parameters()).device
    x = x01.to(device)
    n = grid * grid
    order = torch.argsort(sal_grid.reshape(-1).to(device), descending=True)   # most-salient first
    mean_col = x.mean(dim=(1, 2))

    def _curve(seq: torch.Tensor) -> torch.Tensor:
        imgs = x[None].repeat(n + 1, 1, 1, 1)                     # (N+1,3,H,W)
        for k in range(1, n + 1):
            p = int(seq[k - 1]); r, c = divmod(p, grid)
            imgs[k:, :, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = mean_col[:, None, None]
        with torch.no_grad():
            out = [model(normalize(imgs[i:i + batch])).softmax(-1)[:, int(target)]
                   for i in range(0, n + 1, batch)]
        return torch.cat(out)

    morf = _curve(order)
    lerf = _curve(order.flip(0))
    return morf.cpu().numpy(), lerf.cpu().numpy(), float(morf[0])


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # numpy 2.x renamed trapz


def dapc_of(morf: np.ndarray, lerf: np.ndarray) -> float:
    """LeRF area − MoRF area, trapezoid over occlusion-fraction ∈ [0,1]."""
    n = len(morf) - 1
    return float(_trapz(lerf, dx=1.0 / n) - _trapz(morf, dx=1.0 / n))


# ── npz load / merge / save (extension-friendly) ───────────────────────────────
def load_store(model: str) -> dict:
    p = npz_path(model)
    if not p.exists():
        return {}
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def save_store(model: str, store: dict) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path(model), **store)


# ── CLI: select ────────────────────────────────────────────────────────────────
@app.command()
def select(model: str = typer.Option(..., help="M1|M2|M3|M4"),
           device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
           force: bool = typer.Option(False, "--force")):
    """Persist the N=64 correctly-classified, seed-0 image indices for a model."""
    spec = MODELS[model]
    store = load_store(model)
    if "image_ids" in store and not force:
        print(f"[{model}] already has {len(store['image_ids'])} indices — use --force to redo")
        return
    m, normalize, ds, ncls, label = load(model, device)
    idxs, preds, probs = select_indices(m, normalize, ds, device)
    print(f"[{model}] selected {len(idxs)}/{N_IMAGES} correct images from {len(ds)} ({label})")
    grid = int(ds[0][0].shape[-1]) // int(m.backbone.patch_embed.patch_size[0]
                                          if isinstance(m.backbone.patch_embed.patch_size, (tuple, list))
                                          else m.backbone.patch_embed.patch_size)
    store.update({
        "image_ids": np.array(idxs, dtype=np.int64),
        "preds": np.array(preds, dtype=np.int64),
        "clean_prob": np.array(probs, dtype=np.float64),
    })
    store["provenance"] = json.dumps({
        "model": model, "label": label, "base": spec.base, "dataset": spec.dataset,
        "model_source": spec.model_source, "checkpoint": spec.checkpoint,
        "ds_extra": spec.ds_extra, "num_classes": int(ncls), "n_images": len(idxs),
        "seed": SEED, "occlusion": "image-mean fill", "patch_agg": "max",
        "metric": "DAPC = trapz(LeRF)-trapz(MoRF) over occlusion-fraction; curves length N+1 (index0=clean=1.0)",
        "rise": {"n_masks": RISE_N_MASKS, "s": RISE_S, "p": RISE_P},
        "random_K": RANDOM_K, "lrp_composite": LRP_COMPOSITE,
        "chefer_composite": CHEFER_COMPOSITE, "git_commit": _git_commit(), "when": _now(),
    })
    save_store(model, store)
    print(f"[{model}] wrote {npz_path(model)}")


# ── CLI: run ───────────────────────────────────────────────────────────────────
@app.command()
def run(model: str = typer.Option(..., help="M1|M2|M3|M4"),
        methods: str = typer.Option(",".join(METHODS), help="comma list"),
        img_start: int = typer.Option(0),
        img_end: int = typer.Option(N_IMAGES),
        device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
        force: bool = typer.Option(False, "--force", help="recompute even if present")):
    """Compute MoRF/LeRF curves + DAPC for the given methods/images, merge into
    the model npz. Skips (method, image) pairs already stored unless --force."""
    import lrp_configs
    import experiments.xai_methods as xm
    from crp.attribution import CondAttribution

    want = [m.strip() for m in methods.split(",") if m.strip()]
    store = load_store(model)
    if "image_ids" not in store:
        raise typer.BadParameter(f"run `select --model {model}` first")
    image_ids = list(map(int, store["image_ids"]))
    preds = list(map(int, store["preds"]))
    n = len(image_ids)
    img_end = min(img_end, n)

    m, normalize, ds, ncls, label = load(model, device)
    attribution = CondAttribution(m)
    lrp_comp = lrp_configs.get(LRP_COMPOSITE).composite()
    chefer_comp = lrp_configs.get(CHEFER_COMPOSITE).composite()
    softmax_layers = xm.softmax_layer_names(m)
    x0, _ = ds[image_ids[0]]
    n_prefix, grid, patch = xm.model_geometry(m, x0[None])
    input_size = int(x0.shape[-1])
    npatch = grid * grid
    store["grid"] = np.int64(grid); store["patch"] = np.int64(patch)
    store["n_patch"] = np.int64(npatch)

    for meth in want:
        km, kl, kd = f"curve_morf__{meth}", f"curve_lerf__{meth}", f"dapc__{meth}"
        if km not in store:
            store[km] = np.full((n, npatch + 1), np.nan)
            store[kl] = np.full((n, npatch + 1), np.nan)
            store[kd] = np.full((n,), np.nan)
        # extend width if a prior model-array had different N (shouldn't happen per model)
        for i in range(img_start, img_end):
            if not force and np.isfinite(store[kd][i]):
                continue
            x01, _ = ds[image_ids[i]]
            xn = normalize(x01[None].to(device))
            target = preds[i]
            if meth == "random":
                morfs, lerfs, ds_ = [], [], []
                for k in range(RANDOM_K):
                    sg = saliency_patch("random", model=m, normalize=normalize,
                                        attribution=attribution, lrp_comp=lrp_comp,
                                        chefer_comp=chefer_comp, softmax_layers=softmax_layers,
                                        x01=x01, xn=xn, target=target, n_prefix=n_prefix,
                                        grid=grid, patch=patch, input_size=input_size,
                                        img_seed=image_ids[i] * 17 + k)
                    a, b, _ = perturbation_curves(m, normalize, x01, target, sg, grid=grid, patch=patch)
                    morfs.append(a); lerfs.append(b); ds_.append(dapc_of(a, b))
                mo, le = np.mean(morfs, 0), np.mean(lerfs, 0)
                store[km][i], store[kl][i], store[kd][i] = mo, le, float(np.mean(ds_))
            else:
                sg = saliency_patch(meth, model=m, normalize=normalize, attribution=attribution,
                                    lrp_comp=lrp_comp, chefer_comp=chefer_comp,
                                    softmax_layers=softmax_layers, x01=x01, xn=xn, target=target,
                                    n_prefix=n_prefix, grid=grid, patch=patch, input_size=input_size,
                                    img_seed=image_ids[i])
                a, b, _ = perturbation_curves(m, normalize, x01, target, sg, grid=grid, patch=patch)
                store[km][i], store[kl][i], store[kd][i] = a, b, dapc_of(a, b)
        done = int(np.isfinite(store[kd]).sum())
        print(f"[{model}/{meth}] {done}/{n} images  mean DAPC={np.nanmean(store[kd]):+.4f}")
        save_store(model, store)   # checkpoint after each method
    print(f"[{model}] saved {npz_path(model)}")


# ── CLI: verify-chefer (repo-match sanity) ─────────────────────────────────────
@app.command("verify-chefer")
def verify_chefer(model: str = typer.Option("M1"),
                  device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu")):
    """Print, for one image, the correlation between our faithful CVPR'21 Chefer
    (grad ⊙ R_A) and the ICCV'21 generic variant (grad ⊙ raw-attention) so the
    documented difference is quantified."""
    import lrp_configs
    import experiments.xai_methods as xm
    from crp.attribution import CondAttribution
    m, normalize, ds, ncls, label = load(model, device)
    attribution = CondAttribution(m)
    chefer_comp = lrp_configs.get(CHEFER_COMPOSITE).composite()
    softmax_layers = xm.softmax_layer_names(m)
    store = load_store(model)
    ids = list(map(int, store["image_ids"]))[:8] if "image_ids" in store else list(range(8))
    corrs = []
    for i in ids:
        x01, _ = ds[i]
        xn = normalize(x01[None].to(device))
        n_prefix, grid, patch = xm.model_geometry(m, x01[None])
        with torch.no_grad():
            pred = int(m(xn).argmax(-1))
        cvpr = xm.chefer_transformer_attribution(m, attribution, chefer_comp, xn, pred,
                                                 n_prefix=n_prefix, grid=grid,
                                                 softmax_layers=softmax_layers)[0].cpu().numpy().ravel()
        iccv = xm.chefer_relevance(m, xn, [pred], n_prefix=n_prefix, grid=grid)[0].cpu().numpy().ravel()
        corrs.append(float(np.corrcoef(cvpr, iccv)[0, 1]))
    print(f"[{model}] CVPR'21(grad⊙R_A) vs ICCV'21(grad⊙attn) patch-saliency corr "
          f"over {len(corrs)} imgs: mean={np.mean(corrs):.3f} min={np.min(corrs):.3f} max={np.max(corrs):.3f}")


# ── CLI: summarize ─────────────────────────────────────────────────────────────
@app.command()
def summarize():
    """Write the tidy CSV data/results/benchmark/iddapc_summary.csv."""
    import csv
    rows = []
    for model in MODELS:
        store = load_store(model)
        if "image_ids" not in store:
            continue
        for meth in METHODS:
            kd, km, kl = f"dapc__{meth}", f"curve_morf__{meth}", f"curve_lerf__{meth}"
            if kd not in store:
                continue
            dapc = store[kd]; mask = np.isfinite(dapc)
            if mask.sum() == 0:
                continue
            npatch = store[km].shape[1] - 1
            morf_auc = np.array([_trapz(store[km][i], dx=1.0 / npatch) for i in np.where(mask)[0]])
            lerf_auc = np.array([_trapz(store[kl][i], dx=1.0 / npatch) for i in np.where(mask)[0]])
            rows.append({
                "model": model, "method": meth,
                "dapc_mean": f"{np.mean(dapc[mask]):.6f}", "dapc_std": f"{np.std(dapc[mask]):.6f}",
                "morf_auc_mean": f"{np.mean(morf_auc):.6f}", "lerf_auc_mean": f"{np.mean(lerf_auc):.6f}",
                "n": int(mask.sum()),
            })
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = RES_DIR / "iddapc_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "dapc_mean", "dapc_std",
                                          "morf_auc_mean", "lerf_auc_mean", "n"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['model']} {r['method']:8s} DAPC={r['dapc_mean']} ±{r['dapc_std']} (n={r['n']})")


# ── CLI: figures ───────────────────────────────────────────────────────────────
METHOD_COLORS = {"lrp": "#d62728", "chefer": "#1f77b4", "rollout": "#2ca02c",
                 "rise": "#ff7f0e", "random": "#7f7f7f"}
METHOD_LABEL = {"lrp": "LRP (cp_lrp_baseline)", "chefer": "Chefer CVPR'21",
                "rollout": "Attn rollout", "rise": "RISE", "random": "Random (floor)"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


@app.command()
def figures():
    """Per-model MoRF/LeRF curve panels + a cross-model DAPC bar chart; copy the
    PDFs into the journal-figures directory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [mm for mm in MODELS if "image_ids" in load_store(mm)
               and any(f"dapc__{me}" in load_store(mm) for me in METHODS)]
    JOURNAL_FIG.mkdir(parents=True, exist_ok=True)

    # (a) per-model curve panels (mean MoRF solid, LeRF dashed per method)
    for model in present:
        store = load_store(model)
        methods_here = [me for me in METHODS if f"dapc__{me}" in store
                        and np.isfinite(store[f"dapc__{me}"]).any()]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for me in methods_here:
            km, kl, kd = f"curve_morf__{me}", f"curve_lerf__{me}", f"dapc__{me}"
            mask = np.isfinite(store[kd])
            mo = np.nanmean(store[km][mask], 0); le = np.nanmean(store[kl][mask], 0)
            frac = np.linspace(0, 1, len(mo))
            c = METHOD_COLORS[me]
            ax.plot(frac, mo, color=c, lw=1.8, label=f"{METHOD_LABEL[me]} (DAPC={np.nanmean(store[kd]):+.3f})")
            ax.plot(frac, le, color=c, lw=1.2, ls="--", alpha=0.7)
        ax.set_xlabel("fraction of patches occluded")
        ax.set_ylabel("predicted-class probability")
        ax.set_title(f"{model}: {MODELS[model].label}\nMoRF (solid) vs LeRF (dashed)")
        ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper right")
        _save(fig, FIG_DIR / f"iddapc_curves_{model}")
        plt.close(fig)
        # journal copy
        import shutil
        shutil.copy(FIG_DIR / f"iddapc_curves_{model}.pdf", JOURNAL_FIG / f"iddapc_curves_{model}.pdf")

    # (b) DAPC bar chart, methods grouped per model
    fig, ax = plt.subplots(figsize=(8, 4.2))
    methods_all = [me for me in METHODS if any(f"dapc__{me}" in load_store(mm) for mm in present)]
    x = np.arange(len(present)); w = 0.8 / max(len(methods_all), 1)
    for j, me in enumerate(methods_all):
        vals, errs = [], []
        for model in present:
            store = load_store(model)
            d = store.get(f"dapc__{me}")
            if d is None or not np.isfinite(d).any():
                vals.append(np.nan); errs.append(0); continue
            vals.append(np.nanmean(d)); errs.append(np.nanstd(d) / np.sqrt(np.isfinite(d).sum()))
        ax.bar(x + j * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=2,
               color=METHOD_COLORS[me], label=METHOD_LABEL[me])
    ax.set_xticks(x); ax.set_xticklabels(present)
    ax.set_ylabel("DAPC (LeRF−MoRF area; higher = better)")
    ax.set_title("Insertion-Deletion DAPC per method × model")
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    _save(fig, FIG_DIR / "iddapc_bars")
    plt.close(fig)
    import shutil
    shutil.copy(FIG_DIR / "iddapc_bars.pdf", JOURNAL_FIG / "iddapc_bars.pdf")
    print(f"wrote figures to {FIG_DIR} and copied PDFs to {JOURNAL_FIG}")


if __name__ == "__main__":
    app()
