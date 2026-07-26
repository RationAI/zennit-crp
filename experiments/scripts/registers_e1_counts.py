"""Registers E1: register/outlier-token DETECTION as a single-question experiment.

RQ: how many high-norm outlier ("register/scratch-pad") tokens appear after each
attention half-block and each MLP half-block, and do DINOv3 backbones with
built-in register tokens contain fewer of them than standard ViTs?

Criterion (identical to journal entry Ic / step1c redo, fixed for
comparability): per SAMPLE and per SITE, mu/sigma over that sample's PATCH-token
L2 norms; a token is an outlier iff norm > mu + 4*sigma. CLS is excluded; for
DINOv3 the 4 register tokens are ALSO excluded from the patch statistics but
tracked separately.

Recording sites (24 per model, network order)::

    site 2b   = "post_attn"  b : state after the ATTN residual add
                                 = INPUT of blocks[b].norm2 (forward pre-hook)
    site 2b+1 = "post_mlp"   b : state after the MLP residual add
                                 = OUTPUT of blocks[b]      (forward hook)

This identification is verified numerically per model on the first batch:
``blocks[b] output == norm2_input + ls2/gamma_2 * mlp(norm2(norm2_input))``
(drop_path is identity in eval), which holds for both timm ``Block`` (M1/M2,
LayerScale = ``ls2`` module, Identity for AugReg) and timm ``EvaBlock``
(DINOv3, LayerScale = ``gamma_2`` parameter). Max abs deviation is stored in
the npz meta.

Models (N=256 each, seed 0, round-robin class-diverse, indices persisted;
FunnyBirds models share indices, ImageNet models share indices, because the
selection depends only on dataset labels + seed):

* ``vit_small_funny_birds`` — finetuned probe, FunnyBirds TEST split.
* ``vit_base_imagenet``     — timm ImageNet-1k classifier, val subset
                              (n_per_class=10 pool).
* ``dinov3_small_funny_birds`` — timm ``vit_small_patch16_dinov3`` pretrained
  BACKBONE (no head: detection uses token norms only, so the forward is
  ``forward_features``); FunnyBirds test, normalization via timm
  ``resolve_data_config`` for this model (256x256, 16x16=256 patches).
* ``dinov3_base_imagenet`` — timm ``vit_base_patch16_dinov3`` backbone,
  ImageNet val subset (same n_per_class=10 pool).

Run (collect needs the GPU; the rest is CPU)::

    python -m experiments.scripts.registers_e1_counts collect --model <key>
    python -m experiments.scripts.registers_e1_counts analyze
    python -m experiments.scripts.registers_e1_counts figures
    python -m experiments.scripts.registers_e1_counts report

Outputs: ``data/results/registers/e1_counts_<model>.npz`` (norms + flags +
criterion params), ``data/results/registers/e1_per_site_table.csv``,
``data/results/registers/e1_analysis.json``,
``figures/registers/e1_counts/`` (png+pdf),
report ``research/registers/e1_outlier_counts.md`` (written by ``report``).
All commands are idempotent (recompute + overwrite).
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "results" / "registers"
FIG_DIR = REPO_ROOT / "figures" / "registers" / "e1_counts"

CKPT_FUNNY = (REPO_ROOT / "data" / "runs"
              / "finetune_vit_small_funny-birds-train-clean" / "2026-06-03_000556" / "best.pt")

SD_K = 4.0                # criterion: norm > mu_sample + 4*sd_sample (per site)
N_IMAGES = 256
SEED = 0
DEPTH = 12
PLATEAU_BLOCKS = (8, 9, 10, 11)   # plateau % = mean frac over these blocks

MODELS: Dict[str, dict] = {
    "vit_small_funny_birds": dict(
        kind="probe", base="vit_small", dataset="funny_birds",
        checkpoint=str(CKPT_FUNNY), extra={"split": "test"},
        label="ViT-S/16 AugReg · FunnyBirds test", short="ViT-S (std)"),
    "vit_base_imagenet": dict(
        kind="probe", base="vit_base", dataset="imagenet",
        checkpoint=None, extra={"n_per_class": 10},
        label="ViT-B/16 AugReg · ImageNet val", short="ViT-B (std)"),
    "dinov3_small_funny_birds": dict(
        kind="timm", timm_name="vit_small_patch16_dinov3", dataset="funny_birds",
        extra={"split": "test"},
        label="DINOv3 ViT-S/16 (+4 reg) · FunnyBirds test", short="DINOv3-S (+reg)"),
    "dinov3_base_imagenet": dict(
        kind="timm", timm_name="vit_base_patch16_dinov3", dataset="imagenet",
        extra={"n_per_class": 10},
        label="DINOv3 ViT-B/16 (+4 reg) · ImageNet val", short="DINOv3-B (+reg)"),
}

# fixed categorical assignment (dataviz reference palette, slots 1-4, light mode)
COLORS = {
    "vit_small_funny_birds": "#2a78d6",
    "vit_base_imagenet": "#008300",
    "dinov3_small_funny_birds": "#e87ba4",
    "dinov3_base_imagenet": "#eda100",
}

SITE_NAMES = [f"blk{b}.{half}" for b in range(DEPTH) for half in ("attn", "mlp")]
ATTN_SITES = [2 * b for b in range(DEPTH)]
MLP_SITES = [2 * b + 1 for b in range(DEPTH)]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def counts_path(model_key: str) -> Path:
    return OUT_DIR / f"e1_counts_{model_key}.npz"


# ─────────────────────────────────────────────────────────────────────────────
# selection (identical scheme to step 1/1b/1c: round-robin class-diverse)
# ─────────────────────────────────────────────────────────────────────────────

def _ds_labels(ds) -> List[int]:
    if hasattr(ds, "items"):
        return [int(c) for _, c in ds.items]
    if hasattr(ds, "rows"):
        return [int(c) for _, c in ds.rows]
    return [int(ds[i][1]) for i in range(len(ds))]


def pick_class_diverse(ds, n: int, seed: int = 0) -> List[int]:
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


# ─────────────────────────────────────────────────────────────────────────────
# collect (GPU)
# ─────────────────────────────────────────────────────────────────────────────

class HalfBlockNormRecorder:
    """L2 token norms at both half-block boundaries of every block:
    ``blocks[b].norm2`` INPUT (post-attn residual add, pre-hook) and
    ``blocks[b]`` OUTPUT (post-MLP residual add, forward hook)."""

    def __init__(self, blocks):
        self.norms: Dict[int, np.ndarray] = {}
        self.handles = []
        for b, blk in enumerate(blocks):
            def pre_norm2(mod, args, b=b):
                self.norms[2 * b] = args[0].detach().norm(dim=-1).float().cpu().numpy()

            def post_block(mod, args, out, b=b):
                self.norms[2 * b + 1] = out.detach().norm(dim=-1).float().cpu().numpy()

            self.handles.append(blk.norm2.register_forward_pre_hook(pre_norm2))
            self.handles.append(blk.register_forward_hook(post_block))

    def stack(self) -> np.ndarray:                             # (24, B, T)
        return np.stack([self.norms[s] for s in range(2 * DEPTH)])

    def remove(self):
        for h in self.handles:
            h.remove()


def _verify_sites(blocks, x_in, forward_fn) -> float:
    """Numeric check that norm2's input IS the post-attn-residual state:
    blocks[b] out == norm2_in + drop_path2(scale2 * mlp(norm2(norm2_in))).
    Covers timm Block (ls2 module) and EvaBlock (gamma_2 param). Returns the
    max abs deviation over checked blocks (0, depth//2, depth-1)."""
    import torch

    cap: Dict[int, dict] = {}
    hs = []
    for b in (0, DEPTH // 2, DEPTH - 1):
        blk = blocks[b]
        cap[b] = {}
        hs.append(blk.norm2.register_forward_pre_hook(
            lambda m, a, b=b: cap[b].__setitem__("n2_in", a[0].detach())))
        hs.append(blk.register_forward_hook(
            lambda m, a, o, b=b: cap[b].__setitem__("out", o.detach())))
    with torch.no_grad():
        forward_fn(x_in)
    for h in hs:
        h.remove()

    worst = 0.0
    with torch.no_grad():
        for b, d in cap.items():
            blk = blocks[b]
            y = blk.mlp(blk.norm2(d["n2_in"]))
            if getattr(blk, "gamma_2", None) is not None:      # EvaBlock LayerScale
                y = blk.gamma_2 * y
            elif hasattr(blk, "ls2"):                          # timm Block (Identity for AugReg)
                y = blk.ls2(y)
            recon = d["n2_in"] + blk.drop_path2(y)
            worst = max(worst, float((recon - d["out"]).abs().max()))
    return worst


def cmd_collect(args):
    import torch
    from experiments.crp_gallery import load_eval_dataset, load_model
    from experiments.models import backbone_transforms

    spec = MODELS[args.model]
    device = args.device
    if spec["kind"] == "probe":
        model, _, _, label = load_model(
            spec["base"], spec["dataset"], model_source="checkpoint",
            checkpoint=spec["checkpoint"], head="linear", num_classes=None,
            head_kwargs={}, device=device)
        backbone = model.backbone
        forward_fn = lambda x: model(x)                        # noqa: E731
        n_prefix, n_reg = 1, 0
    else:
        import timm
        backbone = timm.create_model(spec["timm_name"], pretrained=True)
        label = spec["label"]
        # E3 extension: optionally swap in a fine-tuned backbone state
        # (best.pt from train_probe finetune; Probe.backbone IS the timm
        # module, so the state dict loads directly).
        ckpt_path = getattr(args, "checkpoint", None)
        if ckpt_path:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            missing, unexpected = backbone.load_state_dict(
                ck["backbone_state_dict"], strict=False)
            assert not missing, f"missing keys loading {ckpt_path}: {missing}"
            assert not unexpected, f"unexpected keys loading {ckpt_path}: {unexpected}"
            label = f"{label} [finetuned: {Path(ckpt_path).parent.name}]"
            print(f"loaded finetuned backbone from {ckpt_path} "
                  f"(val_acc={ck.get('val_acc', float('nan')):.4f})")
        backbone.eval().to(device).requires_grad_(False)
        # no head needed for detection — token norms only, so skip any classifier
        forward_fn = lambda x: backbone.forward_features(x)    # noqa: E731
        n_prefix = int(backbone.num_prefix_tokens)             # 1 CLS + 4 registers
        n_reg = int(backbone.reg_token.shape[1]) if backbone.reg_token is not None else 0
        assert n_prefix == 1 + n_reg == 5

    assert len(backbone.blocks) == DEPTH
    transform, normalize = backbone_transforms(backbone)
    ds = load_eval_dataset(spec["dataset"], transform, extra_kwargs=spec["extra"])
    indices_from = getattr(args, "indices_from", None)
    if indices_from:
        # E3 extension: reuse the exact persisted image selection of a
        # previous run instead of re-deriving it (belt-and-braces — the
        # derivation is deterministic in (labels, seed) anyway).
        indices = [int(i) for i in
                   np.load(indices_from, allow_pickle=True)["ds_indices"]]
        print(f"reusing {len(indices)} indices from {indices_from}")
    else:
        indices = pick_class_diverse(ds, args.n, seed=args.seed)
    labels = np.array([_ds_labels(ds)[i] for i in indices])
    print(f"{label}: N={len(indices)} of {len(ds)}")

    # one-time numeric site verification on the first batch
    x0 = normalize(torch.stack([ds[i][0] for i in indices[:8]]).to(device))
    dev = _verify_sites(backbone.blocks, x0, forward_fn)
    print(f"site check: max |block_out - (norm2_in + mlp_branch)| = {dev:.3e}")
    assert dev < 1e-3, "norm2-input is not the post-attn residual state?"

    rec = HalfBlockNormRecorder(backbone.blocks)
    chunks = []
    with torch.no_grad():
        for s0 in range(0, len(indices), args.batch_size):
            batch = [ds[i][0] for i in indices[s0:s0 + args.batch_size]]
            forward_fn(normalize(torch.stack(batch).to(device)))
            chunks.append(rec.stack())
            print(f"  {s0 + len(batch)}/{len(indices)}", flush=True)
    rec.remove()

    norms = np.concatenate(chunks, axis=1).astype(np.float32)  # (24, N, T)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if getattr(args, "out", None) else counts_path(args.model)
    np.savez_compressed(
        out_path,
        norms=norms,
        ds_indices=np.array(indices, dtype=np.int64),
        labels=labels,
        n_prefix=np.int64(n_prefix), n_reg=np.int64(n_reg),
        sd_k=np.float64(SD_K),
        site_names=np.array(SITE_NAMES),
        meta=np.array([
            f"model={label}",
            "sites: even index 2b = INPUT of blocks[b].norm2 (post-ATTN residual add); "
            "odd index 2b+1 = blocks[b] OUTPUT (post-MLP residual add)",
            f"site_identity_check_max_abs_dev={dev:.3e} (blocks 0,{DEPTH//2},{DEPTH-1})",
            f"token layout: 0=CLS, 1..{n_prefix-1}=registers (DINOv3 only), "
            f"{n_prefix}.. = patches row-major",
            "criterion: per sample AND site, mu/sd over PATCH-token norms only "
            f"(CLS + registers excluded); outlier iff norm > mu + {SD_K}*sd",
            "DINOv3 models run headless (forward_features): detection needs token norms only",
            f"selection=round-robin class-diverse seed={args.seed}, N={len(indices)}"
            + (f" (indices reused from {indices_from})" if indices_from else ""),
            f"dataset_kwargs={spec['extra']}",
            f"checkpoint={getattr(args, 'checkpoint', None)}",
            f"collected={_now()}",
        ]),
    )
    print(f"saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# analyze (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def per_sample_flags(norms: np.ndarray, n_prefix: int):
    """norms (24, N, T) -> (flags (24, N, P), tau (24, N), patch (24, N, P)).
    Per-sample, per-site stats over patch tokens only."""
    patch = norms[:, :, n_prefix:]
    mu = patch.mean(-1)
    sd = patch.std(-1)
    tau = mu + SD_K * sd
    return patch > tau[:, :, None], tau, patch


def _analyze_model(model_key: str) -> dict:
    d = np.load(counts_path(model_key), allow_pickle=True)
    norms, n_prefix, n_reg = d["norms"], int(d["n_prefix"]), int(d["n_reg"])
    flags, tau, patch = per_sample_flags(norms, n_prefix)
    frac = flags.mean(axis=(1, 2))                             # (24,)
    union = flags.any(0)                                       # (N, P)
    cnt = union.sum(1)                                         # per-image union count

    rep = dict(
        label=MODELS[model_key]["label"], N=int(norms.shape[1]),
        n_patch=int(patch.shape[-1]), n_reg=n_reg,
        frac_per_site=[float(f) for f in frac],
        plateau_attn=float(np.mean([frac[2 * b] for b in PLATEAU_BLOCKS])),
        plateau_mlp=float(np.mean([frac[2 * b + 1] for b in PLATEAU_BLOCKS])),
        union_frac=float(union.mean()),
        total_flagged_union=int(union.sum()),
        total_flags_all_sites=int(flags.sum()),
        per_image=dict(mean=float(cnt.mean()), median=float(np.median(cnt)),
                       min=int(cnt.min()), max=int(cnt.max()),
                       frac_images_flagged=float((cnt > 0).mean())),
    )

    if n_reg > 0:                                              # DINOv3: track registers
        reg = norms[:, :, 1:1 + n_reg]                         # (24, N, 4)
        cls = norms[:, :, 0]                                   # (24, N)
        reg_over_tau = (reg > tau[:, :, None])                 # register beats patch criterion
        rep["registers"] = dict(
            median_reg_norm_per_site=[float(x) for x in np.median(reg, axis=(1, 2))],
            median_patch_norm_per_site=[float(x) for x in np.median(patch, axis=(1, 2))],
            median_cls_norm_per_site=[float(x) for x in np.median(cls, axis=1)],
            frac_reg_tokens_over_tau=float(reg_over_tau.mean()),
            frac_samples_any_reg_over_tau_per_site=[float(x) for x in
                                                    reg_over_tau.any(-1).mean(-1)],
            plateau_frac_samples_any_reg_over_tau=float(np.mean(
                [reg_over_tau[2 * b + 1].any(-1).mean() for b in PLATEAU_BLOCKS])),
        )

    # persist flags/counts next to the norms (idempotent overwrite)
    payload = {k: d[k] for k in d.files}
    payload.update(flags=flags, tau=tau.astype(np.float32),
                   frac_per_site=frac.astype(np.float64),
                   union_mask=union, per_image_count=cnt.astype(np.int64),
                   analyzed=np.array([_now()]))
    np.savez_compressed(counts_path(model_key), **payload)
    return rep


def cmd_analyze(args):
    report = {mk: _analyze_model(mk) for mk in MODELS if counts_path(mk).exists()}
    (OUT_DIR / "e1_analysis.json").write_text(json.dumps(report, indent=2))
    print(f"saved {OUT_DIR / 'e1_analysis.json'}")

    with open(OUT_DIR / "e1_per_site_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site"] + [f"pct_{mk}" for mk in report])
        for s, name in enumerate(SITE_NAMES):
            w.writerow([name] + [f"{100 * report[mk]['frac_per_site'][s]:.4f}"
                                 for mk in report])
    print(f"saved {OUT_DIR / 'e1_per_site_table.csv'}")

    for mk, rep in report.items():
        print(f"\n== {mk} (N={rep['N']}, P={rep['n_patch']})")
        print(f"   plateau attn {100*rep['plateau_attn']:.3f}%  "
              f"mlp {100*rep['plateau_mlp']:.3f}%  union {100*rep['union_frac']:.3f}%  "
              f"per-image mean {rep['per_image']['mean']:.1f} "
              f"[{rep['per_image']['min']}..{rep['per_image']['max']}]")


# ─────────────────────────────────────────────────────────────────────────────
# figures (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {FIG_DIR / stem}.png/.pdf")


def cmd_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((OUT_DIR / "e1_analysis.json").read_text())
    x = np.arange(2 * DEPTH)

    # 1) outlier fraction per site — one line per model, no threshold lines
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    for mk, rep in report.items():
        ax.plot(x, 100 * np.asarray(rep["frac_per_site"]), lw=2, marker="o", ms=4,
                color=COLORS[mk], label=MODELS[mk]["short"])
    ax.set_xticks(x[::2], [str(b) for b in range(DEPTH)])
    ax.set_xlabel("site in network order (tick = post-attn half of block; "
                  "next point = post-MLP half)")
    ax.set_ylabel("patch-token outliers (%)")
    ax.set_title("E1 — outlier fraction per half-block site "
                 f"(per-sample criterion: norm > μ + {SD_K:g}σ, "
                 "CLS/registers excluded; N=256/model)")
    _style(ax)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "e1_fraction_per_site")
    plt.close(fig)

    # 2) DINOv3: register-token vs patch-token norms across sites
    dkeys = [mk for mk in report if report[mk].get("n_reg", 0) > 0]
    fig, axes = plt.subplots(1, len(dkeys), figsize=(6.0 * len(dkeys), 4.0),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, mk in zip(axes, dkeys):
        rg = report[mk]["registers"]
        d = np.load(counts_path(mk), allow_pickle=True)
        reg = d["norms"][:, :, 1:1 + int(d["n_reg"])]          # (24, N, 4)
        med_tau = np.median(d["tau"], axis=1)
        c = COLORS[mk]
        for r in range(reg.shape[-1]):
            ax.plot(x, np.median(reg[:, :, r], axis=1), lw=1.2, color=c,
                    alpha=0.8, label="register tokens (4)" if r == 0 else None)
        ax.plot(x, rg["median_patch_norm_per_site"], lw=2, color="#52514e",
                label="patch tokens (median)")
        ax.plot(x, med_tau, lw=2, ls="--", color="#0b0b0b",
                label="patch outlier threshold μ+4σ (median)")
        ax.plot(x, rg["median_cls_norm_per_site"], lw=1.4, ls=":", color="#52514e",
                label="CLS token")
        ax.set_yscale("log")
        ax.set_xticks(x[::2], [str(b) for b in range(DEPTH)])
        ax.set_xlabel("site (block; attn/MLP halves interleaved)")
        ax.set_title(MODELS[mk]["short"])
        _style(ax)
    axes[0].set_ylabel("token L2 norm (median over N=256)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("E1 — DINOv3: built-in register tokens carry the high norms; "
                 "patch tokens stay below the outlier threshold", y=1.02)
    _save(fig, "e1_dinov3_registers")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# report (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def decision_table_md(report: dict) -> str:
    rows = ["| model | plateau % (attn sites) | plateau % (MLP sites) | any-site union % | "
            "total flagged tokens | per-image mean | per-image min..max | "
            "register norms (DINOv3) |",
            "|---|---|---|---|---|---|---|---|"]
    for mk, rep in report.items():
        if rep.get("n_reg", 0) > 0:
            rg = rep["registers"]
            reg_cell = (f"median reg {max(rg['median_reg_norm_per_site']):.0f} vs patch "
                        f"{max(rg['median_patch_norm_per_site']):.0f} (site max); "
                        f"reg>τ in {100*rg['plateau_frac_samples_any_reg_over_tau']:.0f}% "
                        "of samples (plateau)")
        else:
            reg_cell = "—"
        pi = rep["per_image"]
        rows.append(
            f"| {MODELS[mk]['short']} | {100*rep['plateau_attn']:.3f} | "
            f"{100*rep['plateau_mlp']:.3f} | {100*rep['union_frac']:.3f} | "
            f"{rep['total_flagged_union']} | {pi['mean']:.1f} | "
            f"{pi['min']}..{pi['max']} | {reg_cell} |")
    return "\n".join(rows)


def cmd_report(args):
    report = json.loads((OUT_DIR / "e1_analysis.json").read_text())
    md = REPO_ROOT / "research" / "registers" / "e1_outlier_counts.md"
    parts = [
        "# E1 — register-outlier detection: half-block counts, standard ViT vs DINOv3",
        f"_Generated {_now()} by `experiments/scripts/registers_e1_counts.py` "
        "(collect / analyze / figures / report)._",
        "",
        "## Experiment card",
        "**RQ.** How many high-norm outlier (register/scratch-pad) tokens appear after each "
        "attention half-block and each MLP half-block, and do DINOv3 backbones with built-in "
        "register tokens contain fewer of them than standard ViTs?",
        "",
        "**H1.** DINOv3 backbones (register tokens present) contain substantially fewer "
        "patch-token outliers than standard AugReg ViTs at every depth. "
        "**H0.** comparable fractions. "
        "**Falsified if** DINOv3 patch-outlier plateau fractions are within a factor ~2 of "
        "the standard ViTs'.",
        "",
        "**Criterion** (journal entry Ic, per-sample): at each of the 24 sites, μ/σ "
        f"over that sample's patch-token L2 norms; outlier iff norm > μ + {SD_K:g}σ. "
        "CLS excluded everywhere; DINOv3's 4 register tokens are excluded from patch "
        "statistics and tracked separately.",
        "",
        "## Method",
        "- Sites: `blocks[b].norm2` INPUT = state after the attn residual add (forward "
        "pre-hook); `blocks[b]` OUTPUT = state after the MLP residual add. 12 blocks × 2 "
        "= 24 sites per model, network order.",
        "- Site identity was verified numerically per model on the first batch: "
        "block output equals `norm2_input + ls2/γ₂·mlp(norm2(norm2_input))` "
        "(max abs deviation stored in each npz's meta; both timm `Block` (M1/M2) and "
        "`EvaBlock` (DINOv3, LayerScale `gamma_2`, rotary pos-emb) follow this structure).",
        "- Models: ViT-S/16 FunnyBirds probe (test split), ViT-B/16 timm ImageNet val "
        "(n_per_class=10 pool), DINOv3 ViT-S/16 and ViT-B/16 timm pretrained backbones "
        "run headless via `forward_features` (norms only — no classification head "
        "needed for detection); DINOv3 preprocessing from timm `resolve_data_config` "
        "(256×256 → 256 patches vs 197-token 224×224 for the standard ViTs).",
        "- N=256 images/model, seed 0, round-robin class-diverse; indices persisted in the "
        "npz. The two FunnyBirds models see the same images, likewise the two ImageNet "
        "models.",
        f"- Plateau % = mean outlier fraction over blocks {list(PLATEAU_BLOCKS)} of the "
        "given half (attn sites / MLP sites).",
        "",
        "## Decision table",
        decision_table_md(report),
        "",
    ]
    # verdict
    std_att = [report[k]["plateau_attn"] for k in report if report[k]["n_reg"] == 0]
    std_mlp = [report[k]["plateau_mlp"] for k in report if report[k]["n_reg"] == 0]
    din_att = [report[k]["plateau_attn"] for k in report if report[k]["n_reg"] > 0]
    din_mlp = [report[k]["plateau_mlp"] for k in report if report[k]["n_reg"] > 0]
    ratio_att = min(std_att) / max(max(din_att), 1e-12)
    ratio_mlp = min(std_mlp) / max(max(din_mlp), 1e-12)
    h1 = min(ratio_att, ratio_mlp) > 2.0
    parts += [
        "## Verdict",
        f"- min(standard)/max(DINOv3) plateau ratio: attn sites {ratio_att:.1f}×, "
        f"MLP sites {ratio_mlp:.1f}× (falsification bound: factor ~2).",
        f"- **H1 {'SUPPORTED' if h1 else 'NOT supported'}**: DINOv3 patch-outlier plateaus "
        f"are {'well beyond' if h1 else 'within'} a factor 2 of the standard ViTs'.",
        "",
        "## Files",
        "- Arrays: `data/results/registers/e1_counts_<model>.npz` (norms 24×N×T, "
        "flags, τ, indices, criterion params), `e1_analysis.json`, "
        "`e1_per_site_table.csv` (24 sites × 4 models).",
        "- Figures: `figures/registers/e1_counts/e1_fraction_per_site.{png,pdf}`, "
        "`e1_dinov3_registers.{png,pdf}` (copies in the paper's journal-figures).",
        "",
        "## Per-site table",
        "See `data/results/registers/e1_per_site_table.csv`; headline figure "
        "`e1_fraction_per_site` plots the same 24×4 numbers.",
    ]
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(parts) + "\n")
    print(f"saved {md}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("collect")
    pc.add_argument("--model", choices=list(MODELS), required=True)
    pc.add_argument("--n", type=int, default=N_IMAGES)
    pc.add_argument("--seed", type=int, default=SEED)
    pc.add_argument("--batch-size", type=int, default=32)
    pc.add_argument("--device", default="cuda")
    pc.add_argument("--checkpoint", default=None,
                    help="(timm models) best.pt from train_probe finetune — "
                         "load its backbone_state_dict before collecting (E3)")
    pc.add_argument("--indices-from", default=None,
                    help="npz whose ds_indices to reuse verbatim (E3)")
    pc.add_argument("--out", default=None,
                    help="output npz path override (default: e1_counts_<model>.npz)")
    pc.set_defaults(fn=cmd_collect)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    sub.add_parser("figures").set_defaults(fn=cmd_figures)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
