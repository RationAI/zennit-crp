"""Registers E3: does END-TO-END FINETUNING reintroduce patch-token outliers
that the pretrained DINOv3 backbone does not have?

Two-arm design (scout note ``research/registers/finetune_registers_scout.md``):

* **v1**  — pretrained DINOv3-S/16 backbone. REUSED from E1
  (``e1_counts_dinov3_small_funny_birds.npz``), never recomputed.
* **v2a** — Arm A "naive finetune": M1 protocol, backbone_lr 5e-4, LLRD 0.7,
  onecycle, 25 epochs (``train_probe finetune --from-scratch``).
* **v2b** — Arm B "conservative finetune": literature recipe (Feng et al.
  2510.17201 analogue), backbone_lr 5e-6, head_lr 1e-3, otherwise Arm A.
* **v3**  — frozen-probe control: Phase-2 CLS-token linear probe. Its backbone
  is BYTE-IDENTICAL to v1, so its detection arrays must match v1 exactly —
  collected independently as a pipeline sanity check.

Everything measurement-side is E1 verbatim (same 24 half-block sites, same
per-sample mu+4*sigma patch-norm criterion, same N=256 FunnyBirds-test images
— indices reused from the E1 npz). Collection reuses
``registers_e1_counts.cmd_collect`` via its --checkpoint/--indices-from/--out
extensions.

Extra diagnostics (DINOv3-report style, scout recommendation): per-site
patch-norm max/mean ratio (from the collected norms — no extra forward) and
final-output CLS-patch cosine (one forward sweep per variant).

Run (collect + diagnose need the GPU)::

    python -m experiments.scripts.registers_e3_finetune collect --variant v2a \
        --checkpoint data/runs/finetune_vit_dinov3_small_.../best.pt
    python -m experiments.scripts.registers_e3_finetune collect --variant v3
    python -m experiments.scripts.registers_e3_finetune diagnose
    python -m experiments.scripts.registers_e3_finetune analyze
    python -m experiments.scripts.registers_e3_finetune figures
    python -m experiments.scripts.registers_e3_finetune report

Outputs: ``data/results/registers/e3_finetune_{v2a,v2b,v3}.npz``,
``e3_analysis.json``, ``e3_diagnostics.json``,
``figures/registers/e3_finetune/`` (png+pdf, paper copies
``e3_fraction_finetune.pdf`` / ``e3_registers_finetune.pdf``),
note ``research/registers/e3_finetune_reintroduction.md`` (via ``report``).
All commands idempotent (recompute + overwrite).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from experiments.scripts import registers_e1_counts as e1
from experiments.scripts.registers_e1_counts import (
    DEPTH, PLATEAU_BLOCKS, SD_K, SITE_NAMES, _now, per_sample_flags,
)

REPO_ROOT = e1.REPO_ROOT
OUT_DIR = e1.OUT_DIR
FIG_DIR = REPO_ROOT / "figures" / "registers" / "e3_finetune"
PAPER_FIG_DIR = Path("/home/claude/workspaces/crp-paper/iclr2026/journal-figures")

E1_KEY = "dinov3_small_funny_birds"          # the pretrained row we compare to
E1_STD_KEY = "vit_small_funny_birds"         # M1 standard-ViT reference line

VARIANTS: Dict[str, dict] = {
    "v1": dict(npz=e1.counts_path(E1_KEY), needs_checkpoint=False,
               label="pretrained DINOv3-S (E1 row, reused)",
               short="pretrained", color="#e87ba4", ls="-"),
    "v2a": dict(npz=OUT_DIR / "e3_finetune_v2a.npz", needs_checkpoint=True,
                label="Arm A — naive finetune (M1 protocol, backbone_lr 5e-4)",
                short="finetuned A (lr 5e-4)", color="#eda100", ls="-"),
    "v2b": dict(npz=OUT_DIR / "e3_finetune_v2b.npz", needs_checkpoint=True,
                label="Arm B — conservative finetune (backbone_lr 5e-6)",
                short="finetuned B (lr 5e-6)", color="#008300", ls="-"),
    "v3": dict(npz=OUT_DIR / "e3_finetune_v3.npz", needs_checkpoint=False,
               label="frozen-probe control (backbone identical to v1)",
               short="frozen-probe control", color="#0b0b0b", ls=":"),
}
STD_COLOR = "#2a78d6"                        # ViT-S standard reference (E1 slot)


def _npz_checkpoint(npz_path: Path) -> Optional[str]:
    """Recover the ``checkpoint=...`` meta line stored by collect."""
    d = np.load(npz_path, allow_pickle=True)
    for line in d["meta"]:
        if str(line).startswith("checkpoint="):
            val = str(line)[len("checkpoint="):]
            return None if val == "None" else val
    return None


def _build_backbone(checkpoint: Optional[str], device: str):
    import timm
    import torch
    backbone = timm.create_model("vit_small_patch16_dinov3", pretrained=True)
    if checkpoint:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = backbone.load_state_dict(
            ck["backbone_state_dict"], strict=False)
        assert not missing and not unexpected, (missing, unexpected)
    return backbone.eval().to(device).requires_grad_(False)


# ─────────────────────────────────────────────────────────────────────────────
# collect — thin wrapper over registers_e1_counts.cmd_collect
# ─────────────────────────────────────────────────────────────────────────────

def cmd_collect(args):
    spec = VARIANTS[args.variant]
    if args.variant == "v1":
        raise SystemExit("v1 is the E1 row — reused, never recollected")
    if spec["needs_checkpoint"] and not args.checkpoint:
        raise SystemExit(f"--checkpoint required for {args.variant}")
    if not spec["needs_checkpoint"] and args.checkpoint:
        raise SystemExit(f"{args.variant} must run the PRETRAINED backbone "
                         "(no --checkpoint)")
    inner = argparse.Namespace(
        model=E1_KEY, n=e1.N_IMAGES, seed=e1.SEED,
        batch_size=args.batch_size, device=args.device,
        checkpoint=args.checkpoint,
        indices_from=str(e1.counts_path(E1_KEY)),
        out=str(spec["npz"]),
    )
    e1.cmd_collect(inner)


# ─────────────────────────────────────────────────────────────────────────────
# diagnose — CLS-patch cosine at the final normed output (one sweep/variant)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_diagnose(args):
    import torch
    from experiments.datasets import load_eval_dataset
    from experiments.models import backbone_transforms

    indices = [int(i) for i in
               np.load(e1.counts_path(E1_KEY), allow_pickle=True)["ds_indices"]]
    out: Dict[str, dict] = {}
    for vk, spec in VARIANTS.items():
        if not spec["npz"].exists():
            print(f"skip {vk}: {spec['npz']} missing")
            continue
        ckpt = _npz_checkpoint(spec["npz"]) if vk != "v1" else None
        backbone = _build_backbone(ckpt, args.device)
        transform, normalize = backbone_transforms(backbone)
        ds = load_eval_dataset("funny_birds", transform,
                               extra_kwargs={"split": "test"})
        n_prefix = int(backbone.num_prefix_tokens)     # 5 = 1 CLS + 4 reg
        cos_per_img = []
        with torch.no_grad():
            for s0 in range(0, len(indices), args.batch_size):
                x = torch.stack([ds[i][0] for i in indices[s0:s0 + args.batch_size]])
                f = backbone.forward_features(normalize(x.to(args.device)))
                cls = torch.nn.functional.normalize(f[:, 0], dim=-1)
                patch = torch.nn.functional.normalize(f[:, n_prefix:], dim=-1)
                cos = (patch * cls[:, None, :]).sum(-1).mean(-1)   # (B,)
                cos_per_img.append(cos.float().cpu().numpy())
        cos_all = np.concatenate(cos_per_img)
        out[vk] = dict(
            checkpoint=ckpt, n=len(cos_all),
            cls_patch_cosine_mean=float(cos_all.mean()),
            cls_patch_cosine_median=float(np.median(cos_all)),
            cls_patch_cosine_p95=float(np.percentile(cos_all, 95)),
            note="cosine(CLS, patch) at forward_features output (post final "
                 "norm), mean over patches then stats over the E1 N=256 images",
        )
        print(f"{vk}: mean CLS-patch cosine = {out[vk]['cls_patch_cosine_mean']:.4f}")
        del backbone
        torch.cuda.empty_cache()
    (OUT_DIR / "e3_diagnostics.json").write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_DIR / 'e3_diagnostics.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# analyze
# ─────────────────────────────────────────────────────────────────────────────

def _variant_report(vk: str) -> dict:
    import torch
    spec = VARIANTS[vk]
    d = np.load(spec["npz"], allow_pickle=True)
    norms, n_prefix, n_reg = d["norms"], int(d["n_prefix"]), int(d["n_reg"])
    flags, tau, patch = per_sample_flags(norms, n_prefix)
    frac = flags.mean(axis=(1, 2))
    union = flags.any(0)
    cnt = union.sum(1)
    maxmean = (patch.max(-1) / patch.mean(-1))         # (24, N)
    reg = norms[:, :, 1:1 + n_reg]
    reg_over_tau = reg > tau[:, :, None]

    ckpt = _npz_checkpoint(spec["npz"])
    val_acc = None
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        val_acc = float(ck["val_acc"])

    rep = dict(
        label=spec["label"], checkpoint=ckpt, val_acc=val_acc,
        N=int(norms.shape[1]), n_patch=int(patch.shape[-1]), n_reg=n_reg,
        frac_per_site=[float(f) for f in frac],
        plateau_attn=float(np.mean([frac[2 * b] for b in PLATEAU_BLOCKS])),
        plateau_mlp=float(np.mean([frac[2 * b + 1] for b in PLATEAU_BLOCKS])),
        union_frac=float(union.mean()),
        total_flagged_union=int(union.sum()),
        per_image=dict(mean=float(cnt.mean()), median=float(np.median(cnt)),
                       min=int(cnt.min()), max=int(cnt.max()),
                       frac_images_flagged=float((cnt > 0).mean())),
        patch_maxmean_ratio_per_site=[float(x) for x in np.median(maxmean, axis=1)],
        registers=dict(
            median_reg_norm_per_site=[float(x) for x in np.median(reg, axis=(1, 2))],
            median_patch_norm_per_site=[float(x) for x in np.median(patch, axis=(1, 2))],
            median_cls_norm_per_site=[float(x) for x in np.median(norms[:, :, 0], axis=1)],
            frac_reg_tokens_over_tau=float(reg_over_tau.mean()),
            plateau_frac_samples_any_reg_over_tau=float(np.mean(
                [reg_over_tau[2 * b + 1].any(-1).mean() for b in PLATEAU_BLOCKS])),
            registers_are_top4=_registers_are_top4(norms, n_reg),
        ),
    )
    return rep


def _registers_are_top4(norms: np.ndarray, n_reg: int) -> dict:
    """Per plateau MLP site: fraction of images whose top-4 non-CLS tokens by
    norm are exactly the 4 register tokens."""
    out = {}
    for b in PLATEAU_BLOCKS:
        s = 2 * b + 1
        toks = norms[s, :, 1:]                          # registers + patches
        top4 = np.argsort(-toks, axis=1)[:, :4]
        out[f"blk{b}.mlp"] = float(np.all(top4 < n_reg, axis=1).mean())
    return out


def cmd_analyze(args):
    report = {}
    for vk, spec in VARIANTS.items():
        if spec["npz"].exists():
            report[vk] = _variant_report(vk)
        else:
            print(f"skip {vk}: {spec['npz']} missing")

    # sanity: v3 (frozen-probe control) must equal v1 EXACTLY
    if "v1" in report and "v3" in report:
        n1 = np.load(VARIANTS["v1"]["npz"], allow_pickle=True)["norms"]
        n3 = np.load(VARIANTS["v3"]["npz"], allow_pickle=True)["norms"]
        exact = bool(np.array_equal(n1, n3))
        report["v3_equals_v1"] = dict(
            norms_exactly_equal=exact,
            max_abs_diff=float(np.abs(n1 - n3).max()) if n1.shape == n3.shape else None,
        )
        print(f"v3 == v1 exactly: {exact}")

    (OUT_DIR / "e3_analysis.json").write_text(json.dumps(report, indent=2))
    print(f"saved {OUT_DIR / 'e3_analysis.json'}")
    for vk in ("v1", "v2a", "v2b", "v3"):
        if vk not in report:
            continue
        r = report[vk]
        print(f"{vk:4s} plateau attn {100*r['plateau_attn']:.3f}%  "
              f"mlp {100*r['plateau_mlp']:.3f}%  union {100*r['union_frac']:.3f}%  "
              f"per-image mean {r['per_image']['mean']:.1f}  "
              f"val_acc={r['val_acc']}")


# ─────────────────────────────────────────────────────────────────────────────
# figures
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, stem: str, paper_name: Optional[str] = None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {FIG_DIR / stem}.png/.pdf")
    if paper_name and PAPER_FIG_DIR.is_dir():
        shutil.copy(FIG_DIR / f"{stem}.pdf", PAPER_FIG_DIR / paper_name)
        print(f"copied → {PAPER_FIG_DIR / paper_name}")


def cmd_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((OUT_DIR / "e3_analysis.json").read_text())
    e1_report = json.loads((OUT_DIR / "e1_analysis.json").read_text())
    x = np.arange(2 * DEPTH)

    # 1) fraction per site: pretrained vs both finetune arms (+ v3 markers),
    #    M1 standard-ViT as reference line
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(x, 100 * np.asarray(e1_report[E1_STD_KEY]["frac_per_site"]), lw=1.6,
            ls="--", color=STD_COLOR, alpha=0.9,
            label="reference: standard ViT-S, no registers (M1, E1 row)")
    for vk in ("v1", "v2a", "v2b"):
        if vk not in report:
            continue
        spec = VARIANTS[vk]
        ax.plot(x, 100 * np.asarray(report[vk]["frac_per_site"]), lw=2,
                marker="o", ms=4, color=spec["color"], ls=spec["ls"],
                label=spec["short"])
    if "v3" in report:
        ax.plot(x, 100 * np.asarray(report["v3"]["frac_per_site"]), lw=0,
                marker="x", ms=6, color=VARIANTS["v3"]["color"],
                label="frozen-probe control (= pretrained exactly)")
    ax.set_xticks(x[::2], [str(b) for b in range(DEPTH)])
    ax.set_xlabel("site in network order (tick = post-attn half of block; "
                  "next point = post-MLP half)")
    ax.set_ylabel("patch-token outliers (%)")
    ax.set_title("E3 — patch-token outlier fraction per half-block site, "
                 f"pre vs post finetune (norm > μ + {SD_K:g}σ per sample; "
                 "N=256 FunnyBirds test)")
    e1._style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _save(fig, "e3_fraction_finetune", "e3_fraction_finetune.pdf")
    plt.close(fig)

    # 2) register norms pre/post finetune — one panel per model variant
    panels = [vk for vk in ("v1", "v2a", "v2b") if vk in report]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, vk in zip(axes, panels):
        spec = VARIANTS[vk]
        d = np.load(spec["npz"], allow_pickle=True)
        n_reg = int(d["n_reg"])
        reg = d["norms"][:, :, 1:1 + n_reg]
        n_prefix = int(d["n_prefix"])
        _, tau, patch = per_sample_flags(d["norms"], n_prefix)
        for r in range(n_reg):
            ax.plot(x, np.median(reg[:, :, r], axis=1), lw=1.2,
                    color=spec["color"], alpha=0.85,
                    label="register tokens (4)" if r == 0 else None)
        ax.plot(x, np.median(patch, axis=(1, 2)), lw=2, color="#52514e",
                label="patch tokens (median)")
        ax.plot(x, np.median(tau, axis=1), lw=2, ls="--", color="#0b0b0b",
                label="patch outlier threshold μ+4σ (median)")
        ax.plot(x, np.median(d["norms"][:, :, 0], axis=1), lw=1.4, ls=":",
                color="#52514e", label="CLS token")
        ax.set_yscale("log")
        ax.set_xticks(x[::2], [str(b) for b in range(DEPTH)])
        ax.set_xlabel("site (block; attn/MLP halves interleaved)")
        acc = report[vk]["val_acc"]
        ax.set_title(spec["short"] + (f"  (val {acc:.3f})" if acc else ""))
        e1._style(ax)
    axes[0].set_ylabel("token L2 norm (median over N=256)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("E3 — register vs patch token norms before/after end-to-end "
                 "finetune (DINOv3-S/16, FunnyBirds)", y=1.02)
    _save(fig, "e3_registers_finetune", "e3_registers_finetune.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────────────

def cmd_report(args):
    report = json.loads((OUT_DIR / "e3_analysis.json").read_text())
    e1_report = json.loads((OUT_DIR / "e1_analysis.json").read_text())
    diag = (json.loads((OUT_DIR / "e3_diagnostics.json").read_text())
            if (OUT_DIR / "e3_diagnostics.json").exists() else {})
    std = e1_report[E1_STD_KEY]

    def row(vk):
        r = report[vk]
        rg = r["registers"]
        acc = f"{r['val_acc']:.4f}" if r["val_acc"] is not None else "—"
        cos = (f"{diag[vk]['cls_patch_cosine_mean']:.3f}" if vk in diag else "—")
        top4 = np.mean(list(rg["registers_are_top4"].values()))
        mm = max(r["patch_maxmean_ratio_per_site"])
        return (f"| {VARIANTS[vk]['short']} | {acc} | "
                f"{100*r['plateau_attn']:.3f} | {100*r['plateau_mlp']:.3f} | "
                f"{100*r['union_frac']:.3f} | {r['per_image']['mean']:.1f} "
                f"({r['per_image']['min']}..{r['per_image']['max']}) | "
                f"{100*top4:.0f}% | {mm:.2f} | {cos} |")

    table = [
        "| variant | val top-1 | plateau % (attn) | plateau % (MLP) | "
        "union % | per-image mean (min..max) | registers=top-4 | "
        "patch max/mean (site max, median) | CLS-patch cos |",
        "|---|---|---|---|---|---|---|---|---|",
    ] + [row(vk) for vk in ("v1", "v2a", "v2b", "v3") if vk in report]
    table.append(
        f"| _reference: standard ViT-S (M1, no registers)_ | 0.9738 | "
        f"{100*std['plateau_attn']:.3f} | {100*std['plateau_mlp']:.3f} | "
        f"{100*std['union_frac']:.3f} | {std['per_image']['mean']:.1f} "
        f"({std['per_image']['min']}..{std['per_image']['max']}) | — | — | — |")

    eq = report.get("v3_equals_v1", {})

    def verdict_lines():
        v1p = report["v1"]["plateau_mlp"]
        std_p = std["plateau_mlp"]
        lines = []
        for vk, name in (("v2a", "Arm A (naive, lr 5e-4)"),
                         ("v2b", "Arm B (conservative, lr 5e-6)")):
            if vk not in report:
                continue
            p = report[vk]["plateau_mlp"]
            frac_of_std = p / max(std_p, 1e-12)
            vs_pre = p / max(v1p, 1e-12) if v1p > 0 else float("inf")
            reappeared = p > 0.1 * std_p       # >10% of the standard-ViT level
            lines.append(
                f"- **{name}**: plateau MLP outlier fraction "
                f"{100*p:.3f}% vs pretrained {100*v1p:.3f}% and standard-ViT "
                f"{100*std_p:.3f}% → {frac_of_std:.1%} of the no-register "
                f"level ({'×%.1f' % vs_pre if v1p > 0 else '∞×'} the "
                f"pretrained level). Patch outliers "
                f"**{'REAPPEARED' if reappeared else 'did NOT reappear'}** "
                f"(criterion: >10% of the standard-ViT plateau).")
        return lines

    md = REPO_ROOT / "research" / "registers" / "e3_finetune_reintroduction.md"
    parts = [
        "# E3 — does end-to-end finetuning reintroduce register outliers?",
        f"_Generated {_now()} by `experiments/scripts/registers_e3_finetune.py` "
        "(collect / diagnose / analyze / figures / report)._",
        "",
        "## Experiment card",
        "**RQ.** The pretrained DINOv3-S/16 backbone has (essentially) no "
        "high-norm patch-token outliers — its 4 register tokens absorb that "
        "role (E1). Does END-TO-END finetuning on FunnyBirds re-create patch "
        "outliers, and does a conservative finetune recipe prevent it?",
        "",
        "**H1.** End-to-end finetuning reintroduces patch-token outliers "
        "(the register mechanism does not survive task finetuning at "
        "standard LR). **H0.** The registers keep absorbing the outlier "
        "role; patch tokens stay clean.",
        "",
        "**Falsified if** (H1 rejected): after finetuning, the plateau "
        "patch-outlier fraction stays below 10% of the standard-ViT (M1) "
        "plateau level AND the 4 register tokens remain the top-4 non-CLS "
        "norms in the plateau blocks.",
        "",
        "## Method",
        "- Measurement identical to E1 (`registers_e1_counts.py`): 24 "
        "half-block sites, per-sample per-site patch-norm criterion "
        f"μ+{SD_K:g}σ (CLS + 4 registers excluded from patch stats), N=256 "
        "FunnyBirds test images, SAME indices as E1 (reused verbatim from "
        "the E1 npz), site identity re-verified numerically per variant.",
        "- v2a Arm A: `train_probe finetune --from-scratch` M1 protocol "
        "(25 ep, backbone_lr 5e-4, head_lr 5e-3, LLRD 0.7, wd 0.05, "
        "bs 64×2, bf16, onecycle 0.1, RandAugment, ls 0.1, seed 0), "
        "registers trainable.",
        "- v2b Arm B: same, but backbone_lr 5e-6, head_lr 1e-3 "
        "(conservative literature recipe; scout note).",
        "- v3: frozen-backbone CLS-token linear probe (Phase 2) — backbone "
        "identical to pretrained; collected independently as a pipeline "
        "sanity check (must equal v1 exactly).",
        "- Diagnostics (DINOv3-report style): per-site patch-norm max/mean "
        "ratio (median over images, site max reported) and mean CLS-patch "
        "cosine at the final normed output.",
        "",
        "## Decision table",
        *table,
        "",
        f"Sanity — v3 arrays equal v1 exactly: "
        f"**{eq.get('norms_exactly_equal', 'not checked')}** "
        f"(max abs diff {eq.get('max_abs_diff', 'n/a')}).",
        "",
        "## Verdict",
        *verdict_lines(),
        "",
        "## Files",
        "- Arrays: `data/results/registers/e3_finetune_{v2a,v2b,v3}.npz`, "
        "`e3_analysis.json`, `e3_diagnostics.json`; v1 = "
        "`e1_counts_dinov3_small_funny_birds.npz` (reused).",
        "- Figures: `figures/registers/e3_finetune/e3_fraction_finetune.{png,pdf}`, "
        "`e3_registers_finetune.{png,pdf}`; paper copies "
        "`iclr2026/journal-figures/e3_{fraction,registers}_finetune.pdf`.",
    ]
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(parts) + "\n")
    print(f"saved {md}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("collect")
    pc.add_argument("--variant", choices=["v2a", "v2b", "v3"], required=True)
    pc.add_argument("--checkpoint", default=None,
                    help="best.pt of the matching finetune arm (v2a/v2b)")
    pc.add_argument("--batch-size", type=int, default=32)
    pc.add_argument("--device", default="cuda")
    pc.set_defaults(fn=cmd_collect)
    pd = sub.add_parser("diagnose")
    pd.add_argument("--batch-size", type=int, default=32)
    pd.add_argument("--device", default="cuda")
    pd.set_defaults(fn=cmd_diagnose)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    sub.add_parser("figures").set_defaults(fn=cmd_figures)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
