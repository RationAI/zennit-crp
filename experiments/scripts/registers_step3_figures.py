"""XAI-37 step 3 — figures for the H_B occlusion test (png+pdf, self-explanatory).

Reads the arrays produced by ``registers_step3_occlusion.py`` (scan/selection/
lrp/occlude/analysis npz under data/results/registers/) and writes:

* dp_by_condition       — delta target-prob distributions per occlusion condition
* faithfulness_scatter  — LRP |R| mass on outlier patches vs measured delta-p
* relocation            — relocation/persistence rates + delta-p split by relocation
* examples              — 3 example images: outlier patches, LRP map, occluded input

CPU-only (dataset images are re-read for the example montage; no forwards).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RES = REPO_ROOT / "data" / "results" / "registers"
FIG = REPO_ROOT / "figures" / "registers" / "step3_occlusion"
FIG.mkdir(parents=True, exist_ok=True)

GRID, PATCH = 14, 16
# dataviz palette (light mode): blue = outlier-patch occlusions, orange = controls
C_OUT, C_CTL, C_MUT = "#2a78d6", "#eb6834", "#6b6b6b"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {FIG / name}.png/.pdf")


def main():
    sel = np.load(RES / "step3_selection.npz")
    occ = np.load(RES / "step3_occlusion.npz")
    ana = np.load(RES / "step3_analysis.npz")
    rel = np.load(RES / "step3_lrp.npz")["rel"]          # (128, 14, 14)
    N = len(sel["ds_idx"])

    conds = ["a", "b", "a_all", "c", "d"]
    labels = {
        "a": "a: primary outlier\n(neighbor-mean)",
        "b": "b: primary outlier\n(matched noise)",
        "a_all": "a_all: ALL outlier\npatches (median 4)",
        "c": "c: random background\npatch (control)",
        "d": "d: top-relevance\npatch (control)",
    }
    colors = {"a": C_OUT, "b": C_OUT, "a_all": C_OUT, "c": C_CTL, "d": C_CTL}
    dps = {k: ana[f"dp_{k}"] for k in conds}

    # ── figure 1: delta-p per condition ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = np.random.default_rng(0)
    for i, k in enumerate(conds):
        y = dps[k]
        ax.scatter(i + rng.uniform(-0.16, 0.16, N), y, s=7, alpha=0.35,
                   color=colors[k], linewidths=0, zorder=2)
        med, q1, q3 = np.median(y), *np.percentile(y, [25, 75])
        ax.plot([i - 0.28, i + 0.28], [med, med], color="black", lw=1.6, zorder=3)
        ax.plot([i - 0.28, i - 0.28, i + 0.28, i + 0.28, i - 0.28],
                [q1, q3, q3, q1, q1], color="black", lw=0.7, zorder=3)
        ax.annotate(f"median\n{med:+.4f}", (i, -0.075), ha="center", va="top",
                    fontsize=7.5, color="black")
    ax.axhline(0, color=C_MUT, lw=0.8, zorder=1)
    ax.set_xticks(range(len(conds)), [labels[k] for k in conds], fontsize=7.5)
    ax.set_ylim(-0.12, 0.05)
    ax.set_ylabel(r"$\Delta$ p(target)  (occluded $-$ clean)")
    ax.set_title("Occluding outlier-token patches barely moves the prediction "
                 "(ViT-B/16, N=128, median clean p=0.85)\n"
                 "blue = outlier-patch occlusions, orange = control patches; "
                 "box = IQR, line = median; y clipped at $-0.12$", fontsize=9)
    n_clip = sum((dps[k] < -0.12).sum() for k in conds)
    ax.annotate(f"{n_clip} pts below axis", (0.99, 0.02), xycoords="axes fraction",
                ha="right", fontsize=7, color=C_MUT)
    save(fig, "dp_by_condition")

    # ── figure 2: faithfulness scatter ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    panels = [
        ("frac_prim", "dp_a", "primary outlier patch\n(condition a)", axes[0]),
        ("frac_out", "dp_a_all", "all outlier patches\n(condition a_all)", axes[1]),
    ]
    for fr_key, dp_key, ttl, ax in panels:
        x, y = ana[fr_key], np.abs(ana[dp_key])
        ax.scatter(x, y, s=12, alpha=0.55, color=C_OUT, linewidths=0)
        rho, p = stats.spearmanr(x, y)
        ax.set_title(f"{ttl}\nSpearman $\\rho$={rho:+.2f} (p={p:.2g})", fontsize=8.5)
        ax.set_xlabel("LRP |R| mass fraction on patch(es)")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.axvline(1 / 196, color=C_MUT, lw=0.8, ls="--")
        ax.annotate("uniform share", (1 / 196, ax.get_ylim()[1]), fontsize=7,
                    color=C_MUT, rotation=90, va="top", ha="right")
    axes[0].set_ylabel(r"|$\Delta$ p(target)|  (symlog)")
    fig.suptitle("High LRP relevance on outlier patches is NOT matched by causal "
                 "effect (cp_lrp_baseline, y=target)", fontsize=9.5, y=1.12)
    save(fig, "faithfulness_scatter")

    # ── figure 3: relocation guard ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]
    rates = [
        ("new outlier position\nappeared (relocation)", ana["reloc"].mean()),
        ("occluded patch STILL\nan outlier", ana["persist_primary"].mean()),
        ("prediction preserved\n(argmax = target)", (occ["pred_a"] == sel["target"]).mean()),
    ]
    ys = np.arange(len(rates))[::-1]
    ax.barh(ys, [r for _, r in rates], height=0.55, color=C_OUT)
    for y0, (_, r) in zip(ys, rates):
        ax.annotate(f"{r:.0%}", (r + 0.02, y0), va="center", fontsize=8.5)
    ax.set_yticks(ys, [n for n, _ in rates], fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("After occluding the primary outlier patch\n(condition a, clean thresholds)",
                 fontsize=8.5)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    groups = [("relocated\n(n=%d)" % ana["reloc"].sum(), dps["a"][ana["reloc"]]),
              ("not relocated\n(n=%d)" % (~ana["reloc"]).sum(), dps["a"][~ana["reloc"]])]
    for i, (name, y) in enumerate(groups):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=8, alpha=0.4,
                   color=C_OUT, linewidths=0)
        ax.plot([i - 0.22, i + 0.22], [np.median(y)] * 2, color="black", lw=1.6)
        ax.annotate(f"{np.median(y):+.4f}", (i + 0.26, np.median(y)), fontsize=7.5,
                    va="center")
    u = stats.mannwhitneyu(groups[0][1], groups[1][1])
    ax.axhline(0, color=C_MUT, lw=0.8)
    ax.set_xticks([0, 1], [g[0] for g in groups], fontsize=8)
    ax.set_ylim(-0.05, 0.03)
    ax.set_ylabel(r"$\Delta$ p(target), condition a")
    ax.set_title(f"$\\Delta$p is negligible with AND without relocation\n"
                 f"(Mann-Whitney p={u.pvalue:.2f}) — hydra effect does not\n"
                 f"explain the null", fontsize=8.5)
    save(fig, "relocation")

    # ── figure 4: examples ───────────────────────────────────────────────────
    from experiments.crp_gallery import load_eval_dataset, load_model  # noqa: F401
    from experiments.model_io import backbone_transforms
    import timm
    tm = timm.create_model("vit_base_patch16_224", pretrained=False)
    transform, _ = backbone_transforms(tm)
    ds = load_eval_dataset("imagenet", transform, {"n_per_class": 10})

    # pick 3 images with the largest LRP mass on the primary outlier patch
    order = np.argsort(-ana["frac_prim"])[:3]
    fig, axes = plt.subplots(3, 3, figsize=(7.6, 7.9))
    for row, i in enumerate(order):
        di, p0 = int(sel["ds_idx"][i]), int(sel["primary"][i])
        x = ds[di][0].numpy().transpose(1, 2, 0)
        r, c = divmod(p0, GRID)
        ax = axes[row, 0]
        ax.imshow(x)
        for q in np.flatnonzero(sel["out_mask"][i]):
            qr, qc = divmod(int(q), GRID)
            lw, col = (2.0, C_OUT) if q == p0 else (1.2, "#9dc3ee")
            ax.add_patch(plt.Rectangle((qc * PATCH, qr * PATCH), PATCH, PATCH,
                                       fill=False, color=col, lw=lw))
        ax.set_title("clean + outlier patches\n(thick = primary)" if row == 0 else "",
                     fontsize=8)
        ax.set_ylabel(f"img {di}", fontsize=8)

        ax = axes[row, 1]
        rm = rel[i]
        v = np.abs(rm).max()
        ax.imshow(rm, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   color="black", lw=1.5))
        ax.set_title("LRP patch relevance (14x14)\nred=+, blue=$-$" if row == 0 else "",
                     fontsize=8)
        ax.annotate(f"{ana['frac_prim'][i]:.0%} of image |R|\non boxed patch",
                    (0.03, 0.03), xycoords="axes fraction", fontsize=7,
                    color="black",
                    bbox=dict(fc="white", alpha=0.8, ec="none", pad=1.5))

        ax = axes[row, 2]
        xa = x.copy()
        nb = [q for q in
              [(r + dr) * GRID + (c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if (dr, dc) != (0, 0) and 0 <= r + dr < GRID and 0 <= c + dc < GRID]
              if q not in set(np.flatnonzero(sel["out_mask"][i]).tolist())]
        fill = np.mean([x[(q // GRID) * PATCH:(q // GRID + 1) * PATCH,
                          (q % GRID) * PATCH:(q % GRID + 1) * PATCH].mean((0, 1))
                        for q in nb], axis=0) if nb else x.mean((0, 1))
        xa[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = fill
        ax.imshow(xa)
        ax.set_title("condition a (occluded)" if row == 0 else "", fontsize=8)
        ax.annotate(f"$\\Delta$p = {ana['dp_a'][i]:+.4f}", (0.03, 0.03),
                    xycoords="axes fraction", fontsize=8,
                    bbox=dict(fc="white", alpha=0.8, ec="none", pad=1.5))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Examples with the LARGEST LRP relevance on the primary outlier "
                 "patch — occlusion still changes nothing", fontsize=9.5)
    fig.tight_layout()
    save(fig, "examples")


if __name__ == "__main__":
    main()
