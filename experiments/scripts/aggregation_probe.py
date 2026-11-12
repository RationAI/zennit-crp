"""Control for the ranking's token aggregation.

The detector score used throughout is the **signed sum over tokens** of the
class-conditional relevance at the probe site. If an LRP ranking looks like
chance in the middle blocks, an obvious alternative explanation is that the
signed sum cancels there — positive and negative token contributions of the
same dimension destroying each other — so the failure would belong to the
aggregation, not to the attribution.

This probe recomputes the ranking from the same relevance tensors under three
aggregations and measures each one's top-k agreement with the best ordering
found by search:

    signed   sum_t  R[t, d]            (what the benchmark uses)
    abs      sum_t |R[t, d]|
    maxabs   max_t |R[t, d]|

If all three stay at the chance floor where the signed sum does, cancellation
is ruled out and the failure is the attribution's.

Writes figures/rank_analysis/aggregation_probe/<key>__<concept>/*.{png,pdf}
and prints the table it plots.

Run:  uv run python -m experiments.scripts.aggregation_probe --model-key <key>
          [--concept embed_dim|head] [--sites value qk] [-k 8]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crp.concepts import EmbeddingDimConcept, HeadConcept
from experiments import rank_analysis as ra
from experiments.concept_detector_bench import (
    COMPOSITE, MODELS_CFG, N_IMAGES, layer_name, method_sites, select_correct)
from experiments.datasets import load_eval_dataset
from experiments.gradinput import GradTimesInputAttribution
from experiments.model_datasets import find_by_tag
from experiments.models import backbone_transforms

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "rank_analysis" / "aggregation_probe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGGS = ("signed", "abs", "maxabs")
STYLE = {"signed": ("#1f77b4", "o-"), "abs": ("#ff7f0e", "s--"),
         "maxabs": ("#2ca02c", "^:")}


def aggregate(rel: torch.Tensor, how: str, head_dim: int | None):
    """(B, N, embed_dim) relevance -> (D,) detector scores."""
    r = rel[0]                                    # (N, embed_dim)
    if how == "signed":
        v = r.sum(0)
    elif how == "abs":
        v = r.abs().sum(0)
    else:
        v = r.abs().max(0).values
    if head_dim:                                  # head detectors: pool each slice
        v = v.reshape(-1, head_dim)
        v = v.sum(1) if how != "maxabs" else v.max(1).values
    return v.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="vit_small_funny_birds")
    ap.add_argument("--concept", choices=["embed_dim", "head"], default="embed_dim")
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("-k", type=int, default=None)
    a = ap.parse_args()

    cells = ra.load_cells(a.model_key, a.concept)
    sites = a.sites or cells.sites(ra.ATTENTION_SITES)
    k = a.k or ra.report_k(cells.D)
    cfg = next(c for c in MODELS_CFG if c[0] == a.model_key)
    label = cfg[4]

    model = find_by_tag(a.model_key, device=DEVICE).model.eval()
    transform, normalize = backbone_transforms(model.backbone)
    ds = load_eval_dataset(cfg[2], transform, cfg[3])
    num_heads = model.backbone.blocks[0].attn.num_heads
    embed_dim = int(model.backbone.embed_dim)
    head_dim = embed_dim // num_heads if a.concept == "head" else None
    concept_obj = (HeadConcept(num_heads=num_heads) if a.concept == "head"
                   else EmbeddingDimConcept(num_heads=num_heads))
    picks = select_correct(model, normalize, ds, N_IMAGES)
    if [i for i, _, _ in picks] != cells.image_ids:
        raise SystemExit("image sample differs from the stored benchmark")

    attribution = GradTimesInputAttribution(model)
    outdir = OUT / f"{a.model_key}__{a.concept}"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    for method in ra.LRP_METHODS:
        msites = [s for s in sites if s in method_sites(method, sites)]
        msites = [s for s in msites if (method, s, cells.blocks(s)[-1]) in cells.order]
        if not msites:
            continue
        for site in msites:
            blocks = [b for b in cells.blocks(site)
                      if ra.best_search_order(cells, site, b) is not None]
            if not blocks:
                continue
            scores = {g: np.zeros((len(picks), len(blocks))) for g in AGGS}
            for j, (idx, pred, _) in enumerate(picks):
                x = ds[idx][0].unsqueeze(0).to(DEVICE)
                xin = normalize(x).clone().detach().requires_grad_(True)
                layers = [layer_name(site, b) for b in blocks]
                res = attribution(xin, [{"y": [int(pred)]}], COMPOSITE[method](),
                                  record_layer=layers)
                for bi, b in enumerate(blocks):
                    rel = res.relevances[layer_name(site, b)]
                    ref = ra.best_search_order(cells, site, b)[j]
                    for g in AGGS:
                        o = np.argsort(-aggregate(rel, g, head_dim), kind="stable")
                        scores[g][j, bi] = len(set(o[:k].tolist())
                                               & set(ref[:k].tolist())) / k
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            for g in AGGS:
                col, ls = STYLE[g]
                m = scores[g].mean(0)
                se = scores[g].std(0, ddof=1) / np.sqrt(scores[g].shape[0])
                ax.plot(blocks, m, ls, ms=3.5, lw=1.4, color=col, label=g)
                ax.fill_between(blocks, m - 1.96 * se, m + 1.96 * se,
                                color=col, alpha=0.13, lw=0)
            ax.axhline(k / cells.D, color="k", lw=1.0, ls=":")
            ax.text(0.995, k / cells.D, " chance (k/D)", ha="right", va="bottom",
                    fontsize=7.5, transform=ax.get_yaxis_transform())
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("transformer block")
            ax.set_ylabel(f"top-{k} overlap with the best ordering")
            ax.set_title(
                f"Does the token aggregation explain the failure? · {label}\n"
                f"{ra.METHOD_LABEL[method]}, site {ra.SITE_LABEL[site]}, "
                f"{a.concept} detectors (D={cells.D}), n={len(picks)} images; "
                f"mean, 95% CI", fontsize=9.5)
            ax.legend(fontsize=8, frameon=False, title="token aggregation",
                      title_fontsize=8)
            ax.grid(alpha=0.25, lw=0.5)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(outdir / f"aggregation_{method}_{site}.{ext}", dpi=160)
            plt.close(fig)
            print(f"\n{a.model_key} · {method} · {site} · top-{k} overlap "
                  f"(chance {k / cells.D:.3f})")
            print("  block " + " ".join(f"{b:>6d}" for b in blocks))
            for g in AGGS:
                print(f"  {g:7s}" + " ".join(f"{v:6.3f}" for v in scores[g].mean(0)))
    print(f"\nfigures -> {outdir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
