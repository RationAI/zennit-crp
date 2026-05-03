"""Per-remedy diagnostic sweep for DINOv3 ViT-L/16 + AttnLRP.

Loads frozen DINOv3 backbone + the linear-probe head trained by
``train_dinov3_probe.py``, picks a small sample of correctly-classified
images, and runs every named remedy composite from
``crp.transformer_patches`` through ``diagnose_attribution``. Aggregates
the per-layer health stats and writes a structured markdown report to
``tutorials/vit_crp/dinov3_variants/RESULTS.md``.

Usage::

    uv run python experiments/run_dinov3_remedy_eval.py \\
        --probe data/vit_large_patch16_dinov3_probe_imagenette.pt \\
        --n-samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import timm
from timm.data import resolve_data_config, create_transform

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from datasets import load as load_dataset  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from crp.attribution import CondAttribution  # noqa: E402
from crp.attention_concepts import HeadConcept  # noqa: E402
from crp.transformer_patches import (  # noqa: E402
    AttnLRPEpsilonComposite,
    AttnLRPGammaComposite,
    AttnLRPMatmulFactor2Composite,
    AttnLRPSignedEpsilonComposite,
    AttnLRPRopeDetachComposite,
    AttnLRPLayerScaleUniformComposite,
    AttnLRPLinearGammaComposite,
    AttnLRPCombinedComposite,
)
from dinov3_diagnose import (  # noqa: E402
    diagnose_attribution,
    format_report,
    diverges_at,
)


# Each entry: (label, factory). Factory takes no args. Each composite is
# a single-responsibility class — see crp/transformer_patches.py.
REMEDIES = [
    ("baseline_epsilon",          lambda: AttnLRPEpsilonComposite()),
    ("baseline_gamma",            lambda: AttnLRPGammaComposite(gamma=0.25)),
    ("matmul_factor_2",           lambda: AttnLRPMatmulFactor2Composite()),
    ("signed_epsilon",            lambda: AttnLRPSignedEpsilonComposite()),
    ("rope_detach",               lambda: AttnLRPRopeDetachComposite()),
    ("layerscale_uniform",        lambda: AttnLRPLayerScaleUniformComposite()),
    ("linear_gamma_005",          lambda: AttnLRPLinearGammaComposite(gamma=0.05)),
    # one combined "kitchen sink" so the user can see whether the remedies
    # interact constructively when stacked
    ("combined_all",              lambda: AttnLRPCombinedComposite(
        matmul_factor_2=True, signed_epsilon=True, rope_detach=True,
        layerscale_uniform=True, residual_lrp="ratio",
    )),
    # pair-combinations on top of the only individually-surviving remedy
    # — does adding each other remedy on top of layerscale help or hurt?
    ("layerscale+signed",         lambda: AttnLRPCombinedComposite(
        layerscale_uniform=True, signed_epsilon=True, residual_lrp="ratio",
    )),
    ("layerscale+rope_detach",    lambda: AttnLRPCombinedComposite(
        layerscale_uniform=True, rope_detach=True, residual_lrp="ratio",
    )),
    ("layerscale+matmul",         lambda: AttnLRPCombinedComposite(
        layerscale_uniform=True, matmul_factor_2=True, residual_lrp="ratio",
    )),
    ("layerscale+linear_gamma_005", lambda: AttnLRPCombinedComposite(
        layerscale_uniform=True, linear_gamma=0.05, residual_lrp="ratio",
    )),
    ("layerscale+ratio_residual_only", lambda: AttnLRPCombinedComposite(
        layerscale_uniform=True, residual_lrp="ratio",
    )),
]


def build_model(probe_path: Path, device: str):
    ckpt = torch.load(probe_path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    model = timm.create_model(model_name, pretrained=True, num_classes=ckpt["num_classes"])
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval().to(device)
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=False)
    return model, model_name, transform


def pick_samples(model, dataset, n: int, device: str):
    """Pick `n` correctly-classified images, one per distinct class so the
    sweep doesn't sit on a single basin of attention behaviour."""
    seen_classes: set[int] = set()
    picks = []
    # Stride through the dataset so picks are class-diverse — Imagenette
    # is ordered by class so contiguous indices give the same class.
    n_total = len(dataset)
    stride = max(1, n_total // (n * 6))
    for i in range(0, n_total, stride):
        x, y = dataset[i]
        if int(y) in seen_classes:
            continue
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        pred = logits.argmax(-1).item()
        if pred == int(y):
            picks.append((x.detach().requires_grad_(True), pred))
            seen_classes.add(int(y))
            if len(picks) == n:
                break
    if len(picks) < n:
        print(f"  warning: only {len(picks)} correct class-distinct samples found")
    return picks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path,
                   default=REPO_ROOT / "data/vit_large_patch16_dinov3_probe_imagenette.pt")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "tutorials/vit_crp/dinov3_variants")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"device   : {device}")
    print(f"probe    : {args.probe}")
    print(f"out-dir  : {args.out_dir}")

    print(f"\nloading model + probe head")
    model, model_name, transform = build_model(args.probe, device)
    print(f"  {model_name} (embed_dim={model.embed_dim}, blocks={len(model.blocks)})")

    print(f"\nloading Imagenette val")
    dataset = load_dataset("imagenette", split="val", transform=transform)
    print(f"  {len(dataset)} images")

    print(f"\npicking {args.n_samples} correctly-classified samples")
    samples = pick_samples(model, dataset, args.n_samples, device)
    print(f"  picked {len(samples)} samples (classes: {[s[1] for s in samples]})")

    attribution = CondAttribution(model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dump = {}
    summary_rows = []

    print(f"\nrunning {len(REMEDIES)} remedies × {len(samples)} samples")
    t0 = time.time()
    for label, factory in REMEDIES:
        per_sample = []
        for s_idx, (x, cls) in enumerate(samples):
            composite = factory()
            t_start = time.time()
            report = diagnose_attribution(
                model, attribution, composite, x.clone().requires_grad_(True),
                target_class=cls, composite_label=label,
                num_prefix_tokens=5,
            )
            elapsed = time.time() - t_start
            div = diverges_at(report, threshold=1e10)
            per_sample.append({
                "sample": s_idx,
                "class": cls,
                "elapsed_s": round(elapsed, 2),
                "finite_heatmap": report.finite_heatmap,
                "heatmap_max_abs": report.heatmap_max_abs,
                "heatmap_focus_top10pct": report.heatmap_focus_top10pct,
                "register_leak_share": report.register_leak_share,
                "diverges_at": div,
                "per_layer": [asdict(h) for h in report.per_layer],
                "error": report.error,
            })
            print(f"  {label:25} sample{s_idx} cls{cls}  "
                  f"finite={report.finite_heatmap}  "
                  f"max|R|={report.heatmap_max_abs:.2e}  "
                  f"focus={report.heatmap_focus_top10pct:.3f}  "
                  f"reg_leak={report.register_leak_share:.3f}  "
                  f"div@{div}  ({elapsed:.1f}s)")

        raw_dump[label] = per_sample
        # Aggregate one-row summary per remedy.
        finite_count = sum(1 for s in per_sample if s["finite_heatmap"])
        max_abs_med = _median([s["heatmap_max_abs"] for s in per_sample
                                if s["finite_heatmap"]])
        focus_med = _median([s["heatmap_focus_top10pct"] for s in per_sample
                              if s["finite_heatmap"] and not _isnan(s["heatmap_focus_top10pct"])])
        leak_med = _median([s["register_leak_share"] for s in per_sample
                              if not _isnan(s["register_leak_share"])])
        summary_rows.append({
            "remedy": label,
            "finite_n": f"{finite_count}/{len(per_sample)}",
            "max_abs_median": max_abs_med,
            "focus_top10pct_median": focus_med,
            "register_leak_median": leak_med,
            "first_div_layer": per_sample[0]["diverges_at"] if per_sample else None,
        })

    elapsed_total = time.time() - t0
    print(f"\ntotal: {elapsed_total:.1f}s")

    # Dump raw JSON for downstream notebook generation.
    raw_path = args.out_dir / "diagnostic_raw.json"
    raw_path.write_text(json.dumps(raw_dump, indent=2, default=_json_default))
    print(f"\nwrote {raw_path}")

    # Pretty markdown summary.
    md_path = args.out_dir / "RESULTS.md"
    md_path.write_text(_format_markdown(summary_rows, raw_dump, model_name, args.n_samples))
    print(f"wrote {md_path}")


def _median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _isnan(x):
    return isinstance(x, float) and x != x


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _format_markdown(summary_rows, raw_dump, model_name, n_samples):
    lines = []
    lines.append(f"# DINOv3 LRP remedy diagnostic — `{model_name}`")
    lines.append("")
    lines.append(
        f"Sweep across {len(summary_rows)} remedy composites × {n_samples} "
        f"correctly-classified Imagenette images. Each cell of the diagnostic "
        f"records relevance health at every `blocks.{{i}}.attn.attn_out_tap`. "
        f"See `crp/transformer_patches.py` for the per-remedy composite classes."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Remedy | finite (heatmap) | median max\\|R\\| | median focus@10% | median register-leak | first divergence |")
    lines.append("|--------|------------------|-----------------:|-----------------:|---------------------:|------------------|")
    for r in summary_rows:
        lines.append(
            f"| `{r['remedy']}` | {r['finite_n']} | "
            f"{_fmt_num(r['max_abs_median'])} | "
            f"{_fmt_num(r['focus_top10pct_median'])} | "
            f"{_fmt_num(r['register_leak_median'])} | "
            f"{r['first_div_layer'] or '—'} |"
        )
    lines.append("")
    lines.append(
        "**Interpretation.** `finite=N/N` means the heatmap has no NaN/Inf "
        "after channel sum. `max|R|` near 0 (≪1) suggests under-flow; "
        "`max|R|` ≫ 1e10 is an over-flow regime. `focus@10%` is the share "
        "of total |R| concentrated in the top-10% of pixels — uniform = 0.10, "
        "well-localised = 0.5–0.9. `register-leak` is the fraction of total "
        "|R| living on the 5 cls/register tokens at the deepest recorded "
        "tap (Darcet et al. 2023, arXiv:2309.16588 — register tokens absorb "
        "high-norm artifacts and attach to no input pixel)."
    )
    lines.append("")
    lines.append("## Per-remedy notes")
    lines.append("")
    for r in summary_rows:
        lines.append(f"### `{r['remedy']}`")
        sub = raw_dump[r["remedy"]]
        if not sub:
            lines.append("No samples ran.")
            continue
        s0 = sub[0]
        if s0.get("error"):
            lines.append(f"**Errored:** `{s0['error']}`")
            lines.append("")
            continue
        lines.append(
            f"- {r['finite_n']} samples produce a finite heatmap; "
            f"median max\\|R\\| = {_fmt_num(r['max_abs_median'])}, "
            f"focus = {_fmt_num(r['focus_top10pct_median'])}, "
            f"register-leak = {_fmt_num(r['register_leak_median'])}."
        )
        if r["first_div_layer"]:
            lines.append(
                f"- First per-layer divergence (max\\|R\\| > 1e10) at "
                f"**{r['first_div_layer']}**."
            )
        # Print first sample's per-layer trajectory (deepest 4 + shallowest 4).
        per_layer = s0["per_layer"]
        if per_layer:
            lines.append("")
            lines.append(
                "Per-layer max\\|R\\| trajectory (sample 0, in backward order — "
                "first row is the deepest tap = first to receive relevance):"
            )
            lines.append("")
            lines.append("| layer | shape | max\\|R\\| | mean\\|R\\| | nan |")
            lines.append("|-------|-------|---------:|----------:|----:|")
            n = len(per_layer)
            sel = sorted(set([0, 1, 2, 3, n - 4, n - 3, n - 2, n - 1]))
            sel = [i for i in sel if 0 <= i < n]
            for i in sel:
                h = per_layer[i]
                lines.append(
                    f"| `{h['layer_name']}` | {tuple(h['shape'])} | "
                    f"{_fmt_num(h['max_abs_finite'])} | "
                    f"{_fmt_num(h['mean_abs_finite'])} | "
                    f"{h['nan_count']} |"
                )
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "Raw per-layer dumps in `diagnostic_raw.json`. Each remedy has its "
        "own subfolder under this directory; subfolders for non-working "
        "remedies contain a `FINDINGS.md` only, working ones additionally "
        "contain a `walkthrough.ipynb` demo."
    )
    return "\n".join(lines) + "\n"


def _fmt_num(x):
    if x is None:
        return "—"
    if _isnan(x):
        return "NaN"
    if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
