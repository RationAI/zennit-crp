"""Evaluation matrix for AlphaBeta-on-bilinear vs the standard 2Y+ε rule.

See ``RESEARCH_NOTES.md`` Entry 6 for the derivation and motivation.
This script implements the planned evaluation table on real DINOv3
ViT-L/16 + linear probe, substituting all 24 EvaAttention modules with
:class:`crp.attention_unfolded.EvaAttentionUnfolded` configured for
each of the 4 evaluation variants.

Output: a structured markdown table, raw JSON dump, and per-block
relevance trajectory for sample 0 of each variant. The conclusion is
appended to ``RESEARCH_NOTES.md`` Entry 6 as the experimental result.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config, create_transform

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from crp.attribution import CondAttribution  # noqa: E402
from crp.attention_unfolded import (  # noqa: E402
    EvaAttentionUnfolded,
    EvaAttentionSubstitutionCanonizer,
)
from crp.transformer_patches import AttnLRPCombinedComposite  # noqa: E402
from datasets import load as load_dataset  # noqa: E402
from dinov3_diagnose import diagnose_attribution  # noqa: E402


# Evaluation matrix: (label, matmul_rule_kwargs) per RESEARCH_NOTES Entry 6.
VARIANTS = [
    ("baseline_2y_eps",
     dict(matmul_rule="matmul_factor_2", epsilon=1e-6)),
    ("alphabeta_1_0",
     dict(matmul_rule="alpha_beta", alpha=1.0, beta=0.0, epsilon=1e-6)),
    ("alphabeta_2_-1",
     dict(matmul_rule="alpha_beta", alpha=2.0, beta=-1.0, epsilon=1e-6)),
    ("alphabeta_05_05",
     dict(matmul_rule="alpha_beta", alpha=0.5, beta=0.5, epsilon=1e-6)),
]


def build_model(probe_path: Path, device: str):
    ckpt = torch.load(probe_path, map_location=device, weights_only=False)
    model = timm.create_model(
        ckpt["model_name"], pretrained=True, num_classes=ckpt["num_classes"],
    )
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=False)
    return model, transform, ckpt["model_name"]


def pick_class_distinct_samples(model, dataset, n: int, device: str):
    seen = set()
    picks = []
    stride = max(1, len(dataset) // (n * 6))
    for i in range(0, len(dataset), stride):
        x, y = dataset[i]
        if int(y) in seen:
            continue
        x_dev = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_dev).argmax(-1).item()
        if pred == int(y):
            picks.append((x_dev.detach().requires_grad_(True), pred))
            seen.add(int(y))
            if len(picks) == n:
                break
    return picks


def run_variant(model, attribution, samples, variant_label: str, **unfold_kwargs):
    """Substitute all EvaAttention modules with the unfolded variant
    configured for this evaluation cell, then run the full working
    composite."""
    # Substitute every EvaAttention block with the unfolded variant.
    n_blocks = len(model.blocks)
    sub_canonizer = EvaAttentionSubstitutionCanonizer(
        block_indices=tuple(range(n_blocks)),
        **unfold_kwargs,
    )
    # The composite installs all the OTHER canonizers (LayerNorm, GELU,
    # residual rule, layerscale) plus the linear ε-LRP. The substitution
    # canonizer for attention rides alongside.
    composite = AttnLRPCombinedComposite(
        # Don't install the legacy matmul_factor_2 forward here — the
        # substitution canonizer owns the attention rule.
        layerscale_uniform=True,
        residual_lrp="ratio",
    )
    # Inject the substitution canonizer into the composite's canonizer list.
    composite.canonizers = list(composite.canonizers) + [sub_canonizer]

    per_sample = []
    for s_idx, (x, cls) in enumerate(samples):
        x_run = x.detach().clone().requires_grad_(True)
        t0 = time.time()
        report = diagnose_attribution(
            model, attribution, composite, x_run,
            target_class=cls, composite_label=variant_label,
            num_prefix_tokens=5,
        )
        elapsed = time.time() - t0
        per_sample.append({
            "sample": s_idx,
            "class": cls,
            "elapsed_s": round(elapsed, 2),
            "finite_heatmap": report.finite_heatmap,
            "heatmap_max_abs": report.heatmap_max_abs,
            "heatmap_focus_top10pct": report.heatmap_focus_top10pct,
            "register_leak_share": report.register_leak_share,
            "per_layer": [asdict(h) for h in report.per_layer],
        })
        print(f"  {variant_label:25} sample{s_idx} cls{cls}  "
              f"finite={report.finite_heatmap}  "
              f"max|R|={report.heatmap_max_abs:.3e}  "
              f"focus={report.heatmap_focus_top10pct:.3f}  "
              f"({elapsed:.1f}s)")
    return per_sample


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_path = REPO_ROOT / "data/vit_large_patch16_dinov3_probe_imagenette.pt"
    out_dir = REPO_ROOT / "tutorials/vit_crp/dinov3_variants/alphabeta"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    model, transform, model_name = build_model(probe_path, device)
    dataset = load_dataset("imagenette", split="val", transform=transform)
    samples = pick_class_distinct_samples(model, dataset, n=5, device=device)
    print(f"\npicked {len(samples)} samples (classes: {[s[1] for s in samples]})\n")

    attribution = CondAttribution(model)
    raw = {}
    for label, kw in VARIANTS:
        print(f"running {label} ({kw})")
        raw[label] = run_variant(model, attribution, samples, label, **kw)
        print()

    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=str))
    (out_dir / "RESULTS.md").write_text(_format_markdown(raw, model_name, len(samples)))
    print(f"\nwrote {out_dir / 'raw.json'}")
    print(f"wrote {out_dir / 'RESULTS.md'}")


def _median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _isnan(x):
    return isinstance(x, float) and x != x


def _fmt(x):
    if _isnan(x):
        return "NaN"
    if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.3f}"


def _format_markdown(raw, model_name, n_samples):
    lines = []
    lines.append(f"# AlphaBeta-on-bilinear evaluation — `{model_name}`")
    lines.append("")
    lines.append(
        f"Sweep across {len(VARIANTS)} variants of the bilinear matmul "
        f"rule × {n_samples} class-distinct Imagenette samples. See "
        f"`RESEARCH_NOTES.md` Entry 6 for derivation, motivation, "
        f"evaluation rationale, and acceptance criteria."
    )
    lines.append("")
    lines.append("Substitution: all 24 `EvaAttention` modules replaced with "
                 "`EvaAttentionUnfolded` configured for the variant's "
                 "`matmul_rule`. Other LRP rules unchanged "
                 "(`layerscale_uniform=True`, `residual_lrp='ratio'`, "
                 "`Epsilon` on Linears, `Pass` on LayerNorm).")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| variant | finite | median max\\|R\\| | median focus@10% | median |sum(R)| |")
    lines.append("|---|---|---:|---:|---:|")
    for label, _ in VARIANTS:
        sub = raw[label]
        finite_n = sum(1 for s in sub if s["finite_heatmap"])
        max_med = _median([s["heatmap_max_abs"] for s in sub if s["finite_heatmap"]])
        focus_med = _median([s["heatmap_focus_top10pct"] for s in sub if s["finite_heatmap"] and not _isnan(s["heatmap_focus_top10pct"])])
        # |sum(R)| at input: NOT directly captured by report; we'd need
        # to extract from per_layer or grad. The diagnostic gives heatmap
        # max only. Mark as TBD.
        lines.append(
            f"| `{label}` | {finite_n}/{len(sub)} | {_fmt(max_med)} | "
            f"{_fmt(focus_med)} | (see per_layer in raw.json) |"
        )
    lines.append("")
    lines.append("## Per-block max|R| trajectory (sample 0)")
    lines.append("")
    lines.append("Block-by-block magnitude trajectory shows whether the "
                 "AlphaBeta variants control the per-layer amplification "
                 "documented in `RESEARCH_NOTES.md` Entry 4.")
    lines.append("")
    lines.append("| block | " + " | ".join(f"`{l}`" for l, _ in VARIANTS) + " |")
    lines.append("|---|" + "---:|" * len(VARIANTS))
    n_blocks = len(raw[VARIANTS[0][0]][0]["per_layer"])
    # Show every other block from deepest (last in per_layer = blocks.23)
    # to shallowest (first = blocks.0), since backward order matters.
    for i in range(n_blocks - 1, -1, -2):
        row = [f"blocks.{i}"]
        for label, _ in VARIANTS:
            h = raw[label][0]["per_layer"][i]
            row.append(_fmt(h["max_abs_finite"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Raw per-sample / per-layer dump in `raw.json`. Conclusions "
                 "appended to `RESEARCH_NOTES.md` Entry 6.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
