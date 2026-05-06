"""Milestone G driver — multi-model sweep with residual-LRP.

Adapted from ``run_milestone_d.py``. Adds the ``residual_lrp`` toggle on
both composites; sweeps ``residual_lrp ∈ {None, 'ratio'}`` (we drop
``'symmetric'`` after empirically verifying that, like PA-LRP, it halves
the heatmap by a uniform factor and is AUC-inert: Pearson 1.0 vs the
baseline at every pixel).

Goal: test whether the Otsuki-style ratio split — distributing relevance
at each ``x = x + branch(x)`` proportionally to ``|x|`` and ``|branch|`` —
fixes the milestone-A `kqv_head` AUC anomaly that PA-LRP alone could not
address.

Output: ``data/milestone_g_results.csv``. Schema: same as milestone D plus a
``residual_lrp`` column.

Run::

    uv run python tutorials/vit_crp/run_milestone_g.py
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import timm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crp.attribution import CondAttribution
from crp.transformer_patches import AttnLRPEpsilonComposite
from datasets import load as load_dataset  # noqa: E402
from metrics import (  # noqa: E402
    PER_GRANULARITY_TOP_K,
    run_one_config,
)
from run_milestone_a import _resolve_dataset_kwargs  # noqa: E402
from run_milestone_d import MID_LAYER  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data_dir = Path(__file__).resolve().parents[1] / "data"
    p.add_argument(
        "--dataset",
        choices=("imagenette", "imagenet_val_hf"),
        default="imagenette",
    )
    p.add_argument(
        "--n-per-class",
        type=int,
        default=None,
        help="Default 16 for imagenette, 1 for imagenet_val_hf.",
    )
    p.add_argument(
        "--classes",
        default="",
        help="comma-separated ImageNet-1k class indices to restrict to.",
    )
    p.add_argument("--steps", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--models",
        default="vit_small_patch16_224,vit_base_patch16_224,vit_large_patch16_224",
        help="comma-separated timm model names",
    )
    p.add_argument(
        "--rules",
        default="none,ratio",
        help="comma-separated residual_lrp rules ∈ {none, symmetric, ratio}",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=data_dir / "milestone_g_results.csv",
    )
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    print(f"loading dataset: {args.dataset}")
    ds_kwargs = _resolve_dataset_kwargs(args)
    dataset = load_dataset(args.dataset, **ds_kwargs)
    image_class_pairs = list(dataset.items)
    print(
        f"  {dataset.name}: {len(image_class_pairs)} images from "
        f"{dataset.num_classes} class(es) "
        f"(n_per_class={ds_kwargs['n_per_class']})"
    )

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    rule_args = [None if r == "none" else r for r in rules]

    all_rows: list[dict] = []
    for model_name in (m.strip() for m in args.models.split(",") if m.strip()):
        print(f"\nloading {model_name}")
        model = timm.create_model(model_name, pretrained=True).eval().to(device)
        layer_idx = MID_LAYER.get(model_name, len(model.blocks) // 2)
        attn = model.blocks[layer_idx].attn
        num_heads, head_dim = attn.num_heads, attn.head_dim
        print(
            f"  block={layer_idx}/{len(model.blocks)}  num_heads={num_heads}  "
            f"head_dim={head_dim}"
        )
        attribution = CondAttribution(model, device=torch.device(device))

        for rule in rule_args:
            rng = random.Random(args.seed)
            composite = AttnLRPEpsilonComposite(residual_lrp=rule)
            label = "AttnLRPEps" + (f"+res={rule}" if rule else "")
            print(f"\n────── {model_name} | {label} ──────")
            rows = run_one_config(
                model=model,
                attribution=attribution,
                composite=composite,
                composite_label=label,
                gamma_label=None,
                image_class_pairs=image_class_pairs,
                layer_idx=layer_idx,
                num_heads=num_heads,
                head_dim=head_dim,
                top_k=PER_GRANULARITY_TOP_K,
                n_steps=args.steps,
                rng=rng,
                device=device,
            )
            for r in rows:
                r["model"] = model_name
                r["residual_lrp"] = rule or "none"
                r["layer"] = layer_idx
            all_rows.extend(rows)

        del model, attribution
        if device == "cuda":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {args.out} ({len(all_rows)} rows)")

    summarise(all_rows)


def summarise(rows: list[dict]) -> None:
    """Per-(model, rule, granularity) verdict against the all-four ordering."""
    keys = sorted({(r["model"], r["residual_lrp"], r["concept_def"]) for r in rows})
    print()
    print("=" * 115)
    print(
        f"{'model':28} {'rule':>10} {'granularity':>10}   "
        f"{'del_true':>9} {'del_rand':>9} {'del_gap':>9}   "
        f"{'ins_true':>9} {'ins_rand':>9} {'ins_gap':>9}   verdict"
    )
    print("=" * 115)
    for model_name, rule, cd in keys:
        true_rows = [
            r for r in rows
            if r["model"] == model_name and r["residual_lrp"] == rule
            and r["concept_def"] == cd and r["mode"] == "true"
        ]
        rand_rows = [
            r for r in rows
            if r["model"] == model_name and r["residual_lrp"] == rule
            and r["concept_def"] == cd and r["mode"] == "random"
        ]
        if not true_rows:
            continue
        d_t = sum(r["deletion_auc"] for r in true_rows) / len(true_rows)
        d_r = sum(r["deletion_auc"] for r in rand_rows) / len(rand_rows)
        i_t = sum(r["insertion_auc"] for r in true_rows) / len(true_rows)
        i_r = sum(r["insertion_auc"] for r in rand_rows) / len(rand_rows)
        del_ok = d_t < d_r
        ins_ok = i_t > i_r
        verdict = "OK" if del_ok and ins_ok else (
            "del_FAIL" if not del_ok else "ins_FAIL"
        )
        print(
            f"{model_name:28} {rule:>10} {cd:>10}   "
            f"{d_t:>9.4f} {d_r:>9.4f} {d_r - d_t:>+9.4f}   "
            f"{i_t:>9.4f} {i_r:>9.4f} {i_t - i_r:>+9.4f}   {verdict}"
        )
    print("=" * 115)


if __name__ == "__main__":
    main()
