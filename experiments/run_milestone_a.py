"""Milestone A driver — γ-LRP sweep on faithfulness AUC.

Per ``FUTURE_STATE.md`` Milestone A:

* Re-run the deletion / insertion AUC benchmark on ``vit_base_patch16_224``
  across the four attention-concept granularities, sweeping the LRP
  composite over ε-LRP and γ-LRP at γ ∈ {0.0, 0.1, 0.25, 0.5}.
* On a non-trivial sample (≥64 images, ≥4 distinct ImageNet classes).
* Verify ``deletion_AUC(true) < deletion_AUC(random)`` for **all four**
  granularities under the chosen composite + γ.

Inputs
------

A class-keyed image directory at ``--curated-dir`` (default
``data/curated_milestone_a/<class_idx>/<image>`` under the repo root).
If it does not exist, this script builds it from the Imagenette ``val/``
split that the walkthrough notebook downloads to ``data/imagenette2-160/``.

Outputs
-------

* ``--out`` (default ``data/milestone_a_results.csv``): one row per
  ``(composite, gamma, image, target_class, concept_def, mode)``.
* A summary table printed to stdout: per-(granularity, composite, γ)
  mean deletion / insertion AUC for ``true`` and ``random`` with the gap.

Run::

    uv run python experiments/run_milestone_a.py
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import timm

# Allow sibling-module imports regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crp.attribution import CondAttribution
from datasets import load as load_dataset  # noqa: E402
from metrics import (  # noqa: E402
    PER_GRANULARITY_TOP_K,
    build_composite,
    run_one_config,
)


# ── default class subsets per dataset ─────────────────────────────────────────


# Imagenette has 10 classes; we pick a 4-class subset that's visually and
# semantically distinct so even short sweeps see meaningful variance.
IMAGENETTE_DEFAULT_CLASSES = (
    217,  # English springer
    482,  # cassette player
    569,  # garbage truck
    701,  # parachute
)


def _resolve_dataset_kwargs(args) -> dict:
    """Build kwargs for ``datasets.load`` from CLI args, applying sensible
    per-dataset defaults when the user hasn't overridden them."""
    classes = (
        [int(c) for c in args.classes.split(",") if c]
        if args.classes
        else (list(IMAGENETTE_DEFAULT_CLASSES) if args.dataset == "imagenette" else None)
    )
    n_per_class = args.n_per_class
    if n_per_class is None:
        # Default: 16 imgs/class for imagenette (4 classes × 16 = 64), and
        # 1 img/class for the full-imagenet sweep (~1000-img class-balanced
        # set). Override via --n-per-class.
        n_per_class = 16 if args.dataset == "imagenette" else 1
    return dict(classes=classes, n_per_class=n_per_class, seed=args.seed)


# ── summary table ─────────────────────────────────────────────────────────────


def summarise(rows: list[dict]) -> None:
    """Per-(granularity, composite, γ) summary; flag failures of the
    Milestone A acceptance criterion (true < random on deletion AUC)."""
    keys = sorted({(r["concept_def"], r["composite"], r["gamma"]) for r in rows})
    print("\n" + "=" * 100)
    print(f"{'concept_def':>10} {'composite':>22} {'gamma':>6}   "
          f"{'del_true':>9} {'del_rand':>9} {'del_gap':>9}   "
          f"{'ins_true':>9} {'ins_rand':>9} {'ins_gap':>9}   verdict")
    print("=" * 100)
    failures: list[tuple] = []
    for k in keys:
        cd, comp, gamma = k
        true_rows = [
            r for r in rows
            if r["concept_def"] == cd and r["composite"] == comp
            and r["gamma"] == gamma and r["mode"] == "true"
        ]
        rand_rows = [
            r for r in rows
            if r["concept_def"] == cd and r["composite"] == comp
            and r["gamma"] == gamma and r["mode"] == "random"
        ]
        if not true_rows or not rand_rows:
            continue
        d_true = sum(r["deletion_auc"] for r in true_rows) / len(true_rows)
        d_rand = sum(r["deletion_auc"] for r in rand_rows) / len(rand_rows)
        i_true = sum(r["insertion_auc"] for r in true_rows) / len(true_rows)
        i_rand = sum(r["insertion_auc"] for r in rand_rows) / len(rand_rows)
        del_ok = d_true < d_rand
        ins_ok = i_true > i_rand
        verdict = "OK" if del_ok and ins_ok else (
            "del_FAIL" if not del_ok else "ins_FAIL"
        )
        print(
            f"{cd:>10} {comp:>22} {str(gamma):>6}   "
            f"{d_true:>9.4f} {d_rand:>9.4f} {d_rand - d_true:>+9.4f}   "
            f"{i_true:>9.4f} {i_rand:>9.4f} {i_true - i_rand:>+9.4f}   {verdict}"
        )
        if not (del_ok and ins_ok):
            failures.append(k)
    print("=" * 100)
    if failures:
        print(f"Acceptance failures ({len(failures)}):")
        for cd, comp, gamma in failures:
            print(f"  {cd:>10} {comp:>22} γ={gamma}")
    else:
        print("ALL configs satisfy del(true) < del(random) AND ins(true) > ins(random).")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data_dir = Path(__file__).resolve().parents[1] / "data"
    p.add_argument(
        "--dataset",
        choices=("imagenette", "imagenet_val_hf"),
        default="imagenette",
        help="evaluation set source. ``imagenette`` (default) is fast and "
             "auto-downloads (~98 MB); ``imagenet_val_hf`` is the un-gated "
             "HuggingFace mirror of the full ImageNet-1k val (~830 MB).",
    )
    p.add_argument(
        "--n-per-class",
        type=int,
        default=None,
        help="images per class. Default 16 for imagenette (4 classes × 16 = "
             "64 imgs), 1 for imagenet_val_hf (1000 classes × 1 = 1000 imgs).",
    )
    p.add_argument(
        "--classes",
        default="",
        help="comma-separated ImageNet-1k class indices to restrict to. "
             "Default: a 4-class Imagenette subset for --dataset=imagenette, "
             "all classes for imagenet_val_hf.",
    )
    p.add_argument("--layer", type=int, default=6)
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="single int top-k for every granularity (clamped to concept count). "
             "Default: per-granularity sensible defaults from PER_GRANULARITY_TOP_K.",
    )
    p.add_argument("--steps", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument(
        "--gammas",
        default="0.0,0.1,0.25,0.5",
        help="comma-separated γ values for AttnLRPGammaComposite",
    )
    p.add_argument(
        "--include-epsilon",
        action="store_true",
        default=True,
        help="also evaluate AttnLRPEpsilonComposite as a sanity-check baseline",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=data_dir / "milestone_a_results.csv",
    )
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # 1. Resolve evaluation set.
    print(f"loading dataset: {args.dataset}")
    ds_kwargs = _resolve_dataset_kwargs(args)
    dataset = load_dataset(args.dataset, **ds_kwargs)
    image_class_pairs = list(dataset.items)
    print(
        f"  {dataset.name}: {len(image_class_pairs)} images from "
        f"{dataset.num_classes} class(es) "
        f"(n_per_class={ds_kwargs['n_per_class']})"
    )

    # 2. Model + attribution (re-used across composites).
    print(f"loading {args.model}")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    attn = model.blocks[args.layer].attn
    num_heads, head_dim = attn.num_heads, attn.head_dim
    print(f"  layer={args.layer}  num_heads={num_heads}  head_dim={head_dim}")
    attribution = CondAttribution(model, device=torch.device(device))

    # 3. Build sweep configs.
    configs: list[tuple[str, float | None]] = []
    if args.include_epsilon:
        configs.append(("epsilon", None))
    for g in (float(x) for x in args.gammas.split(",")):
        configs.append(("gamma", g))

    top_k_arg: int | dict[str, int] = args.top_k if args.top_k is not None else PER_GRANULARITY_TOP_K
    print(f"sweeping {len(configs)} configs: {configs}")
    print(f"top_k per granularity: {top_k_arg}")

    all_rows: list[dict] = []
    for comp_name, gamma in configs:
        rng = random.Random(args.seed)  # reset → identical random concepts across configs
        composite = build_composite(
            comp_name, gamma if gamma is not None else 0.0, epsilon=1e-6
        )
        composite_label = type(composite).__name__
        print(
            f"\n────── {composite_label}  γ={gamma}  "
            f"({len(image_class_pairs)} images × {4} granularities) ──────"
        )
        rows = run_one_config(
            model=model,
            attribution=attribution,
            composite=composite,
            composite_label=composite_label,
            gamma_label=gamma,
            image_class_pairs=image_class_pairs,
            layer_idx=args.layer,
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k_arg,
            n_steps=args.steps,
            rng=rng,
            device=device,
        )
        all_rows.extend(rows)

    # 4. Write combined CSV.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {args.out} ({len(all_rows)} rows)")

    # 5. Summary.
    summarise(all_rows)


if __name__ == "__main__":
    main()
