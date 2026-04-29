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
``tutorials/vit_crp/data/curated_milestone_a/<class_idx>/<image>``). If it
does not exist, this script builds it from the Imagenette ``val/`` split,
which the walkthrough notebook downloads to
``tutorials/vit_crp/data/imagenette2-160/``.

Outputs
-------

* ``--out`` (default ``tutorials/vit_crp/data/milestone_a_results.csv``):
  one row per ``(composite, gamma, image, target_class, concept_def, mode)``.
* A summary table printed to stdout: per-(granularity, composite, γ)
  mean deletion / insertion AUC for ``true`` and ``random`` with the gap.

Run::

    uv run python tutorials/vit_crp/run_milestone_a.py
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import timm
from PIL import Image  # noqa: F401  (used transitively via metrics)

# Allow `import metrics` regardless of CWD (tutorials/ is not a package).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crp.attribution import CondAttribution
from metrics import (  # noqa: E402
    PER_GRANULARITY_TOP_K,
    build_composite,
    iter_image_classes,
    run_one_config,
)


# ── Imagenette WordNet → ImageNet-1k class index (subset of notebook map) ────


# Picked to be visually + semantically distinct and to cover enough img/class.
CURATED_CLASSES = {
    217: "n02102040",  # English springer
    482: "n02979186",  # cassette player
    569: "n03417042",  # garbage truck
    701: "n03888257",  # parachute
}


def build_curated_subset(
    imagenette_root: Path, dest: Path, n_per_class: int = 16, seed: int = 0
) -> None:
    """Symlink ``n_per_class`` images per ``CURATED_CLASSES`` from imagenette
    val/ into ``dest/<class_idx>/``. Idempotent."""
    val = imagenette_root / "val"
    if not val.is_dir():
        raise SystemExit(
            f"imagenette val/ not found at {val}. "
            "Run the walkthrough notebook section 2 to download it, or "
            "extract imagenette2-160.tgz under tutorials/vit_crp/data/."
        )
    rng = random.Random(seed)
    dest.mkdir(parents=True, exist_ok=True)
    for cls_idx, wnid in CURATED_CLASSES.items():
        cls_dir = dest / str(cls_idx)
        cls_dir.mkdir(exist_ok=True)
        existing = list(cls_dir.glob("*.JPEG"))
        if len(existing) >= n_per_class:
            print(f"  class {cls_idx} ({wnid}): {len(existing)} symlinks already present")
            continue
        src = val / wnid
        if not src.is_dir():
            raise SystemExit(f"missing imagenette class dir: {src}")
        candidates = sorted(src.glob("*.JPEG"))
        chosen = rng.sample(candidates, k=min(n_per_class, len(candidates)))
        for p in chosen:
            link = cls_dir / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
        print(f"  class {cls_idx} ({wnid}): {len(list(cls_dir.glob('*.JPEG')))} images")


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
    here = Path(__file__).resolve().parent
    p.add_argument(
        "--imagenette-root",
        type=Path,
        default=here / "data" / "imagenette2-160",
        help="extracted imagenette2-160 directory (containing train/ + val/)",
    )
    p.add_argument(
        "--curated-dir",
        type=Path,
        default=here / "data" / "curated_milestone_a",
        help="class-keyed dir of evaluation images",
    )
    p.add_argument("--n-per-class", type=int, default=16)
    p.add_argument("--block", type=int, default=6)
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
        default=here / "data" / "milestone_a_results.csv",
    )
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # 1. Build curated subset (idempotent).
    print("building curated subset")
    build_curated_subset(
        args.imagenette_root, args.curated_dir, args.n_per_class, args.seed
    )
    image_class_pairs = iter_image_classes(args.curated_dir)
    print(
        f"  curated: {len(image_class_pairs)} images from "
        f"{len(set(c for _, c in image_class_pairs))} class(es)"
    )

    # 2. Model + attribution (re-used across composites).
    print(f"loading {args.model}")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    attn = model.blocks[args.block].attn
    num_heads, head_dim = attn.num_heads, attn.head_dim
    layer_name = f"blocks.{args.block}.attn.qkv_tap"
    print(f"  layer={layer_name}  num_heads={num_heads}  head_dim={head_dim}")
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
            layer_name=layer_name,
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
