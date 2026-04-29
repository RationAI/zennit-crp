"""Aggregate ``milestone_a_results.csv`` into a markdown faithfulness table.

Reads the CSV produced by :mod:`run_milestone_a` and emits two tables:

* **AUC table** — per (granularity, composite, γ): ``del_true``, ``del_rand``,
  ``ins_true``, ``ins_rand``, plus the gaps. Used in the PR description.
* **Acceptance table** — same rows, just the verdict
  (`OK` if ``del_true < del_rand`` AND ``ins_true > ins_rand``;
  `del_FAIL`/`ins_FAIL` otherwise).

Usage::

    uv run python experiments/aggregate_milestone_a.py \\
        --in data/milestone_a_results.csv \\
        --out data/milestone_a_table.md
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


GRANULARITIES = ("head", "head_dim", "kqv_head", "kqv_head_dim")
COMPOSITE_LABEL = {
    ("AttnLRPEpsilonComposite", None): "ε-LRP",
}


def _composite_label(comp: str, gamma) -> str:
    if comp == "AttnLRPEpsilonComposite":
        return "ε-LRP"
    g = float(gamma) if gamma not in (None, "") else None
    return f"γ={g:g}"


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["concept_def"], r["composite"], r.get("gamma") or None)
        grouped[key].append(r)

    out: list[dict] = []
    for (cd, comp, gamma), group in grouped.items():
        true_rows = [r for r in group if r["mode"] == "true"]
        rand_rows = [r for r in group if r["mode"] == "random"]
        if not true_rows or not rand_rows:
            continue
        d_t = mean(float(r["deletion_auc"]) for r in true_rows)
        d_r = mean(float(r["deletion_auc"]) for r in rand_rows)
        i_t = mean(float(r["insertion_auc"]) for r in true_rows)
        i_r = mean(float(r["insertion_auc"]) for r in rand_rows)
        del_ok = d_t < d_r
        ins_ok = i_t > i_r
        out.append({
            "granularity": cd,
            "composite": comp,
            "gamma": gamma,
            "label": _composite_label(comp, gamma),
            "n_images": len(true_rows),
            "del_true": d_t,
            "del_rand": d_r,
            "del_gap": d_r - d_t,
            "ins_true": i_t,
            "ins_rand": i_r,
            "ins_gap": i_t - i_r,
            "verdict": (
                "OK" if del_ok and ins_ok
                else ("del_FAIL" if not del_ok else "ins_FAIL")
            ),
        })
    return out


def order_columns(agg: list[dict]) -> tuple[list[str], list[str]]:
    """Stable, paper-friendly column order: ε-LRP first, then γ ascending."""
    rule_keys = sorted({r["label"] for r in agg}, key=lambda s: (s != "ε-LRP", s))
    return GRANULARITIES, rule_keys


def emit_markdown(agg: list[dict]) -> str:
    granularities, rule_labels = order_columns(agg)

    by = {(r["granularity"], r["label"]): r for r in agg}

    lines = []
    lines.append("## Milestone A — faithfulness AUC sweep on `vit_base_patch16_224`")
    lines.append("")
    if agg:
        lines.append(
            f"_Sample: {agg[0]['n_images']} images per (granularity, rule), "
            "block 6, deletion/insertion steps = 14, per-granularity top-k = "
            "{head: 4, kqv: 1, kqv_head: 8, head_dim: 8}._"
        )
        lines.append("")

    # Deletion AUC table
    lines.append("### Deletion AUC — `true` (lower better) vs `random`, gap = rand − true")
    lines.append("")
    header = ["granularity"] + [f"{r}: true / rand / gap" for r in rule_labels]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for g in granularities:
        cells = [g]
        for rl in rule_labels:
            r = by.get((g, rl))
            if not r:
                cells.append("—")
            else:
                cells.append(
                    f"{r['del_true']:.4f} / {r['del_rand']:.4f} / "
                    f"{r['del_gap']:+.4f}"
                )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Insertion AUC table
    lines.append("### Insertion AUC — `true` (higher better) vs `random`, gap = true − rand")
    lines.append("")
    header = ["granularity"] + [f"{r}: true / rand / gap" for r in rule_labels]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for g in granularities:
        cells = [g]
        for rl in rule_labels:
            r = by.get((g, rl))
            if not r:
                cells.append("—")
            else:
                cells.append(
                    f"{r['ins_true']:.4f} / {r['ins_rand']:.4f} / "
                    f"{r['ins_gap']:+.4f}"
                )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Verdict
    lines.append("### Verdict — `del(true) < del(rand)` AND `ins(true) > ins(rand)`")
    lines.append("")
    header = ["granularity"] + rule_labels
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for g in granularities:
        cells = [g]
        for rl in rule_labels:
            r = by.get((g, rl))
            cells.append(r["verdict"] if r else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Aggregate verdict per rule
    lines.append("### Acceptance per rule (across all four granularities)")
    lines.append("")
    lines.append("| rule | passing granularities | failures |")
    lines.append("|---|---|---|")
    for rl in rule_labels:
        passing = [g for g in granularities if (g, rl) in by and by[(g, rl)]["verdict"] == "OK"]
        failing = [
            f"{g} ({by[(g, rl)]['verdict']})"
            for g in granularities
            if (g, rl) in by and by[(g, rl)]["verdict"] != "OK"
        ]
        lines.append(
            f"| {rl} | {len(passing)}/{len(granularities)} "
            f"({', '.join(passing) or '—'}) | {', '.join(failing) or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=data_dir / "milestone_a_results.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=data_dir / "milestone_a_table.md",
    )
    args = p.parse_args()

    with args.in_path.open() as f:
        rows = list(csv.DictReader(f))
    agg = aggregate(rows)
    md = emit_markdown(agg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"wrote {args.out}")
    print()
    print(md)


if __name__ == "__main__":
    main()
