"""Is the head-output site a distinct measurement site, or a copy of the value site?

A head detector can be measured at two places inside the attention block: the
value rows it owns (``value``), and its slice of the merged per-head context
before the output projection (``head_out``). Zeroing a head's slice of V and
zeroing its slice of Z are the same intervention — the context of head h is a
convex combination of that head's value rows alone, so deleting those rows
deletes exactly that slice of Z and nothing else. The two sites should
therefore produce the same ablation curve, and the question is whether the
*ranking* read off them differs, since relevance is read at different points of
the backward pass.

This script answers both from the stored results, with no model loaded:

    ablation   greedy DAPC and greedy ordering, head_out vs value
    ranking    LRP ordering and psi, head_out vs value, per method

If both agree everywhere, head_out carries no information the value site does
not already carry, and reporting it as a separate site would double-count.

Run:  uv run python -m experiments.scripts.headout_equivalence
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[2] / "data" / "results" / "benchmark"
KEYS = ["vit_small_funny_birds", "vit_base_imagenet",
        "vit_dinov3_small_funny_birds", "vit_dinov3_base_imagenet"]
SEARCHES = ("optimal", "optimal_dual", "optimal_dyna")
LRP = ("cp_lrp", "chefer")


def _npz(p):
    return np.load(p, allow_pickle=True) if p.exists() else None


def ablation_rows(key):
    z = _npz(RES / f"cdet_dapc_{key}__head_optimal.npz")
    if z is None:
        return []
    ks, out = set(z.files), []
    for m in SEARCHES:
        dd, oo, n = [], 0, 0
        for b in range(12):
            for j in range(64):
                a, c = (f"{m}__head_out__b{b}__img{j}__dapc",
                        f"{m}__value__b{b}__img{j}__dapc")
                if a in ks and c in ks:
                    dd.append(abs(float(z[a]) - float(z[c])))
                ao, co = (f"{m}__head_out__b{b}__img{j}__order",
                          f"{m}__value__b{b}__img{j}__order")
                if ao in ks and co in ks:
                    n += 1
                    oo += int(np.array_equal(np.asarray(z[ao]), np.asarray(z[co])))
        if dd or n:
            out.append((m, len(dd), max(dd) if dd else float("nan"), oo, n))
    return out


def ranking_rows(key):
    z = _npz(RES / f"cdet_orders_{key}__head.npz")
    if z is None:
        return []
    ks, out = set(z.files), []
    for m in LRP:
        same = tot = 0
        rel = 0.0
        for b in range(12):
            a, c = f"{m}__head_out__b{b}__order", f"{m}__value__b{b}__order"
            if a not in ks or c not in ks:
                continue
            A, C = np.asarray(z[a]), np.asarray(z[c])
            tot += A.shape[0]
            same += int((A == C).all(axis=1).sum())
            PA = np.asarray(z[f"{m}__head_out__b{b}__psi"], float)
            PC = np.asarray(z[f"{m}__value__b{b}__psi"], float)
            sc = np.maximum(np.abs(PA).max(1), 1e-12)
            rel = max(rel, float((np.abs(PA - PC).max(1) / sc).max()))
        if tot:
            out.append((m, same, tot, rel))
    return out


def main():
    print("head_out vs value, head detectors\n")
    any_data = False
    for key in KEYS:
        ab, rk = ablation_rows(key), ranking_rows(key)
        if not ab and not rk:
            print(f"{key}: no head_out results stored\n")
            continue
        any_data = True
        print(key)
        for m, n_d, max_d, n_same, n_ord in ab:
            print(f"  ablation  {m:13s} {n_d:3d} paired DAPC, max |diff| "
                  f"{max_d:.3e}   identical orders {n_same}/{n_ord}")
        for m, same, tot, rel in rk:
            print(f"  ranking   {m:13s} identical orders {same}/{tot}, "
                  f"max relative psi difference {rel:.3e}")
        print()
    if any_data:
        print("An identical ordering means the two sites rank head detectors the\n"
              "same way, so a psi difference that is a positive rescaling changes\n"
              "nothing that is used. Where both ablation and ranking agree, the\n"
              "head_out site is redundant and must not be reported separately.")


if __name__ == "__main__":
    main()
