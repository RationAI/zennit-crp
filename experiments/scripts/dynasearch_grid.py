"""Tighten the attainable-DAPC bound across the benchmark grid with dynasearch.

The bound used to normalise every reported DAPC is the best ordering any search
found, and so far that is the plain greedy of
:mod:`experiments.concept_detector_optimal`, which optimises the MoRF curve
alone and is not even a 1-opt local optimum for DAPC. Dynasearch
(:mod:`experiments.dynasearch_order`) improves an existing ordering on the
adjacent-transposition neighbourhood, scoring the real objective, so it can be
**seeded from the stored greedy order** — no greedy re-run, and the cost is
``iters * 2(D-1)`` forward passes per (cell, image) instead of ``D(D+1)/2``.

Every combo the greedy side-car already holds an order for is eligible; the
scope is restricted to the attention-internal sites, which are what the paper
claims about. Results are committed to the same side-car under the method name
``optimal_dyna``, so :mod:`experiments.rank_analysis` picks them up with no
change (``SEARCH_METHODS`` takes the max over whatever is present).

Run:  uv run python -m experiments.scripts.dynasearch_grid --model-key <key>
          [--concept embed_dim|head] [--sites value qk ...] [--max-iters 40]
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import torch

from experiments.concept_detector_optimal import (
    DEVICE, OptimalStore, ZeroChannelsHook, capture_block_inputs, dapc_of,
    layer_name, load, make_cached_forward, prob_curve)
from experiments.dynasearch_order import dynasearch
from experiments.rank_analysis import ATTENTION_SITES

SEED_METHOD = "optimal"          # the greedy orders already on disk
DYNA_METHOD = "optimal_dyna"


def seed_combos(sto: OptimalStore, sites):
    """(site, block, image) combos that have a stored greedy order."""
    out = []
    for k in sto.store:
        m = re.match(rf"{SEED_METHOD}__(.+?)__b(\d+)__img(\d+)__order$", k)
        if m and m.group(1) in sites:
            out.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--concept", choices=["embed_dim", "head"], default="embed_dim")
    ap.add_argument("--sites", nargs="*", default=list(ATTENTION_SITES))
    ap.add_argument("--max-iters", type=int, default=40)
    a = ap.parse_args()

    sto = OptimalStore(a.model_key, a.concept)
    combos = seed_combos(sto, set(a.sites))
    if not combos:
        raise SystemExit(f"no stored {SEED_METHOD} orders for sites={a.sites} "
                         f"in {sto.path.name}")
    todo = [c for c in combos
            if not sto.has(f"{DYNA_METHOD}__{c[0]}__b{c[1]}__img{c[2]}__dapc")]
    print(f"[{a.model_key}/{a.concept}] {len(combos)} seeded combos, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        return

    model, ds, normalize, D, picks, handles, head_dim = load(a.model_key, a.concept)
    caches, gains = {}, []
    try:
        for n, (site, b, j) in enumerate(todo, 1):
            t0 = time.time()
            idx, pred, _ = picks[j]
            xn = normalize(ds[idx][0].unsqueeze(0)).to(DEVICE)
            if j not in caches:
                caches[j] = capture_block_inputs(model, xn)
            hook = ZeroChannelsHook(head_dim=head_dim)
            hh = model.get_submodule(layer_name(site, b)).register_forward_hook(hook)
            try:
                fwd = make_cached_forward(model, caches[j], b)
                seed = np.asarray(
                    sto.get(f"{SEED_METHOD}__{site}__b{b}__img{j}__order"), np.int64)
                order, dapc, _log = dynasearch(model, xn, pred, hook, D, seed,
                                               forward_fn=fwd, max_iters=a.max_iters)
                ot = torch.from_numpy(np.asarray(order, np.int64)).long()
                morf = prob_curve(model, xn, pred, ot, hook, D, fwd)
                lerf = prob_curve(model, xn, pred, ot.flip(0), hook, D, fwd)
            finally:
                hh.remove()
            chk = dapc_of(morf, lerf)
            if abs(chk - dapc) > 1e-4:
                raise SystemExit(f"dynasearch DAPC {dapc:+.6f} disagrees with the "
                                 f"recomputed curves {chk:+.6f} at {site} b{b} img{j}")
            # a combo can hold a seed order without its DAPC: the side-car commits
            # the expensive order first, so a mid-combo crash leaves that gap.
            bk = f"{SEED_METHOD}__{site}__b{b}__img{j}__dapc"
            base = float(sto.get(bk)) if sto.has(bk) else float("nan")
            if np.isfinite(base):
                gains.append(chk - base)
            pre = f"{DYNA_METHOD}__{site}__b{b}__img{j}"
            sto.commit(pre + "__order", np.asarray(order, np.int64), flush=False)
            sto.commit(pre + "__morf", morf, flush=False)
            sto.commit(pre + "__lerf", lerf, flush=False)
            sto.commit(pre + "__dapc", np.float32(chk), flush=False)
            if n % 16 == 0 or n == len(todo):
                sto.flush()       # one network-storage rewrite per batch
            print(f"[{a.model_key}] {site} b{b} img#{j}: greedy={base:+.4f} -> "
                  f"dyna={chk:+.4f} ({chk-base:+.4f}) {time.time()-t0:.1f}s "
                  f"[{n}/{len(todo)}]", flush=True)
    finally:
        for h in handles:
            h.remove()

    # layer aggregates, bench-shaped, over the images present for each cell
    cells = sorted({(s, b) for s, b, _ in combos})
    for site, b in cells:
        rows = [(j, f"{DYNA_METHOD}__{site}__b{b}__img{j}")
                for _s, _b, j in combos if (_s, _b) == (site, b)]
        rows = [(j, p) for j, p in rows if sto.has(p + "__dapc")]
        if not rows:
            continue
        rows.sort()
        sto.commit(f"{DYNA_METHOD}__{site}__b{b}__morf",
                   np.stack([sto.get(p + "__morf") for _j, p in rows]), flush=False)
        sto.commit(f"{DYNA_METHOD}__{site}__b{b}__lerf",
                   np.stack([sto.get(p + "__lerf") for _j, p in rows]), flush=False)
        sto.commit(f"{DYNA_METHOD}__{site}__b{b}__dapc",
                   np.array([float(sto.get(p + "__dapc")) for _j, p in rows], np.float32),
                   flush=False)
    sto.flush()
    g = np.array(gains)
    if g.size:
        print(f"[{a.model_key}/{a.concept}] done: {len(cells)} cells; gain over "
              f"greedy on {g.size} combos: median={np.median(g):+.4f} "
              f"mean={g.mean():+.4f} max={g.max():+.4f} min={g.min():+.4f}", flush=True)
    else:
        print(f"[{a.model_key}/{a.concept}] done: {len(cells)} cells; no stored "
              "greedy DAPC to compare against", flush=True)


if __name__ == "__main__":
    main()
