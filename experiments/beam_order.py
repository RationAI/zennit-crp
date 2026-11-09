"""Beam search over component orderings, the estimator of Edin et al. (ACL 2025).

Our score is two-sided: DAPC(pi) = mean(LeRF) - mean(MoRF), where MoRF[k] is
the readout after removing the first k components of pi and LeRF[k] after
removing the last k. Because LeRF(pi) is the MoRF curve of the reversed
ordering,

    DAPC(pi) = mean(MoRF(reverse pi)) - mean(MoRF(pi)),

so an ordering with a low MoRF area is a good candidate on both terms at once:
removing its head destroys the prediction fast, and removing its tail (which is
the head of the reverse) destroys it slowly.

The beam therefore builds the ordering left to right minimising the running sum
of MoRF points, which is the one-sided objective Edin et al. search, and the
two-sided score is computed once per surviving beam at the end. Width B costs
about B * n^2 / 2 forward passes; every candidate extension at a beam level
goes through in one batch.

Caveat recorded deliberately: the exact beam width, tie-breaking and partial
score of the published method were not read from the paper here. This is a
faithful implementation of the described procedure, not a reproduction of their
configuration.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.concept_detector_optimal import dapc_of, prob_curve, tau


def beam_order(model, xn, pred, hook, D, *, width=5, forward_fn=None,
               verbose=False):
    """Return (order, dapc, morf_sum_of_best) for the best beam member."""
    # beam: list of (running MoRF sum, removed bool mask, order list)
    f0 = float(tau(model, xn, pred, torch.ones(1, D), hook, D, forward_fn)[0])
    beams = [(f0, np.zeros(D, dtype=bool), [])]

    for step in range(D):
        cand_rows, meta = [], []
        for bi, (ssum, removed, order) in enumerate(beams):
            free = np.flatnonzero(~removed)
            if free.size == 0:
                continue
            keep = np.ones((free.size, D), dtype=np.float32)
            keep[:, removed] = 0.0
            keep[np.arange(free.size), free] = 0.0      # also remove the candidate
            cand_rows.append(torch.from_numpy(keep))
            meta.extend((bi, int(c)) for c in free)
        if not cand_rows:
            break
        vals = np.asarray(tau(model, xn, pred, torch.cat(cand_rows), hook, D,
                              forward_fn))
        scored = [(beams[bi][0] + float(v), bi, c) for v, (bi, c) in zip(vals, meta)]
        scored.sort(key=lambda t: t[0])                 # minimise retained mass
        nxt, seen = [], set()
        for ssum, bi, c in scored:
            _, removed, order = beams[bi]
            key = (tuple(sorted(order + [c])),)         # dedup identical prefix sets
            if key in seen:
                continue
            seen.add(key)
            rm = removed.copy(); rm[c] = True
            nxt.append((ssum, rm, order + [c]))
            if len(nxt) == width:
                break
        beams = nxt
        if verbose and (step + 1) % 64 == 0:
            print(f"      beam step {step+1}/{D}", flush=True)

    best = None
    for ssum, _, order in beams:
        o = np.asarray(order, dtype=np.int64)
        m = prob_curve(model, xn, pred, o, hook, D, forward_fn)
        l = prob_curve(model, xn, pred, o[::-1].copy(), hook, D, forward_fn)
        d = dapc_of(m, l)
        if best is None or d > best[1]:
            best = (o, d, ssum)
    return best


def exhaustive_best(model, xn, pred, hook, D, *, forward_fn=None):
    """Exact optimum over all D! orderings. Only sane for D <= 8."""
    from itertools import permutations
    import math
    assert math.factorial(D) <= 50000, f"{D}! is too many orderings"
    best = None
    for perm in permutations(range(D)):
        o = np.asarray(perm, dtype=np.int64)
        m = prob_curve(model, xn, pred, o, hook, D, forward_fn)
        l = prob_curve(model, xn, pred, o[::-1].copy(), hook, D, forward_fn)
        d = dapc_of(m, l)
        if best is None or d > best[1]:
            best = (o, d)
    return best
