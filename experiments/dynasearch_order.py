"""Iterated dynasearch on the adjacent-transposition neighbourhood, as a
stronger reference ordering than the plain greedy of
``experiments.concept_detector_optimal``.

Why this neighbourhood. DAPC is a *prefix* objective: with
``morf[k]`` = readout after removing the first k detectors of the ordering and
``lerf[k]`` = readout after removing the last k, DAPC = mean(lerf) - mean(morf).
Swapping the ordering's positions j and j+1 therefore changes exactly two
points of the two curves — ``morf[j+1]`` and ``lerf[n-j-1]`` — and leaves every
other point bit-identical, because every other prefix and suffix contains the
same SET of detectors as before. One adjacent swap costs 2 forward passes, and
the whole neighbourhood costs 2(n-1): about 1/n of a single greedy run.

Two swaps at positions j and j' with |j - j'| >= 2 touch disjoint curve points
and neither alters the other's changed set, so their deltas are exactly
additive. Choosing the best compound move is then a maximum-weight independent
set on a path, solved by an O(n) DP. That picks the best of ~1.6^n compound
moves for the price of one neighbourhood scan, with no approximation.

Structure and the additivity argument follow the dynasearch of Congram, Potts
and van de Velde (INFORMS Journal on Computing 14(1), 2002), introduced for
single-machine total weighted tardiness — the same prefix-sum objective family.
The plain greedy is known to be only a tight 4-approximation on this family
(Feige, Lovasz, Tetali, Algorithmica 40(4), 2004), and its tight instances are
exactly early near-ties, which is where ours fails.

Note the plain greedy optimises the MoRF curve alone and never looks at the
LeRF term, so it is not even a 1-opt local optimum for DAPC. Dynasearch scores
the real objective at every step.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.concept_detector_optimal import (
    DEVICE, dapc_of, keep_of, prob_curve, tau,
)


def _changed_sets(order: np.ndarray, D: int):
    """Keep-masks for the two sets each adjacent swap changes.

    Swap at position j (0-based, j = 0..D-2) of ``order`` gives
      new prefix of size j+1 : order[:j] + [order[j+1]]
      new suffix of size D-j-1: order[j+2:] + [order[j]]
    Returns (pref_masks, suff_masks), each (D-1, D) float keep-masks.
    """
    n = D
    pref = np.ones((n - 1, n), dtype=np.float32)
    suff = np.ones((n - 1, n), dtype=np.float32)
    for j in range(n - 1):
        # prefix: everything before j, plus the item that moves into slot j
        pref[j, order[:j]] = 0.0
        pref[j, order[j + 1]] = 0.0
        # suffix: everything after j+1, plus the item that moves into slot j+1
        suff[j, order[j + 2:]] = 0.0
        suff[j, order[j]] = 0.0
    return torch.from_numpy(pref), torch.from_numpy(suff)


def _best_compound(delta: np.ndarray):
    """Max-weight independent set on a path: pick swap positions at least two
    apart maximising the summed delta. Returns the chosen positions."""
    n = len(delta)
    w = np.maximum(delta, 0.0)
    best = np.zeros(n + 1)
    take = np.zeros(n + 1, dtype=bool)
    for j in range(n):
        skip = best[j]
        use = w[j] + (best[j - 1] if j >= 1 else 0.0)
        if use > skip:
            best[j + 1], take[j] = use, True
        else:
            best[j + 1] = skip
    chosen, j = [], n - 1
    while j >= 0:
        if take[j] and w[j] > 0:
            chosen.append(j)
            j -= 2
        else:
            j -= 1
    return sorted(chosen)


def dynasearch(model, xn, pred, hook, D, seed_order, *, forward_fn=None,
               max_iters=40, verbose=False):
    """Improve ``seed_order`` in place-free fashion. Returns (order, dapc, log)."""
    order = np.asarray(seed_order, dtype=np.int64).copy()
    morf = prob_curve(model, xn, pred, order, hook, D, forward_fn)
    lerf = prob_curve(model, xn, pred, order[::-1].copy(), hook, D, forward_fn)
    cur = dapc_of(morf, lerf)
    log = [cur]
    for it in range(max_iters):
        pref, suff = _changed_sets(order, D)
        p_new = np.asarray(tau(model, xn, pred, pref, hook, D, forward_fn))
        s_new = np.asarray(tau(model, xn, pred, suff, hook, D, forward_fn))
        # DAPC = (sum lerf - sum morf) / (n+1); swap j moves morf[j+1], lerf[D-j-1]
        d_morf = p_new - morf[1:D]                      # index j -> morf[j+1]
        d_lerf = s_new - lerf[D - 1:0:-1]               # index j -> lerf[D-j-1]
        delta = (d_lerf - d_morf) / (D + 1)
        chosen = _best_compound(delta)
        if not chosen:
            break
        for j in chosen:
            order[j], order[j + 1] = order[j + 1], order[j]
            morf[j + 1] = p_new[j]
            lerf[D - j - 1] = s_new[j]
        cur = dapc_of(morf, lerf)
        log.append(cur)
        if verbose:
            print(f"    iter {it+1}: {len(chosen)} swaps, dapc={cur:+.4f}", flush=True)
    return order, cur, log
