"""Rank-level analysis of the concept-detector benchmark.

Two estimands, both computed per cell = (model, concept, site, block) with the
**image as the independent unit** — never pooled over blocks, because the same
images recur in every block and blocks are not replicates of each other.

1. ``normalised_dapc`` — DAPC relative to the best ordering any search found.
   DAPC is exactly antisymmetric: swapping MoRF for LeRF negates it, so
   ``DAPC(reverse pi) = -DAPC(pi)`` and therefore
   ``min_pi DAPC = -max_pi DAPC``, with a random ordering at 0 in expectation.
   One search direction is consequently enough to normalise: dividing by the
   best DAPC found puts every cell on a common [-1, 1] scale whose zero is
   chance and whose +1 is the best ordering available. Values above 1 are
   possible and meaningful — they say the ranking beat our search, i.e. the
   bound is an estimate from below, not the attainable optimum.

2. ``overlap_at_k`` — how much two rankings agree on their top k detectors,
   ``|top_k(a) & top_k(b)| / k``. Used two ways: between an LRP ranking and the
   best search ordering (does the ranking find the detectors that matter?), and
   between images for one ranking (is the selected set a property of the model
   or of the image?). Both share the same chance floor: for two independent
   uniform k-subsets of D detectors, ``E|A & B| = k^2 / D``, so the expected
   overlap fraction is ``k / D``.

The aggregate of a per-image quantity is reported as a point estimate with a
percentile bootstrap CI over images, so the sample size that backs each number
is the image count of that cell and nothing larger.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES_DIR = REPO / "data" / "results" / "benchmark"

LRP_METHODS = ("cp_lrp", "chefer")
SEARCH_METHODS = ("optimal", "optimal_dual", "optimal_dyna")
# Scope: the paper's claims are about detectors inside the attention block,
# measured before the output projection. residual/proj_drop are post-projection
# and belong to the supplementary material.
ATTENTION_SITES = ("value", "qk", "query", "key", "head_out")
POST_PROJECTION_SITES = ("proj_drop", "residual")

SITE_LABEL = {"value": "value", "qk": "query/key", "query": "query", "key": "key",
              "head_out": "head output", "proj_drop": "proj_drop",
              "residual": "residual"}
METHOD_LABEL = {"cp_lrp": "CP-LRP", "chefer": "Chefer", "optimal": "greedy",
                "optimal_dyna": "dynasearch",
                "optimal_dual": "dual greedy", "random": "random"}


# ── loading ───────────────────────────────────────────────────────────────────
def _suffixes(concept: str):
    return ("__head", "__head_optimal") if concept == "head" else ("", "__optimal")


@dataclass
class Cells:
    """Everything stored for one (model, concept), keyed by cell."""
    key: str
    concept: str
    D: int
    image_ids: list
    dapc: dict          # (method, site, block) -> (n_images,) float
    order: dict         # (method, site, block) -> (n_images, D) int

    @property
    def n_images(self):
        return len(self.image_ids)

    def sites(self, scope=ATTENTION_SITES):
        have = {s for _m, s, _b in self.dapc}
        return [s for s in scope if s in have]

    def blocks(self, site):
        return sorted({b for _m, s, b in self.dapc if s == site})


def load_cells(key: str, concept: str = "embed_dim") -> Cells:
    """Merge the benchmark store, the search side-car and the ranking dump."""
    suf, osuf = _suffixes(concept)
    bench = RES_DIR / f"cdet_dapc_{key}{suf}.npz"
    z = np.load(bench, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    dapc, order = {}, {}

    for k in z.files:
        m = re.match(r"(.+?)__(.+?)__b(\d+)__dapc$", k)
        if m:
            dapc[(m.group(1), m.group(2), int(m.group(3)))] = np.asarray(z[k], float)

    # search side-car: per-image keys, so collect then stack in image order
    opt = RES_DIR / f"cdet_dapc_{key}{osuf}.npz"
    if opt.exists():
        zo = np.load(opt, allow_pickle=True)
        acc_d, acc_o = {}, {}
        for k in zo.files:
            m = re.match(r"(.+?)__(.+?)__b(\d+)__img(\d+)__(dapc|order)$", k)
            if not m:
                continue
            cell = (m.group(1), m.group(2), int(m.group(3)))
            (acc_d if m.group(5) == "dapc" else acc_o).setdefault(cell, {})[
                int(m.group(4))] = zo[k]
        n = len(meta["image_ids"])
        for cell, per_img in acc_d.items():
            v = np.full(n, np.nan)
            for j, x in per_img.items():
                v[j] = float(x)
            dapc[cell] = v
        for cell, per_img in acc_o.items():
            js = sorted(per_img)
            o = np.full((n, len(per_img[js[0]])), -1, np.int64)
            for j in js:
                o[j] = np.asarray(per_img[j], np.int64)
            order[cell] = o

    # LRP ranking dump (experiments/scripts/dump_lrp_orders.py)
    ords = RES_DIR / f"cdet_orders_{key}{suf}.npz"
    if ords.exists():
        zr = np.load(ords, allow_pickle=True)
        for k in zr.files:
            m = re.match(r"(.+?)__(.+?)__b(\d+)__order$", k)
            if m:
                order[(m.group(1), m.group(2), int(m.group(3)))] = np.asarray(
                    zr[k], np.int64)

    return Cells(key=key, concept=concept, D=int(meta["D"]),
                 image_ids=list(meta["image_ids"]), dapc=dapc, order=order)


# ── aggregation over the independent unit (images) ────────────────────────────
def bootstrap_ci(v, stat=np.median, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap over images. Returns (point, lo, hi, n)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if v.size == 1:
        return float(stat(v)), float("nan"), float("nan"), 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    draws = stat(v[idx], axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(v)), float(lo), float(hi), int(v.size)


# ── estimand 1: normalised DAPC ───────────────────────────────────────────────
def signal_threshold(c: Cells, site, block, floor=0.05, sigmas=3.0):
    """Smallest bound worth dividing by, for this cell.

    A random ordering scores 0 in expectation, so the random spread measures how
    much DAPC a cell produces from nothing. Requiring the bound to clear
    ``sigmas`` times that spread keeps the normalisation from dividing signal by
    noise where the detectors barely matter — collapsed early blocks, mainly.
    ``floor`` applies when the cell has no random control stored."""
    r = c.dapc.get(("random", site, block))
    if r is None or np.sum(np.isfinite(r)) < 2:
        return floor
    return max(floor, sigmas * float(np.nanstd(r, ddof=1)))


def bound_per_image(c: Cells, site, block, eps=None):
    """Best DAPC any search reached, per image. NaN where no search ran and
    where the best does not clear the cell's signal threshold."""
    cand = [c.dapc[(m, site, block)] for m in SEARCH_METHODS
            if (m, site, block) in c.dapc]
    if not cand:
        return None
    if eps is None:
        eps = signal_threshold(c, site, block)
    stack = np.vstack(cand)
    if not np.isfinite(stack).any():
        return None                       # no search ran for this cell
    with np.errstate(invalid="ignore"):
        b = np.nanmax(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    return np.where(np.isfinite(b) & (b > eps), b, np.nan)


def normalised_dapc(c: Cells, method, site, block, **kw):
    """(point, lo, hi, n_used, n_dropped) of per-image DAPC / best-found DAPC."""
    if (method, site, block) not in c.dapc:
        return None
    b = bound_per_image(c, site, block)
    if b is None:
        return None
    v = c.dapc[(method, site, block)] / b
    pt, lo, hi, n = bootstrap_ci(v, **kw)
    return pt, lo, hi, n, int(np.sum(~np.isfinite(v)))


# ── estimand 2: top-k overlap ─────────────────────────────────────────────────
def overlap_at_k(order_a, order_b, k):
    """|top_k(a) & top_k(b)| / k, per row (image)."""
    a, b = np.asarray(order_a), np.asarray(order_b)
    out = np.empty(a.shape[0])
    for j in range(a.shape[0]):
        out[j] = len(set(a[j, :k].tolist()) & set(b[j, :k].tolist())) / k
    return out


def chance_overlap(k, D):
    """Expected overlap fraction and its sd for two independent uniform
    k-subsets of D: |A & B| is hypergeometric(D, k, k)."""
    e = k / D
    if D <= 1:
        return e, 0.0
    var = k * (k / D) * (1 - k / D) * (D - k) / (D - 1)
    return e, float(np.sqrt(var) / k)


INDEPENDENT_SEARCHES = ("optimal", "optimal_dual")   # dyna is seeded from greedy


def search_pair_overlap(c: Cells, site, block, k):
    """Overlap@k between two independently constructed searches: the level a
    ranking can reach without being wrong. Dynasearch is excluded because it
    starts from the greedy order and would inflate the agreement."""
    have = [m for m in INDEPENDENT_SEARCHES if (m, site, block) in c.order]
    if len(have) < 2:
        return None
    v = overlap_at_k(c.order[(have[0], site, block)],
                     c.order[(have[1], site, block)], k)
    return float(v.mean()), have


def best_search_order(c: Cells, site, block):
    """Per image, the stored order of whichever search scored highest."""
    have = [m for m in SEARCH_METHODS
            if (m, site, block) in c.order and (m, site, block) in c.dapc]
    if not have:
        return None
    sc = np.vstack([c.dapc[(m, site, block)] for m in have])
    pick = np.nanargmax(np.where(np.isfinite(sc), sc, -np.inf), axis=0)
    ref = np.stack([c.order[(have[p], site, block)][j]
                    for j, p in enumerate(pick)])
    return ref


def agreement_with_search(c: Cells, method, site, block, k, stat=np.mean, **kw):
    """Overlap@k between an LRP ranking and the best search ordering."""
    ref = best_search_order(c, site, block)
    if ref is None or (method, site, block) not in c.order:
        return None
    return bootstrap_ci(overlap_at_k(c.order[(method, site, block)], ref, k),
                        stat=stat, **kw)


def cross_image_stability(c: Cells, method, site, block, k, max_pairs=120, seed=0):
    """Overlap@k between the top-k sets of two different images, same ranking.
    Chance floor is the same k/D: a ranking that selects the same detectors
    regardless of the image sits near 1, one that re-picks per image near k/D."""
    if (method, site, block) not in c.order:
        return None
    o = c.order[(method, site, block)]
    n = o.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    tops = [set(o[i, :k].tolist()) for i in range(n)]
    v = np.array([len(tops[i] & tops[j]) / k for i, j in pairs])
    # pairs share images, so a bootstrap over pairs would understate the error;
    # report the mean and the spread across pairs instead of a CI.
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan"), v.size


def cell_n(c: Cells, site, block):
    """Images that back this cell: those whose bound clears the threshold."""
    b = bound_per_image(c, site, block)
    return 0 if b is None else int(np.sum(np.isfinite(b)))


# (2) absolute k, pre-specified per granularity; the chance floor k/D is
# reported alongside every value so cells with different D stay readable.
def report_k(D):
    return 2 if D <= 16 else 8


def k_grid(D):
    """Powers of two up to D/4, plus 1; small D gets every sensible k."""
    if D <= 16:
        return [k for k in (1, 2, 3, 4, 6, 8, 12) if k <= D // 2 or k == 1]
    ks, k = [1], 2
    while k <= max(4, D // 4):
        ks.append(k)
        k *= 2
    return ks
