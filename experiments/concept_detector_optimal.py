"""Heuristic optimal ranking for the concept-detector DAPC benchmark.

Journal `exp:insertion-deletion-bench`, paragraph "Optimal ranking -
heuristic search": a greedy search for the removal ranking that maximises
DAPC = area(LeRF) − area(MoRF). Two variants, written verbatim in the
formulation's members (p_current = M(x|y) at the current removal state,
τ(S)(x|y) = predicted-class probability with detector set S removed,
Δ = p_current − τ, the decrease taken from the prediction):

* Variant A (O(n^2), ``greedy_order``): repeat — evaluate Δ_c for every
  remaining single detector at the current state, remove the argmax
  (highest decrease taken first). n + (n−1) + ... = n(n+1)/2 evaluations.
* Variant B (O(n^3), ``pair_order``): repeat — evaluate Δ_{c1,c2} for every
  pair, find the pair with the largest joint decrease, then remove the pair
  member with the higher individual decrease.
  (Spec text says "minimizes the Δ_{c1,c2}" — read as maximising the
  decrease, i.e. minimising the ablated probability τ(c1,c2)(x|y); the
  literal reading would pick the pair that changes the class LEAST and
  contradicts the goal.)

Everything is measured on predicted-class softmax probabilities (competing
classes are accounted for by the renormalisation), never on logits.

Occlusion semantics (checked at startup, ``occlusion_check``): detector =
one embedding-dim channel of the (B, N, D) tensor at the probe site;
removal = zero that channel across all tokens — exactly the
``EmbeddingDimConcept`` masking semantics applied forward-side
(``concept.mask`` itself is a backward relevance hook and cannot be used in
no_grad evaluation). Perturbation is batched: every row of a keep-mask is
an independent eval.

SELF-CONTAINED: imports nothing from experiments.concept_detector_bench
(the shared ≤15-line primitives are inlined here verbatim); only the
project-wide model/dataset loaders and the attention-substitution
canonizer are shared infrastructure. Results land in
``data/results/benchmark/cdet_dapc_<key>__optimal.npz`` and the bench
renderer merges this side-car.

CHECKPOINTING / RESUME: the OptimalStore re-saves after every finished
sub-result. The expensive artefact of a combo is the removal *order*
(~1–15 min); it is committed the moment it exists, and the cheap curve
evaluation (~seconds) afterwards, so a crash mid-combo resumes by
re-using the stored order. Combos with curves committed are skipped
entirely. Restart by simply running the same command again.

Usage:
    VIRTUAL_ENV=$PWD/.venv .venv/bin/python -m experiments.concept_detector_optimal \
        --action run --model-key vit_small_funny_birds --n-imgs 16
    ... --action probe --mode pair --site residual --block 0   (single combo pilot/pair demo)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.datasets import load_eval_dataset
from experiments.models import backbone_transforms
from experiments.model_datasets import find_by_tag
from zennit_extensions.canonisation.canonizers import VanillaViTAttentionSubstitutionCanonizer

REPO = Path(__file__).resolve().parents[1]
RES_DIR = REPO / "data" / "results" / "benchmark"
if not torch.cuda.is_available():
    raise RuntimeError("concept_detector_optimal requires CUDA (spec workloads are batched GPU forwards)")
DEVICE = "cuda"

BATCH = 256          # keep-mask rows per forward in tau() (rows are batch-independent)
N_IMAGES = 16        # bench protocol: 16 correctly-classified images
SEED = 0             # same seed as the bench → same picks
BLOCKS = list(range(12))
ALL_SITES = ["residual", "proj_drop", "value", "qk"]

# (zoo key, dataset key, ds_extra) — M1 = ViT-S/FB, M2 = ViT-B/ImageNet
MODELS_CFG = [
    ("vit_base_imagenet", "imagenet", {}),
    ("vit_small_funny_birds", "funny_birds", {"split": "test"}),
]


def layer_name(site: str, b: int) -> str:
    return {
        "residual": f"backbone.blocks.{b}",
        "proj_drop": f"backbone.blocks.{b}.attn.proj_drop",
        "qk": f"backbone.blocks.{b}.attn.q_lrp_probe",
        "value": f"backbone.blocks.{b}.attn.v_lrp_probe",
    }[site]


# ── forward occlusion (zero-ablation) ────────────────────────────────────────
class ZeroChannelsHook:
    """Batched keep-mask multiplication on the probed layer's output.

    keep: (R, D) float where row r zeroes the removed detectors of eval r.
    Applied as out * keep.unsqueeze(1) so out (R, N, D) keeps every token but
    zeroes the removed channels. (Removal == "channel zeroed across all
    tokens", cf. EmbeddingDimConcept's relevance-stream masking.)
    """

    def __init__(self):
        self.keep = None

    def __call__(self, module, inp, out):
        if self.keep is None:
            return out
        assert out.shape[-1] == self.keep.shape[-1], \
            f"keep-mask width {self.keep.shape[-1]} != layer width {out.shape[-1]}"
        return out * self.keep.unsqueeze(1)


def occlusion_check(model, xn, pred, site, b, D):
    """Fail-fast spec verification: removing detector d zeroes exactly channel
    d across all tokens at the probe site, leaves the rest untouched, and the
    measured value is a softmax **probability** in [0, 1]."""
    mod = model.get_submodule(layer_name(site, b))
    hook = ZeroChannelsHook()
    hh = mod.register_forward_hook(hook)
    seen = {}

    def rec(module, inp, out):
        seen["shape"] = out.shape
    hr = mod.register_forward_hook(rec)
    try:
        base = tau(model, xn, pred, torch.ones(1, D), hook, D)[0]
        d = 0
        rows = torch.ones(2, D)
        rows[1, d] = 0.0
        tau(model, xn, pred, rows, hook, D)
        assert seen["shape"][-1] == D, f"probe width {seen['shape'][-1]} != D={D}"
        assert 0.0 <= float(base) <= 1.0, f"not a probability: {base}"
    finally:
        hh.remove()
        hr.remove()
    print(f"  [occlusion-check] {site} b{b}: probe (B,N,{D}); keep-mask dims line up; "
          f"base prob={base:.3f} OK", flush=True)


# ── DAPC metric pieces ───────────────────────────────────────────────────────
_trapz = getattr(np, "trapezoid", None) or np.trapezoid


def dapc_of(morf: np.ndarray, lerf: np.ndarray) -> float:
    n = len(morf) - 1
    return float(_trapz(lerf, dx=1.0 / n) - _trapz(morf, dx=1.0 / n))


def cumulative_keep(D: int, order: np.ndarray) -> torch.Tensor:
    """(D+1, D) keep-masks: row k has channels order[:k] zeroed (removed)."""
    order = torch.as_tensor(order)
    rank = torch.empty(D, dtype=torch.long)
    rank[order] = torch.arange(D)
    k = torch.arange(D + 1).unsqueeze(1)
    return (rank.unsqueeze(0) >= k).float()


def prob_curve(model, xn, pred, order, hook, D) -> np.ndarray:
    """Predicted-class probability along cumulative removals of ``order``."""
    keep = cumulative_keep(D, order)
    assert keep.shape == (D + 1, D), f"keep {tuple(keep.shape)} != ({D+1},{D})"
    return tau(model, xn, pred, keep, hook, D)


# ── data selection (same protocol as the bench: first 16 correctly-classified) ─
def select_correct(model, normalize, ds, n):
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED)).tolist()
    picks = []
    with torch.no_grad():
        for i in perm:
            x, y = ds[i]
            p = model(normalize(x.unsqueeze(0).to(DEVICE))).softmax(-1)[0]
            pred = int(p.argmax())
            if pred == int(y):
                picks.append((i, pred, float(p[pred])))
                if len(picks) >= n:
                    break
    return picks


# ── the measurement primitive ────────────────────────────────────────────────
def keep_of(removed: np.ndarray, D: int) -> torch.Tensor:
    """(D,) keep row of the current removal state: 1 = kept, 0 = removed."""
    keep = torch.ones(D)
    keep[removed] = 0.0
    return keep


def tau(model, xn, pred, keep_rows, hook, D):
    """τ(S)(x|y): predicted-class softmax probability under keep-mask rows.

    keep_rows: (R, D), row r = one perturbation state (1 = detector kept,
    0 = removed). Rows are batch-independent, chunked to the GPU.
    """
    assert keep_rows.ndim == 2 and keep_rows.shape[1] == D, \
        f"keep_rows {tuple(keep_rows.shape)} — expected (R, {D})"
    assert xn.shape[0] == 1, "tau evaluates one image at a time"
    out = torch.empty(keep_rows.shape[0])
    with torch.no_grad():
        for s in range(0, keep_rows.shape[0], BATCH):
            kb = keep_rows[s:s + BATCH].to(DEVICE)
            hook.keep = kb
            logits = model(xn.expand(kb.shape[0], -1, -1, -1))
            out[s:s + kb.shape[0]] = logits.softmax(-1)[:, pred].cpu()
    hook.keep = None
    return out.numpy()


# ── Variant A: O(n^2) greedy ─────────────────────────────────────────────────
# repeat:  Δ_c = p_current − τ(c)(x|y)     (decrease for the target class)
#          c*  = argmax_c Δ_c              (highest decrease taken first)
#          remove c*, repeat on the new state, until all detectors are removed.
def greedy_order(model, xn, pred, hook, D: int, steps_cap: int = 0) -> np.ndarray:
    removed = np.zeros(D, dtype=bool)
    order: list[int] = []
    t0 = time.time()
    for step in range(min(D, steps_cap or D)):
        candidates = np.flatnonzero(~removed)
        keep = keep_of(removed, D)                       # zero what `order` removed
        # τ(c)(x|y) for all remaining c at the current state, one row per candidate
        rows = keep.repeat(len(candidates), 1)
        rows[torch.arange(len(candidates)), torch.from_numpy(candidates)] = 0.0
        p_ablated = tau(model, xn, pred, rows, hook, D)          # τ(c)(x|y) each c
        p_current = tau(model, xn, pred, keep[None], hook, D)[0]  # M(x|y) at state
        delta = p_current - p_ablated                             # Δ_c
        c = int(candidates[int(np.argmax(delta))])                # argmax Δ_c
        removed[c] = True
        order.append(c)
        if (step + 1) % 32 == 0:
            print(f"      step {step+1}/{min(D, steps_cap or D)} "
                  f"({(time.time()-t0)/(step+1):.2f}s/step)", flush=True)
    full = np.array(order + np.flatnonzero(~removed).tolist(), dtype=np.int64)
    assert len(set(full.tolist())) == D == len(full) or steps_cap, "not a permutation"
    return full


# ── dual greedy: O(n^2), two detectors ranked per turn ───────────────────────
# same per-turn evaluation as variant A, but both ends of the ranking are
# built at once. Per turn, at the current removal state:
#   Δ_c = p_current − τ(c)(x|y)  for every remaining c,
#   h = argmax_c Δ_c  → next-from-top    (strongest: removed first under MoRF)
#   l = argmin_c Δ_c  → next-from-bottom (weakest:  removed last under MoRF,
#                                        i.e. first under LeRF)
# final ranking = head ++ reversed(tail). Complexity O(n^2) like variant A,
# with ~half the state evaluations (two ranks per turn; one odd leftover
# goes to the head for D odd).
def dual_order(model, xn, pred, hook, D: int, steps_cap: int = 0) -> np.ndarray:
    removed = np.zeros(D, dtype=bool)
    head: list[int] = []
    tail: list[int] = []
    t0 = time.time()
    turns = 0
    while np.count_nonzero(~removed) >= 2 and (not steps_cap or turns < steps_cap):
        candidates = np.flatnonzero(~removed)
        keep = keep_of(removed, D)                       # zero what head removed
        # τ(c)(x|y) for all remaining c at the current state, one row per candidate
        rows = keep.repeat(len(candidates), 1)
        rows[torch.arange(len(candidates)), torch.from_numpy(candidates)] = 0.0
        p_ablated = tau(model, xn, pred, rows, hook, D)          # τ(c)(x|y) each c
        p_current = tau(model, xn, pred, keep[None], hook, D)[0]  # M(x|y) at state
        delta = p_current - p_ablated                             # Δ_c
        hi = int(np.argmax(delta))                                # strongest → head
        rest_hi = np.ones(len(candidates), dtype=bool)
        rest_hi[hi] = False                                       # exclude h this turn
        li = int(np.flatnonzero(rest_hi)[int(np.argmin(delta[rest_hi]))])  # weakest
        h, l = int(candidates[hi]), int(candidates[li])
        removed[h] = removed[l] = True
        head.append(h)
        tail.append(l)
        turns += 1
        if turns % 32 == 0:
            print(f"      turn {turns}/{D//2} "
                  f"({(time.time()-t0)/turns:.2f}s/turn, 2 ranks/turn)", flush=True)
    head.extend(np.flatnonzero(~removed).tolist())   # odd leftover (D odd) or pilot tail
    full = np.array(head + tail[::-1], dtype=np.int64)
    assert steps_cap or len(set(full.tolist())) == D == len(full), "not a permutation"
    return full


# ── Variant B: O(n^3) pair search ────────────────────────────────────────────
# repeat:  Δ_{c1,c2} = p_current − τ(c1,c2)(x|y)   (joint decrease per pair)
#          (c1*,c2*) = argmax Δ_{c1,c2}            (most damaging pair)
#          Δ_c1 = p_current − τ(c1), Δ_c2 = p_current − τ(c2)
#          remove the pair member with the higher individual Δ, and repeat.
def pair_order(model, xn, pred, hook, D: int, steps_cap: int = 0) -> np.ndarray:
    removed = np.zeros(D, dtype=bool)
    order: list[int] = []
    t0 = time.time()
    for step in range(min(D, steps_cap or D)):
        candidates = np.flatnonzero(~removed)
        r = len(candidates)
        keep = keep_of(removed, D)
        if r == 1:
            removed[candidates[0]] = True
            order.append(int(candidates[0]))     # last detector: no pair left
            continue
        # Δ_{c1,c2} over all remaining pairs (outer loop on c1 so the
        # r-choose-2 keep-mask rows never materialise at once)
        best_delta, best_pair = -np.inf, None
        for i in range(r):
            n_j = r - i - 1
            if n_j == 0:
                continue
            rows = keep.repeat(n_j, 1)                     # pairs (c_i, c_j), j > i
            rows[torch.arange(n_j), torch.from_numpy(candidates[i:i + 1]).expand(n_j)] = 0.0
            rows[torch.arange(n_j), torch.from_numpy(candidates[i + 1:])] = 0.0
            p_pair = tau(model, xn, pred, rows, hook, D)   # τ(c1,c2)(x|y)
            j_rel = int(np.argmin(p_pair))                 # max Δ  ==  min ablated prob
            delta_pair = -float(p_pair[j_rel])             # up to +p_current const
            if delta_pair > best_delta:
                best_delta = delta_pair
                best_pair = (int(candidates[i]), int(candidates[i + 1 + j_rel]))
        c1, c2 = best_pair                                 # argmax Δ_{c1,c2}
        # individually: Δ_c1 vs Δ_c2 at the current state
        rows = keep.repeat(2, 1)
        rows[0, c1] = 0.0
        rows[1, c2] = 0.0
        p_solo = tau(model, xn, pred, rows, hook, D)       # τ(c1), τ(c2)
        p_current = tau(model, xn, pred, keep[None], hook, D)[0]
        delta_1, delta_2 = p_current - p_solo[0], p_current - p_solo[1]
        c = c1 if delta_1 >= delta_2 else c2               # higher Δ removed
        removed[c] = True
        order.append(c)
        print(f"      step {step+1}/{min(D, steps_cap or D)} "
              f"({(time.time()-t0)/(step+1):.2f}s/step, pairs/step {r*(r-1)//2})", flush=True)
    full = np.array(order + np.flatnonzero(~removed).tolist(), dtype=np.int64)
    assert steps_cap or len(set(full.tolist())) == D == len(full), f"order not a permutation of {D}"
    return full


# ── incremental result store (checkpointing) ─────────────────────────────────
class OptimalStore:
    """Side-car npz re-saved after every finished sub-result → resume-safe.

    Per combo keys:  <method>__<site>__b<blk>__img<j>__{order,morf,lerf,dapc}
    Per-layer keys (bench-shaped, (n_imgs,*)): <method>__<site>__b<blk>__{morf,lerf,dapc}
    """

    METHOD = {"greedy": "optimal", "pair": "optimal_pair", "dual": "optimal_dual"}

    def __init__(self, key: str):
        self.path = RES_DIR / f"cdet_dapc_{key}__optimal.npz"
        self.store = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=True)
            self.store = {k: z[k] for k in z.files}
            print(f"resume: {len(self.store)} keys in {self.path.name}", flush=True)

    def has(self, k: str) -> bool:
        return k in self.store

    def get(self, k: str):
        return self.store[k]

    def commit(self, key: str, obj) -> None:
        """Atomic checkpoint: write a temp file, then replace. A crash
        mid-commit leaves the last complete checkpoint loadable."""
        self.store[key] = obj
        tmp = self.path.with_name(self.path.stem + ".tmp")
        np.savez(tmp, **self.store)                  # np.savez appends .npz
        tmp.with_suffix(".tmp.npz").replace(self.path)


def combo_keys(mkey, site, b, j):
    base = f"{mkey}__{site}__b{b}__img{j}"
    return {k: f"{base}__{k}" for k in ("order", "morf", "lerf", "dapc")}


ORDER_BUILDERS = {"greedy": greedy_order, "pair": pair_order, "dual": dual_order}


def run_combo(model, x, xn, pred, site, b, D, mode, sto, j, steps_cap=0) -> dict:
    """One (image, site, block) combo with order-level checkpoint reuse."""
    assert mode in ORDER_BUILDERS, f"unknown mode {mode!r}"
    keys = combo_keys(OptimalStore.METHOD[mode], site, b, j)
    hook = ZeroChannelsHook()
    hh = model.get_submodule(layer_name(site, b)).register_forward_hook(hook)
    try:
        if sto.has(keys["order"]):
            order = sto.get(keys["order"])            # resume after mid-combo crash
        else:
            order = ORDER_BUILDERS[mode](model, xn, pred, hook, D, steps_cap=steps_cap)
            sto.commit(keys["order"], order)          # commit expensive artefact 1st
        if steps_cap:
            return {"order": order}
        order_t = torch.from_numpy(order).long()         # one numpy→torch hop
        morf = prob_curve(model, xn, pred, order_t, hook, D)
        lerf = prob_curve(model, xn, pred, order_t.flip(0), hook, D)
        out = {"order": order, "morf": morf, "lerf": lerf, "dapc": dapc_of(morf, lerf)}
        for k in ("morf", "lerf", "dapc"):
            sto.commit(keys[k], out[k])
        return out
    finally:
        hh.remove()


# ── model loading (shared infrastructure, canonized like the bench) ──────────
def load(key):
    model = find_by_tag(key, device=DEVICE).model.eval()
    transform, normalize = backbone_transforms(model.backbone)
    cfg = next(c for c in MODELS_CFG if c[0] == key)
    ds = load_eval_dataset(cfg[1], transform, cfg[2])
    D = int(model.backbone.embed_dim)
    picks = select_correct(model, normalize, ds, N_IMAGES)
    canon = VanillaViTAttentionSubstitutionCanonizer(block_indices=None)
    handles = canon.apply(model)
    print(f"[{key}] loaded; D={D}, {len(picks)} picks (seed {SEED})", flush=True)
    return model, ds, normalize, D, picks, handles


def action_probe(key, site, block, mode, img_i, steps_cap):
    model, ds, normalize, D, picks, handles = _checked(key, site, block)
    try:
        idx, pred, _ = picks[img_i]
        print(f"probe[{mode}] {key} {site} b{block} img#{img_i} (ds {idx}), D={D}", flush=True)
        x = ds[idx][0].unsqueeze(0)
        xn = normalize(x).to(DEVICE)
        sto = OptimalStore(key)
        t0 = time.time()
        out = run_combo(model, x, xn, pred, site, block, D, mode, sto, img_i,
                        steps_cap=steps_cap)
        dt = time.time() - t0
        print(f"PROBE_RESULT {mode} {key} D={D} wall={dt:.1f}s", flush=True)
        if not steps_cap:
            print(f"  {site} b{block} img#{img_i}: {mode} dapc={out['dapc']:+.3f}", flush=True)
    finally:
        for h in handles:
            h.remove()


def action_run(key, mode, sites, blocks, n_imgs):
    model, ds, normalize, D, picks, handles = _checked(key, sites[0], blocks[0])
    sto = OptimalStore(key)
    mkey = OptimalStore.METHOD[mode]
    try:
        if not sto.has("meta"):
            sto.commit("meta", json.dumps({"key": key, "mode": mode, "D": D,
                                           "image_ids": [i for i, _, _ in picks],
                                           "preds": [p for _, p, _ in picks]}))
        for site in sites:
            for b in blocks:
                morf_rows, lerf_rows, dapc_rows, pending = [], [], [], []
                for j, (idx, pred, _) in enumerate(picks[:n_imgs]):
                    keys = combo_keys(mkey, site, b, j)
                    if sto.has(keys["dapc"]):
                        morf_rows.append(sto.get(keys["morf"]))
                        lerf_rows.append(sto.get(keys["lerf"]))
                        dapc_rows.append(sto.get(keys["dapc"]))
                    else:
                        pending.append((j, idx, pred))
                for j, idx, pred in pending:
                    t0 = time.time()
                    x = ds[idx][0].unsqueeze(0)
                    xn = normalize(x).to(DEVICE)
                    out = run_combo(model, x, xn, pred, site, b, D, mode, sto, j)
                    print(f"[{key}] {mkey} {site} b{b} img#{j} "
                          f"dapc={out['dapc']:+.3f} ({time.time()-t0:.0f}s)", flush=True)
                    morf_rows.append(out["morf"])
                    lerf_rows.append(out["lerf"])
                    dapc_rows.append(out["dapc"])
                if dapc_rows:
                    sto.commit(f"{mkey}__{site}__b{b}__morf", np.stack(morf_rows))
                    sto.commit(f"{mkey}__{site}__b{b}__lerf", np.stack(lerf_rows))
                    sto.commit(f"{mkey}__{site}__b{b}__dapc",
                               np.array(dapc_rows, np.float32))
                    print(f"[{key}] {mkey} {site} b{b} done "
                          f"({len(dapc_rows)} images)", flush=True)
    finally:
        for h in handles:
            h.remove()


def _checked(key, site, b):
    """Load model/ds and run the spec occlusion check on one probe site once."""
    model, ds, normalize, D, picks, handles = load(key)
    idx, pred, _ = picks[0]
    xn = normalize(ds[idx][0].unsqueeze(0)).to(DEVICE)
    occlusion_check(model, xn, pred, site, b, D)
    return model, ds, normalize, D, picks, handles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=list(ORDER_BUILDERS), default="greedy")
    ap.add_argument("--action", choices=["probe", "run"], default="run")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--site", default="residual")
    ap.add_argument("--block", type=int, default=11)
    ap.add_argument("--img", type=int, default=0)
    ap.add_argument("--steps-cap", type=int, default=0,
                    help="limit removals (probe); 0 = full order")
    ap.add_argument("--sites", nargs="*", default=ALL_SITES)
    ap.add_argument("--blocks", nargs="*", type=int, default=BLOCKS)
    ap.add_argument("--n-imgs", type=int, default=N_IMAGES)
    args = ap.parse_args()
    if args.action == "probe":
        action_probe(args.model_key, args.site, args.block, args.mode, args.img,
                     args.steps_cap)
    else:
        action_run(args.model_key, args.mode, args.sites, args.blocks, args.n_imgs)


if __name__ == "__main__":
    main()
