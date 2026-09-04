"""Heuristic optimal rankings for the concept-detector DAPC benchmark.

Journal `exp:insertion-deletion-bench`, "Optimal ranking - heuristic search"
(+ "Optimal ranking - dual greedy"): greedy searches for the removal ranking
that maximises DAPC = area(LeRF) − area(MoRF). All quantities are measured on
predicted-class softmax probabilities (τ), never logits.

Terms (shared by all variants, defined once at the top):
    p_current     = M(x|y) at the current removal state
    tau(keep)     = pred-class softmax prob under a batch of keep-masks
    Δ_c           = p_current − τ(c)(x|y)   (decrease taken from the prediction)

Variants (all built from the same `marginal_deltas` primitive):
    greedy  O(n^2)      per step remove argmax Δ_c (most damaging first)
    dual    O(n^2)      per step rank TWO: argmax Δ → head (MoRF-first),
                        argmin Δ → tail (LeRF-first)
    pair    O(n^3)      per step remove the most damaging pair's stronger member
                        (infeasible for grids; kept for spec completeness)

Occlusion: detector = one embed-dim channel at the (B, N, D) probe site;
removal = zero that channel across all tokens (forward-side equivalent of
EmbeddingDimConcept's relevance-stream masking; cf. `occlusion_check`).

Checkpoint/resume: OptimalStore commits the ORDER immediately after it is
built (the expensive part) and curves right after; a rerun skips finished
combos. Restart with the same command.

CLI: --action run --model-key <tag> --mode {greedy,dual,pair} --n-imgs N
     --action probe --site residual --block 11 --img 0
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

# ── terms / constants ────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
RES_DIR = REPO / "data" / "results" / "benchmark"
if not torch.cuda.is_available():
    raise RuntimeError("concept_detector_optimal requires CUDA")
DEVICE = "cuda"

BATCH = 256            # keep-mask rows per forward in tau()
N_IMAGES = 16          # bench protocol images
SEED = 0               # same seed as the bench → identical picks
BLOCKS = list(range(12))
ALL_SITES = ["residual", "proj_drop", "value", "qk"]

MODELS_CFG = [
    ("vit_base_imagenet", "imagenet", {}),
    ("vit_small_funny_birds", "funny_birds", {"split": "test"}),
]

LAYER_NAME = {
    "residual": "backbone.blocks.{b}",
    "proj_drop": "backbone.blocks.{b}.attn.proj_drop",
    "qk": "backbone.blocks.{b}.attn.q_lrp_probe",
    "value": "backbone.blocks.{b}.attn.v_lrp_probe",
}


def layer_name(site: str, b: int) -> str:
    return LAYER_NAME[site].format(b=b)


# ── occlusion (forward zero-ablation) ────────────────────────────────────────
class ZeroChannelsHook:
    """out * keep.unsqueeze(1): row r zeroes the removed channels of eval r."""

    def __init__(self):
        self.keep = None

    def __call__(self, module, inp, out):
        if self.keep is None:
            return out
        assert out.shape[-1] == self.keep.shape[-1], \
            f"keep-mask width {self.keep.shape[-1]} != layer width {out.shape[-1]}"
        return out * self.keep.unsqueeze(1)


def keep_of(removed: np.ndarray, D: int) -> torch.Tensor:
    """(D,) keep row of the current removal state: 1 = kept, 0 = removed."""
    keep = torch.ones(D)
    keep[removed] = 0.0
    return keep


# ── tau: measured quantity (softmax probability) ─────────────────────────────
def tau(model, xn, pred, keep_rows, hook, D):
    """τ(S)(x|y): pred-class softmax prob under keep-mask rows (R, D)."""
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


# ── marginal deltas: the shared heuristic term ───────────────────────────────
def marginal_deltas(model, xn, pred, removed, candidates, hook, D):
    """Δ_c = p_current − τ(c)(x|y) for every candidate at the current state."""
    keep = keep_of(removed, D)
    rows = keep.repeat(len(candidates), 1)
    rows[torch.arange(len(candidates)), torch.from_numpy(candidates)] = 0.0
    p_ablated = tau(model, xn, pred, rows, hook, D)
    p_current = tau(model, xn, pred, keep[None], hook, D)[0]
    return p_current - p_ablated


# ── DAPC metric pieces ───────────────────────────────────────────────────────
_trapz = getattr(np, "trapezoid", None) or np.trapezoid


def dapc_of(morf: np.ndarray, lerf: np.ndarray) -> float:
    n = len(morf) - 1
    return float(_trapz(lerf, dx=1.0 / n) - _trapz(morf, dx=1.0 / n))


def cumulative_keep(D: int, order) -> torch.Tensor:
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


# ── Variant A: O(n^2) greedy ─────────────────────────────────────────────────
# repeat:  Δ_c = p_current − τ(c)(x|y);  c* = argmax Δ_c;  remove c*.
def greedy_order(model, xn, pred, hook, D: int, steps_cap: int = 0) -> np.ndarray:
    removed = np.zeros(D, dtype=bool)
    order: list[int] = []
    t0 = time.time()
    for step in range(min(D, steps_cap or D)):
        candidates = np.flatnonzero(~removed)
        delta = marginal_deltas(model, xn, pred, removed, candidates, hook, D)
        c = int(candidates[int(np.argmax(delta))])                # argmax Δ_c
        removed[c] = True
        order.append(c)
        if (step + 1) % 32 == 0:
            print(f"      step {step+1}/{min(D, steps_cap or D)} "
                  f"({(time.time()-t0)/(step+1):.2f}s/step)", flush=True)
    return _full_permutation(order, removed, D, steps_cap)


# ── dual greedy: O(n^2), two detectors ranked per turn ───────────────────────
# per turn, same state evaluation as Variant A:
#   h = argmax_c Δ_c → next-from-top    (strongest: removed first under MoRF)
#   l = argmin_c Δ_c → next-from-bottom (weakest:  removed last under MoRF,
#                                        i.e. first under LeRF)
# final ranking = head ++ reversed(tail); one odd leftover goes to the head.
def dual_order(model, xn, pred, hook, D: int, steps_cap: int = 0) -> np.ndarray:
    removed = np.zeros(D, dtype=bool)
    head: list[int] = []
    tail: list[int] = []
    t0, turns = time.time(), 0
    while np.count_nonzero(~removed) >= 2 and (not steps_cap or turns < steps_cap):
        candidates = np.flatnonzero(~removed)
        delta = marginal_deltas(model, xn, pred, removed, candidates, hook, D)
        hi = int(np.argmax(delta))                                # strongest → head
        rest = np.ones(len(candidates), dtype=bool)
        rest[hi] = False                                          # exclude h this turn
        li = int(np.flatnonzero(rest)[int(np.argmin(delta[rest]))])  # weakest
        h, l = int(candidates[hi]), int(candidates[li])
        removed[h] = removed[l] = True
        head.append(h)
        tail.append(l)
        turns += 1
        if turns % 32 == 0:
            print(f"      turn {turns}/{D//2} "
                  f"({(time.time()-t0)/turns:.2f}s/turn, 2 ranks/turn)", flush=True)
    head.extend(np.flatnonzero(~removed).tolist())
    full = np.array(head + tail[::-1], dtype=np.int64)
    assert steps_cap or len(set(full.tolist())) == D == len(full), "not a permutation"
    return full


# ── Variant B pair search: O(n^3) ────────────────────────────────────────────
# per step: Δ_{c1,c2} = p_current − τ(c1,c2)(x|y) for every pair;
# (c1*,c2*) = argmax Δ; compare the pair's individual Δ, remove the stronger.
# (spec text "minimizes the Δ": read as max decrease, i.e. min ablated prob.)
def pair_order(model, xn, pred, hook, D: int, steps_cap: int = 0) -> np.ndarray:
    removed = np.zeros(D, dtype=bool)
    order: list[int] = []
    t0 = time.time()
    for step in range(min(D, steps_cap or D)):
        candidates = np.flatnonzero(~removed)
        if len(candidates) == 1:
            removed[candidates[0]] = True
            order.append(int(candidates[0]))     # last detector: no pair left
            continue
        keep = keep_of(removed, D)
        best_delta, best_pair = -np.inf, None
        for i in range(len(candidates)):
            n_j = len(candidates) - i - 1
            if n_j == 0:
                continue
            rows = keep.repeat(n_j, 1)                     # pairs (c_i, c_j), j > i
            rows[torch.arange(n_j), torch.from_numpy(candidates[i:i + 1]).expand(n_j)] = 0.0
            rows[torch.arange(n_j), torch.from_numpy(candidates[i + 1:])] = 0.0
            p_pair = tau(model, xn, pred, rows, hook, D)   # τ(c1,c2)(x|y)
            j_rel = int(np.argmin(p_pair))                 # max Δ  ==  min ablated prob
            if -p_pair[j_rel] > best_delta:
                best_delta = -float(p_pair[j_rel])
                best_pair = (int(candidates[i]), int(candidates[i + 1 + j_rel]))
        c1, c2 = best_pair                                 # argmax Δ_{c1,c2}
        delta_1, delta_2 = marginal_deltas(model, xn, pred, removed,
                                           np.array([c1, c2]), hook, D)
        c = c1 if delta_1 >= delta_2 else c2               # higher individual Δ removed
        removed[c] = True
        order.append(c)
        print(f"      step {step+1}/{min(D, steps_cap or D)} "
              f"({(time.time()-t0)/(step+1):.2f}s/step, "
              f"pairs/step {len(candidates)*(len(candidates)-1)//2})", flush=True)
    return _full_permutation(order, removed, D, steps_cap)


def _full_permutation(order: list[int], removed: np.ndarray, D: int,
                      steps_cap: int) -> np.ndarray:
    full = np.array(order + np.flatnonzero(~removed).tolist(), dtype=np.int64)
    if not steps_cap:
        assert len(set(full.tolist())) == D == len(full), f"order not a permutation of {D}"
    return full


# ── data selection: first N_IMAGES correctly-classified, bench seed ──────────
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


# ── spec sanity: dims line up, readout is a probability ──────────────────────
def occlusion_check(model, xn, pred, site, b, D):
    mod = model.get_submodule(layer_name(site, b))
    hook = ZeroChannelsHook()
    hh = mod.register_forward_hook(hook)
    seen = {}

    def rec(module, inp, out):
        seen["shape"] = out.shape
    hr = mod.register_forward_hook(rec)
    try:
        base = tau(model, xn, pred, torch.ones(1, D), hook, D)[0]
        assert seen["shape"][-1] == D, f"probe width {seen['shape'][-1]} != D={D}"
        assert 0.0 <= float(base) <= 1.0, f"not a probability: {base}"
    finally:
        hh.remove()
        hr.remove()
    print(f"  [occlusion-check] {site} b{b}: probe (B,N,{D}); dims line up; "
          f"base prob={base:.3f} OK", flush=True)


# ── incremental result store (atomic checkpointing) ──────────────────────────
class OptimalStore:
    """Side-car npz re-saved atomically after every finished sub-result.

    Keys per combo:  <method>__<site>__b<blk>__img<j>__{order,morf,lerf,dapc}
    Layer aggregates (bench-shaped): <method>__<site>__b<blk>__{morf,lerf,dapc}
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
        self.store[key] = obj
        tmp = self.path.with_name(self.path.stem + ".tmp")
        np.savez(tmp, **self.store)
        tmp.with_suffix(".tmp.npz").replace(self.path)


def combo_keys(mkey, site, b, j):
    base = f"{mkey}__{site}__b{b}__img{j}"
    return {k: f"{base}__{k}" for k in ("order", "morf", "lerf", "dapc")}


ORDER_BUILDERS = {"greedy": greedy_order, "pair": pair_order, "dual": dual_order}


def run_combo(model, x, xn, pred, site, b, D, mode, sto, j, steps_cap=0) -> dict:
    """One (image, site, block) combo with order-level checkpoint reuse."""
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


# ── model loading + action drivers ───────────────────────────────────────────
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


def _checked(key, site, b):
    """Load model/ds and run the spec occlusion check on one probe site once."""
    model, ds, normalize, D, picks, handles = load(key)
    idx, pred, _ = picks[0]
    xn = normalize(ds[idx][0].unsqueeze(0)).to(DEVICE)
    occlusion_check(model, xn, pred, site, b, D)
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
