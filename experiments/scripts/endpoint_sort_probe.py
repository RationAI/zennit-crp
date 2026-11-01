"""Two-endpoint sort: w_i = [f({i})-f(0)] + [f(V)-f(V\{i})], ascending.

Exactly optimal for DAPC when f has no interactions of order >= 3, and costs
2n+2 forward passes in ONE batched round (vs n(n+1)/2 for greedy).
Compared here against greedy, dynasearch and CP-LRP on the residual site.
"""
import sys, time
import numpy as np, torch
from experiments.concept_detector_optimal import (
    _checked, capture_block_inputs, make_cached_forward, layer_name,
    ZeroChannelsHook, prob_curve, dapc_of, tau)
from experiments.dynasearch_order import dynasearch

KEY, SITE, NIMG = "vit_small_funny_birds", "residual", 4
BLOCKS = [int(b) for b in sys.argv[1:]] or [0, 11]
base = f"/home/claude/workspaces/zennit-crp/data/results/benchmark/cdet_dapc_{KEY}"
bench = np.load(base + ".npz", allow_pickle=True)
opt = np.load(base + "__optimal.npz", allow_pickle=True)

for blk in BLOCKS:
    model, ds, normalize, D, picks, handles, head_dim = _checked(KEY, SITE, blk)
    hook = ZeroChannelsHook(head_dim=head_dim)
    h = model.get_submodule(layer_name(SITE, blk)).register_forward_hook(hook)
    try:
        res = {k: [] for k in ("greedy", "dyna", "cp_lrp", "endp", "endp_dyna")}
        t0 = time.time()
        for j in range(NIMG):
            idx, pred, _ = picks[j]
            xn = normalize(ds[idx][0].unsqueeze(0)).to("cuda")
            fwd = make_cached_forward(model, capture_block_inputs(model, xn), blk)
            # --- one batched round: f(0), f(V), f({i}) and f(V\{i}) for all i
            rows = torch.ones(2 * D + 2, D)
            rows[1] = 0.0                                   # f(V): all removed
            for i in range(D):
                rows[2 + i, i] = 0.0                        # f({i})
                rows[2 + D + i] = 0.0
                rows[2 + D + i, i] = 1.0                    # f(V\{i})
            v = np.asarray(tau(model, xn, pred, rows, hook, D, fwd))
            f0, fV = v[0], v[1]
            w = (v[2:2 + D] - f0) + (fV - v[2 + D:])
            endp = np.argsort(w)                            # ascending
            m = prob_curve(model, xn, pred, endp, hook, D, fwd)
            l = prob_curve(model, xn, pred, endp[::-1].copy(), hook, D, fwd)
            res["endp"].append(dapc_of(m, l))
            _, ed, _ = dynasearch(model, xn, pred, hook, D, endp,
                                  forward_fn=fwd, max_iters=150)
            res["endp_dyna"].append(ed)
            res["greedy"].append(float(opt[f"optimal__{SITE}__b{blk}__img{j}__dapc"]))
            res["cp_lrp"].append(float(bench[f"cp_lrp__{SITE}__b{blk}__dapc"][j]))
            g_order = opt[f"optimal__{SITE}__b{blk}__img{j}__order"]
            _, gd, _ = dynasearch(model, xn, pred, hook, D, g_order,
                                  forward_fn=fwd, max_iters=150)
            res["dyna"].append(gd)
        print(f"b{blk:<2d} " + "  ".join(
            f"{k} {np.mean(v):+.4f}" for k, v in res.items())
            + f"  ({time.time()-t0:.0f}s)", flush=True)
    finally:
        h.remove()
        for hh in handles:
            hh.remove()
print("DONE")
