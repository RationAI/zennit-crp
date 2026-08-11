"""XAI-37 — Step 3 of the register-token study: H_B occlusion test.

Does the input CONTENT at outlier-token positions causally matter for the
prediction?  H_B says yes (high input relevance there is faithful); the
registers paper (Darcet et al., 2309.16588) predicts no — outlier patches are
chosen *because* their local information is redundant/discardable.

Phased CLI so each GPU chunk runs under the shared GPU lock separately:

    python -m experiments.scripts.registers_step3_occlusion scan      # GPU
    python -m experiments.scripts.registers_step3_occlusion select --blocks 8,9,10,11   # CPU
    python -m experiments.scripts.registers_step3_occlusion lrp       # GPU
    python -m experiments.scripts.registers_step3_occlusion occlude   # GPU
    (analysis/figures live in a separate CPU step driven by the note)

Protocol
--------
* Model: timm ViT-B/16 ImageNet-1k (experiments.models.ImagenetViTBase).
* Data: imagenet_val_hf, n_per_class=10 (10k pool), shuffled with a fixed seed.
* Outlier tokens: per-block L2 norms of the block-output patch tokens (CLS
  excluded), threshold = mean + 4*sd over the scan pool per block; per-image
  outlier set = union over the chosen detection blocks.
* N=128 correctly-classified images with >=1 outlier patch.
* Occlusions in PIXEL space ([0,1], before normalization), 16x16 patches on
  the 14x14 grid:
    a      primary outlier patch <- constant fill = mean color of its (non-
           outlier) 8-neighbors
    a_all  ALL outlier patches   <- same neighbor-mean occluder
    b      primary outlier patch <- Gaussian noise matched to image mean/std
    c      control: random non-outlier, non-outlier-adjacent, below-median-
           relevance patch <- neighbor-mean occluder
    d      control: top-LRP-relevance patch (non-outlier, non-adjacent)
           <- neighbor-mean occluder
* Relocation guard: for condition (a) re-run the norm maps with the CLEAN
  thresholds; relocation = outlier at a position not in the clean outlier set.
* Faithfulness: per-image LRP (cp_lrp_baseline, condition y=target) patch
  relevance; outlier-patch |R| mass fraction vs measured delta-p.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "data" / "results" / "registers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCAN_NPZ = OUT_DIR / "step3_scan.npz"
SEL_NPZ = OUT_DIR / "step3_selection.npz"
LRP_NPZ = OUT_DIR / "step3_lrp.npz"
OCC_NPZ = OUT_DIR / "step3_occlusion.npz"

SEED = 0
N_SCAN = 1024          # scan-pool size for norm statistics + selection
N_SELECT = 128
GRID = 14              # 14x14 patch grid
PATCH = 16             # 16px patches
N_BLOCKS = 12
SIGMA_K = 4.0          # mean + 4*sd criterion


# ─────────────────────────────────────────────────────────────────────────────
# shared loading
# ─────────────────────────────────────────────────────────────────────────────

def load_everything(device: str):
    from experiments.models import ImagenetViTBase, backbone_transforms
    from experiments.datasets import load_eval_dataset
    model = ImagenetViTBase(device=device)
    label = f"vit_base · {model.head_name} · imagenet"
    transform, normalize = backbone_transforms(model.backbone)
    ds = load_eval_dataset("imagenet", transform, {"n_per_class": 10})
    print(f"model: {label}  dataset: {len(ds)} images")
    return model, ds, normalize


def scan_order(n: int) -> np.ndarray:
    """Fixed shuffled order over the dataset (the loader is ordered by class)."""
    return np.random.default_rng(SEED).permutation(n)


class NormHooks:
    """Forward hooks capturing per-block patch-token L2 norms (CLS excluded)."""

    def __init__(self, model, blocks):
        self.norms = {}
        self.handles = []
        for b in blocks:
            def hook(mod, inp, out, b=b):
                self.norms[b] = out[:, 1:, :].norm(dim=-1).detach()
            self.handles.append(model.backbone.blocks[b].register_forward_hook(hook))

    def pop(self):
        n, self.norms = self.norms, {}
        return n

    def remove(self):
        for h in self.handles:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# phase 1: scan — norms + predictions over the pool
# ─────────────────────────────────────────────────────────────────────────────

def cmd_scan(device: str):
    model, ds, normalize = load_everything(device)
    order = scan_order(len(ds))[:N_SCAN]
    hooks = NormHooks(model, range(N_BLOCKS))

    all_norms, all_pred, all_tgt, all_prob = [], [], [], []
    bs = 64
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            x = torch.stack([ds[int(i)][0] for i in idx]).to(device)
            y = torch.tensor([ds[int(i)][1] for i in idx])
            logits = model(normalize(x))
            prob = logits.softmax(-1)
            n = hooks.pop()
            all_norms.append(torch.stack([n[b] for b in range(N_BLOCKS)], 1).half().cpu())
            all_pred.append(logits.argmax(-1).cpu())
            all_tgt.append(y)
            all_prob.append(prob[torch.arange(len(y)), y].cpu())
            print(f"  scanned {s + len(idx)}/{len(order)}")
    hooks.remove()

    norms = torch.cat(all_norms).numpy()               # (N, 12, 196) fp16
    np.savez_compressed(
        SCAN_NPZ, ds_idx=order, norms=norms,
        pred=torch.cat(all_pred).numpy(), target=torch.cat(all_tgt).numpy(),
        prob=torch.cat(all_prob).numpy())
    print(f"saved {SCAN_NPZ}")


# ─────────────────────────────────────────────────────────────────────────────
# phase 2 (CPU): stats + selection
# ─────────────────────────────────────────────────────────────────────────────

def block_stats(norms: np.ndarray):
    """Per-block pool stats + mean+4sd thresholds. norms: (N, 12, 196)."""
    flat = norms.astype(np.float32).reshape(norms.shape[0], N_BLOCKS, -1)
    mean = flat.mean(axis=(0, 2))
    sd = flat.std(axis=(0, 2))
    thr = mean + SIGMA_K * sd
    out_frac = (flat > thr[None, :, None]).mean(axis=(0, 2))
    img_frac = (flat > thr[None, :, None]).any(axis=2).mean(axis=0)
    p999 = np.percentile(flat, 99.9, axis=(0, 2))
    med = np.median(flat, axis=(0, 2))
    return mean, sd, thr, out_frac, img_frac, p999, med


def cmd_select(blocks_arg: str):
    d = np.load(SCAN_NPZ)
    norms = d["norms"]
    mean, sd, thr, out_frac, img_frac, p999, med = block_stats(norms)
    print(f"{'blk':>3} {'mean':>7} {'sd':>7} {'thr':>7} {'p99.9':>8} {'p99.9/med':>9} "
          f"{'tok>thr%':>8} {'img>thr%':>8}")
    for b in range(N_BLOCKS):
        print(f"{b:>3} {mean[b]:>7.1f} {sd[b]:>7.1f} {thr[b]:>7.1f} {p999[b]:>8.1f} "
              f"{p999[b] / med[b]:>9.2f} {100 * out_frac[b]:>7.3f}% {100 * img_frac[b]:>7.1f}%")
    if blocks_arg == "auto":
        print("\nrun again with explicit --blocks after inspecting the table")
        return
    blocks = [int(b) for b in blocks_arg.split(",")]
    print(f"detection blocks: {blocks}")

    flat = norms.astype(np.float32)                    # (N, 12, 196)
    out_mask = np.zeros((flat.shape[0], GRID * GRID), bool)
    peak = np.zeros((flat.shape[0], GRID * GRID), np.float32)
    for b in blocks:
        out_mask |= flat[:, b] > thr[b]
        peak = np.maximum(peak, flat[:, b] / thr[b])   # threshold-relative peak
    correct = d["pred"] == d["target"]
    has_out = out_mask.any(1)
    eligible = np.where(correct & has_out)[0]
    print(f"pool {len(flat)}: correct {correct.mean():.1%}, with-outliers "
          f"{has_out.mean():.1%}, eligible {len(eligible)}")
    sel = eligible[:N_SELECT]
    if len(sel) < N_SELECT:
        print(f"WARNING only {len(sel)} eligible")
    primary = np.array([np.flatnonzero(out_mask[i])[np.argmax(peak[i][out_mask[i]])]
                        for i in sel])
    np.savez_compressed(
        SEL_NPZ, scan_row=sel, ds_idx=d["ds_idx"][sel], target=d["target"][sel],
        prob_clean_scan=d["prob"][sel], out_mask=out_mask[sel], primary=primary,
        n_outliers=out_mask[sel].sum(1), blocks=np.array(blocks), thr=thr,
        pool_mean=mean, pool_sd=sd)
    print(f"saved {SEL_NPZ}: {len(sel)} images, median outliers/img "
          f"{np.median(out_mask[sel].sum(1)):.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# phase 3 (GPU): per-image conditional LRP -> 14x14 patch relevance
# ─────────────────────────────────────────────────────────────────────────────

def cmd_lrp(device: str):
    from zennit_extensions.lrp_composites import CPLRPComposite
    from crp.attribution import CondAttribution
    model, ds, normalize = load_everything(device)
    sel = np.load(SEL_NPZ)
    attribution = CondAttribution(model)
    composite = CPLRPComposite()
    rel = np.zeros((len(sel["ds_idx"]), GRID, GRID), np.float32)
    for i, (di, t) in enumerate(zip(sel["ds_idx"], sel["target"])):
        x = ds[int(di)][0][None].to(device)
        xn = normalize(x).requires_grad_()
        attr = attribution(xn, [{"y": [int(t)]}], composite)
        h = attr.heatmap[0].detach().float().cpu().numpy()      # (224, 224)
        rel[i] = h.reshape(GRID, PATCH, GRID, PATCH).sum((1, 3))
        if (i + 1) % 16 == 0:
            print(f"  lrp {i + 1}/{len(rel)}")
    np.savez_compressed(LRP_NPZ, rel=rel)
    print(f"saved {LRP_NPZ}")


# ─────────────────────────────────────────────────────────────────────────────
# phase 4 (GPU): occlusion conditions + relocation guard
# ─────────────────────────────────────────────────────────────────────────────

def neighbors(p: int):
    r, c = divmod(p, GRID)
    return [nr * GRID + nc for nr in range(r - 1, r + 2) for nc in range(c - 1, c + 2)
            if (nr, nc) != (r, c) and 0 <= nr < GRID and 0 <= nc < GRID]


def patch_px(x: torch.Tensor, p: int) -> torch.Tensor:
    r, c = divmod(p, GRID)
    return x[:, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]


def fill_patch(x: torch.Tensor, p: int, val: torch.Tensor):
    r, c = divmod(p, GRID)
    x[:, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = val


def neighbor_mean_fill(x: torch.Tensor, p: int, exclude: set) -> torch.Tensor:
    """Constant fill colour = per-channel mean over valid (non-excluded)
    8-neighbor patches; falls back to whole-image mean."""
    nb = [q for q in neighbors(p) if q not in exclude]
    if not nb:
        return x.mean((1, 2), keepdim=True)
    vals = torch.stack([patch_px(x, q).mean((1, 2)) for q in nb])
    return vals.mean(0)[:, None, None]


def cmd_occlude(device: str):
    model, ds, normalize = load_everything(device)
    sel = np.load(SEL_NPZ)
    rel = np.load(LRP_NPZ)["rel"].reshape(-1, GRID * GRID)
    blocks = [int(b) for b in sel["blocks"]]
    thr = sel["thr"]
    hooks = NormHooks(model, blocks)
    conds = ["clean", "a", "a_all", "b", "c", "d"]
    N = len(sel["ds_idx"])
    probs = {k: np.zeros(N, np.float32) for k in conds}
    preds = {k: np.zeros(N, np.int64) for k in conds}
    ctrl_patch = np.zeros((N, 2), np.int64)            # chosen (c, d) patches
    reloc_mask = np.zeros((N, GRID * GRID), bool)      # condition-a outlier map
    torch.manual_seed(SEED)

    for i in range(N):
        di, t = int(sel["ds_idx"][i]), int(sel["target"][i])
        x = ds[di][0]                                   # (3, 224, 224) in [0,1]
        omask = sel["out_mask"][i]
        oset = set(np.flatnonzero(omask).tolist())
        adj = set().union(*[set(neighbors(q)) for q in oset]) | oset
        p0 = int(sel["primary"][i])

        variants = {"clean": x}
        xa = x.clone(); fill_patch(xa, p0, neighbor_mean_fill(x, p0, oset))
        variants["a"] = xa
        xaa = x.clone()
        for q in oset:
            fill_patch(xaa, q, neighbor_mean_fill(x, q, oset))
        variants["a_all"] = xaa
        g = torch.Generator().manual_seed(SEED + i)
        noise = x.mean((1, 2))[:, None, None] + x.std((1, 2))[:, None, None] * \
            torch.randn((3, PATCH, PATCH), generator=g)
        xb = x.clone(); fill_patch(xb, p0, noise.clamp(0, 1))
        variants["b"] = xb
        # control c: random non-outlier, non-adjacent, below-median-relevance
        r = np.abs(rel[i])
        cand = [q for q in range(GRID * GRID) if q not in adj and r[q] <= np.median(r)]
        pc = int(np.random.default_rng(1000 + i).choice(cand))
        xc = x.clone(); fill_patch(xc, pc, neighbor_mean_fill(x, pc, oset))
        variants["c"] = xc
        # control d: top-relevance patch outside the outlier neighborhood
        candd = [q for q in range(GRID * GRID) if q not in adj]
        pd_ = int(candd[int(np.argmax(r[candd]))])
        xd = x.clone(); fill_patch(xd, pd_, neighbor_mean_fill(x, pd_, oset))
        variants["d"] = xd
        ctrl_patch[i] = (pc, pd_)

        xb_ = torch.stack([variants[k] for k in conds]).to(device)
        with torch.no_grad():
            logits = model(normalize(xb_))
            pr = logits.softmax(-1)
        n = hooks.pop()
        a_row = conds.index("a")
        occ_out = np.zeros(GRID * GRID, bool)
        for b in blocks:
            occ_out |= (n[b][a_row].float().cpu().numpy() > thr[b])
        reloc_mask[i] = occ_out
        for j, k in enumerate(conds):
            probs[k][i] = pr[j, t].item()
            preds[k][i] = int(logits[j].argmax().item())
        if (i + 1) % 16 == 0:
            print(f"  occluded {i + 1}/{N}")
    hooks.remove()

    np.savez_compressed(
        OCC_NPZ, conds=np.array(conds), ctrl_patch=ctrl_patch,
        reloc_mask=reloc_mask,
        **{f"prob_{k}": probs[k] for k in conds},
        **{f"pred_{k}": preds[k] for k in conds})
    print(f"saved {OCC_NPZ}")
    for k in conds[1:]:
        dp = probs[k] - probs["clean"]
        print(f"  {k:>5}: median dp {np.median(dp):+.4f}  mean {dp.mean():+.4f}  "
              f"pred-preserved {(preds[k] == sel['target']).mean():.1%}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["scan", "select", "lrp", "occlude"])
    ap.add_argument("--blocks", default="auto")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.phase == "scan":
        cmd_scan(a.device)
    elif a.phase == "select":
        cmd_select(a.blocks)
    elif a.phase == "lrp":
        cmd_lrp(a.device)
    elif a.phase == "occlude":
        cmd_occlude(a.device)


if __name__ == "__main__":
    main()
