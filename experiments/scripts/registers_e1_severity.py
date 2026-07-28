"""E1 empirical severity: per-image extreme-norm ratio from stored E1 norms.

rho_i = max_t ||h(t)|| / median_t ||h(t)||  over PATCH tokens, site-max per
image. Threshold-free signature of the scratch-pad phenomenon (Darcet et al.:
standard large ViTs show order-of-magnitude high-norm tokens). Also the
register-carried ratio for DINOv3 (max over the 4 register tokens / patch
median) — shows where the extreme norm goes instead.

Reads the E1 count arrays (norms per site x image x token); no GPU. Writes
data/results/registers/e1_severity.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "registers"
MODELS = [
    ("m1", "ViT-S std / FB", "e1_counts_m1_vit_small_fb"),
    ("m2", "ViT-B std / IN", "e1_counts_vit_base_imagenet"),
    ("m3", "DINOv3-S ft / FB", "e1_counts_m3_dinov3s_fb"),
    ("m4", "DINOv3-B / IN", "e1_counts_dinov3_base_imagenet"),
]


def _stats(v):
    return {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)), "max": float(v.max())}


def main():
    out = {}
    for tag, label, fname in MODELS:
        d = np.load(RES / f"{fname}.npz", allow_pickle=True)
        norms = d["norms"]                       # (n_sites, n_img, n_tok)
        n_prefix, n_reg = int(d["n_prefix"]), int(d["n_reg"])
        patch = norms[:, :, n_prefix:]           # patch tokens only
        med = np.median(patch, axis=2)
        rho = patch.max(axis=2) / med            # (S, N) per site per image
        rho_img = rho.max(axis=0)                # site-max per image
        rec = {"label": label, "n_img": int(norms.shape[1]),
               "patch_extreme_ratio": _stats(rho_img)}
        if n_reg > 0:
            reg = norms[:, :, 1:1 + n_reg]       # register tokens (after cls)
            reg_ratio = (reg.max(axis=2) / med).max(axis=0)
            rec["register_extreme_ratio"] = _stats(reg_ratio)
        out[tag] = rec
    (RES / "e1_severity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
