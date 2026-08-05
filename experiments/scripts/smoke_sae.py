"""Smoke test for the SAE splice + SAE-basis CRP wiring.

Trains a quick SAE on one block's proj_drop activations (dsprites probe),
splices it in, and checks:
  1. reconstruction faithfulness — splice output â ≈ a (low FVU);
  2. relevance conservation — Σ R(SAE features f) ≈ R at the splice output
     (the decoder's γ-rule should conserve), the property that makes this CRP
     rather than gradient attribution.

Run: uv run python -m experiments.scripts.smoke_sae
"""
from __future__ import annotations

import torch

from experiments.concept_flipping import load_probe, DATASETS, REPO_ROOT
from experiments import sae as sae_mod
from crp.attribution import CondAttribution
from crp.concepts import EmbeddingDimConcept
from zennit_extensions.lrp_composites import CPLRPComposite
from timm.data import resolve_data_config, create_transform
from experiments.datasets import load as load_dataset

BLOCK = 11
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    model, ck, _ = load_probe("dsprites", DEV)
    mods = sae_mod.site_modules(model, "proj_drop")
    d = model.backbone.embed_dim
    transform = create_transform(**resolve_data_config({}, model=model.backbone), is_training=False)
    ds = load_dataset("dsprites", root=REPO_ROOT / "data", transform=transform, target="shape")

    print("collecting activations…")
    acts = sae_mod.collect_activations(model, ds, mods, n_images=400, device=DEV)
    print(f"acts[{BLOCK}] shape {tuple(acts[BLOCK].shape)}")

    print("training quick SAE (block 11, 1200 steps)…")
    raw, metrics = sae_mod.train_sae(acts[BLOCK], m=8 * d, l1_coeff=1e-3, steps=1200,
                                     lr=1e-3, batch=4096, device=DEV, resample_every=600)
    print(f"  FVU={metrics['fvu']:.4f}  L0={metrics['l0']:.1f}  dead={metrics['dead']}/{8*d}")

    # one image, target class 0
    x = ds[0][0].unsqueeze(0).to(DEV)

    # capture the RAW site activation before splicing (recon-fidelity reference)
    a0 = {}
    h = mods[BLOCK].register_forward_hook(lambda m, i, o: a0.__setitem__(0, o.detach()))
    with torch.no_grad():
        model(x)
    h.remove()

    # splice
    splice = sae_mod.SAESplice(raw, inner=None).to(DEV).eval()
    model.backbone.blocks[BLOCK].attn.proj_drop = splice
    rec_feat = f"backbone.blocks.{BLOCK}.attn.proj_drop.features"
    rec_out = f"backbone.blocks.{BLOCK}.attn.proj_drop"

    with torch.no_grad():
        ahat = splice(a0[0])
    rel_err = (ahat - a0[0]).norm().item() / (a0[0].norm().item() + 1e-12)
    print(f"splice recon on image: ||â−a||/||a|| = {rel_err:.4f}")

    composite = CPLRPComposite()
    attribution = CondAttribution(model)
    xg = x.clone().requires_grad_(True)
    res = attribution(xg, [{"y": [0]}], composite, record_layer=[rec_feat, rec_out])
    Rf = res.relevances[rec_feat]   # (1, N, m)
    Ro = res.relevances[rec_out]    # (1, N, d)
    sum_f = Rf.sum().item()
    sum_o = Ro.sum().item()
    print(f"\nconservation:  Σ R(features) = {sum_f:+.4f}   Σ R(splice out) = {sum_o:+.4f}   "
          f"ratio = {sum_f / (sum_o + 1e-12):.4f}")

    # per-latent attribution sanity
    conc = EmbeddingDimConcept(num_heads=6)
    rel = conc.attribute(Rf, abs_norm=False)[0]
    print(f"per-latent relevance: n={rel.numel()}  nonzero={(rel.abs()>1e-9).sum().item()}  "
          f"top|R|={rel.abs().max().item():.4f}")
    print("\nSMOKE_OK")


if __name__ == "__main__":
    main()
