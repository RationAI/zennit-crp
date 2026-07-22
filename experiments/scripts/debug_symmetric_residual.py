"""Debug: is `residual_lrp='symmetric'` a wiring no-op, or a global rescale?

For one image, attribute with the `attnlrp_gamma_residual_none` and
`attnlrp_gamma_residual_symmetric` composites and compare the per-block
embedding-dim relevance vectors at the proj_drop probe sites.

Hypotheses:
  (A) no-op / canonizer not firing  -> rel_sym == rel_none exactly.
  (B) global positive rescale        -> rel_sym == k_b * rel_none with k_b>0
                                        constant within a block; ranking identical.
If (B), the symmetric rule IS applied; the concept-flipping metric is just
invariant to it (ranking-based, scale-free).
"""
import numpy as np
import torch

import lrp_configs
from crp.attribution import CondAttribution
from experiments.concept_flipping import (
    load_probe, concept_detectors, DATASETS,
)
from timm.data import create_transform, resolve_data_config
from experiments.datasets import load as load_dataset
from experiments import concept_flipping as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEY = "dsprites"
CONCEPT = "embed_dim"


def rels_for(config_name, model, attribution, sites, concept, x, c):
    composite = lrp_configs.get(config_name).composite()
    xg = x.clone().requires_grad_(True)
    rel_layers = attribution(xg, [{"y": [c]}], composite, record_layer=sites).relevances
    out = {}
    for b, s in enumerate(sites):
        out[b] = concept.attribute(rel_layers[s], abs_norm=False)[0].detach().cpu().numpy()
    return out


def main():
    model, ck, _ = load_probe(DATASETS[KEY][2], DEVICE)
    n_blocks = len(model.backbone.blocks)
    sites = [f"backbone.blocks.{b}.attn.{lrp_configs.get('attnlrp_gamma_residual_none').site}"
             for b in range(n_blocks)]
    attribution = CondAttribution(model)
    concept, D, n_grid = concept_detectors(
        CONCEPT, model.backbone.blocks[0].attn.num_heads, model.backbone.embed_dim, DEVICE)

    transform = create_transform(**resolve_data_config({}, model=model.backbone), is_training=False)
    ds_name, ds_kw, _ = DATASETS[KEY]
    ds = load_dataset(ds_name, root=cf.REPO_ROOT / "data", transform=transform, **ds_kw)
    sel = cf.select_correct(model, ds, list(range(int(ck["num_classes"]))), 1, DEVICE)
    c = next(iter(sel))
    x = ds[sel[c][0]][0].unsqueeze(0).to(DEVICE)

    r_none = rels_for("attnlrp_gamma_residual_none", model, attribution, sites, concept, x, c)
    r_sym = rels_for("attnlrp_gamma_residual_symmetric", model, attribution, sites, concept, x, c)

    print(f"image class={c}  blocks={n_blocks}  embed_dim={D.shape[0]}")
    print(f"{'blk':>3} {'max|none|':>11} {'max|d|':>10} {'ratio sym/none':>16} "
          f"{'ratio std':>10} {'rank match':>11}")
    for b in range(n_blocks):
        a, s = r_none[b], r_sym[b]
        diff = np.abs(a - s).max()
        mask = np.abs(a) > 1e-12
        ratio = s[mask] / a[mask]
        rmean = ratio.mean() if mask.any() else float("nan")
        rstd = ratio.std() if mask.any() else float("nan")
        # ranking identical?
        rank_match = np.array_equal(np.argsort(a), np.argsort(s))
        print(f"{b:>3} {np.abs(a).max():11.4e} {diff:10.3e} {rmean:16.6f} "
              f"{rstd:10.2e} {str(rank_match):>11}")


if __name__ == "__main__":
    main()
