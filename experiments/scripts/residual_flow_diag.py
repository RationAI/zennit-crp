"""Residual-flow diagnostic: how LRP relevance splits between SKIP and BRANCH.

For each transformer block of a finetuned ViT probe, the canonized block forward
(:class:`zennit_ext.TimmBlockResidualCanonizer`) routes both residual additions
through recordable :class:`~zennit_ext.attention_unfolded.ResidualAdd` modules::

    out1 = _lrp_res1(x,   drop_path1(ls1(attn(norm1(x)))))   # attn residual
    out2 = _lrp_res2(out1, drop_path2(ls2(mlp(norm2(out1)))))  # mlp residual

At attribution time the composite's residual rule (``ResidualRatio`` for
``cp_lrp_baseline``) splits the relevance arriving at each add output
*element-wise* between the skip operand and the branch operand. This script
records, per block ``b`` and per residual site:

* ``R_add``    — relevance at the add output (``backbone.blocks.b._lrp_res1`` /
  ``backbone.blocks.b._lrp_res2``), shape (B, N_tokens, D);
* ``R_branch`` — relevance at the branch endpoint
  (``backbone.blocks.b.attn.proj_drop`` for the attention branch,
  ``backbone.blocks.b.mlp.drop2`` for the MLP branch). Between these endpoints
  and the add's branch input sit only Identity (``ls*``, ``drop_path*``) and
  eval-mode Dropout modules, so the recorded gradient IS the branch summand of
  the elementwise split (verified numerically: ``max|R(proj_drop)-R(ls1)| = 0``);
* ``R_skip``   — derived as ``R_add - R_branch`` (exact, because the residual
  rule splits elementwise: R_add = R_to_skip + R_to_branch per token per dim).

Per embedding dimension d we store token-summed signed and absolute relevance
for both paths, per sample, and the branch fraction
``f = |R_branch| / (|R_branch| + |R_skip|)`` (on absolute token sums).

Conservation: the skip/branch decomposition is exact by construction of the
rule; the reported conservation error is the *propagation* drift of total
relevance between consecutive cuts of the network (Gamma-rule bias absorption),
i.e. |sum R(add1_b) - sum R(add2_b)| and |sum R(add2_{b-1}) - sum R(add1_b)|
relative to the block-output total.

Usage (repo root on PYTHONPATH)::

    python -m experiments.scripts.residual_flow_diag compute --n-samples 96
    python -m experiments.scripts.residual_flow_diag render
    python -m experiments.scripts.residual_flow_diag all --n-samples 96
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ_DIR = REPO_ROOT / "data" / "results" / "residual_flow"
DEFAULT_WEB_DIR = REPO_ROOT / "webapp" / "residual_flow"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def npz_path(base: str, dataset: str, config: str, out_dir: Path) -> Path:
    return out_dir / f"residual_flow_{base}_{dataset}_{config}.npz"


# ─────────────────────────────────────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────────────────────────────────────

def pick_class_diverse(ds, n: int, seed: int = 0) -> List[int]:
    """Round-robin over classes (like ``crp_gallery.pick_samples``) so the
    sample is class-diverse even when n < n_classes * per-class count."""
    if hasattr(ds, "items"):
        labels = [int(c) for _, c in ds.items]
    elif hasattr(ds, "rows"):
        labels = [int(c) for _, c in ds.rows]
    else:
        labels = [int(ds[i][1]) for i in range(len(ds))]
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for i in rng.permutation(len(labels)):
        by_class.setdefault(labels[i], []).append(int(i))
    classes = sorted(by_class)
    out: List[int] = []
    while len(out) < n and any(by_class[c] for c in classes):
        for c in classes:
            if by_class[c]:
                out.append(by_class[c].pop(0))
                if len(out) >= n:
                    break
    return out


def compute(args) -> Path:
    import torch
    import lrp_configs
    from crp.attribution import CondAttribution
    from experiments.model_io import DATASETS, load_probe, backbone_transforms
    from experiments.datasets import load as load_dataset

    device = args.device
    tag = DATASETS[args.dataset][2]
    model, ck, ck_path = load_probe(tag, device, base=args.base)
    n_blocks = len(model.backbone.blocks)
    embed_dim = int(model.backbone.embed_dim)
    transform, normalize = backbone_transforms(model.backbone)
    # Held-out eval images: the FunnyBirds *test* split (the dataset's validation
    # split; it contains zero part-ablated images, so no clean_only filter needed).
    ds_name, ds_kw, _ = DATASETS[args.dataset]
    ds_kw = dict(ds_kw)
    if args.dataset == "funny_birds":
        ds_kw = {"split": "test"}
    ds = load_dataset(ds_name, root=REPO_ROOT / "data", transform=transform, **ds_kw)
    idxs = pick_class_diverse(ds, args.n_samples, seed=args.seed)
    print(f"{len(idxs)} class-diverse samples (round-robin), "
          f"model={ck['base']}·{ck['head']}, D={embed_dim}")

    cfg = lrp_configs.get(args.config)
    attribution = CondAttribution(model)

    # (block, kind) → (add layer, branch layer); network order: attn then mlp.
    sites = []
    for b in range(n_blocks):
        sites.append((b, "attn", f"backbone.blocks.{b}._lrp_res1",
                      f"backbone.blocks.{b}.attn.proj_drop"))
        sites.append((b, "mlp", f"backbone.blocks.{b}._lrp_res2",
                      f"backbone.blocks.{b}.mlp.drop2"))
    record = sorted({l for _, _, a, br in sites for l in (a, br)})
    check_layers = ["backbone.blocks.0.ls1", f"backbone.blocks.{n_blocks - 1}.ls1"]

    S, n_sites = len(idxs), len(sites)
    branch_signed = np.zeros((n_sites, S, embed_dim), np.float32)
    skip_signed = np.zeros_like(branch_signed)
    branch_abs = np.zeros_like(branch_signed)
    skip_abs = np.zeros_like(branch_signed)
    tot_add = np.zeros((n_sites, S), np.float32)          # sum tokens+dims at add out
    sample_target = np.zeros(S, np.int64)
    sample_pred = np.zeros(S, np.int64)
    sample_logit = np.zeros(S, np.float32)
    endpoint_err = 0.0                                     # max|R(proj_drop)-R(ls1)|

    bs = args.batch_size
    for i0 in range(0, S, bs):
        chunk = idxs[i0:i0 + bs]
        xs, ys = zip(*[(ds[i][0], int(ds[i][1])) for i in chunk])
        x = torch.stack(list(xs)).to(device)
        xin = normalize(x).requires_grad_(True)
        conds = [{"y": [y]} for y in ys]
        rec = record + (check_layers if i0 == 0 else [])
        res = attribution(xin, conds, cfg.composite(), record_layer=rec)
        missing = [l for l in record if l not in res.relevances]
        if missing:
            raise RuntimeError(f"recording failed for layers: {missing}")
        if i0 == 0:
            e1 = (res.relevances["backbone.blocks.0.attn.proj_drop"]
                  - res.relevances["backbone.blocks.0.ls1"]).abs().max()
            e2 = (res.relevances[f"backbone.blocks.{n_blocks-1}.attn.proj_drop"]
                  - res.relevances[f"backbone.blocks.{n_blocks-1}.ls1"]).abs().max()
            endpoint_err = float(torch.maximum(e1, e2))
        pred = res.prediction.detach()
        for j, y in enumerate(ys):
            k = i0 + j
            sample_target[k] = y
            sample_pred[k] = int(pred[j].argmax())
            sample_logit[k] = float(pred[j, y])
        for si, (b, kind, add_l, br_l) in enumerate(sites):
            r_add = res.relevances[add_l]                  # (B, N, D)
            r_br = res.relevances[br_l]
            r_skip = r_add - r_br                          # exact elementwise split
            branch_signed[si, i0:i0 + len(chunk)] = r_br.sum(1).cpu().numpy()
            skip_signed[si, i0:i0 + len(chunk)] = r_skip.sum(1).cpu().numpy()
            branch_abs[si, i0:i0 + len(chunk)] = r_br.abs().sum(1).cpu().numpy()
            skip_abs[si, i0:i0 + len(chunk)] = r_skip.abs().sum(1).cpu().numpy()
            tot_add[si, i0:i0 + len(chunk)] = r_add.sum((1, 2)).cpu().numpy()
        print(f"  batch {i0 // bs + 1}/{(S + bs - 1) // bs} done", flush=True)

    # Propagation (conservation) drift between consecutive cuts, per sample:
    # within block (add2 → add1, through skip+MLP), and across the attn side
    # (add2 of block b-1 → add1 of block b). Relative to the final-block total.
    ref = np.abs(tot_add[-1])                              # (S,)
    t1 = tot_add[0::2]                                     # add1 per block  (n_blocks, S)
    t2 = tot_add[1::2]                                     # add2 per block
    drift_mlp = np.abs(t1 - t2) / ref                      # within-block
    drift_attn = np.abs(t2[:-1] - t1[1:]) / ref            # across blocks
    meta = {
        "base": args.base, "dataset": args.dataset, "config": args.config,
        "split": "test" if args.dataset == "funny_birds" else str(ds_kw),
        "checkpoint": str(ck_path), "n_samples": S, "n_blocks": n_blocks,
        "embed_dim": embed_dim, "seed": args.seed,
        "composite_desc": cfg.description,
        "endpoint_identity_err": endpoint_err,
        "accuracy_on_sample": float((sample_pred == sample_target).mean()),
        "drift_mlp_median": float(np.median(drift_mlp)),
        "drift_mlp_max": float(drift_mlp.max()),
        "drift_attn_median": float(np.median(drift_attn)),
        "drift_attn_max": float(drift_attn.max()),
        "generated": _now(),
    }
    out = npz_path(args.base, args.dataset, args.config, Path(args.out_dir))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        branch_signed=branch_signed, skip_signed=skip_signed,
        branch_abs=branch_abs, skip_abs=skip_abs, tot_add=tot_add,
        site_block=np.array([b for b, _, _, _ in sites], np.int64),
        site_kind=np.array([k for _, k, _, _ in sites]),
        site_add_layer=np.array([a for _, _, a, _ in sites]),
        site_branch_layer=np.array([br for _, _, _, br in sites]),
        sample_ds_index=np.array(idxs, np.int64),
        sample_target=sample_target, sample_pred=sample_pred,
        sample_logit=sample_logit,
        meta=np.array(json.dumps(meta)),
    )
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta, indent=2))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Render (static, fully self-contained Bokeh page)
# ─────────────────────────────────────────────────────────────────────────────

# Reference dataviz palette (documented validated instance): diverging
# blue ↔ neutral-gray ↔ red; chrome/ink tokens.
C_SKIP, C_MID, C_BRANCH = "#2a78d6", "#f0efec", "#e34948"
SURFACE, PLANE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def diverging_hex(vals: np.ndarray) -> List[str]:
    """Map values in [0,1] onto the blue↔gray↔red diverging ramp (0.5 = gray)."""
    lo, mid, hi = _hex2rgb(C_SKIP), _hex2rgb(C_MID), _hex2rgb(C_BRANCH)
    v = np.clip(vals, 0, 1)
    t = np.abs(v - 0.5) * 2
    pole = np.where(v[..., None] < 0.5, lo, hi)
    rgb = (mid * (1 - t[..., None]) + pole * t[..., None]).round().astype(int)
    return ["#%02x%02x%02x" % tuple(c) for c in rgb.reshape(-1, 3)]


def _style(fig):
    fig.background_fill_color = SURFACE
    fig.border_fill_color = PLANE
    fig.outline_line_color = None
    for ax in (fig.xaxis, fig.yaxis):
        ax.axis_line_color = BASE
        ax.major_tick_line_color = BASE
        ax.minor_tick_line_color = None
        ax.major_label_text_color = MUTED
        ax.axis_label_text_color = INK2
    fig.grid.grid_line_color = None
    fig.title.text_color = INK2
    fig.title.text_font_size = "12px"
    fig.toolbar.logo = None


def render(args) -> Path:
    from bokeh.embed import file_html
    from bokeh.layouts import column, row
    from bokeh.models import (ColumnDataSource, CustomJS, Div, HoverTool,
                              Range1d, TabPanel, Tabs)
    from bokeh.plotting import figure
    from bokeh.resources import INLINE

    src_npz = npz_path(args.base, args.dataset, args.config, Path(args.out_dir))
    z = np.load(src_npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    ba, sa = z["branch_abs"], z["skip_abs"]                # (24, S, D)
    bs_, ss_ = z["branch_signed"], z["skip_signed"]
    site_block, site_kind = z["site_block"], z["site_kind"]
    n_sites, S, D = ba.shape
    site_labels = [f"block {b} · {k}" for b, k in zip(site_block, site_kind)]

    f = ba / (ba + sa + 1e-12)                             # (24, S, D)
    med = np.median(f, axis=1)                             # (24, D)
    mean = f.mean(axis=1)
    q25, q75 = np.percentile(f, [25, 75], axis=1)
    iqr = q75 - q25
    med_bsig = np.median(bs_, axis=1)
    med_ssig = np.median(ss_, axis=1)

    # Histogram of f over samples, per cell — feeds the hover-linked histogram.
    nbins = 16
    bin_idx = np.clip((f * nbins).astype(int), 0, nbins - 1)     # (24, S, D)
    hist = np.zeros((n_sites, D, nbins), np.int32)
    for si in range(n_sites):
        for bi in range(nbins):
            hist[si, :, bi] = (bin_idx[si] == bi).sum(axis=0)

    order_sorted = np.argsort(-mean.mean(axis=0))          # dims by mean f, desc
    x_of_dim_sorted = np.empty(D, int)
    x_of_dim_sorted[order_sorted] = np.arange(D)

    ys, dims = np.mgrid[0:n_sites, 0:D]
    ys, dims = ys.ravel(), dims.ravel()
    alpha = 0.25 + 0.75 * (1 - np.clip(iqr / 0.5, 0, 1))
    cell = ColumnDataSource(dict(
        x_idx=dims.astype(float), x_sort=x_of_dim_sorted[dims].astype(float),
        y=ys.astype(float), dim=dims,
        site=[site_labels[i] for i in ys],
        color=diverging_hex(med.ravel()), alpha=alpha.ravel(),
        median=med.ravel().round(3), mean=mean.ravel().round(3),
        iqr=iqr.ravel().round(3),
        bsig=med_bsig.ravel().round(4), ssig=med_ssig.ravel().round(4),
        hist=[hist[y, d].tolist() for y, d in zip(ys, dims)],
    ))

    hist_src = ColumnDataSource(dict(
        left=(np.arange(nbins) / nbins).tolist(),
        right=((np.arange(nbins) + 1) / nbins).tolist(),
        top=[0] * nbins,
    ))
    hist_title = Div(text="<i>hover a cell to see its f-distribution</i>",
                     styles={"color": INK2, "font-size": "12px"})
    hover_js = CustomJS(args=dict(src=cell, hsrc=hist_src, title=hist_title), code="""
        const inds = cb_data.index.indices;
        if (inds.length == 0) { return; }
        const i = inds[0];
        hsrc.data = {left: hsrc.data.left, right: hsrc.data.right,
                     top: src.data.hist[i]};
        title.text = "<b>" + src.data.site[i] + " · dim " + src.data.dim[i] +
            "</b> — median f = " + Number(src.data.median[i]).toFixed(3) +
            ", IQR = " + Number(src.data.iqr[i]).toFixed(3);
        hsrc.change.emit();
    """)

    tooltips = [("site", "@site"), ("dim", "@dim"),
                ("median f", "@median"), ("mean f", "@mean"), ("IQR", "@iqr"),
                ("median signed R(branch)", "@bsig"),
                ("median signed R(skip)", "@ssig")]

    # Row marginals: mean over samples of total |R| per path per site.
    tot_b = ba.sum(axis=2).mean(axis=1)                    # (24,)
    tot_s = sa.sum(axis=2).mean(axis=1)

    def heat_tab(xfield: str, title: str) -> TabPanel:
        p = figure(width=1180, height=620, title=title,
                   x_range=Range1d(-0.5, D - 0.5),
                   y_range=Range1d(n_sites - 0.5, -0.5),
                   tools="hover,pan,box_zoom,wheel_zoom,reset",
                   active_scroll=None)
        r = p.rect(x=xfield, y="y", width=1, height=1, source=cell,
                   fill_color="color", fill_alpha="alpha", line_color=None)
        hv = p.select_one(HoverTool)
        hv.tooltips = tooltips
        hv.renderers = [r]
        hv.callback = hover_js
        p.yaxis.ticker = list(range(n_sites))
        p.yaxis.major_label_overrides = {i: l for i, l in enumerate(site_labels)}
        p.xaxis.axis_label = ("dims sorted by mean branch fraction (desc)"
                              if xfield == "x_sort" else "embedding dim index")
        _style(p)
        m = figure(width=240, height=620, title="mean Σ_dims |R| per path",
                   y_range=p.y_range, tools="")
        m.hbar(y=np.arange(n_sites) - 0.18, height=0.32, right=tot_b,
               fill_color=C_BRANCH, line_color=None, legend_label="branch")
        m.hbar(y=np.arange(n_sites) + 0.18, height=0.32, right=tot_s,
               fill_color=C_SKIP, line_color=None, legend_label="skip")
        m.yaxis.visible = False
        m.legend.location = "top_right"
        m.legend.label_text_color = INK2
        m.legend.label_text_font_size = "11px"
        m.legend.background_fill_alpha = 0.6
        _style(m)
        return TabPanel(child=row(p, m), title=title)

    tabs = Tabs(tabs=[
        heat_tab("x_sort", "dims sorted by branch fraction"),
        heat_tab("x_idx", "dims by index"),
    ])

    # Hover-linked histogram of f over the samples for one cell.
    ph = figure(width=430, height=260, title="branch fraction f — per-sample histogram",
                x_range=Range1d(0, 1), tools="")
    ph.quad(left="left", right="right", bottom=0, top="top", source=hist_src,
            fill_color=C_BRANCH, fill_alpha=0.75, line_color=SURFACE, line_width=2)
    ph.xaxis.axis_label = "f = |R_branch| / (|R_branch| + |R_skip|)"
    ph.yaxis.axis_label = f"images (of {S})"
    ph.y_range.start = 0
    _style(ph)

    # Summary: per site, distribution over dims of median-f (box + whiskers).
    q = np.percentile(med, [5, 25, 50, 75, 95], axis=1)    # (5, 24)
    ps = figure(width=560, height=620, title="per-site distribution over dims of median f",
                x_range=Range1d(0, 1), y_range=Range1d(n_sites - 0.5, -0.5), tools="")
    yy = np.arange(n_sites)
    ps.segment(q[0], yy, q[4], yy, line_color=BASE, line_width=2)
    ps.hbar(y=yy, height=0.55, left=q[1], right=q[3],
            fill_color=diverging_hex(q[2]), line_color=SURFACE, line_width=1)
    ps.segment(q[2], yy - 0.32, q[2], yy + 0.32, line_color=INK, line_width=2)
    ps.yaxis.ticker = list(range(n_sites))
    ps.yaxis.major_label_overrides = {i: l for i, l in enumerate(site_labels)}
    ps.xaxis.axis_label = "median branch fraction f (5–95% whiskers, IQR box, median tick)"
    _style(ps)

    header = Div(text=f"""
      <h1 style="font-family:system-ui;color:{INK};margin:0 0 4px">Residual skip vs branch — LRP relevance flow</h1>
      <p style="font-family:system-ui;color:{INK2};max-width:1100px;margin:0">
      Per embedding dimension and per residual site of
      <b>{meta['base']}</b> on <b>{meta['dataset']}</b> ({meta['split']} split,
      N={meta['n_samples']} class-diverse images), composite
      <b>{meta['config']}</b>: the fraction of LRP relevance the residual rule
      routes through the <span style="color:{C_BRANCH}"><b>branch</b></span>
      (attention resp. MLP) vs the
      <span style="color:{C_SKIP}"><b>skip</b></span> path,
      f&nbsp;=&nbsp;|R<sub>branch</sub>| / (|R<sub>branch</sub>|+|R<sub>skip</sub>|)
      on token-summed absolute relevance. Cell color = median f over images
      (blue&nbsp;0&nbsp;=&nbsp;all&nbsp;skip · gray&nbsp;0.5 ·
      red&nbsp;1&nbsp;=&nbsp;all&nbsp;branch); cell opacity fades with the
      across-image IQR (solid = consistent, faint = unstable). Hover any cell
      for stats + its per-image histogram. Generated {meta['generated']}.
      </p>""")

    methods = Div(text=f"""
      <div style="font-family:system-ui;color:{INK2};max-width:1100px;font-size:13px">
      <h2 style="color:{INK};font-size:16px">Methods</h2>
      <p><b>Model.</b> Finetuned probe <code>{meta['checkpoint']}</code>
      ({meta['base']}, {meta['n_blocks']} blocks, D={meta['embed_dim']}), accuracy on
      the {meta['n_samples']}-image sample: {meta['accuracy_on_sample']:.3f}.
      Attribution: <code>crp.CondAttribution</code> conditioned on the true class
      (<code>{{"y": [target]}}</code>), composite <code>{meta['config']}</code>
      ("{meta['composite_desc']}").</p>
      <p><b>Recording.</b> The <code>TimmBlockResidualCanonizer</code> routes both
      residual additions of every block through recordable <code>ResidualAdd</code>
      modules. Per block b the recorded layers are the add outputs
      <code>backbone.blocks.b._lrp_res1</code> (attn) and
      <code>backbone.blocks.b._lrp_res2</code> (MLP), and the branch endpoints
      <code>backbone.blocks.b.attn.proj_drop</code> and
      <code>backbone.blocks.b.mlp.drop2</code>. Between each endpoint and the add's
      branch input sit only Identity (<code>ls*</code>, <code>drop_path*</code>) and
      eval-mode Dropout modules, so the endpoint gradient equals the branch summand
      of the residual rule's elementwise split — verified numerically:
      max |R(proj_drop) − R(ls1)| = {meta['endpoint_identity_err']:.2e}.</p>
      <p><b>Skip derivation.</b> The composite's residual rule
      (<code>ResidualRatio</code>) splits the relevance at the add output
      elementwise: R<sub>add</sub> = R<sub>skip</sub> + R<sub>branch</sub> per token
      per dimension. R<sub>skip</sub> is therefore derived exactly as
      R<sub>add</sub> − R<sub>branch</sub> (no approximation in the decomposition
      itself). Per dimension we sum over the 197 tokens — signed sums and
      absolute-value sums are both stored; f uses the absolute sums.</p>
      <p><b>Conservation.</b> The decomposition at each add is exact by construction;
      the residual (propagation) error is the drift of <i>total</i> relevance between
      consecutive network cuts, caused by the Gamma rule's bias absorption:
      within-block (add2 → add1, through skip + MLP) median
      {meta['drift_mlp_median']:.3f} / max {meta['drift_mlp_max']:.3f}; across the
      attention side (add2 of block b−1 → add1 of block b) median
      {meta['drift_attn_median']:.3f} / max {meta['drift_attn_max']:.3f}
      (relative to the final-block total). This drift is a property of the LRP
      recipe, not of the skip/branch split reported here.</p>
      <p><b>Data.</b> Raw arrays (signed + absolute token-summed relevance per
      site × image × dim, plus indices/targets/logits):
      <code>data/results/residual_flow/{src_npz.name}</code>.
      Script: <code>experiments/scripts/residual_flow_diag.py</code>.</p>
      </div>""")

    page = column(header, tabs, row(column(hist_title, ph), ps), methods,
                  styles={"background": PLANE, "padding": "18px"})
    html = file_html(page, resources=INLINE,
                     title="Residual skip vs branch — LRP relevance flow")
    out_dir = Path(args.webapp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict(base="vit_small", dataset="funny_birds", config="cp_lrp_baseline")
    for name in ("compute", "render", "all"):
        p = sub.add_parser(name)
        for k, v in common.items():
            p.add_argument(f"--{k}", default=v)
        p.add_argument("--out-dir", default=str(DEFAULT_NPZ_DIR))
        if name in ("compute", "all"):
            p.add_argument("--n-samples", type=int, default=96)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--device", default="cuda")
            p.add_argument("--seed", type=int, default=0)
        if name in ("render", "all"):
            p.add_argument("--webapp-dir", default=str(DEFAULT_WEB_DIR))
    args = ap.parse_args()
    if args.cmd in ("compute", "all"):
        compute(args)
    if args.cmd in ("render", "all"):
        render(args)


if __name__ == "__main__":
    main()
