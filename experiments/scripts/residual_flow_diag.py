"""Residual-flow diagnostic: how LRP relevance splits between skip and branch.

Every canonized transformer block routes its two residual additions through
recordable ``ResidualAdd`` modules (``_lrp_res1`` attention-side, ``_lrp_res2``
MLP-side). The composite's residual rule splits the relevance arriving at each
add output *elementwise* between the skip and branch operands, so per residual
site and per embedding dimension::

    R_skip   = R_add - R_branch                        (exact split)
    f_branch = |R_branch| / (|R_branch| + |R_skip|)

Pipeline:

* ``compute`` — batched conditional LRP (true-class conditioning) over a
  class-diverse, correctly-classified sample; records token-summed (patch
  tokens only) signed/absolute relevance per site x sample x dim, plus the
  propagation-drift stats; writes ONE npz per model carrying that model's own
  residual-site list (``site_block`` / ``site_kind`` / layer names).
* ``render`` — turns the npz into a self-contained Bokeh page. Every row on
  the page is built from the site list stored in the npz, so the page always
  shows exactly the layers of the evaluated model — no fixed slot grid.

Usage (repo root on PYTHONPATH)::

    python -m experiments.scripts.residual_flow_diag compute --n-samples 96
    python -m experiments.scripts.residual_flow_diag render
    python -m experiments.scripts.residual_flow_diag all --n-samples 96
"""
from __future__ import annotations

import argparse
import html
import json
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ_DIR = REPO_ROOT / "data" / "results" / "residual_flow"
DEFAULT_WEB_DIR = REPO_ROOT / "webapp" / "residual_flow"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def npz_path(base: str, dataset: str, config: str, out_dir: Path,
             tag: str | None = None) -> Path:
    tag = tag or f"{base}_{dataset}"
    return out_dir / f"residual_flow_{tag}_{config}.npz"


# ---------------------------------------------------------------------------
# Step 1 — residual sites: which layers exist on THIS model
# ---------------------------------------------------------------------------

def site_label(block: int, kind: str) -> str:
    """Short display label of one residual site (single format-string home)."""
    return f"block {block} · {kind}"


class ResidualSite(NamedTuple):
    """One recordable residual addition of one block, with its branch endpoint."""
    block: int
    kind: str          # "attn" | "mlp"
    add_layer: str     # ResidualAdd output layer
    branch_layer: str  # branch-summand endpoint layer (arch-dependent)

    @property
    def label(self) -> str:
        return site_label(self.block, self.kind)


def is_eva_block(model) -> bool:
    """timm ``EvaBlock`` (DINOv3) blocks carry LayerScale (``gamma_1``)."""
    return hasattr(model.backbone.blocks[0], "gamma_1")


def branch_endpoint_suffixes(model) -> tuple[str, str]:
    """(attn, mlp) suffixes of the branch-summand endpoint modules.

    timm ``Block``: ``ls1``/``ls2`` are Identity, so ``attn.proj_drop`` /
    ``mlp.drop2`` ARE the add's branch summands. In ``EvaBlock`` (DINOv3)
    LayerScale sits between those modules and the add (routed through a
    Uniform-rule ``LayerScaleMul`` by the canonizer), so the summand is the
    LayerScaleMul output, recorded as ``_lrp_ls1`` / ``_lrp_ls2``.
    """
    if is_eva_block(model):
        blk0 = model.backbone.blocks[0]
        assert blk0.gamma_1 is not None and blk0.gamma_2 is not None, \
            "EvaBlock without LayerScale: endpoint choice unhandled"
        return "._lrp_ls1", "._lrp_ls2"
    return ".attn.proj_drop", ".mlp.drop2"


def list_residual_sites(model) -> List[ResidualSite]:
    """All recordable residual sites of this model, in network order
    (block 0 attn, block 0 mlp, block 1 attn, block 1 mlp, ...)."""
    attn_ep, mlp_ep = branch_endpoint_suffixes(model)
    sites: List[ResidualSite] = []
    for b in range(len(model.backbone.blocks)):
        sites.append(ResidualSite(b, "attn", f"backbone.blocks.{b}._lrp_res1",
                                  f"backbone.blocks.{b}{attn_ep}"))
        sites.append(ResidualSite(b, "mlp", f"backbone.blocks.{b}._lrp_res2",
                                  f"backbone.blocks.{b}{mlp_ep}"))
    return sites


def endpoint_check_pairs(model) -> List[tuple[str, str]]:
    """(upper, lower) layer-name pairs between which only Identity / eval-mode
    Dropout modules sit — their recorded relevance must agree (verified
    numerically on the first batch). Checks the two edge blocks only."""
    n_blocks = len(model.backbone.blocks)
    if is_eva_block(model):
        return [(f"backbone.blocks.{b}.drop_path{i}",
                 f"backbone.blocks.{b}._lrp_ls{i}")
                for b in (0, n_blocks - 1) for i in (1, 2)]
    return [(f"backbone.blocks.{b}.attn.proj_drop", f"backbone.blocks.{b}.ls1")
            for b in (0, n_blocks - 1)] + \
           [(f"backbone.blocks.{b}.mlp.drop2", f"backbone.blocks.{b}.ls2")
            for b in (0, n_blocks - 1)]


# ---------------------------------------------------------------------------
# Step 2 — model, dataset, sample selection
# ---------------------------------------------------------------------------

def load_model_for_args(args, device):
    """Resolve the eval model for (base, dataset, checkpoint).

    ``vit_dinov3_base_in1k`` is a special assembly: timm DINOv3-B/16 backbone
    + the public canvit IN1k linear head on the final-norm CLS token, wrapped
    in one module so ``CondAttribution`` sees a single classifier. Otherwise
    loads a finetuned probe via ``experiments.model_io.load_probe``
    (``--checkpoint`` pins the exact run instead of the newest-glob default).
    """
    import torch.nn as nn
    from experiments.model_io import DATASETS, load_probe

    if args.base == "vit_dinov3_base_in1k":
        import timm
        from experiments.scripts.eval_dinov3_in1k_probe import (
            HEAD_REPOS, load_in1k_linear_head)
        timm_name = "vit_base_patch16_dinov3.lvd1689m"

        class _DinoV3CLSFullProbe(nn.Module):
            """DINOv3 backbone features → final-norm CLS token → linear head."""

            def __init__(self, backbone: nn.Module, head: nn.Module):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                return self.head(self.backbone.forward_features(x)[:, 0])

        backbone = timm.create_model(timm_name, pretrained=True,
                                     num_classes=0, img_size=256)
        head = load_in1k_linear_head(timm_name, device)
        model = _DinoV3CLSFullProbe(backbone, head).eval().to(device)
        model.requires_grad_(False)
        ck = {"base": args.base, "head": "canvit_in1k_linear_cls",
              "num_classes": 1000, "dataset": "imagenet_val_hf"}
        return model, ck, Path(f"timm:{timm_name} + hf:{HEAD_REPOS[timm_name]}")

    tag = DATASETS[args.dataset][2]
    ckpt = Path(args.checkpoint) if getattr(args, "checkpoint", None) else None
    model, ck, ck_path = load_probe(tag, device, base=args.base, path=ckpt)
    if ckpt is not None:
        assert Path(ck_path).resolve() == ckpt.resolve(), (ck_path, ckpt)
    return model, ck, ck_path


def dataset_eval_kwargs(dataset: str) -> dict:
    """Eval-split loader kwargs for a ``DATASETS`` entry. FunnyBirds evaluates
    on the *test* split (it contains zero part-ablated images)."""
    from experiments.model_io import DATASETS
    if dataset == "funny_birds":
        return {"split": "test"}
    return dict(DATASETS[dataset][1])


def pick_class_diverse(ds, n: int, seed: int = 0) -> List[int]:
    """Round-robin over classes (like ``crp_gallery.pick_samples``) so the
    sample stays class-diverse even when n < n_classes * per-class count."""
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


def pick_correct_class_diverse(model, ds, idx_order, n: int, normalize, device,
                               batch: int = 64) -> List[int]:
    """Filter a class-diverse candidate ordering down to the first ``n``
    correctly-classified images (round-robin order preserved)."""
    import torch
    sel: List[int] = []
    scanned = 0
    with torch.no_grad():
        for i0 in range(0, len(idx_order), batch):
            chunk = idx_order[i0:i0 + batch]
            x = torch.stack([ds[i][0] for i in chunk]).to(device)
            y = [int(ds[i][1]) for i in chunk]
            pred = model(normalize(x)).argmax(-1).cpu()
            sel.extend(i for i, p, t in zip(chunk, pred.tolist(), y) if p == t)
            scanned += len(chunk)
            if len(sel) >= n:
                break
    print(f"  correct-classified selection: scanned {scanned} candidates "
          f"→ kept {min(len(sel), n)}")
    if len(sel) < n:
        raise RuntimeError(f"only {len(sel)} correctly-classified images found")
    return sel[:n]


# ---------------------------------------------------------------------------
# Step 3 — conditional attribution: record skip/branch relevance per site
# ---------------------------------------------------------------------------

class RecordedFlow(NamedTuple):
    """Per-site relevance sums over the sample, plus bookkeeping.

    ``branch_*`` / ``skip_*`` have shape (n_sites, n_samples, embed_dim):
    per-dim sums over PATCH tokens (prefix rows excluded). ``tot_add`` of
    shape (n_sites, n_samples) keeps ALL token rows and dims — used for the
    conservation-drift stats.
    """
    branch_signed: np.ndarray
    skip_signed: np.ndarray
    branch_abs: np.ndarray
    skip_abs: np.ndarray
    tot_add: np.ndarray
    sample_target: np.ndarray
    sample_pred: np.ndarray
    sample_logit: np.ndarray
    endpoint_identity_err: float


def _ingest_batch(store: RecordedFlow, res, sites: List[ResidualSite],
                  sl: slice, n_prefix: int) -> None:
    """Split one batch's add relevance into skip/branch and accumulate the
    per-dim token sums for every site. R_skip = R_add - R_branch is exact
    because the residual rule splits elementwise."""
    for si, site in enumerate(sites):
        r_add = res.relevances[site.add_layer]         # (B, N_tokens, D)
        r_br = res.relevances[site.branch_layer]
        r_skip = r_add - r_br
        br_patch = r_br[:, n_prefix:]                  # patch-token rows only
        sk_patch = r_skip[:, n_prefix:]
        store.branch_signed[si, sl] = br_patch.sum(1).cpu().numpy()
        store.skip_signed[si, sl] = sk_patch.sum(1).cpu().numpy()
        store.branch_abs[si, sl] = br_patch.abs().sum(1).cpu().numpy()
        store.skip_abs[si, sl] = sk_patch.abs().sum(1).cpu().numpy()
        store.tot_add[si, sl] = r_add.sum((1, 2)).cpu().numpy()


def record_residual_flow(attribution, cfg, ds, idxs, normalize, sites, model,
                         device, batch_size: int) -> RecordedFlow:
    """Run batched conditional LRP (conditioned on the true class) and
    accumulate the per-site relevance sums for every sample."""
    import torch
    record_layers = sorted({l for s in sites
                            for l in (s.add_layer, s.branch_layer)})
    check_pairs = endpoint_check_pairs(model)
    check_layers = sorted({l for pair in check_pairs for l in pair})
    n_prefix = int(getattr(model.backbone, "num_prefix_tokens", 1))
    embed_dim = int(model.backbone.embed_dim)
    n_sites, S = len(sites), len(idxs)

    store = RecordedFlow(
        branch_signed=np.zeros((n_sites, S, embed_dim), np.float32),
        skip_signed=np.zeros((n_sites, S, embed_dim), np.float32),
        branch_abs=np.zeros((n_sites, S, embed_dim), np.float32),
        skip_abs=np.zeros((n_sites, S, embed_dim), np.float32),
        tot_add=np.zeros((n_sites, S), np.float32),
        sample_target=np.zeros(S, np.int64),
        sample_pred=np.zeros(S, np.int64),
        sample_logit=np.zeros(S, np.float32),
        endpoint_identity_err=0.0,  # set after the loop via _replace
    )
    endpoint_err = 0.0
    for i0 in range(0, S, batch_size):
        chunk = idxs[i0:i0 + batch_size]
        xs, ys = zip(*[(ds[i][0], int(ds[i][1])) for i in chunk])
        x = torch.stack(list(xs)).to(device)
        xin = normalize(x).requires_grad_(True)
        rec = record_layers + (check_layers if i0 == 0 else [])
        res = attribution(xin, [{"y": [y]} for y in ys], composite_cls(),
                          record_layer=rec)
        missing = [l for l in record_layers if l not in res.relevances]
        if missing:
            raise RuntimeError(f"recording failed for layers: {missing}")
        if i0 == 0:
            errs = [(res.relevances[a] - res.relevances[b]).abs().max()
                    for a, b in check_pairs]
            endpoint_err = float(torch.stack(errs).max())
            print(f"  endpoint-identity check over {len(check_pairs)} pairs: "
                  f"max err = {endpoint_err:.3e}")
        pred = res.prediction.detach()
        for j, y in enumerate(ys):
            store.sample_target[i0 + j] = y
            store.sample_pred[i0 + j] = int(pred[j].argmax())
            store.sample_logit[i0 + j] = float(pred[j, y])
        _ingest_batch(store, res, sites, slice(i0, i0 + len(chunk)), n_prefix)
        done, total = i0 // batch_size + 1, (S + batch_size - 1) // batch_size
        print(f"  batch {done}/{total} done", flush=True)
    # NamedTuples are immutable: _replace shares the preallocated arrays.
    return store._replace(endpoint_identity_err=endpoint_err)


# ---------------------------------------------------------------------------
# Step 4 — conservation drift, metadata, save
# ---------------------------------------------------------------------------

def propagation_drift(tot_add: np.ndarray, sites: List[ResidualSite]):
    """|Δ total relevance| between consecutive add cuts, relative to the final
    block's total (Gamma-rule bias absorption — a property of the LRP recipe,
    not of the exact elementwise skip/branch split).

    Returns ``(within_block, across_blocks)``: within block b = between its
    mlp and attn adds (through skip + MLP); across = between block b-1's mlp
    add and block b's attn add (through skip + attention).
    """
    attn = [i for i, s in enumerate(sites) if s.kind == "attn"]
    mlp = [i for i, s in enumerate(sites) if s.kind == "mlp"]
    ref = np.abs(tot_add[mlp[-1]])
    within_block = np.abs(tot_add[attn] - tot_add[mlp]) / ref
    across_blocks = np.abs(tot_add[mlp][:-1] - tot_add[attn][1:]) / ref
    return within_block, across_blocks


def build_metadata(args, ck_path, model, cfg, ds_kw, idxs,
                   store: RecordedFlow, drift) -> dict:
    within_block, across_blocks = drift
    n_prefix = int(getattr(model.backbone, "num_prefix_tokens", 1))
    ep_attn, ep_mlp = branch_endpoint_suffixes(model)
    return {
        "base": args.base, "dataset": args.dataset, "config": args.config,
        "model_tag": args.model_tag,
        "split": "test" if args.dataset == "funny_birds" else str(ds_kw),
        "checkpoint": str(ck_path), "n_samples": len(idxs),
        "n_blocks": len(model.backbone.blocks),
        "embed_dim": int(model.backbone.embed_dim), "seed": args.seed,
        "composite_desc": (composite_cls.__doc__ or "").strip().splitlines()[0],
        "block_type": "EvaBlock" if is_eva_block(model) else "Block",
        "branch_endpoints": [ep_attn, ep_mlp],
        "num_prefix_tokens_excluded": n_prefix,
        "token_rows": f"patch tokens only (prefix rows 0..{n_prefix - 1} = cls"
                      f"{'+register' if n_prefix > 1 else ''} excluded from "
                      "per-dim sums; tot_add keeps all rows)",
        "selection": "class-diverse round-robin, correctly-classified only",
        "endpoint_identity_err": store.endpoint_identity_err,
        "accuracy_on_sample":
            float((store.sample_pred == store.sample_target).mean()),
        "drift_mlp_median": float(np.median(within_block)),
        "drift_mlp_max": float(within_block.max()),
        "drift_attn_median": float(np.median(across_blocks)),
        "drift_attn_max": float(across_blocks.max()),
        "generated": _now(),
    }


def save_npz(args, sites, idxs, store: RecordedFlow, meta: dict) -> Path:
    out = npz_path(args.base, args.dataset, args.config, Path(args.out_dir),
                   tag=args.model_tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        branch_signed=store.branch_signed, skip_signed=store.skip_signed,
        branch_abs=store.branch_abs, skip_abs=store.skip_abs,
        tot_add=store.tot_add,
        site_block=np.array([s.block for s in sites], np.int64),
        site_kind=np.array([s.kind for s in sites]),
        site_add_layer=np.array([s.add_layer for s in sites]),
        site_branch_layer=np.array([s.branch_layer for s in sites]),
        sample_ds_index=np.array(idxs, np.int64),
        sample_target=store.sample_target,
        sample_pred=store.sample_pred,
        sample_logit=store.sample_logit,
        meta=np.array(json.dumps(meta)),
    )
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta, indent=2))
    return out


def compute(args) -> Path:
    from zennit_extensions.lrp_composites import AttnLRPBaselineComposite, CheferLRPComposite, CPLRPComposite
    composites = {"cp_lrp_baseline": CPLRPComposite, "attnlrp_baseline": AttnLRPBaselineComposite,
                  "chefer_lrp": CheferLRPComposite}
    from crp.attribution import CondAttribution
    from experiments.datasets import load as load_dataset
    from experiments.model_io import DATASETS, backbone_transforms

    device = args.device
    model, ck, ck_path = load_model_for_args(args, device)
    print(f"checkpoint/model source: {ck_path}")

    ds_kw = dataset_eval_kwargs(args.dataset)
    transform, normalize = backbone_transforms(model.backbone)
    ds = load_dataset(DATASETS[args.dataset][0], root=REPO_ROOT / "data",
                      transform=transform, **ds_kw)
    candidates = pick_class_diverse(ds, len(ds), seed=args.seed)
    idxs = pick_correct_class_diverse(model, ds, candidates, args.n_samples,
                                      normalize, device)
    print(f"{len(idxs)} class-diverse correctly-classified samples, "
          f"model={ck['base']}·{ck['head']}, D={int(model.backbone.embed_dim)}")

    sites = list_residual_sites(model)
    composite_cls = composites[args.config]
    attribution = CondAttribution(model)
    store = record_residual_flow(attribution, cfg, ds, idxs, normalize, sites,
                                 model, device, args.batch_size)
    drift = propagation_drift(store.tot_add, sites)
    meta = build_metadata(args, ck_path, model, cfg, ds_kw, idxs,
                          store, drift)
    return save_npz(args, sites, idxs, store, meta)


# ---------------------------------------------------------------------------
# Render — every panel is built from the site list stored in the npz, so the
# page shows exactly the layers of the evaluated model.
# ---------------------------------------------------------------------------

C_SKIP, C_MID, C_BRANCH = "#2a78d6", "#f0efec", "#e34948"
SURFACE, PLANE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED, BASE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
HIST_BINS = 16

TOOLTIPS = [("site", "@site"), ("dim", "@dim"),
            ("median f", "@median"), ("mean f", "@mean"), ("IQR", "@iqr"),
            ("median signed R(branch)", "@bsig"),
            ("median signed R(skip)", "@ssig")]

_BOKEH = None


def _import_bokeh():
    """Import bokeh lazily so ``compute`` never needs it installed."""
    global _BOKEH
    if _BOKEH is None:
        from bokeh import embed, layouts, models, plotting, resources
        _BOKEH = types.SimpleNamespace(
            figure=plotting.figure, file_html=embed.file_html,
            column=layouts.column, row=layouts.row, INLINE=resources.INLINE,
            **{name: getattr(models, name) for name in (
                "ColumnDataSource", "CustomJS", "Div", "HoverTool",
                "Range1d", "TabPanel", "Tabs")})
    return _BOKEH


# Step R1 — load one model's recorded flow + per-cell statistics -------------

class FlowData(NamedTuple):
    """One model's recorded relevance flow."""
    branch_abs: np.ndarray      # (n_sites, n_samples, n_dims)
    skip_abs: np.ndarray
    branch_signed: np.ndarray
    skip_signed: np.ndarray
    site_labels: List[str]      # one label per residual site of THIS model
    meta: dict

    @property
    def n_sites(self) -> int:
        return self.branch_abs.shape[0]

    @property
    def n_samples(self) -> int:
        return self.branch_abs.shape[1]

    @property
    def n_dims(self) -> int:
        return self.branch_abs.shape[2]


def _escape_meta_html(meta: dict) -> dict:
    """HTML-escape every npz-supplied string in ``meta`` — the values land in
    Bokeh Div text and hover tooltips, which render as HTML. Non-strings
    (numbers, bools) pass through; list items are escaped element-wise."""
    def esc(v):
        return html.escape(v, quote=False) if isinstance(v, str) else v
    return {k: ([esc(x) for x in v] if isinstance(v, list) else esc(v))
            for k, v in meta.items()}


def load_flow(npz: Path) -> FlowData:
    """Load the npz; site labels are read from the stored site list and every
    npz-supplied string is HTML-escaped (see ``_escape_meta_html``)."""
    z = np.load(npz, allow_pickle=False)
    labels = [site_label(int(b), html.escape(str(k), quote=False))
              for b, k in zip(z["site_block"], z["site_kind"])]
    return FlowData(z["branch_abs"], z["skip_abs"], z["branch_signed"],
                    z["skip_signed"], labels,
                    _escape_meta_html(json.loads(str(z["meta"]))))


class FractionStats(NamedTuple):
    """Across-sample summaries of the per-cell branch fraction f."""
    f: np.ndarray               # (n_sites, n_samples, n_dims)
    median: np.ndarray          # (n_sites, n_dims)
    mean: np.ndarray            # (n_sites, n_dims)
    iqr: np.ndarray             # (n_sites, n_dims)
    med_branch_signed: np.ndarray
    med_skip_signed: np.ndarray
    sort_pos: np.ndarray        # dim → x position when sorted by mean f (desc)


def branch_fraction_stats(flow: FlowData) -> FractionStats:
    f = flow.branch_abs / (flow.branch_abs + flow.skip_abs + 1e-12)
    q25, q75 = np.percentile(f, [25, 75], axis=1)
    order = np.argsort(-f.mean(axis=1).mean(axis=0))
    sort_pos = np.empty(flow.n_dims, int)
    sort_pos[order] = np.arange(flow.n_dims)
    return FractionStats(f, np.median(f, axis=1), f.mean(axis=1), q75 - q25,
                         np.median(flow.branch_signed, axis=1),
                         np.median(flow.skip_signed, axis=1), sort_pos)


def per_cell_histograms(f: np.ndarray, n_bins: int) -> np.ndarray:
    """Histogram of f over samples for every (site, dim) cell."""
    n_sites, _, n_dims = f.shape
    bin_idx = np.clip((f * n_bins).astype(int), 0, n_bins - 1)
    hist = np.zeros((n_sites, n_dims, n_bins), np.int32)
    for si in range(n_sites):
        for bi in range(n_bins):
            hist[si, :, bi] = (bin_idx[si] == bi).sum(axis=0)
    return hist


# Step R2 — small rendering helpers -------------------------------------------

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


def label_site_axis(axis, site_labels: List[str]) -> None:
    """One tick per residual site of this model, labeled from its npz."""
    axis.ticker = list(range(len(site_labels)))
    axis.major_label_overrides = dict(enumerate(site_labels))


def build_cell_source(flow: FlowData, stats: FractionStats, hist: np.ndarray):
    """One bokeh source row per (site, dim) cell of this model's grid."""
    bk = _import_bokeh()
    ys, dims = np.mgrid[0:flow.n_sites, 0:flow.n_dims]
    ys, dims = ys.ravel(), dims.ravel()
    alpha = 0.25 + 0.75 * (1 - np.clip(stats.iqr / 0.5, 0, 1))
    return bk.ColumnDataSource(dict(
        x_idx=dims.astype(float), x_sort=stats.sort_pos[dims].astype(float),
        y=ys.astype(float), dim=dims,
        site=[flow.site_labels[i] for i in ys],
        color=diverging_hex(stats.median.ravel()), alpha=alpha.ravel(),
        median=stats.median.ravel().round(3), mean=stats.mean.ravel().round(3),
        iqr=stats.iqr.ravel().round(3),
        bsig=stats.med_branch_signed.ravel().round(4),
        ssig=stats.med_skip_signed.ravel().round(4),
        hist=[hist[y, d].tolist() for y, d in zip(ys, dims)],
    ))


# Step R3 — the four panels ----------------------------------------------------

def build_heatmap_tab(flow: FlowData, cell, hover_js, xfield: str,
                      title: str):
    """Heatmap tab: sites x dims colored by median f, with the path totals."""
    bk = _import_bokeh()
    y_range = bk.Range1d(flow.n_sites - 0.5, -0.5)
    p = bk.figure(width=1180, height=620, title=title,
                  x_range=bk.Range1d(-0.5, flow.n_dims - 0.5),
                  y_range=y_range,
                  tools="hover,pan,box_zoom,wheel_zoom,reset",
                  active_scroll=None)
    r = p.rect(x=xfield, y="y", width=1, height=1, source=cell,
               fill_color="color", fill_alpha="alpha", line_color=None)
    hv = p.select_one(bk.HoverTool)
    hv.tooltips = TOOLTIPS
    hv.renderers = [r]
    hv.callback = hover_js
    label_site_axis(p.yaxis, flow.site_labels)
    p.xaxis.axis_label = ("dims sorted by mean branch fraction (desc)"
                          if xfield == "x_sort" else "embedding dim index")
    _style(p)
    return bk.TabPanel(
        child=bk.row(p, build_path_totals_panel(flow, y_range)), title=title)


def build_path_totals_panel(flow: FlowData, y_range):
    """Row marginal: sample-mean of total |R| per path (branch vs skip)."""
    bk = _import_bokeh()
    tot_branch = flow.branch_abs.sum(axis=2).mean(axis=1)
    tot_skip = flow.skip_abs.sum(axis=2).mean(axis=1)
    m = bk.figure(width=240, height=620, title="mean Σ_dims |R| per path",
                  y_range=y_range, tools="")
    m.hbar(y=np.arange(flow.n_sites) - 0.18, height=0.32, right=tot_branch,
           fill_color=C_BRANCH, line_color=None, legend_label="branch")
    m.hbar(y=np.arange(flow.n_sites) + 0.18, height=0.32, right=tot_skip,
           fill_color=C_SKIP, line_color=None, legend_label="skip")
    m.yaxis.visible = False
    m.legend.location = "top_right"
    m.legend.label_text_color = INK2
    m.legend.label_text_font_size = "11px"
    m.legend.background_fill_alpha = 0.6
    _style(m)
    return m


def build_hover_histogram(flow: FlowData, cell):
    """Per-sample f histogram linked to the heatmap hover. Returns
    (title div, figure, hover callback) — one shared instance for all tabs."""
    bk = _import_bokeh()
    hist_src = bk.ColumnDataSource(dict(
        left=(np.arange(HIST_BINS) / HIST_BINS).tolist(),
        right=((np.arange(HIST_BINS) + 1) / HIST_BINS).tolist(),
        top=[0] * HIST_BINS))
    title = bk.Div(text="<i>hover a cell to see its f-distribution</i>",
                   styles={"color": INK2, "font-size": "12px"})
    hover_js = bk.CustomJS(args=dict(src=cell, hsrc=hist_src, title=title),
                           code="""
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
    p = bk.figure(width=430, height=260,
                  title="branch fraction f — per-sample histogram",
                  x_range=bk.Range1d(0, 1), tools="")
    p.quad(left="left", right="right", bottom=0, top="top", source=hist_src,
           fill_color=C_BRANCH, fill_alpha=0.75, line_color=SURFACE,
           line_width=2)
    p.xaxis.axis_label = "f = |R_branch| / (|R_branch| + |R_skip|)"
    p.yaxis.axis_label = f"images (of {flow.n_samples})"
    p.y_range.start = 0
    _style(p)
    return title, p, hover_js


def build_site_distribution_panel(flow: FlowData, stats: FractionStats):
    """Per site: distribution over dims of the per-dim median f
    (5–95% whiskers, IQR box colored by its median, median tick)."""
    bk = _import_bokeh()
    q = np.percentile(stats.median, [5, 25, 50, 75, 95], axis=1)
    yy = np.arange(flow.n_sites)
    p = bk.figure(width=560, height=620,
                  title="per-site distribution over dims of median f",
                  x_range=bk.Range1d(0, 1),
                  y_range=bk.Range1d(flow.n_sites - 0.5, -0.5), tools="")
    p.segment(q[0], yy, q[4], yy, line_color=BASE, line_width=2)
    p.hbar(y=yy, height=0.55, left=q[1], right=q[3],
           fill_color=diverging_hex(q[2]), line_color=SURFACE, line_width=1)
    p.segment(q[2], yy - 0.32, q[2], yy + 0.32, line_color=INK, line_width=2)
    label_site_axis(p.yaxis, flow.site_labels)
    p.xaxis.axis_label = ("median branch fraction f "
                          "(5–95% whiskers, IQR box, median tick)")
    _style(p)
    return p


# Step R4 — page texts and assembly -------------------------------------------

def build_header_div(meta: dict):
    bk = _import_bokeh()
    return bk.Div(text=f"""
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


def build_methods_div(meta: dict, npz_name: str):
    bk = _import_bokeh()
    return bk.Div(text=f"""
      <div style="font-family:system-ui;color:{INK2};max-width:1100px;font-size:13px">
      <h2 style="color:{INK};font-size:16px">Methods</h2>
      <p><b>Model.</b> Finetuned probe <code>{meta['checkpoint']}</code>
      ({meta['base']}, {meta['n_blocks']} blocks, D={meta['embed_dim']}), accuracy on
      the {meta['n_samples']}-image sample: {meta['accuracy_on_sample']:.3f}.
      Attribution: <code>crp.CondAttribution</code> conditioned on the true class
      (<code>{{"y": [target]}}</code>), composite <code>{meta['config']}</code>
      ("{meta['composite_desc']}").</p>
      <p><b>Recording.</b> The block residual canonizer
      (<code>TimmBlockResidualCanonizer</code> /
      <code>EvaBlockResidualCanonizer</code> for {meta['block_type']})
      routes both residual additions of every block through recordable
      <code>ResidualAdd</code> modules. Per block b the recorded layers are the
      add outputs <code>backbone.blocks.b._lrp_res1</code> (attn) and
      <code>backbone.blocks.b._lrp_res2</code> (MLP), and the branch endpoints
      <code>backbone.blocks.b{meta['branch_endpoints'][0]}</code> and
      <code>backbone.blocks.b{meta['branch_endpoints'][1]}</code>.
      Between each endpoint and the add's branch input sit only Identity
      (<code>drop_path*</code>) and eval-mode Dropout modules — for Eva blocks the
      endpoint is the <code>LayerScaleMul</code> output, i.e. already ABOVE the
      Uniform LayerScale rule — so the endpoint gradient equals the branch summand
      of the residual rule's elementwise split — verified numerically:
      max endpoint-identity error = {meta['endpoint_identity_err']:.2e}.</p>
      <p><b>Skip derivation.</b> The composite's residual rule
      (<code>ResidualRatio</code>) splits the relevance at the add output
      elementwise: R<sub>add</sub> = R<sub>skip</sub> + R<sub>branch</sub> per token
      per dimension. R<sub>skip</sub> is therefore derived exactly as
      R<sub>add</sub> − R<sub>branch</sub> (no approximation in the decomposition
      itself). Per dimension we sum over tokens
      ({meta['token_rows']}) — signed sums and absolute-value sums are both
      stored; f uses the absolute sums.</p>
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
      <code>data/results/residual_flow/{npz_name}</code>.
      Script: <code>experiments/scripts/residual_flow_diag.py</code>.</p>
      </div>""")


def render(args) -> Path:
    bk = _import_bokeh()
    src_npz = npz_path(args.base, args.dataset, args.config, Path(args.out_dir),
                       tag=args.model_tag)
    flow = load_flow(src_npz)
    stats = branch_fraction_stats(flow)
    cell = build_cell_source(flow, stats, per_cell_histograms(stats.f, HIST_BINS))
    hist_title, hist_fig, hover_js = build_hover_histogram(flow, cell)
    tabs = bk.Tabs(tabs=[
        build_heatmap_tab(flow, cell, hover_js, "x_sort",
                          "dims sorted by branch fraction"),
        build_heatmap_tab(flow, cell, hover_js, "x_idx", "dims by index"),
    ])
    page = bk.column(
        build_header_div(flow.meta), tabs,
        bk.row(bk.column(hist_title, hist_fig),
               build_site_distribution_panel(flow, stats)),
        build_methods_div(flow.meta, src_npz.name),
        styles={"background": PLANE, "padding": "18px"})
    html = bk.file_html(page, resources=bk.INLINE,
                        title="Residual skip vs branch — LRP relevance flow")
    out_dir = Path(args.webapp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (args.page_name or "index.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        p.add_argument("--model-tag", default=None,
                       help="output naming tag (npz + page); default <base>_<dataset>")
        if name in ("compute", "all"):
            p.add_argument("--checkpoint", default=None,
                           help="explicit best.pt path (overrides newest-run glob)")
            p.add_argument("--n-samples", type=int, default=96)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--device", default="cuda")
            p.add_argument("--seed", type=int, default=0)
        if name in ("render", "all"):
            p.add_argument("--webapp-dir", default=str(DEFAULT_WEB_DIR))
            p.add_argument("--page-name", default=None,
                           help="output html filename (default index.html)")
    args = ap.parse_args()
    if args.cmd in ("compute", "all"):
        compute(args)
    if args.cmd in ("render", "all"):
        render(args)


if __name__ == "__main__":
    main()
