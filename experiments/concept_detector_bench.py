"""Concept-detector Insertion-Deletion (DAPC) benchmark — self-contained script.

Journal `exp:insertion-deletion-bench` (concept-detector section): rank a layer's
parallel concept detectors (embedding dims) by an XAI score, then ZERO their
activations cumulatively most-relevant-first (MoRF) and least-relevant-first
(LeRF), tracking retained predicted-class probability. DAPC = area(LeRF) −
area(MoRF) (higher = better). Random ranking is the baseline floor.

Methods: cp_lrp (StopGradient on Q/K → skipped at query/key), chefer (all sites),
random. Detectors = embedding dims, flipped one-by-one. Sites per block: residual
add, attn.proj_drop, value/query/key probes.

Run:  uv run python -m experiments.concept_detector_bench
Produces:  data/results/benchmark/cdet_dapc_<key>.npz , figures/concept_detector_bench/*,
           webapp/concept_detector_bench/index.html
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crp.concepts import EmbeddingDimConcept, HeadConcept
from experiments.datasets import load_eval_dataset
from experiments.gradinput import GradTimesInputAttribution
from experiments.models import backbone_transforms
from experiments.model_datasets import find_by_tag
from zennit_extensions.canonisation.canonizers import (
    EvaAttentionSubstitutionCanonizer, VanillaViTAttentionSubstitutionCanonizer)
from zennit_extensions.lrp_composites import CPLRPComposite, CheferLRPComposite

REPO = Path(__file__).resolve().parents[1]
RES_DIR = REPO / "data" / "results" / "benchmark"
FIG_DIR = REPO / "figures" / "concept_detector_bench"
WEB_DIR = REPO / "webapp" / "concept_detector_bench"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_IMAGES = 16
K_RANDOM = 5
SEED = 0
CHUNK = 64
BLOCKS = list(range(12))

# (zoo key, short tag, dataset key, ds_extra, label)
MODELS_CFG = [
    ("vit_base_imagenet", "m2_vitb_in", "imagenet", {}, "M2 · ViT-B/16 · ImageNet-1k val"),
    ("vit_small_funny_birds", "m1_vits_fb", "funny_birds", {"split": "test"},
     "M1 · ViT-S/16 · FunnyBirds test"),
    ("vit_dinov3_base_imagenet", "m6_dinob_in", "imagenet", {},
     "M6 · DINOv3-B/16 · ImageNet-1k val"),
    ("vit_dinov3_small_funny_birds", "m5_dinos_fb", "funny_birds", {"split": "test"},
     "M5 · DINOv3-S/16 · FunnyBirds test"),
]

# Vanilla ViT: q and k merge into one "qk" site — under zero-ablation, dropping
# embedding-dim d from q and from k delete the identical term q_id·k_jd, so the
# perturbation is bit-identical (ranked via the q-probe). DINOv3/Eva applies RoPE
# before the score, so that equivalence breaks; q and k are kept as SEPARATE
# "query"/"key" sites there.
ALL_SITES = ["residual", "proj_drop", "value", "qk"]            # vanilla ViT
ALL_SITES_EVA = ["residual", "proj_drop", "value", "query", "key"]  # DINOv3 (split q/k)
COMPARABLE = ["residual", "proj_drop", "value"]   # cp_lrp valid sites (no StopGradient)
QK_SITES = ("qk", "query", "key")                 # StopGradient sites cp_lrp can't attribute
METHODS = ["cp_lrp", "chefer", "optimal", "optimal_dual", "random"]


def is_eva(model) -> bool:
    """DINOv3/Eva backbones use timm EvaAttention (RoPE, register tokens)."""
    from timm.models.eva import EvaAttention
    return isinstance(model.backbone.blocks[0].attn, EvaAttention)


def method_sites(method: str, sites):
    """Sites a method can attribute: cp_lrp is undefined at q/k (StopGradient)."""
    if method == "cp_lrp":
        return [s for s in sites if s in COMPARABLE]
    return list(sites)
COMPOSITE = {"cp_lrp": CPLRPComposite, "chefer": CheferLRPComposite}
METHOD_LABEL = {"cp_lrp": "CP-LRP", "chefer": "Chefer", "optimal": "Optimal (greedy)",
                "optimal_dual": "Optimal (greedy, dual)", "random": "Random"}
# one-line answers to "what exactly was computed?", shown under the selectors
METHOD_INFO = {
    "cp_lrp": ("CP-LRP concept relevance (cp_lrp composite, grad×input convention): "
               "detectors = embedding-dim channels at the probe site; ranking ψ = per-dim "
               "SUM over tokens of class-conditional relevance (token-sum, not max; signed, unclamped)"),
    "chefer": ("Chefer et al. 2021 transformer attribution (relevance⊗grad of the attention map); "
               "same detector/ranking protocol (token-sum over signed relevance)"),
    "random": "uniform random ranking, mean over K=5 seeds (baseline floor)",
    "optimal": ("heuristic-optimal greedy O(n²): per step remove the detector with the largest "
                "current drop Δ = p(state) − p(state∖{c}); removal state accumulates"),
    "optimal_dual": ("dual greedy O(n²), two detectors per step: argmax Δ to the head (MoRF-first) "
                     "and argmin Δ to the tail (LeRF-first) evaluated at the same removal state"),
}
SITE_INFO = {
    "residual": "site: block output (residual stream after skip-add)",
    "proj_drop": "site: attention output projection (attn.proj_drop)",
    "value": "site: value probe in the unfolded-attention substitution (pre-projection V)",
    "qk": "site: query/key probe (q and k ablation of the same dim is bit-identical; ranked via q)",
    "query": "site: query probe (DINOv3: q and k split — RoPE breaks q==k equivalence)",
    "key": "site: key probe (DINOv3: q and k split — RoPE breaks q==k equivalence)",
}
SITE_LABEL = {"residual": "residual (block out)", "proj_drop": "attn.proj_drop",
              "value": "value", "qk": "query/key (shared dim)",
              "query": "query", "key": "key"}


def layer_name(site: str, b: int) -> str:
    return {
        "residual": f"backbone.blocks.{b}",
        "proj_drop": f"backbone.blocks.{b}.attn.proj_drop",
        "qk": f"backbone.blocks.{b}.attn.q_lrp_probe",
        "query": f"backbone.blocks.{b}.attn.q_lrp_probe",
        "key": f"backbone.blocks.{b}.attn.k_lrp_probe",
        "value": f"backbone.blocks.{b}.attn.v_lrp_probe",
    }[site]


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def dapc_of(morf: np.ndarray, lerf: np.ndarray) -> float:
    n = len(morf) - 1
    return float(_trapz(lerf, dx=1.0 / n) - _trapz(morf, dx=1.0 / n))


# ── data selection ────────────────────────────────────────────────────────────
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


# ── ranking psi (per-detector importance under an LRP composite) ───────────────
def rank_psi(attribution, normalize, x, cls, method, num_heads, sites, blocks, concept):
    """``concept``: a Concept instance whose ``attribute`` reduces a
    (B, N, embed_dim) relevance to (B, n_detectors) — EmbeddingDimConcept
    (n=embed_dim) or HeadConcept (n=num_heads)."""
    comp = COMPOSITE[method]()
    layers = [layer_name(s, b) for s in sites for b in blocks]
    xin = normalize(x).clone().detach().requires_grad_(True)
    res = attribution(xin, [{"y": [int(cls)]}], comp, record_layer=layers)
    return {(s, b): concept.attribute(res.relevances[layer_name(s, b)],
                                      abs_norm=False)[0].detach().cpu()
            for s in sites for b in blocks}


# ── zero-ablation perturbation curve ───────────────────────────────────────────
# Removal of detector d == EmbeddingDimConcept's masking semantics applied
# forward-side: concepts.py mask() zeroes the dim's channels across all tokens
# in the (backward) relevance stream for conditional attribution; this hook
# zeroes the same channels in the forward stream for eval-time occlusion
# (no backward exists under no_grad, so concept.mask itself cannot be called
# here). The concept class IS used at its own stage: rank_psi scores detectors
# via EmbeddingDimConcept(...).attribute; the probe sites (residual/proj_drop
# and the unfolded-attention v/q probes) are exactly the (B, N, embed_dim)
# sites the concept classes address. Detections are per-dim; a head-level bench
# would only widen keep to the head's dim slice (HeadConcept's mapping).
class ZeroChannelsHook:
    def __init__(self, head_dim=None):
        self.keep = None       # (chunk, D_search) float, per-sample keep-mask
        self.head_dim = head_dim   # head concept: expand each head bit over its dims

    def __call__(self, module, inp, out):
        if self.keep is None:
            return out
        keep = self.keep
        if self.head_dim is not None:
            # (chunk, num_heads) -> (chunk, num_heads*head_dim) contiguous slices
            keep = keep.repeat_interleave(self.head_dim, dim=-1)
        return out * keep.unsqueeze(1)


def cumulative_keep(D, order):
    """(D+1, D) keep-mask: row k has channels order[:k] zeroed (removed)."""
    rank = torch.empty(D, dtype=torch.long)
    rank[order] = torch.arange(D)
    k = torch.arange(D + 1).unsqueeze(1)
    return (rank.unsqueeze(0) >= k).float()


def prob_curve(model, normalize, x, pred, order, hook):
    D = order.shape[0]
    keep = cumulative_keep(D, order)
    xn = normalize(x).to(DEVICE)
    probs = torch.empty(D + 1)
    with torch.no_grad():
        for s in range(0, D + 1, CHUNK):
            kb = keep[s:s + CHUNK].to(DEVICE)
            hook.keep = kb
            logits = model(xn.expand(kb.shape[0], -1, -1, -1))
            probs[s:s + kb.shape[0]] = logits.softmax(-1)[:, pred].cpu()
    hook.keep = None
    return probs.numpy()


# ── per-model run ──────────────────────────────────────────────────────────────
def run_model(key, tag, dataset, extra, label, concept="embed_dim",
              sites=None, blocks=None):
    is_head = concept == "head"
    suffix = "__head" if is_head else ""
    out_path = RES_DIR / f"cdet_dapc_{key}{suffix}.npz"
    if out_path.exists():
        print(f"[{tag}] exists → skip")
        return
    print(f"[{tag}] loading {key}… (concept={concept})")
    model = find_by_tag(key, device=DEVICE).model.eval()
    transform, normalize = backbone_transforms(model.backbone)
    ds = load_eval_dataset(dataset, transform, extra)
    num_heads = model.backbone.blocks[0].attn.num_heads
    embed_dim = int(model.backbone.embed_dim)
    run_blocks = list(blocks) if blocks is not None else BLOCKS
    # DINOv3/Eva splits q/k into separate query/key sites (RoPE); vanilla merges to qk.
    run_sites = list(sites) if sites is not None else (ALL_SITES_EVA if is_eva(model) else ALL_SITES)

    # head concept: detectors = heads (D_search=num_heads); the occlusion hook
    # widens each head bit over its contiguous head_dim slice of embed_dim.
    if is_head:
        head_dim = embed_dim // num_heads
        D = num_heads
        concept_obj = HeadConcept(num_heads=num_heads)
    else:
        head_dim = None
        D = embed_dim
        concept_obj = EmbeddingDimConcept(num_heads=num_heads)

    picks = select_correct(model, normalize, ds, N_IMAGES)
    print(f"[{tag}] {len(picks)} correct images, D={D}, heads={num_heads}, "
          f"embed_dim={embed_dim}, sites={run_sites}, blocks={run_blocks}")

    # (1) rankings for the two LRP methods (composite context canonizes the model).
    # Both composites are grad×input → relevance = g×activation (GradTimesInputAttribution).
    attribution = GradTimesInputAttribution(model)
    psi = {}
    for method in ("cp_lrp", "chefer"):
        m_sites = method_sites(method, run_sites)
        if not m_sites:
            continue
        for j, (idx, pred, _) in enumerate(picks):
            x = ds[idx][0].unsqueeze(0).to(DEVICE)
            psi[(method, j)] = rank_psi(attribution, normalize, x, pred, method,
                                        num_heads, m_sites, run_blocks, concept_obj)
        print(f"[{tag}] ranked {method}")

    # (2) zero-ablation curves on the attention-unfolded model. Apply both attention
    # canonizers; their isinstance filters are disjoint (TimmAttention vs EvaAttention),
    # so exactly one fires per model family (unfolds the q/k/v probe sites).
    handles = []
    for c in (VanillaViTAttentionSubstitutionCanonizer(block_indices=None),
              EvaAttentionSubstitutionCanonizer(block_indices=None)):
        handles += c.apply(model)
    store = {}
    try:
        # optimal / optimal_dual curves come from the side-car (concept_detector_optimal);
        # here we compute only the ranking-based LRP methods + random.
        bench_methods = [m for m in METHODS if m == "random" or (m, 0) in psi]
        for method in bench_methods:
            for site in (s for s in method_sites(method, run_sites)):
                for b in run_blocks:
                    mod = model.get_submodule(layer_name(site, b))
                    hook = ZeroChannelsHook(head_dim=head_dim)
                    hh = mod.register_forward_hook(hook)
                    morf = np.zeros((len(picks), D + 1), np.float32)
                    lerf = np.zeros_like(morf)
                    dapc = np.zeros(len(picks), np.float32)
                    for j, (idx, pred, _) in enumerate(picks):
                        x = ds[idx][0].unsqueeze(0)
                        if method == "random":
                            ms, ls = [], []
                            for kk in range(K_RANDOM):
                                g = torch.Generator().manual_seed(SEED * 9973 + idx * 131 + kk)
                                o = torch.randperm(D, generator=g)
                                ms.append(prob_curve(model, normalize, x, pred, o, hook))
                                ls.append(prob_curve(model, normalize, x, pred, o.flip(0), hook))
                            m, l = np.mean(ms, 0), np.mean(ls, 0)
                        else:
                            o = torch.argsort(psi[(method, j)][(site, b)], descending=True)
                            m = prob_curve(model, normalize, x, pred, o, hook)
                            l = prob_curve(model, normalize, x, pred, o.flip(0), hook)
                        morf[j], lerf[j], dapc[j] = m, l, dapc_of(m, l)
                    hh.remove()
                    p = f"{method}__{site}__b{b}"
                    store[p + "__morf"] = morf
                    store[p + "__lerf"] = lerf
                    store[p + "__dapc"] = dapc
                print(f"[{tag}] {method} · {site} done")
    finally:
        for h in handles:
            h.remove()

    meta = {"key": key, "tag": tag, "label": label, "dataset": dataset, "D": D,
            "concept": concept, "embed_dim": embed_dim, "num_heads": int(num_heads),
            "sites": run_sites, "blocks": run_blocks, "seed": SEED, "k_random": K_RANDOM,
            "image_ids": [i for i, _, _ in picks], "preds": [p for _, p, _ in picks]}
    RES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta=json.dumps(meta), **store)
    print(f"[{tag}] saved {out_path}")


# ── aggregation ────────────────────────────────────────────────────────────────
class _Store(dict):
    @property
    def files(self):
        return list(self.keys())


def load_model_store(key, concept="embed_dim"):
    """Bench npz merged with the optional optimal-ranking side-car
    (``cdet_dapc_<key>[__head]_optimal.npz``, produced by
    :mod:`experiments.concept_detector_optimal`; keys ``optimal__*``)."""
    suffix = "__head" if concept == "head" else ""
    z = np.load(RES_DIR / f"cdet_dapc_{key}{suffix}.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    store = _Store({k: z[k] for k in z.files})
    # side-car name must match OptimalStore: "__head_optimal" (head) / "__optimal" (embed).
    sidecar = "__head_optimal" if concept == "head" else "__optimal"
    opt = RES_DIR / f"cdet_dapc_{key}{sidecar}.npz"
    if opt.exists():
        zo = np.load(opt, allow_pickle=True)
        for k in zo.files:
            if k != "meta":
                store[k] = zo[k]
    return meta, store


def mean_dapc(z, method, site, b):
    k = f"{method}__{site}__b{b}__dapc"
    return float(z[k].mean()) if k in z.files else float("nan")


def combined_scores(z, sites, blocks):
    """Baseline-subtracted mean over (comparable ∩ run) sites × blocks, per method."""
    comp = [s for s in sites if s in COMPARABLE] or sites
    rand = np.array([[mean_dapc(z, "random", s, b) for b in blocks] for s in comp])
    out = {}
    for method in ("cp_lrp", "chefer", "optimal", "optimal_dual"):
        m = np.array([[mean_dapc(z, method, s, b) for b in blocks] for s in comp])
        out[method] = float(np.nanmean(m - rand))
    return out


# ── figures (match prior AOPC style: MoRF red / LeRF blue, faint ±1σ) ──────────
def _save(fig, path_noext):
    path_noext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_noext.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(path_noext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def curve_figure(z, tag, site, method, D, blocks, fig_dir):
    ncol = min(4, len(blocks))
    nrow = int(np.ceil(len(blocks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.25 * ncol, 2.8 * nrow), squeeze=False)
    x = np.arange(D + 1)                     # absolute count of detectors removed
    for ax in axes.flat:
        ax.axis("off")
    for i, b in enumerate(blocks):
        ax = axes.flat[i]
        ax.axis("on")
        mo = z[f"{method}__{site}__b{b}__morf"]
        le = z[f"{method}__{site}__b{b}__lerf"]
        for cur, col, lab in ((mo, "tab:red", "MoRF"), (le, "tab:blue", "LeRF")):
            mean, sd = cur.mean(0), cur.std(0)
            ax.plot(x, mean, color=col, lw=1.6, label=lab)
            ax.fill_between(x, mean - sd, mean + sd, color=col, alpha=0.10, lw=0)
        ax.set_title(f"block {b}", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{tag} · {SITE_LABEL[site]} · {METHOD_LABEL[method]} — "
                 f"predicted-class prob vs fraction of detectors zeroed", fontsize=11)
    fig.supxlabel("detectors removed (absolute count)", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    _save(fig, fig_dir / f"cdet_curves_{tag}_{site}_{method}")


def bars_figure(z, tag, sites, blocks, fig_dir):
    xs = np.arange(len(sites))
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = {"cp_lrp": "tab:green", "chefer": "tab:orange",
              "optimal": "tab:purple", "optimal_dual": "tab:olive", "random": "0.6"}
    for i, method in enumerate(METHODS):
        vals, errs = [], []
        for s in sites:
            # tolerant to partially-computed methods (e.g. optimal mid-run):
            # missing blocks contribute NaN and shrink the s.e.m. base
            if s not in method_sites(method, sites):
                vals.append(np.nan); errs.append(0); continue
            per_block = [mean_dapc(z, method, s, b) for b in blocks]
            n_ok = int(np.sum(~np.isnan(per_block)))
            vals.append(float(np.nanmean(per_block)) if n_ok else np.nan)
            errs.append(float(np.nanstd(per_block) / np.sqrt(n_ok)) if n_ok > 1 else 0)
        ax.bar(xs + (i - 2) * width, vals, width, yerr=errs, capsize=2,
               label=METHOD_LABEL[method], color=colors[method])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([SITE_LABEL[s] for s in sites], fontsize=8, rotation=15)
    ax.set_ylabel("mean DAPC (higher = better)")
    ax.set_title(f"{tag} — DAPC per site (mean over {len(blocks)} blocks × {N_IMAGES} images)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, fig_dir / f"cdet_bars_{tag}")


# ── web page (minimal, static; model/site/method selectors swap the curve img) ─
def build_web(models_meta, sites, blocks, web_dir, concept="embed_dim"):
    detector_word = "attention heads" if concept == "head" else "embedding-dim channels"
    comp = [s for s in sites if s in COMPARABLE] or sites
    rows_html, opts_model = [], []
    web_scores, curve_avail = {}, {}
    for key, tag, label in models_meta:
        _, z = load_model_store(key, concept)
        sc = combined_scores(z, sites, blocks)
        web_scores[tag] = sc
        # a method's curve grid exists only once all run blocks are stored
        for s in sites:
            for m in METHODS:
                curve_avail[f"{tag}_{s}_{m}"] = s in method_sites(m, sites) and all(
                    f"{m}__{s}__b{b}__morf" in z.files for b in blocks)
        opts_model.append(f'<option value="{tag}">{label}</option>')
        # per (site, block) DAPC table
        trows = []
        for s in sites:
            for b in blocks:
                cells = []
                vals = {m: mean_dapc(z, m, s, b) for m in METHODS}
                best = max((v for v in vals.values() if not np.isnan(v)), default=float("nan"))
                for m in METHODS:
                    v = vals[m]
                    txt = "—" if np.isnan(v) else f"{v:+.3f}"
                    strong = (not np.isnan(v) and v == best and m != "random")
                    cells.append(f"<td>{'<b>'+txt+'</b>' if strong else txt}</td>")
                trows.append(f'<tr data-model="{tag}"><td class="ln">{layer_name(s, b)}</td>'
                             + "".join(cells) + "</tr>")
        rows_html.append("".join(trows))
    table_body = "".join(rows_html)
    scores_json = json.dumps(web_scores)
    avail_json = json.dumps(curve_avail)
    method_info_json = json.dumps(METHOD_INFO)
    site_info_json = json.dumps(SITE_INFO)
    model_opts = "".join(opts_model)
    cachebust = int(max((p.stat().st_mtime for p in web_dir.glob("*.png")), default=0))
    site_opts = "".join(f'<option value="{s}">{SITE_LABEL[s]}</option>' for s in sites)
    method_opts = "".join(f'<option value="{m}">{METHOD_LABEL[m]}</option>'
                          for m in ("chefer", "cp_lrp", "optimal", "optimal_dual", "random"))
    default_tag = models_meta[0][1]
    comp_label = "/".join(comp)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Concept-detector Insertion-Deletion (DAPC) — {concept}</title>
<style>
 body {{ font-family: system-ui, sans-serif; background:#f9f9f7; color:#0b0b0b; margin:0; padding:0 0 40px; }}
 header {{ padding:12px 20px; border-bottom:1px solid #e1e0d9; background:#fcfcfb; }}
 header h1 {{ font-size:16px; margin:0 0 4px; }} header .sub {{ color:#52514e; font-size:12px; }}
 .wrap {{ max-width:1120px; margin:0 auto; padding:16px 20px; }}
 .controls {{ display:flex; gap:14px; flex-wrap:wrap; align-items:center; margin:14px 0; }}
 select {{ font:inherit; font-size:13px; padding:4px 8px; border:1px solid #c3c2b7; border-radius:6px; background:#fcfcfb; }}
 img {{ max-width:100%; border:1px solid #e1e0d9; border-radius:6px; background:#fff; }}
 h2 {{ font-size:14px; margin:22px 0 8px; }}
 table {{ border-collapse:collapse; font-size:12px; width:100%; }}
 th,td {{ border:1px solid #e1e0d9; padding:3px 7px; text-align:right; }}
 td.ln {{ text-align:left; font-family:ui-monospace,monospace; font-size:11px; color:#333; }}
 tr:nth-child(even) {{ background:#fcfcfb; }}
 .scores {{ font-size:13px; margin:10px 0; }} .scores b {{ font-size:15px; }}
 .info {{ font-size:12px; color:#52514e; margin:6px 0; }}
</style></head><body>
<header><h1>Concept-detector Insertion-Deletion — DAPC ({concept})</h1>
<div class="sub">Rank a layer's detectors ({detector_word}), zero-ablate MoRF/LeRF, area between the curves (DAPC; higher = better).
Occlusion: detector zeroed at the probe site; measured value = softmax probability of the predicted class; curve x-axis = absolute number of detectors removed.
CP-LRP skips query/key (StopGradient). N={N_IMAGES} images, detectors one-by-one, seed {SEED}.</div></header>
<div class="wrap">
 <div class="controls">
  <label>model <select id="model">{model_opts}</select></label>
  <label>site <select id="site">{site_opts}</select></label>
  <label>method <select id="method">{method_opts}</select></label>
 </div>
 <div class="info" id="info"></div>
 <div class="scores" id="scores"></div>
 <img id="curve" alt="curves">
 <div id="nocurve" style="display:none;font-size:13px;color:#52514e;margin:10px 0">
   No perturbation curve for this combination &mdash; either it is undefined by
   design (CP-LRP at query/key: StopGradient keeps the softmax a graph constant)
   or its grid is still being computed (optimal-greedy variants mid-run). The
   per-layer table below shows &ldquo;&mdash;&rdquo; for missing values.</div>
 <h2>DAPC per site (mean over blocks)</h2>
 <img id="bars" alt="bars">
 <h2>Per-layer DAPC (mean over images) — <span id="tblmodel"></span></h2>
 <table><thead><tr><th style="text-align:left">layer</th><th>CP-LRP</th><th>Chefer</th><th>Optimal (greedy)</th><th>Optimal (greedy, dual)</th><th>Random</th></tr></thead>
 <tbody id="tbody">{table_body}</tbody></table>
</div>
<script>
const SCORES = {scores_json};
const AVAIL = {avail_json};
const METHOD_INFO = {method_info_json};
const SITE_INFO = {site_info_json};
const M=document.getElementById("model"), S=document.getElementById("site"), Me=document.getElementById("method");
const fmt = v => (v==null || isNaN(v)) ? "—" : (+v).toFixed(3);
function refresh() {{
  const tag=M.value, site=S.value, method=Me.value;
  const modelText = M.options[M.selectedIndex].text;
  document.getElementById("info").textContent =
    modelText + " · " + (SITE_INFO[site]||"") + " · " + (METHOD_INFO[method]||"");
  // show the note (not a 404) for combos with no perturbation grid:
  // CP-LRP at q/k by design (StopGradient), or optimal still mid-run.
  const noCurve = !AVAIL[`${{tag}}_${{site}}_${{method}}`];
  const img = document.getElementById("curve");
  img.style.display = noCurve ? "none" : "";
  img.src = noCurve ? "" : `cdet_curves_${{tag}}_${{site}}_${{method}}.png?v={cachebust}`;
  document.getElementById("nocurve").style.display = noCurve ? "" : "none";
  document.getElementById("bars").src=`cdet_bars_${{tag}}.png?v={cachebust}`;
  document.getElementById("tblmodel").textContent=tag;
  for (const r of document.querySelectorAll("#tbody tr")) r.style.display = (r.dataset.model===tag)?"":"none";
  const s=SCORES[tag]||{{}};
  document.getElementById("scores").innerHTML =
    `combined score (baseline-subtracted mean over ${{'{comp_label}'}}): `+
    `Optimal (greedy, dual) <b>${{fmt(s.optimal_dual)}}</b> · `+
    `Optimal (greedy) <b>${{fmt(s.optimal)}}</b> · `+
    `Chefer <b>${{fmt(s.chefer)}}</b> · CP-LRP <b>${{fmt(s.cp_lrp)}}</b> · Random 0.000`;
}}
M.value="{default_tag}"; [M,S,Me].forEach(e=>e.addEventListener("change",refresh)); refresh();
</script></body></html>"""
    web_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / "index.html").write_text(html)


def make_outputs(concept="embed_dim"):
    suffix = "__head" if concept == "head" else ""
    fig_dir = FIG_DIR.with_name(FIG_DIR.name + ("_head" if concept == "head" else ""))
    web_dir = WEB_DIR.with_name(WEB_DIR.name + ("_head" if concept == "head" else ""))
    fig_dir.mkdir(parents=True, exist_ok=True)
    models_meta = []
    run_sites, run_blocks = ALL_SITES, BLOCKS
    for key, tag, dataset, extra, label in MODELS_CFG:
        if not (RES_DIR / f"cdet_dapc_{key}{suffix}.npz").exists():
            print(f"[{tag}] no result npz → skip outputs")
            continue
        meta, z = load_model_store(key, concept)
        D = meta["D"]
        # Render every site/block the merged store actually has data for (any method),
        # not just the bench run's sites — so heuristic (optimal) sites present only in
        # the side-car (e.g. residual/proj_drop for head) still appear on the page.
        import re as _re
        present = [(_m.group(1), int(_m.group(2)))
                   for k in z.files
                   if (_m := _re.match(r"\w+__(\w+)__b(\d+)__dapc$", k))]
        site_order = {s: i for i, s in enumerate(ALL_SITES)}
        run_sites = sorted({s for s, _ in present}, key=lambda s: site_order.get(s, 99)) \
            or list(meta.get("sites", ALL_SITES))
        run_blocks = sorted({b for _, b in present}) or list(meta.get("blocks", BLOCKS))
        for site in run_sites:
            for method in METHODS:
                if site in method_sites(method, run_sites) and all(
                        f"{method}__{site}__b{b}__morf" in z for b in run_blocks):
                    curve_figure(z, tag, site, method, D, run_blocks, fig_dir)
        bars_figure(z, tag, run_sites, run_blocks, fig_dir)
        models_meta.append((key, tag, label))
        sc = combined_scores(z, run_sites, run_blocks)
        print(f"[{tag}] combined  chefer={sc['chefer']:+.3f}  cp_lrp={sc['cp_lrp']:+.3f}"
              f"  optimal={sc['optimal']:+.3f}  optimal_dual={sc['optimal_dual']:+.3f}")
    if models_meta:
        # copy the figures next to the html so the static page is self-contained
        for key, tag, label in models_meta:
            for p in fig_dir.glob(f"cdet_*{tag}*.png"):
                web_dir.mkdir(parents=True, exist_ok=True)
                (web_dir / p.name).write_bytes(p.read_bytes())
        build_web(models_meta, run_sites, run_blocks, web_dir, concept)
        print(f"web → {web_dir/'index.html'}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept", choices=["embed_dim", "head"], default="embed_dim")
    ap.add_argument("--sites", nargs="*", default=None,
                    help=f"subset of {ALL_SITES} (default: all)")
    ap.add_argument("--blocks", type=int, nargs="*", default=None,
                    help="subset of blocks 0-11 (default: all)")
    ap.add_argument("--no-web", action="store_true",
                    help="skip building the web page")
    args = ap.parse_args()
    for key, tag, dataset, extra, label in MODELS_CFG:
        run_model(key, tag, dataset, extra, label, concept=args.concept,
                  sites=args.sites, blocks=args.blocks)
    if not args.no_web:
        make_outputs(args.concept)


if __name__ == "__main__":
    main()
