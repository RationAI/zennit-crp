"""Build the static SAE-downstream report (webapp/sae_downstream/index.html).

Reads the artefacts written by ``experiments.sae_downstream`` under
``data/sae_downstream/<case>/`` (``meta.json`` + ``reps_block*.npz``) and renders
ONE standalone HTML page with TWO sections (funny_birds, imagenet), each holding:

  * an eval table (ORIGINAL vs DECOMPOSED model, all SAEs inserted) + Δ,
  * a per-block downstream-FVU summary,
  * a decomposition-quality table (ORIG vs DECODED) at the representative SAE'd
    layers, and
  * side-by-side INTERACTIVE 3D UMAP scatter (ORIG | DECODED), coloured by class,
    embedded as standalone Plotly JS (no server, no CDN dependency beyond the
    single embedded plotly bundle on the first plot).

Manifold rule (per the brief): TWO manifolds only — ORIG (clean model M) vs
DECODED (downstream-loss SAE's decoded M'). No standard-SAE manifold.

Usage::

    python -m experiments.scripts.build_sae_downstream_report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "sae_downstream"
WEB_DIR = REPO_ROOT / "webapp" / "sae_downstream"
CASES = ["funny_birds", "imagenet"]

UMAP_MAX = 1500          # points per manifold panel (subsample for responsiveness)
UMAP_SEED = 0


# ─────────────────────────────────────────────────────────────────────────────
# UMAP 3D
# ─────────────────────────────────────────────────────────────────────────────

def umap3d(X: np.ndarray, seed: int = UMAP_SEED) -> np.ndarray:
    import umap
    n = len(X)
    reducer = umap.UMAP(n_components=3, n_neighbors=min(15, max(2, n - 1)),
                        min_dist=0.1, random_state=seed, metric="euclidean")
    return reducer.fit_transform(X)


def _subsample(X: np.ndarray, y: np.ndarray, n_max: int, seed: int = UMAP_SEED):
    if len(X) <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_max, replace=False)
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Plotly 3D side-by-side
# ─────────────────────────────────────────────────────────────────────────────

def manifold_pair_html(orig: np.ndarray, decoded: np.ndarray, y: np.ndarray,
                       title: str, include_plotlyjs) -> str:
    """Return embedded HTML for a side-by-side ORIG | DECODED 3D UMAP scatter."""
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    # subsample jointly so both panels share the SAME images/colours
    if len(orig) > UMAP_MAX:
        rng = np.random.default_rng(UMAP_SEED)
        idx = rng.choice(len(orig), UMAP_MAX, replace=False)
        orig_s, dec_s, y_s = orig[idx], decoded[idx], y[idx]
    else:
        orig_s, dec_s, y_s = orig, decoded, y

    e_orig = umap3d(orig_s)
    e_dec = umap3d(dec_s)

    classes = np.unique(y_s)
    # qualitative palette (cycled)
    import plotly.express as px
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    cmap = {int(c): palette[i % len(palette)] for i, c in enumerate(classes)}

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("ORIG (original model M)",
                                        "DECODED (downstream-SAE M')"),
                        horizontal_spacing=0.02)

    def add(emb, col, showleg):
        for c in classes:
            mask = y_s == c
            fig.add_trace(go.Scatter3d(
                x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
                mode="markers", name=f"class {int(c)}",
                legendgroup=f"c{int(c)}", showlegend=showleg,
                marker=dict(size=2.5, color=cmap[int(c)], opacity=0.75),
                hovertemplate=f"class {int(c)}<extra></extra>"),
                row=1, col=col)

    add(e_orig, 1, True)
    add(e_dec, 2, False)
    n_leg = len(classes) <= 30
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        height=560, margin=dict(l=0, r=0, t=70, b=0),
        showlegend=n_leg,
        legend=dict(itemsizing="constant", font=dict(size=9)),
        scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""),
        scene2=dict(xaxis_title="", yaxis_title="", zaxis_title=""),
    )
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=False,
                       div_id=None)


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def eval_table(meta: dict) -> str:
    ev = meta.get("eval")
    if not ev:
        return "<p><em>No eval recorded.</em></p>"
    o, d = ev["orig"], ev["decomposed_all_saes"]
    rows = [
        ("ORIGINAL model (no SAE)", o["top1"], o["top5"], o["n"]),
        ("DECOMPOSED (all downstream SAEs inserted)", d["top1"], d["top5"], d["n"]),
    ]
    body = "".join(
        f"<tr><td>{name}</td><td>{t1:.4f}</td><td>{t5:.4f}</td><td>{n}</td></tr>"
        for name, t1, t5, n in rows)
    delta = ev.get("delta_top1", d["top1"] - o["top1"])
    return (
        "<table><thead><tr><th>Configuration</th><th>Top-1</th><th>Top-5</th>"
        f"<th>n images</th></tr></thead><tbody>{body}</tbody></table>"
        f"<p><strong>Δ Top-1 (decomposed − original): {delta:+.4f}</strong></p>")


def dfvu_table(meta: dict) -> str:
    sae = meta.get("sae", {}).get("downstream", {})
    if not sae:
        return "<p><em>No per-block SAE metrics.</em></p>"
    blocks = sorted(sae, key=int)
    rows = "".join(
        f"<tr><td>{b}</td><td>{sae[b]['downstream_fvu']:.4f}</td>"
        f"<td>{sae[b]['recon_fvu']:.3f}</td><td>{sae[b]['l0']:.1f}</td>"
        f"<td>{sae[b]['dead']}/{sae[b]['m']}</td></tr>"
        for b in blocks)
    dfvus = [sae[b]["downstream_fvu"] for b in blocks]
    l0s = [sae[b]["l0"] for b in blocks]
    summ = (f"<p>downstream-FVU: min {min(dfvus):.4f}, median "
            f"{np.median(dfvus):.4f}, max {max(dfvus):.4f} &nbsp;|&nbsp; "
            f"mean L0 {np.mean(l0s):.1f} (of m={sae[blocks[0]]['m']})</p>")
    return (
        "<table><thead><tr><th>Block i</th><th>downstream-FVU</th>"
        "<th>self-recon-FVU</th><th>code L0</th><th>dead/m</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>{summ}")


def quality_table(meta: dict) -> str:
    q = meta.get("quality", {})
    if not q:
        return "<p><em>No decomposition-quality metrics.</em></p>"
    rows = []
    for b in sorted(q, key=int):
        qb = q[b]
        for variant in ("orig", "decoded_downstream"):
            if variant not in qb:
                continue
            v = qb[variant]
            label = "ORIG" if variant == "orig" else "DECODED"
            rows.append(
                f"<tr><td>{b}</td><td>{label}</td>"
                f"<td>{v['knn_purity']:.3f}</td><td>{v['silhouette']:.3f}</td>"
                f"<td>{v['participation_ratio']:.1f}</td>"
                f"<td>{v['linear_probe_acc']:.3f}</td></tr>")
    return (
        "<table><thead><tr><th>Block</th><th>Rep</th><th>kNN purity↑</th>"
        "<th>Silhouette↑</th><th>Particip. ratio</th>"
        "<th>Linear-probe acc↑</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>")


def hypothesis_verdict(meta: dict) -> str:
    """Summarise whether DECODED beats ORIG on the quality metrics."""
    q = meta.get("quality", {})
    if not q:
        return ""
    wins = {"knn_purity": 0, "silhouette": 0, "linear_probe_acc": 0,
            "participation_ratio": 0}
    tot = 0
    deltas = {k: [] for k in wins}
    for b in q:
        qb = q[b]
        if "orig" not in qb or "decoded_downstream" not in qb:
            continue
        tot += 1
        o, d = qb["orig"], qb["decoded_downstream"]
        for k in ("knn_purity", "silhouette", "linear_probe_acc"):
            deltas[k].append(d[k] - o[k])
            if d[k] > o[k]:
                wins[k] += 1
        # lower participation ratio = lower effective dim = "decomposes more easily"
        deltas["participation_ratio"].append(d["participation_ratio"] - o["participation_ratio"])
        if d["participation_ratio"] < o["participation_ratio"]:
            wins["participation_ratio"] += 1
    if tot == 0:
        return ""
    parts = []
    for k in ("knn_purity", "silhouette", "linear_probe_acc", "participation_ratio"):
        md = float(np.mean(deltas[k]))
        arrow = "lower better" if k == "participation_ratio" else "higher better"
        parts.append(f"<li><code>{k}</code> ({arrow}): DECODED wins "
                     f"{wins[k]}/{tot} blocks, mean Δ = {md:+.4f}</li>")
    return ("<p><strong>Hypothesis check</strong> (DECODED cleaner / decomposes "
            "more easily than ORIG):</p><ul>" + "".join(parts) + "</ul>")


# ─────────────────────────────────────────────────────────────────────────────
# Page assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_case_section(case: str, include_plotlyjs) -> str:
    cdir = DATA_DIR / case
    meta_p = cdir / "meta.json"
    if not meta_p.is_file():
        return f"<section><h2>{case}</h2><p><em>No data.</em></p></section>", include_plotlyjs
    meta = json.loads(meta_p.read_text())
    note = meta.get("note", case)
    dataset = meta.get("dataset", "")

    manifolds = ""
    rep_files = sorted(cdir.glob("reps_block*.npz"),
                       key=lambda p: int(p.stem.replace("reps_block", "")))
    for p in rep_files:
        b = int(p.stem.replace("reps_block", ""))
        z = np.load(p)
        if "decoded_downstream" not in z:
            continue
        orig, dec, y = z["orig"], z["decoded_downstream"], z["labels"]
        title = (f"{case} — block {b} CLS token · 3D UMAP · "
                 f"n={len(orig)} images, {len(np.unique(y))} classes")
        manifolds += f"<h4>Block {b}</h4>"
        manifolds += manifold_pair_html(orig, dec, y, title, include_plotlyjs)
        include_plotlyjs = False  # embed the plotly bundle only once

    sec = f"""
<section id="{case}">
  <h2>{case}</h2>
  <p class="note">{note}<br>Eval/rep dataset: <code>{dataset}</code></p>

  <h3>A. Sanity — original vs decomposed model</h3>
  {eval_table(meta)}

  <h3>Per-block downstream reconstruction (FVU)</h3>
  {dfvu_table(meta)}

  <h3>C. Decomposition quality — ORIG vs DECODED</h3>
  {quality_table(meta)}
  {hypothesis_verdict(meta)}

  <h3>B. Manifold comparison (headline) — 3D UMAP, ORIG | DECODED, by class</h3>
  <p>Interactive: drag to rotate, scroll to zoom. Two manifolds only
  (original model representation vs downstream-SAE decoded representation).</p>
  {manifolds}
</section>
"""
    return sec, include_plotlyjs


METHODS = """
<section id="methods">
<h2>Methods &amp; assumptions</h2>
<ul>
<li><strong>Object of study.</strong> Per-block SAEs on transformer block OUTPUTS
(post residual add). The SAE is a pass-through M' = decode(encode(M)); the studied
artefact is the DECODED representation M', not the codes.</li>
<li><strong>Downstream loss (NOT standard reconstruction).</strong>
L = ‖D<sub>next</sub>(M) − D<sub>next</sub>(decode(encode(M)))‖² + λ·‖encode(M)‖₁,
where D<sub>next</sub> = block<sub>i+1</sub> then SAE<sub>i+1</sub> (if present). M
itself may change; what must be preserved is the next stage's output. Model weights
are FROZEN — only the SAE trains.</li>
<li><strong>Iterative-from-output training.</strong> SAEs are trained in DESCENDING
block order (i = N−2 … 0). When training SAE<sub>i</sub>, all already-trained
downstream SAEs (i+1 … N−2) are inserted and FROZEN, so the downstream target is the
representation actually propagated in the decomposed model (already changed by the
downstream SAEs). SAEs are NOT trained independently/in parallel against the clean
model.</li>
<li><strong>Baseline = the original model.</strong> The only comparison is ORIG
(original model representation, no SAE) vs DECODED (downstream-loss SAE's decoded
representation). <strong>NO standard-reconstruction-loss SAE control was trained</strong>
(a standard SAE optimises ‖M−M'‖, a different objective, not a meaningful comparison
here).</li>
<li><strong>Blocks SAE'd.</strong> 0…N−2 (the last block feeds only the classifier,
so it has no "next block"). For 12-block ViTs that is blocks 0…10.</li>
<li><strong>Representations stored.</strong> At a few representative SAE'd layers
(early / mid / late) we store ORIG = clean M and DECODED = the deployed decomposed
model's block-output at that site (all SAEs active). The CLS token (token 0) is used
for the manifold &amp; quality analysis.</li>
<li><strong>Manifold viz.</strong> 3D UMAP, side-by-side ORIG | DECODED, coloured by
class. Rendered as standalone interactive 3D with Plotly (embedded JS, no server).
Bokeh has no native 3D scatter, so Plotly was used for the interactive 3D requirement;
all data are embedded in the HTML.</li>
<li><strong>Decomposition-quality metrics.</strong> kNN class-purity (k=10),
silhouette-by-class, linear-probe accuracy (logistic regression, 70/30 split), and
participation ratio (effective dimensionality (Σλ)²/Σλ²). "Cleaner / decomposes more
easily" is operationalised as: higher class-purity / silhouette / linear-probe
accuracy (classes better separated) and/or lower participation ratio (the
representation lives on a lower-dimensional, more axis-aligned manifold) at matched
downstream-FVU.</li>
<li><strong>ImageNet case.</strong> timm <code>vit_base_patch16_224</code> (ImageNet-1k
pretrained); eval top-1/top-5 on an ImageNet-val subset (HF mirror
<code>evanarlian/imagenet_1k_resized_256</code>). FunnyBirds case: vit_small + linear
probe (50 classes), eval accuracy on a FunnyBirds subset.</li>
</ul>
</section>
"""


def build(css: str):
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    include = "inline"  # embed plotly.js once, in the first plot
    sections = ""
    for case in CASES:
        sec, include = build_case_section(case, include)
        sections += sec
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Downstream-loss SAEs on frozen ViTs</title>
<style>{css}</style>
<style>
  body {{ max-width: 1100px; margin: 1.5rem auto; padding: 0 1rem; }}
  table {{ font-size: 0.85rem; }}
  .note {{ color: #555; }}
  h2 {{ border-top: 2px solid #ccc; padding-top: 1rem; margin-top: 2.5rem; }}
  section#methods {{ background: #f7f7f9; padding: 0.5rem 1.2rem 1rem; border-radius: 8px; }}
</style>
</head><body>
<h1>Downstream-loss SAEs on frozen ViTs</h1>
<p>Per-block sparse autoencoders trained on block outputs with a <strong>downstream
reconstruction loss</strong>, <strong>iteratively from the output</strong>, model
frozen. Headline question: does the downstream loss make the DECODED representations
<em>cleaner / decompose more easily</em> than the ORIGINAL model representation?</p>
<nav><a href="#funny_birds">FunnyBirds (vit_small)</a> ·
<a href="#imagenet">ImageNet (vit_base)</a> ·
<a href="#methods">Methods</a></nav>
{sections}
{METHODS}
<footer><p class="note">Generated by
<code>experiments/scripts/build_sae_downstream_report.py</code> from
<code>data/sae_downstream/</code>.</p></footer>
</body></html>"""
    out = WEB_DIR / "index.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    css_src = REPO_ROOT / "webapp" / "crp_gallery" / "pico.min.css"
    css = css_src.read_text() if css_src.is_file() else ""
    # also copy pico for reference (kept inline above, but copy for parity)
    build(css)
