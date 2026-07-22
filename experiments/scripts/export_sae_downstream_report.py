"""Build the static report page for the downstream-loss SAE study.

Reads ``data/sae_downstream/<case>/meta.json`` + ``reps_block*.npz`` (written by
``experiments.sae_downstream``) and emits ``webapp/sae_downstream/index.html``: a
single static page with TWO sections (funny_birds, imagenet), each holding
  * sanity eval table (original vs decomposed-all-SAEs classification metrics),
  * per-block downstream-FVU / L0 summary table,
  * decomposition-quality metric table (ORIG vs DECODED vs standard-SAE control),
  * interactive 3D manifold plots (UMAP) of ORIG vs DECODED (vs standard),
    side by side, coloured by class label.

3D rendering note
-----------------
The brief asked for "interactive 3D bokeh". Bokeh has NO native 3D scatter
(its renderer is 2D; 3D needs a fragile vis.js custom extension). We therefore
render the genuine, rotatable, standalone-HTML interactive 3D scatter with
**plotly** (``include_plotlyjs="cdn"``, embedded as ``full_html=False`` divs).
This honours the intent (interactive, 3D, standalone, class-coloured) with a
robust library. Documented as an assumption on the page.

Run::

    python -m experiments.scripts.export_sae_downstream_report
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "sae_downstream"
WEB_DIR = REPO_ROOT / "webapp" / "sae_downstream"
FIG_DIR = REPO_ROOT / "figures" / "sae_downstream"

CASES = [("funny_birds", "vit_small + linear probe — FunnyBirds (50 classes)"),
         ("imagenet", "timm vit_base_patch16_224 — ImageNet-1k val")]

REP_NAMES = {  # npz key -> (display label, plot colour-set ok)
    "orig": "ORIGINAL  M  (clean model)",
    "decoded_downstream": "DECODED  M'  (downstream-loss SAE, deployed)",
    "decoded_standard": "DECODED  M'  (standard-recon SAE, control)",
}


# ── 3D embedding ─────────────────────────────────────────────────────────────

def embed3d(X: np.ndarray, y: np.ndarray, seed: int = 0, n_max: int = 1500):
    """3D UMAP of X (subsample to n_max for speed). Returns (emb, y_sub)."""
    import umap
    rng = np.random.default_rng(seed)
    if len(X) > n_max:
        idx = rng.choice(len(X), n_max, replace=False)
        X, y = X[idx], y[idx]
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                        random_state=seed, metric="cosine")
    return reducer.fit_transform(X), y


# ── plotly 3D scatter ────────────────────────────────────────────────────────

def scatter3d_html(emb: np.ndarray, labels: np.ndarray, title: str) -> str:
    import plotly.graph_objects as go
    uniq = np.unique(labels)
    # categorical colour by label
    fig = go.Figure()
    import plotly.express as px
    palette = (px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
               + px.colors.qualitative.Light24)
    for i, c in enumerate(uniq):
        mask = labels == c
        fig.add_trace(go.Scatter3d(
            x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
            mode="markers", name=str(int(c)),
            marker=dict(size=2.5, color=palette[i % len(palette)], opacity=0.8),
            showlegend=len(uniq) <= 20,
            hovertemplate=f"class {int(c)}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=0, r=0, t=28, b=0), height=420,
        scene=dict(xaxis_title="", yaxis_title="", zaxis_title="",
                   xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                   zaxis=dict(showticklabels=False)),
        legend=dict(font=dict(size=8), itemsizing="constant"),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, default_height="420px")


# ── tables ───────────────────────────────────────────────────────────────────

def _f(x, p=4):
    try:
        return f"{float(x):.{p}f}"
    except (TypeError, ValueError):
        return "—"


def eval_table(meta: dict) -> str:
    ev = meta.get("eval")
    if not ev:
        return "<p class='empty'>no eval recorded</p>"
    rows = [("Original model", ev["orig"]),
            ("Decomposed (all downstream-loss SAEs)", ev["decomposed_all_saes"])]
    if "decomposed_standard" in ev:
        rows.append(("Decomposed (all standard-recon SAEs, control)", ev["decomposed_standard"]))
    body = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{_f(d['top1'])}</td>"
        f"<td>{_f(d['top5'])}</td><td>{d['n']}</td></tr>"
        for name, d in rows)
    delta = ev.get("delta_top1")
    cap = (f"<tr><td><b>Δ top-1 (decomposed − original)</b></td>"
           f"<td colspan=3><b>{_f(delta)}</b></td></tr>")
    return ("<table><thead><tr><th>model</th><th>top-1</th><th>top-5</th>"
            f"<th>n</th></tr></thead><tbody>{body}{cap}</tbody></table>")


def sae_table(meta: dict) -> str:
    sae = meta.get("sae", {}).get("downstream", {})
    std = meta.get("sae", {}).get("standard", {})
    if not sae:
        return "<p class='empty'>no SAE metrics</p>"
    blocks = sorted(sae, key=int)
    head = ("<tr><th>block i</th><th>downstream FVU</th><th>M-space recon FVU</th>"
            "<th>L0</th><th>dead / m</th><th>cond. on SAE_{i+1}</th>")
    if std:
        head += "<th>std FVU (ctrl)</th><th>std L0</th>"
    head += "</tr>"
    rows = []
    for b in blocks:
        d = sae[b]
        r = (f"<tr><td>{b}</td><td>{_f(d['downstream_fvu'])}</td>"
             f"<td>{_f(d['recon_fvu'], 3)}</td><td>{_f(d['l0'], 1)}</td>"
             f"<td>{d['dead']} / {d['m']}</td>"
             f"<td>{'yes' if d.get('has_downstream_sae') else 'no'}</td>")
        if std and b in std:
            r += f"<td>{_f(std[b]['downstream_fvu'])}</td><td>{_f(std[b]['l0'], 1)}</td>"
        elif std:
            r += "<td>—</td><td>—</td>"
        rows.append(r + "</tr>")
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def quality_table(meta: dict) -> str:
    q = meta.get("quality")
    if not q:
        return "<p class='empty'>no quality metrics</p>"
    keys = ["knn_purity", "silhouette", "participation_ratio", "linear_probe_acc"]
    kn = {"knn_purity": "kNN purity ↑", "silhouette": "silhouette ↑",
          "participation_ratio": "particip. ratio (eff. dim) ↓",
          "linear_probe_acc": "linear-probe acc ↑"}
    repkeys = [("orig", "ORIG"), ("decoded_downstream", "DECODED (downstream)"),
               ("decoded_standard", "DECODED (standard ctrl)")]
    out = []
    for b in sorted(q, key=int):
        qb = q[b]
        head = "<tr><th>representation</th>" + "".join(f"<th>{kn[k]}</th>" for k in keys) + "</tr>"
        rows = []
        for rk, rl in repkeys:
            if rk not in qb:
                continue
            cells = "".join(f"<td>{_f(qb[rk][k], 2 if k=='participation_ratio' else 3)}</td>"
                            for k in keys)
            rows.append(f"<tr><td>{rl}</td>{cells}</tr>")
        out.append(f"<h4>block {b}</h4><table><thead>{head}</thead>"
                   f"<tbody>{''.join(rows)}</tbody></table>")
    return "".join(out)


# ── per-case section ─────────────────────────────────────────────────────────

def build_section(case: str, title: str) -> str:
    cdir = DATA_DIR / case
    meta_path = cdir / "meta.json"
    if not meta_path.is_file():
        return f"<section><h2>{html.escape(title)}</h2><p class='empty'>not computed yet</p></section>"
    meta = json.loads(meta_path.read_text())
    parts = [f"<section id='{case}'>", f"<h2>{html.escape(title)}</h2>"]
    parts.append(f"<p class='meta-line'>dataset: {html.escape(str(meta.get('dataset','?')))} · "
                 f"{html.escape(str(meta.get('note','')))}</p>")

    parts.append("<h3>A · Sanity: original vs decomposed classification</h3>")
    parts.append(eval_table(meta))
    parts.append("<h3>Per-block SAE training summary (iterative-from-output)</h3>")
    parts.append(sae_table(meta))

    parts.append("<h3>C · Decomposition-quality metrics</h3>")
    parts.append("<p class='meta-line'>On the CLS-token representation of the eval "
                 "subset. <b>kNN purity</b> = mean fraction of a point's 10 nearest "
                 "neighbours sharing its class (higher ⇒ classes locally cleaner). "
                 "<b>silhouette</b> = class separation/cohesion (higher ⇒ tighter, "
                 "better-separated clusters). <b>participation ratio</b> = effective "
                 "dimensionality (Σλ)²/Σλ² of the covariance (lower ⇒ rep concentrated "
                 "on fewer directions = a simpler / more easily decomposed manifold). "
                 "<b>linear-probe acc</b> = held-out logistic-regression accuracy on the "
                 "rep (higher ⇒ classes more linearly accessible). \"Decomposes more "
                 "easily\" ≡ higher kNN/silhouette/linear-probe AND/OR lower "
                 "participation ratio for DECODED vs ORIG.</p>")
    parts.append(quality_table(meta))

    parts.append("<h3>B · Manifold comparison (interactive 3D UMAP)</h3>")
    parts.append("<p class='meta-line'>Rotatable / zoomable. Each point = one image's "
                 "CLS-token representation at the block, coloured by class. ORIG = clean "
                 "model M; DECODED = the deployed decomposed model's rep M' at that site "
                 "(downstream-loss SAE, with all SAEs active). Drag to rotate.</p>")
    rep_blocks = sorted([int(p.stem.replace("reps_block", ""))
                         for p in cdir.glob("reps_block*.npz")])
    band = {rep_blocks[0]: "early", rep_blocks[len(rep_blocks)//2]: "mid",
            rep_blocks[-1]: "late"} if rep_blocks else {}
    for b in rep_blocks:
        z = np.load(cdir / f"reps_block{b}.npz")
        y = z["labels"]
        present = [k for k in REP_NAMES if k in z.files]
        cols = []
        for k in present:
            emb, ysub = embed3d(z[k], y)
            cols.append((REP_NAMES[k], scatter3d_html(emb, ysub, REP_NAMES[k])))
        grid = "".join(f"<div class='cell'><div class='cell-h'>{html.escape(lbl)}</div>{h}</div>"
                       for lbl, h in cols)
        parts.append(f"<h4>block {b} <span class='pill'>{band.get(b,'')}</span></h4>"
                     f"<div class='grid3'>{grid}</div>")

    parts.append("</section>")
    return "\n".join(parts)


# ── page ─────────────────────────────────────────────────────────────────────

PAGE = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Downstream-loss SAEs on frozen ViTs</title>
<link rel="stylesheet" href="pico.min.css">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body {{ max-width: 1180px; margin: 0 auto; padding: 1rem; }}
  table {{ font-size: 0.85rem; }}
  .meta-line {{ font-size: 0.85rem; color: var(--pico-muted-color); }}
  .grid3 {{ display: grid; gap: 0.5rem; grid-template-columns: 1fr; }}
  @media (min-width: 900px) {{ .grid3 {{ grid-template-columns: repeat(2, 1fr); }} }}
  .cell {{ border: 1px solid var(--pico-muted-border-color); border-radius: 6px; padding: 0.25rem; }}
  .cell-h {{ font-size: 0.78rem; font-weight: 600; margin: 0.2rem 0.4rem; }}
  .pill {{ display:inline-block; padding:0.05rem 0.5rem; border-radius:1rem;
           background: var(--pico-secondary-background); font-size:0.7rem; }}
  section {{ margin-bottom: 2.5rem; }}
  details {{ font-size: 0.85rem; }}
  h4 {{ margin-top: 1rem; }}
</style>
</head>
<body>
<header>
<h1>Downstream-loss SAEs on frozen Vision Transformers</h1>
<p class="meta-line">Preliminary study. Per-block sparse autoencoders trained so the
DECODED representation preserves the <b>downstream</b> (next-stage) output rather than
its own value, on a frozen ViT. Two model cases below.</p>
</header>

<section>
<h2>Methods &amp; assumptions</h2>
<details open>
<summary>What was done (click to collapse)</summary>
{methods}
</details>
</section>

{sections}

<footer><p class="meta-line">Generated by
<code>experiments/scripts/export_sae_downstream_report.py</code> from
<code>experiments/sae_downstream.py</code> outputs.</p></footer>
</body>
</html>
"""

METHODS = """
<ul>
<li><b>Setup.</b> For a frozen ViT, at each block <i>i</i> (output after the residual
add) an SAE is spliced as a pass-through M' = decode(encode(M)). Model weights stay
FROZEN; only the SAE trains.</li>
<li><b>Downstream loss.</b> Instead of vanilla ‖M − M'‖², we minimise the DOWNSTREAM
reconstruction ‖D<sub>next</sub>(M) − D<sub>next</sub>(M')‖² + λ·‖encode(M)‖₁, where
D<sub>next</sub> = block<sub>i+1</sub> followed by SAE<sub>i+1</sub> if present. M may
change; what the consuming stage produces must not.</li>
<li><b>★ Iterative-from-output training (critical).</b> The per-block SAEs are NOT
independent: an SAE changes what flows downstream. So they are trained in DESCENDING
block order i = N−2 … 0. When training SAE<sub>i</sub>, all already-trained downstream
SAEs (i+1 … N−2) are inserted and FROZEN, and the target D<sub>next</sub>(M) is the
DEPLOYED next-stage output (block<sub>i+1</sub> then SAE<sub>i+1</sub>). This makes the
composite self-consistent. M itself is collected from the clean model because the
downstream SAEs sit AFTER block i (upstream SAEs 0…i−1 do not exist yet during
SAE<sub>i</sub>'s training).</li>
<li><b>SAE.</b> Bricken-style untied-L1 SAE (unit-norm decoder, tied pre-decoder bias),
expansion m = {expansion}·d. Activations centered + RMS-scaled for training; the scaling
is folded into the deployed params. Dead-latent resampling at half-training.</li>
<li><b>Hyper-params.</b> λ = {l1} (chosen from a sweep for a sane sparsity/fidelity
tradeoff), {steps} steps, lr {lr}, image-batch {img_batch}, trained on a {n_train}-image
SUBSET (preliminary / cheap). SAEs on blocks 0…N−2 (the last block feeds only the head).</li>
<li><b>Eval (A).</b> Original vs decomposed (all SAEs spliced) top-1/top-5 on the eval
subset; per-block downstream FVU = ‖D<sub>next</sub>(M)−D<sub>next</sub>(M')‖²/var(D<sub>next</sub>(M)).</li>
<li><b>Manifold (B).</b> At early/mid/late SAE'd blocks we store the CLS-token of ORIG
(clean M) and DECODED (the deployed decomposed model's rep M' at that site). 3D UMAP
(cosine metric), interactive, coloured by class.</li>
<li><b>Quality (C).</b> kNN class-purity, silhouette, participation ratio (effective
dim), held-out linear-probe accuracy on ORIG vs DECODED (vs a standard-recon SAE
control trained at the same sites).</li>
<li><b>Assumptions / caveats.</b> (1) Preliminary: small subset, few steps — numbers are
indicative. (2) "Interactive 3D bokeh" was requested; Bokeh has no native 3D scatter, so
the genuine rotatable 3D is rendered with <b>plotly</b> (documented deviation). (3)
CLS-token used as the per-image representation for the manifold/quality (cleaner class
structure than mean-pooling). (4) Manifold/quality computed on the eval subset.</li>
</ul>
"""


def main():
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    # copy styling
    src_css = REPO_ROOT / "webapp" / "crp_gallery" / "pico.min.css"
    if src_css.is_file():
        (WEB_DIR / "pico.min.css").write_bytes(src_css.read_bytes())

    # pull hyper-params from whichever case has them
    hp = {}
    for case, _ in CASES:
        mp = DATA_DIR / case / "meta.json"
        if mp.is_file():
            m = json.loads(mp.read_text())
            ds = m.get("sae", {}).get("downstream", {})
            if ds:
                any_b = next(iter(ds.values()))
                hp = dict(expansion=any_b["m"] // any_b["d"], l1=any_b["l1_coeff"],
                          steps=any_b["steps"], lr=any_b["lr"],
                          img_batch=any_b["img_batch"], n_train="≈800")
                break
    hp = hp or dict(expansion="?", l1="?", steps="?", lr="?", img_batch="?", n_train="?")
    methods = METHODS.format(**hp)

    sections = "\n".join(build_section(c, t) for c, t in CASES)
    page = PAGE.format(methods=methods, sections=sections)
    (WEB_DIR / "index.html").write_text(page)
    print(f"wrote {WEB_DIR/'index.html'} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
