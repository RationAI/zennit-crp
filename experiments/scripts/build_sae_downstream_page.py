"""Build the static SAE-downstream results page from the stored reps + meta.

For each case (funny_birds, imagenet) and each stored rep block, fit a 3D t-SNE
on the ORIGINAL representation and on the downstream-SAE DECODED representation
(cls-token, fixed test subset) and render them side-by-side as interactive 3D
plotly scenes. Assemble one HTML with both sections + the eval / decomposition-
quality tables + a methods note.

(Spec asked for bokeh; bokeh has no native 3D scatter, so plotly is used for the
interactive 3D — it emits the same standalone JS/HTML for webshare. Reducer is
t-SNE since umap-learn isn't installed; the spec allowed "UMAP or t-SNE".)

Run: VIRTUAL_ENV=/home/claude/venvs/zennit-crp \
     /home/claude/venvs/zennit-crp/bin/python -m experiments.scripts.build_sae_downstream_page
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "sae_downstream"
OUT = REPO / "webapp" / "sae_downstream"
CASES = [("funny_birds", "ViT-S/16 + linear probe · FunnyBirds (50 classes)"),
         ("imagenet", "ViT-B/16 ImageNet-1k pretrained · ImageNet-val (1000 classes)")]
REP_BLOCKS = [1, 5, 10]
N_PTS = 1000          # subsample points per scene
SEED = 0


def tsne3(X: np.ndarray) -> np.ndarray:
    """3D t-SNE with PCA pre-reduction (fast, stable)."""
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=SEED).fit_transform(X)
    perp = max(5, min(30, (len(X) - 1) // 3))
    return TSNE(n_components=3, init="pca", perplexity=perp,
                random_state=SEED).fit_transform(X)


def scene_pair(orig: np.ndarray, dec: np.ndarray, lab: np.ndarray, title: str):
    rng = np.random.default_rng(SEED)
    if len(orig) > N_PTS:
        idx = rng.choice(len(orig), N_PTS, replace=False)
        orig, dec, lab = orig[idx], dec[idx], lab[idx]
    eo, ed = tsne3(orig), tsne3(dec)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("ORIGINAL  M", "DECODED  M′ (downstream-SAE)"))
    mk = dict(size=2.2, color=lab, colorscale="Turbo", opacity=0.8, showscale=False)
    fig.add_trace(go.Scatter3d(x=eo[:, 0], y=eo[:, 1], z=eo[:, 2], mode="markers",
                               marker=mk, hovertext=lab, name="orig"), 1, 1)
    fig.add_trace(go.Scatter3d(x=ed[:, 0], y=ed[:, 1], z=ed[:, 2], mode="markers",
                               marker=mk, hovertext=lab, name="decoded"), 1, 2)
    fig.update_layout(title=title, height=460, margin=dict(l=0, r=0, t=46, b=0),
                      showlegend=False)
    for s in ("scene", "scene2"):
        fig.update_layout({s: dict(xaxis_title="", yaxis_title="", zaxis_title="",
                                   xaxis_showticklabels=False, yaxis_showticklabels=False,
                                   zaxis_showticklabels=False)})
    return fig


def eval_table(meta: dict) -> str:
    e = meta.get("eval", {})
    if not e:
        return ""
    o, d = e["orig"], e["decomposed_all_saes"]
    return (f"<table><tr><th>model</th><th>top-1</th><th>top-5</th></tr>"
            f"<tr><td>original</td><td>{o['top1']:.3f}</td><td>{o['top5']:.3f}</td></tr>"
            f"<tr><td>decomposed (all SAEs)</td><td>{d['top1']:.3f}</td><td>{d['top5']:.3f}</td></tr>"
            f"<tr><td>Δ top-1</td><td colspan=2>{e['delta_top1']:+.3f}</td></tr></table>")


def fvu_table(meta: dict) -> str:
    ds = meta.get("sae", {}).get("downstream", {})
    if not ds:
        return ""
    rows = "".join(
        f"<tr><td>{b}</td><td>{ds[str(b)]['downstream_fvu']:.3f}</td>"
        f"<td>{ds[str(b)]['l0']:.0f}</td><td>{ds[str(b)]['dead']}</td></tr>"
        for b in sorted((int(k) for k in ds)))
    return ("<table><tr><th>block</th><th>downstream FVU</th><th>L0</th><th>dead</th></tr>"
            + rows + "</table>")


def quality_table(meta: dict) -> str:
    q = meta.get("quality", {})
    if not q:
        return ""
    head = ("<tr><th>block</th><th>variant</th><th>kNN purity</th>"
            "<th>linear-probe</th><th>silhouette</th><th>participation ratio</th></tr>")
    rows = ""
    for b in sorted(q, key=int):
        for k, lbl in (("orig", "original"), ("decoded_downstream", "decoded M′")):
            if k in q[b]:
                v = q[b][k]
                rows += (f"<tr><td>{b}</td><td>{lbl}</td><td>{v['knn_purity']:.3f}</td>"
                         f"<td>{v['linear_probe_acc']:.3f}</td><td>{v['silhouette']:.3f}</td>"
                         f"<td>{v['participation_ratio']:.1f}</td></tr>")
    return "<table>" + head + rows + "</table>"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # vendor the lightweight CSS next to the page (reuse the gallery's copy)
    css_src = REPO / "webapp" / "crp_gallery" / "pico.min.css"
    if css_src.is_file():
        (OUT / "pico.min.css").write_bytes(css_src.read_bytes())
    sections = []
    gidx = 0   # global plot index for unique ids
    for key, title in CASES:
        meta_p = DATA / key / "meta.json"
        if not meta_p.is_file():
            continue
        meta = json.loads(meta_p.read_text())
        plots = []
        for b in REP_BLOCKS:
            p = DATA / key / f"reps_block{b}.npz"
            if not p.is_file():
                continue
            z = np.load(p)
            if "decoded_downstream" not in z:
                continue
            fig = scene_pair(z["orig"], z["decoded_downstream"], z["labels"], f"block {b}")
            # Lazy: embed the figure spec as JSON; a single IntersectionObserver
            # renders it only when scrolled into view and purges it when off-screen,
            # so live WebGL contexts stay ≤ a couple (6 figs × 2 scenes = 12 contexts
            # at once exceeds the browser cap and silently drops some plots).
            pid = f"plot{gidx}"
            gidx += 1
            plots.append(
                f'<h4>Block {b}</h4>'
                f'<div class="lazyplot" id="{pid}" style="min-height:480px"></div>'
                f'<script type="application/json" data-for="{pid}">{fig.to_json()}</script>')
        sections.append(f"""
        <section>
          <h2>{title}</h2>
          <div class="grid2">
            <div><h3>Classification sanity</h3>{eval_table(meta)}</div>
            <div><h3>Per-block SAE</h3>{fvu_table(meta)}</div>
          </div>
          <h3>Decomposition quality (original vs decoded)</h3>
          {quality_table(meta)}
          <h3>Representation manifolds (3D t-SNE, colored by class)</h3>
          {''.join(plots)}
        </section>""")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Downstream-loss SAE — preliminary</title>
<link rel="stylesheet" href="pico.min.css">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script>window.MathJax={{tex:{{inlineMath:[['$','$']],displayMath:[['$$','$$']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
 body{{max-width:1100px;margin:0 auto;padding:1rem}}
 .grid2{{display:grid;gap:1rem;grid-template-columns:1fr}}
 @media(min-width:760px){{.grid2{{grid-template-columns:1fr 1fr}}}}
 table{{font-size:.85rem}} th,td{{text-align:right;padding:.15rem .5rem}} td:first-child,th:first-child{{text-align:left}}
 section{{border-top:2px solid var(--pico-muted-border-color);margin-top:2rem;padding-top:1rem}}
 h4{{margin:.6rem 0 .2rem}} .eq{{overflow-x:auto}}
</style></head><body>
<header><h1>Downstream-loss SAE — preliminary results</h1>

<h3>Setup</h3>
<p>Frozen ViT, residual blocks $B_0,\\dots,B_{{L-1}}$. Let $M_i=B_i(\\cdot)\\in\\mathbb R^{{N\\times d}}$
be block $i$'s output (after the residual add — the tensor copied to the next block and the skip).
An SAE is the map $S_\\theta(M)=W_{{\\rm dec}}\\,\\sigma(W_{{\\rm enc}}(M-b_{{\\rm dec}})+b_{{\\rm enc}})+b_{{\\rm dec}}$,
$\\sigma=\\mathrm{{ReLU}}$, code $f=\\sigma(\\cdot)\\in\\mathbb R^{{m}}$. We splice $S$ in as a pass-through at $M_i$.</p>

<h3>Objective</h3>
<p>A standard SAE minimises self-reconstruction $\\lVert M-S(M)\\rVert^2$. <b>Here instead</b> the SAE
is trained so the <i>consuming</i> layer's output is preserved — $M_i$ itself may change, but $B_{{i+1}}$
must not notice:</p>
<p class="eq">$$\\mathcal L_i=\\big\\lVert\\,\\widetilde B_{{i+1}}(M_i)-\\widetilde B_{{i+1}}(S_{{\\theta_i}}(M_i))\\,\\big\\rVert_2^2\\;+\\;\\lambda\\lVert f_i\\rVert_1 .$$</p>
<p><b>Iterative-from-output:</b> train descending $i=L\\!-\\!2,\\dots,0$; the consumer is the <i>deployed</i>
next stage $\\widetilde B_{{i+1}}=S_{{\\theta_{{i+1}}}}\\!\\circ B_{{i+1}}$ with the already-trained, frozen
downstream SAE in place (so the target is the representation actually propagated in the decomposed model).
Model weights stay frozen; only $\\theta_i$ trains. The last block is skipped (its output feeds only the
cls-token head). <b>Object of study:</b> the <i>decoded</i> $M'_i=S_{{\\theta_i}}(M_i)$ — not the codes $f_i$.</p>

<h3>Metrics</h3>
<p class="eq">downstream&nbsp;FVU $=\\dfrac{{\\lVert B(M)-B(M')\\rVert^2}}{{\\mathrm{{Var}}\\,B(M)}}$ &nbsp;·&nbsp;
participation ratio $=\\dfrac{{(\\sum_k\\lambda_k)^2}}{{\\sum_k\\lambda_k^2}}$ (eigvals $\\lambda_k$ of $\\mathrm{{Cov}}$, = effective dim) &nbsp;·&nbsp;
kNN purity, linear-probe accuracy, silhouette — on $M$ vs $M'$.</p>

<details><summary>Configuration &amp; assumptions</summary>
<ul>
<li><b>Baseline = original model representation</b> $M$ (no standard-reconstruction SAE control).</li>
<li>Dictionary $m=\\tfrac12 d$ (undercomplete bottleneck): $m{{=}}192$ (ViT-S, $d{{=}}384$), $m{{=}}384$ (ViT-B, $d{{=}}768$).
Per-block centre + global-RMS normalisation; $\\lambda=10^{{-4}}$ (FunnyBirds) / $10^{{-5}}$ (ImageNet); 2500 steps, Adam $10^{{-3}}$.</li>
<li>FunnyBirds: full clean train set. ImageNet: 5/class val subset.</li>
<li>Manifolds: 3D <b>t-SNE</b> (PCA-50 pre-reduction) on a {N_PTS}-point class-spanning subset; plotly 3D
(bokeh has no native 3D scatter); reducer t-SNE as umap-learn is not installed.</li>
</ul></details></header>
{''.join(sections)}
<footer><small>Generated by experiments/scripts/build_sae_downstream_page.py</small></footer>
<script>
// Render each 3D plot only while on-screen; purge off-screen to free WebGL
// contexts (browsers cap simultaneous contexts → otherwise some plots fail to load).
const _obs = new IntersectionObserver((entries) => {{
  entries.forEach((e) => {{
    const d = e.target;
    if (e.isIntersecting) {{
      if (!d.dataset.rendered) {{
        const s = JSON.parse(document.querySelector('script[data-for="' + d.id + '"]').textContent);
        Plotly.newPlot(d, s.data, s.layout, {{responsive: true, displaylogo: false}});
        d.dataset.rendered = "1";
      }}
    }} else if (d.dataset.rendered) {{
      Plotly.purge(d);
      d.dataset.rendered = "";
    }}
  }});
}}, {{rootMargin: "150px"}});
document.querySelectorAll('.lazyplot').forEach((d) => _obs.observe(d));
</script>
</body></html>"""
    (OUT / "index.html").write_text(page)
    print(f"wrote {OUT/'index.html'} ({len(page)} bytes), sections={len(sections)}")


if __name__ == "__main__":
    main()
