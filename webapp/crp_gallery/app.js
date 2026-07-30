// Static CRP gallery — reads manifest.json and drives cascading selects.
// No build step, no framework. Empty manifest ⇒ empty selects.
// Cascade: model+dataset → instance (composite × basis) → image sample → layer.
"use strict";

const $ = (id) => document.getElementById(id);
let MANIFEST = { models: {} };

function opt(value, label) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = label;
  return o;
}

function fillSelect(sel, items, labelFn) {
  sel.innerHTML = "";
  if (!items.length) {
    sel.appendChild(opt("", "—"));
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  for (const it of items) sel.appendChild(opt(it, labelFn ? labelFn(it) : it));
}

function curModel() { return MANIFEST.models[$("sel-model").value]; }
function curInstance() { const m = curModel(); return m && m.instances[$("sel-instance").value]; }
function curSample() { const i = curInstance(); return i && i.samples[$("sel-sample").value]; }

// "aggregate" first, then the fixed images in insertion order.
function sampleOrder(inst) {
  const keys = Object.keys(inst.samples);
  keys.sort((a, b) => (a === "aggregate" ? -1 : b === "aggregate" ? 1 : 0));
  return keys;
}

function onModelChange() {
  const m = curModel();
  const instances = m ? Object.keys(m.instances) : [];
  fillSelect($("sel-instance"), instances, (k) => m.instances[k].label);
  onInstanceChange();
}

function onInstanceChange() {
  const inst = curInstance();
  renderComposite(inst);
  const samples = inst ? sampleOrder(inst) : [];
  fillSelect($("sel-sample"), samples, (k) => inst.samples[k].label);
  onSampleChange();
}

function onSampleChange() {
  const inst = curInstance();
  const s = curSample();
  // Chosen input image + its own overall relevance heatmap (single-image only).
  const figs = $("sample-figs");
  const thumb = $("sample-thumb");
  const heatFig = $("sample-heat-fig");
  const heat = $("sample-heat");
  const snote = $("sample-note");
  const normFig = $("sample-norm-fig");
  const norm = $("sample-norm");
  const normLink = $("sample-norm-link");
  if (s && s.image) {
    thumb.src = s.image;
    if (s.heat) { heat.src = s.heat; heatFig.hidden = false; }
    else { heatFig.hidden = true; heat.removeAttribute("src"); }
    if (s.normmap) { norm.src = s.normmap; normLink.href = s.normmap; normFig.hidden = false; }
    else { normFig.hidden = true; norm.removeAttribute("src"); }
    const oodBadge = $("sample-ood");
    if (s.ood_tokens != null) { oodBadge.textContent = `OOD patch tokens: ${s.ood_tokens}`; oodBadge.hidden = false; }
    else { oodBadge.textContent = ""; oodBadge.hidden = true; }
    renderXai(s);
    figs.hidden = false;
    snote.textContent = "Local analysis: detectors ranked by relevance to THIS input. "
      + "Per figure, column 1 = the query input, columns 2.. = that detector's dataset "
      + "representatives. Sub-rows: image · conditional relevance heatmap · receptive-field "
      + "crop (image clipped to the heatmap's high-relevance region, low relevance faded).";
    snote.hidden = false;
  } else {
    figs.hidden = true; thumb.removeAttribute("src"); heat.removeAttribute("src");
    snote.hidden = true;
    $("xai-block").hidden = true;
  }
  // Layers under this (instance, sample), ordered by block.
  const layers = s ? Object.keys(s.layers) : [];
  layers.sort((a, b) => (s.layers[a].block - s.layers[b].block));
  fillSelect($("sel-layer"), layers, (ln) => {
    const L = s.layers[ln];
    return `block ${L.block} · ${L.site}`;
  });
  onLayerChange();
}

const XAI_LABELS = { lrp: "LRP / CRP", chefer: "Chefer", rollout: "Rollout", occlusion: "Occlusion Δp⁺" };

// Labelled competing-saliency row: input | LRP | Chefer | rollout | occlusion.
function renderXai(s) {
  const block = $("xai-block");
  const row = $("xai-row");
  row.innerHTML = "";
  if (!s || !s.xai) { block.hidden = true; return; }
  const order = (MANIFEST.xai_order || ["lrp", "chefer", "rollout", "occlusion"]).filter((m) => s.xai[m]);
  if (!order.length) { block.hidden = true; return; }
  const caps = MANIFEST.xai_captions || {};
  const cell = (src, label, prov) => {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = src; img.alt = label; img.loading = "lazy";
    const cap = document.createElement("figcaption");
    cap.innerHTML = `<span class="m">${label}</span>` + (prov ? `<span class="prov">${prov}</span>` : "");
    fig.appendChild(img); fig.appendChild(cap);
    return fig;
  };
  if (s.image) row.appendChild(cell(s.image, "input", "the query image"));
  for (const m of order) row.appendChild(cell(s.xai[m], XAI_LABELS[m] || m, caps[m] || ""));
  block.hidden = false;
}

function renderComposite(inst) {
  const box = $("composite");
  const note = $("layer-note");
  if (!inst) { box.hidden = true; note.hidden = true; return; }
  const k = inst.composite || {};
  box.hidden = false;
  $("comp-name").textContent = inst.label || k.name || "";
  $("comp-desc").textContent = k.description || "";
  $("comp-class").textContent = k.class || "";
  $("comp-site").textContent = "site: " + (k.site || "");
  $("comp-isolates").textContent = k.isolates ? ("isolates: " + k.isolates) : "reference";
  $("comp-source").textContent = k.build_source || "(source unavailable)";
  if (inst.concept_desc) {
    note.innerHTML = `<strong>${inst.basis}</strong> — ${inst.concept_desc}`;
    note.hidden = false;
  } else {
    note.hidden = true;
  }
}

function onLayerChange() {
  const s = curSample();
  const entries = $("entries");
  entries.innerHTML = "";
  const ln = $("sel-layer").value;
  const L = s && ln ? s.layers[ln] : null;
  const list = (L && L.entries) || [];
  $("empty-msg").hidden = list.length > 0;
  for (const e of list) {
    const fig = document.createElement("figure");
    const a = document.createElement("a");
    a.href = e.pdf;
    const img = document.createElement("img");
    img.src = e.png;
    img.alt = `detector ${e.id}`;
    img.loading = "lazy";
    a.appendChild(img);
    const cap = document.createElement("figcaption");
    const rel = (e.relevance != null) ? ` · rel ${(+e.relevance).toPrecision(3)}` : "";
    const rank = (e.rank != null) ? ` · rank ${e.rank}` : "";
    cap.innerHTML = `detector <strong>#${e.id}</strong>${rank}${rel} · [<a href="${e.pdf}">pdf</a>]`;
    fig.appendChild(a);
    fig.appendChild(cap);
    entries.appendChild(fig);
  }
}

async function main() {
  try {
    const res = await fetch("manifest.json", { cache: "no-store" });
    if (res.ok) MANIFEST = await res.json();
  } catch (_) { /* keep empty */ }
  $("stamp").textContent = MANIFEST.generated || "";
  const models = Object.keys(MANIFEST.models || {});
  fillSelect($("sel-model"), models, (md) => (MANIFEST.models[md].label || md));
  $("sel-model").addEventListener("change", onModelChange);
  $("sel-instance").addEventListener("change", onInstanceChange);
  $("sel-sample").addEventListener("change", onSampleChange);
  $("sel-layer").addEventListener("change", onLayerChange);
  onModelChange();
}

main();
