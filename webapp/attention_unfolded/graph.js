// Attention Unfolded — engine. Builds the cell graph, renders matrices, handles selection.
// Model specifics (dims, tokens, flags, notes) come from window.MODELS[id] (see model_*.js).
(function () {
  'use strict';

  // ---------- numerics ----------
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const r2 = x => Math.round(x * 100) / 100;
  const fmt = x => r2(x).toFixed(2).replace('-', '−');
  const par = x => x < 0 ? `(${fmt(x)})` : fmt(x);

  // ---------- state ----------
  let cells, els, model, stageOrder;

  function mk(id, label, val, { deps = [], heads = [], expr = '', kind = 'mid', stage = '' } = {}) {
    const d = new Set(), hs = new Set(heads);
    for (const x of deps) { const c = typeof x === 'string' ? cells.get(x) : x; d.add(c.id); c.deps.forEach(y => d.add(y)); c.heads.forEach(h => hs.add(h)); }
    const cell = { id, label, val, deps: d, heads: hs, expr, kind, stage };
    cells.set(id, cell);
    if (!stageOrder.includes(stage)) stageOrder.push(stage);
    return cell;
  }

  // ---------- op ledger (forward order of the unfolded module) ----------
  function ledgerFor(m) {
    const rows = [];
    let n = 0;
    const add = (module, kind, what, where) => { n += 1; rows.push({ n, module, kind, what, where }); return n; };
    const L = {};
    L.qkv = add('qkv', 'Linear', `qkv = a·W_qkvᵀ${m.qkvBias ? ' + b_qkv' : ''}, (${m.N},${m.C})→(${m.N},${3 * m.C})${m.qkvBias ? '' : ' — no bias'}`, 'stage 1');
    L.split = add('split', 'reindex', `chunk 3 on last dim → q_flat, k_flat, v_flat, each (${m.N},${m.C})`, 'stage 2');
    const p0 = n + 1; n += 3; rows.push({ n: `${p0}–${n}`, module: 'q/k/v_lrp_probe', kind: 'Identity', what: `hookable taps on the flat (${m.N},${m.C}) tensors, no math`, where: 'stage 2' }); L.probe = [p0, p0 + 1, p0 + 2];
    L.toheads = add('_to_heads ×3', 'reindex', `(${m.N},${m.C})→(${m.H},${m.N},${m.HD}): head h takes channels h·${m.HD} .. h·${m.HD}+${m.HD - 1}`, m.rope ? 'stage 3' : 'stage 3');
    const qn = n + 1; n += 2; rows.push({ n: `${qn}–${n}`, module: 'q_norm, k_norm', kind: 'Identity', what: m.qkNormNote, where: m.rope ? 'stage 3' : 'stage 3' }); L.qnorm = qn; L.knorm = qn + 1;
    if (m.rope) {
      L.ropeq = add('rope_q', 'elementwise', `RoPE on q, patch tokens only (${m.rope.prefix} prefix tokens skipped); rotate-half pairs (d, d+${m.HD / 2})`, 'stage 3');
      L.ropek = add('rope_k', 'elementwise', 'RoPE on k, same angles', 'stage 3');
      L.typeas = add('q.type_as(v), k.type_as(v)', 'Identity', 'dtype cast, no-op in fp32', 'stage 3');
    }
    const sN = m.rope ? 4 : 3;
    L.scale = add('scale_q', 'elementwise', `q̃ = q · ${fmt(m.scale)} (hd^-½, constant, not a parameter)`, `stage ${sN}`);
    L.scores = add('qk_scores', 'bilinear matmul', `S_h = q̃_h · k_hᵀ, (${m.H},${m.N},${m.N})`, `stage ${sN}`);
    L.mask = add('add_mask', 'Identity', 'S + attn_mask; attn_mask=None → S unchanged', `stage ${sN + 1}`);
    L.softmax = add('softmax', 'elementwise-per-row', 'w_h[i,:] = softmax(S_h[i,:]) over keys j', `stage ${sN + 1}`);
    L.drop = add('attn_drop', 'Identity', 'dropout, identity at eval', `stage ${sN + 1}`);
    L.context = add('context', 'bilinear matmul', `ctx_h = w_h · v_h, (${m.H},${m.N},${m.HD})`, `stage ${sN + 2}`);
    L.reshape = add('reshape', 'reindex', `merge heads (${m.H},${m.N},${m.HD})→(${m.N},${m.C}): o[i, h·${m.HD}+d] = ctx_h[i,d]`, `stage ${sN + 3}`);
    L.norm = add('norm', 'Identity', m.postNormNote, `stage ${sN + 3}`);
    L.proj = add('proj', 'Linear', `attn = o·W_projᵀ + b_proj, (${m.N},${m.C})→(${m.N},${m.C})`, `stage ${sN + 3}`);
    L.projdrop = add('proj_drop', 'Identity', 'dropout, identity at eval', `stage ${sN + 3}`);
    return { rows, L, total: n, sN };
  }

  // ---------- RoPE constants (DINOv3 rotate-half layout) ----------
  function ropeTables(m) {
    // returns per token: angles[d] for d < HD/2, from coords [y,x] in [-1,1]
    const { HD, rope } = m; const nper = HD / 4;
    const periods = Array.from({ length: nper }, (_, k) => Math.pow(rope.temperature, 2 * k / (HD / 2)));
    return m.tokens.map(t => {
      if (!t.coord) return null;
      const ang = [];
      for (let axis = 0; axis < 2; axis++) for (let k = 0; k < nper; k++) ang.push(2 * Math.PI * t.coord[axis] / periods[k]);
      return { ang, periods };
    });
  }

  // ---------- forward pass, cell by cell ----------
  function build(m) {
    cells = new Map(); els = new Map(); stageOrder = [];
    const { N, C, H, HD } = m; const SCALE = m.scale;
    const rnd = mulberry32(m.seed);
    const randn = () => { const u = 1 - rnd(), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const headOf = p => Math.floor((p % C) / HD);
    const partOf = p => ['q', 'k', 'v'][Math.floor(p / C)];
    const led = ledgerFor(m); const L = led.L;
    const tk = i => m.tokens[i].label;

    const a = Array.from({ length: N }, (_, i) => Array.from({ length: C }, (_, c) => mk(`a_${i}_${c}`, `a[${i},${c}]`, r2(randn()), { kind: 'input', stage: 'a', expr: `a[${i},${c}] = norm1(x)[${i},${c}]  — block-side input (${tk(i)}), taken as given\n(depends on x[${i},:] and γ1[${c}], β1[${c}]; out of scope here)` })));

    const Wqkv = Array.from({ length: 3 * C }, (_, p) => Array.from({ length: C }, (_, c) => mk(`Wqkv_${p}_${c}`, `W_qkv[${p},${c}]`, r2(randn() * 0.6), { kind: 'param', heads: [headOf(p)], stage: 'W_qkv', expr: `parameter · row ${p} = ${partOf(p)} of head ${headOf(p)}, head-channel d=${p % HD}; column ${c} reads a[·,${c}]` })));
    const bqkv = m.qkvBias ? Array.from({ length: 3 * C }, (_, p) => mk(`bqkv_${p}`, `b_qkv[${p}]`, r2(randn() * 0.3), { kind: 'param', heads: [headOf(p)], stage: 'b_qkv', expr: `parameter · bias of ${partOf(p)} head ${headOf(p)} d=${p % HD}` })) : null;
    const qkvT = Wqkv.map((row, p) => Array.from({ length: N }, (_, i) => {
      let s = bqkv ? bqkv[p].val : 0; const terms = [];
      for (let c = 0; c < C; c++) { s += row[c].val * a[i][c].val; terms.push(`${par(row[c].val)}·${par(a[i][c].val)}`); }
      return mk(`qkvT_${p}_${i}`, `qkvᵀ[${p},${i}]`, s, { deps: [...row, ...a[i], ...(bqkv ? [bqkv[p]] : [])], stage: 'qkvᵀ', expr: `op ${L.qkv} qkv (Linear${m.qkvBias ? '' : ', bias=False'})\nqkvᵀ[${p},${i}] = Σ_c W_qkv[${p},c]·a[${i},c]${bqkv ? ` + b_qkv[${p}]` : ''}\n= ${terms.join(' + ')}${bqkv ? ` + ${par(bqkv[p].val)}` : ''}\n= ${fmt(s)}\n\nthis is ${partOf(p)}_flat[${i}, ${p % C}] → head ${headOf(p)}, d=${p % HD}; token ${i} = ${tk(i)}` });
    }));

    const flat = {};
    [['q', 0], ['k', 1], ['v', 2]].forEach(([nm, s]) => {
      flat[nm] = Array.from({ length: N }, (_, i) => Array.from({ length: C }, (_, c) => { const p = s * C + c, src = qkvT[p][i];
        return mk(`${nm}flat_${i}_${c}`, `${nm}_flat[${i},${c}]`, src.val, { deps: [src], stage: `${nm}_flat`, expr: `op ${L.split} split (chunk 3, last dim) → op ${L.probe[s]} ${nm}_lrp_probe (Identity)\n${nm}_flat[${i},${c}] = qkv[${i},${p}] = qkvᵀ[${p},${i}] = ${fmt(src.val)}\n\nno math; hookable "${nm === 'q' ? 'qk' : nm === 'v' ? 'value' : 'k'}" probe site. head ${headOf(c)}, d=${c % HD}` }); }));
    });

    // rope constants
    let ropeT = null, cosC = null, sinC = null;
    if (m.rope) {
      ropeT = ropeTables(m);
      cosC = []; sinC = [];
      m.tokens.forEach((t, i) => {
        if (!ropeT[i]) { cosC.push(null); sinC.push(null); return; }
        const { ang, periods } = ropeT[i]; const nper = HD / 4;
        cosC.push(ang.map((th, d) => mk(`cos_${i}_${d}`, `cos θ[${i},${d}]`, Math.cos(th), { kind: 'const', stage: 'RoPE cos/sin', expr: `RoPE constant (not a parameter)\nθ[${i},${d}] = 2π · coord_${d < nper ? 'y' : 'x'}(${tk(i)}) / period_${d % nper}\n= 2π · ${par(t.coord[d < nper ? 0 : 1])} / ${fmt(periods[d % nper])} = ${fmt(th)} rad\ncos θ = ${fmt(Math.cos(th))}\n\nperiod_k = T^(2k/(hd/2)), T=${m.rope.temperature}; coords are 0.5-centred patch positions mapped to [−1,1] (make_coords_dinov3, 'separate')` })));
        sinC.push(ang.map((th, d) => mk(`sin_${i}_${d}`, `sin θ[${i},${d}]`, Math.sin(th), { kind: 'const', stage: 'RoPE cos/sin', expr: `RoPE constant (not a parameter)\nθ[${i},${d}] = ${fmt(th)} rad (see cos cell)\nsin θ = ${fmt(Math.sin(th))}` })));
      });
    }
    function rope(nm, h, i, src, opn) {
      // src: array of HD cells for token i, head h. returns rotated cells
      const half = HD / 2;
      if (!m.rope || !ropeT[i]) return src.map((c, d) => mk(`${nm}r_${h}_${i}_${d}`, `${nm}${h}ʳ[${i},${d}]`, c.val, { deps: [c], stage: `${nm}ʳ h${h}`, expr: `op ${opn} rope_${nm} (RotaryEmbedding)\n${nm}${h}ʳ[${i},${d}] = ${nm}${h}[${i},${d}] = ${fmt(c.val)}\n\ntoken ${i} = ${tk(i)} is a prefix token (cls/register): RoPE skipped, identity.` }));
      const out = [];
      for (let d = 0; d < HD; d++) {
        const lo = d < half, j = lo ? d : d - half, cs = cosC[i][j], sn = sinC[i][j];
        const A = src[d], B = src[lo ? d + half : d - half];
        const val = lo ? A.val * cs.val - B.val * sn.val : A.val * cs.val + B.val * sn.val;
        out.push(mk(`${nm}r_${h}_${i}_${d}`, `${nm}${h}ʳ[${i},${d}]`, val, { deps: [A, B, cs, sn], stage: `${nm}ʳ h${h}`, expr: `op ${opn} rope_${nm} (apply_rot_embed_cat, half=True)\n${nm}${h}ʳ[${i},${d}] = ${nm}[${i},${d}]·cos θ[${i},${j}] ${lo ? '−' : '+'} ${nm}[${i},${lo ? d + half : d - half}]·sin θ[${i},${j}]\n= ${par(A.val)}·${par(cs.val)} ${lo ? '−' : '+'} ${par(B.val)}·${par(sn.val)}\n= ${fmt(val)}\n\nrotate-half: channel ${d} pairs with channel ${lo ? d + half : d - half}; both turn by θ[${i},${j}] (${j < HD / 4 ? 'y' : 'x'} axis). Constants only, no parameters; same angles in every head.` }));
      }
      return out;
    }

    const heads = [];
    for (let h = 0; h < H; h++) {
      const q = Array.from({ length: N }, (_, i) => Array.from({ length: HD }, (_, d) => { const c = h * HD + d, src = flat.q[i][c];
        return mk(`q_${h}_${i}_${d}`, `q${h}[${i},${d}]`, src.val, { deps: [src], stage: `q h${h}`, expr: `op ${L.toheads} _to_heads (reshape (${N},${C})→(${N},${H},${HD}), transpose) → op ${L.qnorm} q_norm (Identity)\nq${h}[${i},${d}] = q_flat[${i},${c}] = ${fmt(src.val)}   (c = h·hd + d = ${h}·${HD} + ${d})` }); }));
      const k = Array.from({ length: N }, (_, j) => Array.from({ length: HD }, (_, d) => { const c = h * HD + d, src = flat.k[j][c];
        return mk(`k_${h}_${j}_${d}`, `k${h}[${j},${d}]`, src.val, { deps: [src], stage: `k h${h}`, expr: `op ${L.toheads} _to_heads → op ${L.knorm} k_norm (Identity)\nk${h}[${j},${d}] = k_flat[${j},${c}] = ${fmt(src.val)}   (pure reindex)` }); }));
      const v = Array.from({ length: N }, (_, j) => Array.from({ length: HD }, (_, d) => { const c = h * HD + d, src = flat.v[j][c];
        return mk(`v_${h}_${j}_${d}`, `v${h}[${j},${d}]`, src.val, { deps: [src], stage: `v h${h}`, expr: `op ${L.toheads} _to_heads\nv${h}[${j},${d}] = v_flat[${j},${c}] = ${fmt(src.val)}   (pure reindex)` }); }));
      let qr = q, kr = k;
      if (m.rope) { qr = q.map((row, i) => rope('q', h, i, row, L.ropeq)); kr = k.map((row, j) => rope('k', h, j, row, L.ropek)); }
      const qt = qr.map((row, i) => row.map((src, d) => mk(`qt_${h}_${i}_${d}`, `q̃${h}[${i},${d}]`, SCALE * src.val, { deps: [src], stage: `q̃ h${h}`, expr: `${m.rope ? `op ${L.typeas} type_as (Identity) → ` : ''}op ${L.scale} scale_q (ScaleByConstant, elementwise)\nq̃${h}[${i},${d}] = ${m.rope ? 'q' + h + 'ʳ' : 'q' + h}[${i},${d}] · hd^-½ = ${par(src.val)} · ${fmt(SCALE)} = ${fmt(SCALE * src.val)}\n\nscale is a graph constant, not a parameter.` })));
      const kT = Array.from({ length: HD }, (_, d) => Array.from({ length: N }, (_, j) => { const src = kr[j][d];
        return mk(`kT_${h}_${d}_${j}`, `k${h}${m.rope ? 'ʳ' : ''}ᵀ[${d},${j}]`, src.val, { deps: [src], stage: `kᵀ h${h}`, expr: `${m.rope ? `op ${L.typeas} type_as (Identity) → ` : ''}.transpose(-2,-1)\nk${h}${m.rope ? 'ʳ' : ''}ᵀ[${d},${j}] = ${m.rope ? 'k' + h + 'ʳ' : 'k' + h}[${j},${d}] = ${fmt(src.val)}   (pure reindex)` }); }));
      const S = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => { let s = 0; const t = [];
        for (let d = 0; d < HD; d++) { s += qt[i][d].val * kT[d][j].val; t.push(`${par(qt[i][d].val)}·${par(kT[d][j].val)}`); }
        return mk(`S_${h}_${i}_${j}`, `S${h}[${i},${j}]`, s, { deps: [...qt[i], ...kT.map(r => r[j])], stage: `S h${h}`, expr: `op ${L.scores} qk_scores (BilinearMatmul)\nS${h}[${i},${j}] = Σ_d q̃${h}[${i},d]·k${h}ᵀ[d,${j}]\n= ${t.join(' + ')}\n= ${fmt(s)}\n\nquery ${tk(i)} · key ${tk(j)}, head ${h}.` }); }));
      const w = S.map((row, i) => { const mx = Math.max(...row.map(c => c.val)); const ex = row.map(c => Math.exp(c.val - mx)); const Z = ex.reduce((x, y) => x + y, 0);
        return row.map((c, j) => mk(`w_${h}_${i}_${j}`, `w${h}[${i},${j}]`, ex[j] / Z, { deps: row, stage: `w h${h}`, expr: `op ${L.mask} add_mask (+None → Identity) → op ${L.softmax} softmax (last dim = keys) → op ${L.drop} attn_drop (Identity, eval)\nw${h}[${i},${j}] = exp(S${h}[${i},${j}]) / Σ_j' exp(S${h}[${i},j'])\n= exp(${fmt(c.val)}) / (${row.map(x => `exp(${fmt(x.val)})`).join(' + ')})\n= ${fmt(ex[j] / Z)}\n\ndepends on the whole key row S${h}[${i},:] — all ${N} tokens.` })); });
      const ctx = Array.from({ length: N }, (_, i) => Array.from({ length: HD }, (_, d) => { let s = 0; const t = [];
        for (let j = 0; j < N; j++) { s += w[i][j].val * v[j][d].val; t.push(`${par(w[i][j].val)}·${par(v[j][d].val)}`); }
        return mk(`ctx_${h}_${i}_${d}`, `ctx${h}[${i},${d}]`, s, { deps: [...w[i], ...v.map(r => r[d])], stage: `ctx h${h}`, expr: `op ${L.context} context (BilinearMatmul)\nctx${h}[${i},${d}] = Σ_j w${h}[${i},j]·v${h}[j,${d}]\n= ${t.join(' + ')}\n= ${fmt(s)}\n\nweighted over ALL key tokens j; still head ${h} only.` }); }));
      heads.push({ q, k, v, qr, kr, qt, kT, S, w, ctx });
    }

    const oT = Array.from({ length: C }, (_, c) => Array.from({ length: N }, (_, i) => { const h = Math.floor(c / HD), d = c % HD, src = heads[h].ctx[i][d];
      return mk(`oT_${c}_${i}`, `oᵀ[${c},${i}]`, src.val, { deps: [src], stage: 'oᵀ', expr: `op ${L.reshape} reshape (ReshapeMergeHeads) → op ${L.norm} norm (Identity)\noᵀ[${c},${i}] = ctx${h}[${i},${d}] = ${fmt(src.val)}   (c = h·hd + d = ${h}·${HD} + ${d})` }); }));
    const Wproj = Array.from({ length: C }, (_, c) => Array.from({ length: C }, (_, c2) => mk(`Wproj_${c}_${c2}`, `W_proj[${c},${c2}]`, r2(randn() * 0.6), { kind: 'param', heads: [Math.floor(c2 / HD)], stage: 'W_proj', expr: `parameter · output channel ${c}; column ${c2} reads o[·,${c2}] = ctx of head ${Math.floor(c2 / HD)}, d=${c2 % HD}\n(W_proj[:, ${Math.floor(c2 / HD) * HD}:${Math.floor(c2 / HD) * HD + HD}] is head ${Math.floor(c2 / HD)}'s output projection W_O)` })));
    const bproj = Array.from({ length: C }, (_, c) => mk(`bproj_${c}`, `b_proj[${c}]`, r2(randn() * 0.3), { kind: 'param', stage: 'b_proj', expr: `parameter · output bias, channel ${c}; belongs to no head` }));
    const attnT = Wproj.map((row, c) => Array.from({ length: N }, (_, i) => { let s = bproj[c].val; const t = [];
      for (let c2 = 0; c2 < C; c2++) { s += row[c2].val * oT[c2][i].val; t.push(`${par(row[c2].val)}·${par(oT[c2][i].val)}`); }
      const byHead = Array.from({ length: H }, (_, h) => `[head ${h}] ${t.slice(h * HD, (h + 1) * HD).join(' + ')}`);
      return mk(`attnT_${c}_${i}`, `attnᵀ[${c},${i}]`, s, { deps: [...row, ...oT.map(r => r[i]), bproj[c]], stage: 'attnᵀ', expr: `op ${L.proj} proj (Linear) → op ${L.projdrop} proj_drop (Identity, eval)\nattnᵀ[${c},${i}] = Σ_c' W_proj[${c},c']·oᵀ[c',${i}] + b_proj[${c}]\n= ${byHead.join('\n+ ')}\n+ ${par(bproj[c].val)}\n= ${fmt(s)}\n\nFIRST place a number depends on both heads' parameters: the sum over c' runs across the head-0 columns and head-1 columns of W_proj.` });
    }));

    return { a, Wqkv, bqkv, qkvT, flat, heads, oT, Wproj, bproj, attnT, cosC, sinC, ropeT, led };
  }

  // ---------- rendering ----------
  function matrix({ title, shape, cap, grid, rowLabels, colLabels, rowClass, colClass }) {
    const cols = grid[0].length;
    const el = document.createElement('div'); el.className = 'matrix';
    el.style.gridTemplateColumns = `74px repeat(${cols},52px)`;
    const t = document.createElement('div'); t.className = 'title'; t.innerHTML = `${title}${shape ? `<small>${shape}</small>` : ''}`; el.appendChild(t);
    const corner = document.createElement('div'); corner.className = 'lbl col'; el.appendChild(corner);
    colLabels.forEach((l, c) => { const d = document.createElement('div'); d.className = 'lbl col' + (colClass ? ' ' + colClass(c) : ''); d.textContent = l; el.appendChild(d); });
    grid.forEach((row, r) => {
      const rl = document.createElement('div'); rl.className = 'lbl row' + (rowClass ? ' ' + rowClass(r) : ''); rl.textContent = rowLabels[r]; el.appendChild(rl);
      row.forEach(cell => {
        const d = document.createElement('div');
        if (!cell) { d.className = 'cell empty'; d.textContent = '—'; el.appendChild(d); return; }
        d.className = 'cell ' + cell.kind; d.tabIndex = 0;
        d.dataset.id = cell.id; d.dataset.heads = [...cell.heads].sort().join('');
        d.textContent = fmt(cell.val); d.title = cell.label;
        d.addEventListener('click', () => select(cell.id));
        d.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(cell.id); } });
        if (!els.has(cell.id)) els.set(cell.id, []); els.get(cell.id).push(d); el.appendChild(d);
      });
    });
    if (cap) { const c = document.createElement('div'); c.className = 'cap'; c.textContent = cap; el.appendChild(c); }
    return el;
  }
  function arrow(lines) { const a = document.createElement('div'); a.className = 'arrow'; a.innerHTML = lines.map(l => l === '→' ? '<b>⟶</b>' : `<span class="${/Identity/.test(l) ? 'id' : ''}">${l}</span>`).join(''); return a; }
  function mmBlock(cornerText, left, top, result) {
    const b = document.createElement('div'); b.className = 'mm';
    const c = document.createElement('div'); c.className = 'corner'; c.textContent = cornerText; b.appendChild(c);
    b.appendChild(top);
    const lg = document.createElement('div'); lg.className = 'opgroup'; (Array.isArray(left) ? left : [left]).forEach(x => lg.appendChild(x)); b.appendChild(lg);
    b.appendChild(result);
    return b;
  }
  function stage(title, eq, note, content) {
    const s = document.createElement('section'); s.className = 'stage';
    s.innerHTML = `<div class="stage-head"><h2>${title}</h2><span class="eq">${eq}</span>${note ? `<span class="note">${note}</span>` : ''}</div>`;
    const sc = document.createElement('div'); sc.className = 'scroll'; (Array.isArray(content) ? content : [content]).forEach(x => sc.appendChild(x)); s.appendChild(sc);
    return s;
  }
  function rowOf(items) { const r = document.createElement('div'); r.className = 'row-of'; items.forEach(x => r.appendChild(x)); return r; }
  function group(items) { const g = document.createElement('div'); g.className = 'opgroup'; items.forEach(x => g.appendChild(x)); return g; }

  function render(m, G) {
    const main = document.getElementById('main'); main.innerHTML = '';
    const { N, C, H, HD } = m; const L = G.led.L; const sN = G.led.sN;
    const headOf = p => Math.floor((p % C) / HD), partOf = p => ['q', 'k', 'v'][Math.floor(p / C)];
    const tokL = m.tokens.map(t => t.label), qryL = tokL.map(s => 'q:' + s), keyL = tokL.map(s => 'k:' + s);
    const chL = n => Array.from({ length: n }, (_, i) => `c${i}`), dL = n => Array.from({ length: n }, (_, i) => `d${i}`);
    const rowsQKV = G.Wqkv.map((_, p) => `p${p} ${partOf(p)}·h${headOf(p)}·d${p % HD}`);
    const hc = p => 'h' + headOf(p);

    // stage 0: ledger
    {
      const s = document.createElement('section'); s.className = 'stage';
      s.innerHTML = `<div class="stage-head"><h2>0 · Forward order, all ${G.led.total} ops of <span class="eq">${m.source}.forward</span></h2><span class="note">Exactly the module calls in the source, in order. Identity ops are still listed so nothing is hidden; they are drawn as labelled arrows below.</span></div>`;
      const sc = document.createElement('div'); sc.className = 'scroll';
      const tb = document.createElement('table'); tb.className = 'ops';
      tb.innerHTML = '<tr><th>#</th><th>module</th><th>kind</th><th>what it does (eval)</th><th>drawn in</th></tr>' + G.led.rows.map(r => `<tr><td>${r.n}</td><td>${r.module}</td><td class="${r.kind === 'Identity' ? 'k-id' : /Linear|matmul/.test(r.kind) ? 'k-lin' : ''}">${r.kind}</td><td>${r.what}</td><td>${r.where}</td></tr>`).join('');
      sc.appendChild(tb); s.appendChild(sc); main.appendChild(s);
    }
    // stage 1: qkv
    {
      const aT = G.a[0].map((_, c) => G.a.map(row => row[c]));
      const left = [matrix({ title: 'W_qkv', shape: `(${3 * C}×${C})`, cap: `parameters. Rows 0–${C - 1} → q, ${C}–${2 * C - 1} → k, ${2 * C}–${3 * C - 1} → v; within each, rows 0–${HD - 1} belong to head 0, rows ${HD}–${2 * HD - 1} to head 1.`, grid: G.Wqkv, rowLabels: rowsQKV, colLabels: chL(C), rowClass: hc })];
      if (G.bqkv) left.push(matrix({ title: '+ b_qkv', shape: `(${3 * C})`, cap: 'bias per row', grid: G.bqkv.map(x => [x]), rowLabels: G.bqkv.map((_, p) => `p${p}`), colLabels: [''], rowClass: hc }));
      const top = matrix({ title: 'aᵀ', shape: `(${C}×${N})`, cap: 'a = norm1(x), block-side input, taken as given. Columns = tokens.', grid: aT, rowLabels: chL(C), colLabels: tokL });
      const res = matrix({ title: 'qkvᵀ', shape: `(${3 * C}×${N})`, cap: `= W_qkv · aᵀ${G.bqkv ? ' + b_qkv' : ''}. Each row = one output channel for all tokens.`, grid: G.qkvT, rowLabels: rowsQKV, colLabels: tokL, rowClass: hc });
      main.appendChild(stage(`1 · op ${L.qkv} qkv (Linear${m.qkvBias ? '' : ', no bias'})`, `qkvᵀ[p,i] = Σ_c W_qkv[p,c]·a[i,c]${m.qkvBias ? ' + b_qkv[p]' : ''}`, 'Shown transposed so parameter rows stay rows.', mmBlock('rows of W_qkv × columns of aᵀ', left, top, res)));
    }
    // stage 2: split + probes
    {
      const colsFlat = chL(C).map((s, c) => `${s} h${headOf(c)}`);
      const items = [['q', 'q_lrp_probe', '"qk" site'], ['k', 'k_lrp_probe', '—'], ['v', 'v_lrp_probe', '"value" site']].map(([nm, probe, site], s) =>
        group([arrow([`qkvᵀ rows ${s * C}–${s * C + C - 1}`, '→', `op ${L.split} split`, `op ${L.probe[s]} ${probe}`, '(Identity)']),
          matrix({ title: `${nm}_flat`, shape: `(${N}×${C})`, cap: `untransposed back to (tokens × channels). ${site !== '—' ? 'CRP ' + site + '.' : ''}`, grid: G.flat[nm], rowLabels: tokL, colLabels: colsFlat, colClass: c => 'h' + headOf(c) })]));
      main.appendChild(stage(`2 · op ${L.split} split, ops ${L.probe[0]}–${L.probe[2]} probes`, `q_flat = qkv[:, 0:${C}], k_flat = qkv[:, ${C}:${2 * C}], v_flat = qkv[:, ${2 * C}:${3 * C}]`, `Pure reindex plus three Identity taps. Columns 0–${HD - 1} of each are head 0, ${HD}–${2 * HD - 1} head 1.`, rowOf(items)));
    }
    // stage 3 (rope only): to_heads + RoPE
    if (m.rope) {
      const half = HD / 2, nper = HD / 4;
      const patch = m.tokens.map((t, i) => i).filter(i => G.ropeT[i]);
      const angL = Array.from({ length: half }, (_, d) => `θ${d} ${d < nper ? 'y' : 'x'}`);
      const cosM = matrix({ title: 'cos θ', shape: `(${patch.length}×${half})`, cap: 'RoPE constants per patch token, per pair index. Prefix tokens have no row: not rotated.', grid: patch.map(i => G.cosC[i]), rowLabels: patch.map(i => tokL[i]), colLabels: angL });
      const sinM = matrix({ title: 'sin θ', shape: `(${patch.length}×${half})`, cap: `θ[i,d] = 2π·coord_axis(i)/period_k; period_k = ${m.rope.temperature}^(2k/${half}), k = d mod ${nper}. hd/4 = ${nper} period(s) here.`, grid: patch.map(i => G.sinC[i]), rowLabels: patch.map(i => tokL[i]), colLabels: angL });
      const blocks = [group([cosM, sinM])];
      for (let h = 0; h < H; h++) { const hh = G.heads[h];
        blocks.push(group([
          matrix({ title: `q${h}`, shape: `(${N}×${HD})`, cap: `= q_flat[:, ${h * HD}:${h * HD + HD}] after op ${L.toheads} _to_heads, op ${L.qnorm} q_norm (Identity)`, grid: hh.q, rowLabels: tokL, colLabels: dL(HD) }),
          arrow([`op ${L.ropeq} rope_q`, 'rotate-half', `(d, d+${half})`, '→']),
          matrix({ title: `q${h}ʳ`, shape: `(${N}×${HD})`, cap: 'rotated query. Prefix rows unchanged.', grid: hh.qr, rowLabels: tokL, colLabels: dL(HD) }),
          matrix({ title: `k${h}`, shape: `(${N}×${HD})`, cap: `= k_flat[:, ${h * HD}:${h * HD + HD}] after op ${L.toheads} _to_heads, op ${L.knorm} k_norm (Identity)`, grid: hh.k, rowLabels: tokL, colLabels: dL(HD) }),
          arrow([`op ${L.ropek} rope_k`, 'same θ', '→']),
          matrix({ title: `k${h}ʳ`, shape: `(${N}×${HD})`, cap: 'rotated key', grid: hh.kr, rowLabels: tokL, colLabels: dL(HD) }),
        ]));
      }
      main.appendChild(stage(`3 · ops ${L.toheads}–${L.typeas}: to_heads, qk-norm, RoPE, type_as`, `qʳ[i,d] = q[i,d]·cosθ[i,d] − q[i,d+${half}]·sinθ[i,d];  qʳ[i,d+${half}] = q[i,d+${half}]·cosθ[i,d] + q[i,d]·sinθ[i,d]`, `Per token, per head: a 2-D rotation of channel pairs by position-dependent constant angles. ${m.rope.prefix} prefix token(s) (cls/registers) are passed through. Same angles for every head; no parameters. Applied to q and k only, never v.`, rowOf(blocks)));
    }
    // scale + scores
    {
      const blocks = [];
      for (let h = 0; h < H; h++) { const hh = G.heads[h];
        const src = m.rope ? hh.qr : hh.q;
        const qm = matrix({ title: m.rope ? `q${h}ʳ` : `q${h}`, shape: `(${N}×${HD})`, cap: m.rope ? 'from stage 3' : `= q_flat[:, ${h * HD}:${h * HD + HD}]`, grid: src, rowLabels: tokL, colLabels: dL(HD) });
        const ar = arrow(m.rope ? [`op ${L.typeas} type_as`, '(Identity)', `op ${L.scale} scale_q`, `× ${fmt(m.scale)}`, '→'] : [`op ${L.qnorm} q_norm`, '(Identity)', `op ${L.scale} scale_q`, `× ${fmt(m.scale)}`, '→']);
        const qt = matrix({ title: `q̃${h}`, shape: `(${N}×${HD})`, cap: 'scaled query', grid: hh.qt, rowLabels: tokL, colLabels: dL(HD) });
        const top = matrix({ title: `k${h}${m.rope ? 'ʳ' : ''}ᵀ`, shape: `(${HD}×${N})`, cap: m.rope ? 'rotated key from stage 3, transposed' : `= k_flat[:, ${h * HD}:${h * HD + HD}]ᵀ after op ${L.knorm} k_norm (Identity)`, grid: hh.kT, rowLabels: dL(HD), colLabels: keyL });
        const res = matrix({ title: `S${h}`, shape: `(${N}×${N})`, cap: 'scores, query × key', grid: hh.S, rowLabels: qryL, colLabels: keyL });
        blocks.push(mmBlock(`head ${h}: ${m.rope ? '' : `op ${L.toheads} _to_heads, then `}q̃ · kᵀ`, [qm, ar, qt], top, res));
      }
      main.appendChild(stage(`${sN} · ops ${m.rope ? L.typeas : L.toheads}–${L.scores}: ${m.rope ? 'type_as, ' : 'to_heads, qk-norm, '}scale, scores`, `S_h[i,j] = Σ_d (${fmt(m.scale)}·q${m.rope ? 'ʳ' : ''}_h[i,d])·k${m.rope ? 'ʳ' : ''}_h[j,d]`, `${m.rope ? '' : `Head h takes channels h·${HD} .. h·${HD}+${HD - 1} of q_flat, k_flat, v_flat. `}Heads never touch each other here.`, rowOf(blocks)));
    }
    // softmax
    {
      const items = [];
      for (let h = 0; h < H; h++) { const hh = G.heads[h];
        items.push(group([matrix({ title: `S${h}`, shape: `(${N}×${N})`, cap: `scores from stage ${sN}`, grid: hh.S, rowLabels: qryL, colLabels: keyL }),
          arrow([`op ${L.mask} add_mask`, '+ None (Identity)', `op ${L.softmax} softmax`, 'over keys', `op ${L.drop} attn_drop`, '(Identity)', '→']),
          matrix({ title: `w${h}`, shape: `(${N}×${N})`, cap: 'attention weights, each row sums to 1', grid: hh.w, rowLabels: qryL, colLabels: keyL })]));
      }
      main.appendChild(stage(`${sN + 1} · ops ${L.mask}–${L.drop}: mask, softmax, dropout`, 'w_h[i,j] = exp(S_h[i,j]) / Σ_j\' exp(S_h[i,j\'])', 'Only nonlinearity in the branch. Each w row depends on its whole S row, so on every key token; still one head.', rowOf(items)));
    }
    // context
    {
      const blocks = [];
      for (let h = 0; h < H; h++) { const hh = G.heads[h];
        blocks.push(mmBlock(`head ${h}: op ${L.context} context, w · v`,
          matrix({ title: `w${h}`, shape: `(${N}×${N})`, cap: `weights from stage ${sN + 1}`, grid: hh.w, rowLabels: qryL, colLabels: keyL }),
          matrix({ title: `v${h}`, shape: `(${N}×${HD})`, cap: `= v_flat[:, ${h * HD}:${h * HD + HD}] via op ${L.toheads} _to_heads${m.rope ? ' (no RoPE on v)' : ''}`, grid: hh.v, rowLabels: keyL, colLabels: dL(HD) }),
          matrix({ title: `ctx${h}`, shape: `(${N}×${HD})`, cap: 'per-head context', grid: hh.ctx, rowLabels: tokL, colLabels: dL(HD) })));
      }
      main.appendChild(stage(`${sN + 2} · op ${L.context} context`, 'ctx_h[i,d] = Σ_j w_h[i,j]·v_h[j,d]', 'Token mixing: ctx for token i sums over all key tokens j. Head-local.', rowOf(blocks)));
    }
    // merge + proj
    {
      const hOf = c => Math.floor(c / HD);
      const left = [matrix({ title: 'W_proj', shape: `(${C}×${C})`, cap: `parameters. Columns 0–${HD - 1} read head 0, columns ${HD}–${2 * HD - 1} read head 1: W_proj[:, 0:${HD}] is head 0's W_O, W_proj[:, ${HD}:${2 * HD}] is head 1's.`, grid: G.Wproj, rowLabels: chL(C).map(s => 'out ' + s), colLabels: chL(C).map((s, c) => `${s} h${hOf(c)}`), colClass: c => 'h' + hOf(c) }),
        matrix({ title: '+ b_proj', shape: `(${C})`, cap: 'no head', grid: G.bproj.map(x => [x]), rowLabels: chL(C), colLabels: [''] })];
      const top = matrix({ title: 'oᵀ', shape: `(${C}×${N})`, cap: `= [ctx0 | ctx1]ᵀ, op ${L.reshape} reshape (merge heads) then op ${L.norm} norm (Identity)`, grid: G.oT, rowLabels: chL(C).map((s, c) => `${s} = h${hOf(c)} d${c % HD}`), colLabels: tokL, rowClass: c => 'h' + hOf(c) });
      const res = matrix({ title: 'attnᵀ', shape: `(${C}×${N})`, cap: `= W_proj · oᵀ + b_proj, then op ${L.projdrop} proj_drop (Identity). Attention-branch output${m.afterNote ? '; ' + m.afterNote : '.'}`, grid: G.attnT, rowLabels: chL(C), colLabels: tokL });
      const call = document.createElement('div'); call.className = 'callout';
      call.innerHTML = `<b>Where heads mix.</b> Nothing before this line combines parameters of different heads. Each attnᵀ[c,i] = Σ_{c'∈head 0} W_proj[c,c']·o[i,c'] + Σ_{c'∈head 1} W_proj[c,c']·o[i,c'] + b_proj[c]: a head-0 term plus a head-1 term, added. The mixing is a plain sum of two per-head projections. ${m.mixNote || ''}`;
      main.appendChild(stage(`${sN + 3} · ops ${L.reshape}–${L.projdrop}: merge heads, norm, proj, dropout`, 'attnᵀ[c,i] = Σ_c\' W_proj[c,c\']·oᵀ[c\',i] + b_proj[c]', '', [mmBlock('rows of W_proj × columns of oᵀ', left, top, res), call]));
    }
  }

  // ---------- interaction ----------
  function select(id) {
    const c = cells.get(id); if (!c) return;
    clearHighlight();          // a cell click returns the grid to its default
    const down = new Set(); for (const [oid, o] of cells) if (o.deps.has(id)) down.add(oid);
    for (const [oid, list] of els) for (const el of list) {
      el.classList.remove('sel', 'down', 'up', 'dim');
      if (oid === id) el.classList.add('sel');
      else if (down.has(oid)) el.classList.add('down');
      else if (c.deps.has(oid)) el.classList.add('up');
      else el.classList.add('dim');
    }
    document.getElementById('sel-label').textContent = `${c.label} = ${fmt(c.val)}`;
    const hs = [...c.heads].sort();
    const hl = document.getElementById('sel-heads');
    if (c.kind === 'param') hl.innerHTML = hs.length ? `parameter of <b class="h${hs[0]}">head ${hs[0]}</b>` : 'parameter, belongs to no head';
    else if (c.kind === 'input') hl.innerHTML = 'input; no attention parameters';
    else if (c.kind === 'const') hl.innerHTML = 'constant; no parameters, shared by all heads';
    else hl.innerHTML = hs.length === 0 ? 'depends on no head parameters' : hs.length === 1 ? `depends on <b class="h${hs[0]}">head ${hs[0]}</b> parameters only` : `depends on <b class="h0">head 0</b> and <b class="h1">head 1</b> parameters — mixed`;
    document.getElementById('sel-expr').textContent = c.expr;
    const cnt = {}; for (const oid of down) { const s = cells.get(oid).stage; cnt[s] = (cnt[s] || 0) + 1; }
    const total = {}; for (const o of cells.values()) total[o.stage] = (total[o.stage] || 0) + 1;
    document.getElementById('sel-counts').innerHTML = stageOrder.filter(s => cnt[s]).map(s => `<span>${s}</span><span>${cnt[s]} / ${total[s]}</span>`).join('') || '<span>none</span><span></span>';
  }
  function clear() {
    for (const list of els.values()) for (const el of list) el.classList.remove('sel', 'down', 'up', 'dim');
    document.getElementById('sel-label').textContent = '—';
    document.getElementById('sel-heads').textContent = 'Click any cell.';
    document.getElementById('sel-expr').textContent = '';
    document.getElementById('sel-counts').innerHTML = '';
  }


  // ---------- component-decomposition highlighting ----------
  // A concept detector is a group of model parameters plus the site where its
  // output first appears. Each formalism below partitions the attention block
  // into such groups; the partition stops at the first op that mixes groups,
  // which is why none of them reach attnᵀ (the output projection sums across
  // heads) and why the q/k formalisms stop before S (the score sums over dims).
  const DECOMPOSITIONS = {
    value_dim: {
      label: 'Value dim',
      title: 'value dimension d',
      note: 'W_qkv row 2C+d and its bias → v_flat[:,d] → v_h[:,d mod hd] → ctx_h[:,d mod hd] → oᵀ[d,:] → W_proj column d. One group per embedding dimension of the value stream. Stops at W_proj, the first op that mixes.',
      count: m => m.C,
      of: (P, m) => {
        const n = k => parseInt(P[k], 10);
        switch (P[0]) {
          case 'Wqkv': return Math.floor(n(1) / m.C) === 2 ? n(1) % m.C : null;
          case 'bqkv': return Math.floor(n(1) / m.C) === 2 ? n(1) % m.C : null;
          case 'qkvT': return Math.floor(n(1) / m.C) === 2 ? n(1) % m.C : null;
          case 'vflat': return n(2);
          case 'v': return n(1) * m.HD + n(3);
          case 'ctx': return n(1) * m.HD + n(3);
          case 'oT': return n(1);
          case 'Wproj': return n(2);
          default: return null;
        }
      },
    },
    qk_dim: {
      label: 'Query/key dim',
      title: 'query–key dimension d',
      note: 'W_qkv rows d (query) and C+d (key) with their biases → q_flat[:,d], k_flat[:,d] → per-head q,k column d mod hd → scaled query and transposed key. Stops before S: the score sums over all dimensions, so no single dimension owns it. Under zero-ablation, deleting dim d from q and from k removes the same product term, which is why the vanilla ViT treats them as one group.',
      count: m => m.C,
      of: (P, m) => {
        const n = k => parseInt(P[k], 10);
        const part = p => Math.floor(p / m.C);
        switch (P[0]) {
          case 'Wqkv': case 'bqkv': case 'qkvT':
            return part(n(1)) < 2 ? n(1) % m.C : null;
          case 'qflat': case 'kflat': return n(2);
          case 'q': case 'k': case 'qr': case 'kr': case 'qt':
            return n(1) * m.HD + n(3);
          case 'kT': return n(1) * m.HD + n(2);
          default: return null;
        }
      },
    },
    query_dim: {
      label: 'Query dim',
      title: 'query dimension d',
      note: 'W_qkv row d and its bias → q_flat[:,d] → q_h[:,d mod hd] → rotated and scaled query. Separate from the key dimension because RoPE is applied before the score, so deleting dim d from q and from k are no longer the same intervention.',
      count: m => m.C,
      of: (P, m) => {
        const n = k => parseInt(P[k], 10);
        switch (P[0]) {
          case 'Wqkv': case 'bqkv': case 'qkvT':
            return Math.floor(n(1) / m.C) === 0 ? n(1) % m.C : null;
          case 'qflat': return n(2);
          case 'q': case 'qr': case 'qt': return n(1) * m.HD + n(3);
          default: return null;
        }
      },
    },
    key_dim: {
      label: 'Key dim',
      title: 'key dimension d',
      note: 'W_qkv row C+d and its bias → k_flat[:,d] → k_h[:,d mod hd] → rotated key and its transpose.',
      count: m => m.C,
      of: (P, m) => {
        const n = k => parseInt(P[k], 10);
        switch (P[0]) {
          case 'Wqkv': case 'bqkv': case 'qkvT':
            return Math.floor(n(1) / m.C) === 1 ? n(1) % m.C : null;
          case 'kflat': return n(2);
          case 'k': case 'kr': return n(1) * m.HD + n(3);
          case 'kT': return n(1) * m.HD + n(2);
          default: return null;
        }
      },
    },
    head: {
      label: 'Head',
      title: 'attention head h',
      note: 'every W_qkv row belonging to head h — query, key and value alike — plus W_proj columns h·hd … h·hd+hd−1, and every per-head tensor in between: q, k, v, scores, weights, context. Stops at attnᵀ, where W_proj sums across head columns: that is the first number depending on more than one head.',
      count: m => m.H,
      of: (P, m) => {
        const n = k => parseInt(P[k], 10);
        const hOfRow = p => Math.floor((p % m.C) / m.HD);
        switch (P[0]) {
          case 'Wqkv': case 'bqkv': case 'qkvT': return hOfRow(n(1));
          case 'qflat': case 'kflat': case 'vflat': return Math.floor(n(2) / m.HD);
          case 'q': case 'k': case 'v': case 'qr': case 'kr': case 'qt':
          case 'kT': case 'S': case 'w': case 'ctx': return n(1);
          case 'oT': return Math.floor(n(1) / m.HD);
          case 'Wproj': return Math.floor(n(2) / m.HD);
          default: return null;
        }
      },
    },
    value_head: {
      label: 'Value-head',
      title: 'value rows of head h',
      note: 'the value half of a head: W_qkv value rows of head h and their biases → v_flat columns of head h → v_h → ctx_h → oᵀ rows of head h → W_proj columns of head h. The query and key rows of the same head are excluded, so the attention weights are left untouched. Zeroing this group zeroes ctx_h entirely, which is why the head-output site and the value site give bit-identical ablations.',
      count: m => m.H,
      of: (P, m) => {
        const n = k => parseInt(P[k], 10);
        switch (P[0]) {
          case 'Wqkv': case 'bqkv': case 'qkvT':
            return Math.floor(n(1) / m.C) === 2
              ? Math.floor((n(1) % m.C) / m.HD) : null;
          case 'vflat': return Math.floor(n(2) / m.HD);
          case 'v': case 'ctx': return n(1);
          case 'oT': return Math.floor(n(1) / m.HD);
          case 'Wproj': return Math.floor(n(2) / m.HD);
          default: return null;
        }
      },
    },
  };
  // formalisms offered per model: RoPE splits query and key, vanilla shares them
  const DECOMP_FOR = m => m.rope
    ? ['value_dim', 'query_dim', 'key_dim', 'head', 'value_head']
    : ['value_dim', 'qk_dim', 'head', 'value_head'];

  let activeDecomp = null;

  function clearHighlight() {
    if (!activeDecomp) return;
    activeDecomp = null;
    for (const list of els.values())
      for (const el of list) { el.classList.remove('hl'); el.style.removeProperty('--hl'); }
    document.querySelectorAll('[data-decomp]').forEach(b => b.classList.remove('on'));
    const n = document.getElementById('decomp-note');
    if (n) { n.textContent = ''; n.classList.remove('shown'); }
  }

  function applyHighlight(key) {
    if (activeDecomp === key) { clearHighlight(); return; }
    clearHighlight();
    const spec = DECOMPOSITIONS[key];
    if (!spec) return;
    clear();                   // drop the lineage colouring; show only the groups
    activeDecomp = key;
    for (const [id, list] of els) {
      const g = spec.of(id.split('_'), model);
      if (g === null || g === undefined || Number.isNaN(g)) continue;
      for (const el of list) {
        el.classList.add('hl');
        el.style.setProperty('--hl', `var(--p${g % 10})`);
      }
    }
    document.querySelectorAll('[data-decomp]').forEach(
      b => b.classList.toggle('on', b.dataset.decomp === key));
    const n = document.getElementById('decomp-note');
    if (n) {
      n.innerHTML = `<b>${spec.title}</b> — ${spec.note} `
        + `<span class="swatches">${Array.from({ length: spec.count(model) }, (_, i) =>
            `<i style="background:var(--p${i % 10})" title="component ${i}"></i>`).join('')}`
        + `</span> ${spec.count(model)} components`;
      n.classList.add('shown');
    }
  }

  function buildDecompPicker() {
    const bar = document.getElementById('decomp-picker');
    if (!bar) return;
    bar.innerHTML = '';
    for (const key of DECOMP_FOR(model)) {
      const b = document.createElement('button');
      b.type = 'button'; b.dataset.decomp = key;
      b.textContent = DECOMPOSITIONS[key].label;
      b.addEventListener('click', () => applyHighlight(key));
      bar.appendChild(b);
    }
  }

  function show(id) {
    model = window.MODELS[id] || Object.values(window.MODELS)[0];
    document.querySelectorAll('[data-model]').forEach(b => b.classList.toggle('on', b.dataset.model === model.id));
    document.getElementById('model-desc').innerHTML = model.desc;
    document.getElementById('dims').textContent = model.dimsLine;
    const G = build(model); render(model, G);
    buildDecompPicker();
    select(model.defaultCell || 'Wqkv_0_0');
    if (location.hash !== '#' + model.id) history.replaceState(null, '', '#' + model.id);
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-model]').forEach(b => b.addEventListener('click', () => show(b.dataset.model)));
    document.getElementById('clear').addEventListener('click', clear);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') clear(); });
    window.addEventListener('hashchange', () => show(location.hash.slice(1)));
    show(location.hash.slice(1));
  });
})();
