# AGENTS.md — zennit-crp (RationAI fork)

CRP = Concept Relevance Propagation on top of [zennit](https://github.com/chr5tphr/zennit) LRP.
This fork extends CRP from CNNs to **vision transformers** (AttnLRP). Everything is
already implemented — **find the tool below before writing new code.**

**Authoritative source = the code in `crp/` + `zennit_ext/` + the two upstream notebooks**
(`tutorials/attributions.ipynb`, `tutorials/feature_visualization.ipynb`).

---

## Mental model (4 stages)

1. **Composite** (zennit) = the LRP rule set + canonizers. You build it once.
2. **Concept** = how a layer's units are grouped (a channel, a head, a token …) and
   how to *mask* / *rank* them. `mask` plugs into attribution; `attribute`/`reference_sampling` rank.
3. **CondAttribution** = run conditional LRP: backprop relevance, masked to chosen
   concepts at chosen layers → a heatmap (+ recorded activations/relevances).
4. **FeatureVisualization** = sweep a dataset once to **index** the top reference
   samples per concept, then retrieve/plot them (RelMax/ActMax, concept atlas).


**Design rules for LRP rule wiring (read before adding modules/rules):**
- **One module type, many rules — pick the rule in the `layer_map`, never via a
  module-per-rule.** If a module can take different LRP rules (e.g. a residual
  `x+branch`), define it ONCE (`ResidualAdd`) and select the rule in the composite
  `layer_map`: `(ResidualAdd, ResidualRatio | Uniform | ResidualL1)`. Do NOT create
  a separate module class per rule. 
- A canonizer installs the single module type so the
  add is hookable; it does NOT choose the rule.
- **Never predefine Composite variants unprompted.**

---

## 2. Conditional attribution — `crp/attribution.py` (`CondAttribution`)

```python
attribution = CondAttribution(model)            # no setup needed
attr = attribution(data, conditions, composite,
                   record_layer=[...], mask_map=ChannelConcept.mask)
attr.heatmap, attr.prediction, attr.activations[lname], attr.relevances[lname]
```
- **`conditions`** = list of dicts. `"y"` = output target class; other keys are
  `layer_name: [concept_ids]`. One dict → one heatmap.
  `[{"features.40": [35], "features.36": [24], "y": [46]}]`.
  Multi-layer = cascading; **list higher layers first**.
- **`mask_map`** = the concept's `mask` (default `ChannelConcept.mask`; for ViT pass
  `concept.mask`). This is what restricts relevance to the chosen concept ids.
- **`record_layer`** = layers whose `activations`/`relevances` you want back.
- **`start_layer` / `init_rel`** = begin backprop mid-network instead of at `"y"`.
- **`exclude_parallel=True`** (default) zeroes parallel branches between conditioned
  layers (isolates the path); `False` = standard full backward.
- **`attribution.generate(data, conditions, composite, batch_size=…)`** = generator
  that expands many conditions over one forward pass (≈2× faster than looping calls).
  Use it to score all channels of a layer.

---

## 3. Concepts — `crp/concepts.py` (CNN + ViT)

- **`ChannelConcept`** (CNN/generic): one concept = one channel. `mask`, `attribute`,
  `reference_sampling`.
- **ViT** — three classes, all on the `(B, N, embed_dim)` relevance at a probe site:

  | Class | ctor | one concept = |
  |---|---|---|
  | `HeadConcept(num_heads, token_filter=slice(None))` | per attention head |
  | `EmbeddingDimConcept(num_heads, token_filter=…)` | per embedding dim (`head_of(dim)` decodes its head) |
  | `TokenConcept(token_filter=…)` | per token position |

  `token_filter` slices the token axis: `slice(0,1)` cls, register/spatial ranges, etc.
  **Probe sites** (from `zennit_ext/attention_unfolded.py` substitution canonizers, auto-installed
  by the composites): `blocks.N.attn.{q_lrp_probe,k_lrp_probe,v_lrp_probe}` (Q/K/V token
  sequences) or `blocks.N.attn.proj_drop` (block output).

---

## 4. Feature visualization / reference sampling — `crp/visualization.py`

```python
fv = FeatureVisualization(attribution, dataset, layer_map,
                          preprocess_fn=normalize, path="fv_out")
fv.run(composite, 0, len(dataset))              # PRE-BUILD the index (once, slow)
ref = fv.get_max_reference(concept_ids, layer_name, "relevance", (0,8),
                           composite=composite, plot_fn=vis_img_heatmap)
plot_grid(ref)
```
- **`layer_map`** = `{layer_name: concept}` (which concept defines each layer).
- **Dataset must yield UN-normalized images**; `preprocess_fn` applies normalization
  internally (so reference crops render correctly).
- **`fv.run(composite, start, end, batch_size=32, checkpoint=500)`** sweeps the dataset
  and writes the top-40-per-concept index to disk under `path/` (`RelMax*/`, `ActMax*/`,
  `RelStats*/`, `ActStats*/` — dataset indices, relevances, receptive-field neurons).
  Run once; retrieval reads from disk.
- **`get_max_reference(ids, layer, mode, r_range, composite=None, rf=False, plot_fn=…)`**:
  `mode="relevance"` (RelMax, faithful) vs `"activation"` (ActMax). `composite` →
  compute conditional heatmaps on the crops (omit → raw images). `rf=True` → crop to the
  top neuron's receptive field. `plot_fn`: `vis_img_heatmap` (image+heatmap) or
  `vis_opaque_img` (masked crop) from `crp/image.py`; `None` → raw tensors.
- **Concept atlas / per-class:** `compute_stats(concept_id, layer, top_N=…)` → which
  target classes a concept serves; `get_stats_reference(...)` → reference crops per target.
- **Caching:** pass `cache=ImageCache(path)` (`crp/cache.py`); `precompute_ref(layer_ids,
  composite, plot_list=[vis_opaque_img], …)` bulk-fills it for an atlas.
- Backing stores: `Maximization`/`Statistics` (`crp/maximization.py`, `statistics.py`);
  readers in `crp/helper.py` (`load_maximization`, `load_statistics`, `find_files`).

---

## 5. Rendering & helpers — `crp/image.py`, `crp/helper.py`

`imgify(tensor, cmap="bwr", symmetric=True)` heatmap→PIL · `plot_grid({id:[imgs]})` concept
grid · `vis_img_heatmap` / `vis_opaque_img` the two `plot_fn`s · `get_crop_range` rf box.
`get_layer_names(model, [nn.Conv2d, nn.Linear])` lists attributable layers ·
`abs_norm`/`max_norm` relevance normalizers.

---

## Figures — output convention (paper evidence)

When you plot or show a result, treat the figure as durable evidence for the
paper, not a throwaway:

- **Always write BOTH `.png` and `.pdf`.** PNG for quick viewing / sharing; PDF
  (vector) is the paper-ready artefact that must survive even if the source data
  are deleted. Generate the PDF *while you are already plotting* — never defer it
  ("regenerate later" loses the data).
- **Store under the committed, top-level `figures/` tree — NOT under `data/`.**
  `figures/` is git-tracked (only `figures/comparison.png`, the demo throwaway, is
  ignored) so a figure is committed paper evidence. `data/` is gitignored — durable
  (it sits on the persistent storage root, survives pod bounces) but *not* in git,
  so a figure left only there is not part of the paper record. Organise by
  experiment then run: `figures/<experiment>/<config>_<concept>/<plot>_<dataset>.{png,pdf}`.
- **Overwrite stale figures; clean up when data are invalidated or superseded.**
  Generators should wipe their output subdir before re-rendering (see
  `experiments/scripts/export_flipping_figures.py`, which `shutil.rmtree`s its
  `<config>_<concept>` dir first) so removed datasets leave no orphans.
- **Make figures self-explanatory**: descriptive title (what / config / concept /
  dataset / n), axis labels with units, a labelled legend, and the metric/band
  meaning in the title or subtitle — a reader should not need a notebook caption
  to understand the figure.
- Notebooks are for interactive exploration

---

## What-to-use-when

| Goal | Tool |
|---|---|
| Heatmap for class / channel / head | `CondAttribution(...)` with `conditions` + `mask_map` |
| Rank concepts in a layer | `concept.attribute(attr.relevances[l])` + `topk` |
| Score all channels fast | `attribution.generate(...)` |
| Which lower concepts build this one | `trace_model_graph` → `AttributionGraph` |
| Top examples per concept (across a dataset) | `FeatureVisualization.run` → `get_max_reference` |
| Activation vs relevance examples | `mode="activation"` vs `"relevance"` |
| Concept↔class atlas | `compute_stats` / `get_stats_reference` |
| Zoom to the neuron's receptive field | `rf=True` |
| ViT instead of CNN | swap composite → `AttnLRP*Composite`, concept → `HeadConcept`/…, layer → a `*_probe`/`proj_drop` site, `mask_map=concept.mask` |

## Storage & persistence

Persist experiment results systematically. The pod's local disk is **ephemeral**
(wiped on bounce); durable storage is a separate configured location. Which is
which is **declared per-deployment, never detected at runtime** — two env knobs in
`.env` (see `.env.example`), read by `experiments/storage.py`:

- `ZENNIT_PERSIST_ROOT` — durable storage that survives a bounce (this deployment:
  the NFS/GPFS workspace; default `<repo>/data`).
- `ZENNIT_SCRATCH_ROOT` — fast, ephemeral scratch, wiped on bounce (this
  deployment: node-local overlay; default `~/.cache/zennit-crp`).

Rules:
- **Durable outputs** — results, checkpoints, figures, and anything a web page
  references (parquets, `figures/`, `webapp/*/figures/`, `public/`) — go
  under the persistent root / repo. Web presentations reference that copy. **Never
  leave the only copy of a result on scratch.** (The repo tree is already the
  persistent root here, so writing repo-relative is correct by default.)
- **Expensive regenerable caches** (FeatureVisualization indices, activation
  dumps) — build under `SCRATCH_ROOT` (fast, and it avoids the small-file wedge
  that thousands of tiny writes cause on network storage), then mirror to the
  persistent root with `storage.persist(...)`; refill scratch on startup with
  `storage.hydrate(...)` so a post-bounce run reuses the index instead of
  recomputing. See the FV cache wiring in `experiments/crp_gallery.py` (`CACHE_ROOT`
  = scratch, `CACHE_MIRROR` = persistent; `storage.sync` around `fv.run`).
- Do **not** add filesystem-type probing (`findmnt`, mount parsing) to choose
  locations — the roots are configuration, not something to discover.

## Environment (uv-managed)
- The venv is `.venv/` at the repo root, managed by **uv**: `UV_LINK_MODE=copy uv sync`
  creates/updates it from `pyproject.toml` + `uv.lock` (copy mode because the repo sits
  on NFS — hardlinks from the uv cache cross devices). It lives on the persistent
  workspace, so it **survives pod bounces** — no per-bounce recreation.
- ALL dependencies are tracked in `pyproject.toml` (`[project.dependencies]` for
  runtime, `[dependency-groups] dev` for tooling — uv installs dev by default).
  Never `pip install` ad hoc; add to `pyproject.toml` and `uv sync`.
- Run things with `uv run <cmd>` or `.venv/bin/python -m …` — both work.
- Notebooks: the IDE auto-detects `.venv` as the kernel — no per-project
  kernelspec registration needed (or wanted).

## Pointers & gotchas
- Start from `tutorials/attributions.ipynb` then `feature_visualization.ipynb` (CNN
  fundamentals); ViT entry point is `tutorials/vit_crp/walkthrough.ipynb`.
- Notebooks are edited **directly** (no `_build_*.py` generators anymore) — keep outputs.
- `experiments/` = faithfulness sweeps/audits (not needed to use the lib). `research/` =
  historical design notes (stale APIs). `tests/`: `uv run pytest tests/` (the ViT suite;
  legacy `test_attribution.py`/`test_integration.py` predate this branch).
