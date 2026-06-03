# AGENTS.md — zennit-crp (RationAI fork)

CRP = Concept Relevance Propagation on top of [zennit](https://github.com/chr5tphr/zennit) LRP.
This fork extends CRP from CNNs to **vision transformers** (AttnLRP). Everything is
already implemented — **find the tool below before writing new code.**

**Authoritative source = the code in `crp/` + the two upstream notebooks**
(`tutorials/attributions.ipynb`, `tutorials/feature_visualization.ipynb`). The prose
notes in `research/` (CURRENT_STATE, UNFOLDING_ATTENTION_REFACTOR, …) are historical
record and contain stale API names — do not copy APIs from them.

---

## Mental model (4 stages)

1. **Composite** (zennit) = the LRP rule set + canonizers. You build it once.
2. **Concept** = how a layer's units are grouped (a channel, a head, a token …) and
   how to *mask* / *rank* them. `mask` plugs into attribution; `attribute`/`reference_sampling` rank.
3. **CondAttribution** = run conditional LRP: backprop relevance, masked to chosen
   concepts at chosen layers → a heatmap (+ recorded activations/relevances).
4. **FeatureVisualization** = sweep a dataset once to **index** the top reference
   samples per concept, then retrieve/plot them (RelMax/ActMax, concept atlas).

---

## 1. Composites (the LRP rules) — zennit + `crp/transformer_patches.py`

- **CNN / generic:** use zennit composites directly, e.g.
  `EpsilonPlusFlat([SequentialMergeBatchNorm()])`. Canonizers go in the list.
- **ViT:** use this fork's composites (canonizers incl. attention-unfolding are
  pre-bundled — no manual setup). Three, all in `crp/transformer_patches.py`:

  | Composite | Use when |
  |---|---|
  | `AttnLRPEpsilonComposite(epsilon=1e-6, *, palrp=False, residual_lrp=None)` | baseline ε-LRP |
  | `AttnLRPGammaComposite(gamma=0.25, *, palrp=…, residual_lrp=…)` | paper γ default |
  | `AttnLRPCombinedComposite(*, alpha=.5, beta=.5, layerscale_uniform=False, linear_gamma=None, palrp=False, residual_lrp=None)` | **canonical recipe-builder** — used by the walkthrough |

  Kwargs `palrp` (pos-embed PA-LRP) and `residual_lrp ∈ {None,'symmetric','ratio'}`
  are opt-in remedies (AUC trade-offs documented in `research/CURRENT_STATE.md`). All
  rules/canonizers are scoped to `composite.context(model)` — no global mutation.

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

**Rank concepts:** `ChannelConcept().attribute(attr.relevances[lname], abs_norm=True)`
→ `(B, C)` per-channel relevance; `torch.topk`/`argsort` for the top concepts.

**Cross-layer decomposition:** `crp/graph.py` `trace_model_graph(model, sample, layer_names)`
→ `ModelGraph`; feed to `AttributionGraph(attribution, graph, layer_map)` and call with
`(sample, composite, concept_id, layer_name, target, width=[5,2])` → `(nodes, connections)`
hierarchy. Use for "which lower-layer concepts build this one".

---

## 3. Concepts — `crp/concepts.py` (CNN) · `crp/attention_concepts.py` (ViT)

- **`ChannelConcept`** (CNN/generic): one concept = one channel. `mask`, `attribute`,
  `reference_sampling`.
- **ViT** — three classes, all on the `(B, N, embed_dim)` relevance at a probe site:

  | Class | ctor | one concept = |
  |---|---|---|
  | `HeadConcept(num_heads, token_filter=slice(None))` | per attention head |
  | `EmbeddingDimConcept(num_heads, token_filter=…)` | per embedding dim (`head_of(dim)` decodes its head) |
  | `TokenConcept(token_filter=…)` | per token position |

  `token_filter` slices the token axis: `slice(0,1)` cls, register/spatial ranges, etc.
  **Probe sites** (from `crp/attention_unfolded.py` substitution canonizers, auto-installed
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

## Pointers & gotchas
- Start from `tutorials/attributions.ipynb` then `feature_visualization.ipynb` (CNN
  fundamentals); ViT entry point is `tutorials/vit_crp/walkthrough.ipynb`.
- Notebooks are edited **directly** (no `_build_*.py` generators anymore) — keep outputs.
- `experiments/` = faithfulness sweeps/audits (not needed to use the lib). `research/` =
  historical design notes (stale APIs). `tests/`: `uv run pytest tests/` (the ViT suite;
  legacy `test_attribution.py`/`test_integration.py` predate this branch).
- Env: `uv run python …` / `uv sync`. ViT extras in `pyproject.toml` `vit` group.
