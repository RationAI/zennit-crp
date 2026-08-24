# Current State — Vision-Transformer CRP

Branch: `transformer-multi-concept` (off `transformer` by Jiri Hofirek). PR #2 (https://github.com/RationAI/zennit-crp/pull/2), still in draft.

## What this fork adds

A complete, idiomatic-zennit AttnLRP implementation for vision transformers, with three concept-detector granularities, all reading from the unfolded-attention probe sites.

> **Note:** sections 1–5 below were written for the iter-10 four-class
> tap design and were superseded by the *unfolding-attention* refactor
> (see [`UNFOLDING_ATTENTION_REFACTOR.md`](UNFOLDING_ATTENTION_REFACTOR.md)).
> The headline descriptions here have been updated to the current
> three-class API; the dated iteration log further down is kept verbatim
> as the historical record. The authoritative reference is the
> `crp/attention_concepts.py` docstring and the walkthrough notebook.

### 1. Concept classes — `crp/attention_concepts.py`

Three classes, each subclassing `Concept` directly (no shared base, no flags). All operate on the same `(B, N, embed_dim)` relevance tensor and can be hooked at any unfolded-attention probe site (`q_lrp_probe` / `k_lrp_probe` / `v_lrp_probe` / `proj_drop`):

| Class | Selects on | Granularity | `attribute()` shape |
|---|---|---|---|
| `HeadConcept`         | `embed_dim` | per attention head      | `(B, num_heads)`   |
| `EmbeddingDimConcept` | `embed_dim` | per embedding dimension | `(B, embed_dim)`   |
| `TokenConcept`        | `N`         | per token position      | `(B, N_filtered)`  |

* `HeadConcept` slices `embed_dim` into `num_heads` contiguous head segments and sums over `head_dim` (and the filtered token axis). `EmbeddingDimConcept` keeps each `embed_dim` index as its own concept. `TokenConcept` selects token positions along `N` instead of subspaces along `embed_dim`.
* All three take a `token_filter` slice to restrict the token axis (cls / register / spatial). `HeadConcept` / `EmbeddingDimConcept` also take `num_heads`.
* Concept ids are integers (heads / dims / token positions depending on the class).

### 2. AttnLRP rules — `crp/transformer_patches.py`

Idiomatic zennit Canonizer + Hook + Composite stack. Three layers:

* **Autograd Functions** (forward-graph rules):
  * `_IdentityRuleFn` — AttnLRP identity rule for activations (Eq. 9).
  * `_DivideGradientFn` — AttnLRP uniform rule for bilinears (Eq. 7).
* **Canonizers** (model graph + forward swaps, instance-level, scoped):
  * Attention substitution — `TimmAttentionSubstitutionCanonizer` / `EvaAttentionSubstitutionCanonizer` (in `crp/attention_unfolded.py`) swap each Attention block for an unfolded equivalent exposing `q_lrp_probe` / `k_lrp_probe` / `v_lrp_probe` / `proj_drop` (`LRPInspectionLayer` submodules). (The old tap-injection `AttentionTapsCanonizer` / `QKVTapCanonizer` were removed in the unfolding refactor.)
  * `TimmViTCanonizer(CompositeCanonizer)` — bundles the per-instance `forward` swaps (`LayerNorm` / `GELU` / `Dropout` / block-residual). Forward-swap reverts on `composite.context()` exit.
* **LRP Hooks** (gradient×input formulation):
  * `GradientTimesInputBasicHook(BasicHook)` — pure subclass overriding `forward` (saves output) and `backward` (multiplies grad_output by output, runs zennit's standard backward through ParamMod-modified module, divides relevance by input).
  * `GTIEpsilon` — ε-LRP in GTI form, drop-in for `zennit.rules.Epsilon`.
  * `GTIGamma` — γ-LRP in GTI form (single-branch positive-weight clamp; AttnLRP §3.2.1, default γ=0.25).
* **Composites** (`LayerMapComposite`, canonizer pre-bundled):
  * `AttnLRPEpsilonComposite` — Linear/Conv2d → `GTIEpsilon`; activations → `Pass`.
  * `AttnLRPGammaComposite` — Linear/Conv2d → `GTIGamma`; activations → `Pass`.
  * `AttnLRPCombinedComposite` — composes the individually-validated remedies; used by the walkthrough notebook.

There is **no process-global mutation** of zennit or timm. All rules and forward swaps are scoped to `composite.context(model)` and reverted on exit.

### 3. CondAttribution context-manager order — `crp/attribution.py`

`_attribute` and `generate` enter the user composite **first** (canonizer applies, the `q_lrp_probe` / `k_lrp_probe` / `v_lrp_probe` / `proj_drop` probes become real submodules), then register recording-layer hooks and `mask_composite` **inside** that scope so name-resolution finds canonizer-created submodules:

```python
with composite.context(self.model) as modified:
    handles, layer_out = self._append_recording_layer_hooks(...)
    with mask_composite.context(self.model):
        # forward + backward
```

Without this ordering, `MaskHook` on the canonizer-installed taps silently no-ops (`NameMapComposite` finds no module).

### 4. Tests — `tests/`

* `tests/test_attention_concepts.py` — pure-tensor unit tests (no model) for `HeadConcept` / `EmbeddingDimConcept` / `TokenConcept`: mask zeroing of non-selected heads/dims/positions, `token_filter` handling (in both `mask` and `attribute`), `attribute` shape and abs-norm sums-to-1, invalid-id raises, and the `head_of` dim→head decoder.
* `tests/test_attention_unfolded.py` — tests for the unfolded-attention substitution (forward equivalence, probe-site exposure).
* `tests/test_vit_integration.py` — integration tests on a small ViT (random init): attention-substitution canonizer register/remove, forward-swap reversibility, end-to-end attribution per concept class, ε vs γ composite sanity, and a conservation diagnostic.

Run the ViT-CRP suite with `uv run pytest tests/`.

The 6 legacy tests in `tests/test_attribution.py` and `tests/test_integration.py` predate this work and fail under the current zennit version (positional canonizer arg removed); out of scope here.

### 5. Tutorials — `tutorials/vit_crp/`

User-facing capabilities only — minimal, focused, committed-as-they-are:

* `walkthrough.ipynb` — end-to-end notebook (dataset selection → composite → FV index per granularity → top-concept identification → reference samples → conditional heatmap → single-image comparative heatmaps across the four granularities, the previous `demo.py` content folded in). Tracked directly; edit in Jupyter.

### 6. Experiments — `experiments/`

Sweeps and audits that drove design decisions; **not** prerequisites for using the library. Each script reads/writes under `data/` (gitignored).

* `datasets.py` — uniform loader. ``load("imagenette", n_per_class=16, classes=[217,482,569,701])`` → 64-image dev/CI subset; ``load("imagenet_val", n_per_class=1)`` → 1000-image class-balanced benchmark sample. Imagenette auto-downloads; imagenet_val is gated and expects manual setup. Ships the canonical 1000-WordNet-ID list at `_data/imagenet_synsets.txt`.
* `metrics.py` — Petsiuk deletion/insertion AUC machinery (per-granularity top-k, random-concept baseline, ε / γ composite factory). Imported by all milestone drivers; also runnable as a single-config CLI.
* `run_milestone_a.py` — γ-LRP sweep on `vit_base_patch16_224` (Milestone A). `--dataset {imagenette|imagenet_val}` switch.
* `run_milestone_g.py` — multi-model residual-LRP (symmetric / ratio) sweep (Milestone G). Same dataset switch.
* `aggregate_milestone_a.py` — turn the milestone-A CSV into a markdown table for the PR description.
* `conservation_check.py` — diagnostic CLI that complements `tests/test_vit_integration.py::TestConservation`.

See [`experiments/README.md`](experiments/README.md) for run-time details.

### 7. Generated data — `data/` (gitignored)

Single top-level dir for everything that gets generated:

* `data/imagenette2-160/` — downloaded by the walkthrough notebook (~98 MB).
* `data/curated_milestone_a/<class_idx>/*.JPEG` — symlink farm built by `experiments/run_milestone_a.py`.
* `data/feature_visualization/<concept>/` — per-granularity FV indices.
* `data/milestone_*_results.csv`, `data/milestone_a_table.md` — sweep outputs.

Both scripts and the notebook compute paths under `<repo>/data/`; running from the repo root just works.

### 8. Tooling — `pyproject.toml`, `uv.lock`

Dependency management is `uv add` / `uv sync`. `timm` and `Pillow` are pinned in main `dependencies`. Optional extras: `vit` (HF `transformers`, reserved for the future HF-ViT canonizer), `dev` (pytest, ruff), `notebook` (jupyter, ipykernel, ipywidgets, torchvision), `fast_img`. Lockfile committed for reproducibility.

## What was removed

* `crp.concepts.AttentionHeadConcept` (POC) — superseded by `HeadConcept`. The POC hooked the post-`proj` attention output, where `Linear(D, D)` mixes all heads, so a head-stripe mask there did not isolate head `h`. The four classes in `crp.attention_concepts` hook either `attn_out_tap` (pre-`proj`, per-head stripes preserved) or `qkv_tap` (pre-attention K/Q/V).
* `crp.attribution.AttentionAttribution` — convenience wrapper that bound `AttentionHeadConcept.mask` as default. Use `CondAttribution` with `mask_map=concept.mask` directly.
* `crp.transformer_patches.{monkey_patch, monkey_patch_zennit, prepare_timm_vit, inject_qkv_taps, _build_default_map, get_default_map, _check_already_patched, replace_module, wrap_attention_forward, cp_*}` — the entire monkey-patch infrastructure plus its LLaMA / GPT-2 / Qwen2 / Gemma3 patch maps (none of those models were exercised on this branch). Replaced by the Canonizer + Hook + Composite stack.
* `KQVConcept` (per-block coarse Q/K/V concept, iter-10) — granularity below the attention head was conceptually dubious (Q-across-all-heads isn't a single concept detector). Replaced by the four-class `KQV_SPLIT × DIM_SPLIT` matrix.

## Iteration history (this branch)

| Iter | Commit | Highlights |
|---|---|---|
| 1 | `5bdeff2` | Multi-granularity attention concepts, timm forward patch, integration tests |
| 2 | `ffb04ce` | Visualisation demo + faithfulness metrics + README |
| 3a | `420d3d5` | Align `timm_attention_forward` with timm 1.0.x; correct heatmap shape |
| 3b | `841fd35` | Override `reference_sampling` on `_BaseAttentionConcept` (FV compatibility) |
| 3c | `a760d0f` | Walkthrough notebook + uv-managed deps |
| 4  | `e055e48` | Replace monkey-patching with Canonizer + Hook + Composite (idiomatic zennit) |
| 5  | `8019975` / `3c80950` | γ-LRP variant (`GTIGamma` / `AttnLRPGammaComposite`) + drop legacy classes + state docs refresh |
| 6  | `526c77a` | Generalise `FeatureVisualization._attribution_on_reference` — pull `mask_map` from `self.layer_map[layer_name]` instead of hardcoded `ChannelConcept.{mask,mask_rf}`. Restores per-reference-sample conditional heatmaps for the four attention concepts. |
| 7  | `c7cd0d7` | Milestone A faithfulness sweep on `vit_base_patch16_224` (64 imgs × 4 classes × {ε, γ ∈ 0.0/0.1/0.25/0.5} × 4 granularities × {true, random}, per-granularity top-k). Methodology fix (`resolve_top_k`), per-granularity top-k defaults, `run_milestone_a.py` driver, `aggregate_milestone_a.py` table emitter. Drop stale `IMPLEMENTATION_PLAN.md`. |
| 8  | `4835c3c` | Milestone D — conservation diagnostic. `tests/test_vit_integration.py::TestConservation` + `tutorials/vit_crp/conservation_check.py` (both since removed). Found the pipeline far from conservative; dominant leak = un-hooked residual adds (~2×/block). *(An earlier uniform-½ "PA-LRP" sketch in this same commit was NOT the paper's method — it had no positional sink, so it only rescaled heatmaps by ½ and was AUC-inert by construction. Superseded by the paper-faithful PA-LRP in `zennit_extensions/rules/palrp.py` + `VanillaViTPosEmbedCanonizer`, opt-in via layer_map.)* |
| 9  | `c38923e` | Milestone G — residual-LRP. `_ResidualRatioFn` (Otsuki ratio split, ∝ `|x|` vs `|branch|`) + `vit_block_forward_{symmetric,ratio}` swaps + `TimmViTCanonizer(residual_lrp=…)` toggle. `run_milestone_g.py` sweep. Symmetric is AUC-inert (Pearson=1.0 — uniform ½ rescale, rank-preserving). **Ratio fixes the kqv_head AUC anomaly at all three model sizes and gets vit_small to 4/4 OK** (was 2/4). Trade-off: breaks `head` on vit_base (del_gap −0.0075) and degrades vit_large further. Default kept off; opt-in via `residual_lrp='ratio'`. |
| 10 | `c608a8e` | Concept refactor per design review. **Removed `KQVConcept`** (per-block coarse Q/K/V wasn't a meaningful concept detector). **Renamed old `HeadDimConcept` → `KQVHeadDimConcept`** and introduced **new `HeadConcept` and `HeadDimConcept` reading at the per-head output tokens**: a new `attn_out_tap` (`nn.Identity` between `attn @ v` and `self.proj`) is now the default tap for output-side concepts. Single `_AttentionConcept` base class with two boolean flags `KQV_SPLIT` and `DIM_SPLIT`; the four concrete classes are flag-only. `AttentionTapsCanonizer` (rename of `QKVTapCanonizer`) installs both taps; back-compat alias kept. Concepts auto-register attention dims when constructed with the model: `HeadConcept(model)`. Tests fully rewritten; tutorials, demo CLI, milestone drivers, walkthrough notebook, README updated. |
| 11 | `66129e8` | Repo layout cleanup. **Top-level `data/`** (single `.gitignore` entry) replaces nested `tutorials/vit_crp/data/` + `tutorials/vit_crp/FeatureVisualization/`. **`experiments/`** dir holds milestone drivers + metrics + conservation_check + aggregator (moved from `tutorials/vit_crp/`); `tutorials/vit_crp/` keeps only `walkthrough.ipynb` + `demo.py`. `_build_notebook.py` deleted — the notebook is tracked directly going forward. New `experiments/README.md`; `tutorials/vit_crp/README.md` rewritten to focus on the notebook + demo. Path defaults in scripts derive `<repo>/data/` from `__file__`; the notebook walks up to `pyproject.toml` to find the repo root. |
| 12 | (this commit) | Dataset abstraction (phase 1 of full-ImageNet support). New `experiments/datasets.py` exposes `load("imagenette", ...)` (auto-downloaded) and `load("imagenet_val", ...)` (gated; manual setup expected, code-ready, **not auto-downloaded**). Both yield a `CuratedDataset` (PIL image + ImageNet-1k class idx, also a `torch.utils.data.Dataset`). Canonical 1000-WordNet-ID list shipped at `experiments/_data/imagenet_synsets.txt`. All milestone drivers gain `--dataset {imagenette\|imagenet_val}` + `--n-per-class` + `--classes` flags; the old symlink-farm `build_curated_subset` is gone. `demo.py` deleted (single-image comparison folds into the walkthrough notebook in phase 2). |

## Public API (current — unfolding refactor)

```python
from crp.attention_concepts import HeadConcept, EmbeddingDimConcept, TokenConcept
from crp.attribution import CondAttribution
from crp.transformer_patches import AttnLRPCombinedComposite
from crp.visualization import FeatureVisualization

composite = AttnLRPCombinedComposite()     # canonizers (incl. attention substitution) pre-bundled
attribution = CondAttribution(model)       # no model-time setup needed

# One concept = one attention head. Hook at any probe site:
# q_lrp_probe / k_lrp_probe / v_lrp_probe (Q/K/V sequences) or proj_drop (block output).
concept = HeadConcept(num_heads=12)
layer_name = "blocks.6.attn.proj_drop"     # or "...q_lrp_probe", etc.

# Conditional heatmap on head 3 at this site:
result = attribution(
    data,
    [{layer_name: [3], "y": [281]}],
    composite,
    mask_map=concept.mask,
)

# Index reference samples across a dataset:
fv = FeatureVisualization(attribution, dataset, {layer_name: concept},
                          preprocess_fn=preprocess_fn, path="fv_head_proj")
fv.run(composite, 0, len(dataset))
ref_c = fv.get_max_reference([0, 1, 2], layer_name, "relevance", (0, 4),
                              composite=composite)

# Read the K/Q/V subspaces instead — same concept class, different probe site:
v_layer = "blocks.6.attn.v_lrp_probe"
result = attribution(
    data,
    [{v_layer: [1], "y": [281]}],          # head 1 of the V projection
    composite,
    mask_map=concept.mask,
)
```

## Milestone A — faithfulness sweep finding (iter 7)

Driver: `tutorials/vit_crp/run_milestone_a.py`.
Aggregator: `tutorials/vit_crp/aggregate_milestone_a.py` →
`data/milestone_a_table.md`.
Raw CSV: `data/milestone_a_results.csv` (2560 rows).

Acceptance criterion (FUTURE_STATE.md A2): `del_AUC(true) < del_AUC(random)`
**and** `ins_AUC(true) > ins_AUC(random)` for **all four** granularities
under the chosen composite. Result per rule (3/4 is the best any single rule
achieves; **no rule satisfies the all-four criterion**):

| rule    | passing granularities | failure |
|---|---|---|
| ε-LRP   | head, kqv, head_dim | kqv_head: del_gap **−0.0088**, ins_gap **−0.0120** (random WINS both) |
| γ=0.0   | head, kqv, head_dim | kqv_head (identical to ε-LRP — sanity-check confirms `GammaMod(0, min=0)` ≡ `NoMod`) |
| γ=0.1   | head, kqv_head, head_dim | kqv: ins_gap **−0.0005** (within noise) |
| γ=0.25  | (none) | head, kqv: small del_gaps; kqv_head, head_dim: marginal flips |
| γ=0.5  | kqv_head | head, kqv, head_dim: most gaps collapse to ±0.000 (γ over-flattens) |

Most informative cell — **head_dim under ε-LRP**: del_gap +0.060, ins_gap
+0.025, the strongest signal in the matrix. Confirms the fine-grained
concept structure is faithful when given enough room to discriminate.

Most surprising — **kqv_head under ε-LRP fails on both axes**. The 36
(part, head) concepts at top-8 produce union heatmaps where the 8
relevance-ranked concepts cover **less** of the model's predictive evidence
than 8 randomly-selected ones. Reproducible (γ=0.0 reproduces ε exactly).
Two plausible explanations, both deferred:

1. **Positional-encoding leakage** — relevance flowing through `pos_embed`
   is treated as a constant by AttnLRP §3 and lost from the conservation
   accounting. PA-LRP (Bakish et al., arXiv:2506.02138) addresses this with
   per-layer positional sinks and paper-faithful ε/uniform rules; the
   implementation lives in `zennit_extensions/rules/palrp.py`
   (`PosEmbedSink` Eq. 5, `RotaryRopeSink` Eq. 10), opt-in via `layer_map`,
   structure exposed by `VanillaViTPosEmbedCanonizer` (input-level PE) and
   the existing `RotaryEmbedding` modules (RoPE).
2. **Union-of-top-k saturation** — at 8/36 concepts, the 8 random concepts
   already cover most of the model's spatial attention; the discriminative
   ranking signal is washed out by the union. Smaller top-k (1, 2) or a
   pixel-ranked Petsiuk variant (rank pixels of the single top-1
   conditional heatmap) would test this.

The AttnLRP §3.2.1 γ ≈ 0.25 default does **not** transfer to the four
attention-concept granularities under the union-of-top-k Petsiuk
methodology. γ ∈ {0.25, 0.5} consistently makes the gap worse for
head/kqv/head_dim while only marginally helping kqv_head; the cause is
plausibly that γ-LRP redistributes relevance toward positive-weight
contributions in a way that flattens the per-concept ranking specificity.

**Default**: `AttnLRPEpsilonComposite` (3/4 OK, only kqv_head fails). Keep
`AttnLRPGammaComposite(gamma=0.25)` available for users who want the
paper-default rule but flag the AUC behaviour. Re-evaluate after Milestone D.

## Milestone D — conservation diagnostic (iter 8)

### What landed

* **Conservation diagnostic** — `tests/test_vit_integration.py::TestConservation`
  (3 tests, gating off — current pipeline is far from conservative; loose
  assertions for regression detection only). Companion CLI:
  `tutorials/vit_crp/conservation_check.py` (both since removed).
* **Paper-faithful PA-LRP** — `zennit_extensions/rules/palrp.py`
  (`PosEmbedSink` Eq. 5, `RotaryRopeSink` Eq. 10) plus
  `VanillaViTPosEmbedCanonizer` (input-level PE) and the existing
  `RotaryEmbedding` modules (RoPE). Opt-in via `layer_map`; default recipes
  unchanged (structure installed, no rule mapped).

### Conservation finding

`R_input.sum() / target_logit` ratios on a real Imagenette image (target
class 217), pretrained models, ε-LRP:

| model | ratio |
|---|---|
| vit_tiny | −14.6 |
| vit_small | 3.0e8 |
| vit_base | −223 |

Far from 1.0 — the dominant leak is the unhooked residual additions inside
each block (`x = x + attn(x)` and `x = x + mlp(x)` are plain tensor `+`, no
LRP rule applied), which add ~2× per block. γ-LRP magnitudes are
catastrophic. Documented in test docstrings.

> **Note on an earlier, superseded sketch.** This commit originally shipped a
> uniform-½ "PA-LRP" that wrapped `x + pos_embed` in `divide_gradient(_, 2)`
> with **no positional sink**. That is *not* the paper's method (Bakish et al.,
> arXiv:2506.02138, Eq. 5 is an ε-proportional split with a per-layer sink).
> Without a sink the positional half was discarded, so the heatmap came out
> as `baseline × 0.5` at every pixel — a uniform rescale that is AUC-inert
> by construction (rank-based AUC is blind to a constant scale). The
> "PA-LRP is AUC-inert / mathematically inert" conclusion recorded in
> earlier drafts of this section was an artifact of that wrong
> implementation, **not** a property of the paper's method. The
> paper-faithful implementation now in `zennit_extensions/rules/palrp.py`
> supersedes it.

### Multi-scale finding (`kqv_head` AUC anomaly)

`del_gap` for `kqv_head` (ε-LRP, top-8, mid-block):

| model | n_concepts | top_k_coverage | del_gap | ins_gap | verdict |
|---|---|---|---|---|---|
| vit_small (12 blocks, 6 heads) | 18 | 8/18 = 44 % | **−0.0011** | +0.0005 | del_FAIL (mildest) |
| vit_base (12 blocks, 12 heads) | 36 | 8/36 = 22 % | −0.0088 | −0.0120 | del_FAIL + ins_FAIL |
| vit_large (24 blocks, 16 heads) | 48 | 8/48 = 17 % | **−0.0165** | +0.0026 | del_FAIL (worst) |

The saturation hypothesis (smaller-`n` → more overlap with random) predicted
**vit_small** would be **worst**. The data shows the opposite: vit_large is
**worst**, vit_small **mildest**. Saturation is not the cause. Scale
amplifies the failure, suggesting accumulated LRP-rule error along the
24-block backward chain rather than concept-set redundancy.

`vit_large` also flips `head_dim` from `OK` (vit_base, +0.060) to
`del_FAIL` (vit_large, −0.033). The 24-block backward chain accumulates
enough rule error that the ranking on the 3072 head_dim concepts inverts.

### Decision

* **kqv_head and (vit_large) head_dim AUC remain open.** The probable
  cause — un-hooked residual additions accumulating ~2×/block — is a
  separate fix (residual-LRP via the `ResidualAdd` module + a residual
  rule in the `layer_map`, now shipped as `ResidualRatio` / `EpsilonAdd` /
  `CheferAdd`). Tracked in FUTURE_STATE.md as the next milestone.

  **Update (2026-08-24):** The Chefer rules (`zennit_extensions/rules/chefer2021.py`)
  and composite (`zennit_extensions/lrp_composites/chefer2021.py`) were rewritten to
  be **code-exact** (released implementation, not the paper): matmul z-rule + ÷2
  (not Eq. 9 normalization), Add global-sum renorm, Linear = ZPlus(zero_params=['bias']).
  Validated against the reference clone's own tensors via
  `tutorials/vit_crp/chefer_reference.ipynb` — all 6 `example.ipynb` heatmaps
  reconstructed to pearson r ≥ 0.999998.

## Milestone G — residual-LRP (iter 9)

Adapted from a ResNet residual-LRP scheme used in our adjacent project:
two rules, `'symmetric'` (uniform halving — equivalent to
`divide_gradient(_, 2)` per residual) and `'ratio'` (Otsuki proportional
split, ``R_x ∝ |x|`` and ``R_branch ∝ |branch|``). Both implemented as
autograd Functions in the forward pass, swapped in via an
`AttributeCanonizer` on timm `Block.forward`. Composite kwarg
`residual_lrp ∈ {None, 'symmetric', 'ratio'}` (None default).

### `'symmetric'` is AUC-inert

`divide_gradient(_, 2)` at every residual is a uniform multiplicative
factor on the entire backward chain. Pearson 1.0000 / Spearman 1.0000 vs
the baseline heatmap on a single image — same identical-AUC pathology as
PA-LRP. **Implemented but not run through the multi-model sweep**;
a no-op for ranking.

### `'ratio'` AUC findings (multi-model sweep, 64 imgs, ε-LRP)

Per-(model, granularity) verdict, comparing `residual_lrp=None` to
`residual_lrp='ratio'`. ✅ = OK, ❌ = FAIL.

| model | granularity (concepts/k) | baseline | ratio |
|---|---|---|---|
| vit_small (12L, 6H) | head (6/4) | ❌ ins_FAIL | ✅ OK |
| | kqv (3/1) | ✅ OK | ✅ OK |
| | kqv_head (18/8) | ❌ del_FAIL | ✅ OK (del_gap +0.028) |
| | head_dim (1152/8) | ✅ OK | ✅ OK |
| | **summary** | **2/4** | **4/4** |
| vit_base (12L, 12H) | head (12/4) | ✅ OK | ❌ del_FAIL (−0.008) |
| | kqv (3/1) | ✅ OK | ✅ OK |
| | kqv_head (36/8) | ❌ del_FAIL (−0.009) | ✅ OK (del_gap +0.005) |
| | head_dim (2304/8) | ✅ OK | ✅ OK |
| | **summary** | **3/4** | **3/4** (different cell) |
| vit_large (24L, 16H) | head (16/4) | ✅ OK | ❌ del_FAIL |
| | kqv (3/1) | ✅ OK | ❌ del_FAIL |
| | kqv_head (48/8) | ❌ del_FAIL | ❌ ins_FAIL |
| | head_dim (3072/8) | ❌ del_FAIL | ❌ ins_FAIL |
| | **summary** | **2/4** | **0/4** |

Raw CSV: `data/milestone_g_results.csv` (3072 rows).

### Reading

* **vit_small fully fixed.** The Otsuki ratio split is a strict
  improvement (2/4 → 4/4) at this scale. The original Milestone A `head`
  ins_FAIL and the deeper kqv_head del_FAIL both close.
* **vit_base trade-off.** Ratio splits the kqv_head failure (closes it
  cleanly, +0.013 swing on del_gap). Cost: `head` granularity
  marginally fails (del_gap −0.008, within noise of the +0.005 ratio
  result on kqv_head). Net 3/4 ↔ 3/4 — different cell fails.
* **vit_large breaks more cells.** With 24 blocks, the ratio rule
  attenuates relevance more aggressively; small absolute discrimination
  signal at this depth flips on multiple granularities. The
  cumulative-attenuation hypothesis (deeper backward chain → more
  relevance compressed near zero by the proportional split) fits.

### Decision

`residual_lrp='ratio'` is shipped as **opt-in** (`residual_lrp=None`
default on both composites). It is a real fix at the smaller scales but a
regression at vit_large; I won't override the user's heatmap by default.

For vit_small users facing the kqv_head/head failures, the recommendation
is `AttnLRPEpsilonComposite(residual_lrp='ratio')` — strict improvement.

### Open questions raised by the result

* Why does vit_large degrade? Likely cumulative attenuation: with 24
  block-pair residuals (48 ratio splits) and `|x| ≫ |branch|` after
  LayerNorm-ed inputs, the branch-relevance gets repeatedly down-weighted.
  Worth a per-layer relevance dump to confirm.
* Could a **scale-aware** ratio (clip the `|x|`/`|branch|` ratio to a
  bounded range) keep the small-scale gains without the large-scale
  collapse? Worth a single image probe before another full sweep.
* The `head`-on-vit_base regression sits at the boundary of statistical
  noise (sample n=64, gap −0.008). One more sweep with a different
  random seed and 128 imgs would settle whether it's a real flip.

## Outstanding work

Roadmap moved to YouTrack **XAI-21** (paper plan; see `scout_novelty_crp_vit.md` for the novelty verdict). `FUTURE_STATE.md` is retired. Milestone A is **investigated, not closed**;
Milestone D's conservation diagnostic landed; its earlier uniform-½ "PA-LRP"
sketch was superseded by the paper-faithful implementation in
`zennit_extensions/rules/palrp.py` (opt-in). Milestone G is
**closed** (ratio rule shipped opt-in; partial AUC fix; open questions
above). Next: methodology check (Milestone H — pixel-rank Petsiuk and
signed-vs-abs ranking) and Milestone B (richer baselines).
