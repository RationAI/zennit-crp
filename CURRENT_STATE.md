# Current State — Vision-Transformer CRP

Branch: `transformer-multi-concept` (off `transformer` by Jiri Hofirek). PR #2 (https://github.com/RationAI/zennit-crp/pull/2), still in draft.

## What this fork adds

A complete, idiomatic-zennit AttnLRP implementation for vision transformers, with four concept-detector granularities, all sharing one named hook tap.

### 1. Concept classes — `crp/attention_concepts.py`

`_BaseAttentionConcept(ChannelConcept)` and four subclasses, all hooking the same named tap (`<…attn>.qkv_tap`):

| Class | Granularity | `attribute()` shape |
|---|---|---|
| `HeadConcept`     | one concept per attention head                               | `(B, num_heads)` |
| `KQVConcept`      | three concepts per block (whole Q / K / V)                   | `(B, 3)` |
| `KQVHeadConcept`  | per `(part, head)` — `3 × num_heads`                          | `(B, 3, num_heads)` |
| `HeadDimConcept`  | per `(part, head, dim)` — `3 × num_heads × head_dim`         | `(B, 3, num_heads, head_dim)` |

The base class implements `mask`, `attribute`, and `reference_sampling` (the last is required by `crp.maximization.Maximization` / `FeatureVisualization`); subclasses only define `_concept_to_slices` (mask layout) and `_per_token_relevance` (reshape + sum axes). `KQVHeadConcept` and `HeadDimConcept` accept both tuple ids and flat int ids — flat ints decode in row-major order matching the `attribute()` flatten.

### 2. AttnLRP rules — `crp/transformer_patches.py`

Idiomatic zennit Canonizer + Hook + Composite stack. Three layers:

* **Autograd Functions** (forward-graph rules):
  * `_IdentityRuleFn` — AttnLRP identity rule for activations (Eq. 9).
  * `_DivideGradientFn` — AttnLRP uniform rule for bilinears (Eq. 7).
* **Canonizers** (model graph + forward swaps, instance-level, scoped):
  * `QKVTapCanonizer(Canonizer)` — installs `qkv_tap = nn.Identity()` per Attention on register; `del module._modules["qkv_tap"]` on remove. Honors a user-pre-injected tap (idempotent).
  * `TimmViTCanonizer(CompositeCanonizer)` — bundles `QKVTapCanonizer` with four `AttributeCanonizer` instances that swap `forward` per-instance on `Attention`, `LayerNorm`, `GELU`, `Dropout`. Forward-swap reverts on `composite.context()` exit.
* **LRP Hooks** (gradient×input formulation):
  * `GradientTimesInputBasicHook(BasicHook)` — pure subclass overriding `forward` (saves output) and `backward` (multiplies grad_output by output, runs zennit's standard backward through ParamMod-modified module, divides relevance by input).
  * `GTIEpsilon` — ε-LRP in GTI form, drop-in for `zennit.rules.Epsilon`.
  * `GTIGamma` — γ-LRP in GTI form (single-branch positive-weight clamp; AttnLRP §3.2.1, default γ=0.25).
* **Composites** (`LayerMapComposite`, canonizer pre-bundled):
  * `AttnLRPEpsilonComposite` — Linear/Conv2d → `GTIEpsilon`; activations → `Pass`.
  * `AttnLRPGammaComposite` — Linear/Conv2d → `GTIGamma`; activations → `Pass`.

There is **no process-global mutation** of zennit or timm. All rules and forward swaps are scoped to `composite.context(model)` and reverted on exit.

### 3. CondAttribution context-manager order — `crp/attribution.py`

`_attribute` and `generate` enter the user composite **first** (canonizer applies, `qkv_tap` becomes a real submodule), then register recording-layer hooks and `mask_composite` **inside** that scope so name-resolution finds canonizer-created submodules:

```python
with composite.context(self.model) as modified:
    handles, layer_out = self._append_recording_layer_hooks(...)
    with mask_composite.context(self.model):
        # forward + backward
```

Without this ordering, `MaskHook` on `qkv_tap` silently no-ops (`NameMapComposite` finds no module).

### 4. Tests — `tests/`

* `tests/test_attention_concepts.py` — 38 unit tests (pure tensor, no model): mask shape and slice coverage, batch isolation, int/tuple alias, `attribute` shape and aggregation-vs-manual, abs-norm sums to 1, conservation across granularities, `reference_sampling` shape and ordering, flat-id decode.
* `tests/test_vit_integration.py` — 14 integration tests on `vit_tiny_patch16_224` (random init): canonizer register/remove cycle, idempotent pre-injected tap, forward-swap reversibility, end-to-end attribution per granularity (heatmap shape `(B, H, W)`), per-concept relevance shape, ε vs γ composite end-to-end and numerical-difference sanity.

All 52 ViT-CRP tests green.

The 6 legacy tests in `tests/test_attribution.py` and `tests/test_integration.py` predate this work and fail under the current zennit version (positional canonizer arg removed); out of scope here.

### 5. Tutorials — `tutorials/vit_crp/`

* `walkthrough.ipynb` — end-to-end notebook (Imagenette download → composite → FV index per granularity → top-concept identification → reference samples → conditional heatmap). Source kept in `_build_notebook.py` for reviewable diffs.
* `demo.py` — single-image CLI demo across the four granularities.
* `metrics.py` — deletion / insertion AUC faithfulness benchmark (Petsiuk et al.) with random-concept baseline.

### 6. Tooling — `pyproject.toml`, `uv.lock`

Dependency management is `uv add` / `uv sync`. Optional extras: `vit` (timm + transformers), `dev` (pytest, ruff), `notebook` (jupyter, ipykernel, ipywidgets, torchvision), `fast_img`. Lockfile committed for reproducibility.

## What was removed

* `crp.concepts.AttentionHeadConcept` (POC) — superseded by `HeadConcept`. The POC hooked the post-`proj` attention output, where `Linear(D, D)` mixes all heads, so a head-stripe mask there did not isolate head `h`. The four classes in `crp.attention_concepts` hook the pre-attention `qkv_tap` instead.
* `crp.attribution.AttentionAttribution` — convenience wrapper that bound `AttentionHeadConcept.mask` as default. Use `CondAttribution` with `mask_map=concept.mask` directly.
* `crp.transformer_patches.{monkey_patch, monkey_patch_zennit, prepare_timm_vit, inject_qkv_taps, _build_default_map, get_default_map, _check_already_patched, replace_module, wrap_attention_forward, cp_*}` — the entire monkey-patch infrastructure plus its LLaMA / GPT-2 / Qwen2 / Gemma3 patch maps (none of those models were exercised on this branch). Replaced by the Canonizer + Hook + Composite stack.

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
| 8  | (this commit) | Milestone D — conservation test + PA-LRP. `PALRPCanonizer` (uniform rule at `x + pos_embed`, factor 2). `TimmViTCanonizer(palrp=…)`, both composites take `palrp` kwarg. Conservation diagnostic in `tests/test_vit_integration.py` and `tutorials/vit_crp/conservation_check.py`. Multi-model sweep `run_milestone_d.py` on `vit_small/base/large` × ± PA-LRP. Findings: PA-LRP halves heatmap uniformly → AUC unchanged (Pearson=1.0, argsort identical empirically). kqv_head failure persists at every model scale (vit_large worst, vit_small mildest — opposite of saturation hypothesis). |

## Public API (post-iter-5)

```python
from crp.attention_concepts import HeadConcept, KQVConcept, KQVHeadConcept, HeadDimConcept
from crp.attribution import CondAttribution
from crp.transformer_patches import AttnLRPEpsilonComposite, AttnLRPGammaComposite
from crp.visualization import FeatureVisualization

# Choose composite. γ for ViT linears (AttnLRP §3.2.1 recommends γ ≈ 0.25):
composite = AttnLRPGammaComposite()        # canonizer pre-bundled
attribution = CondAttribution(model)       # no model-time setup needed

concept = KQVHeadConcept()
concept.register_from_model(model)

# Conditional heatmap on the top KQV-head concept:
result = attribution(
    data,
    [{"blocks.6.attn.qkv_tap": [("v", 1)], "y": [281]}],
    composite,
    mask_map=concept.mask,
)

# Index reference samples across a dataset:
fv = FeatureVisualization(attribution, dataset, {"blocks.6.attn.qkv_tap": concept},
                          preprocess_fn=preprocess_fn, path="fv_kqv_head")
fv.run(composite, 0, len(dataset))
ref_c = fv.get_max_reference([0, 1, 2], "blocks.6.attn.qkv_tap", "relevance", (0, 4),
                              composite=composite)
```

## Milestone A — faithfulness sweep finding (iter 7)

Driver: `tutorials/vit_crp/run_milestone_a.py`.
Aggregator: `tutorials/vit_crp/aggregate_milestone_a.py` →
`tutorials/vit_crp/data/milestone_a_table.md`.
Raw CSV: `tutorials/vit_crp/data/milestone_a_results.csv` (2560 rows).

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
   accounting. PA-LRP (Bakish et al., NeurIPS 2025; arXiv 2506.02138) adds a
   uniform-rule canonizer for it. Triggers FUTURE_STATE.md Milestone D.
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

## Milestone D — conservation + PA-LRP (iter 8)

### What landed

* **Conservation diagnostic** — `tests/test_vit_integration.py::TestConservation`
  (3 tests, gating off — current pipeline is far from conservative; loose
  assertions for regression detection only). Companion CLI:
  `tutorials/vit_crp/conservation_check.py`.
* **PA-LRP**: `vit_pos_embed_palrp` swap, `PALRPCanonizer` integrated
  through a `palrp: bool` kwarg on `TimmViTCanonizer`,
  `AttnLRPEpsilonComposite`, `AttnLRPGammaComposite`. Default off — PA-LRP
  is opt-in until conservation justifies turning it on.

### Conservation finding

`R_input.sum() / target_logit` ratios on a real Imagenette image (target
class 217), pretrained models:

| model | ε | ε+PA-LRP | γ=0.25 | γ=0.25+PA-LRP |
|---|---|---|---|---|
| vit_tiny | −14.6 | −7.3 | −1.7e31 | −8.8e30 |
| vit_small | 3.0e8 | 1.5e8 | NaN | NaN |
| vit_base | −223 | −112 | −1.6e35 | −7.8e34 |

PA-LRP halves the ratio **exactly** (mathematical: it halves the gradient
once at the additive `pos_embed` step). It does not approach 1.0 — the
remaining ~100× drift is dominated by the unhooked residual additions
inside each block (`x = x + attn(x)` and `x = x + mlp(x)` are plain
tensor `+`, no LRP rule applied), which add ~2× per block. γ-LRP magnitudes
are catastrophic and unfixable by PA-LRP. Documented in test docstrings.

### PA-LRP × AUC finding

Run `tutorials/vit_crp/run_milestone_d.py` (multi-model: vit_small / base /
large × ε-LRP × {palrp off, on}, same 64-image curated subset and 14-step
deletion/insertion as milestone A). 3072 rows in
`tutorials/vit_crp/data/milestone_d_results.csv`.

**PA-LRP changes nothing about AUC** — every (model, granularity) row in
the summary table reproduces bit-identically across `palrp=False` and
`palrp=True` (del_AUC, ins_AUC at 4 dp). Independently verified on a
single image: with PA-LRP the heatmap = baseline × 0.5 at every pixel
(Pearson 1.0000, `argsort` identical). PA-LRP halves a constant factor;
AUC is rank-based; rank is preserved. PA-LRP is a conservation-magnitude
fix, not a faithfulness fix — for the milestone-A AUC anomaly it is
mathematically inert.

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

* **Default kept as `AttnLRPEpsilonComposite(palrp=False)`**. PA-LRP has
  no AUC effect; turning it on by default would be an opaque ½×
  rescaling of every heatmap with no upside.
* **Milestone D is closed**: PA-LRP is implemented, opt-in, tested. The
  kqv_head failure mode it was hypothesised to fix is unrelated to
  pos_embed.
* **kqv_head and (vit_large) head_dim AUC remain open.** The probable
  cause — un-hooked residual additions accumulating ~2×/block — is a
  separate fix (residual-LRP via a `BlockResidualCanonizer` that wraps the
  `x = x + branch(x)` step in `divide_gradient(2)`). Tracked in
  FUTURE_STATE.md as the next milestone.

## Outstanding work

See `FUTURE_STATE.md`. Milestone A is **investigated, not closed**;
Milestone D is **closed** but the underlying AUC anomaly it was meant to
fix turned out to be unrelated to PA-LRP. The next high-likelihood fix is
residual-LRP (un-hooked `x = x + f(x)` in each transformer block).
