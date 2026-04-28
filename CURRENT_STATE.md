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
| 5  | (this commit) | γ-LRP variant (`GTIGamma` / `AttnLRPGammaComposite`) + drop legacy classes + state docs refresh |

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

## Outstanding work

See `FUTURE_STATE.md`.
