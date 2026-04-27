# Current State — Vision Transformer CRP

Branch: `transformer-multi-concept` (off `transformer` by Jiri Hofirek). PR target: `main`/`master` (TBC).

## What exists on `transformer` branch

### POC: attention-head as concept detector
- **`crp/concepts.py:149-276` — `AttentionHeadConcept`** (subclass of `ChannelConcept`).
  - `register_num_heads(layer_name, num_heads)` (`:160`) — manual registration.
  - `_resolve_num_heads(layer_name)` (`:181`) — auto-discovery via `num_heads`/`n_heads`/`num_attention_heads` attrs.
  - `mask(batch_id, concept_ids, layer_name)` (`:201-257`) — zeroes grad outside `[..., h*d_h:(h+1)*d_h]` slice. Works on the post-concat `(B, N, D)` tensor.
  - `attribute(...)` (`:259-276`) — reshape `(B, N, D) → (B, N, H, d_h)`, sum over `dim=(1,3)` (seq + head_dim), abs-normalise.

### LXT / AttnLRP rules (`crp/transformer_patches.py`, NEW, ~452 LOC)
- Implements AttnLRP rules (Achtibat et al., ICML 2024):
  - **Identity rule** for elementwise non-linearities (GELU, ReLU) via `_IdentityRuleFn` (`:18-33`) — saves `output/(input+ε)`.
  - **Uniform rule** for bilinear ops via `_DivideGradientFn` (`:36-48`):
    - Q,K each get factor 4 in `wrap_attention_forward` (`:181-189`) — they re-enter via softmax `QK^T` so 2×2.
    - V gets factor 2 — single bilinear with attention weights.
  - **Stop-gradient** in normalisation (LN/RMSNorm) — `:119-120, 130`.
- `monkey_patch_zennit()` (`:299-320`) — overrides `BasicHook.backward` so LRP is computed as `grad·output / (input+ε)` (gradient-times-input framework).
- Default patch maps for ViT (torchvision), LLaMA, Qwen2, GPT-2, Gemma3 (`:326-409`).
- For ViT: patches `torch.nn.MultiheadAttention` with CP-LRP variant (stop-gradient on Q,K).

### Other transformer-branch additions
- `crp/attribution.py` (+93) — `AttentionAttribution` (`:681-768`) wraps `CondAttribution` and instantiates `AttentionHeadConcept` as default.
- `crp/hooks.py` (+42) — `MaskHook.backward()` (`:39-44`) applies mask functions sequentially during backward.

## Correctness assessment (vs CRP + AttnLRP + PA-LRP)

> **⚠ Semantic correctness finding (added on review).** The POC's `MaskHook` is registered on the **`Attention` / `nn.MultiheadAttention` module's output**, which is **post-proj** (i.e. after `out_proj` or `attn.proj` has mixed all heads). Masking head-stripe `[..., h*d_h:(h+1)*d_h]` of the post-proj gradient does **not** isolate head `h`'s contribution — `proj` is a full `Linear(D,D)` and mixes all `H` heads' outputs. Three semantically distinct mask points exist:
> 1. **Post-proj head-stripe** *(what the POC does)* — masks an arbitrary channel slice of the attention block output. Concept = "channel slice of post-attention block", *not* "head".
> 2. **Post-concat / pre-proj head-stripe** — directly isolates head `h`'s output. Equivalent to "head h's contribution to the residual stream before mixing".
> 3. **Pre-attention `qkv` output head-stripes (Q[h], K[h], V[h] together)** — isolates head `h`'s Q/K/V neurons. Cleanest match to the user's spec phrase *"the full triple of weight matrices considered as a single unit"*.
>
> **Plan**: keep the POC class intact for backward compatibility; introduce a new `HeadConcept` (and the K/Q/V family) that hooks on a `qkv_tap = nn.Identity()` injected at point #3. `HeadConcept` masks all three Q/K/V head-stripes simultaneously per the spec's "single unit" phrasing.

| Item | Status |
|---|---|
| Head aggregation axes | ✅ axes-correct given POC's hook point; but hook point itself is **wrong for spec** (see finding above) |
| Mask on post-concat tensor | ❌ POC actually masks post-**proj**, not post-concat. Doesn't isolate head h |
| Q,K factor 4 / V factor 2 | ✅ matches AttnLRP Eq. 15 |
| Identity rule for GELU/LN | ✅ standard AttnLRP |
| Stop-gradient on LN denom | ⚠ defensible CP-LRP variant; differs from textbook identity rule but conserves |
| ε-LRP for Linear | ✅ inherited from zennit |
| γ-LRP for ViT-Linear | ❌ not patched. AttnLRP §3.2.1 recommends γ≈0.25 for ViT to mitigate gradient shattering |
| Positional-encoding rule | ❌ not handled. PA-LRP (Bakish et al., NeurIPS 2025) Eq. 5 — additive PE is treated as constant (LXT does same; conservation violation possible). Optional add. |
| Dropout disabled in LRP | ✅ `dropout_forward` returns input (`:159-161`) |
| `concept_id` schema | ⚠ flat `List[int]` — only encodes head index. Needs extension for multi-axis concepts |

## Gaps for the two new concept definitions

1. **`mask()` is hard-coded to one head_dim slice** — needs to dispatch on a richer concept-id schema (`(part, head, dim)` triples).
2. **`attribute()` aggregation axes are hard-coded** — needs parameterised reduction.
3. **Hook point is post-concat** — for K/Q/V split we need taps on the outputs of the q/k/v projections (or slices of the fused `qkv` tensor) before they're combined.
4. **Registration carries only `num_heads`** — needs `head_dim` and per-layer concept-detector type.

## Test coverage

- `tests/test_attribution.py` — `SimpleModel` (linear). No transformer.
- `tests/test_integration.py` — `FashionModel` (Conv2D + MLP). No transformer.
- **No transformer tests at all.** `AttentionHeadConcept`, `AttentionAttribution`, `transformer_patches` are untested.

## Environment

- `setup.py` pins `torch>=1.7,<2.0`, `numpy<=1.23.5`, `zennit<=0.4.6`, `python>=3.8`.
- These pins are too tight for modern timm / HF ViT. Need to bump (or override) for the env we develop in.
- No `pyproject.toml` yet.

## Risks / open questions

- **PR target**: user said "main" but repo has `master` (default), `BP`, `transformer`. Confirm target before opening PR.
- **Setup.py bump scope**: do we modernise the pins (touches upstream), or add a separate `pyproject.toml` for dev only?
- **γ-LRP for ViT linears**: add now, or keep for follow-up? Affects all 3 concept definitions equally so adding once is cheap.
- **Patch-input attribution for visualisations**: head-level concepts produce per-token relevance; per-token-dim concepts produce per-token-per-dim relevance. Heatmaps need pixel-space relevance — that requires running the full LRP backward to the input, which the current pipeline already does (via zennit). No new code needed for that, just use the pixel-relevance maps with the conditional masks of each concept def.
