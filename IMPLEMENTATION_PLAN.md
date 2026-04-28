# Implementation Plan — Vision Transformer CRP

End-to-end plan for the `transformer-multi-concept` branch. Phases sequence; sub-tasks within a phase may parallelise.

> **Status (2026-04-27)**: Phases 0 (partial — env setup deferred to glibc host),
> 1 (refactor), 2 (KQV concept), 3 (HeadDim concept), 4 (visualisation),
> 5 (faithfulness metrics + random baseline) are **implemented**. Still TODO:
> γ-LRP composite (Phase 1 step 5), stability metric, localisation metric, README
> integration, PR open.

Conventions:
- Target ViT: `timm`'s `vit_base_patch16_224` (or HF `google/vit-base-patch16-224`). `attn` modules expose `qkv` (fused `Linear(D, 3D)`), `proj` (output `Linear(D, D)`), `num_heads`, `head_dim`.
- Hyperparameter defaults: ε=1e-6, γ=0.25 for ViT-Linear (Phase 1), uniform factors (4, 4, 2) for (Q, K, V), abs_norm=True in `Concept.attribute`.

---

## Phase 0 — Environment + smoke test

1. Add `pyproject.toml` for uv (relaxed deps: `torch>=2.0`, `zennit>=0.5`, `numpy>=1.24`, `python>=3.10`, plus dev: `timm`, `transformers`, `pytest`, `matplotlib`, `pillow`).
2. `uv venv` + `uv pip install -e .` + dev extras.
3. Run existing `tests/test_attribution.py` and `tests/test_integration.py`. Fix any breakage from pin bumps.
4. Run a transformer smoke: load `vit_base_patch16_224`, run one forward + one CRP attribution with `AttentionHeadConcept` on a single image. Capture the output shape.

**Exit criteria**: existing tests pass, head-level CRP attribution returns sensible shape on a real ViT.

---

## Phase 1 — Refactor for extensibility (low-risk, no behaviour change)

Goal: make `AttentionHeadConcept` parametric so (B) and (C) are subclasses, not rewrites.

1. Promote concept-id schema from `List[int]` → `List[Tuple]` (or named-tuple). Backwards compat: a bare `int` is interpreted as `(int,)`.
2. Extract aggregation reduction axes into a method `_aggregate_axes(relevance_shape) → tuple[int, ...]`. Default in `AttentionHeadConcept` returns the current `(seq_dim, head_dim_axis)` pair.
3. Extract slice-to-mask logic into `_concept_slice(concept_id, layer_name) → tuple[slice, ...]`. Default returns the head-stripe slice.
4. Add `head_dim` to the registry alongside `num_heads`.
5. Optional: introduce a γ-rule patch for ViT Linear layers in `transformer_patches.py` (AttnLRP §3.2.1). Pass `gamma=0.25` from a config entry.
6. **Tests**: golden test that the refactored `AttentionHeadConcept` produces bit-identical output to the pre-refactor version on the smoke ViT (regression guard).

---

## Phase 2 — Concept definition (B): K/Q/V as separate detectors

Granularity: implement two variants, controlled by an arg.
- **B1 — whole-projection**: 3 concepts per block (one for all of Q, one for all of K, one for V).
- **B2 — per-head Q/K/V**: 3·num_heads concepts per block. Most likely the useful one.

Both use the same hook taps:

1. **Hook taps**: in the ViT attention forward, expose Q/K/V outputs as separately-hookable named tensors. Implement via a small monkey-patch of `timm.models.vision_transformer.Attention.forward`:
   - After `qkv = self.qkv(x).reshape(B, N, 3, H, d_h).permute(2, 0, 3, 1, 4)`, register backward hooks on `qkv[0]`, `qkv[1]`, `qkv[2]` (which are Q, K, V respectively after the unbinding — verify exact layout).
   - Each tap is named `<block_name>.attn.q_out`, `.k_out`, `.v_out`.
2. **`KQVConcept(AttentionHeadConcept)`** in `crp/concepts.py`:
   - `concept_id = (part, head)` where `part ∈ {'q','k','v'}` and `head ∈ {0..H-1}` (variant B2). For B1, `head=None`.
   - `_concept_slice` returns the head stripe for B2, the whole tensor for B1.
   - `attribute()` aggregates over `(N, d_h)` per (part, head) → `(B, 3, H)` for B2.
3. **Forward-rule check**: Q/K/V outputs are the immediate post-`qkv` linear products, then enter the bilinear attention block. The existing uniform-rule factors (4, 4, 2) apply downstream of these taps, so masking *here* preserves AttnLRP's conservation argument for the masked-in concept (Phase 0/3 sanity test will verify).
4. **Tests**:
   - Shape: `(B, 3, H)` for B2, `(B, 3)` for B1.
   - Concept superposition: sum of per-head Q relevances should equal whole-Q relevance (variant B1=Σ B2 over heads), ε tolerance.
   - Sanity: a target class that strongly correlates with one head (per the existing POC heatmap) should produce strong `(q, h*)` and `(k, h*)` relevances for that head.

---

## Phase 3 — Concept definition (C): per-token-dim columns/rows

Concept = one column of W_Q (or W_K, W_V), per head. Row vs column resolution: **column** is correct (output-neuron axis; matches CRP's CNN-channel convention; AttnLRP Eq. 15 uses the output index as the per-neuron index).

1. **Reuse Phase 2 hook taps** — the same Q/K/V output tensors have the per-dim axis exposed as the last dim of `(B, N, H, d_h)`.
2. **`HeadDimConcept(KQVConcept)`** in `crp/concepts.py`:
   - `concept_id = (part, head, dim)` with `dim ∈ {0..d_h-1}`.
   - `_concept_slice` returns a single-element slice on the `d_h` axis.
   - `attribute()` aggregates over `(N,)` only → `(B, 3, H, d_h)`.
3. **Tests**:
   - Shape: `(B, 3, H, d_h)`.
   - Conservation: sum over `dim` axis equals corresponding (part, head) score from Phase 2.
   - Reference sampling preserves the `d_h` axis (the existing `ChannelConcept.reference_sampling` collapses spatial dims via argmax — needs override for per-dim concepts).

---

## Phase 4 — Visualisation pipeline

1. Pick 8–16 ImageNet-val images that activate diverse classes (e.g. animals + objects). Cache locally.
2. Pipeline: for each (image, target class):
   - Run CRP with each of (A) head, (B2) per-head Q/K/V, (C) per-head per-dim concepts.
   - For top-K most-relevant concepts under each definition, produce conditional heatmaps in pixel space.
3. Comparative figure: rows = images, columns = concept definitions, top-K=3 concepts per definition. One figure per attention block of interest (focus on mid + late blocks, e.g. blocks 6, 9, 11 of vit_b_16).
4. Save figures + metadata to `tutorials/vit_concept_comparison/`.

---

## Phase 5 — Quantitative metrics + baselines

Metrics:
1. **Faithfulness — deletion/insertion** (Petsiuk et al.): mask out the top-k% most-relevant patches, measure target-class probability drop. AUC over k.
2. **Localisation — pointing game** on ImageNet-S (or annotation-augmented val): does the heatmap argmax fall inside the bounding box?
3. **Stability** under input perturbation: small Gaussian noise on input, measure cosine sim of relevance maps.

Baselines:
- Gradient-only (no LRP).
- Grad-CAM at attention output.
- Occlusion (sliding window).
- **Random concept assignment** (shuffle concept ids, recompute) — this is the "is the concept structure meaningful" test, key for A/B/C comparison.

Outputs: a CSV per (model, image-set, concept-def, metric) → mean ± std. Plus a summary table in the PR description.

---

## Phase 6 — Tests + sanity checks (continuous, not a phase boundary)

Add throughout:
- Phase 1: regression test for refactored AttentionHeadConcept.
- Phase 2: shape + conservation tests for `KQVConcept`.
- Phase 3: shape + conservation tests for `HeadDimConcept`; reference-sampling shape test.
- Phase 5: reproducibility — set seeds; metric values within tolerance across runs.

End-to-end test: `pytest tests/` runs all of the above + a tiny ViT integration that exercises (A)+(B)+(C).

---

## Phase 7 — Docs + PR

1. Update `README.md` with a "Vision Transformer Concepts" section.
2. Add a tutorial notebook `tutorials/vit_crp_concepts.ipynb` demonstrating the 3 concept defs end-to-end.
3. Final pass on `CURRENT_STATE.md` to reflect what's now done.
4. Squash/clean commits, push `transformer-multi-concept`, open PR.

---

## Cross-phase risk register

- **timm Attention layout**: confirm whether `qkv[0]` is Q after `permute(2, 0, 3, 1, 4)` — there are minor layout diffs across timm versions. Phase 2 step 1 verifies.
- **Hook ordering**: backward hooks on intermediate tensors fire in reverse construction order; ensure mask hooks register *after* the LXT uniform-rule hooks.
- **γ-LRP destabilisation**: γ=0.25 is a reasonable default but may shift relevance distributions vs current POC outputs. Phase 1 step 5 is gated behind a config flag that defaults to current behaviour to minimise diff.
- **ImageNet val access**: if not pre-mirrored on the dev box, use a tiny curated subset (8–16 images) checked into `tests/data/`. Avoid re-downloading the full set.
- **PA-LRP positional rule**: deferred. Add only if Phase 5 conservation checks fail materially.
