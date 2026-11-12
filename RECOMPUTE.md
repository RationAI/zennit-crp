# Recompute list — everything invalidated by the canonizer/composite fixes

Cause: (1) `LayerNormSubstitutionCanonizer` never fired on timm models (LNs got
the `Pass` fallback, not the intended σ-detach + ε rule); (2) γ on attention
projections silently no-opped on fused-MHA models. All journal results computed
through the LRP composites on timm zoo models carry the old (Pass-LN) behavior.

## Journal entries to redo (Performed part)

- [ ] **Residual relevance flow across the model set** (`exp:residual-flow`) — per-dimension skip/branch split under cp_lrp; all 4 models.
- [ ] **Register outliers I** — only the saliency/colocation half (norm-based detection is forward-only, stays valid).
- [ ] **Register outliers II** — residual mechanism + composite lever (fully LRP-based).
- [ ] **Register outliers III** — occlusion runs stay valid; redo the LRP-map comparisons.
- [ ] **Register outliers IV** — cross-method saliency comparison (LRP rows).
- [ ] **E1** (scratch-pad phenomenon, register models) — relevance-based parts; token-norm counts stay valid.
- [ ] **E2** (layer-resolved outlier→saliency, cross-method) — LRP rows.
- [ ] **Insertion-Deletion benchmark, run 1** — `lrp` and `chefer` rows on all 4 models (`rollout`/`rise`/`random` rows are composite-independent; per-model npz is method-keyed, so only the two rows need recomputation + summary/figures refresh).

## NOT invalidated

- Model records M1–M6 (training only).
- Register outliers Ib, Ic (token norms, forward-only).
- Suggested part (S1–S4) — not performed yet; will run on the fixed stack.

## Gallery (wiped, rebuild from scratch on the fixed stack)

- [ ] All CRP gallery instances: 4 model+dataset pairs × {proj_drop, residual} × both FV classes (original + condition-class-aligned); jobs.jsonl removed — recompute specs from scratch.
- [ ] FV indices (scratch + persistent mirror) — deleted; rebuilt on first gallery run.

## Non-journal artefacts computed with the old stack (redo if still wanted)

- [ ] γ/ε sweep (`gamma_sweep_M2*.npz`, `figures/benchmark/gamma_sweep_M2*`).
- [ ] Concept-flipping parquets (`data/results/concept_flipping/`) — if any figures still cited.
