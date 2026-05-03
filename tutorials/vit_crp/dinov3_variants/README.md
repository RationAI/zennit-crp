# DINOv3 ViT-L/16 — AttnLRP variant matrix

Findings from the systematic diagnostic sweep over the 5 originally-proposed
"remedies" plus the baseline composites and combinations, on
`vit_large_patch16_dinov3` + linear probe trained on Imagenette.

## TL;DR — what works

```python
from crp.transformer_patches import AttnLRPCombinedComposite
composite = AttnLRPCombinedComposite(
    matmul_factor_2=True,        # AttnLRP Prop 3.3 bilinear rule
    layerscale_uniform=True,     # AttnLRP Eq. 7 uniform on LayerScale γ
    residual_lrp="ratio",        # Otsuki ratio split on residual additions
)
```

Median across 5 distinct Imagenette classes: `max|R| ≈ 1.5e+02`,
focus@10% ≈ 0.81 (vs 0.10 random), no NaN.

→ See [`working_combo/`](working_combo/FINDINGS.md) +
[`working_combo/walkthrough.ipynb`](working_combo/walkthrough.ipynb)
for the full demo.

## What changed since the first iteration of this file

A rule audit (driven by the user's correctly-flagged concern that LRP
should redistribute by *weight proportions*, not gradients) found two
real implementation bugs:

1. **`GradientTimesInputBasicHook` (`GTIEpsilon`/`GTIGamma`) was not
   LRP-ε.** It introduced extra `* output` and `/ stab(input)` factors
   that don't appear in the rule (Bach et al. 2015; Montavon et al.
   2019, iphome.hhi.de/samek/pdf/MonXAI19.pdf). Conservation audit:
   `experiments/audit_gti_hook.py` shows the broken hook violates
   `sum(R_in) ≈ sum(R_out)` by 100–200% on ordinary inputs. Fix:
   `GTIEpsilon`/`GTIGamma` are now thin aliases over zennit's stock
   `Epsilon`/`Gamma` (which we verified preserves conservation to
   ~1e-7 on the same audit).
2. **`_MatmulFactor2Fn` was missing the operand multiplication.** The
   AttnLRP Prop 3.3 rule is
   `R_A = A · (R_Y/(2Y+ε) @ B^T)`; we returned just `R_Y/(2Y+ε) @ B^T`.
   That broke the chain composition with the rest of the LRP graph.
   Fix: include the `A ·` / `B ·` factors so the rule returns
   relevance in pure R form.
3. **`_IdentityRuleFn` used `output/stab(input)` instead of
   `output/stab(output)`.** Old form over-dampened relevance based on
   input magnitude. Fixed; conservation holds across the GELU step now.

After these fixes, the originally-proposed "remedies" reorganise into:

* **Required AttnLRP rules** (not optional):
  * `matmul_factor_2` — without it bare matmul has no LRP rule.
  * `residual_lrp='ratio'` — without it residuals double R per step.
  * `layerscale_uniform` — required for any model with LayerScale γ.
* **No-ops in our config** (kept as labelled composites for paper parity):
  * `signed_epsilon` — zennit's `Stabilizer` is already sign-aware.
  * `rope_detach` — RoPE has no learnable params; detach changes nothing.
* **Alternative rule choice** (not strictly better/worse):
  * `linear_gamma_005` — γ-LRP on Linears with γ=0.05.

## Per-folder findings

| Folder | Composite | Status (after rule audit) |
|---|---|---|
| [`baseline_epsilon/`](baseline_epsilon/FINDINGS.md) | `AttnLRPEpsilonComposite()` | ❌ NaN — missing matmul + residual rules |
| [`baseline_gamma/`](baseline_gamma/FINDINGS.md) | `AttnLRPGammaComposite()` | ❌ NaN — same reason |
| [`matmul_factor_2/`](matmul_factor_2/FINDINGS.md) | `AttnLRPMatmulFactor2Composite()` | ❌ alone, ✅ with ratio + layerscale |
| [`signed_epsilon/`](signed_epsilon/FINDINGS.md) | `AttnLRPSignedEpsilonComposite()` | ❌ no-op |
| [`rope_detach/`](rope_detach/FINDINGS.md) | `AttnLRPRopeDetachComposite()` | ❌ no-op |
| [`layerscale_uniform/`](layerscale_uniform/FINDINGS.md) | `AttnLRPLayerScaleUniformComposite()` | ⚠️ alone (still ~1e+13), ✅ as part of working combo |
| [`linear_gamma_005/`](linear_gamma_005/FINDINGS.md) | `AttnLRPLinearGammaComposite(γ=0.05)` | ⚠️ finite (~1e+5), but worse than ε in working combo |
| [`combined_all/`](combined_all/FINDINGS.md) | `AttnLRPCombinedComposite(...)` (all 4 remedies + ratio) | ✅ same as working_combo (the no-op remedies are no-ops) |
| **[`working_combo/`](working_combo/FINDINGS.md)** | `AttnLRPCombinedComposite(matmul + layerscale + ratio)` | **✅ canonical recipe** |

## Conservation status

| metric | value |
|---|---|
| `sum(R_input) / target_logit` | within ~2 OOM of 1.0 on vit_base + working recipe |
| per-Linear ε-LRP conservation | ~1e-7 (zennit `Epsilon`, audited) |
| per-bilinear matmul rule conservation | exact in absence of ε |
| per-residual ratio-rule conservation | exact in absence of ε |

Full conservation (within ε) requires also installing the PA-LRP rule
on the absolute positional embedding — DINOv3 uses RoPE so this is
moot; for standard timm ViT add `palrp=True`.

## Register-token observations (Darcet et al. ICLR 2024, arXiv:2309.16588)

The notebook in `working_combo/` includes a per-token relevance dump
at one mid-stack tap. Sample-0 result: cls token |R| ≈ 16,
register tokens 4-10 each, average patch token ≈ 22. Register tokens
do receive sensible relevance under the proper recipe (consistent
with the registers paper's claim that they absorb high-norm
artifacts); the diagnostic separates them from the spatial heatmap so
the spatial pattern is not confounded by these intentionally
non-spatial tokens.

## How to regenerate

```
uv run python experiments/run_dinov3_remedy_eval.py --n-samples 5
uv run python tutorials/vit_crp/dinov3_variants/working_combo/_build_notebook.py
uv run jupyter nbconvert --to notebook --execute \
    tutorials/vit_crp/dinov3_variants/working_combo/walkthrough.ipynb \
    --output walkthrough.ipynb --ExecutePreprocessor.timeout=300
```

## Audit scripts (under `experiments/`)

* `audit_gti_hook.py` — proves the original GTI hook violated LRP-ε
  conservation by 100-200% even on healthy inputs.
* `audit_identity_rule.py` — shows the old `_IdentityRuleFn` damped
  R by `f(x)/stab(x)` instead of preserving identity behaviour.
* `dinov3_diagnose.py` — per-layer diagnostic harness used throughout.
* `run_dinov3_remedy_eval.py` — runs the full sweep that produced
  `RESULTS.md` and `diagnostic_raw.json` in this folder.
