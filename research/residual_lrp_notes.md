# Residual-split LRP rules — research notes

Notes from exploring how to split relevance at a ViT residual add `y = x + branch`
(x = skip/stream, branch = sublayer output). For the experiment plumbing see
`lrp_configs/` and `zennit_ext/attnlrp_rules.py`.

## The trilemma (the core result)

For a single signed relevance value per neuron you **cannot** have all of:

1. **conservation** of the signed sum `R_x + R_branch = R_y` (LRP's "sums to the logit");
2. **boundedness** (no magnitude explosion under cancellation `x ≈ −branch`);
3. **sign-aware** split (an opposing branch gets opposite-sign relevance), proportional to magnitudes.

Conservation + sign-proportional ⟹ the signed z-rule `R_x = R_y·x/(x+branch)`, which
explodes as `y → 0` (pole). Bounding it forces either a discontinuous step or washes
out the opposition. Dropping signs gives the magnitude rule (bounded, conserving, but
sign-blind). The corners:

| rule | conserve ΣR | bounded | sign-aware |
|---|---|---|---|
| z / ε (AttnLRP default on linear) | ✅ | ❌ pole | ✅ |
| **Otsuki ratio** `|x|/(|x|+|b|)` (our production) | ✅ | ✅ | ❌ sign-blind |
| magnitude-blind symmetric ½/½ | ✅ | ✅ | ❌ |

## On cancellation (why it is not a bug)

Otsuki ratio makes both shares carry `sign(R_y)`, so the *split* creates no opposing
signs. But the block's internal linears flip signs regardless, and at the residual
**re-merge** (`R^l = R_skip + R_block`) opposite signs cancel — for *every* method,
Otsuki included. Under **signed-sum** conservation this is correct accounting, not a
loss: the signed sum still telescopes to the logit; cancellation just represents
competing evidence. It only looks like a "problem" if you demand `Σ|R|` conservation.

## The absolute-value / L1 alternatives (explored, then dropped)

Goal was sign-aware + bounded + continuous. Findings:

- **signed numerator / abs denominator** `R_x = R_y·x/(|x|+|b|)`: bounded, continuous,
  sign follows the operand, `x=0 ⇒ R_x=0`. But conserves only **L1 mass locally**
  (`|R_x|+|R_b| = |R_y|`), NOT the signed sum (`R_x+R_b = R_y·(x+b)/(|x|+|b|)`; flips
  to `−R_y` in the `--` quadrant). This is the rule **kept** as
  `lrp_configs.attnlrp_gamma_residual_l1` (`ResidualL1` / module `L1ResidualAdd`) for
  later manual inspection only — NOT production.
- **cancellation-ratio blend** `ρ·(x/y) + (1−ρ)·|x|/S`, `ρ=|y|/S`: conserves ΣR,
  bounded, but has a **discontinuous step** at `y=0` (from `sign(y)`); the smooth `ρ²`
  variant removes the step but **washes out** the opposition. Dropped.
- **renormalised α/β** (separate same-sign pools, divide by Z): conserves ΣR, bounded,
  sign-aware — but degrades to magnitude split at same-sign nodes (under-credits the
  dominant operand) and `α=β=1` reintroduces a `Z→0` blowup. Dropped.
- **global L1-contractive composite** (L1 rule everywhere): measured on dsprites — input
  retains **~0.1%** of the logit mass vs ~15% for std AttnLRP-γ (bias absorption). The
  L1 residual's *signed* shares cancel at every re-merge, so the local
  `|R_x|+|R_b|=|R_y|` identity does **not** telescope (`Σ|R|` strictly contracts at
  multi-input merges). So L1 is **not** a viable global conservation rule. Dropped
  (code removed).

## Conclusion / decisions

- **Production stays Otsuki `ratio`.** It's the conserving, bounded corner; sign-blind by
  design (the absolute value is what guarantees conservation regardless of operand sign).
- To keep `Σ|R|` AND sign you must leave the single-signed-field setting entirely:
  magnitude-only (lose sign) or **two-rail** propagation (separate non-negative `R⁺`,`R⁻`,
  never subtracted; ≈ αβ/RAP). Not pursued here.
- **Kept for manual inspection only:** `attnlrp_gamma_residual_l1` (`x/(|x|+|branch|)`).
  Note: its global mass collapses (above), but per-block ranking is scale-invariant, so
  its concept-flipping AOPC ranking may still be sensible — untested, to inspect manually.

Refs: Otsuki et al., *LRP with Conservation Property for ResNet*, ECCV 2024
(arXiv 2407.09115); Montavon et al., *LRP Overview*, Springer LNCS 11700 (2019);
Achtibat et al., *AttnLRP*, ICML 2024 (arXiv 2402.05602).

## 2026-07-22 — Empirical skip/branch split measured (diagnostic shipped)

`experiments/scripts/residual_flow_diag.py` (N=96 funny_birds test, vit_small
probe acc 0.979, `cp_lrp_baseline`, true-class conditioning). Skip derived
exactly: `R_skip = R_add − R_branch` via the `ResidualAdd` records (endpoint→add
identity verified, error 0.0). Interactive grid: `webapp/residual_flow/`
(served as `zennit-residual`); raw arrays `data/results/residual_flow/*.npz`.

Findings: network skip-dominated (branch share 0.25–0.42 at 21/24 sites);
exceptions block-0 MLP (0.57) and block-11 attn (0.63); attn branches more
skip-dominated than MLP at equal depth; U-shaped depth profile (blocks 4–8 most
skip-heavy); branch relevance 95% positive. Propagation drift between cuts
(Gamma bias absorption): median ≈2.2%, max 16%.

Paper use: quantifies the "explaining dead blocks is useless" concern per
block/dim; motivates block-output as the complete-cut site; cross-layer flipping
comparisons should note branch share.
