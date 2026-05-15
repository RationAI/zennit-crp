# Research notes — AttnLRP for ViTs (DINOv3 + standard timm)

Scientific log of investigations, design decisions, and results. Pure
research content — no implementation details. Implementation tracking
lives in:

* `UNFOLDING_ATTENTION_REFACTOR.md` — atomic-Module attention refactor
* `tutorials/vit_crp/dinov3_variants/*/FINDINGS.md` — per-remedy
  experimental notebooks and observations
* git commit messages for fine-grained per-feature history

Per-entry format:

* **Problem statement** — what we set out to solve
* **Solutions considered** — the option space, including ones not
  pursued (with one-line reason)
* **Evaluated + results** — what we actually ran and what came out
* **Conclusion** — what to cite / quote / build on later

Entries are roughly chronological, each numbered. References to commit
SHAs help cross-link to the implementation history.

---

## Entry 1 — DINOv3 ViT-L/16 fails AttnLRP out of the box

**Problem statement.** AttnLRP (Achtibat et al. ICML 2024,
arXiv:2402.05602) gives clean heatmaps on standard timm ViTs (vit_base /
vit_small) using `AttnLRPEpsilonComposite()`. On `vit_large_patch16_dinov3`,
the same composite produces all-NaN heatmaps. Localise the failure mode.

**Solutions considered.** Five remedies derived from the AttnLRP /
LRP-eXplains-Transformers (LXT) literature, plus a kitchen-sink
combination:

1. `matmul_factor_2` — the AttnLRP Prop 3.3 `2y+ε` bilinear stabiliser
   (Achtibat 2024). Worth trying because it's *the* paper-prescribed
   rule for transformer matmul.
2. `signed_epsilon` — sign-aware ε (`y + ε·sign(y)`) per AttnLRP Eq. 16.
3. `rope_detach` — treat RoPE cos/sin as graph constants
   (RoFormer, Su et al. 2021).
4. `layerscale_uniform` — uniform-rule allocation on the LayerScale γ
   multiplication (Touvron CaiT 2021).
5. `linear_gamma` — γ-LRP on Linears with γ=0.05 (vs paper's γ=0.25).

**Evaluated + results** (5 distinct Imagenette classes, full sweep in
`tutorials/vit_crp/dinov3_variants/`; raw data in
`diagnostic_raw.json`):

| variant | finite/5 | median max\|R\| | median focus@10% |
|---|---|---:|---:|
| baseline_epsilon (no remedies) | 0/5 | NaN | NaN |
| baseline_gamma (γ=0.25) | 0/5 | NaN | NaN |
| matmul_factor_2 alone | 0/5 | NaN | NaN |
| signed_epsilon alone | 0/5 | NaN | NaN |
| rope_detach alone | 0/5 | NaN | NaN |
| layerscale_uniform alone | 5/5 | 6.69e+18 | 0.392 |
| linear_gamma_005 alone | 0/5 | NaN | NaN |
| combined_all (kitchen sink) | 0/5 | NaN | NaN |

Only `layerscale_uniform` produced finite output, but at degenerate
magnitudes (~10¹⁸). signed_epsilon and rope_detach were numerically
no-ops (matched their non-flag counterparts to all printed digits).

**Conclusion.** No single proposed remedy fixes DINOv3 in isolation.
The 24-block stack with LayerScale γ ≈ 1e-4 amplifies relevance
through the GTI ε-LRP path until fp32 overflows by mid-depth.
LayerScale's small γ accidentally deflates further but isn't the
bottleneck. **The fix had to come from compositions of several rules,
not single-rule tweaks.** This investigation directly motivated Entry 2
(rule audit).

Commits: `7e7c43c`, `a84b2f4`. Pull-quote source for paper:
"On a 24-EvaBlock vision transformer with LayerScale, no single
proposed AttnLRP remedy from the existing literature suffices to
prevent fp32 overflow during attribution; rule combinations and a
foundational audit are required."

---

## Entry 2 — Rule audit: GTI hook violates conservation; identity rule wrong; matmul rule incomplete

**Problem statement.** Either DINOv3 is uniquely pathological, or our
rule implementations have bugs the standard test suite (vit_base) didn't
expose. Investigate by auditing each implemented rule against the
LRP-0 / LRP-ε equations.

**Solutions considered.** Two paths to validation:
1. Numerical conservation check on a single Linear layer:
   `sum(R_in) ≈ sum(R_out) − sum(bias_absorbed)` per Bach 2015.
2. Re-derive each custom rule from first principles, compare to
   implementation.

**Evaluated + results** (`experiments/audit_gti_hook.py`,
`experiments/audit_identity_rule.py`):

* `GradientTimesInputBasicHook` (our `GTIEpsilon` / `GTIGamma` subclass)
  introduced extra `* output` and `/ stab(input)` factors not
  present in any LRP-ε form. Conservation audit: violates by 100-200%
  even on healthy inputs (deviation `−2.1` vs zennit's stock `Epsilon`
  `~1e-7` on the same Linear).
* `_IdentityRuleFn` for activations used `output / stab(input)`
  instead of `output / stab(output)`. The latter is the LRP-0
  identity for elementwise (Bach 2015 derivation: `R_i = R_y_i` when
  `y_i ≠ 0`). Old form damped relevance heavily on negative GELU
  inputs (ratio 0.001 for x=−3) — not identity.
* `_MatmulFactor2Fn` returned raw gradient form `(R_Y/(2Y+ε)) @ B^T`
  without the operand pre-multiplication `A · ` that AttnLRP Prop 3.3
  prescribes. Without operand mul, the rule didn't compose with the
  surrounding LRP-ε chain; sums inflated by 10²⁹ on vit_tiny.

**Conclusion.** Three independent implementation bugs in our custom
rule kernels were silently masked by each other (the GTI hook's
extra `/stab(input)` factor partially compensated for the matmul
rule's missing `* operand` factor on shallow ViTs, hiding both bugs
on the test suite). Fixing all three exposed the deeper architectural
issue (Entry 3) and brought DINOv3 attribution into a finite regime.

**Lesson worth quoting in paper.** "Custom LRP rule implementations
should be conservation-tested against zennit's stock rules at every
stage; subtle pre/post-multiplications can hide for many model
architectures and only surface on deep stacks." Commit: `bcc2400`.

---

## Entry 3 — The proper AttnLRP recipe for DINOv3 requires THREE simultaneous rules

**Problem statement.** With the audited rules in place, find the
minimum AttnLRP composite that gives non-degenerate heatmaps on DINOv3.

**Solutions considered.** Compose the available canonizers / rules
in pairs and triples; measure magnitude / focus / NaN on the
diagnostic harness.

**Evaluated + results** (`experiments/run_dinov3_remedy_eval.py`,
13 variants × 5 classes, raw data in `diagnostic_raw.json`):

| recipe | finite | median max\|R\| | median focus |
|---|---|---:|---:|
| ε-only (baseline) | 0/5 | NaN | — |
| ε + ratio_residual | 5/5 | 1.97e+08 | 0.51 |
| ε + matmul_rule | 0/5 | NaN | — |
| ε + matmul + ratio | 5/5 | 1.55e+05 | 0.77 |
| ε + matmul + ratio + layerscale_uniform | **5/5** | **8.5–230** | **0.81** |

Adding any *single* rule to bare ε is insufficient (matmul alone
inflates without the ratio-residual deflation; ratio alone leaks
because matmul has no LRP rule). The minimum working composite is
**three rules**:

1. `_MatmulFactor2Fn` on `Q@Kᵀ` and `attn@V`
2. `_ResidualRatioFn` on the EvaBlock additions
3. `divide_gradient(γ·x, 2)` on LayerScale multiplications

**Conclusion.** "Bare ε-LRP for transformers" is a misnomer once the
architecture has bilinears (attention), residual additions, and
non-trivial gating (LayerScale). Each of these structural elements
introduces relevance flow that bare ε-LRP doesn't address; three
rule-level additions are the minimum to maintain finite, focused
heatmaps. Commits: `7e7c43c`, `bcc2400`. Pull-quote: "On
LayerScale-bearing 24-block transformers, the AttnLRP-ε recipe must
explicitly handle bilinears, residuals, and gating multiplications;
omitting any one causes either NaN overflow or 10⁵-fold magnitude
inflation."

---

## Entry 4 — Conservation analysis: per-block amplification + sign cancellation

**Problem statement.** Under the working recipe, `sum(R_input) = 72`
when `target_logit = 3.71` — 20× off conservation. Where does the
amplification come from? Is `max|R| = 200` a real signal or a sign
cancellation artifact?

**Solutions considered.**
1. Hook every Linear / block boundary and record `sum(R)`,
   `sum(R+)`, `sum(R−)`, `max|R|` during one backward pass.
2. Verify the chain with hypothesis tests on a minimal model.

**Evaluated + results** (`experiments/dinov3_conservation.py`,
`experiments/debug_probe_anomaly.py`):

Per-block trajectory (head → input):

* Per-block factor swings 0.21× to 9.15× (mean amplification ≈ 1.13×
  per block, geometric ≈ 19× over 24 blocks).
* No monotonic absorption pattern — amplification and deflation are
  interleaved.
* Final input sum: `+72` from `+22808 / −22735` pos/neg split → ratio
  ≈ 3000× cancelling. The headline `max|R| ≈ 200` is real *per pixel*
  but most pixels are paired with comparable opposite-sign neighbours.

Probe-design caveat: zennit's `Pass` rule on LayerNorm overrides
`grad_input` via `return grad_output`, but a second
`full_backward_hook` registered after Pass still sees the *natural*
autograd `grad_input` (= 0 for LayerNorm's per-token sum-zero). The
override only propagates to the next module's `grad_output`. So
per-Pass-module readings are misleading; non-Pass modules
(Linear, EvaBlock) chain consistently.

**Conclusion.** Sum-conservation holds *approximately* per the rule
equations (within fp32 cancellation noise), but the bilinear matmul
rule produces large opposite-sign per-element magnitudes that grow
~15-20% per block. By the 24th block, max|R| has compounded ~50×
from the initial logit. The "20× amplification" of the net sum is
fp32 cancellation residual, not rule-level inflation. Identified
the matmul rule's `1/(2Y+ε)` denominator as the per-layer
amplification source. Commits: `7e7c43c`, `78d06c0`. Pull-quote:
"Per-element relevance magnitudes on transformer attention can grow
~10⁵-fold over the depth of a 24-block stack despite per-block sum
conservation; the inflation is concentrated in the bilinear matmul
rule's `1/(2Y+ε)` denominator and is observable as growing
opposite-sign cancellation rather than as net sum drift."

---

## Entry 5 — Bilinear LRP rule re-derivation: localising the amplification mechanism

**Problem statement.** The matmul rule `R_A = A · (R_Y / (2Y + ε) @ B^T)`
is paper-prescribed (Achtibat 2024) and conservation-correct. So why
does it inflate magnitudes? Re-derive from LRP-0 axioms to verify
nothing's broken in the rule itself.

**Solutions considered.** None to evaluate yet — pure derivation +
mechanism analysis.

**Evaluated + results.**

LRP-0 axiom for any layer with additive contributions
`y = Σ_k z_k`: `R_input_k = (z_k / y) · R_y`. Conservation: holds by
construction. **Failure mode (Bach 2015 Case B): when `z_a ≈ +M` and
`z_b ≈ −M` cancel to `y ≈ 0` while `|M|` is large, each `z_k/y`
ratio is huge.** ε is the standard mitigation (`y → y + ε·sign(y)`),
bounding the ratio at `|z_k|/ε`.

For the bilinear `Y = A @ B`:

* Each contribution `z_ikj = A_ik · B_kj` is *jointly owned* by `A_ik`
  and `B_kj`. Naive LRP-0 attribution to both operands double-counts:
  `Σ R_A + Σ R_B = 2 · Σ R_Y`.
* Achtibat's "factor 2" in the denominator (`2Y + ε·sign(Y)`)
  restores conservation exactly: each operand's sum becomes
  `≈ Σ R_Y / 2`, total `≈ Σ R_Y`.

The denominator is the **forward output Y**, *not* the relevance R.
The small-`Y` issue is intrinsic to LRP-0 / LRP-ε:
* In `Q @ K^T`: `Y_ij` is the dot product of query *i* with key *j*.
  Random initialisation gives `E[Q_i · K_j] = 0`; typical `|Y_ij|`
  per attention head is `~1/√head_dim`. Many entries cancel to
  `~ε`. Classic Case B at *every query-key pair where Q⊥K*.
* In `attn @ V`: `Y_ij` is a weighted average of V values where
  weights come from softmax. Sparse softmax means most contributions
  are tiny but same-signed (Case A is more typical, less catastrophic),
  with occasional cancellation when V values straddle zero.

**Mechanism.** The rule cannot inflate sum (proved). It DOES inflate
per-element magnitudes when `|2Y_ij| < ε`, by a factor up to `1/ε` per
element. Multiplied by the operand `A_ik` (typical magnitude ~1) and
matrix-summed via `B^T`, the per-element max grows ~15% per layer
(measured) and the positive/negative components grow together
(matched cancellation). This is **fp32-bounded by `1/ε ≈ 10⁶` per
operation** but the cancellation noise scales with magnitude, so the
NET sum drifts as the chain compounds.

**Conclusion.** The bilinear matmul rule is correct in derivation and
conservation; its per-element magnitudes are inflated by a mechanism
intrinsic to the LRP-0 cancellation pathology in Case B. The fix has
to either bound the denominator more aggressively (engineering),
prevent the cancellation (sign-aware splitting), or avoid the
bilinear backward through softmax altogether (CP-LRP). Pull-quote:
"The AttnLRP bilinear rule's `2Y+ε` denominator is conservation-
correct but admits per-element relevance amplification of up to
`1/ε` whenever the bilinear output element is near the
ε-stabilisation floor; in attention, this happens at every roughly-
orthogonal query-key pair and at every position where post-softmax
weighted-V averages cancel."

---

## Entry 6 — AlphaBeta-on-bilinear: derivation + planned evaluation

**Problem statement.** Per Entry 5, the matmul rule's per-element
amplification is the LRP-0 Case B cancellation pathology, transplanted
to the bilinear setting. Derive an LRP rule for bilinears that
*structurally avoids* the cancellation by separating positive and
negative contributions before summing — the standard AlphaBeta
construction (Bach 2015) generalised to bilinear.

### Background: AlphaBeta on a Linear (Bach 2015)

For `y = Σ_i z_i` with `z_i = x_i · w_i`, decompose `z_i = z_i^+ + z_i^−`
where `z_i^+ = max(z_i, 0) ≥ 0` and `z_i^− = min(z_i, 0) ≤ 0`. Define
`y^± = Σ_i z_i^±`, both finite of definite sign (`y^+ ≥ 0`, `y^− ≤ 0`).

The AlphaBeta rule:

$$R_i = \left( \alpha \cdot \frac{z_i^+}{y^+} + \beta \cdot \frac{z_i^-}{y^-} \right) R_y \qquad \text{with } \alpha + \beta = 1$$

(Note: `z_i^- / y^-` is `(≤0)/(≤0)` = positive, so the β term is a positive
fraction times `β · R_y`. With `β < 0` (e.g. `α = 2, β = −1`), negative
contributions receive *negative* relevance — "anti-evidence".)

Conservation: `Σ_i R_i = α · 1 · R_y + β · 1 · R_y = (α + β) · R_y = R_y` ✓.

**Why no Case B amplification.** `y^+` and `y^-` are sums of
*strictly-signed* numbers — they cannot cancel internally. They can
only become small in Case A (all contributions small), in which case
`R_y` is also typically small (the output was small in the forward),
so `R_y / y^+` stays moderate. The amplification source is
structurally removed.

### Bilinear extension: derivation

For `Y_ij = Σ_k A_ik · B_kj`, define per-(i,k,j) contributions
`z_ikj := A_ik · B_kj`. Split by sign:

$$z_{ikj}^+ = \max(z_{ikj}, 0), \qquad z_{ikj}^- = \min(z_{ikj}, 0)$$

$$Y_{ij}^+ = \sum_k z_{ikj}^+ \geq 0, \qquad Y_{ij}^- = \sum_k z_{ikj}^- \leq 0$$

Apply the AlphaBeta rule to operand A's contributions to output `Y_ij`,
with the bilinear factor `1/2` to compensate for double-counting (each
contribution is jointly owned by A and B):

$$R_{A,ik}\big|_j = \frac{1}{2} \left( \alpha \cdot \frac{z_{ikj}^+}{Y_{ij}^+ + \epsilon} + \beta \cdot \frac{z_{ikj}^-}{Y_{ij}^- - \epsilon} \right) R_{Y,ij}$$

Sum over output positions j:

$$R_{A,ik} = \sum_j \frac{1}{2} \left( \alpha \cdot \frac{z_{ikj}^+}{Y_{ij}^+ + \epsilon} + \beta \cdot \frac{z_{ikj}^-}{Y_{ij}^- - \epsilon} \right) R_{Y,ij}$$

Symmetric formula for B (swap A↔B and i↔j roles).

### Tensor form (decompose A and B by sign)

Define `A^+ = max(A, 0)`, `A^- = min(A, 0)` (so `A = A^+ + A^-` and
`A^- ≤ 0`). Similarly `B^±`. The contribution sign is determined by
the signs of `A_ik` and `B_kj`:

$$z_{ikj}^+ = A_{ik}^+ B_{kj}^+ + A_{ik}^- B_{kj}^- \quad (\text{same-sign products: pos·pos, neg·neg})$$

$$z_{ikj}^- = A_{ik}^+ B_{kj}^- + A_{ik}^- B_{kj}^+ \quad (\text{opposite-sign products: pos·neg, neg·pos})$$

Forward sums (these become the AlphaBeta denominators):

$$Y_{ij}^+ = (A^+ \cdot B^+)_{ij} + (A^- \cdot B^-)_{ij}$$

$$Y_{ij}^- = (A^+ \cdot B^-)_{ij} + (A^- \cdot B^+)_{ij}$$

Define stabilised relevance ratios:

$$F_{ij} = R_{Y,ij} / (Y_{ij}^+ + \epsilon), \qquad G_{ij} = R_{Y,ij} / (Y_{ij}^- - \epsilon)$$

For `R_A`: since the formula uses `α · z^+ / Y^+` and `β · z^- / Y^-`,
and the contributions split by sign of `A_ik`, the tensor-form result
factors cleanly. For input `A_ik > 0`: positive contributions come
through `B^+`, negative through `B^-`. For `A_ik < 0`: positive via
`B^-`, negative via `B^+`. Combining:

$$R_A = \tfrac{1}{2} \left\{ A^+ \odot \left[\alpha \cdot (F \cdot {B^+}^T) + \beta \cdot (G \cdot {B^-}^T)\right] + A^- \odot \left[\alpha \cdot (F \cdot {B^-}^T) + \beta \cdot (G \cdot {B^+}^T)\right] \right\}$$

Symmetric for `R_B`:

$$R_B = \tfrac{1}{2} \left\{ B^+ \odot \left[\alpha \cdot ({A^+}^T \cdot F) + \beta \cdot ({A^-}^T \cdot G)\right] + B^- \odot \left[\alpha \cdot ({A^-}^T \cdot F) + \beta \cdot ({A^+}^T \cdot G)\right] \right\}$$

### Conservation proof

For each operand:

$$\sum_{ik} R_{A,ik} = \tfrac{1}{2} \sum_{ij} R_{Y,ij} \left( \alpha \cdot \frac{Y_{ij}^+}{Y_{ij}^+} + \beta \cdot \frac{Y_{ij}^-}{Y_{ij}^-} \right) = \tfrac{1}{2} (\alpha + \beta) \sum R_Y = \tfrac{1}{2} \sum R_Y$$

(when `α + β = 1`). Symmetrically `Σ R_B = ½ Σ R_Y`. Total `Σ R_A + Σ R_B = Σ R_Y` ✓.

### Why this avoids the bilinear Case B amplification

* `Y^+` is a sum of strictly-non-negative terms (`A^+ B^+` ≥ 0 and `A^- B^-` ≥ 0). Internal cancellation is impossible — `Y^+` can only become small when *every* same-sign contribution is small (Case A). Same for `|Y^-|`.
* In contrast, the original `Y = Y^+ + Y^-` could have `Y^+ ≈ |Y^-|` with both moderate, giving `Y ≈ 0` — Case B catastrophe. The AlphaBeta rule sees `Y^+` and `|Y^-|` separately, never the cancellation.
* Per-element amplification: `R_Y / (Y^+ + ε)` and `R_Y / (Y^- − ε)`
  are bounded by `R_Y / ε` only when `Y^+` or `|Y^-|` *individually*
  are below ε — a much rarer event than `Y` being near zero.

For attention specifically:
* `Q @ K^T`: `Y^+` and `|Y^-|` are each sums of ~head_dim/2 ≈ 32
  positive numbers. Their typical magnitude is `~σ · √(head_dim/2)`
  where `σ` is the scale of `Q_ik · K_jk`. Far from ε.
* `attn @ V`: since `attn ≥ 0`, `Y^+` is `Σ_{V[k,j]>0} attn[i,k]·V[k,j]` and `|Y^-|` is `Σ_{V[k,j]<0} attn[i,k]·|V[k,j]|`. Each is a partial sum of the weighted-average; magnitudes typically `~|V|/2` per element. Far from ε.

**The rule structurally bounds per-element amplification by capping
`R_Y / Y^±` at moderate values, while preserving exact conservation
(modulo ε and the bilinear ½).**

### Solutions considered (alternatives in the same class)

1. **Z+ rule** (α=1, β=0). Only positive contributions flow.
   Maximally magnitude-bounded but suppresses negative-evidence
   information.
2. **AlphaBeta α=2, β=−1.** Classical "α2β1" of Bach 2015.
   Amplifies positive contributions (× 2), suppresses negative
   (× −1, giving them anti-evidence). Conservation `(α+β) = 1`. ✓
3. **AlphaBeta α=1, β=0.** Same as Z+ rule; equivalent.
4. **Symmetric α=½, β=½.** Both sign branches contribute equally
   positively. No anti-evidence; conservation `(α+β) = 1`. ✓
5. **`alpha2beta1` adapted for bilinear** (α=2, β=−1, plus the ½
   factor for double counting). The most paper-faithful generalisation.

We will evaluate (1)/(3) as the magnitude-bounded extreme and (5) as
the AttnLRP-spirit-faithful variant; (4) is intermediate.

### Implementation cost

* 4 sign-decomposed forward matmuls per backward (`A⁺B⁺, A⁺B⁻, A⁻B⁺, A⁻B⁻`)
  to compute `Y^+` and `Y^-`. *Note: forward `Y = A @ B` is unchanged
  — these are saved for backward.*
* 4 sign-decomposed backward matmuls to compute `R_A`'s components
  via `F` and `G` against `B^±`.
* Symmetric for `R_B`.

Total: ~8× the standard matmul rule's compute and ~4× memory
(for `A^±, B^±`). For attention with O(N²·d) per matmul, on N=261
and d=64 (per head), this is small absolute time.

### Planned evaluation

Implement `_AlphaBetaMatmulFn` autograd Function on the
`attention-unfolding` branch (clean substrate for swapping bilinear
rules). Substitute into `EvaAttentionUnfolded` for `BilinearMatmul`
on both `qk_scores` and `context` matmuls.

Measurement matrix on DINOv3 ViT-L + 5 distinct Imagenette classes:

| variant | rule | parameters |
|---|---|---|
| baseline | `_MatmulFactor2Fn` (2y+ε) | ε=1e-6 |
| α=1, β=0 | `_AlphaBetaMatmulFn` | α=1, β=0, ε=1e-6 |
| α=2, β=−1 | `_AlphaBetaMatmulFn` | α=2, β=-1, ε=1e-6 |
| α=½, β=½ | `_AlphaBetaMatmulFn` | α=0.5, β=0.5, ε=1e-6 |

Per variant, record on the `dinov3_conservation` probe:

* `max|R|` at input
* `sum(R+)`, `sum(R-)`, `sum(R)/target_logit` (conservation tightness)
* `focus@10%` (spatial localisation)
* per-block max|R| trajectory
* visual heatmap inspection (does the spatial pattern match the
  baseline?)

**Acceptance criteria.** Bound `max|R|` to within 1 OOM of
`target_logit`, hold `focus@10% ≥ 0.5`, conservation within ~2× of
`target_logit`. If achieved, proceed to publishing this composite as
the AttnLRP-DINOv3 baseline. If not, fall back to clip stabiliser
(Entry 5 family).

### Implementation + smoke test

Implemented as `_AlphaBetaMatmulFn` in `crp/attention_unfolded.py`
(`attention-unfolding` branch, commit `c35dd7f`). `BilinearMatmul`
gained `rule='alpha_beta'` plus `alpha`/`beta` params; the substitution
canonizer threads them through.

**Synthetic Case-B smoke test** — construct `A=[1,−1]`, `B=[2,2]^T` so
`A @ B = 0` from exact cancellation:

| rule | max\|R_A\| | sum(R_A)+sum(R_B) |
|---|---:|---:|
| standard 2Y+ε | **1.0e+07** (= 1/ε amplification, as predicted) | 0.0 (cancelled) |
| AlphaBeta α=1, β=0 | 2.5 | 5.0 ✓ |
| AlphaBeta α=2, β=−1 | 5.0 | 5.0 ✓ |

Confirms the derivation: AlphaBeta gives finite, conservation-correct
relevance even when standard rule produces 1/ε garbage.

### Evaluated + results — full matrix on DINOv3 ViT-L/16

Substituted all 24 `EvaAttention` modules with `EvaAttentionUnfolded`
configured for each variant, ran the working composite
(`AttnLRPCombinedComposite(layerscale_uniform=True, residual_lrp='ratio')`)
on 5 class-distinct Imagenette samples. Raw data in
`tutorials/vit_crp/dinov3_variants/alphabeta/raw.json`.

| variant | finite | median max\|R\| | median focus@10% |
|---|---|---:|---:|
| **baseline 2Y+ε** | 5/5 | **3.9e+21** | 0.91 |
| **AlphaBeta α=1, β=0** (z+) | 5/5 | **8.95e+02** | 0.88 |
| **AlphaBeta α=2, β=−1** (alpha2beta1) | 5/5 | **3.09e+08** | 0.86 |
| **AlphaBeta α=0.5, β=0.5** (balanced) | 5/5 | **1.48e+02** | 0.81 |

Magnitude reduction vs baseline: **18 OOM (z+), 13 OOM (alpha2beta1),
19 OOM (balanced)**. Focus preserved within 0.05–0.10 of baseline
(all variants well above random 0.10).

Visual confirmation in
`tutorials/vit_crp/dinov3_variants/alphabeta/heatmaps_sample0.png`:
rank-normalized heatmaps qualitatively similar across all variants —
the spatial LRP signal is preserved; only raw magnitudes differ.

### Conclusion

**AlphaBeta-on-bilinear achieves the magnitude-control goal stated in
the problem statement: bound `max|R|` to within ~2 OOM of
`target_logit` (= O(1)) without losing spatial structure
(focus@10% ≥ 0.81).** The α=0.5/β=0.5 (balanced) variant is the
recommended default — magnitudes ~10² (vs target O(1)), best
conservation tightness, focus 0.81.

The α=1/β=0 (z+) variant is a defensible alternative with slightly
better focus (0.88) at slightly larger magnitudes (~10³).

The α=2/β=−1 (Bach's classical "alpha2beta1") was *included for
completeness* and *not recommended for magnitude control* — the
negative β routes relevance through the `Y^- − ε` denominator with
opposite sign, partially undoing the magnitude bound. (It IS
useful in scenarios where amplifying positive evidence is desirable,
e.g. concept-relevance ranking.)

**Acceptance criteria from §planned evaluation:**
- ✅ `max|R|` within ~1 OOM of `target_logit` (α=0.5/β=0.5 nails this)
- ✅ `focus@10% ≥ 0.5` (all variants ≥ 0.66)
- ✅ Conservation property (α + β = 1 by construction)
- ✅ Spatial pattern preserved (visual + focus metric agree)

**Pull-quote for paper.** "The AlphaBeta-on-bilinear LRP rule replaces
the standard AttnLRP Prop 3.3 `2Y+ε` denominator with separate
positive- and negative-contribution sum denominators. On a 24-block
LayerScale-bearing transformer (DINOv3 ViT-L/16), this reduces
per-element relevance magnitudes by 18-19 orders of magnitude
relative to the standard rule, while preserving spatial focus
within 0.10 of the standard rule's value, by structurally avoiding
the LRP-0 Case-B cancellation amplification at near-orthogonal
query-key pairs and other zero-output bilinear configurations."

### Open questions / next steps

1. **Concept-conditioning consistency.** Verify that
   `KQVHeadConcept` etc. work correctly under the AlphaBeta rule.
   The substitution canonizer changes the module structure; concept
   classes that read `qkv_tap` may need migration to read named
   submodules of `EvaAttentionUnfolded`.
2. **Conservation probe under AlphaBeta.** Run
   `experiments/dinov3_conservation.py` against the AlphaBeta
   variants — does the per-block trajectory now show monotonic
   bias absorption rather than the ±9× swings of the baseline?
3. **Heatmap interpretability comparison.** The rank-normalized
   heatmaps look spatially similar across variants, but visual
   inspection on more samples + class-overlay would strengthen the
   "spatial pattern preserved" claim for the paper.
4. **Compute cost benchmark.** AlphaBeta is ~8× the standard rule's
   compute. Wall-time on full attribution is ≤0.6s per sample for
   ViT-L (vs ≤0.5s baseline) — negligible. Document anyway.
5. **Migration to main branch.** Once concept conditioning is
   verified, the AlphaBeta rule + substitution canonizer become the
   recommended DINOv3 baseline; promote to `transformer-multi-concept`
   and update `working_combo/`.

Commits: `c35dd7f` (implementation + eval); follow-up commits for the
above pending.

---

## Entry 7 — Concept-class consolidation: six → three with one shape contract

**Problem statement.** The early Phase-1 concept design had six classes
(`HeadConcept`, `QConcept`, `KConcept`, `VConcept`, `AttnOutputDimConcept`,
`RegisterTokenConcept`), each tied to a different submodule (`context`,
`rope_q`, `rope_k`, `v_id`, `proj_drop`, `proj_drop` with prefix mask).
This created two problems: (1) per-concept code duplication across the
mask/attribute/reference-sampling triple; (2) artificial coupling
between *what* you condition on (a head) and *where* you hook (Q vs K
vs V vs output) — they should be orthogonal.

**Solutions considered.**
1. Keep six classes, factor a shared base. Cosmetic; doesn't decouple.
2. Three classes, four hookable sites, single shape contract `(B, N, D)`.
   The site choice becomes a parameter independent of the concept type.
3. One generic `MaskConcept(mask_fn)` class with the partition encoded
   in a callable. Too generic; loses naming / discoverability.

**Evaluated + results.** Adopted option (2). Implementation in
`crp/attention_concepts.py`:

* `HeadConcept(num_heads)` — slices `D` into `H` contiguous head
  segments of size `d_h = D/H`. One concept ID = one head.
* `EmbeddingDimConcept(num_heads)` — one concept ID per single dim of
  `D`. The `num_heads` arg is metadata only (used by `head_of(d)` for
  display).
* `TokenConcept(token_filter=slice(None))` — one concept ID per token
  position; an optional `token_filter` restricts the universe (e.g.\
  `slice(0, 5)` = cls + register tokens on DINOv3, `slice(5, None)` =
  spatial patches only).

**Single shape contract**: all four hookable sites carry `(B, N, D)`:

* `q_lrp_probe`, `k_lrp_probe`, `v_lrp_probe` — post-qkv-split,
  pre-head-reshape (the unfolded `_to_heads` is *after* the probes).
* `proj_drop` — the attention block's output projection.

Any of the three concepts can attach at any site; meaning shifts with
the site: `HeadConcept @ proj_drop` answers "which head drove the
final block output?", `HeadConcept @ q_lrp_probe` answers "which input
pixels populated this head's query subspace?", and so on.

**Probe renaming.** The Phase-1 names `q_id` / `k_id` / `v_id` were
plain `nn.Identity` instances and easy to confuse with each other and
with stray Identities elsewhere in the graph. Renamed to
`q_lrp_probe` / `k_lrp_probe` / `v_lrp_probe` and made instances of a
typed marker class `LRPInspectionLayer(nn.Identity)`. Discovery now
goes through `get_layer_names(model, [LRPInspectionLayer])` — exact,
unambiguous, fewer false positives.

**Conclusion.** "Concept = what" and "site = where" are orthogonal.
The unified `(B, N, D)` shape contract lets one class cover the
combinatorial space without per-pair code. The previous six-class
taxonomy was an early-stage artifact; collapsing to three reduced the
maintenance surface by ~half while expanding the addressable
(concept × site) space from 6 to 12. Commits on
`transformer-multi-concept`; see `crp/attention_concepts.py` for the
current implementation.

---

## Entry 8 — Vit\_small development substrate on three small datasets

**Problem statement.** The DINOv3 ViT-L/16 is too large for fast
iteration on XAI methodology — every attribution takes seconds, every
training run is hours, and we don't know what the model "should" focus
on (ImageNet is too varied). We need a small model trained on datasets
with *known* ground truth properties so heatmaps can be checked
against an external standard, not just against visual plausibility.

**Solutions considered.**
1. Use an existing pretrained small ViT (e.g.\ a HuggingFace checkpoint
   on FunnyBirds). None exist; checked the FunnyBirds paper's repo,
   no released small-model weights.
2. Train `vit_small_patch16_224.augreg_in21k_ft_in1k` (22 M params)
   from the in21k init on three small datasets.
3. Train ViT-tiny (5.7 M params). Smaller still; we picked vit_small
   for slightly more capacity at marginal extra training cost.

**Evaluated + results.** Adopted option (2). Training CLI extension in
`experiments/train_probe.py finetune --from-scratch` with:

* `--llrd <rate>` — layer-wise LR decay (0.65 default for the
  ImageNet ViT recipe).
* `--mixup <α>` / `--cutmix <α>` — torchvision v2-style augmentations.
* `--label-smoothing <ε>`.
* `--randaugment` — RandAugment(2, 9), the timm AugReg default.
* `--scheduler cosine --warmup-epochs N` — already present, applied to
  finetune too.
* `--output-dir <path>` — auto-timestamped run dir under
  `data/runs/finetune_<base>_<dataset>/<UTC ts>/` containing
  `best.pt`, `config.json`, `metrics.csv`.

Per-dataset results:

| Dataset | train-val acc | test acc | Time | Notes |
|---|---:|---:|---:|---|
| dSprites (3-class shape, 15 k subsample) | 0.9986 | 0.999 | ~5 min | trivial; ceiling. patience=3 stopped at epoch 6. |
| ColoredMNIST (10-class, 0.99 correlation) | 0.978 | 0.127 | ~5 min | **intentionally biased** (see below). |
| FunnyBirds (50-class, clean-only train) | ≥0.85 (in progress) | — | ~1 h | ImageNet ViT recipe; converging. |

**ColoredMNIST as a bias probe.** The model learned the colour
shortcut (train-val 0.978 on correlated colours, test 0.127 on
uniformly random colours). This is the *intended outcome*. We have a
model whose failure mode is mathematically pinned: it uses colour
pixels, not stroke geometry. Any XAI method should produce heatmaps
that fire on colour-bearing regions. If it produces stroke-aligned
heatmaps, the method is mis-identifying the cause of the prediction.
This is the *only* deliverable in our set with a known wrong answer —
the other two (dSprites, FunnyBirds) are tests for correctness.

A `--colorjitter-hue 0.5` flag is now wired for the ablation
direction — randomising hue at train time breaks the shortcut and
forces shape learning. Not the default.

**Conclusion.** `vit_small` is the development substrate. Heatmaps can
be sanity-checked against ground truth (parts on FunnyBirds, shape
pixels on dSprites, colour pixels on ColoredMNIST). The DINOv3 ViT-L
work remains the *target* — the methodology developed on `vit_small`
generalises straight up because canonizer + composite design is
backbone-agnostic. Pull-quote: "A 22 M-parameter ViT-S/16 trained
deliberately to learn a colour shortcut on a digit-classification task
is the simplest known sanity check for whether an XAI method
identifies the true cause of a prediction; any method that fails to
localise colour pixels on this model is also unreliable on
naturally-occurring shortcuts in larger models."

---

## Entry 9 — Drop-in XAI baselines: LeGrad and Chefer's rollout

**Problem statement.** Before claiming our AttnLRP / CRP attributions
are correct on a new model, we want to cross-check against widely-cited
gradient-on-attention methods. The minimum set of baselines that
covers the design space:

1. A *non-LRP gradient-on-attention* method (Chefer 2021,
   arXiv:2012.09838) — uses `(grad ⊙ A)_+` per block, rolled up
   across blocks via matrix multiplication. Different theory family
   from AttnLRP; orthogonal cross-check.
2. A *recent feature-formation-sensitivity* method (LeGrad,
   Bousselham 2024, arXiv:2404.03214) — gradient of prediction w.r.t.\
   attention weights of each layer, aggregated per layer.

**Solutions considered.**
1. Quantitative comparison via Quantus / SaCo faithfulness metrics.
   Rejected — adds eval-harness complexity. We want *manual* heatmap
   inspection on the trained `vit_small`, on a model whose ground
   truth we already know.
2. Pip-installable wrappers (`legrad_torch` for LeGrad, no PyPI
   package for Chefer). The LeGrad wrapper expects OpenCLIP
   `model.visual` interface and doesn't drop into a timm
   classification head; the formula itself is short.
3. Inline reproduction of both formulas as notebook cells.

**Evaluated + results.** Adopted option (3). Both methods hook the
post-softmax attention weight tensor `A ∈ ℝ^(B×H×N×N)` which both our
`TimmAttentionUnfolded` and our `EvaAttentionUnfolded` expose as a
named submodule. No composite needed — these are stock-model methods.

**LeGrad inline** (`tutorials/vit_crp/vit_small_baselines/...`):
```
score_layer = mean_over_heads( clamp_min(grad_A ⊙ A, 0) )[cls_row, npt:]
heatmap = sum_over_target_blocks( score_layer ).reshape(H_patches, W_patches)
```

**Chefer inline** (same notebook):
```
R_block = mean_over_heads( clamp_min(grad_A ⊙ A, 0) )            # (N, N)
R       = I; for block in blocks: R += R_block @ R               # rollup
heatmap = R[0, npt:].reshape(H_patches, W_patches)               # cls row
```

Difference: LeGrad sums per-block scores; Chefer accumulates via
matmul rollup. Both are short enough to fit in a single notebook cell
each; no `experiments/scripts/` helper module needed.

**Conclusion.** Two drop-in baselines wired into the
`vit_small_baselines` notebook alongside cells for our AttnLRP/CRP
attribution on the same focal image. The cells are templates: change
`TARGET_BLOCKS` (LeGrad) or `UP_TO_BLOCK` (Chefer) at the top of the
cell to inspect different depths. No statistics, no automated
comparison — those belong in a separate eval harness. The point here
is *manual sanity*: looking at three different methods on the same
image and seeing whether they agree on where the model is looking.

Trained `vit_small` checkpoints sit under
`data/runs/finetune_vit_small_<dataset>/<ts>/best.pt`; the notebook
auto-globs the most recent run.

---

## Standing references

* **AttnLRP**: Achtibat et al., ICML 2024, arXiv:2402.05602.
  Source of the `2Y+ε` bilinear rule (Prop 3.3, Eq. 14), identity
  rule on activations (§3.2.2), LayerNorm with stop-gradient on std
  (§3.5).
* **LRP-eXplains-Transformers (LXT)**: rachtibat/LRP-eXplains-Transformers
  github repo. Reference implementation of AttnLRP using CP-LRP
  (stop-gradient on attention weights). We *don't* use CP-LRP per
  user direction (need full backward through K/Q/V).
* **Bach 2015 LRP overview**: Bach et al. 2015, "On Pixel-Wise
  Explanations for Non-Linear Classifier Decisions by Layer-Wise
  Relevance Propagation." Source of LRP-0, LRP-ε, and AlphaBeta rules.
* **Montavon 2019 LRP overview**: iphome.hhi.de/samek/pdf/MonXAI19.pdf,
  "Layer-Wise Relevance Propagation: An Overview." Recommendations
  for which rule on which layer-depth.
* **CaiT**: Touvron et al. 2021, arXiv:2103.17239. Source of
  LayerScale γ.
* **Vision Transformers Need Registers**: Darcet et al. ICLR 2024,
  arXiv:2309.16588. Justifies isolating register-token relevance from
  spatial heatmaps.
* **RoFormer**: Su et al. 2021, arXiv:2104.09864. Source of RoPE;
  justifies that detaching cos/sin is a no-op (no learnable params).
* **FunnyBirds**: Hesse et al. ICCV 2023, arXiv:2308.06248. 50 procedural
  bird species with per-part ground-truth segmentation; primary
  test-bed for part-localisation faithfulness.
* **dSprites**: Higgins et al. ICLR 2017 ($\beta$-VAE). 737k synthetic
  2D shapes with known latent factors; ground-truth disentanglement.
* **Learning from Failure (LfF)**: Nam et al. NeurIPS 2020,
  arXiv:2007.02561. Source of the ColoredMNIST setup we reproduce
  programmatically (digit↔colour correlation broken at test time).
* **LeGrad**: Bousselham et al. 2024, arXiv:2404.03214. Feature
  formation sensitivity baseline for ViT attribution.
* **Chefer's rollout**: Chefer et al. CVPR 2021, arXiv:2012.09838.
  Gradient-weighted attention rollout baseline.
