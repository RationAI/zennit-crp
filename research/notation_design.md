# Notation design — perturbation / concept-detector experiments

Incremental redesign of the maths in `crp-paper/iclr2026/experiment-journal.tex`
(sections: perturbation notation, correlation of attribution and removal effect,
insertion-deletion benchmark for concept detectors, S3 block flipping).

Ground rules agreed so far:

- Plain letters with `\in \mathbb{R}^{shape}`. No function-space / index-set
  constructions, no `\bigsqcup`, no invented shorthands.
- Signed quantities stay signed.
- No normalisation of the readout; it already lives in [0,1].
- Readout class may differ per experiment.
- Units of one experiment must be parallel (no causal path between them).

One decision at a time. `[ ]` open, `[x]` settled, `[-]` dropped.

## A. Core objects
- [x] A1 Activation value: letter, shape declaration, input argument
- [x] A2 Indexing entries of an activation (one entry, a row/column, a subset)
- [x] A3 Naming *which* activation (block index + site kind)
- [x] A4 Model, readout value, class argument
- [x] A5 Input, label, evaluation sample

## B. Perturbation
- [x] B1 Concept detector as a set of coordinates
- [x] B2 Family of detectors and its size (follows from B1)
- [ ] B3 Where a detector's coordinates may live; stating parallelism
- [ ] B4 The removal operation: perturbed activation or perturbed model?
- [ ] B5 Fill value and its variants (zero, mean, sign flip, patch)
- [ ] B6 Removing several detectors at once
- [ ] B7 Effect of a removal, signed, relative to already-removed detectors

## C. Ranking and metric
- [ ] C1 Attribution score: symbol and arguments
- [ ] C2 Ranking, prefix, reversal, ties
- [ ] C3 Perturbation curve and its x-axis (count vs fraction)
- [ ] C4 Area and the DAPC/SRG score
- [ ] C5 Aggregation over the sample; random baseline

## D. Algorithms and other experiments
- [ ] D1 Greedy, dual greedy, pairwise greedy
- [ ] D2 Correlation of score and effect
- [ ] D3 S3 block flipping (units are whole activations, not parallel)

## Decisions

### A1 — activation value
`Z \in \mathbb{R}^{T \times D}` — capital letter, input implicit (fixed by a
sentence, not carried through the formulas). Entry `Z_{t,d}`, perturbed `\tilde{Z}`.
Rejected: `a`/`h` (lowercase reads as a scalar; `h` collides with a head index),
`a(x)` (explicit input forces `[a(x)]_{t,d}` everywhere).

### A2 — group of entries
A group of coordinates is a **set of positions**
`E \subseteq \{1,\dots,T\} \times \{1,\dots,D\}`.

    channel $d$   :  $E = \{(t,d) : t = 1,\dots,T\}$
    head $j$      :  $E = \{(t,d) : d \in S_j\}$
    token $t$     :  $E = \{(t,d) : d = 1,\dots,D\}$
    disjoint      :  $E_1 \cap E_2 = \emptyset$
    combine       :  $E_1 \cup E_2$
    removal reads :  "set $Z_{t,d}$ to zero for every $(t,d) \in E$"

Rejected: slice subscripts `Z_{:,S}` and Python brackets `Z[:,S]` (neither
expresses a patch spanning both axes); binary masks (arithmetic instead of
naming, and disjointness becomes `M_1 \odot M_2 = 0`).

### A3 — which activation
No fixed decoration. The symbol stays bare, the site and block are fixed by a
sentence, and a superscript for whatever dimension an equation needs (block,
site) is introduced in place, where that equation appears.

### A4 — model and readout
`f \in [0,1]^{C}` — the model's output for the fixed input, declared once to be
a softmax probability; `f_y` is the readout for class `y`, and
`y = \arg\max_c f_c` where an experiment uses the predicted class. Which class
`y` denotes is stated per experiment. Perturbed form provisionally `\tilde{f}_y`,
settled in B4.

The **logit** is never a curve value: every curve, area and delta uses the
probability (bench takes `logits.softmax(-1)[:, pred]`; concept flipping is
probability-only because a logit ratio breaks once the logit is negative). The
logit occurs only as the LRP backward seed (true-class logit, carrying its
value), so it is introduced by a sentence in the two sections that describe
seeding, per the A3 rule.

### A5 — sample and aggregation
Standard practice, no special decision: the sample is a series
`(x_1,y_1),\dots,(x_N,y_N)`, a subscript enumerates it, and aggregations are
ordinary sums over that subscript.

### Terminology (fixed)
**Concept detector** throughout — never "unit". Its elements are
**coordinates**. A detector's activation or significance is read from the
values of the computation graph at those coordinates.

### B1/B2 — concept detector and family
A concept detector *is* a set of coordinates,
`c \subseteq \{1,\dots,T\} \times \{1,\dots,D\}`; the family is
`\mathcal{C} = \{c_1,\dots,c_n\}`, matching the letters already in the journal.

    channel $d$   :  $c = \{(t,d) : t = 1,\dots,T\}$
    head $j$      :  $c = \{(t,d) : d \in S_j\}$
    disjoint      :  $c \cap c' = \emptyset$
    removal reads :  "zero $Z_{t,d}$ for every $(t,d) \in c$"

Rejected: letter `E`; a detector as an abstract object with an attached
coordinate map `E(c)`.

---

# Constraints the notation must satisfy

Collected from every experiment in the journal: the two insertion-deletion
benchmarks, correlation of attribution and effect, concept flipping, the
heuristic-optimal rankings, the CRP gallery spec, and S1-S4.

## Objects that need a name

**Model side**
- The network, and the ability to name *several* models under comparison (M1, M2, M7).
- Input image, and separately the normalised input (the gallery spec distinguishes
  `x_n` from `\tilde{x}_n`; heatmaps are taken w.r.t. the normalised one).
- Class, in three distinct roles: true label, predicted class, conditioning class.
  The gallery uses the true label and a query's conditioning class in the same
  formula.
- Probability readout for a class. [settled: `f`, `f_y`]
- Logit, used only as the backward seed value.
- Model parameters, grouped two ways: the weight group producing one output
  channel (defines a projection-site detector) and the *module class*
  (linear, attention op, LayerNorm, softmax, SAE decoder) that S4 audits.

**Activation side**
- Activation tensor at a probe site, shape tokens x channels.
- Probe-site identity: kind (residual, proj_drop, value, qk, query, key)
  crossed with block index; also a *set* of probe layers, as the gallery
  sweeps several.
- Backward tensor at the same site, and relevance as their product.
- Positive part of relevance, since the gallery clamps and the flipping
  experiment deliberately does not.
- Per-sample L1 normalisation of detector scores.
- Token subsets: CLS, registers, spatial, outlier tokens.
- Residual branch output, the unit of S3, which is a whole activation.
- Input patch partition, the unit of the saliency-mask benchmark.

**Concept side**
- Concept detector as a set of coordinates. [settled: `c`, family `\mathcal{C}`]
- The basis that generates a family: axis-aligned vs SAE. S2 reports per
  (layer, basis).
- A CAV: a direction in activation space, not a coordinate set, which must
  coexist with detectors without being one.
- Reference set / top-K store per (layer, detector, mode), with the four
  modes RelSum, RelMax, ActSum, ActMax.
- Attribution score per detector.
- Ranking of detectors, its reverse, its prefixes.
- Propagation-rule variant (cp_lrp, chefer, attnlrp_gamma, epsilon-variant)
  as a named object that changes what the score is.
- CRP conditional set, the gallery's theta = {class, layer: dims}.

## Statements that must be writable

1. Overwrite the values at a detector's coordinates and rerun, making clear
   that only that site changes and every bypassing path stays live.
2. Do it cumulatively for a growing set of detectors in a given order.
3. State that a family is disjoint and parallel, and why that matters.
4. Effect of a removal, signed, both marginal and relative to what is already
   removed.
5. Rank by a score, reverse a ranking, take a prefix.
6. Curve against number removed; area; area between the two curves.
7. Random-ranking baseline and its spread.
8. Greedy constructions that refer to the current removal state.
9. Aggregate over images, blocks and sites, while keeping per-image values for
   the Wilcoxon test and the bootstrap band.
10. Correlate scores against effects across detectors.
11. Detector score as a token-axis reduction (sum or max) of relevance or
    activation, in a signed or positive-clamped variant.
12. Conditional backward: seed at a class logit, mask to a detector at a layer.
13. Relevance entering and leaving a module, hence absorption between two cuts,
    per module class (S4).
14. Share of prediction carried by a residual branch.
15. Token-pooled activations, a fitted linear direction, and its cosine with a
    basis axis or an SAE decoder column.
16. Statements about which tokens a saliency mask highlights (E2).
17. Compare methods on one instance, and compare models.
18. Say that the same construction applies to input patches and to internal
    detectors (the open todo in the benchmark section).

## Collisions to avoid

Already live in the journal, or introduced by the draft rewrite:

| Symbol | Competing meanings |
|---|---|
| `M` | model (old text), quantity in {R, A} (gallery), model names M1/M2/M7 |
| `t` | token index, and `t_n` the true label (gallery) |
| `A` | activation (gallery), area under the curve (draft) |
| `g` | backward stream (gallery), "rest of the network" (draft) |
| `\rho` | token-axis reduction (gallery), perturbation curve (draft) |
| `R` | relevance (gallery), and any readout letter |
| `h` | hidden state, and attention head index |
| `D` | channel count, and the evaluation sample |
| `\tau` | token index (gallery), and the perturbation operator (old text) |

Subscript slots are scarce: sample index, class, coordinate pair, block. The
gallery spec already resolves this as superscript = layer, subscript = sample,
brackets = coordinates, e.g. `A^l_n[\tau, d]`. Any new notation should extend
that precedent rather than compete with it.
