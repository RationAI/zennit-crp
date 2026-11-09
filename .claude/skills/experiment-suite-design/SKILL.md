---
name: experiment-suite-design
description: Design the SMALLEST experiment that decides a research goal — one research question, one hypothesis, one experiment by default, with controls, confound handling, sample-size reasoning and a stated decision rule. Use when planning a study/benchmark/ablation, when the user says "design the experiments", "plan the suite", "what experiments do we need", or before any experimental effort.
---

## The rule that governs every other rule

**One research question. One hypothesis. One experiment.** That is the target,
not a starting point to grow from. A design that needs more has to justify the
extra, and the justification is never "it would also be interesting".

Occam's razor applies to the design, not only to the explanation:

- Between two hypotheses that predict the same observation, take the one with
  fewer moving parts and fewer free parameters, and test it first. The
  elaborate one earns its turn only after the simple one fails.
- Between two measurements that decide the same question, take the cheaper and
  more direct one. A derived quantity with three preprocessing steps is a
  worse instrument than a number read straight off the run, even when it is
  more elegant.
- Between two experiments that would both settle the question, run the one
  whose negative result is more informative.
- An additional condition, site, model or granularity is only in scope if the
  question cannot be decided without it. "While we are at it" is how suites
  become untestable.

**Scope cuts before experiments multiply.** When one experiment cannot decide
the question, the first move is to narrow the question, not to add a second
experiment. A narrower question with a clean answer beats a broad one with a
tangle of partial answers.

**Stop rule.** If the design has grown past one experiment, write one sentence
saying which single result would make the extras unnecessary. If that sentence
is easy to write, cut to that result and stop. If it cannot be written, the
question is still too broad — go back and narrow it.

## Order of operations — questions before hypotheses before designs

### 1. Research questions (the critical step)

Formulate **one** RQ. Additional RQs are admissible only when the *same*
experiment, with no added conditions, no extra runs and no extra analysis
machinery, answers them as a by-product. If answering the second question
requires one more configuration, it is a second experiment — drop it or
replace the first question with it.

Each RQ must be:
- **Self-contained**: understandable without reading the codebase or prior
  chat — a new collaborator or reviewer parses it in one read.
- **Singular**: one question. "Does X help and why" is two RQs.
- **Answerable by observation**: its answer is a measurement outcome, not an
  opinion. If no measurement can answer it, it is a design goal, not an RQ.
- **Plainly worded**: technical terms only where unavoidable, each defined at
  first use.

Bad: "How faithful is CRP on ViTs?" (not self-contained: which notion of
faithful? which models?). Good: "When concept detectors at a fixed ViT layer
are deleted in relevance order, does the class probability fall faster than
when deleted in activation-magnitude order?"

### 2. Hypothesis (one)

For each RQ write H1/H0 with mechanism:
- **H1**: directional, falsifiable, with expected magnitude when defensible
  ("relevance-ordered AOPC exceeds activation-ordered by >2× the random-
  ordering CI width").
- **H0**: what no-effect concretely looks like in the metric.
- **Mechanism** ("because..."): why H1 should hold — one sentence; this is
  what distinguishes a hypothesis from a guess and drives the design.
- **Falsification criterion**: the concrete observable that kills H1. If you
  cannot write one, the hypothesis is not testable — rewrite or demote to
  exploration.
- May be technical, but every symbol/term explained in-line.

Write **one** hypothesis. Quality gate: testability, falsifiability,
parsimony, explanatory power, consistency with known results, novelty vs the
literature (check the related-work notes/briefs before claiming novelty).

Competing hypotheses are the one sanctioned exception, and they come with a
condition: state 2–3 only when a **single** experiment discriminates between
them because their predictions differ on the **same** measurement. If telling
them apart needs a separate run per hypothesis, that is a suite masquerading
as a design — pick the simplest explanation, test it alone, and leave the rest
for after it fails. Designing to discriminate is right; designing to confirm a
favourite is not.

### 3. Variable matrix (per experiment)

| Variable | Role | Operationalization |
|---|---|---|
| e.g. residual-split rule | independent | lrp_configs name |
| e.g. AOPC score | dependent | area between MoRF/LeRF prob curves, N images |
| e.g. model, dataset, N | control (fixed) | pinned checkpoint / split / seed |
| e.g. pretrain provenance | confound | held constant or measured & reported |
| e.g. detector count per basis | confound | normalized axis (%-relevance) |

Every independent variable varies ONE thing per configuration (the
lrp_configs one-knob convention IS this principle). A configuration that
changes two things aliases their effects — the fractional-factorial
"aliasing" trap in ML clothing: you cannot then attribute the outcome to
either knob.

### 4. Design principles, ML translation of Fisher

- **Randomization** → seeds and sample selection. Sample selection rule +
  seed pinned and persisted (indices to npz). Anything "chosen by hand" is a
  confound unless it is the explicit subject.
- **Replication at the right level** → the ML pseudoreplication trap:
  per-token or per-patch measurements within one image are NOT independent;
  the independent unit is usually the image (or the seed, for training runs).
  N = number of images/seeds, and statistics aggregate per-unit first
  (per-image score → median + bootstrap CI), never pooled tokens.
- **Blocking** → known nuisance axes (class, dataset difficulty, outlier-
  token presence) get stratified sampling (class-diverse round-robin) or
  conditioning in the analysis (e.g. condition on outlier masks when no
  clean model exists).
- **Controls** — every effect claim needs at least one: random ordering /
  random subset (chance floor, K≥10 or closed-form expectation), competing
  standard method (activation ranking, rollout), null/randomized model where
  cheap. A number without a control is not a result.
- **Confounds checklist for this domain**: pretrain provenance (verify
  checkpoint lineage — do not trust memory), dimensionality mismatch across
  bases/granularities, conservation leaks (report capture fractions),
  rule-choice interactions, register/outlier tokens, train/test
  contamination, GPU nondeterminism (fix seeds; note where exact determinism
  is not achievable).

### 5. Sample size & decision rules

No formal power analysis pretense; instead: pilot on toy data → observe
variance → choose N so the bootstrap CI is decisively smaller than the
effect claimed; state the decision rule BEFORE the full run (test, threshold,
correction across multiple comparisons — Wilcoxon + Holm is the repo
convention). Pre-registration equivalent: the experiment card
(`remind-me-to-experiment`) or a journal design entry written before compute.

### 6. If more than one experiment is unavoidable

Reaching this section means the stop rule above did not cut the design down.
Say explicitly why one experiment cannot decide the question before going on.
Then:

- Order by dependency and by information per GPU-hour: the cheap
  discriminating run first, the expensive confirmation only for what survives.
- Factor shared infrastructure: one data pass serving several measurements.
- Every experiment states its kill condition — the earlier result that makes
  it pointless, so it is dropped rather than run out of habit.
- Two experiments is a lot. Three is a campaign and needs the user to have
  asked for a campaign.

## Output format

A suite document (research/<topic>_suite.md, or directly YouTrack issue
bodies when the user asks for tracked work) containing per experiment:

```
### E<n>: <title>
RQ: ...
H1 / H0 / Mechanism / Falsified if: ...
Variable matrix: (table)
Controls: ...
Inputs: model+ckpt | dataset+split+N+seed | configs | script
Decision rule: metric, test, threshold
Depends on / killed by: ...
Cost estimate: ...
```

Plus a header: goal, the RQ, staging order if there is more than one
experiment, total compute estimate.

**Before writing the output, answer these three in one line each.** If any
answer is unconvincing, the design is not ready.

1. Which single number decides the question, and what value of it means yes?
2. What is the simplest thing that could explain the expected result, and does
   this design rule it out?
3. What was cut to get here? (If nothing was cut, the design was not
   minimised.)

## Endpoint: designs only, never execution (Adam, 2026-07-26)

This skill ALWAYS ends with pre-registered designs, never with running
anything. Each proposed experiment is written as a **suggested entry** in
the experiment journal's "Suggested — awaiting confirmation" part
(`experiment-journal` skill: Hypothesis / Design & rationale / Planned
metrics & evaluation; Conclusion =
`\todo[inline]{Not performed yet, awaiting confirmation.}`), committed and
pushed for Adam's rigorous review. Execution of any entry starts only after
Adam approves it (then it flows through `remind-me-to-experiment` step 4+
and moves to the Performed part on completion).
