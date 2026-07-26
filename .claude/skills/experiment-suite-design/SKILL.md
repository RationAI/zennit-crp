---
name: experiment-suite-design
description: Design a coherent experiment suite for a research goal — precise self-contained research questions first, then falsifiable hypotheses (technical but clearly explained), then designs with controls, confound handling, sample-size reasoning, and staged compute. Use when planning a study/benchmark/ablation campaign, when the user says "design the experiments", "plan the suite", "what experiments do we need", or before any multi-experiment effort.
---

# Design an experiment suite

Adapted for computer-science / ML-interpretability work from:
K-Dense `hypothesis-generation` + `experimental-design` skills and
poemswe `hypothesis-testing` skill (github.com/K-Dense-AI/scientific-agent-skills,
github.com/poemswe/co-researcher). Wet-lab machinery replaced by ML
equivalents; the logic (Fisher's principles, falsifiability, confound
discipline) is retained.

## Order of operations — questions before hypotheses before designs

### 1. Research questions (the critical step)

Formulate 1–5 RQs. Each RQ must be:
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

### 2. Hypotheses (one or more per RQ)

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

Quality gate (from the source skills, CS-tuned): testability, falsifiability,
parsimony, explanatory power, consistency with known results, novelty vs the
literature (check the related-work notes/briefs before claiming novelty).
When a phenomenon has competing explanations, write 2–3 COMPETING hypotheses
whose predictions differ, and design the experiment to discriminate — not to
confirm the favorite (cf. register study: H_A artifact vs H_B real signal,
settled by occlusion + mechanism test).

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

### 6. Suite assembly & staging

- Order experiments by dependency and by information-per-GPU-hour: cheap
  discriminating experiments first (screening on toy datasets = the
  fractional-factorial screen), expensive confirmations (ImageNet-scale)
  only for hypotheses that survive.
- Shared infrastructure factored: one data pass serving many measurements
  (e.g. single FV run per model×config covering all sites/granularities).
- Every experiment cites its kill condition: what earlier result would make
  it pointless (do not run experiment 4 if experiment 2 falsified its
  premise).

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

Plus a suite header: goal, RQ list, staging order, total compute estimate.

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
