---
name: experiment-suite-design
description: Design the smallest experiment that decides a question — one research question, one hypothesis, one experiment by default, with controls, confound handling, sample-size reasoning and a stated decision rule. Use when planning a study, trial, benchmark or ablation, when the user says "design the experiments", "plan the suite", "what experiments do we need", or before any experimental effort.
---

# Design an experiment

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
- An additional condition, group or level is only in scope if the question
  cannot be decided without it. "While we are at it" is how designs become
  untestable.

**Narrow the question before adding an experiment.** When one experiment
cannot decide the question, the first move is to narrow the question, not to
add a second experiment. A narrow question with a clean answer beats a broad
one with a tangle of partial answers.

**Stop rule.** If the design has grown past one experiment, write one sentence
naming the single result that would make the extras unnecessary. If that
sentence is easy to write, cut to that result and stop. If it cannot be
written, the question is still too broad — go back and narrow it.

## Order of operations — question, then hypothesis, then design

### 1. The research question

Formulate **one**. A second question is admissible only when the *same*
experiment answers it as a by-product, with no added conditions, no extra runs
and no extra analysis machinery. If answering it requires one more
configuration, it is a second experiment — drop it, or replace the first
question with it.

The question must be:

- **Self-contained** — understandable without reading the code, the lab
  notebook or the prior conversation. A new collaborator or reviewer parses it
  in one read.
- **Singular** — one question. "Does X help, and why" is two.
- **Answerable by observation** — its answer is a measurement outcome, not an
  opinion. If no measurement can answer it, it is a design goal, not a
  research question.
- **Plainly worded** — technical terms only where unavoidable, each defined at
  first use.

Bad: "How effective is the new method?" — which notion of effective, measured
on what, against what. Good: "When the method is applied to the same inputs as
the standard baseline, is the measured error lower?"

### 2. The hypothesis

Write **one**, with four parts:

- **H1** — directional and falsifiable, with an expected magnitude where one
  can be defended ("the treated group's mean exceeds the control's by more
  than twice the width of the control's confidence interval").
- **H0** — what no effect concretely looks like in the chosen measurement.
- **Mechanism** — one sentence on why H1 should hold. This is what separates a
  hypothesis from a guess, and it is what drives the design.
- **Falsification criterion** — the concrete observation that kills H1. If you
  cannot write one, the hypothesis is not testable: rewrite it, or demote it to
  exploration.

Quality gate: testability, falsifiability, parsimony, explanatory power,
consistency with established results, and novelty against the existing
literature — check the literature before claiming novelty.

Competing hypotheses are the one sanctioned exception, and they carry a
condition: state two or three only when a **single** experiment discriminates
between them, because their predictions differ on the **same** measurement. If
telling them apart needs a separate run per hypothesis, that is a suite
wearing the costume of a design — take the simplest explanation, test it
alone, and keep the rest for after it fails. Designing to discriminate is
right; designing to confirm a favourite is not.

### 3. One change per condition

Each condition differs from the baseline in exactly one respect. When two
things change together, no observation can separate them: teach one group with
a new method and also give it extra hours, and a higher score belongs to
neither the method nor the hours. The outcome is then uninterpretable, however
large it is.

This is the most common way a design fails, and it is invisible once the data
are in — so check it before the run, not after.

### 4. Design principles

- **Randomisation.** Assignment and selection are made by a rule, not by
  hand, and the rule is recorded so the selection can be reproduced. Anything
  chosen by hand is a confound unless it is the explicit subject of the study.
- **Count the right thing.** Decide what one piece of evidence is, and count
  those. A piece of evidence is something that could have come out differently
  on its own. Twenty questions put to ten people is ten pieces of evidence,
  not two hundred — one person's answers move together. Summarise each one
  first, then compare across them. Counting the measurements instead of the
  things measured produces error bars far too narrow to believe.
- **Balance what you cannot remove.** Some unwanted variation is known and
  impossible to eliminate: people differ in age, machines differ in speed,
  material differs by batch. Random assignment balances it only on average,
  and one draw can still put every older participant in the same group. So
  split by that variable first and assign conditions within each split, or run
  every condition on every machine. Where neither is practical, record the
  variable and account for it when analysing.
- **Controls.** Every effect claim needs at least one: a chance baseline (what
  the measurement reads when there is no effect, by simulation or closed
  form), the standard method it is supposed to beat, or a deliberately
  broken version of the system. A number without a control is not a result.
- **Confounds to rule out explicitly.** Provenance of the materials or data
  (verify it; do not trust memory). Scale or units differing between
  conditions. Information leaking from what is measured into what is
  predicted. Interaction with a setting assumed to be neutral. Contamination
  between the material used to build the system and the material used to
  evaluate it. Nondeterminism in the apparatus — pin what can be pinned, and
  state plainly where exact repeatability is not achievable.

### 5. Sample size and the decision rule

Do not stage a formal power analysis you cannot support. Instead: run a pilot
on cheap material, observe the variance, then choose the sample size so the
confidence interval is decisively narrower than the effect being claimed.

State the decision rule **before** the full run: which number, which test,
which threshold, and how multiple comparisons are corrected. Write it down
where it cannot be quietly revised after the data arrive — that record is the
whole point of pre-registration.

### 6. If more than one experiment is unavoidable

Reaching this section means the stop rule did not cut the design down. Say
explicitly why one experiment cannot decide the question before going on.
Then:

- Order by dependency, and by information gained per unit of cost: the cheap
  discriminating run first, the expensive confirmation only for what survives
  it.
- Factor shared effort — one pass over the material serving several
  measurements.
- Give every experiment a kill condition: the earlier result that would make
  it pointless, so it is dropped rather than run out of habit.
- Two experiments is a lot. Three is a campaign, and needs the user to have
  asked for a campaign.

## Output

One design document per experiment:

```
### E<n>: <title>
Question: ...
H1 / H0 / Mechanism / Falsified if: ...
What changes between conditions: ...
Controls: ...
Inputs: material, selection rule, sample size, settings, procedure
Decision rule: measurement, test, threshold
Depends on / killed by: ...
Cost estimate: ...
```

With a header giving the goal, the question, and the total cost estimate —
plus the ordering, if there is more than one experiment.

**Before writing the output, answer these two in one line each.** If any
answer is unconvincing, the design is not ready.

1. Which single number decides the question, and what value of it means yes?
2. What is the simplest thing that could explain the expected result, and does
   this design rule it out?

## Endpoint: designs, never execution

This skill always ends with a written design, never with running anything.
Each experiment is recorded as a proposal, awaiting review — hypothesis,
design and rationale, planned measurement and analysis, with the conclusion
left explicitly unfilled. Execution begins only once the proposal has been
approved; the result is then recorded against the design as written, including
where it diverged.
