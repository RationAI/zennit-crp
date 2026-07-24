---
name: remind-me-to-experiment
description: Scientific-method gate. Fires whenever the user asks for experimental or analysis work (run/measure/compare/evaluate/investigate/check whether/see if...) — STOP before executing and force a falsifiable hypothesis + pinned experiment spec, so every piece of work ends as a compact reportable experiment with a verdict. Does not apply to pure infrastructure, refactoring, writing, or bugfixes.
---

# Remind me to experiment

Purpose: no experimental work proceeds on vibes. Every run must be reportable
afterwards as: question → hypothesis → pinned inputs → result → verdict
(supported / falsified / inconclusive), journal-ready
(see the `experiment-journal` skill).

## When this gate applies

Any request whose deliverable is an empirical claim: "run X", "measure Y",
"compare A vs B", "does Z help", "check whether", "see if", "investigate",
"evaluate", "try". NOT for: infrastructure, refactors, paper writing,
bugfixes, figure regeneration of existing results.

## Procedure

1. **Before touching code or GPU, produce the EXPERIMENT CARD** (compact, in
   the reply — not a file):

   ```
   ── EXPERIMENT CARD ─────────────────────────────
   RQ:   <one self-contained question a non-expert collaborator understands>
   H1:   <falsifiable statement, direction + rough magnitude if possible>
   H0:   <what "no effect" looks like>
   Falsified if: <concrete observable outcome that kills H1>
   Inputs: model+ckpt | dataset+split+N+selection(+seed) | config | script/CLI
   Metric + decision rule: <statistic, threshold/CI/test, and at what N>
   Controls/baselines: <random / activation / prior-config / null model — which and why>
   Confounds guarded: <the 1-3 that matter here>
   Cost: <GPU-hours order of magnitude>
   ────────────────────────────────────────────────
   ```

2. **Triage by what the user gave you:**
   - Request already implies H and spec → fill the card yourself from context,
     show it, proceed immediately. The card is a restatement, not a quiz.
   - H is clear but underspecified inputs → fill gaps with repo defaults
     (EXPERIMENTS.md registry, lrp_configs, existing N conventions), note the
     defaults on the card, proceed.
   - No falsifiable H exists (pure exploration) → say so explicitly, reframe
     as a QUESTION card ("RQ + what observation would settle it"), and ask the
     user ONLY if the reframing changes what to compute. Exploratory work is
     allowed — it just must be labeled exploratory and still ends with
     documented inputs + findings.
   - The ask would produce an unfalsifiable claim ("make the maps nicer") →
     push back once with a measurable proxy proposal (e.g. localisation mass,
     flipping score), then follow the user's call.

3. **While running**: keep the card's inputs binding — if any input changes
   mid-run (N, config, model), update the card in the next message and say
   why.

4. **After the run**: report against the card — verdict per H1
   (supported / falsified / inconclusive + the number that decided it), then
   invoke the `experiment-journal` skill to append the entry. A falsified
   hypothesis is a full-value result; never soften it into "partially
   confirmed".

## Rules

- One card per experiment; a sweep over a matrix is one card with the matrix
  in Inputs.
- Never fabricate a hypothesis post-hoc to fit results (HARKing). The card
  precedes the run in the conversation record; if the interesting finding is
  unexpected, journal it as exploratory and write the NEW hypothesis as a
  follow-up card.
- If the user explicitly says "skip the card / just run it", comply — but the
  journal entry afterwards still records inputs and the (retrospective)
  question.
