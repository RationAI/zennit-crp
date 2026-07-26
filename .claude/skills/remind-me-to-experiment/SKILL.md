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
   - Request already implies H and spec → fill the card yourself from context.
   - H is clear but underspecified inputs → fill gaps with repo defaults
     (EXPERIMENTS.md registry, lrp_configs, existing N conventions), note the
     defaults on the card.
   - No falsifiable H exists (pure exploration) → say so explicitly, reframe
     as a QUESTION card ("RQ + what observation would settle it"). Exploratory
     work must be labeled exploratory.
   - The ask would produce an unfalsifiable claim ("make the maps nicer") →
     push back once with a measurable proxy proposal (e.g. localisation mass,
     flipping score), then follow the user's call.

3. **STOP — pre-register, do not run (Adam, 2026-07-26).** The default
   endpoint of this skill is a DESIGN, not a result. Write the experiment as
   a **suggested entry** in the journal's "Suggested — awaiting confirmation"
   part (see the `experiment-journal` skill: full Hypothesis / Design &
   rationale / Planned metrics & evaluation subsections; Conclusion =
   `\todo[inline]{Not performed yet, awaiting confirmation.}`), commit+push
   the journal, tell the user which entry to review, and END. No compute is
   spent on the experiment itself. ONLY proceed to execution when (a) the
   user explicitly approves that entry, or (b) the user's request itself
   explicitly ordered immediate execution ("run it now", "no approval
   needed") — in which case still record the entry first, then run.

4. **While running** (post-approval): the entry's inputs are binding — if any
   input changes mid-run (N, config, model), update the entry and say why.

5. **After the run**: move the entry from the Suggested part to the Performed
   part (end, chronological), fill Results + a bare-answer Conclusion, report
   the verdict per H1 (supported / falsified / inconclusive + the number that
   decided it). A falsified hypothesis is a full-value result; never soften
   it into "partially confirmed".

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
