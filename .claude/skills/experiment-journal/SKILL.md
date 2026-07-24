---
name: experiment-journal
description: Append a reproducible experiment entry (hypothesis, pinned-inputs design, in-body results + figures, terse conclusion) to the LaTeX research journal in the crp-paper repo. Use whenever an experiment finishes, when the user says "journal this", "add journal entry", "record the experiment", or after any analysis producing findings worth revisiting.
---

# Add an experiment-journal entry

The journal is `/home/claude/workspaces/crp-paper/iclr2026/experiment-journal.tex`
— chronological, append-only, one `\section` per experiment. Its purpose:
every finding revisitable, every experiment reproducible from the entry alone.
Web links are auxiliary (temporary); results live in the document body.

## Entry procedure

1. **Gather the pinned inputs** (all mandatory; dig them out of the run, do
   not approximate):
   - model + exact checkpoint path (or timm name + "pretrained")
   - dataset, split, N, sample-selection rule (+ seed / persisted indices)
   - LRP/CRP config name from `lrp_configs/` (and any non-default knobs)
   - script path + exact CLI invocation
   - git commit hash of `zennit-crp` containing the code that ran (commit
     first if the code is uncommitted — an entry must never reference
     unversioned code)
   - result-array paths under `data/results/...`
2. **Copy key figures** (1–3 per entry, png) into
   `/home/claude/workspaces/crp-paper/iclr2026/journal-figures/`. Prefer the
   single figure that carries the conclusion. `\graphicspath` already points
   there — reference by basename without extension.
3. **Append the entry** at the END of the document, directly above the
   "Append new entries BELOW" marker comment, using the template below.
   Date = experiment execution date (not writing date). Never edit or reorder
   existing entries; corrections go in a new dated entry referencing the old.
4. **Static checks** (no LaTeX toolchain on the pod — do not attempt
   `pdflatex`): balanced `\begin{...}`/`\end{...}`; every
   `\includegraphics` target exists in `journal-figures/`; underscores in
   paths escaped (`\_`); `%` written as `\%`.
5. **Commit + push both repos**: the figure/data/code side in `zennit-crp`
   (if anything new), and the journal in `crp-paper` (`git add
   iclr2026/experiment-journal.tex iclr2026/journal-figures/ && git commit &&
   git pull --no-rebase && git push origin master`). The paper repo syncs to
   Overleaf via GitHub — pushing is what makes the entry visible to the user.
   Both repos need the local git identity already configured; commit messages
   end with the standard Co-Authored-By/Claude-Session trailer.

## Entry template

```latex
% ============================================================================
\section{YYYY-MM-DD --- <short experiment title>}
% ============================================================================

\subsection*{Hypothesis}
<One falsifiable statement. If exploratory, state the question instead and
say so.>

\subsection*{Design \& rationale}
<How the experiment answers the hypothesis and why this design. End with:>
\textbf{Inputs}: <model+checkpoint>; <dataset, split, N, selection rule,
seed>; <LRP config>; script \texttt{<path>} (\texttt{<CLI>}); code commit
\texttt{<hash>}; arrays \texttt{<data/results/... paths>}.

\subsection*{Results}
<Key numbers IN TEXT or a booktabs table — enough to re-derive the
conclusion without opening files. Then figures:>
\begin{center}
\includegraphics[width=0.7\textwidth]{<figure-basename>}
\end{center}
<Optional auxiliary web link via \url{...}, clearly secondary.>

\subsection*{Conclusion}
<Brief, terse, descriptive. Only claims the Results support. Note open
caveats in one clause.>
```

## Rules

- No claim without data in the entry; no data without provenance (commit +
  paths).
- Terse throughout; the Conclusion is 1–3 sentences.
- One experiment = one entry; a multi-part study is multiple entries.
- Failed/negative experiments get entries too — same rigor.

## Presentation conventions (Adam, 2026-07-24 review)

- **Tables for numbers**: whenever multiple numbers are compared, present
  them as a booktabs table, not inline prose.
- **Conclusions are bare answers**: separate short sentences containing ONLY
  the final answer to the hypothesis/question. All explanation and reasoning
  belongs in the Results body BEFORE the conclusion, never inside it.
- **Saliency-map provenance**: every figure showing saliency/relevance maps
  carries a note stating exactly where they come from — method, site/layer,
  composite/config name, conditioning (class? concept?), key params — placed
  UNDER the figure as a description (never overlaid on the visual).
- **Figure descriptions**: every figure gets a `\noindent Figure: ...`
  description paragraph explaining axes, color encoding, and what to look at.
- **Clickable navigation**: hyperref labels everywhere — `\label{exp:...}` on
  every entry section, `\label{model:...}` on model records,
  `\label{fig:...}` on figures where referenced; cross-reference with `\ref`
  so the PDF is navigable by clicking.
- **Statistical terms**: name the test and correction explicitly and add a
  half-sentence plain-language gloss at first use in an entry (e.g.
  "Holm–Bonferroni step-down correction across the 196 positions"); report
  underflowed p-values as bounds, never as 0.
