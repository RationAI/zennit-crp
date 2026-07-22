# CAV self-consistency check for CRP representants — protocol

*Designed 2026-07-22 (Adam's sanity-check framing). Status: planned, not run.
Complementary independent-source CRP-vs-TCAV comparison: backlog (see
`scout_benchmark_cav_datasets.md` §4).*

## Hypothesis

**H1 (linear coherence of RelMax representants).** For a concept detector `c`
(layer `L`, basis `B` ∈ {embed_dim, head, sae}), the top-N RelMax representant
set `S_N` is *linearly coherent*: a linear direction (CAV) fit on a K-subset of
`S_N` against random negatives generalizes — images retrieved from a held-out
pool by CAV similarity overlap with the held-out representants `S_N \ S_K` far
above chance, and score high relevance `R_c`.

Failure is a reportable finding, not a broken experiment: either the detector is
polysemantic (S_N = several clusters, no single direction) or relevance-based
selection is not aligned with any linear feature — both directly qualify the
"what does this detector mean" claim.

## Steps (each with purpose)

1. **Detector selection**: top-n detectors per (layer, basis) by the flipping
   ranking. *Purpose: test detectors that matter for decisions, not arbitrary ones.*
2. **Representant split**: `S_N` = top-N from the FV index (N ≤ 40 =
   `Maximization.SAMPLE_SIZE`). Split by **rank interleaving** (odd ranks → fit
   set `S_K`, even → held-out) so both halves share the same relevance
   distribution. *Purpose: prevents "fit on best, test on worst" rank confound.*
3. **Feature space**: activations at the same site/layer `L`, spatial-mean over
   (filtered) tokens → one vector per image. *Purpose: CAV lives where the
   detector lives; token pooling matches how the detector's relevance is summed.*
4. **CAV fit**: logistic regression (L2), `S_K` positives vs M = 10·K random
   negatives from the dataset (excluding `S_N`). TCAV convention. *Purpose:
   standard, citable CAV estimator.*
5. **Retrieval**: score pool `P` = dataset ∖ `S_K` by ⟨a(x), v_c⟩; take top-M′,
   M′ = |S_N ∖ S_K|·5. *Purpose: generalization query — can the direction alone
   find the concept?*
6. **Primary metric**: overlap |top-M′ ∩ (S_N∖S_K)| vs hypergeometric null
   (pool size |P|, draws M′, successes |S_N∖S_K|); report p per detector +
   fraction of detectors with p < 0.01. *Purpose: exact chance-corrected test of
   H1 — this is Adam's "significant overlap with remaining representants".*
7. **Secondary metrics**: (a) mean `R_c`(retrieved) vs `R_c`(random), Mann-
   Whitney — *retrieved images fire the detector in relevance terms, not just
   activation similarity*; (b) direction alignment: cosine(v_c, e_c) for
   axis-aligned bases, cosine(v_c, decoder column d_c) for SAE — *does the CAV
   rediscover the basis vector? An interpretable scalar per detector.*
8. **Controls**: (i) N≥10 random-positive-set CAVs → null distribution for all
   metrics (TCAV convention); (ii) same pipeline with **ActMax** representants —
   *does relevance-selected S_N define a more generalizable direction than
   activation-selected? Direct RelMax-vs-ActMax evidence.*

## Costs

Activations at site L for the eval split (one forward sweep, reusable across
detectors and bases); logistic fits trivial; relevance of retrieved sets = one
conditional backward per (detector, image) on small sets. Runs on funny_birds +
imagenet-subset within existing infra.
