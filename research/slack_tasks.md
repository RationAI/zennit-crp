# Slack task queue (#xai-methods-concepts)

Big tasks the Slack monitor was asked to do but can't run inside a monitor pass.
The main zennit worker drains this: pick the oldest unchecked item, do it, reply
in the Slack thread with results, then check it off (`- [x]`).

Format: `- [ ] YYYY-MM-DD — <requester> — <ask> — thread: <link or thread_ts>`

## Queue
<!-- monitor appends below -->
- [x] 2026-06-07 — Tomáš Brázdil (informal go-ahead, thread also seen by Adam/Vít/Vojtěch) — Implement & smoke-test the SAE×CRP pipeline per `research/sae_crp_plan.md`: (1) collect activations at probe site `proj_drop` on dsprites; (2) train a small L1 SAE (expansion α=8) on one block (block 11) — smoke run; (3) add `SAEConcept(Concept)` mirroring `EmbeddingDimConcept` but in SAE feature space, expose feature activations as named layer `sae.features`, ε-LRP rule on decoder so logit-initiated relevance reaches `f` with conservation; (4) eval Q1 faithfulness `AOPC_most − AOPC_least` (SAE vs `embed_dim` same probe site) + Q2 conservation `ΣR(f_i) ≈ R(probe)`. Validate relevance conservation on the smoke run BEFORE scaling to all blocks + all 3 datasets. NOTE: probe site `proj_drop` is the agent's proposed default but Adam/Vít have not explicitly confirmed proj_drop vs residual stream — keep it a settable param; if residual is wanted, swap site. Report results back in-thread. — thread: 1780649439.329549
  DONE 2026-06-09: scaled to BOTH sites (proj_drop + residual) × {embed_dim, sae} × 3 datasets, all 12 blocks (`experiments/sae.py`, `--concept sae --site` in `concept_flipping.py`). 72 SAEs FVU 0.005–0.044. 4 grid figures published https://claude-bajger.dyn.cloud.e-infra.cz/zennit-flip/ and posted in-thread (p1780991554430289). Conservation Q2 only weak (~2% reaches latents; folded decode-bias≈mean absorbs rest under γ-rule) — ranking/curves valid, but a clean ΣR(f)≈R(probe) claim needs a bias-aware decoder rule. Splice = reconstruction passthrough w/ γ-rule (not ε as planned).
