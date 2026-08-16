# Slack monitor notes (#xai-methods-concepts)

Durable hand-off from the `slack-xai-monitor` cron (runs each wake as an
isolated subagent) to the main dev session. The subagent appends dated, terse,
factual bullets here for anything the dev session should eventually act on or
know — decisions, team questions to answer in code, results the monitor posted,
reported bugs, requested experiments. Routine "nothing new / stayed silent"
wakes write nothing.

<!-- newest at top -->

- 2026-06-12: Monitor REPLIED (thread 1781022185.367649, reply p1781252584860119) to
  Jiri Hofirek re prostate ViT heatmaps. He reported pa-LRP (positional-embedding
  absorption) didn't reduce positive relevance on background; his image came from config
  `attnlrp_gamma_palrp`. Pointed out that config = reference recipe `attnlrp_gamma` (full
  bilinear attention, the noise source) + pa-LRP on `x+pos_embed` only; pa-LRP can't fix
  background leak while full bilinear attention runs underneath. Recommended switching to
  `cp_lrp_baseline` (value-path-only, StopGradient on Q/K) first, then revisit pos_embed.
  Open issue: even cp_lrp_baseline may still leak some positive relevance to background on
  prostate ViT — unresolved, dev session may want to investigate.
  FOLLOW-UPS (2026-06-12, same thread): Adam noted (a) saliency map mirrors the patch grid
  with a regular repeating texture over background tiles — model seems to distinguish bg but
  relevance *intensity* isn't normalized right; (b) IMPORTANT caveat: Jiri Hofirek works on
  his OWN commit with local changes the agent's committed-branch view may not reflect. Monitor
  replied (p1781252844963439) acknowledging analysis is vs committed branch, asked Jiri to
  push/share diff of lrp_configs/ + zennit_ext/, and offered a quick per-patch relevance
  quant breakdown (epithelium vs stroma vs background) on the committed branch. SUGGESTED
  DEV-SESSION TASK if requested: run that per-patch relevance statistic to check the
  normalization hypothesis.

- 2026-06-09: Dev session ran a LRP×SAE related-work scan (thread 1780649439.329549,
  reply p1781048011005439). Published summary https://claude-bajger.dyn.cloud.e-infra.cz/zennit-litreview/
  + markdown `research/related_work_lrp_sae.md`. VERDICT: our output→SAE-basis CRP
  contribution is novel — no paper does conservative output-logit→SAE-latent relevance
  decomposition + AOPC-vs-axis-aligned on a ViT. Closest: CaFE (2509.00749, SAE+AttnLRP
  on ViT but feature→input direction), ClassifSAE (2506.23951, decision→SAE but ablation
  not LRP, LLM), Sparse-but-not-Simpler (2603.15919, SAE+attribution as separate axes),
  Interpreto (2512.09730, no LRP, modules unlinked, NLP). Actionable: LRP-for-Autoencoders
  (2303.11734) Deep-Taylor root-point recipe = the principled fix for our weak Q2 decoder-
  bias conservation. Monitor: do not re-reply; relay any team response.

- 2026-06-09: Dev session DRAINED the SAE×CRP task (thread 1780649439.329549). Scaled
  the smoke to BOTH probe sites (proj_drop + residual stream — answers the open
  proj_drop-vs-residual question by doing both) × {axis-aligned embed_dim, SAE α=8} ×
  3 datasets × 12 blocks. New code: `experiments/sae.py` (L1 SAE train + `SAESplice`
  reconstruction passthrough exposing `features` tap), `concept_flipping.py` gains
  `--concept sae` + `--site {proj_drop,residual}` + `--max-steps`. 72 SAEs trained,
  FVU 0.005–0.044. 4 grid figures (3 datasets joined horizontally) published via
  webshare: https://claude-bajger.dyn.cloud.e-infra.cz/zennit-flip/ and posted
  in-thread (msg p1780991554430289). FINDINGS: SAE basis faithful, clearest on
  residual stream (colored_mnist/funny_birds MoRF gap). CAVEAT: conservation Q2 weak
  (~2% of probe relevance reaches latents — folded decode-bias≈mean absorbs the rest
  under the γ-rule); ranking-based flipping curves unaffected, but a clean
  ΣR(f_i)≈R(probe) claim needs a bias-aware decoder rule (used γ, not the planned ε).
  Monitor: do not re-reply; relay any team response.

- 2026-06-06: Dev session posted the REFINED SAE×CRP plan in thread 1780649439.329549
  (msg p1780728662808969). Full markdown committed at `research/sae_crp_plan.md`.
  Sharpened vs the earlier step-plan: novelty positioning (CRP output→SAE-basis
  decomposition, distinct from CaFE arXiv 2509.00749 feature→input, and from
  gradient Sparse Feature Circuits 2403.19647); concrete LRP impl = splice trained
  SAE at probe site as recordable `sae.features` layer with ε-rule on decoder so
  output-logit relevance lands on SAE latents WITH conservation; SAEConcept mirrors
  EmbeddingDimConcept; eval = AOPC(SAE) vs AOPC(embed_dim) + conservation check.
  STILL AWAITING team sign-off on scope + probe site (proj_drop vs residual) before
  queueing to slack_tasks.md. Monitor: do not re-reply, just relay any team response.

- 2026-06-05: Tomas Brazdil raised (thread on ts 1780649439) whether axis-aligned
  concept detection (per tensor component) is outdated vs Anthropic's "rotated"
  concepts. Monitor replied: our concepts (ChannelConcept/HeadConcept/
  EmbeddingDimConcept/TokenConcept in crp/concepts.py) are all axis-aligned;
  Anthropic argues features are non-axis-aligned due to superposition (Elhage
  et al. Toy Models of Superposition 2022) and recover monosemantic features via
  dictionary learning / SAEs (Bricken et al. 2023, Templeton et al. 2024).
  POSSIBLE FOLLOW-UP for dev session: combine LRP/CRP with an SAE basis on the
  residual stream — propagate relevance into SAE features instead of raw dims.
  Monitor offered a small experiment on probes in data/ if wanted.
- 2026-06-05: Monitor went live. Adam Bajger announced/connected the cron agent
  to the channel; Vit Musil asked where the repo is. Monitor replied with confirmation
  + repo URL (https://github.com/RationAI/zennit-crp, active branch
  `transformer-multi-concept`, forked from upstream `rachtibat/zennit-crp`).
- 2026-06-05 context: Adam posted concept-flipping / AOPC results (per-block and
  per-head) for 3 datasets (dSprites, FunnyBirds 10/50-class, colored MNIST).
  Concept detectors measured per-dimension on attention-block output (values +
  skip, no propagation through attention). LRP config appears to give the models
  some per-concept semantics; per-head comparison in progress. (ViT-small, 6 heads.)
- 2026-06-05: Tomas Brazdil asked how our axis-aligned concept detection relates to Anthropic results (superposition / "rotated" concepts). Prior Claude agent answered well (Toy Models of Superposition 2022; Towards/Scaling Monosemanticity 2023/2024; SAE/dictionary learning on residual stream; proposed combining with LRP/CRP by propagating relevance into SAE features). Adam asked that agent to write a detailed concrete experiment plan before running. Monitor stayed silent (already addressed). Possible future dev work: SAE-feature CRP pipeline using probes in data/.

- 2026-06-05: Tomáš Brazdil ptal na Anthropic "rotated"/superposition koncepty vs náš axis-aligned přístup. Adam (dev session) odpověděl referencemi (Elhage 2022, Bricken 2023, Templeton 2024) a navrhl SAE-do-CRP experiment; pak požádal agenta o podrobný konkrétní plán provedení. Monitor odpověděl v threadu detailním step-by-step plánem (sběr aktivací na proj_drop probe → trénink SAE → nová SAEConcept třída po vzoru EmbeddingDimConcept → AOPC/concept-flipping vyhodnocení), s návrhem smoke run na dsprites/1 blok a ověřením konzervace relevance. Čeká na souhlas s rozsahem/probe site před zařazením do fronty.

- 2026-06-05: Vít Musil & Adam requested the slack agent always self-identify (disclaimer line) before replying in Adam's name, so it's clear who is writing. Updated SLACK_CRON.md Rules with an "Always self-identify first" rule (disclaimer line at top of every post). Replied in thread 1780649439.329549.
- 2026-06-05: Thread context — Tomáš Brázdil asked whether axis-aligned concept detection is outdated vs Anthropic's "rotated" concepts (superposition). Adam (via Claude) proposed a concrete SAEConcept plan (sparse autoencoder on proj_drop activations, new Concept subclass, AOPC eval). Not yet queued to slack_tasks.md — awaiting team sign-off on probe-site choice (proj_drop) and scope.

- 2026-06-06: Agent (prior run, 08:51) posted a sharpened SAE×CRP plan in thread 1780649439.329549; full markdown committed at research/sae_crp_plan.md (branch transformer-multi-concept). Key framing: headline "LRP through SAE on ViT" is NOT novel (CaFE); defensible contribution = the CRP move (decompose OUTPUT relevance onto SAE basis + quantitative faithfulness vs axis-aligned CRP + conservation guarantee). Concrete LRP execution: insert SAE as reconstruction pass-through at probe site, expose feature activations f as named layer sae.features, ε-LRP rule on decoder so logit-initiated relevance reaches f conservatively. Eval Q1 AOPC_most−AOPC_least (SAE vs embed_dim same probe site); Q2 conservation ΣR(f_i)≈R(probe). Smoke: dsprites/block 11/L1 SAE α=8. STILL awaiting team sign-off on probe site (proj_drop vs residual stream) before queueing to slack_tasks.md. No human reply in last 8h; monitor stayed silent.

- 2026-06-07: Vojtech Kur pinged Adam with paper openreview bZ0MXXoldX = Bakish/Zimerman/Chefer/Wolf, "Revisiting LRP: Positional Attribution as the Missing Ingredient for Transformer Explainability" (NeurIPS 2025). Adds positional-encoding-aware LRP rules for transformers. Relevant to lost relevance on bias/positional embeddings in ViT branch transformer-multi-concept. Agent replied in-thread with reference + offered comparison to current LRP config.

- 2026-06-07: Tomáš Brázdil (17:30, thread 1780649439.329549) gave informal go-ahead for SAE×CRP ("blbost to neudělat, TB 0.8 promile research mode"). Monitor treated this as team sign-off, QUEUED the task to research/slack_tasks.md (dsprites/block11/L1 SAE α=8 on proj_drop, SAEConcept mirroring EmbeddingDimConcept, ε-LRP on decoder w/ conservation check ΣR(f_i)≈R(probe), AOPC SAE vs embed_dim) and replied in-thread that it's queued for the zennit dev session. OPEN: probe site proj_drop is default but Adam/Vít haven't explicitly confirmed proj_drop vs residual stream — left as settable param. Dev session: drain slack_tasks.md, report results in-thread.
- 2026-06-07: In paper thread (1780844248), Adam (human) corrected the agent: he had already seen openreview bZ0MXXoldX and has it implemented; "agent se asi moc nedíval do kódu". Monitor stayed silent (no question to agent; correction already +1'd). Possible dev note: positional-encoding-aware LRP from that NeurIPS25 paper may already be implemented in branch transformer-multi-concept — verify.

- 2026-06-07: Vojtěch Kůr shared paper openreview `bZ0MXXoldX` (Bakish/Zimerman/Chefer/Wolf, "Revisiting LRP: Positional Attribution...", NeurIPS 2025), addressed to Adam. Agent replied with reference + relevance note. Adam corrected: "Ano, viděl jsem. Mám to naimplementováno. Agent se asi moc nedíval do kódu." → positional LRP rule is ALREADY implemented in the `transformer-multi-concept` branch; agent should consult code before claiming a method is missing. Dev session: confirm where positional attribution LRP lives in the codebase.
- 2026-06-07: Tomáš Brázdil gave green light on SAE×CRP ("blbost to neudělat"); agent confirmed SAE×CRP queued in research/slack_tasks.md (smoke run: dsprites/block 11/L1 SAE α=8/probe site proj_drop). No re-queue needed.
- 2026-06-08: Adam upozornil (thread ts 1780844248), že paper Bakish et al. "Revisiting LRP: Positional Attribution..." (NeurIPS 2025, openreview bZ0MXXoldX) už má naimplementovaný a agent se nedíval do kódu, aby si toho všiml. Lekce pro dev session: zkontrolovat aktuální stav positional-LRP v `transformer-multi-concept` před nabízením implementace. (Past 8h cutoff — no reply.)

- 2026-06-09: Matej Pekar joined #xai-methods-concepts; needs CRP heatmaps on RationAI/vit-patch16-224-prostate (timm ViT) by ~Thu 11.6 (patologové look Tue). transformer-multi-concept branch gives ugly maps; Adam suspects wrong composite advised to Matej. Agent replied: the LXT-article-reproducing recipe is `cp_lrp_baseline` (lrp_configs/cp_lrp_baseline.py) — value-path-only CP-LRP (StopGradient Q/K), γ=0.10, ratio residual, site proj_drop; "cleanest heatmaps". Likely fix: Matej may be using `attnlrp_gamma` (full bilinear) instead of `cp_lrp_baseline`. Jiri to look tomorrow AM; reporting at Fri meeting.

- 2026-06-12: Prostate-ViT thread (parent ts 1781022185). Jiří Hofírek reported (10:18) his heatmap result is from config `attnlrp_gamma_palrp` — an earlier uniform-½ "PA-LRP" sketch (NOT the paper's method; no positional sink, so it only rescaled heatmaps by ½). The "didn't help" observation was an artifact of that wrong implementation, not of PA-LRP itself. The config has since been removed; the paper-faithful PA-LRP now lives in `zennit_extensions/rules/palrp.py` (opt-in). Prior agent runs (10:23–10:27) recommended switching to `cp_lrp_baseline` (value-path-only, StopGradient Q/K) FIRST, then revisit positional. Adam noted Jiří works on an uncommitted local commit the agent can't see; agent caveated its analysis reads committed state only and offered (pending) a quantitative per-patch relevance analysis (epithelium vs stroma vs background) on the committed branch. No new human follow-up after 10:27 within the 8h window — monitor stayed silent this run.

- 2026-06-12 (21:46): Adam (human, top-level msg to Vojtěch Kůr) voiced concern that CaFE (arxiv 2509.00749) is essentially what he wanted to do — "agent tvrdí, že ne, ale z úvodu to vypadá, že ano" — and will read it carefully. Monitor verified CaFE's abstract via WebFetch and replied in-thread (ts 1781293984.631359 under parent 1781293579) reaffirming the prior differentiation: CaFE starts FROM a SAE feature activation and attributes back to input patches (feature→input) via Effective Receptive Field, validated by patch-insertion, on CLIP-ViT, with NO relevance conservation and NO SAE-vs-axis-aligned comparison. Our approach is the opposite axis: start at the OUTPUT logit and CRP/LRP-decompose onto the learned SAE basis (output→SAE) with conservation + concept-flipping/AOPC vs axis-aligned. Overlap is thematic (SAE + LRP-style attribution on ViT), not in what is computed. Agent invited correction if careful reading finds an output→basis decomposition w/ conservation in the paper. No further human reply within 3×3min polling — monitor ended.

- 2026-06-16 (10:36 CEST): Matej Pekar posted an operational FREEZE request to the channel: "Prosím nezasahujte do `zennit-crp` repa branch `transformer-multiconcept-branch`. Instaluje se to z commitu na ECDP, tak aby se to nerozbilo." => DEV SESSION / WORKER: do NOT push/commit/modify the zennit-crp branch transformer-multi-concept right now — it is being installed from a pinned commit on ECDP and must stay stable. Hold off on any repo-mutating work (e.g. draining slack_tasks.md SAE×CRP task) until Matej/team lift the freeze. Local read-only `uv run` experiments are fine; just no branch changes. Monitor stayed silent (heads-up to team, not a question to the agent).

- 2026-06-16 (12:23 CEST): Follow-up to Matej's freeze (thread parent 1781605395.838229). Adam asked whether the ECDP install pins zennit-crp via commit hash or branch name; Matej confirmed COMMIT HASH ("Takže stačí nesmažu ten branch"). => Operational clarification for dev session: the freeze pins a specific commit, so the branch ref itself is safe to keep but must NOT be force-pushed/rebased/deleted in a way that disturbs the pinned commit. Pure logistics between two humans; not addressed to agent. Monitor stayed silent.

- 2026-06-22: Matej Pekár replied in the downstream-loss SAE thread saying "the objective is wrong" and gave L_i = ||S_{θ_{i+1}}(M_{i+1}) − B̃_{i+1}(S_{θ_i}(M_i))||² + λ||f_i||₁. Verified against experiments/sae_downstream.py (transformer-multi-concept, lines 315–320, train loop descending l660–688): the code ALREADY implements exactly this — target=next_sae(block_{i+1}(M_i))=S_{θ_{i+1}}(M_{i+1}), pred=B̃_{i+1}(SAE_i(M_i)). No bug; only the abbreviated inline equation in the original agent post dropped the downstream-SAE tilde. Replied in-thread confirming + reconciling notation.
- 2026-06-22: Adam instructed the agent NOT to use the <@U0B8DJ1CJAE|Claude> mention (reserved Slack app) — write "Claude"/names plainly without @ in future posts.
