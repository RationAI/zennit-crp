# Scout report: novelty of "CRP on ViTs" (XAI-21 paper claim)

*Sonnet literature-scout output, 2026-07-22. 15 queries, 12 fetches. Distilled
verdict first; overlap table verbatim below.*

## Verdict

**Blanket claim "CRP was not previously applied to ViTs" is NOT defensible.**
Same lab (Achtibat/Dreyer/Lapuschkin/Samek) has adjacent work from three sides;
one independent 2025 paper (CaFE) already propagates relevance through an SAE
spliced into a ViT.

**Defensible reformulation** (scout's wording, keep): first to instantiate the
FULL CRP apparatus (conditional heatmaps + RelMax galleries) at the granularity
of specific ViT internal sites (embed-dim / head / value / query / block
output); first systematic concept-flipping faithfulness benchmark of such
site-specific latent ViT concepts; first doing both raw latents AND CRP over an
SAE basis — unifying fragments from AttnLRP, SemanticLens, CaFE.

## Overlap table (condensed)

| Paper | Venue | Overlap w/ (a) sites, (b) faithfulness bench, (c) SAE×CRP |
|---|---|---|
| CRP original (arXiv:2206.03208) | NMI 2023 | none — CNNs only |
| **AttnLRP** (arXiv:2402.05602) | ICML 2024 | (a) PARTIAL — latent relevance in ViTs shown, "opens door" phrasing; latent concept work done on LLMs only. (b,c) none |
| LXT library | — | engine only, no CRP concept tooling for ViT sites |
| **SemanticLens** (arXiv:2501.05398) | NMI 2025 | (a) PARTIAL — RelMax on ViT/DINOv2 "components" (neuron/channel), no site taxonomy, no faithfulness study |
| PURE (arXiv:2404.06453) | CVPR-W 2024 | none — CNN only |
| L-CRP | CVPR-W 2023 | none — CNN backbones only |
| **CaFE** (arXiv:2509.00749) | ExCV-W 2025 | **(c) DIRECT** — SAE on CLIP-ViT-L/14 patch tokens, relevance backprop through SAE encoder + ViT via AttnLRP, insertion-AUC faithfulness. Flat SAE features, no site taxonomy, no CRP/RelMax apparatus. STRONGEST COUNTEREXAMPLE — read fully, differentiate explicitly |
| IMPACT / "Sparse but not Simpler" (arXiv:2603.15919) | 2026 | (b) PARTIAL/DIRECT — SAE-on-ViT (DeiT-III) + insertion/deletion metrics; NOT LRP-based |
| PatchSAE (arXiv:2412.05276) | 2024 | (c) PARTIAL — SAE spatial *activation* maps, no relevance |
| MCD / HU-MCD (arXiv:2301.11911, 2503.18629) | 2023/CVPR 2025 | CNN-only relevance-concept subspaces |
| ICE/CRAFT/ACE, FACE | various | concept-deletion-curve paradigm, activation-based, CNN-oriented |
| SaCo (arXiv:2404.01415) | CVPR 2024 | input-level ViT faithfulness metric only |

Also relevant from other scout: **Pruning by Explaining Revisited**
(arXiv:2408.12568, ECCV-W 2024) — CRP-relevance-ranked structure deletion on
ViTs vs magnitude/random, for pruning. Closest to the flipping mechanics.

## Must-read (full text, Opus pass) before writing intro

1. AttnLRP — arXiv:2402.05602 (backbone; cite precisely what it does NOT do)
2. SemanticLens — arXiv:2501.05398 (component-granularity boundary to argue against)
3. CaFE — arXiv:2509.00749 (strongest (c) counterexample)
4. IMPACT — arXiv:2603.15919 (SAE+faithfulness on ViT, non-LRP)
5. CRP original — arXiv:2206.03208 (terminology)
   (+ Pruning by Explaining Revisited — arXiv:2408.12568, flipping mechanics)
